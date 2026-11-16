import json
import os
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import AsyncGroq

JEE_BASE = "https://pfqwhrjalqnvscewfurq.supabase.co/functions/v1/j"
MODEL = "openai/gpt-oss-120b"
MAX_ROUNDS = 4  # lets the model chain checklist -> mark
HTTP_TIMEOUT = 15

SUBJECTS = ["physics", "chemistry", "maths", "prerna"]
STATUSES = ["done", "in_progress", "not_started"]

SYSTEM_PROMPT = """You are a JEE syllabus tracker assistant. You have tools to read and update
the user's JEE progress in Supabase (Physics, Chemistry, Maths, Prerna).

Rules:
- To mark a chapter, ALWAYS call jee_checklist for that subject first to get
  the exact chapter name, then call jee_mark with the exact name.
- Never invent chapter names. Never guess.
- Be concise. After tool results come back, summarize what happened in 1-3
  sentences, including the new remaining count when relevant.
- You can also chat normally on non-tracker questions; if no tool is needed,
  just reply. You may gently steer back to JEE topics when relevant.
- For overall progress or the full syllabus, pass subject="all".
- Never send null for any argument. Every parameter you send must match its
  declared type and enum exactly.
- Never expose the API key or raw URLs to the user.
"""

app = FastAPI(title="JEE Syllabus AI Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Tool definitions
# --------------------------------------------------------------------------
def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_SUBJECT_PROP = {"type": "string", "enum": SUBJECTS}
# Groq validates tool arguments strictly: an optional string param that the model
# fills with null fails with "expected string, but got null". So optional params
# take an explicit "all" value instead of being omittable.
_SUBJECT_OPT = {
    "type": "string",
    "enum": [*SUBJECTS, "all"],
    "description": "Subject name, or 'all' for every subject.",
}

TOOLS = [
    _tool(
        "jee_progress",
        "Get overall progress (total, done, remaining chapters, per-subject breakdown). "
        "Optionally filter by subject.",
        {"subject": _SUBJECT_OPT},
        ["subject"],
    ),
    _tool(
        "jee_checklist",
        "Get the full chapter checklist for a subject, including exact chapter names, "
        "ids and status.",
        {"subject": _SUBJECT_PROP},
        ["subject"],
    ),
    _tool(
        "jee_pending",
        "List the pending (not yet done) chapters for a subject.",
        {"subject": _SUBJECT_PROP},
        ["subject"],
    ),
    _tool(
        "jee_mark",
        "Mark a chapter's status. Use the EXACT chapter name. If unsure, call "
        "jee_checklist first and copy the name verbatim. Never invent chapter names.",
        {
            "subject": _SUBJECT_PROP,
            "chapter": {"type": "string", "description": "Exact chapter name."},
            "status": {
                "type": "string",
                "enum": STATUSES,
                "description": "New status. Defaults to done.",
            },
        },
        ["subject", "chapter"],
    ),
    _tool(
        "jee_reset",
        "DESTRUCTIVE: clears all ticked/completed chapters for ONE subject. Only call "
        "this if the user's message explicitly asks for the reset AND contains the word "
        "'confirm'. Otherwise do not call it: reply asking them to repeat the request "
        "with the word 'confirm'.",
        {"subject": _SUBJECT_PROP},
        ["subject"],
    ),
    _tool(
        "jee_syllabus",
        "Get the full chapter list with status, optionally for one subject.",
        {"subject": _SUBJECT_OPT},
        ["subject"],
    ),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def scrub(text: str) -> str:
    """Make sure secrets never leak into responses."""
    for var in ("JEE_API_KEY", "GROQ_API_KEY"):
        secret = os.environ.get(var)
        if secret:
            text = text.replace(secret, "***")
    return text


_http: Optional[httpx.AsyncClient] = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    return _http


def filter_progress(data: dict[str, Any], subject: str) -> dict[str, Any]:
    """Upstream may ignore s on a=progress, so narrow to one subject ourselves."""
    rows = data.get("by_subject")
    if not isinstance(rows, list):
        return data
    kept = [
        r for r in rows
        if isinstance(r, dict)
        and str(r.get("subject", "")).strip().lower() == subject
    ]
    if not kept:
        return data
    return {"ok": data.get("ok", True), "subject": subject, "by_subject": kept}


async def call_jee(params: dict[str, str]) -> dict[str, Any]:
    key = os.environ.get("JEE_API_KEY")
    if not key:
        return {"ok": False, "error": "JEE_API_KEY is not configured"}
    query = {**params, "apikey": key}
    try:
        try:
            resp = await get_http().get(JEE_BASE, params=query)
        except RuntimeError:
            # Cached client bound to a closed event loop (serverless reuse): rebuild once.
            global _http
            _http = None
            resp = await get_http().get(JEE_BASE, params=query)
        try:
            data = resp.json()
        except ValueError:
            return {
                "ok": False,
                "error": f"upstream returned non-JSON (HTTP {resp.status_code})",
            }
        return data if isinstance(data, dict) else {"ok": True, "data": data}
    except httpx.HTTPError as e:
        # Do not use str(e): httpx messages can include the full URL (and key).
        return {"ok": False, "error": f"upstream request failed: {type(e).__name__}"}


async def run_tool(name: str, args: dict[str, Any], q: str) -> dict[str, Any]:
    subject = args.get("subject")
    if subject == "all":
        subject = None
    if subject is not None and subject not in SUBJECTS:
        return {"ok": False, "error": f"invalid subject: {subject}"}

    if name == "jee_progress":
        params = {"a": "progress"}
    elif name == "jee_syllabus":
        params = {"a": "syllabus"}
    elif name == "jee_reset":
        if not subject:
            return {"ok": False, "error": "subject is required for reset"}
        if "confirm" not in q.lower():
            return {
                "ok": False,
                "error": "Reset NOT performed. Ask the user to repeat the request "
                         "including the word 'confirm'.",
            }
        params = {"a": "reset"}
    elif name in ("jee_checklist", "jee_pending", "jee_mark"):
        if not subject:
            return {"ok": False, "error": "subject is required"}
        params = {"a": "checklist" if name == "jee_checklist"
                  else "pending" if name == "jee_pending" else "mark"}
        if name == "jee_mark":
            chapter = args.get("chapter")
            if not chapter:
                return {"ok": False, "error": "chapter is required"}
            status = args.get("status") or "done"
            if status not in STATUSES:
                return {"ok": False, "error": f"invalid status: {status}"}
            params["c"] = chapter
            params["st"] = status
    else:
        return {"ok": False, "error": f"unknown tool: {name}"}

    if subject:
        params["s"] = subject
    result = await call_jee(params)
    if name == "jee_progress" and subject and result.get("ok") is not False:
        result = filter_progress(result, subject)
    return result


def parse_args(raw: Optional[str]) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/chat")
async def chat(q: Optional[str] = Query(None)):
    if not q or not q.strip():
        return JSONResponse({"ok": False, "error": "missing q"}, status_code=400)

    try:
        groq = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], timeout=HTTP_TIMEOUT * 2)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        executed: list[dict[str, Any]] = []
        reply: Optional[str] = None

        for _ in range(MAX_ROUNDS):
            try:
                resp = await groq.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    reasoning_effort="low",
                )
            except Exception as e:
                detail = str(e)
                # Groq validates generated tool arguments against the schema and
                # rejects the whole call (e.g. "expected string, but got null").
                # Tell the model what went wrong and let it try again.
                if "tool_use_failed" in detail or "did not match schema" in detail:
                    messages.append({
                        "role": "system",
                        "content": "Your previous tool call was rejected by the "
                                   "schema validator. Call the tool again with every "
                                   "argument matching its declared type and enum "
                                   "exactly. Never send null. Validator said: "
                                   + detail[:300],
                    })
                    continue
                raise
            msg = resp.choices[0].message

            if not msg.tool_calls:
                reply = (msg.content or "").strip()
                break

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                args = parse_args(tc.function.arguments)
                result = await run_tool(tc.function.name, args, q)
                executed.append({"name": tc.function.name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, separators=(",", ":")),
                })

        if reply is None:
            # Tool rounds exhausted: final summary call with no tools.
            final = await groq.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                reasoning_effort="low",
            )
            reply = (final.choices[0].message.content or "").strip()

        return {"ok": True, "query": q, "reply": reply, "tool_calls": executed}

    except Exception as e:
        return JSONResponse({"ok": False, "error": scrub(str(e))}, status_code=500)
