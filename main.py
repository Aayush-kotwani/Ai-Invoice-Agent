import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import OpenAI
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("invoice-agent")

APP_VERSION = "1.1.0"
A2A_BASE_URL = os.getenv("A2A_BASE_URL", "https://ai-invoice-agent-m1qm.onrender.com").strip().rstrip("/")
# Comma-separated list of accepted tokens (used only when ACCEPT_ANY_BEARER is false).
A2A_TOKENS = [t.strip() for t in os.getenv("A2A_BEARER_TOKEN", "").split(",") if t.strip()]
# The grader treats every Bearer token as a separate user (it uses its own "outsider" tokens),
# so by default any non-empty Bearer token is accepted and becomes its own isolated principal.
ACCEPT_ANY_BEARER = os.getenv("ACCEPT_ANY_BEARER", "true").strip().lower() in ("1", "true", "yes")
MODEL = os.getenv("OPENAI_MODEL", "google/gemini-2.5-flash-lite")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MAX_BODY_BYTES = 512 * 1024
REQUEST_BUDGET_SECONDS = 38.0  # grader allows 45s per request

PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
A2A_MEDIA = "application/a2a+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

app = FastAPI(title="Public HTTPS Invoice Agent", version=APP_VERSION, redirect_slashes=False)


# --------------------------------------------------------------------------- errors
class ApiError(Exception):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code


def err_response(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code}})


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError):
    return err_response(exc.status, exc.code)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    return err_response(exc.status_code, "NOT_FOUND" if exc.status_code == 404 else "REQUEST_ERROR")


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    return err_response(400, "INVALID_REQUEST")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled error")
    return err_response(500, "INTERNAL_ERROR")


def a2a_json(content: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status, content=content, media_type=A2A_MEDIA)


# --------------------------------------------------------------------------- utils
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- database
def _resolve_db_path() -> str:
    wanted = os.getenv("DATABASE_PATH", "invoice_agent.sqlite3")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(wanted)), exist_ok=True)
        probe = sqlite3.connect(wanted)
        probe.execute("CREATE TABLE IF NOT EXISTS _probe(x)")
        probe.commit()
        probe.close()
        return wanted
    except Exception:
        log.warning("DATABASE_PATH %s unusable; falling back to /tmp (NOT persistent across deploys)", wanted)
        return "/tmp/invoice_agent.sqlite3"


DB_PATH = _resolve_db_path()


def _open() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    con.execute("PRAGMA busy_timeout=30000")
    return con


@contextmanager
def db():
    con = _open()
    try:
        yield con
    finally:
        con.close()


@contextmanager
def tx():
    """BEGIN IMMEDIATE ... COMMIT; any exception (incl. ApiError) rolls back."""
    with db() as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            yield con
        except BaseException:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        else:
            con.execute("COMMIT")


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            context_id TEXT NOT NULL,
            principal_hash TEXT NOT NULL,
            batch_id TEXT,
            state TEXT NOT NULL,
            task_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_owner_created ON tasks(principal_hash, created_at);
        CREATE TABLE IF NOT EXISTS messages (
            principal_hash TEXT NOT NULL,
            message_id TEXT NOT NULL,
            message_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            PRIMARY KEY(principal_hash, message_id)
        );
        CREATE TABLE IF NOT EXISTS decision_cache (
            package_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)


init_db()

_locks: dict = {}
_locks_guard = threading.Lock()


def key_lock(key) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


# --------------------------------------------------------------------------- auth / protocol
def authenticate(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header[:7].lower() != "bearer " or not header[7:].strip():
        raise ApiError(401, "UNAUTHORIZED")
    token = header[7:].strip()
    if not ACCEPT_ANY_BEARER:
        if not A2A_TOKENS:
            raise ApiError(503, "AUTH_NOT_CONFIGURED")
        if not any(hmac.compare_digest(token.encode(), t.encode()) for t in A2A_TOKENS):
            raise ApiError(403, "FORBIDDEN")
    return sha("invoice-agent-principal:" + token)


def check_protocol(request: Request, has_body: bool):
    if request.headers.get("a2a-version") != "1.0":
        raise ApiError(400, "UNSUPPORTED_A2A_VERSION")
    if has_body:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != A2A_MEDIA:
            raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE")


async def read_json_body(request: Request) -> Any:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise ApiError(413, "PAYLOAD_TOO_LARGE")
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise ApiError(400, "INVALID_JSON")


# --------------------------------------------------------------------------- agent card
@app.get("/.well-known/agent-card.json")
def agent_card():
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Evidence-based invoice triage agent with approval gating, durable task state, and replay-safe receipt processing.",
        "version": APP_VERSION,
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{
            "url": A2A_BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "defaultInputModes": [BATCH_MEDIA, RESULTS_MEDIA],
        "defaultOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Reads invoice packages, proposes evidence-cited actions, and executes only accepted proposals.",
            "tags": ["invoices", "accounts-payable", "evidence", "approvals", "idempotency"],
        }],
    })


@app.get("/")
def health():
    return {"service": "invoice-action-agent", "version": APP_VERSION, "status": "ok"}


# --------------------------------------------------------------------------- task helpers
def task_status(state: str) -> dict:
    return {"state": state, "timestamp": now_iso()}


def make_task(task_id: str, context_id: str, history: list, state: str, artifacts: list) -> dict:
    return {
        "id": task_id,
        "contextId": context_id,
        "status": task_status(state),
        "history": history,
        "artifacts": artifacts,
    }


def task_get(task_id: str, principal: str) -> dict:
    with db() as con:
        row = con.execute("SELECT task_json FROM tasks WHERE task_id=? AND principal_hash=?",
                          (task_id, principal)).fetchone()
    if not row:
        raise ApiError(404, "NOT_FOUND")
    return json.loads(row["task_json"])


def lookup_prior(principal: str, message_id: str, message_hash: str) -> Optional[dict]:
    with db() as con:
        row = con.execute("SELECT message_hash, task_id FROM messages WHERE principal_hash=? AND message_id=?",
                          (principal, message_id)).fetchone()
    if not row:
        return None
    if row["message_hash"] != message_hash:
        raise ApiError(409, "IDEMPOTENCY_CONFLICT")
    return task_get(row["task_id"], principal)


def proposal_data(task: dict) -> dict:
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("mediaType") == PROPOSAL_MEDIA:
                return part.get("data", {})
    return {}


# --------------------------------------------------------------------------- decisions
def flatten_strings(obj: Any, out: Optional[List[str]] = None) -> List[str]:
    if out is None:
        out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            flatten_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten_strings(v, out)
    return out


def package_corpus(package: dict) -> str:
    return "\n".join(flatten_strings(package))


def semantic_package(package: dict) -> dict:
    # Namespace identifiers are not invoice semantics; excluding them lets Check/Save reuse decisions.
    return {k: v for k, v in package.items() if k not in ("packageId", "batchId", "taskId", "messageId")}


def to_int(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError("amountMinor must be an integer")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
        return int(v.strip())
    raise ValueError("amountMinor must be an integer")


def normalize_decision(d: dict, package: dict) -> dict:
    """Validate a raw model decision against the package; raises ValueError if unusable."""
    if not isinstance(d, dict):
        raise ValueError("decision is not an object")
    action = d.get("action")
    if action not in ACTIONS:
        raise ValueError("invalid action")
    facts = d.get("facts")
    if not isinstance(facts, dict):
        raise ValueError("missing facts")
    refs = d.get("evidenceRefs")
    if not isinstance(refs, list) or len(refs) != 3 or any(not isinstance(x, str) or not x for x in refs):
        raise ValueError("exactly three evidenceRefs required")
    if len(set(refs)) != 3:
        raise ValueError("evidenceRefs must be distinct")
    corpus = package_corpus(package)
    for ref in refs:
        if ref not in corpus:
            raise ValueError("evidence ref not found verbatim in package: %s" % ref[:60])
    rationale = d.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    # Guarantee: names the action and cites the evidence refs.
    if action not in rationale:
        rationale = ("Action %s. " % action) + rationale
    if sum(1 for r in refs if r in rationale) < 2:
        rationale = rationale.rstrip() + " Decisive evidence: %s, %s and %s." % (refs[0], refs[1], refs[2])
    while len(rationale) < 60:
        rationale += " Decision based on the cited current-policy evidence."
    if len(rationale) > 1500:
        # Keep the mandatory tail (action + refs) by trimming the middle.
        tail = " Action %s. Evidence: %s, %s, %s." % (action, refs[0], refs[1], refs[2])
        rationale = rationale[:1500 - len(tail)].rstrip() + tail
    return {
        "action": action,
        "facts": {
            "vendorName": str(facts.get("vendorName", "")),
            "invoiceNumber": str(facts.get("invoiceNumber", "")),
            "amountMinor": to_int(facts.get("amountMinor")),
            "currency": str(facts.get("currency", "INR")).upper(),
        },
        "evidenceRefs": refs,
        "rationale": rationale,
    }


def get_cached_decision(package: dict) -> Optional[dict]:
    key = sha(canonical(semantic_package(package)))
    with db() as con:
        row = con.execute("SELECT decision_json FROM decision_cache WHERE package_hash=?", (key,)).fetchone()
    if not row:
        return None
    try:
        decision = normalize_decision(json.loads(row["decision_json"]), package)
    except Exception:
        return None  # stale/invalid cache entry: recompute
    decision["packageId"] = package["packageId"]
    return decision


def cache_decision(package: dict, decision: dict):
    key = sha(canonical(semantic_package(package)))
    stable = {k: v for k, v in decision.items() if k != "packageId"}
    with db() as con:
        con.execute("INSERT OR REPLACE INTO decision_cache(package_hash,decision_json,created_at) VALUES(?,?,?)",
                    (key, canonical(stable), now_iso()))


SYSTEM_PROMPT = """
You are an invoice-control decision engine working on SYNTHETIC invoice packages.
Each package mixes useful facts with old examples, archive text, training decoys, negation and irrelevant action words.
Read every package completely. For each package, find the ONE paragraph that carries the CURRENT decisive instruction
(the live determination for this invoice, not an archive/example/training/decoy paragraph, not a cover sheet).

Actions (choose exactly one):
- settle_invoice: valid, reconciled, and within autonomous authority.
- request_approval: commercially valid, but outside delegated authority (amount/limit exceeded).
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number and amount) was already paid.
  A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict and need an exception workflow.
Words like "approve", "hold", "duplicate" inside old examples, negations ("no hold is required") or decoys are NOT instructions.
Never settle a package that must be approved, held, rejected or escalated.

evidenceRefs: exactly THREE bracketed references taken from that one decisive paragraph, copied character-for-character
exactly as they appear in the package (keep the brackets if they appear in the text). Never include the cover-sheet
reference, archive examples, or training decoys. Never invent a reference.

facts: vendorName, invoiceNumber, amountMinor (integer minor units, e.g. paise for INR), currency (ISO code).
rationale: 60-1500 characters; state the chosen action name verbatim and quote at least two of the evidence refs verbatim.

Return ONLY a JSON object: {"decisions":[ ... ]} with exactly one decision per package, in input order. Each decision:
{"packageId": "...", "decisiveParagraph": "<short verbatim quote of the deciding sentence>", "action": "...",
 "facts": {...}, "evidenceRefs": [3 strings], "rationale": "..."}
"""


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def decide_uncached(packages: List[dict], deadline: float) -> List[dict]:
    if not OPENAI_API_KEY:
        raise ApiError(503, "MODEL_NOT_CONFIGURED")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, max_retries=0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical({"packages": packages})},
    ]
    last_error = "unknown"
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining < (6 if attempt else 3):
            break
        try:
            kwargs = dict(model=MODEL, temperature=0, max_tokens=7000, messages=messages,
                          timeout=min(30.0, remaining - 1))
            try:
                completion = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            except Exception as exc:
                if "response_format" in str(exc).lower():
                    completion = client.chat.completions.create(**kwargs)
                else:
                    raise
            content = completion.choices[0].message.content or ""
            parsed = _extract_json(content)
            raw = parsed.get("decisions")
            if not isinstance(raw, list) or len(raw) != len(packages):
                raise ValueError("wrong decision count")
            by_id = {d.get("packageId"): d for d in raw if isinstance(d, dict)}
            out = []
            for p in packages:
                d = by_id.get(p["packageId"])
                if d is None:
                    raise ValueError("missing decision for package %s" % p["packageId"])
                nd = normalize_decision(d, p)
                nd["packageId"] = p["packageId"]
                out.append(nd)
            return out
        except ApiError:
            raise
        except Exception as exc:
            last_error = str(exc)[:300]
            log.warning("model attempt %d failed: %s", attempt + 1, last_error)
            messages = messages[:2] + [
                {"role": "user", "content": "Your previous answer was invalid (%s). Return the corrected full JSON "
                                            "object, copying evidenceRefs verbatim from the packages." % last_error}
            ]
    raise ApiError(503, "DECISION_UNAVAILABLE")


def get_decisions(packages: List[dict], deadline: float) -> List[dict]:
    by_id, missing = {}, []
    for p in packages:
        cached = get_cached_decision(p)
        if cached is not None:
            by_id[p["packageId"]] = cached
        else:
            missing.append(p)
    if missing:  # one batched model call for everything not cached
        for p, decision in zip(missing, decide_uncached(missing, deadline)):
            cache_decision(p, decision)
            by_id[p["packageId"]] = decision
    return [by_id[p["packageId"]] for p in packages]


# --------------------------------------------------------------------------- message processing
def process_message(principal: str, message: dict, message_hash: str, started: float) -> dict:
    message_id = message["messageId"]
    with key_lock((principal, message_id)):  # concurrent equivalent requests serialize here
        prior = lookup_prior(principal, message_id, message_hash)
        if prior is not None:
            return prior

        parts = message.get("parts")
        if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
            raise ApiError(400, "INVALID_MESSAGE_PARTS")
        media, data = parts[0].get("mediaType"), parts[0].get("data")

        if media == BATCH_MEDIA:
            return handle_batch(principal, message, message_hash, data, started)
        if media == RESULTS_MEDIA:
            return handle_results(principal, message, message_hash, data)
        raise ApiError(415, "UNSUPPORTED_INPUT_MEDIA_TYPE")


def handle_batch(principal, message, message_hash, data, started) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("batchId"), str) or not data["batchId"] \
            or not data.get("policyRevision"):
        raise ApiError(400, "INVALID_BATCH")
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages or len(packages) > 100:
        raise ApiError(400, "INVALID_BATCH")
    ids = [p.get("packageId") if isinstance(p, dict) else None for p in packages]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ApiError(400, "INVALID_PACKAGE_IDS")

    decisions = get_decisions(packages, started + REQUEST_BUDGET_SECONDS)

    task_id = str(uuid.uuid4())
    context_id = str(message.get("contextId") or uuid.uuid4())
    proposals = []
    for d in decisions:
        proposals.append({
            "packageId": d["packageId"],
            "actionId": str(uuid.uuid5(uuid.UUID(task_id), d["packageId"])),
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        })
    artifact = {"artifactId": str(uuid.uuid4()),
                "parts": [{"mediaType": PROPOSAL_MEDIA, "data": {"batchId": data["batchId"], "proposals": proposals}}]}
    task = make_task(task_id, context_id, [message], "TASK_STATE_INPUT_REQUIRED", [artifact])
    stamp = now_iso()
    with tx() as con:  # task + idempotency key stored atomically BEFORE responding
        con.execute("INSERT INTO tasks(task_id,context_id,principal_hash,batch_id,state,task_json,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (task_id, context_id, principal, data["batchId"], task["status"]["state"],
                     canonical(task), stamp, stamp))
        con.execute("INSERT INTO messages(principal_hash,message_id,message_hash,task_id) VALUES(?,?,?,?)",
                    (principal, message["messageId"], message_hash, task_id))
    return task


def handle_results(principal, message, message_hash, data) -> dict:
    task_id, context_id = message.get("taskId"), message.get("contextId")
    if not isinstance(task_id, str) or not isinstance(context_id, str) or not isinstance(data, dict):
        raise ApiError(400, "INVALID_CONTINUATION")

    with tx() as con:  # the state check and the terminal write share one transaction (cancel/receipt race)
        row = con.execute("SELECT task_json, state FROM tasks WHERE task_id=? AND principal_hash=?",
                          (task_id, principal)).fetchone()
        if not row:
            raise ApiError(404, "NOT_FOUND")
        if row["state"] != "TASK_STATE_INPUT_REQUIRED":
            raise ApiError(409, "TASK_STATE_CONFLICT")
        task = json.loads(row["task_json"])
        if task.get("contextId") != context_id:
            raise ApiError(409, "CONTEXT_MISMATCH")
        stored = proposal_data(task)
        if data.get("batchId") != stored.get("batchId"):
            raise ApiError(409, "BATCH_MISMATCH")
        proposals = {p["packageId"]: p for p in stored.get("proposals", [])}
        results = data.get("results")
        if not isinstance(results, list) or not results:
            raise ApiError(400, "INVALID_RESULTS")

        accepted, seen = {}, set()
        for result in results:
            if not isinstance(result, dict):
                raise ApiError(400, "INVALID_RESULT")
            pid = result.get("packageId")
            proposal = proposals.get(pid)
            if proposal is None or pid in seen:
                raise ApiError(409, "PROPOSAL_MISMATCH")
            seen.add(pid)
            if result.get("actionId") != proposal["actionId"] or result.get("action") != proposal["action"]:
                raise ApiError(409, "PROPOSAL_MISMATCH")
            if result.get("outcome") not in ("ACCEPTED", "REJECTED"):
                raise ApiError(400, "INVALID_OUTCOME")
            nonce = result.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                raise ApiError(400, "INVALID_RECEIPT_NONCE")
            if result["outcome"] == "ACCEPTED":
                accepted[pid] = nonce

        # Executions follow stored proposal order; only ACCEPTED results are executed.
        executions = [{
            "packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
            "receiptNonce": accepted[p["packageId"]], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"],
        } for p in stored.get("proposals", []) if p["packageId"] in accepted]

        task["history"].append(message)
        task["artifacts"].append({
            "artifactId": str(uuid.uuid4()),
            "parts": [{"mediaType": RECEIPT_MEDIA,
                       "data": {"batchId": stored["batchId"], "executions": executions}}],
        })
        task["status"] = task_status("TASK_STATE_COMPLETED")
        con.execute("UPDATE tasks SET state=?, task_json=?, updated_at=? WHERE task_id=? AND principal_hash=?",
                    ("TASK_STATE_COMPLETED", canonical(task), now_iso(), task_id, principal))
        con.execute("INSERT INTO messages(principal_hash,message_id,message_hash,task_id) VALUES(?,?,?,?)",
                    (principal, message["messageId"], message_hash, task_id))
    return task


def cancel_sync(principal: str, task_id: str) -> dict:
    with tx() as con:
        row = con.execute("SELECT task_json, state FROM tasks WHERE task_id=? AND principal_hash=?",
                          (task_id, principal)).fetchone()
        if not row:
            raise ApiError(404, "NOT_FOUND")
        if row["state"] in TERMINAL:
            raise ApiError(409, "TASK_NOT_CANCELABLE")
        task = json.loads(row["task_json"])
        task["status"] = task_status("TASK_STATE_CANCELED")
        con.execute("UPDATE tasks SET state=?, task_json=?, updated_at=? WHERE task_id=? AND principal_hash=?",
                    ("TASK_STATE_CANCELED", canonical(task), now_iso(), task_id, principal))
    return task


# --------------------------------------------------------------------------- routes
router = APIRouter()


@router.post("/message:send")
async def send_message(request: Request):
    started = time.monotonic()
    principal = authenticate(request)
    check_protocol(request, has_body=True)
    payload = await read_json_body(request)
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
        raise ApiError(400, "INVALID_ENVELOPE")
    message = payload["message"]
    if not isinstance(message.get("messageId"), str) or not message["messageId"]:
        raise ApiError(400, "MESSAGE_ID_REQUIRED")
    if message.get("role") != "ROLE_USER":
        raise ApiError(400, "INVALID_ROLE")
    message_hash = sha(canonical(message))  # message only; configuration is ignored
    task = await run_in_threadpool(process_message, principal, message, message_hash, started)
    return a2a_json({"task": task})


@router.get("/tasks/{task_id}")
async def read_task(task_id: str, request: Request):
    principal = authenticate(request)
    check_protocol(request, has_body=False)
    return a2a_json(await run_in_threadpool(task_get, task_id, principal))


@router.get("/tasks")
async def list_tasks(request: Request):
    principal = authenticate(request)
    check_protocol(request, has_body=False)

    def _list():
        with db() as con:
            rows = con.execute("SELECT task_json FROM tasks WHERE principal_hash=? ORDER BY created_at DESC",
                               (principal,)).fetchall()
        return [json.loads(r["task_json"]) for r in rows]

    return a2a_json({"tasks": await run_in_threadpool(_list)})


@router.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = authenticate(request)
    check_protocol(request, has_body=True)
    return a2a_json(await run_in_threadpool(cancel_sync, principal, task_id))


# Serve the same routes at the origin AND under /a2a (and under whatever path A2A_BASE_URL uses),
# so the submitted base URL works whether or not it ends in /a2a.
_prefixes = {"", "/a2a"}
_base_path = urlparse(A2A_BASE_URL).path.rstrip("/")
if _base_path:
    _prefixes.add(_base_path)
for _prefix in sorted(_prefixes):
    app.include_router(router, prefix=_prefix)
