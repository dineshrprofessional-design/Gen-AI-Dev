"""Week 10: the docs squad. An orchestrator delegates to two narrow workers over HTTP.

    question -> orchestrator (LLM) --HTTP--> version/deprecation worker (LLM + 3 MCP tools)
                     |          \\--HTTP--> code-sample worker        (LLM + search_docs)
                     v
               final answer in the Week 7 answer contract, like the single agent's

## What each part may do

    orchestrator          tools: delegate_version_check, delegate_code_sample.
                          It cannot search anything itself; it decides, delegates
                          and writes the answer from the workers' reports.
    version/deprecation   tools: search_docs, get_package_versions,
      worker              get_package_deprecation. It reports the target version
                          and what differs or is deprecated. It does not answer.
    code-sample worker    tools: search_docs only. It reports the documented
                          facts and any documented code, in the target version.

The narrow tool sets are the point. Giving the code-sample worker every tool the
single agent has would remove the one plausible source of a win (the brief's
warning). Tools come from the same MCP client the single agent uses, filtered
by an allow-list. A worker asking for a tool outside its list gets an error.

## Context is passed on purpose

The orchestrator hands the code-sample worker the task, the target version and
the version worker's notes: enough not to lose the version (the brief's other
warning), and never whole reference pages. A worker's search results stay in
the worker. The orchestrator gets only the worker's short report back.

## The HTTP boundary, and the injected failure

Workers sit behind a FastAPI app (`build_worker_app`). The orchestrator reaches
it through `starlette.testclient.TestClient`, which is httpx speaking real HTTP
to the app in-process. A `FailurePlan` makes one worker return a genuine
`500 Internal Server Error` for one case. Its body is the generic FastAPI one,
so the orchestrator cannot tell the failure was injected. No HTTP-level retry
exists. If the orchestrator asks again, that is the model's own decision, and
the hand-off log shows it.

## Token accounting, per hop

Every LLM call is metered from the provider's own usage numbers, and every call
belongs to exactly one hop:

    question -> orchestrator          the orchestrator's first lap
    orchestrator -> <worker>          all of that worker's LLM calls for one hand-off
    <worker> -> orchestrator          the orchestrator's next lap, which re-sends the
                                      conversation including that worker's report

The hops sum exactly to the squad's total, which `SquadRun.check_accounting`
asserts.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.testclient import TestClient

from app.mcp.client import McpClient, ToolCallError
from app.rag.week7_agent import _assistant_message, _tool_message
from app.rag.week7_contract import (
    ANSWER_CONTRACT,
    ArmConfig,
    ChatResult,
    Meter,
    TaskOutput,
    ToolCall,
    chat_once,
    load_price,
    parse_task_output,
    user_prompt,
)
from app.rag.week7_questions import Week7Question
from app.rag.week9_agent import McpToolCallRecord, _capped, _chunk_ids, tool_payload

ChatFn = Callable[..., ChatResult]

ARM = "squad"
VERSION, CODE = "version", "code"
WORKER_LABEL = {VERSION: "version/deprecation worker", CODE: "code-sample worker"}


# --------------------------------------------------------------------------
# the workers
# --------------------------------------------------------------------------

VERSION_WORKER_PROMPT = """\
You are the version/deprecation worker in a documentation team. You have one \
job: establish which SDK version a developer question concerns, and what about \
it differs between versions or is deprecated.

Use your tools, and state nothing you have not seen a tool return. search_docs \
searches one documentation version at a time ("v2" or "v3"). The package tools \
report published releases and deprecation notices.

If the question names a version, that is the target. If it compares versions, \
the target is both. Otherwise the target is v3, the newest documented version.

Reply in exactly this format and nothing else:
TARGET_VERSION: <v2, v3 or both>
VERSION_NOTES: <what differs between v2 and v3 for what the question asks, with chunk ids in [brackets]; or: none found>
DEPRECATION: <any deprecation that applies, from a tool result; or: none found>

Do not answer the question itself. Another worker does that.
"""

CODE_WORKER_PROMPT = """\
You are the code-sample worker in a documentation team. You receive one task \
and the target SDK version. Find what the documentation says, and report it.

Use search_docs with api_version set to the target version. When the target is \
both, search each version separately. State nothing you have not seen a passage say.

Copy code only if a passage shows it, verbatim. Never compose a call the \
documentation does not show.

Reply in exactly this format and nothing else:
CONTENT: <the facts that answer the task, naming the version they belong to, with chunk ids in [brackets]>
CODE: <a code sample copied verbatim from a passage, or: none>

If the passages do not contain the answer, reply CONTENT: not documented, and CODE: none.
"""


@dataclass(frozen=True)
class WorkerSpec:
    key: str
    label: str
    prompt: str
    tools: tuple[str, ...]
    max_laps: int = 4

    def task_text(self, arguments: dict[str, Any]) -> str:
        if self.key == VERSION:
            return f"Developer question: {arguments.get('question', '')}"
        return (
            f"Task: {arguments.get('task', '')}\n"
            f"Target version: {arguments.get('target_version', '')}\n"
            f"Version notes from the version worker: {arguments.get('version_notes') or 'none'}"
        )


WORKERS = {
    VERSION: WorkerSpec(VERSION, WORKER_LABEL[VERSION], VERSION_WORKER_PROMPT,
                        ("search_docs", "get_package_versions", "get_package_deprecation")),
    CODE: WorkerSpec(CODE, WORKER_LABEL[CODE], CODE_WORKER_PROMPT, ("search_docs",)),
}


class WorkerResult(BaseModel):
    worker: str
    output: str = ""
    termination: str = ""
    tool_calls: list[McpToolCallRecord] = Field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    rate_limit_wait_s: float = 0.0
    latency_ms: float = 0.0
    transcript: list[dict[str, Any]] = Field(default_factory=list)


def _worker_tool_call(client: McpClient, spec: WorkerSpec, call: ToolCall, records: list, cfg: ArmConfig) -> dict:
    """One tool call from a worker. The allow-list is enforced here, not trusted to the prompt."""
    record = McpToolCallRecord(seq=len(records) + 1, name=call.name)
    records.append(record)
    started = time.perf_counter()
    if call.name not in spec.tools:
        payload = {"error": f"tool {call.name!r} is not available to the {spec.label}; available: {list(spec.tools)}"}
    else:
        try:
            arguments = json.loads(call.arguments_json) if call.arguments_json else {}
        except json.JSONDecodeError as exc:
            payload = {"error": f"arguments were not valid JSON: {exc}"}
        else:
            record.arguments = arguments if isinstance(arguments, dict) else {}
            record.server = client.server_for(call.name) or ""
            payload = tool_payload(client.call_tool(call.name, record.arguments))
            record.result_chunk_ids = _chunk_ids(payload)
    record.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if "error" in payload:
        record.ok, record.error = False, str(payload["error"])
    payload = _capped(payload, cfg.max_tool_result_chars)
    record.result_chars = len(json.dumps(payload, ensure_ascii=False))
    return payload


def run_worker(spec: WorkerSpec, arguments: dict[str, Any], client: McpClient, *,
               chat: ChatFn, cfg: ArmConfig, model: str, price: Any) -> WorkerResult:
    schemas = [s for s in client.openai_tools() if s["function"]["name"] in spec.tools]
    missing = set(spec.tools) - {s["function"]["name"] for s in schemas}
    if missing:
        raise RuntimeError(f"{spec.label}: tools {sorted(missing)} were not discovered over MCP")

    started = time.perf_counter()
    meter = Meter(price=price)
    records: list[McpToolCallRecord] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": spec.prompt},
        {"role": "user", "content": spec.task_text(arguments)},
    ]
    output, termination, waited = "", "max_laps", 0.0
    for _ in range(spec.max_laps):
        result = chat(messages, model=model, tools=schemas, tool_choice="auto",
                      temperature=cfg.temperature, retries=cfg.retries)
        meter.add(result)
        waited += result.rate_limit_wait_s
        if result.error:
            termination = f"chat_error: {result.error}"
            break
        if result.tool_calls:
            messages.append(_assistant_message(result))
            for call in result.tool_calls:
                messages.append(_tool_message(call, _worker_tool_call(client, spec, call, records, cfg)))
            continue
        output = result.content or ""
        messages.append({"role": "assistant", "content": output})
        termination = "report" if output else "empty_response"
        break
    return WorkerResult(
        worker=spec.key, output=output, termination=termination, tool_calls=records,
        llm_calls=meter.calls, prompt_tokens=meter.prompt_tokens, completion_tokens=meter.completion_tokens,
        cached_tokens=meter.cached_tokens, total_tokens=meter.total_tokens, cost_usd=meter.cost_usd,
        rate_limit_wait_s=waited, latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        transcript=messages,
    )


# --------------------------------------------------------------------------
# the HTTP boundary
# --------------------------------------------------------------------------


@dataclass
class FailurePlan:
    """Make one worker answer HTTP 500 for one case. Every attempt is counted."""

    worker: str
    case_id: str
    attempts: int = 0


class WorkerRequest(BaseModel):
    case_id: str
    hop: int
    arguments: dict[str, Any] = Field(default_factory=dict)


def build_worker_app(client: McpClient, *, chat: ChatFn, cfg: ArmConfig, model: str, price: Any,
                     failure: FailurePlan | None = None) -> FastAPI:
    app = FastAPI(title="docs squad workers")

    @app.post("/workers/{worker}")
    def handle(worker: str, request: WorkerRequest):
        spec = WORKERS.get(worker)
        if spec is None:
            return JSONResponse(status_code=404, content={"detail": f"no worker {worker!r}"})
        if failure and failure.worker == worker and failure.case_id == request.case_id:
            failure.attempts += 1
            # FastAPI's own body for an unhandled error, so nothing marks it as injected.
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
        try:
            result = run_worker(spec, request.arguments, client, chat=chat, cfg=cfg, model=model, price=price)
        except ToolCallError as exc:
            return JSONResponse(status_code=502, content={"detail": f"MCP tool call failed: {exc}"})
        return result.model_dump(mode="json")

    return app


# --------------------------------------------------------------------------
# the orchestrator
# --------------------------------------------------------------------------

ORCHESTRATOR_PREAMBLE = """\

You lead a team of two workers, available to you as tools. You cannot search \
the documentation yourself.

1. Call delegate_version_check with the developer's question. It reports the \
target version, and what differs between versions or is deprecated.
2. Call delegate_code_sample with the task, the target version from step 1, and \
the version notes. It reports the documented facts and any documented code \
sample, with chunk ids.
3. Write the answer from the workers' reports only, and cite the chunk ids they cite.
"""

ORCHESTRATOR_TOOLS = [
    {"type": "function", "function": {
        "name": "delegate_version_check",
        "description": "Ask the version/deprecation worker which SDK version the question concerns, and what "
                       "differs between versions or is deprecated. Returns TARGET_VERSION, VERSION_NOTES and DEPRECATION.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The developer's question, verbatim."}},
            "required": ["question"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "delegate_code_sample",
        "description": "Ask the code-sample worker for the documented facts and any documented code sample for "
                       "one task, in one target version. Returns CONTENT and CODE, with chunk ids.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "What to find out, in words."},
            "target_version": {"type": "string", "enum": ["v2", "v3", "both"],
                               "description": "The version the version worker reported. Required: the two "
                                              "versions document different defaults."},
            "version_notes": {"type": "string", "description": "The version worker's notes, so nothing it found is lost."}},
            "required": ["task", "target_version"], "additionalProperties": False}}},
]
DELEGATE = {"delegate_version_check": VERSION, "delegate_code_sample": CODE}


class Hop(BaseModel):
    case_id: str
    seq: int
    edge: str
    actor: str
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    http_status: int | None = None
    note: str = ""


class SquadRun(BaseModel):
    arm: str = ARM
    question_id: str
    question: str
    output: TaskOutput = Field(default_factory=TaskOutput)
    hops: list[Hop] = Field(default_factory=list)
    workers: list[WorkerResult] = Field(default_factory=list)
    delegations: list[dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    rate_limit_wait_s: float = 0.0
    termination: str = ""
    termination_detail: str = ""
    transcript: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def tool_calls(self) -> list[McpToolCallRecord]:
        return [record for worker in self.workers for record in worker.tool_calls]

    def check_accounting(self) -> list[str]:
        problems = []
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            hop_sum = sum(getattr(hop, name) for hop in self.hops)
            if hop_sum != getattr(self, name):
                problems.append(f"{name}: hops sum to {hop_sum}, run total is {getattr(self, name)}")
        return problems


def run_squad(question: Week7Question, http: TestClient, cfg: ArmConfig | None = None, *,
              chat: ChatFn = chat_once, price: Any = None, max_laps: int = 6) -> SquadRun:
    """One question through the squad. Never raises; every ending is a termination."""
    cfg = cfg or ArmConfig()
    model = cfg.resolved_model()
    price = price if price is not None else load_price(model)
    meter = Meter(price=price)
    run = SquadRun(question_id=question.id, question=question.question)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ANSWER_CONTRACT + ORCHESTRATOR_PREAMBLE},
        {"role": "user", "content": user_prompt(question)},
    ]
    started = time.perf_counter()
    waited = 0.0
    incoming = "question"  # what arrived since the orchestrator's previous lap

    def hop(**kwargs) -> Hop:
        h = Hop(case_id=question.id, seq=len(run.hops) + 1, **kwargs)
        run.hops.append(h)
        return h

    for _ in range(max_laps):
        result = chat(messages, model=model, tools=ORCHESTRATOR_TOOLS, tool_choice="auto",
                      temperature=cfg.temperature, retries=cfg.retries)
        lap = meter.add(result)
        waited += result.rate_limit_wait_s
        decided = ("delegates to " + ", ".join(WORKER_LABEL.get(DELEGATE.get(c.name, ""), c.name) for c in result.tool_calls)
                   if result.tool_calls else "final answer" if result.content else "no output")
        hop(edge=f"{incoming} -> orchestrator", actor="orchestrator", llm_calls=1,
            prompt_tokens=lap.prompt_tokens, completion_tokens=lap.completion_tokens, cached_tokens=lap.cached_tokens,
            total_tokens=lap.tokens, cost_usd=round(price.cost(lap.prompt_tokens, lap.completion_tokens, lap.cached_tokens), 8),
            latency_ms=lap.latency_ms, note=decided)
        if result.error:
            run.termination, run.termination_detail = "chat_error", result.error
            break
        if not result.tool_calls:
            if result.content:
                messages.append({"role": "assistant", "content": result.content})
                run.output = parse_task_output(result.content)
                run.termination = "final_answer"
                run.termination_detail = run.output.parse_error or "orchestrator answered"
            else:
                run.termination, run.termination_detail = "empty_response", "orchestrator returned nothing"
            break

        messages.append(_assistant_message(result))
        senders = []
        for call in result.tool_calls:
            worker = DELEGATE.get(call.name)
            if worker is None:
                messages.append(_tool_message(call, {"error": f"unknown tool {call.name!r}; available: {list(DELEGATE)}"}))
                continue
            try:
                arguments = json.loads(call.arguments_json) if call.arguments_json else {}
                if not isinstance(arguments, dict):
                    raise ValueError(f"arguments must be a JSON object, got {type(arguments).__name__}")
            except (json.JSONDecodeError, ValueError) as exc:
                messages.append(_tool_message(call, {"error": f"invalid arguments: {exc}"}))
                continue
            label = WORKER_LABEL[worker]
            senders.append(label)
            t0 = time.perf_counter()
            response = http.post(f"/workers/{worker}", json={"case_id": question.id, "hop": len(run.hops) + 1,
                                                             "arguments": arguments})
            elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
            run.delegations.append({"worker": worker, "arguments": arguments, "http_status": response.status_code})
            if response.status_code == 200:
                done = WorkerResult(**response.json())
                run.workers.append(done)
                waited += done.rate_limit_wait_s
                hop(edge=f"orchestrator -> {label}", actor=label, llm_calls=done.llm_calls,
                    prompt_tokens=done.prompt_tokens, completion_tokens=done.completion_tokens,
                    cached_tokens=done.cached_tokens, total_tokens=done.total_tokens, cost_usd=done.cost_usd,
                    latency_ms=elapsed, http_status=200,
                    note=f"{len(done.tool_calls)} tool call(s), termination {done.termination}")
                content = {"worker": label, "report": done.output or f"(no report: {done.termination})"}
            else:
                detail = response.json().get("detail", response.text) if response.content else ""
                hop(edge=f"orchestrator -> {label}", actor=label, latency_ms=elapsed,
                    http_status=response.status_code, note=f"HTTP {response.status_code}: {detail}")
                content = {"error": f"HTTP {response.status_code} from the {label}: {detail}"}
            messages.append(_tool_message(call, content))
        incoming = " + ".join(senders) if senders else "tool error"
    else:
        run.termination, run.termination_detail = "max_laps", f"no final answer within {max_laps} orchestrator laps"

    if run.termination != "final_answer":
        run.output = TaskOutput(parse_error=f"{run.termination}: {run.termination_detail}")
    for name in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens"):
        setattr(run, name, sum(getattr(h, name) for h in run.hops))
    run.cost_usd = round(sum(h.cost_usd for h in run.hops), 8)
    run.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    run.rate_limit_wait_s = waited
    run.transcript = messages
    return run
