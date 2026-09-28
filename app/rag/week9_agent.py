"""The Week 9 agent: Week 7's tool-calling loop, with its tools discovered over MCP.

    .venv/bin/python -m app.rag.week9_agent --question w7-01
    .venv/bin/python -m app.rag.week9_agent --question w7-01 --detail \\
        --out report/w9/step4_run_w7-01.json --wire report/w9/step4_mcp_messages_w7-01.jsonl
    .venv/bin/python -m app.rag.week9_agent --question w7-04 --mcp-config config/mcp_servers.json

## What this agent knows, and what it discovers

It knows one thing about tools: the path of an MCP config file naming the servers
to start. Everything else (how many tools there are, their names, descriptions
and input schemas) comes from each server's live `tools/list`, through
`app.mcp.client`. No tool name and no server name appears in this file. That is
what should let a second server be added later by editing the config and
nothing else.

    question -> McpClient: initialize, tools/list -> tools offered to the LLM
             -> LLM picks a tool   -> McpClient.call_tool -> tools/call -> server
             -> result appended    -> LLM again ... -> final answer

The LLM call and the MCP calls are separate paths. `chat_once` is an HTTPS call
to an OpenAI-compatible endpoint; `McpClient` is JSON-RPC over the stdin/stdout
of a local subprocess. The model never sees JSON-RPC, and no server sees the model.

## What is reused from Week 7, and why that is not tool coupling

    week7_contract   chat_once (the only model call site), system_prompt,
                     user_prompt, parse_task_output, Meter, load_price, score_output
    week7_agent      Budgets and its three check sites, LapTrace, and the two
                     message builders (the assistant tool-call echo, the tool reply)

None of these names a tool, holds a schema, or executes one; `_Run.ctx` only
needs a `.path`. Reusing them keeps a Week 9 run comparable to a Week 7 one:
same model call, same prompt, same parser, same four budgets, same scorer.
Where tools come from is the only thing that changes.

Importing those modules does load `week7_tools` into this process, because
Week 7's dispatch sits beside what is reused here. Nothing in this file refers
to it. A tool runs only through `McpClient.call_tool`, which means only in a
server process.

## What the model is shown

For a result carrying `structuredContent`, the model is shown that dict,
serialised and capped exactly as `week7_contract.run_tool` does it. For the Step 3
docs server that dict is what the Week 7 implementation returned, so the tool
message is byte-for-byte what Week 7 showed the model. An `isError` result becomes
`{"error": <text>}`, the shape Week 7's dispatch used for tool failures, so
failures read the same way too.

## Failure handling (basic; recoverable diagnostics are a later step)

    cannot start / initialize / tools/list    McpClient raises before any LLM call;
                                              the CLI prints it and exits 2
    two servers list the same tool            same, at connect time
    LLM names a tool nobody listed            {"error": ...} back to the model; no
                                              request is sent
    LLM sends arguments that are not JSON     {"error": ...} back to the model
    server answers isError: true              {"error": ...} back to the model
    tools/call gets no answer at all          run ends, termination="mcp_error"
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mcp.types as types
from pydantic import BaseModel, Field

from app.mcp.client import (
    DEFAULT_CONFIG,
    McpClient,
    McpClientError,
    ToolCallError,
    UnknownToolError,
    configure_logging,
    load_config,
)
from app.rag.week6_assertions import CHUNK_ID
from app.rag.week7_agent import (
    Budgets,
    BudgetTrip,
    LapTrace,
    _assistant_message,
    _Run,
    _tool_message,
    check_mid_lap,
    check_post_call,
    check_pre_call,
)
from app.rag.week7_contract import (
    ArmConfig,
    ChatResult,
    LapUsage,
    Meter,
    Score,
    TaskOutput,
    ToolCall,
    chat_once,
    load_price,
    parse_task_output,
    score_output,
    system_prompt,
    user_prompt,
)
from app.rag.week7_questions import DEFAULT_QUESTIONS_JSONL, Week7Question, load_questions

ARM = "week9_agent"

#: Week 7's four ways to end, plus one: the MCP server stopped answering.
TERMINATIONS = ("final_answer", "budget", "chat_error", "empty_response", "mcp_error")

log = logging.getLogger("week9.agent")

__all__ = ["McpToolCallRecord", "Week9Run", "run_agent", "tool_payload", "TERMINATIONS"]

ChatFn = Callable[..., ChatResult]


# --------------------------------------------------------------------------
# what a run records
# --------------------------------------------------------------------------


class McpToolCallRecord(BaseModel):
    """One tool call as the host saw it. `server` is the one that answered.

    `result_chunk_ids` is what `week7_contract.score_output` reads to check that
    every citation was actually returned by a tool in this run.
    """

    seq: int
    name: str
    server: str = ""              # "" when no connected server listed the name
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str = ""
    latency_ms: float = 0.0
    result_chars: int = 0
    result_chunk_ids: list[str] = Field(default_factory=list)


@dataclass
class _CallLog:
    """What `_Run` needs from a tool context: the calls, and their path."""

    records: list[McpToolCallRecord] = field(default_factory=list)

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.records)


class Week9Run(BaseModel):
    arm: str = ARM
    question_id: str
    question: str

    # Discovery, as it happened on this connection.
    servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    discovered_tools: list[dict[str, str]] = Field(default_factory=list)
    tools_offered: list[dict[str, Any]] = Field(default_factory=list)

    output: TaskOutput = Field(default_factory=TaskOutput)
    tool_calls: list[McpToolCallRecord] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)

    laps: list[LapTrace] = Field(default_factory=list)
    usage: list[LapUsage] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    usage_missing_calls: int = 0
    cost_usd: float = 0.0
    price_verified: bool = False

    latency_ms: float = 0.0
    rate_limit_wait_s: float = 0.0
    elapsed_s: float = 0.0

    termination: str = ""
    termination_detail: str = ""
    budgets: Budgets = Field(default_factory=lambda: Budgets())
    budget_trip: BudgetTrip | None = None

    transcript: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------
# MCP result -> what the model is shown
# --------------------------------------------------------------------------


def tool_payload(result: types.CallToolResult) -> dict[str, Any]:
    """The dict the model is shown for one MCP tool result. See the module docstring."""
    text = "\n".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    if result.isError:
        return {"error": text or "the tool reported an error without a message"}
    if result.structuredContent is not None:
        return dict(result.structuredContent)
    return {"content": text}


def _capped(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    # The same cap, same shape, as week7_contract.run_tool. That function also
    # dispatches to the Week 7 implementations, so it cannot be called here.
    rendered = json.dumps(payload, ensure_ascii=False)
    if len(rendered) > limit:
        return {
            "truncated": True,
            "note": f"result truncated at {limit} chars",
            "content": rendered[:limit],
        }
    return payload


def _chunk_ids(value: Any, found: list[str] | None = None) -> list[str]:
    """Every string in a result that is a corpus chunk id, wherever it sits.

    Keyed on the id format the answer contract cites, not on any one server's
    field names.
    """
    found = [] if found is None else found
    if isinstance(value, dict):
        for item in value.values():
            _chunk_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _chunk_ids(item, found)
    elif isinstance(value, str) and CHUNK_ID.match(value) and value not in found:
        found.append(value)
    return found


def _call_over_mcp(
    client: McpClient, call: ToolCall, calls: _CallLog, cfg: ArmConfig, lap: int,
) -> dict[str, Any]:
    """One model tool call -> one MCP tools/call -> the payload for the model.

    Raises ToolCallError only when the server gave no answer at all; everything
    else, including a tool the model made up, goes back to the model as an error.
    """
    record = McpToolCallRecord(seq=len(calls.records) + 1, name=call.name)
    calls.records.append(record)
    started = time.perf_counter()
    try:
        try:
            arguments = json.loads(call.arguments_json) if call.arguments_json else {}
        except json.JSONDecodeError as exc:
            payload = {"error": f"arguments were not valid JSON: {exc}"}
        else:
            if not isinstance(arguments, dict):
                payload = {"error": f"arguments must be a JSON object, got {type(arguments).__name__}"}
            else:
                record.arguments = arguments
                record.server = client.server_for(call.name) or ""
                try:
                    result = client.call_tool(call.name, arguments)
                except UnknownToolError as exc:
                    payload = {"error": str(exc)}
                else:
                    payload = tool_payload(result)
                    record.result_chunk_ids = _chunk_ids(payload)
    except ToolCallError as exc:
        record.ok, record.error = False, str(exc)
        raise
    finally:
        record.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

    if "error" in payload:
        record.ok, record.error = False, str(payload["error"])
    payload = _capped(payload, cfg.max_tool_result_chars)
    record.result_chars = len(json.dumps(payload, ensure_ascii=False))
    log.info(
        "lap %d: tool result received from %s: %s, %d chars, %d chunk id(s)",
        lap, record.server or "no server", "ok" if record.ok else f"error: {record.error}",
        record.result_chars, len(record.result_chunk_ids),
    )
    return payload


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def run_agent(
    question: Week7Question,
    client: McpClient,
    cfg: ArmConfig | None = None,
    *,
    chat: ChatFn = chat_once,
    budgets: Budgets | None = None,
    price: Any = None,
) -> Week9Run:
    """Run one question to a final answer, or to a clean non-answer.

    `client` must already be connected: discovery happened when it connected,
    and this function offers the LLM exactly what discovery found. Nothing
    raises out of here; every ending is a `termination` with a reason.
    """
    cfg = cfg or ArmConfig()
    budgets = budgets or Budgets()
    model = cfg.resolved_model()
    price = price if price is not None else load_price(model)
    meter = Meter(price=price)
    calls = _CallLog()

    # The only source of tools: what the connected servers answered to tools/list.
    schemas = client.openai_tools()
    offered = ", ".join(schema["function"]["name"] for schema in schemas) or "none"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("agent")},
        {"role": "user", "content": user_prompt(question)},
    ]
    state = _Run(
        budgets=budgets, meter=meter, ctx=calls,
        started=time.perf_counter(), messages=messages,
    )
    laps = state.laps
    output = TaskOutput()
    termination = ""
    detail = ""
    trip: BudgetTrip | None = None

    while not termination:
        trip = check_pre_call(state)
        if trip:
            termination, detail = "budget", trip.render()
            break

        lap = state.iterations + 1
        trace = LapTrace(
            lap=lap,
            messages_in=len(messages),
            tool_messages_in=sum(1 for m in messages if m.get("role") == "tool"),
        )
        log.info("lap %d: calling the LLM with %d discovered tool(s): %s", lap, len(schemas), offered)
        result = chat(
            messages,
            model=model,
            tools=schemas,
            tool_choice="auto",
            temperature=cfg.temperature,
            retries=cfg.retries,
        )
        meter.add(result)

        trace.prompt_tokens = result.prompt_tokens
        trace.completion_tokens = result.completion_tokens
        trace.cached_tokens = result.cached_tokens
        trace.usage_missing = result.usage_missing
        trace.latency_ms = round(result.latency_ms, 3)
        trace.rate_limit_wait_s = result.rate_limit_wait_s
        trace.content_chars = len(result.content or "")
        trace.tool_calls_out = len(result.tool_calls)
        trace.tools_called = [call.name for call in result.tool_calls]
        trace.finish = (
            "error" if result.error
            else "tool_calls" if result.tool_calls
            else "content" if result.content
            else "empty"
        )
        laps.append(trace)

        if result.error:
            termination, detail = "chat_error", result.error
            break

        trip = check_post_call(state)
        if trip:
            termination, detail = "budget", trip.render()
            break

        if result.tool_calls:
            messages.append(_assistant_message(result))
            for call in result.tool_calls:
                log.info("lap %d: LLM selected tool %s with arguments %s", lap, call.name, call.arguments_json)
                try:
                    payload = _call_over_mcp(client, call, calls, cfg, lap)
                except ToolCallError as exc:
                    termination, detail = "mcp_error", str(exc)
                    log.info("lap %d: stopping: %s", lap, exc)
                    break
                messages.append(_tool_message(call, payload))
                trip = check_mid_lap(state)
                if trip:
                    termination, detail = "budget", trip.render()
                    break
            continue

        if result.content:
            messages.append({"role": "assistant", "content": result.content})
            output = parse_task_output(result.content)
            termination = "final_answer"
            detail = output.parse_error or "model answered without requesting more tools"
            log.info("lap %d: final answer generated (the LLM requested no further tool)", lap)
            break

        termination = "empty_response"
        detail = "model returned neither content nor a tool call"

    if termination != "final_answer":
        output = TaskOutput(parse_error=f"{termination}: {detail}")
        log.info("run ended without an answer: %s: %s", termination, detail)

    return Week9Run(
        question_id=question.id,
        question=question.question,
        servers=client.server_info,
        discovered_tools=[{"name": tool.name, "server": tool.server} for tool in client.tools],
        tools_offered=schemas,
        output=output,
        tool_calls=list(calls.records),
        tool_sequence=list(calls.path),
        laps=laps,
        usage=list(meter.laps),
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        cached_tokens=meter.cached_tokens,
        total_tokens=meter.total_tokens,
        usage_missing_calls=meter.usage_missing_calls,
        cost_usd=meter.cost_usd,
        price_verified=price.verified,
        latency_ms=round(state.raw_elapsed_s * 1000.0, 3),
        rate_limit_wait_s=sum(lap.rate_limit_wait_s for lap in laps),
        elapsed_s=round(state.elapsed_s, 3),
        termination=termination,
        termination_detail=detail,
        budgets=budgets,
        budget_trip=trip,
        transcript=messages,
    )


# --------------------------------------------------------------------------
# rendering + CLI
# --------------------------------------------------------------------------


def render_run(run: Week9Run, question: Week7Question, score: Score, detail: bool = False) -> str:
    lines = [f"question    {run.question_id}  {run.question}"]
    for name, info in run.servers.items():
        server = info.get("serverInfo", {})
        lines.append(
            f"server      {name}  initialize ok: protocol {info.get('protocolVersion')}, "
            f"{server.get('name')} {server.get('version')}"
        )
    lines.append(
        f"discovered  {len(run.discovered_tools)} tool(s) via tools/list: "
        + (", ".join(f"{t['name']}@{t['server']}" for t in run.discovered_tools) or "none")
    )
    lines.append("")
    for lap in run.laps:
        asked = ", ".join(lap.tools_called) or "-"
        lines.append(
            f"lap {lap.lap}  LLM in: {lap.messages_in} msgs ({lap.tool_messages_in} tool)  "
            f"out: {lap.finish:<10} tools: {asked:<20} "
            f"tokens {lap.prompt_tokens}+{lap.completion_tokens}  {lap.latency_ms / 1000:.1f}s"
        )
    for record in run.tool_calls:
        lines.append(
            f"  tools/call #{record.seq} {record.name} -> {record.server or '(no server)'}  "
            f"{json.dumps(record.arguments, ensure_ascii=False)}  "
            f"{'ok' if record.ok else 'ERROR ' + record.error}  {record.latency_ms / 1000:.1f}s  "
            f"{record.result_chars} chars  chunk ids: {', '.join(record.result_chunk_ids) or '-'}"
        )
    lines += [
        "",
        f"termination {run.termination}  ({run.termination_detail})",
        f"answer      {run.output.answer or '-'}",
        f"version     {run.output.sdk_version}    citations: {', '.join(run.output.citations) or 'none'}",
        f"path        {list(run.tool_sequence)}    (Week 7 expected: {question.expected_order})",
        f"tokens      {run.total_tokens:,} ({run.prompt_tokens:,} prompt + {run.completion_tokens:,} completion)"
        f"    cost ${run.cost_usd:.6f}    {run.latency_ms / 1000:.1f}s",
        f"score       {'PASS' if score.passed else 'FAIL'}  "
        + "  ".join(f"{c.name}={'ok' if c.ok else 'FAIL'}" for c in score.checks if c.applicable),
    ]
    if detail:
        lines += ["", "transcript:"]
        for message in run.transcript:
            lines.append(json.dumps(message, ensure_ascii=False)[:1500])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer one Week 7 question with tools discovered over MCP.")
    parser.add_argument("--question", required=True, help="a Week 7 question id, e.g. w7-01")
    parser.add_argument("--mcp-config", type=Path, default=DEFAULT_CONFIG, help="which MCP servers to start")
    parser.add_argument("--detail", action="store_true", help="also print the message transcript")
    parser.add_argument("--out", type=Path, help="write the run and its score as JSON")
    parser.add_argument("--wire", type=Path, help="write every JSON-RPC message to this JSONL file")
    args = parser.parse_args(argv)
    configure_logging()

    questions = {q.id: q for q in load_questions(DEFAULT_QUESTIONS_JSONL)}
    question = questions.get(args.question)
    if question is None:
        parser.error(f"unknown question {args.question!r}; known: {', '.join(questions)}")

    try:
        with McpClient(load_config(args.mcp_config)) as client:
            log.info("question %s: %s", question.id, question.question)
            run = run_agent(question, client)
            wire = client.wire
    except McpClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    score = score_output(question, run.output, run.tool_calls)
    print(render_run(run, question, score, detail=args.detail))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"run": run.model_dump(mode="json"), "score": score.model_dump(mode="json")},
                indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}", file=sys.stderr)
    if args.wire:
        wire.write_jsonl(args.wire)
        print(f"wrote {len(wire.messages)} JSON-RPC messages to {args.wire}", file=sys.stderr)
    return 0 if run.termination == "final_answer" else 1


if __name__ == "__main__":
    raise SystemExit(main())
