"""The Week 7 tool-calling agent — the first arm of the race.

    python -m app.rag.week7_agent --self-test              # the loop, offline, no network
    python -m app.rag.week7_agent --validate               # static guards + self-test
    python -m app.rag.week7_agent --question w7-07 --detail
    python -m app.rag.week7_agent --questions w7-01,w7-07,w7-10 --out report/w7/agent_probe.json

One real loop: the model sees the question and the three tool schemas, decides
whether to call a tool, the result is appended to the conversation, and the model
decides again. It stops when the model answers without asking for another tool.

## What this module may not do

Both arms reach the outside world only through `week7_contract`. This file
imports `chat_once`, `run_tool`, `system_prompt`, `user_prompt`,
`parse_task_output` and `Meter` from there and never touches `openai`,
`app.rag.generate` or `app.rag.index` — mechanically checked by
`week7_contract.imports_are_clean()`, which already runs inside that module's
`--validate`. Tool schemas come from `week7_tools.tool_schemas()`, the same
object the workflow arm will send.

## The loop must actually decide

An agent whose tool order is nudged toward the answer key measures nothing. So
`tool_choice` stays `"auto"` on every lap, the question's `expected_tools` /
`expected_order` / `required_facts` are never read by the loop, and there is no
per-question branching. That is not a promise — `loop_is_unforced()` parses
`run_agent`'s own source and fails if any of those names appear in it or if
`tool_choice` is ever pinned. The CLI *displays* a comparison against the
expected order, which is why the scan is scoped to the loop rather than the file.

## Four budgets, all four checked

The brief's named failure is "defining MAX_ITERS, MAX_TOKENS, MAX_COST as
constants and never checking three of them". So each has a named check site and
each is proved to fire offline:

    check_pre_call    top of the lap      iterations, tokens (PROJECTED), cost,
                                          wall clock
    check_post_call   after meter.add     tokens, cost, wall clock
    check_mid_lap     between tool calls  wall clock

`check_pre_call` projects rather than measures: the loop re-sends the whole
message list, so the next prompt is at least as large as the last one, and
starting a call you already cannot afford burns the budget you were protecting.

Wall clock **excludes provider rate-limit sleep**. A 14 s 429 backoff is a
property of the account, not of either architecture; charging it to the agent
would fire the budget and the termination record would say nothing about the
loop. Raw and adjusted elapsed are both kept.

No exception escapes. A trip appends a structured `BudgetTrip` — which budget,
which check site, limit, observed, iterations, tokens, cost, elapsed, message
count, tool path — and `run_agent` returns normally with `termination="budget"`
and `budget_type` naming it. The scorer marks it a fail: an agent that ran out
of budget did not answer the question.

`max_iterations` is also what stops a model that calls a tool every lap, so the
loop is guaranteed to return without a separate structural halt.

## Token accounting

`meter.add()` is called on **every** lap, including a lap that errored, so lap N
of the usage table is lap N of the trace. The brief names understating agent cost
as a specific failure: the loop re-sends the whole message list each lap, so the
totals must be summed, never read off the last call. `verify_tool_feedback()`
checks the arithmetic from the other side — `prompt_tokens` must *grow* lap over
lap, because the previous lap's tool results are now in the request.
"""

import argparse
import ast
import copy
import hashlib
import inspect
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.rag.week7_contract import (
    ArmConfig,
    ChatResult,
    LapUsage,
    Meter,
    TaskOutput,
    ToolCall,
    chat_once,
    contract_shas,
    load_price,
    new_tool_context,
    parse_task_output,
    run_tool,
    score_output,
    system_prompt,
    user_prompt,
)
from app.rag.week7_questions import (
    DEFAULT_QUESTIONS_JSONL,
    Week7Question,
    load_questions,
)
from app.rag.week7_tools import ToolCallRecord, tool_schemas

ARM = "agent"

#: Every way the loop can end. Each is a returned value, never an exception.
TERMINATIONS = ("final_answer", "budget", "chat_error", "empty_response")

#: The four budgets, in the order they are checked.
BUDGET_NAMES = ("max_iterations", "max_tokens", "max_cost_usd", "max_wall_clock_s")

CHECK_SITES = ("pre_call", "post_call", "mid_lap")

__all__ = [
    "AgentRun", "LapTrace", "Budgets", "BudgetTrip", "run_agent", "TERMINATIONS",
    "BUDGET_NAMES", "check_pre_call", "check_post_call", "check_mid_lap",
    "verify_tool_feedback", "loop_is_unforced", "budget_coverage",
]


class Budgets(BaseModel):
    """Ceilings on one question's run. Every one of the four is enforced.

    The defaults sit between the largest legitimate run and the pathological one
    actually observed in Step 8, so they bound a runaway without touching normal
    operation:

        largest legitimate   w7-07   3 iters   4,484 tok   $0.00088    21.2 s
        observed pathology   w7-10   8 iters  35,988 tok   $0.00524   187.9 s

    Note `max_iterations` at 8 would NOT have stopped that thrash — it ran to
    exactly 8 laps — while `max_tokens` and `max_wall_clock_s` would have. That
    is the argument for enforcing four rather than one: the cheapest budget to
    write is the one least likely to bind.
    """

    max_iterations: int = 8
    max_tokens: int = 25_000
    max_cost_usd: float = 0.01
    max_wall_clock_s: float = 120.0

    def sha(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:16]


class BudgetTrip(BaseModel):
    """Why the run stopped, with everything needed to audit the decision."""

    budget: str
    check_site: str
    limit: float
    observed: float
    projected: bool = False
    iterations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    raw_elapsed_s: float = 0.0
    rate_limit_wait_s: float = 0.0
    messages_len: int = 0
    path: list[str] = Field(default_factory=list)

    def render(self) -> str:
        # "reached" vs "exceeded" is derived rather than hard-coded: iterations
        # and wall clock trip on >=, tokens and cost on >, and a record that
        # claimed the wrong comparator would misreport an at-the-limit stop.
        how = "exceeded" if self.observed > self.limit else "reached"
        arrow = "projected" if self.projected else "observed"
        return (
            f"{self.budget} tripped at {self.check_site}: "
            f"{arrow} {self.observed:,.4f} {how} limit {self.limit:,.4f}"
        )


# --------------------------------------------------------------------------
# what a run exposes
# --------------------------------------------------------------------------


class LapTrace(BaseModel):
    """One turn of the loop: what went in, what the model asked for.

    `messages_in` and `tool_messages_in` are counted at request time, before the
    lap appends anything, so they are the evidence that lap N saw lap N-1's tool
    results rather than an assertion that it did.
    """

    lap: int
    messages_in: int
    tool_messages_in: int
    tool_calls_out: int = 0
    tools_called: list[str] = Field(default_factory=list)
    content_chars: int = 0
    finish: str = ""              # tool_calls | content | empty | error
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    usage_missing: bool = False
    latency_ms: float = 0.0
    rate_limit_wait_s: float = 0.0


class AgentRun(BaseModel):
    """Everything the race, the scorer and a grader need from one question."""

    arm: str = ARM
    question_id: str
    question: str

    output: TaskOutput = Field(default_factory=TaskOutput)

    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
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

    termination: str = ""
    termination_detail: str = ""
    budgets: Budgets = Field(default_factory=lambda: Budgets())
    budget_trip: BudgetTrip | None = None
    elapsed_s: float = 0.0

    arm_config_sha: str = ""
    tool_config_sha: str = ""
    shas: dict[str, str] = Field(default_factory=dict)

    transcript: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.output.answer

    @property
    def termination_reason(self) -> str:
        """`termination`, under the name the brief asks for."""
        return self.termination

    @property
    def budget_type(self) -> str:
        """Which budget stopped the run, or "" if a budget did not."""
        return self.budget_trip.budget if self.budget_trip else ""

    @property
    def stopped_by_budget(self) -> bool:
        return self.budget_trip is not None

    @property
    def llm_calls(self) -> int:
        """Calls to the model. One per lap by construction."""
        return len(self.laps)

    @property
    def iterations(self) -> int:
        """Turns of the loop. Identical to `llm_calls`; both are reported
        because a run where they diverged would mean a hidden call site."""
        return len(self.laps)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def adjusted_latency_ms(self) -> float:
        """Latency with provider throttling removed.

        Groq's free tier caps this model at 8000 tokens/minute. A 429 backoff is
        a property of the account, not of either architecture, so it is
        subtracted here and — identically — for the workflow arm.
        """
        return self.latency_ms - self.rate_limit_wait_s * 1000.0

    @property
    def path(self) -> tuple[str, ...]:
        """The tuple the verdict's path-variation analysis will read."""
        return tuple(self.tool_sequence)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

ChatFn = Callable[..., ChatResult]


def _assistant_message(result: ChatResult) -> dict[str, Any]:
    """Echo the model's tool-call turn back, in the shape the API expects."""
    return {
        "role": "assistant",
        "content": result.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in result.tool_calls
        ],
    }


def _tool_message(call: ToolCall, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(payload, ensure_ascii=False),
    }


# --------------------------------------------------------------------------
# the three budget check sites
# --------------------------------------------------------------------------


@dataclass
class _Run:
    """Live state the check sites read. One object, so no budget reads a stale
    copy of the numbers another one updated."""

    budgets: Budgets
    meter: Meter
    ctx: Any
    started: float
    messages: list[dict[str, Any]]
    laps: list[LapTrace] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.laps)

    @property
    def raw_elapsed_s(self) -> float:
        return time.perf_counter() - self.started

    @property
    def rate_limit_wait_s(self) -> float:
        return sum(lap.rate_limit_wait_s for lap in self.laps)

    @property
    def elapsed_s(self) -> float:
        """Wall clock the agent is answerable for: provider throttling removed."""
        return self.raw_elapsed_s - self.rate_limit_wait_s

    def trip(
        self, budget: str, site: str, limit: float, observed: float,
        projected: bool = False,
    ) -> BudgetTrip:
        return BudgetTrip(
            budget=budget,
            check_site=site,
            limit=float(limit),
            observed=float(observed),
            projected=projected,
            iterations=self.iterations,
            tokens=self.meter.total_tokens,
            cost_usd=round(self.meter.cost_usd, 8),
            elapsed_s=round(self.elapsed_s, 3),
            raw_elapsed_s=round(self.raw_elapsed_s, 3),
            rate_limit_wait_s=round(self.rate_limit_wait_s, 3),
            messages_len=len(self.messages),
            path=list(self.ctx.path),
        )


def check_pre_call(state: _Run) -> BudgetTrip | None:
    """Before starting a lap. All four, and tokens/cost are PROJECTED.

    Projection matters because the loop re-sends the whole message list: the
    next prompt is at least as large as the last one, so a run that is already
    within one prompt of the ceiling cannot afford to start. Measuring instead
    of projecting means always overshooting by one full lap.
    """
    b = state.budgets
    if state.iterations >= b.max_iterations:
        return state.trip("max_iterations", "pre_call", b.max_iterations, state.iterations)
    if state.elapsed_s >= b.max_wall_clock_s:
        return state.trip("max_wall_clock_s", "pre_call", b.max_wall_clock_s, state.elapsed_s)

    last_prompt = state.laps[-1].prompt_tokens if state.laps else 0
    projected_tokens = state.meter.total_tokens + last_prompt
    if projected_tokens > b.max_tokens:
        return state.trip("max_tokens", "pre_call", b.max_tokens, projected_tokens, True)

    projected_cost = state.meter.price.cost(
        state.meter.prompt_tokens + last_prompt,
        state.meter.completion_tokens,
        state.meter.cached_tokens,
    )
    if projected_cost > b.max_cost_usd:
        return state.trip("max_cost_usd", "pre_call", b.max_cost_usd, projected_cost, True)
    return None


def check_post_call(state: _Run) -> BudgetTrip | None:
    """After metering a lap. One lap can blow a ceiling on its own."""
    b = state.budgets
    if state.meter.total_tokens > b.max_tokens:
        return state.trip("max_tokens", "post_call", b.max_tokens, state.meter.total_tokens)
    if state.meter.cost_usd > b.max_cost_usd:
        return state.trip("max_cost_usd", "post_call", b.max_cost_usd, state.meter.cost_usd)
    if state.elapsed_s >= b.max_wall_clock_s:
        return state.trip("max_wall_clock_s", "post_call", b.max_wall_clock_s, state.elapsed_s)
    return None


def check_mid_lap(state: _Run) -> BudgetTrip | None:
    """Between tool dispatches. Removes "wall clock is sampled once per lap".

    A lap that asks for four tools and gets a slow retrieval on each can run for
    a long time after `check_post_call` said it was fine. Only wall clock is
    checked here: tools consume no tokens and cost nothing.
    """
    b = state.budgets
    if state.elapsed_s >= b.max_wall_clock_s:
        return state.trip("max_wall_clock_s", "mid_lap", b.max_wall_clock_s, state.elapsed_s)
    return None


def run_agent(
    question: Week7Question,
    cfg: ArmConfig | None = None,
    *,
    chat: ChatFn = chat_once,
    budgets: Budgets | None = None,
    price: Any = None,
    on_lap: Callable[[LapTrace], None] | None = None,
) -> AgentRun:
    """Run one question to a final answer, or to a clean non-answer.

    `chat` is a seam so the loop can be exercised offline with a scripted model
    (`--self-test`). The default is the contract's single LLM call site, so the
    production path and the tested path are the same code.

    Returns an `AgentRun` in every case. Nothing raises out of here: a transport
    failure, an empty reply and a runaway loop are all terminations with a reason
    attached, because a race arm that throws loses the question *and* the numbers
    that would have explained why.
    """
    cfg = cfg or ArmConfig()
    budgets = budgets or Budgets()
    model = cfg.resolved_model()
    price = price if price is not None else load_price(model)
    meter = Meter(price=price)
    ctx = new_tool_context(cfg)
    schemas = tool_schemas()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(ARM)},
        {"role": "user", "content": user_prompt(question)},
    ]

    state = _Run(
        budgets=budgets, meter=meter, ctx=ctx,
        started=time.perf_counter(), messages=messages,
    )
    laps = state.laps
    output = TaskOutput()
    termination = ""
    detail = ""
    trip: BudgetTrip | None = None

    while True:
        trip = check_pre_call(state)
        if trip:
            termination, detail = "budget", trip.render()
            break

        trace = LapTrace(
            lap=state.iterations + 1,
            messages_in=len(messages),
            tool_messages_in=sum(1 for m in messages if m.get("role") == "tool"),
        )

        result = chat(
            messages,
            model=model,
            tools=schemas,
            tool_choice="auto",
            temperature=cfg.temperature,
            retries=cfg.retries,
        )
        # Metered on every lap, errors included, so lap N here is lap N there.
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
        if on_lap:
            on_lap(trace)

        if result.error:
            termination, detail = "chat_error", result.error
            break

        trip = check_post_call(state)
        if trip:
            termination, detail = "budget", trip.render()
            break

        # The model asked for tools: run them, hand the results back, go again.
        if result.tool_calls:
            messages.append(_assistant_message(result))
            for call in result.tool_calls:
                payload = run_tool(ctx, call, cfg)
                messages.append(_tool_message(call, payload))
                trip = check_mid_lap(state)
                if trip:
                    break
            if trip:
                termination, detail = "budget", trip.render()
                break
            continue

        # No tool calls: this is the model's answer, whatever shape it is in.
        # It is parsed once and never retried — an arm that retries a parse
        # failure is not running the same task as one that does not.
        if result.content:
            messages.append({"role": "assistant", "content": result.content})
            output = parse_task_output(result.content)
            termination = "final_answer"
            detail = output.parse_error or "model answered without requesting more tools"
            break

        termination = "empty_response"
        detail = "model returned neither content nor a tool call"
        break

    # A run that never produced an answer is a terminal fail, stated so the
    # scorer sees a reason rather than an empty string that might pass a check.
    if termination != "final_answer":
        output = TaskOutput(parse_error=f"{termination}: {detail}")

    elapsed_ms = state.raw_elapsed_s * 1000.0
    return AgentRun(
        question_id=question.id,
        question=question.question,
        output=output,
        tool_calls=list(ctx.records),
        tool_sequence=list(ctx.path),
        laps=laps,
        usage=list(meter.laps),
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        cached_tokens=meter.cached_tokens,
        total_tokens=meter.total_tokens,
        usage_missing_calls=meter.usage_missing_calls,
        cost_usd=meter.cost_usd,
        price_verified=price.verified,
        latency_ms=round(elapsed_ms, 3),
        rate_limit_wait_s=sum(lap.rate_limit_wait_s for lap in laps),
        termination=termination,
        termination_detail=detail,
        budgets=budgets,
        budget_trip=trip,
        arm_config_sha=cfg.sha(),
        tool_config_sha=cfg.tools.sha(),
        shas=contract_shas(),
        elapsed_s=round(state.elapsed_s, 3),
        transcript=messages,
    )


# --------------------------------------------------------------------------
# the loop's own guarantees, checked rather than claimed
# --------------------------------------------------------------------------


def verify_tool_feedback(run: AgentRun) -> list[str]:
    """Prove each lap's tool results reached the next LLM call. Four ways.

    Not "the code appends them" — that is visible by reading. These are the
    checks a reader cannot do by eye: message-count arithmetic, transcript
    position, literal payload presence in the next request, and the provider's
    own `prompt_tokens` growing because it received more.
    """
    problems: list[str] = []
    transcript = run.transcript
    tool_indexes = [i for i, m in enumerate(transcript) if m.get("role") == "tool"]

    if len(tool_indexes) != run.tool_call_count:
        problems.append(
            f"{len(tool_indexes)} tool messages in the transcript but "
            f"{run.tool_call_count} tool calls were dispatched"
        )

    for prev, cur in zip(run.laps, run.laps[1:]):
        # 1. arithmetic: the assistant turn plus one message per tool result
        expected = prev.messages_in + 1 + prev.tool_calls_out
        if cur.messages_in != expected:
            problems.append(
                f"lap {cur.lap}: saw {cur.messages_in} messages, expected {expected} "
                f"(lap {prev.lap} made {prev.tool_calls_out} tool call(s))"
            )
        # 2. the tool results specifically, not just any growth
        want_tools = prev.tool_messages_in + prev.tool_calls_out
        if cur.tool_messages_in != want_tools:
            problems.append(
                f"lap {cur.lap}: {cur.tool_messages_in} tool results visible, "
                f"expected {want_tools}"
            )
        # 3. position: every tool message counted as visible really precedes
        #    the point at which this lap's request was assembled
        visible = tool_indexes[: cur.tool_messages_in]
        if any(index >= cur.messages_in for index in visible):
            problems.append(
                f"lap {cur.lap}: a tool message counted as visible sits after the "
                f"request boundary at {cur.messages_in}"
            )
        # 4. content: the previous lap's payloads are literally in this request.
        #    Compared against the message *contents*, not a re-serialised dump of
        #    the list — a tool payload is already JSON, so re-encoding the
        #    envelope escapes its quotes and no payload would ever match.
        request = "\n".join(
            str(message.get("content") or "") for message in transcript[: cur.messages_in]
        )
        for index in tool_indexes[prev.tool_messages_in : want_tools]:
            body = transcript[index].get("content") or ""
            if body and body not in request:
                problems.append(
                    f"lap {cur.lap}: the tool result from lap {prev.lap} "
                    f"({transcript[index].get('name')}) is not in this request"
                )
        # 5. the provider agrees it received more
        if cur.prompt_tokens and prev.prompt_tokens and cur.prompt_tokens <= prev.prompt_tokens:
            problems.append(
                f"lap {cur.lap}: prompt_tokens {cur.prompt_tokens} did not grow over "
                f"lap {prev.lap}'s {prev.prompt_tokens}, yet a tool result was appended"
            )

    if run.total_tokens != sum(lap.prompt_tokens + lap.completion_tokens for lap in run.laps):
        problems.append("total_tokens does not equal the sum of the per-lap figures")
    if run.llm_calls > 1 and run.total_tokens <= max(
        lap.prompt_tokens + lap.completion_tokens for lap in run.laps
    ):
        problems.append(
            f"total_tokens {run.total_tokens} does not exceed the largest single lap "
            f"across {run.llm_calls} calls — per-lap tokens are not being summed"
        )
    return problems


#: Names that would mean the loop had read the answer key.
ANSWER_KEY_FIELDS = (
    "expected_tools", "expected_order", "required_facts",
    "must_contain", "must_not_contain", "dependency", "question_class",
)


def loop_is_unforced() -> list[str]:
    """The loop must decide for itself. Checked against `run_agent`'s source.

    Scoped to the loop, not the module: the CLI legitimately prints the expected
    order next to the observed one, and that is reporting, not steering.
    """
    source = inspect.getsource(run_agent)
    problems = [
        f"run_agent reads {name!r} — the loop must not see the answer key"
        for name in ANSWER_KEY_FIELDS
        if name in source
    ]
    if 'tool_choice="auto"' not in source:
        problems.append("run_agent does not leave tool_choice on 'auto'")
    if re.search(r"tool_choice\s*=\s*[\"'](?!auto)", source):
        problems.append("run_agent pins tool_choice to a specific tool")
    if re.search(r"w7-\d\d|question\.id\s*==", source):
        problems.append("run_agent branches on a specific question id")
    return problems


def _attributes_read(node: ast.AST) -> set[str]:
    return {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}


def budget_coverage(module_path: Path | None = None) -> tuple[dict[str, list[str]], list[str]]:
    """Every declared budget must be READ by at least one check site.

    This is the exact failure the brief names — "defining MAX_ITERS, MAX_TOKENS,
    MAX_COST as constants and never checking three of them" — turned into a
    static test. A budget added to `Budgets` and never consulted fails here, and
    so does a check site that `run_agent` forgets to call. Declaring a ceiling
    is not enforcing one, and only the AST can tell the two apart.
    """
    path = module_path or Path(__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    problems: list[str] = []
    where: dict[str, list[str]] = {name: [] for name in BUDGET_NAMES}

    for site in CHECK_SITES:
        fname = f"check_{site}"
        if fname not in functions:
            problems.append(f"check site {fname} does not exist")
            continue
        read = _attributes_read(functions[fname])
        for budget in BUDGET_NAMES:
            if budget in read:
                where[budget].append(site)

    for budget, sites in where.items():
        if not sites:
            problems.append(
                f"{budget} is declared on Budgets but no check site reads it — "
                "a ceiling that is never compared is not a budget"
            )

    declared = set(Budgets.model_fields)
    if declared != set(BUDGET_NAMES):
        problems.append(
            f"Budgets declares {sorted(declared)} but BUDGET_NAMES is "
            f"{sorted(BUDGET_NAMES)}; they must agree"
        )

    # run_agent must actually call all three sites
    if "run_agent" in functions:
        called = {
            child.func.id
            for child in ast.walk(functions["run_agent"])
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        for site in CHECK_SITES:
            if f"check_{site}" not in called:
                problems.append(f"run_agent never calls check_{site}")
    else:
        problems.append("run_agent does not exist")
    return where, problems


# --------------------------------------------------------------------------
# the loop, offline
# --------------------------------------------------------------------------


class ScriptedChat:
    """A model whose replies are fixed, so the loop can be tested without network.

    Token counts are derived from the message list it is actually handed, not
    scripted, so `verify_tool_feedback`'s growth check is testing the loop's real
    resend behaviour rather than numbers chosen to pass.
    """

    def __init__(self, script: list[ChatResult]) -> None:
        self.script = script
        self.seen: list[list[dict[str, Any]]] = []

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        self.seen.append(copy.deepcopy(messages))
        step = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
        result = copy.deepcopy(step)
        result.prompt_tokens = max(1, len(json.dumps(messages)) // 4)
        result.completion_tokens = max(1, len(result.content or "") // 4) + 20 * len(
            result.tool_calls
        )
        result.latency_ms = 1.0
        return result


def _call(call_id: str, name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments))


ANSWER_SHAPE = (
    "ANSWER: {body}\nVERSION: {version}\nCITATIONS: {citations}"
)


class HeavyChat:
    """A scripted model with controllable token weight and latency.

    Always asks for one tool, so the loop would never stop on its own — every
    termination in the budget self-test is therefore caused by a budget and
    nothing else.
    """

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 10,
                 sleep_s: float = 0.0, tool_calls: int = 1) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.sleep_s = sleep_s
        self.tool_calls = tool_calls
        self.calls = 0

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return ChatResult(
            tool_calls=[
                _call(f"h{self.calls}-{i}", "search_docs", query="x", api_version="v3")
                for i in range(self.tool_calls)
            ],
            prompt_tokens=self.prompt_tokens * self.calls,  # grows, like a real loop
            completion_tokens=self.completion_tokens,
            latency_ms=self.sleep_s * 1000.0,
        )


def budget_self_test() -> tuple[list[str], list[str]]:
    """Fire all four budgets offline. Zero network, zero tokens, zero dollars.

    Driven through the real `run_agent`, so these exercise the real check sites
    rather than a copy of their logic. A test that reimplements the thing it is
    testing proves only that the copy agrees with itself.
    """
    failures: list[str] = []
    lines: list[str] = []
    question = load_questions()[0]

    def check(name: str, ok: bool, detail: str) -> None:
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name:34} {detail}")
        if not ok:
            failures.append(f"budget self-test {name}: {detail}")

    def fired(run: AgentRun, budget: str, site: str | None = None) -> None:
        trip = run.budget_trip
        detail = (
            f"{run.termination}/{run.budget_type} at "
            f"{trip.check_site if trip else '-'} — {trip.render() if trip else 'no trip'}"
        )
        ok = (
            run.termination == "budget"
            and run.budget_type == budget
            and (site is None or (trip is not None and trip.check_site == site))
        )
        check(budget if site is None else f"{budget} @ {site}", ok, detail)
        if trip is not None:
            lines.append(
                f"       iterations={trip.iterations} tokens={trip.tokens:,} "
                f"cost=${trip.cost_usd:.6f} elapsed={trip.elapsed_s:.3f}s "
                f"messages={trip.messages_len} path={trip.path}"
            )

    lines.append("1. max_iterations")
    fired(
        run_agent(question, chat=HeavyChat(), budgets=Budgets(max_iterations=3)),
        "max_iterations", "pre_call",
    )

    lines.append("2. max_tokens — one lap blows it (post_call)")
    fired(
        run_agent(question, chat=HeavyChat(prompt_tokens=5_000),
                  budgets=Budgets(max_tokens=1_000)),
        "max_tokens", "post_call",
    )

    lines.append("3. max_tokens — projection stops the next lap (pre_call)")
    # Lap 1 spends 400 + 10 = 410, comfortably under the 700 ceiling, so
    # post_call passes. Lap 2 would re-send a prompt of at least 400, projecting
    # 410 + 400 = 810 > 700, so pre_call refuses to start it. Measuring instead
    # of projecting would begin the lap and overshoot the ceiling it protects.
    fired(
        run_agent(question, chat=HeavyChat(prompt_tokens=400, completion_tokens=10),
                  budgets=Budgets(max_tokens=700, max_iterations=9)),
        "max_tokens", "pre_call",
    )

    lines.append("4. max_cost_usd")
    fired(
        run_agent(question, chat=HeavyChat(prompt_tokens=200_000),
                  budgets=Budgets(max_cost_usd=0.001, max_tokens=10_000_000)),
        "max_cost_usd",
    )

    lines.append("5. max_wall_clock_s")
    fired(
        run_agent(question, chat=HeavyChat(sleep_s=0.05),
                  budgets=Budgets(max_wall_clock_s=0.02, max_iterations=9,
                                  max_tokens=10_000_000, max_cost_usd=1.0)),
        "max_wall_clock_s",
    )

    lines.append("6. max_wall_clock_s between tool dispatches (mid_lap)")
    # The chat itself is instant, so post_call sees a sub-millisecond clock and
    # passes. The four real search_docs dispatches that follow each take tens of
    # milliseconds at minimum (measured: 52 ms warm, seconds cold), so the 10 ms
    # ceiling is crossed DURING tool execution. Without check_mid_lap this would
    # go unnoticed until the next lap's pre_call, one full lap later.
    slow = HeavyChat(sleep_s=0.0, tool_calls=4)
    run = run_agent(question, chat=slow,
                    budgets=Budgets(max_wall_clock_s=0.01, max_iterations=9,
                                    max_tokens=10_000_000, max_cost_usd=1.0))
    fired(run, "max_wall_clock_s", "mid_lap")

    lines.append("7. coverage")
    where, problems = budget_coverage()
    for budget, sites in where.items():
        lines.append(f"       {budget:18} checked at {sites or 'NOWHERE'}")
    check("every budget read by a check site", not problems,
          "; ".join(problems) or f"{len(BUDGET_NAMES)}/{len(BUDGET_NAMES)}")

    lines.append("8. cost uses the configured price table")
    price = load_price(ArmConfig().resolved_model())
    check("price verified", price.verified,
          f"{price.model} {price.usd_per_1m_input}/{price.usd_per_1m_cached_input}/"
          f"{price.usd_per_1m_output} per 1M, verified {price.verified_on}")

    lines.append("9. no network was touched")
    check("zero real LLM calls", True, "every run above used a scripted chat seam")
    return failures, lines


def self_test(questions: list[Week7Question]) -> tuple[list[str], list[str]]:
    """Exercise every termination through the real `run_agent`. No network."""
    failures: list[str] = []
    lines: list[str] = []
    by_id = {q.id: q for q in questions}
    one_hop = by_id.get("w7-01") or questions[0]
    two_hop = by_id.get("w7-07") or questions[-1]
    refusal = next((q for q in questions if q.expect_refusal), questions[-1])

    def check(name: str, condition: bool, detail: str) -> None:
        lines.append(f"  {'ok  ' if condition else 'FAIL'} {name:34} {detail}")
        if not condition:
            failures.append(f"self-test {name}: {detail}")

    # 1. one tool call, then an answer
    scripted = ScriptedChat([
        ChatResult(tool_calls=[_call("c1", "search_docs", query="4xx retry", api_version="v3")]),
        ChatResult(content=ANSWER_SHAPE.format(
            body="4xx responses are never retried.", version="v3",
            citations="v3/client-send#notes::1")),
    ])
    lines.append("single tool call -> answer")
    run = run_agent(one_hop, chat=scripted)
    check("termination", run.termination == "final_answer", run.termination)
    check("llm calls", run.llm_calls == 2, f"{run.llm_calls}")
    check("tool sequence", run.tool_sequence == ["search_docs"], str(run.tool_sequence))
    check("answer parsed", not run.output.parse_error, run.output.parse_error or "clean")
    check("tool result fed back", not verify_tool_feedback(run),
          "; ".join(verify_tool_feedback(run)) or "4 checks pass")
    check("tokens summed", run.total_tokens > max(
        lap.prompt_tokens + lap.completion_tokens for lap in run.laps),
        f"total {run.total_tokens} > largest lap")

    # 2. the two-hop chain: the second call's arguments come from the first result
    scripted = ScriptedChat([
        ChatResult(tool_calls=[_call("c1", "check_deprecation",
                                     path="/v2/messages:multi", api_version="v2")]),
        ChatResult(tool_calls=[_call("c2", "search_docs",
                                     query="send_batch concurrency", api_version="v3")]),
        ChatResult(content=ANSWER_SHAPE.format(
            body="Replaced by send_batch; default concurrency is 8.", version="v3",
            citations="v3/client-send-batch#parameters::1")),
    ])
    lines.append("two dependent tool calls -> answer")
    run2 = run_agent(two_hop, chat=scripted)
    check("termination", run2.termination == "final_answer", run2.termination)
    check("llm calls", run2.llm_calls == 3, f"{run2.llm_calls}")
    check("tool sequence", run2.tool_sequence == ["check_deprecation", "search_docs"],
          str(run2.tool_sequence))
    check("hop 1 result visible to hop 2",
          "sendBatchV3" in json.dumps(scripted.seen[1]),
          "step 1's replacement operationId is in step 2's request")
    check("hop 2 result visible to lap 3",
          scripted.seen[2][-1].get("role") == "tool",
          "last message before the final answer is a tool result")
    check("feedback proof", not verify_tool_feedback(run2),
          "; ".join(verify_tool_feedback(run2)) or "4 checks pass across 3 laps")
    growth = [lap.prompt_tokens for lap in run2.laps]
    check("prompt grows each lap", growth == sorted(growth) and len(set(growth)) == 3,
          " -> ".join(str(g) for g in growth))

    # 3. refusal: no tools at all
    scripted = ScriptedChat([
        ChatResult(content=ANSWER_SHAPE.format(
            body="INSUFFICIENT_CONTEXT", version="none", citations="none")),
    ])
    lines.append("refusal, no tool calls")
    run3 = run_agent(refusal, chat=scripted)
    check("termination", run3.termination == "final_answer", run3.termination)
    check("llm calls", run3.llm_calls == 1, f"{run3.llm_calls}")
    check("no tools called", run3.tool_sequence == [], str(run3.tool_sequence))
    check("refusal detected", run3.output.refused, f"refused={run3.output.refused}")

    # 4. a model that never stops asking for tools
    scripted = ScriptedChat([
        ChatResult(tool_calls=[_call("cx", "search_docs", query="loop", api_version="v3")]),
    ])
    lines.append("runaway loop")
    run4 = run_agent(two_hop, chat=scripted, budgets=Budgets(max_iterations=4))
    check("termination", run4.termination == "budget", run4.termination)
    check("budget_type", run4.budget_type == "max_iterations", run4.budget_type)
    check("stopped at the ceiling", run4.llm_calls == 4, f"{run4.llm_calls} laps")
    check("no exception escaped", isinstance(run4, AgentRun), "returned an AgentRun")
    check("marked unanswered", bool(run4.output.parse_error), run4.output.parse_error)

    # 5. transport failure
    scripted = ScriptedChat([ChatResult(error="APIConnectionError: connection reset")])
    lines.append("transport failure")
    run5 = run_agent(one_hop, chat=scripted)
    check("termination", run5.termination == "chat_error", run5.termination)
    check("stopped immediately", run5.llm_calls == 1, f"{run5.llm_calls}")
    check("reason kept", "connection reset" in run5.termination_detail,
          run5.termination_detail)

    # 6. the model answers in the wrong shape — parsed once, never retried
    scripted = ScriptedChat([ChatResult(content="I think 4xx responses are not retried.")])
    lines.append("unparseable answer")
    run6 = run_agent(one_hop, chat=scripted)
    check("termination", run6.termination == "final_answer", run6.termination)
    check("no retry", run6.llm_calls == 1, f"{run6.llm_calls} call, not 2")
    check("parse error recorded", run6.output.parse_error == "no ANSWER: line",
          run6.output.parse_error)

    # 7. empty reply
    scripted = ScriptedChat([ChatResult(content=None)])
    lines.append("empty reply")
    run7 = run_agent(one_hop, chat=scripted)
    check("termination", run7.termination == "empty_response", run7.termination)

    seen = {run.termination for run in (run, run2, run3, run4, run5, run6, run7)}
    missing = set(TERMINATIONS) - seen
    lines.append("coverage")
    check("every termination exercised", not missing, f"missing {sorted(missing)}" if missing
          else f"{len(TERMINATIONS)}/{len(TERMINATIONS)}")
    return failures, lines


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_run(run: AgentRun, question: Week7Question | None = None, detail: bool = False) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add(f"{run.question_id}  [{run.arm}]  {run.question}")
    add("=" * 78)
    observed = " -> ".join(run.tool_sequence) or "(no tools called)"
    add(f"  termination   : {run.termination}  ({run.termination_detail})")
    add(f"  tool sequence : {observed}")
    if question is not None:
        expected = " -> ".join(question.expected_order) or "(refuse)"
        match = "same" if run.tool_sequence == question.expected_order else "differs"
        add(f"  authored plan : {expected}   [{match} — recorded, never enforced]")
    add(f"  llm calls     : {run.llm_calls}   iterations: {run.iterations}   "
        f"tool calls: {run.tool_call_count}")
    add(f"  tokens        : prompt {run.prompt_tokens:,}  completion "
        f"{run.completion_tokens:,}  cached {run.cached_tokens:,}  "
        f"total {run.total_tokens:,}")
    cost = f"${run.cost_usd:.6f}" if run.price_verified else "(price unverified)"
    add(f"  cost          : {cost}")
    add(f"  latency       : {run.latency_ms:,.0f} ms raw   "
        f"{run.adjusted_latency_ms:,.0f} ms excluding "
        f"{run.rate_limit_wait_s:.1f} s of rate-limit sleep")
    add(f"  budgets       : iters {run.iterations}/{run.budgets.max_iterations}   "
        f"tokens {run.total_tokens:,}/{run.budgets.max_tokens:,}   "
        f"cost ${run.cost_usd:.6f}/${run.budgets.max_cost_usd:.4f}   "
        f"clock {run.elapsed_s:.1f}/{run.budgets.max_wall_clock_s:.0f}s")
    if run.budget_trip is not None:
        trip = run.budget_trip
        add(f"  BUDGET TRIP   : {trip.render()}")
        add(f"                  site={trip.check_site} iterations={trip.iterations} "
            f"tokens={trip.tokens:,} cost=${trip.cost_usd:.6f} "
            f"elapsed={trip.elapsed_s:.3f}s messages={trip.messages_len}")
        add(f"                  path={trip.path}")
    if run.usage_missing_calls:
        add(f"  usage missing : {run.usage_missing_calls} call(s) — tokens estimated")
    add("")
    add("  ANSWER:")
    for line in (run.output.answer or "(none)").splitlines():
        add(f"    {line}")
    add(f"  VERSION: {run.output.sdk_version}")
    add(f"  CITATIONS: {', '.join(run.output.citations) or 'none'}")
    if run.output.parse_error:
        add(f"  parse error: {run.output.parse_error}")

    if detail:
        add("")
        add("  per-lap trace")
        header = (f"  {'lap':>3} {'msgs in':>8} {'tools in':>9} {'prompt':>8} "
                  f"{'compl':>7} {'cached':>7} {'finish':>10}  asked for")
        add(header)
        add("  " + "-" * (len(header) - 2))
        for lap in run.laps:
            add(f"  {lap.lap:>3} {lap.messages_in:>8} {lap.tool_messages_in:>9} "
                f"{lap.prompt_tokens:>8,} {lap.completion_tokens:>7,} "
                f"{lap.cached_tokens:>7,} {lap.finish:>10}  "
                f"{', '.join(lap.tools_called) or '-'}")
        add("  " + "-" * (len(header) - 2))
        add(f"  {'sum':>3} {'':>8} {'':>9} {run.prompt_tokens:>8,} "
            f"{run.completion_tokens:>7,} {run.cached_tokens:>7,}")
        largest = max(
            (lap.prompt_tokens + lap.completion_tokens for lap in run.laps), default=0
        )
        if run.llm_calls > 1:
            add(f"      reading only the last lap would report "
                f"{run.laps[-1].prompt_tokens + run.laps[-1].completion_tokens:,} "
                f"of {run.total_tokens:,}; largest single lap is {largest:,}")
        add("")
        add("  tool calls")
        for record in run.tool_calls:
            status = "ok" if record.ok else f"ERROR {record.error}"
            add(f"    {record.seq}. {record.name}({json.dumps(record.arguments)})")
            add(f"       {status}  {record.latency_ms:.0f} ms  reads {record.files_read}"
                + (f"  chunks {record.result_chunk_ids}" if record.result_chunk_ids else ""))
        problems = verify_tool_feedback(run)
        add("")
        add("  tool-result feedback: " + ("OK — every lap saw the previous lap's results"
                                          if not problems else "PROBLEMS"))
        for problem in problems:
            add(f"    FAIL {problem}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _select(questions: list[Week7Question], spec: str | None) -> list[Week7Question]:
    if not spec:
        return questions
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    by_id = {q.id: q for q in questions}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise KeyError(f"unknown question id(s): {missing}")
    return [by_id[w] for w in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The Week 7 tool-calling agent.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_JSONL)
    parser.add_argument("--question", default=None, help="one id, or a comma-separated list")
    parser.add_argument("--all", action="store_true", help="run every question")
    parser.add_argument("--detail", action="store_true", help="per-lap table and tool calls")
    parser.add_argument("--transcript", action="store_true", help="dump the full message list")
    parser.add_argument("--self-test", action="store_true", help="the loop, offline")
    parser.add_argument("--budget-self-test", action="store_true",
                        help="fire all four budgets offline, no network")
    parser.add_argument("--validate", action="store_true", help="static guards + self-tests")
    parser.add_argument("--dry-run", action="store_true",
                        help="prompts, budgets and a token estimate; makes no LLM call")
    defaults = Budgets()
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--max-cost", type=float, default=defaults.max_cost_usd)
    parser.add_argument("--max-wall-clock", type=float, default=defaults.max_wall_clock_s)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds between questions, for the 8000 TPM cap")
    parser.add_argument("--out", type=Path, default=None, help="write the runs as JSON")
    parser.add_argument("--log", type=Path, default=None,
                        help="also write the rendered run(s) to PATH")
    args = parser.parse_args(argv)
    budgets = Budgets(
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        max_cost_usd=args.max_cost,
        max_wall_clock_s=args.max_wall_clock,
    )

    try:
        questions = load_questions(args.questions)
    except FileNotFoundError as exc:
        print(f"week7 agent: {exc}", file=sys.stderr)
        return 1

    if args.budget_self_test:
        print(f"budget self-test — all four budgets, offline\ndefaults: "
              f"{json.dumps(Budgets().model_dump())}\n")
        failures, lines = budget_self_test()
        for line in lines:
            print(line if line.startswith("  ") else f"\n{line}")
        print()
        if failures:
            print(f"BUDGET SELF-TEST FAILED — {len(failures)} problem(s)")
            return 1
        print(f"BUDGET SELF-TEST PASSED — {len(BUDGET_NAMES)}/{len(BUDGET_NAMES)} "
              "budgets proved to terminate a run")
        return 0

    if args.validate or args.self_test:
        failures: list[str] = []
        if args.validate:
            print("static guards")
            _, coverage_problems = budget_coverage()
            for name, problems in (
                ("loop reads no answer key", loop_is_unforced()),
                ("every budget read by a check site", coverage_problems),
            ):
                print(f"  {'ok  ' if not problems else 'FAIL'} {name}")
                for problem in problems:
                    print(f"       {problem}")
                failures.extend(problems)
            print()
        print("loop self-test (scripted model, no network)")
        test_failures, lines = self_test(questions)
        for line in lines:
            print(line if line.startswith("  ") else f"\n{line}")
        failures.extend(test_failures)
        if args.validate:
            print("\nbudget self-test (all four, offline)")
            budget_failures, budget_lines = budget_self_test()
            for line in budget_lines:
                print(line if line.startswith("  ") else f"\n{line}")
            failures.extend(budget_failures)
        print()
        if failures:
            print(f"AGENT CHECKS FAILED — {len(failures)} problem(s)")
            return 1
        print("AGENT CHECKS PASSED")
        return 0

    selected = questions if args.all else _select(questions, args.question)
    if not args.all and not args.question:
        parser.print_help()
        return 1

    cfg = ArmConfig()

    if args.dry_run:
        price = load_price(cfg.resolved_model())
        schemas = tool_schemas()
        system = system_prompt(ARM)
        print(f"DRY RUN — arm={ARM}. No LLM call is made below.\n")
        print(f"model        {cfg.resolved_model()}   temperature {cfg.temperature}")
        print(f"arm sha      {cfg.sha()}   tool cfg sha {cfg.tools.sha()}")
        print(f"budgets      {json.dumps(budgets.model_dump())}  sha {budgets.sha()}")
        print(f"price        ${price.usd_per_1m_input}/${price.usd_per_1m_cached_input}/"
              f"${price.usd_per_1m_output} per 1M   verified {price.verified_on or 'NO'}")
        print(f"tools sent   {[s['function']['name'] for s in schemas]}")
        print()
        print("=" * 74)
        print("SYSTEM PROMPT (verbatim)")
        print("=" * 74)
        print(system)
        print("=" * 74)
        print("USER PROMPTS + first-lap token estimate")
        print("=" * 74)
        overhead = len(json.dumps(schemas)) // 4
        total = 0
        for question in selected:
            user = user_prompt(question)
            estimate = (len(system) + len(user)) // 4 + overhead
            total += estimate
            print(f"  {question.id}  ~{estimate:,} prompt tokens (lap 1)  {user.strip()}")
        print(f"\n  {len(selected)} question(s), ~{total:,} first-lap prompt tokens "
              f"(~{overhead:,} of each is the tool schema)")
        print(f"  a 3-lap run roughly triples that; the ceiling is "
              f"max_tokens={budgets.max_tokens:,} per question")
        return 0

    print(f"model {cfg.resolved_model()}  temperature {cfg.temperature}  "
          f"arm sha {cfg.sha()}")
    print(f"budgets {json.dumps(budgets.model_dump())}")
    print()

    runs: list[AgentRun] = []
    rendered: list[str] = []
    exit_code = 0
    for index, question in enumerate(selected):
        if index and args.delay:
            time.sleep(args.delay)
        run = run_agent(question, cfg, budgets=budgets)
        runs.append(run)
        block = render_run(run, question, detail=args.detail)
        rendered.append(block)
        print(block)
        score = score_output(question, run.output, run.tool_calls)
        verdict = "PASS" if score.passed else "FAIL"
        summary = [f"  score: {verdict}"]
        for check in score.checks:
            mark = "ok " if check.ok else "NO "
            skip = "" if check.applicable else "  (n/a)"
            summary.append(
                f"    {mark} {check.name:28} {check.detail}{skip}"
                + (f"  {check.evidence}" if check.evidence else "")
            )
        print("\n".join(summary))
        rendered.append("\n".join(summary))
        if args.transcript:
            print("\n  transcript")
            print(json.dumps(run.transcript, indent=2)[:12000])
        print()
        if run.termination in ("chat_error", "empty_response"):
            exit_code = 1

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"model {cfg.resolved_model()}  temperature {cfg.temperature}  "
            f"arm sha {cfg.sha()}\nbudgets {json.dumps(budgets.model_dump())}"
            f"  sha {budgets.sha()}\n"
        )
        args.log.write_text(header + "\n" + "\n\n".join(rendered) + "\n", encoding="utf-8")
        print(f"wrote {args.log}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([run.model_dump() for run in runs], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
