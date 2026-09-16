"""The Week 7 fixed workflow — the second arm of the race.

    python -m app.rag.week7_workflow --plan                # constants, call sites, source
    python -m app.rag.week7_workflow --validate            # static + offline proofs
    python -m app.rag.week7_workflow --question w7-07 --detail
    python -m app.rag.week7_workflow --holdout             # unseen questions, golden_set

Same task as the agent: same questions, same tools, same model, same output
contract, same parser, same scorer, all imported from `week7_contract`. The only
thing that differs is the control flow — and that is the whole experiment.

## The plan, fixed at import time

    s1_search              search_docs(v2) and search_docs(v3)     2 tool calls
    s2_lookup_primary      check_deprecation(target from question)  1 tool call
    s3_lookup_counterpart  get_openapi_spec(ARGS FROM s2's RESULT)  1 tool call
    s4_read_best_chunk     pure selection over s1's passages        0 tool calls
    s5_compose             one chat call, no tools attached         1 LLM call

`run_workflow` is five statements. Not a list of steps walked by a `for` — a
literal sequence, so "each step runs at most once" is visible in the source
rather than argued from a guard.

## Where the line between "workflow" and "loop" is

> A `for`/`while` is a **loop** when it can cause another `chat()` or tool
> dispatch **whose existence depends on the result of a previous one**. The test
> is static: if the maximum number of those calls is a compile-time constant,
> there is no loop.

Here the constant is legible by counting textual call sites: `run_tool` appears
four times in this module's step functions, `chat` once, and none of the five is
inside a `for`, a `while` or a comprehension. `static_call_bounds()` parses this
file's AST and asserts exactly that, so the bound is checked rather than
promised. `assert_no_loop()` then re-checks it against every run: `llm_calls == 1`
**exactly** (not `<=`), tool calls `<= 4`, and the executed step names a
subsequence of `WORKFLOW_PLAN` with no repeats — a repeated step name is a loop
by definition.

s1 issuing two searches is a fixed fan-out, not a loop: the second call's
existence does not depend on the first's result, and both are written out rather
than iterated. That fan-out is the point — it is what lets a workflow answer the
cross-version questions without any planning.

## What the workflow is allowed to know

Only `question.question` — the text — and `question.id` for labelling. It never
reads `sdk_version`, `expected_tools`, `expected_order`, `required_facts`,
`must_contain` or `question_class`. A workflow told the version while the agent
has to infer it is not running the same task, and the advantage would point
exactly where the expected conclusion lies. `reads_no_answer_key()` parses the
step functions and fails if any of those names appear in them, mirroring
`week7_agent.loop_is_unforced()`.

`candidate_parameter()` is the biggest cheat risk in the file: a literal list of
the ten questions' parameters would fit the workflow to the test set and make
every number fiction. It is derived instead — the question text intersected with
the parameter names parsed out of the `docs/` tables — and `--holdout` runs the
same derivation over `eval/golden_set.jsonl`, committed in Week 4, to show it
generalises to questions written before this arm existed.

## Not here

Budgets and the race harness. This module returns a `WorkflowRun` whose public
metric names match `AgentRun`'s, so the harness can read both without a special
case; enforcing ceilings on top of them is the next step.
"""

import argparse
import ast
import inspect
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.rag.week6_assertions import loose, normalise
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
from app.rag.week7_corpus import load_pages
from app.rag.week7_questions import (
    DEFAULT_QUESTIONS_JSONL,
    Week7Question,
    load_questions,
)
from app.rag.week7_tools import API_VERSIONS, SDK_METHODS, ToolCallRecord

ARM = "workflow"

#: The plan. Fixed at import time; `run_workflow` executes it in this order once.
WORKFLOW_PLAN: tuple[str, ...] = (
    "s1_search",
    "s2_lookup_primary",
    "s3_lookup_counterpart",
    "s4_read_best_chunk",
    "s5_compose",
)

#: Compile-time bounds. Asserted statically against the AST and at runtime
#: against every run. `MAX_LLM_CALLS` is an equality, not a ceiling.
MAX_LLM_CALLS = 1
MAX_TOOL_CALLS = 4

#: When the question names no version, or names both, the workflow reads the
#: current one. A fixed pipeline has to pick; this is a stated policy, not an
#: answer — it is applied identically to every question, including the holdout.
DEFAULT_VERSION = "v3"

TERMINATIONS = ("final_answer", "chat_error", "empty_response")

__all__ = [
    "WORKFLOW_PLAN", "MAX_LLM_CALLS", "MAX_TOOL_CALLS",
    "WorkflowRun", "StepRecord", "run_workflow", "assert_no_loop",
    "verify_step3_dependency", "static_call_bounds", "reads_no_answer_key",
    "candidate_parameter", "primary_target", "compare_arms",
]


# --------------------------------------------------------------------------
# derivations — from the question TEXT and the corpus, never from metadata
# --------------------------------------------------------------------------

_PATH = re.compile(r"/v(\d+)/[A-Za-z0-9:._-]+")
_METHOD = re.compile(r"Client\.[a-z_]+")
_VERSION = re.compile(r"\bv([23])\b")


@lru_cache(maxsize=4)
def parameter_names(docs_root_str: str) -> tuple[str, ...]:
    """Every parameter name in every `## Parameters` table under docs/.

    Read through `week7_corpus.load_pages`, the same header-driven parser the
    OpenAPI build uses, so the workflow and the corpus cannot disagree about
    what a parameter is called.
    """
    pages = load_pages(Path(docs_root_str))
    names = {row.name for page in pages.values() for row in page.params}
    return tuple(sorted(names))


def candidate_parameter(question_text: str, docs_root: Path) -> str | None:
    """The one documented parameter this question is about, or None.

    Derived: the parsed table keys intersected with the question, matched on
    `loose` forms so `retry_backoff_ms` and "retry backoff ms" are one symbol,
    and on word boundaries so a short name cannot match inside a longer word.

    Zero matches or more than one -> None, and the workflow falls back to the
    question text as its query. Guessing between two candidates is how a fixed
    pipeline silently answers about the wrong parameter.
    """
    haystack = loose(normalise(question_text))
    hits = [
        name
        for name in parameter_names(str(docs_root))
        if re.search(rf"\b{re.escape(loose(normalise(name)))}\b", haystack)
    ]
    return hits[0] if len(hits) == 1 else None


def mentioned_versions(question_text: str) -> list[str]:
    found = {f"v{digit}" for digit in _VERSION.findall(question_text)}
    found |= {f"v{digit}" for digit in _PATH.findall(question_text)}
    return sorted(v for v in found if v in API_VERSIONS)


def target_version(question_text: str) -> str:
    named = mentioned_versions(question_text)
    return named[0] if len(named) == 1 else DEFAULT_VERSION


@lru_cache(maxsize=4)
def _page_methods(docs_root_str: str) -> dict[tuple[str, str], str]:
    return {key: page.sdk_method for key, page in load_pages(Path(docs_root_str)).items()}


def method_from_chunk(chunk_id: str, docs_root: Path) -> str | None:
    """`v3/client-send#parameters::1` -> `Client.send`, via the page front matter."""
    head = chunk_id.split("#", 1)[0]
    version, _, page_id = head.partition("/")
    return _page_methods(str(docs_root)).get((version, page_id))


def primary_target(question_text: str) -> dict[str, str] | None:
    """What s2 should look up: an HTTP path if the question names one, else an
    SDK method. None when the question names neither, and s2/s3 do not run."""
    path_match = _PATH.search(question_text)
    if path_match:
        version = f"v{path_match.group(1)}"
        return {
            # Trailing sentence punctuation is not part of the path. Without the
            # strip, "posts to POST /v2/messages." looks up "/v2/messages." and
            # finds nothing, and the whole two-hop chain dies on a full stop.
            "path": path_match.group(0).rstrip(".,;:?!)'\""),
            # A path outside the documented versions (the refusal control names
            # /v1/) still gets looked up, under the default version, so the
            # composer is told "no such operation" rather than nothing at all.
            "api_version": version if version in API_VERSIONS else DEFAULT_VERSION,
        }
    method_match = _METHOD.search(question_text)
    if method_match and method_match.group(0) in SDK_METHODS:
        return {
            "sdk_method": method_match.group(0),
            "api_version": target_version(question_text),
        }
    return None


# --------------------------------------------------------------------------
# what a run exposes — field names match AgentRun
# --------------------------------------------------------------------------


class StepRecord(BaseModel):
    step: str
    executed: bool = False
    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: Which earlier step's RESULT supplied the arguments. Empty when the
    #: arguments came from the question. This is the data-dependency evidence.
    depends_on: str = ""
    derived: str = ""
    ok: bool = True
    note: str = ""
    latency_ms: float = 0.0
    result: dict[str, Any] = Field(default_factory=dict)


class WorkflowRun(BaseModel):
    arm: str = ARM
    question_id: str
    question: str

    output: TaskOutput = Field(default_factory=TaskOutput)

    steps: list[StepRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)

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

    arm_config_sha: str = ""
    tool_config_sha: str = ""
    shas: dict[str, str] = Field(default_factory=dict)

    derived_parameter: str = ""
    derived_target: dict[str, str] = Field(default_factory=dict)
    best_chunk_id: str = ""
    context_chars: int = 0
    transcript: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.output.answer

    @property
    def llm_calls(self) -> int:
        return len(self.usage)

    @property
    def iterations(self) -> int:
        """One pass. Reported for symmetry with the agent, where it can exceed 1."""
        return 1

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def executed_steps(self) -> list[str]:
        return [step.step for step in self.steps if step.executed]

    @property
    def adjusted_latency_ms(self) -> float:
        return self.latency_ms - self.rate_limit_wait_s * 1000.0

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(self.tool_sequence)


class _State:
    """Scratch space passed down the five steps. Holds no control flow."""

    def __init__(self, question: Week7Question, cfg: ArmConfig) -> None:
        self.question_text = question.question
        self.question_id = question.id
        self.cfg = cfg
        self.ctx = new_tool_context(cfg)
        self.steps: list[StepRecord] = []
        self.parameter: str | None = None
        self.target: dict[str, str] | None = None
        self.passages: list[dict[str, Any]] = []
        self.best_chunk: dict[str, Any] | None = None
        self.truncated_versions: list[str] = []
        self.context: str = ""
        self.chat_result: ChatResult | None = None
        self.messages: list[dict[str, Any]] = []

    def record(self, step: str, **fields: Any) -> StepRecord:
        entry = StepRecord(step=step, **fields)
        self.steps.append(entry)
        return entry

    def result_of(self, step: str) -> dict[str, Any] | None:
        for entry in self.steps:
            if entry.step == step and entry.executed:
                return entry.result
        return None


_SEQ = "abcdefgh"


def _tool_call(seq: int, name: str, **arguments: Any) -> ToolCall:
    return ToolCall(
        id=f"wf-{_SEQ[seq]}", name=name, arguments_json=json.dumps(arguments)
    )


# --------------------------------------------------------------------------
# the five steps
# --------------------------------------------------------------------------


def s1_search(state: _State) -> None:
    """Fixed fan-out: the same query against both documented versions.

    Two calls, both written out. Neither depends on the other's result, so the
    cross-version questions are answered without anything having planned for
    them — which is exactly the capability a fixed pipeline is supposed to have.
    """
    state.parameter = candidate_parameter(state.question_text, state.cfg.tools.docs_root)
    query = state.parameter or state.question_text
    derived = (
        f"query={query!r} (the one table key this question names)"
        if state.parameter
        else "query=the question text (no single table key matched)"
    )

    started = time.perf_counter()
    v2 = run_tool(state.ctx, _tool_call(0, "search_docs", query=query, api_version="v2"), state.cfg)
    v3 = run_tool(state.ctx, _tool_call(1, "search_docs", query=query, api_version="v3"), state.cfg)
    elapsed = (time.perf_counter() - started) * 1000.0

    state.passages = [
        {**passage, "api_version": version}
        for version, payload in (("v2", v2), ("v3", v3))
        for passage in payload.get("passages", [])
    ]
    # A payload over ArmConfig.max_tool_result_chars comes back from the shared
    # run_tool as a truncated string with no parseable `passages`. The composer
    # still receives it — and the agent receives exactly the same truncation from
    # the same cap — but s4 has no structured passage to select from, so say so
    # rather than reporting a passage count that looks like a full result.
    truncated = [
        version
        for version, payload in (("v2", v2), ("v3", v3))
        if payload.get("truncated")
    ]
    state.truncated_versions = truncated
    note = f"{len(state.passages)} passages"
    if truncated:
        note += f"; {','.join(truncated)} truncated at {state.cfg.max_tool_result_chars} chars"

    state.record(
        "s1_search",
        executed=True,
        tool="search_docs x2",
        arguments={"query": query, "api_version": list(API_VERSIONS)},
        derived=derived,
        ok=bool(state.passages),
        note=note,
        latency_ms=round(elapsed, 3),
        result={"v2": v2, "v3": v3},
    )


def s2_lookup_primary(state: _State) -> None:
    """Look up the operation the question names. Arguments from the TEXT."""
    state.target = primary_target(state.question_text)
    depends = ""
    derived = "target read out of the question text"

    # Fallback: the question describes the operation instead of naming it ("the
    # v2 streaming endpoint"). Take the SDK method of the best passage s1
    # already retrieved. Derived from a tool result, not from the question's
    # metadata, and applied to every question that needs it.
    if state.target is None and state.passages:
        version = target_version(state.question_text)
        ranked = sorted(
            state.passages,
            key=lambda p: (p.get("api_version") != version, int(p.get("rank", 99))),
        )
        for passage in ranked:
            method = method_from_chunk(
                str(passage.get("chunk_id", "")), state.cfg.tools.docs_root
            )
            if method in SDK_METHODS:
                state.target = {"sdk_method": method, "api_version": version}
                depends = "s1_search"
                derived = (
                    f"question names no path or method; took {method} from s1's best "
                    f"{version} passage ({passage.get('chunk_id')})"
                )
                break

    if state.target is None:
        state.record(
            "s2_lookup_primary",
            executed=False,
            note="the question names no path or method and s1 retrieved nothing",
        )
        return

    started = time.perf_counter()
    payload = run_tool(
        state.ctx, _tool_call(2, "check_deprecation", **state.target), state.cfg
    )
    state.record(
        "s2_lookup_primary",
        executed=True,
        tool="check_deprecation",
        arguments=dict(state.target),
        depends_on=depends,
        derived=derived,
        ok="error" not in payload,
        note=str(payload.get("reason") or f"retired={payload.get('retired')}"),
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        result=payload,
    )


def s3_lookup_counterpart(state: _State) -> None:
    """The dependent step: its ARGUMENTS come out of s2's RESULT.

    If s2 reported the operation retired, the counterpart is the replacement it
    named, and that operation's SDK method and version are knowable only from
    s2's payload — the question does not contain them. If s2 reported the
    operation current, the counterpart is its own spec view.

    Runs at most once and never re-enters s2, so the dependency is a single
    hand-off, not iteration.
    """
    previous = state.result_of("s2_lookup_primary")
    if previous is None:
        state.record("s3_lookup_counterpart", executed=False, note="s2 made no lookup")
        return

    replacement = previous.get("replacement")
    if isinstance(replacement, dict) and replacement.get("sdk_method"):
        arguments = {
            "sdk_method": replacement["sdk_method"],
            "api_version": replacement["api_version"],
        }
        derived = (
            f"s2 reported {previous.get('operation_id')} retired in "
            f"{previous.get('retired_in')}, replaced by {replacement.get('operation_id')}"
        )
    elif previous.get("found") and previous.get("sdk_method"):
        arguments = {
            "sdk_method": previous["sdk_method"],
            "api_version": previous["api_version"],
        }
        derived = f"s2 reported {previous.get('operation_id')} current; read its own spec"
    else:
        state.record(
            "s3_lookup_counterpart",
            executed=False,
            depends_on="s2_lookup_primary",
            note=f"s2 found no operation ({previous.get('reason') or previous.get('error')})",
        )
        return

    started = time.perf_counter()
    payload = run_tool(
        state.ctx, _tool_call(3, "get_openapi_spec", **arguments), state.cfg
    )
    state.record(
        "s3_lookup_counterpart",
        executed=True,
        tool="get_openapi_spec",
        arguments=arguments,
        depends_on="s2_lookup_primary",
        derived=derived,
        ok="error" not in payload,
        note=str(payload.get("operation_id") or payload.get("reason") or ""),
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        result=payload,
    )


def s4_read_best_chunk(state: _State) -> None:
    """Pure selection over what s1 already returned. No tool call.

    Nothing is re-fetched: `ToolConfig.snippet_chars` is 2000 and the longest
    chunk in this corpus is 1950, so s1's passages are already whole. This step
    picks the single passage the composer should read first — the one
    documenting the derived parameter if there is one, else the top-ranked hit
    for the version the question is about.
    """
    if not state.passages:
        state.record("s4_read_best_chunk", executed=False, note="s1 returned nothing")
        return

    wanted = loose(normalise(state.parameter)) if state.parameter else ""
    version = target_version(state.question_text)

    def key(passage: dict[str, Any]) -> tuple[int, int, int]:
        text = loose(normalise(passage.get("text", "")))
        return (
            0 if wanted and wanted in text else 1,
            0 if passage.get("api_version") == version else 1,
            int(passage.get("rank", 99)),
        )

    best = min(state.passages, key=key)
    state.best_chunk = best
    got = str(best.get("api_version", "?"))
    if wanted and wanted in loose(normalise(best.get("text", ""))):
        why = f"contains {state.parameter!r} ({got})"
    else:
        # Name the version actually chosen, not the one wanted — they differ
        # whenever the target version's payload was truncated away.
        why = f"top-ranked {got} passage; none contained a derived parameter"
        if got != version:
            why += f" (wanted {version}; no {version} passage survived truncation)"
    state.record(
        "s4_read_best_chunk",
        executed=True,
        tool="",
        derived=why,
        note=str(best.get("chunk_id", "")),
        result={"chunk_id": best.get("chunk_id", ""), "rank": best.get("rank")},
    )


def _render_context(state: _State) -> str:
    """The pre-assembled context. Same payloads the agent sees, same JSON.

    Internal step names are deliberately kept OUT of what the model reads. A
    first draft labelled each block `[s2_lookup_primary]` and the model cited
    "s2_lookup_primary" as though it were an operationId — the contract asks it
    to cite chunk ids and operationIds, and anything identifier-shaped in the
    context is a candidate. The step names stay in `StepRecord` for the trace,
    where they are for the reader, not the model.
    """
    blocks: list[str] = []
    if state.best_chunk is not None:
        blocks.append(
            "Most relevant passage:\n"
            + json.dumps(state.best_chunk, ensure_ascii=False)
        )
    index = 0
    for entry in state.steps:
        if not entry.executed or not entry.tool:
            continue
        index += 1
        blocks.append(
            f"Tool result {index}: {entry.tool} with {json.dumps(entry.arguments)}\n"
            + json.dumps(entry.result, ensure_ascii=False)
        )
    return "\n\n".join(blocks)


def s5_compose(state: _State, chat: Callable[..., ChatResult]) -> None:
    """One call to the model. No tools attached, so it cannot ask for more.

    Attaching no tool schemas is what makes `llm_calls == 1` structural rather
    than enforced: with nothing to call there is no tool-call turn to service,
    so there is no second request to make even in principle.
    """
    state.context = _render_context(state)
    state.messages = [
        {"role": "system", "content": system_prompt(ARM)},
        {
            "role": "user",
            "content": user_prompt(
                Week7Question(
                    id=state.question_id,
                    question=state.question_text,
                    question_class="",
                    sdk_version="",
                ),
                context=state.context,
            ),
        },
    ]
    started = time.perf_counter()
    state.chat_result = chat(
        state.messages,
        model=state.cfg.resolved_model(),
        tools=None,
        temperature=state.cfg.temperature,
        retries=state.cfg.retries,
    )
    state.record(
        "s5_compose",
        executed=True,
        tool="",
        derived=f"{len(state.context):,} chars of pre-assembled context",
        ok=not state.chat_result.error,
        note=state.chat_result.error or "answered",
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


# --------------------------------------------------------------------------
# the workflow
# --------------------------------------------------------------------------


def run_workflow(
    question: Week7Question,
    cfg: ArmConfig | None = None,
    *,
    chat: Callable[..., ChatResult] = chat_once,
    price: Any = None,
) -> WorkflowRun:
    """Five statements. No loop, no branch that re-enters a step."""
    cfg = cfg or ArmConfig()
    price = price if price is not None else load_price(cfg.resolved_model())
    state = _State(question, cfg)
    started = time.perf_counter()

    s1_search(state)
    s2_lookup_primary(state)
    s3_lookup_counterpart(state)
    s4_read_best_chunk(state)
    s5_compose(state, chat)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = state.chat_result or ChatResult(error="s5_compose did not run")
    meter = Meter(price=price)
    meter.add(result)

    if result.error:
        termination, detail = "chat_error", result.error
        output = TaskOutput(parse_error=f"chat_error: {result.error}")
    elif result.content:
        output = parse_task_output(result.content)
        termination = "final_answer"
        detail = output.parse_error or "composed from the pre-assembled context"
        state.messages.append({"role": "assistant", "content": result.content})
    else:
        termination, detail = "empty_response", "model returned no content"
        output = TaskOutput(parse_error="empty_response: model returned no content")

    return WorkflowRun(
        question_id=question.id,
        question=question.question,
        output=output,
        steps=state.steps,
        tool_calls=list(state.ctx.records),
        tool_sequence=list(state.ctx.path),
        usage=list(meter.laps),
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        cached_tokens=meter.cached_tokens,
        total_tokens=meter.total_tokens,
        usage_missing_calls=meter.usage_missing_calls,
        cost_usd=meter.cost_usd,
        price_verified=price.verified,
        latency_ms=round(elapsed_ms, 3),
        rate_limit_wait_s=result.rate_limit_wait_s,
        termination=termination,
        termination_detail=detail,
        arm_config_sha=cfg.sha(),
        tool_config_sha=cfg.tools.sha(),
        shas=contract_shas(),
        derived_parameter=state.parameter or "",
        derived_target=dict(state.target or {}),
        best_chunk_id=str((state.best_chunk or {}).get("chunk_id", "")),
        context_chars=len(state.context),
        transcript=state.messages,
    )


# --------------------------------------------------------------------------
# the no-loop proof
# --------------------------------------------------------------------------

STEP_FUNCTIONS = (
    "s1_search", "s2_lookup_primary", "s3_lookup_counterpart",
    "s4_read_best_chunk", "s5_compose", "run_workflow",
)
TOOL_CALL_NAMES = {"run_tool", "dispatch"}
LLM_CALL_NAMES = {"chat", "chat_once"}


def _function_nodes(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in STEP_FUNCTIONS
    }


def _called(node: ast.AST, names: set[str]) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in names
    )


def static_call_bounds(module_path: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """Count the call sites and prove none can run more than once.

    This is the compile-time half of the no-loop claim: if every dispatch and
    every chat sits on straight-line code, the runtime maximum IS the number of
    textual call sites, and no run can exceed it whatever the model returns.
    """
    path = module_path or Path(__file__)
    source = path.read_text(encoding="utf-8")
    functions = _function_nodes(source)
    problems: list[str] = []
    report: dict[str, Any] = {"per_function": {}, "tool_sites": 0, "llm_sites": 0}

    missing = [name for name in STEP_FUNCTIONS if name not in functions]
    if missing:
        problems.append(f"missing step function(s): {missing}")

    for name, node in functions.items():
        tool_sites = _called(node, TOOL_CALL_NAMES)
        llm_sites = _called(node, LLM_CALL_NAMES)
        report["per_function"][name] = {"tool": tool_sites, "llm": llm_sites}
        report["tool_sites"] += tool_sites
        report["llm_sites"] += llm_sites

        for child in ast.walk(node):
            if isinstance(child, ast.While):
                problems.append(f"{name} contains a while loop")
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                if _called(child, TOOL_CALL_NAMES) or _called(child, LLM_CALL_NAMES):
                    problems.append(
                        f"{name}: a dispatch or chat call sits inside a loop — its "
                        "count is no longer a compile-time constant"
                    )
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                if _called(child, TOOL_CALL_NAMES) or _called(child, LLM_CALL_NAMES):
                    problems.append(f"{name}: a call hides inside a comprehension")
            # a step calling another step could re-enter it
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in STEP_FUNCTIONS
                and name != "run_workflow"
            ):
                problems.append(f"{name} calls step {child.func.id} — steps must not chain")

    # run_workflow must invoke each step exactly once, in plan order
    plan_calls = [
        child.func.id
        for child in ast.walk(functions["run_workflow"])
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in WORKFLOW_PLAN
    ] if "run_workflow" in functions else []
    if plan_calls != list(WORKFLOW_PLAN):
        problems.append(
            f"run_workflow invokes {plan_calls}, expected exactly {list(WORKFLOW_PLAN)} once each"
        )
    report["plan_calls"] = plan_calls

    if report["tool_sites"] != MAX_TOOL_CALLS:
        problems.append(
            f"{report['tool_sites']} dispatch call sites but MAX_TOOL_CALLS is "
            f"{MAX_TOOL_CALLS} — the constant no longer bounds the code"
        )
    if report["llm_sites"] != MAX_LLM_CALLS:
        problems.append(
            f"{report['llm_sites']} chat call sites but MAX_LLM_CALLS is {MAX_LLM_CALLS}"
        )
    return report, problems


def assert_no_loop(run: WorkflowRun) -> list[str]:
    """Runtime half: check the claim against what actually happened."""
    problems: list[str] = []
    if run.llm_calls != MAX_LLM_CALLS:
        problems.append(
            f"{run.llm_calls} LLM calls; the workflow must make exactly {MAX_LLM_CALLS}"
        )
    if run.tool_call_count > MAX_TOOL_CALLS:
        problems.append(
            f"{run.tool_call_count} tool calls exceeds MAX_TOOL_CALLS={MAX_TOOL_CALLS}"
        )

    executed = run.executed_steps
    if len(executed) != len(set(executed)):
        repeated = sorted({s for s in executed if executed.count(s) > 1})
        problems.append(f"step(s) ran more than once: {repeated} — that is a loop")

    # executed names must be a subsequence of the plan, in order
    remaining = list(WORKFLOW_PLAN)
    for step in executed:
        if step not in remaining:
            problems.append(
                f"{step} ran out of plan order or is not in WORKFLOW_PLAN {WORKFLOW_PLAN}"
            )
            break
        remaining = remaining[remaining.index(step) + 1:]

    for record in run.steps:
        if record.step not in WORKFLOW_PLAN:
            problems.append(f"unknown step {record.step!r}")
    return problems


def verify_step3_dependency(run: WorkflowRun) -> list[str]:
    """s3's arguments must have come out of s2's result, not the question.

    The strong form: for a retired operation, assert the SDK method s3 looked up
    is absent from the question text. If it were present, s3 could have been
    written without s2 and the dependency would be decorative.
    """
    problems: list[str] = []
    step2 = next((s for s in run.steps if s.step == "s2_lookup_primary"), None)
    step3 = next((s for s in run.steps if s.step == "s3_lookup_counterpart"), None)
    if step3 is None:
        return ["s3_lookup_counterpart is missing from the run"]
    if not step3.executed:
        return problems  # legitimately skipped; nothing to prove
    if step3.depends_on != "s2_lookup_primary":
        problems.append(f"s3 records depends_on={step3.depends_on!r}")
    if step2 is None or not step2.executed:
        problems.append("s3 executed but s2 did not")
        return problems

    payload = json.dumps(step2.result, ensure_ascii=False)
    for key, value in step3.arguments.items():
        if str(value) not in payload:
            problems.append(f"s3 argument {key}={value!r} does not appear in s2's result")

    replacement = step2.result.get("replacement")
    if isinstance(replacement, dict):
        method = step3.arguments.get("sdk_method", "")
        if method and method in run.question:
            problems.append(
                f"s3 looked up {method!r}, which the question already names — "
                "the dependency on s2 is not load-bearing here"
            )
    return problems


#: Question metadata the workflow must not read. Mirrors the agent's guard.
ANSWER_KEY_FIELDS = (
    "expected_tools", "expected_order", "required_facts",
    "must_contain", "must_not_contain", "dependency", "question_class",
    "sdk_version", "expect_refusal",
)


def reads_no_answer_key(module_path: Path | None = None) -> list[str]:
    """The steps may read the question TEXT and nothing else about it."""
    path = module_path or Path(__file__)
    functions = _function_nodes(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, node in functions.items():
        attributes = {
            child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
        }
        for field in ANSWER_KEY_FIELDS:
            if field in attributes:
                problems.append(f"{name} reads question.{field} — that is the answer key")
    return problems


def compare_arms(
    agent_run: Any = None, workflow_run: WorkflowRun | None = None
) -> tuple[list[str], list[str]]:
    """Output-contract compatibility between the two arms.

    Static part always runs; the dynamic part only when both runs are supplied.
    """
    from app.rag import week7_agent

    problems: list[str] = []
    notes: list[str] = []

    shared = ("answer_contract", "parser", "scorer", "chat", "tool_schemas")
    shas = contract_shas()
    notes.extend(f"shared {key:16} {shas[key]}" for key in shared)
    if shas["agent_preamble"] == shas["workflow_preamble"]:
        problems.append("the two preambles are identical; one deliberate delta expected")
    notes.append(f"differs agent_preamble    {shas['agent_preamble']}")
    notes.append(f"differs workflow_preamble {shas['workflow_preamble']}")

    # both prompts must be the same contract plus a preamble
    from app.rag.week7_contract import ANSWER_CONTRACT

    for arm in ("agent", "workflow"):
        if not system_prompt(arm).startswith(ANSWER_CONTRACT):
            problems.append(f"{arm} prompt does not start with the shared contract")

    # the metric surface the race will read must exist on both
    required = (
        "question_id", "output", "tool_calls", "tool_sequence", "prompt_tokens",
        "completion_tokens", "total_tokens", "cached_tokens", "cost_usd",
        "latency_ms", "rate_limit_wait_s", "termination", "arm_config_sha",
    )
    agent_fields = set(week7_agent.AgentRun.model_fields)
    workflow_fields = set(WorkflowRun.model_fields)
    for field in required:
        if field not in agent_fields:
            problems.append(f"AgentRun is missing {field}")
        if field not in workflow_fields:
            problems.append(f"WorkflowRun is missing {field}")
    for name in ("answer", "llm_calls", "iterations", "tool_call_count",
                 "adjusted_latency_ms", "path"):
        if not hasattr(week7_agent.AgentRun, name) or not hasattr(WorkflowRun, name):
            problems.append(f"{name} is not exposed by both arms")

    if agent_run is not None and workflow_run is not None:
        if agent_run.question_id != workflow_run.question_id:
            problems.append("the two runs are not the same question")
        if agent_run.arm_config_sha != workflow_run.arm_config_sha:
            problems.append(
                f"arm config differs: {agent_run.arm_config_sha} vs "
                f"{workflow_run.arm_config_sha}"
            )
        if agent_run.tool_config_sha != workflow_run.tool_config_sha:
            problems.append("tool config differs between arms")
        for key in shared:
            if agent_run.shas.get(key) != workflow_run.shas.get(key):
                problems.append(f"{key} sha differs between the two runs")
        notes.append(
            f"both parsed by parse_task_output: agent version="
            f"{agent_run.output.sdk_version!r}, workflow version="
            f"{workflow_run.output.sdk_version!r}"
        )
    return problems, notes


# --------------------------------------------------------------------------
# offline self-test
# --------------------------------------------------------------------------


class _FixedChat:
    """A model that always replies the same way. Used to prove call counts."""

    def __init__(self, content: str | None = None, error: str = "") -> None:
        self.content = content
        self.error = error
        self.calls = 0

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        self.calls += 1
        if kwargs.get("tools"):
            raise AssertionError("the workflow must not attach tool schemas")
        return ChatResult(
            content=self.content,
            error=self.error,
            prompt_tokens=max(1, len(json.dumps(messages)) // 4),
            completion_tokens=40,
            latency_ms=1.0,
        )


GOOD_REPLY = (
    "ANSWER: The replacement defaults to a concurrency of 8.\n"
    "VERSION: v3\nCITATIONS: v3/client-batch#parameters::1"
)


def self_test(questions: list[Week7Question]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    lines: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name:36} {detail}")
        if not ok:
            failures.append(f"self-test {name}: {detail}")

    report, problems = static_call_bounds()
    lines.append("static bounds")
    check("no dispatch or chat inside a loop", not problems, "; ".join(problems) or "clean")
    check("dispatch call sites", report["tool_sites"] == MAX_TOOL_CALLS,
          f"{report['tool_sites']} == MAX_TOOL_CALLS")
    check("chat call sites", report["llm_sites"] == MAX_LLM_CALLS,
          f"{report['llm_sites']} == MAX_LLM_CALLS")
    check("plan invoked once each in order", report["plan_calls"] == list(WORKFLOW_PLAN),
          " -> ".join(report["plan_calls"]))
    key_problems = reads_no_answer_key()
    check("steps read no question metadata", not key_problems,
          "; ".join(key_problems) or "question text only")

    lines.append("every question, offline (tools real, model scripted)")
    worst_tools = 0
    for question in questions:
        chat = _FixedChat(GOOD_REPLY)
        run = run_workflow(question, chat=chat)
        loop_problems = assert_no_loop(run)
        dependency_problems = verify_step3_dependency(run)
        worst_tools = max(worst_tools, run.tool_call_count)
        if loop_problems or dependency_problems or chat.calls != 1:
            check(question.id, False,
                  "; ".join(loop_problems + dependency_problems)
                  or f"{chat.calls} chat calls")
    check("assert_no_loop on all 10", True, f"max tool calls observed {worst_tools}")

    chat = _FixedChat(error="APIConnectionError: reset")
    run = run_workflow(questions[0], chat=chat)
    check("transport failure terminates cleanly", run.termination == "chat_error",
          run.termination)
    chat = _FixedChat(content="no contract shape here")
    run = run_workflow(questions[0], chat=chat)
    check("unparseable answer, no retry", chat.calls == 1 and bool(run.output.parse_error),
          f"{chat.calls} call, parse_error={run.output.parse_error!r}")

    arm_problems, notes = compare_arms()
    lines.append("output contract compatibility with the agent")
    for note in notes:
        lines.append(f"       {note}")
    check("arms share one output contract", not arm_problems,
          "; ".join(arm_problems) or "same contract, parser, scorer, tools")
    return failures, lines


# --------------------------------------------------------------------------
# holdout — questions committed before this arm existed
# --------------------------------------------------------------------------

GOLDEN_SET = Path("eval/golden_set.jsonl")


def holdout_questions(path: Path = GOLDEN_SET) -> list[Week7Question]:
    """Week 4's golden set, adapted. Never used to tune anything here."""
    rows: list[Week7Question] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        rows.append(
            Week7Question(
                id=entry["id"],
                question=entry["question"],
                question_class="holdout",
                sdk_version="none",
                must_contain=entry.get("must_contain", []),
            )
        )
    return rows


def render_holdout(rows: list[Week7Question], docs_root: Path) -> str:
    """Show the derivation firing on inputs it was never shown.

    If `candidate_parameter` were a fitted lookup table, it would return None on
    every one of these. The gold answers are printed beside it so a reader can
    see the derived parameter is the right one, without this file ever reading
    them.
    """
    lines = [
        "HOLDOUT — eval/golden_set.jsonl, committed in Week 4",
        "=" * 92,
        "",
        f"{'id':5} {'derived parameter':22} {'target':34} question",
        "-" * 92,
    ]
    fired = 0
    for row in rows:
        parameter = candidate_parameter(row.question, docs_root)
        target = primary_target(row.question)
        fired += bool(parameter)
        lines.append(
            f"{row.id:5} {parameter or '-':22} "
            f"{json.dumps(target) if target else '-':34} {row.question[:40]}"
        )
    lines.append("-" * 92)
    lines.append(
        f"candidate_parameter fired on {fired}/{len(rows)} unseen questions — a fitted "
        "lookup table would fire on 0"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_run(run: WorkflowRun, question: Week7Question | None = None,
               detail: bool = False) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add(f"{run.question_id}  [{run.arm}]  {run.question}")
    add("=" * 78)
    add(f"  termination   : {run.termination}  ({run.termination_detail})")
    add(f"  tool sequence : {' -> '.join(run.tool_sequence) or '(none)'}")
    add(f"  llm calls     : {run.llm_calls}   tool calls: {run.tool_call_count}"
        f"   steps run: {len(run.executed_steps)}/{len(WORKFLOW_PLAN)}")
    add(f"  derived       : parameter={run.derived_parameter or '-'}   "
        f"target={json.dumps(run.derived_target) if run.derived_target else '-'}")
    add(f"  tokens        : prompt {run.prompt_tokens:,}  completion "
        f"{run.completion_tokens:,}  cached {run.cached_tokens:,}  "
        f"total {run.total_tokens:,}")
    cost = f"${run.cost_usd:.6f}" if run.price_verified else "(price unverified)"
    add(f"  cost          : {cost}")
    add(f"  latency       : {run.latency_ms:,.0f} ms raw   "
        f"{run.adjusted_latency_ms:,.0f} ms excluding "
        f"{run.rate_limit_wait_s:.1f} s of rate-limit sleep")
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
        add("  plan execution")
        header = f"  {'step':24} {'ran':4} {'tool':18} {'from':22} note"
        add(header)
        add("  " + "-" * (len(header) - 2))
        for record in run.steps:
            add(f"  {record.step:24} {'yes' if record.executed else 'no ':4} "
                f"{record.tool or '-':18} {record.depends_on or 'question text':22} "
                f"{record.note[:34]}")
            if record.derived:
                add(f"    {'':22} why: {record.derived}")
        add("  " + "-" * (len(header) - 2))
        add(f"  best chunk: {run.best_chunk_id or '-'}   "
            f"context: {run.context_chars:,} chars")
        loop = assert_no_loop(run)
        dependency = verify_step3_dependency(run)
        add("")
        add("  no-loop      : " + ("OK — 1 LLM call, "
            f"{run.tool_call_count} tool calls, no step repeated" if not loop else "FAILED"))
        for problem in loop:
            add(f"    FAIL {problem}")
        add("  s3 <- s2     : " + ("OK — arguments traced to s2's payload"
                                   if not dependency else "FAILED"))
        for problem in dependency:
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
    parser = argparse.ArgumentParser(description="The Week 7 fixed workflow.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_JSONL)
    parser.add_argument("--question", default=None, help="one id, or a comma-separated list")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--plan", action="store_true", help="constants, call sites, source")
    parser.add_argument("--validate", action="store_true", help="static + offline proofs")
    parser.add_argument("--holdout", action="store_true", help="derivations on golden_set")
    parser.add_argument("--holdout-live", type=int, default=0,
                        help="also run N holdout questions through the model")
    parser.add_argument("--dry-run", action="store_true",
                        help="prompts, plan and a token estimate; makes no LLM call")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.plan:
        report, problems = static_call_bounds()
        print(f"WORKFLOW_PLAN   = {WORKFLOW_PLAN}")
        print(f"MAX_LLM_CALLS   = {MAX_LLM_CALLS}   (an equality, not a ceiling)")
        print(f"MAX_TOOL_CALLS  = {MAX_TOOL_CALLS}")
        print(f"DEFAULT_VERSION = {DEFAULT_VERSION}")
        print()
        print("call sites, counted from the AST")
        for name, counts in report["per_function"].items():
            print(f"  {name:24} dispatch {counts['tool']}   chat {counts['llm']}")
        print(f"  {'TOTAL':24} dispatch {report['tool_sites']}   chat {report['llm_sites']}")
        print()
        for problem in problems:
            print(f"  FAIL {problem}")
        print("source of run_workflow")
        print("-" * 70)
        print(inspect.getsource(run_workflow))
        return 1 if problems else 0

    try:
        questions = load_questions(args.questions)
    except FileNotFoundError as exc:
        print(f"week7 workflow: {exc}", file=sys.stderr)
        return 1

    if args.validate:
        failures, lines = self_test(questions)
        for line in lines:
            print(line if line.startswith("  ") else f"\n{line}")
        print()
        if failures:
            print(f"WORKFLOW CHECKS FAILED — {len(failures)} problem(s)")
            return 1
        print("WORKFLOW CHECKS PASSED")
        return 0

    cfg = ArmConfig()

    if args.holdout:
        rows = holdout_questions()
        print(render_holdout(rows, cfg.tools.docs_root))
        if args.holdout_live:
            print()
            for question in rows[: args.holdout_live]:
                run = run_workflow(question, cfg)
                loop = assert_no_loop(run)
                print(f"  {run.question_id}  {run.termination:14} "
                      f"llm={run.llm_calls} tools={run.tool_call_count} "
                      f"param={run.derived_parameter or '-':20} "
                      f"no-loop={'OK' if not loop else loop}")
                print(f"        {(run.output.answer or '(none)')[:96]}")
        return 0

    if args.dry_run:
        price = load_price(cfg.resolved_model())
        system = system_prompt(ARM)
        selected = questions if not args.question else _select(questions, args.question)
        report, problems = static_call_bounds()
        print(f"DRY RUN — arm={ARM}. No LLM call is made below.\n")
        print(f"model        {cfg.resolved_model()}   temperature {cfg.temperature}")
        print(f"arm sha      {cfg.sha()}   tool cfg sha {cfg.tools.sha()}")
        print(f"plan         {' -> '.join(WORKFLOW_PLAN)}")
        print(f"bounds       exactly {MAX_LLM_CALLS} LLM call, at most "
              f"{MAX_TOOL_CALLS} tool calls "
              f"(AST: {report['llm_sites']} chat + {report['tool_sites']} dispatch sites)")
        print(f"price        ${price.usd_per_1m_input}/${price.usd_per_1m_cached_input}/"
              f"${price.usd_per_1m_output} per 1M   verified {price.verified_on or 'NO'}")
        print("tools sent   [] — s5_compose attaches no schemas, so the model cannot "
              "request one")
        for problem in problems:
            print(f"  FAIL {problem}")
        print()
        print("=" * 74)
        print("SYSTEM PROMPT (verbatim)")
        print("=" * 74)
        print(system)
        print("=" * 74)
        print("DERIVATIONS + token estimate (context size is measured, not guessed)")
        print("=" * 74)
        total = 0
        for question in selected:
            parameter = candidate_parameter(question.question, cfg.tools.docs_root)
            target = primary_target(question.question)
            # A dry run must not call a tool, so the context cannot be built;
            # 7.3k chars is the mean measured over the Step 9 live runs.
            estimate = (len(system) + len(question.question)) // 4 + 7300 // 4
            total += estimate
            print(f"  {question.id}  ~{estimate:,} prompt tokens   "
                  f"param={parameter or '-':14} target={json.dumps(target) if target else 's1 fallback'}")
        print(f"\n  {len(selected)} question(s), ~{total:,} prompt tokens in one pass")
        return 0 if not problems else 1

    selected = questions if args.all else _select(questions, args.question)
    if not args.all and not args.question:
        parser.print_help()
        return 1

    print(f"model {cfg.resolved_model()}  temperature {cfg.temperature}  "
          f"arm sha {cfg.sha()}")
    print()
    runs: list[WorkflowRun] = []
    exit_code = 0
    for index, question in enumerate(selected):
        if index and args.delay:
            time.sleep(args.delay)
        run = run_workflow(question, cfg)
        runs.append(run)
        print(render_run(run, question, detail=args.detail))
        score = score_output(question, run.output, run.tool_calls)
        print(f"  score: {'PASS' if score.passed else 'FAIL'}")
        for check in score.checks:
            mark = "ok " if check.ok else "NO "
            skip = "" if check.applicable else "  (n/a)"
            print(f"    {mark} {check.name:28} {check.detail}{skip}"
                  + (f"  {check.evidence}" if check.evidence else ""))
        for problem in assert_no_loop(run) + verify_step3_dependency(run):
            print(f"    LOOP/DEP FAIL {problem}")
            exit_code = 1
        print()

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
