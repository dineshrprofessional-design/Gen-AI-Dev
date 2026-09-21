"""Week 8 trajectory evaluation: did the agent get there for the right reasons?

    python -m app.rag.trajectory_eval --validate                 # spec vs corpus
    python -m app.rag.trajectory_eval --self-test                # engine, offline
    python -m app.rag.trajectory_eval --replay RUNS.json         # score trajectories
    python -m app.rag.trajectory_eval --replay A.json --compare B.json
    python -m app.rag.trajectory_eval --replay RUNS.json --gap   # outcome vs trajectory

Week 7 scored the final ANSWER. This scores the PATH: which tools were called,
with what arguments, in what order, and how many steps it took. The two can
disagree — an agent can guess a right answer down a wrong path, or walk a
perfect path and fumble the wording — and the gap between them is the thing
Week 8 exists to measure.

## The five metrics

  Tool-Choice Accuracy   correct tool calls / total tool calls. A call is
                         correct when the tool is in the case's allowed set and
                         not in its forbidden set.
  Argument Validity      valid calls / total calls. Separate from tool choice:
                         the right tool with a hallucinated path is a different
                         failure from the wrong tool.
  Step Efficiency        min_steps / actual_steps, capped at 1.0, per case then
                         median. 1.00 means no wasted call.
  Cost p50 / Cost Max    both, because a thrashing agent has a normal median
                         and a catastrophic tail. Reporting only p50 hides
                         exactly the failure mode this week targets.

## Replay, not re-run

Everything here reads a saved `AgentRun` dump. No model, no network, no tokens.
The same JSON scores identically on any machine, which is what makes the
before/after comparison a measurement rather than two separate anecdotes.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.rag.percentile import median, percentile
from app.rag.week7_corpus import OPENAPI_JSON
from app.rag.week7_tools import API_VERSIONS, SDK_METHODS, TOOL_NAMES, _openapi

DEFAULT_CASES = Path("eval/week8_cases.yaml")

#: The six failure modes the regression matrix must cover.
FAILURE_MODES = (
    "redundant_thrashing",
    "argument_violation",
    "step_inefficiency",
    "false_positive_bypass",
    "premature_refusal",
    "budget_overrun",
)


# --------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------


class ArgSpec(BaseModel):
    required: list[str] = Field(default_factory=list)
    api_version: list[str] = Field(default_factory=list)
    sdk_method: list[str] = Field(default_factory=list)
    path: list[str] = Field(default_factory=list)


class TrajectoryCase(BaseModel):
    id: str
    question_id: str
    query: str
    cls: str = Field(alias="class")
    min_steps: int
    max_steps: int
    required_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    ordering: list[list[str]] = Field(default_factory=list)
    commutative: bool = False
    arguments: dict[str, ArgSpec] = Field(default_factory=dict)
    rationale: str = ""

    model_config = {"populate_by_name": True}


def load_cases(path: Path = DEFAULT_CASES) -> list[TrajectoryCase]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [TrajectoryCase(**entry) for entry in payload["cases"]]


# --------------------------------------------------------------------------
# corpus vocabulary — what an argument is allowed to say
# --------------------------------------------------------------------------


def corpus_vocabulary() -> dict[str, set[str]]:
    """Every value an argument may legitimately take, read from the corpus."""
    spec = _openapi(str(OPENAPI_JSON))
    return {
        "api_version": set(API_VERSIONS),
        "sdk_method": set(SDK_METHODS),
        "path": set(spec["paths"]),
    }


# --------------------------------------------------------------------------
# per-call judgement
# --------------------------------------------------------------------------


class CallVerdict(BaseModel):
    seq: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_ok: bool = True
    tool_reason: str = ""
    args_ok: bool = True
    args_reason: str = ""
    redundant: bool = False


def judge_call(
    case: TrajectoryCase,
    seq: int,
    tool: str,
    arguments: dict[str, Any],
    vocabulary: dict[str, set[str]],
    seen: list[tuple[str, str]],
) -> CallVerdict:
    """One tool call, judged on choice and arguments independently."""
    verdict = CallVerdict(seq=seq, tool=tool, arguments=arguments)

    # --- tool choice
    if tool not in TOOL_NAMES:
        verdict.tool_ok, verdict.tool_reason = False, f"unknown tool {tool!r}"
    elif tool in case.forbidden_tools:
        verdict.tool_ok, verdict.tool_reason = False, "tool is forbidden for this case"
    elif case.allowed_tools and tool not in case.allowed_tools:
        verdict.tool_ok = False
        verdict.tool_reason = f"tool not in allowed set {case.allowed_tools}"

    # --- arguments: required, then vocabulary
    spec = case.arguments.get(tool)
    problems: list[str] = []
    if spec:
        for name in spec.required:
            if name not in arguments or arguments[name] in (None, ""):
                problems.append(f"missing required argument {name!r}")
    for name, value in arguments.items():
        if not isinstance(value, str):
            continue
        allowed = vocabulary.get(name)
        if allowed is not None and value not in allowed:
            # An argument the QUESTION supplies is not a hallucination, even
            # when the corpus has no such thing: probing /v1/embeddings because
            # the user named it is how you establish absence.
            if value in case.query:
                continue
            problems.append(f"{name}={value!r} is not in the corpus vocabulary")
        if spec:
            narrowed = getattr(spec, name, None)
            if narrowed and value not in narrowed and value not in case.query:
                problems.append(f"{name}={value!r} should be one of {narrowed}")
    if problems:
        verdict.args_ok, verdict.args_reason = False, "; ".join(problems)

    # --- redundancy: the same tool with the same arguments, already asked
    signature = (tool, json.dumps(arguments, sort_keys=True))
    if signature in seen:
        verdict.redundant = True
    seen.append(signature)
    return verdict


# --------------------------------------------------------------------------
# per-case judgement
# --------------------------------------------------------------------------


class TrajectoryResult(BaseModel):
    case_id: str
    question_id: str
    cls: str
    steps: int
    calls: list[CallVerdict] = Field(default_factory=list)

    trajectory_pass: bool = False
    failures: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)

    outcome_pass: bool = False
    step_efficiency: float = 0.0
    redundant_calls: int = 0

    llm_calls: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    termination: str = ""
    budget_type: str = ""

    @property
    def false_positive(self) -> bool:
        """Answer passed, path did not. The gap, one case at a time."""
        return self.outcome_pass and not self.trajectory_pass

    @property
    def false_negative(self) -> bool:
        return self.trajectory_pass and not self.outcome_pass


def _order_ok(case: TrajectoryCase, sequence: list[str]) -> list[str]:
    """Enforce [before, after] pairs, but only where a dependency is real."""
    if case.commutative or not case.ordering:
        return []
    problems = []
    for before, after in case.ordering:
        if before in sequence and after in sequence:
            if sequence.index(before) > sequence.index(after):
                problems.append(
                    f"{after} was called before {before}; step 2's arguments "
                    "cannot have come from step 1"
                )
        elif after in sequence and before not in sequence:
            problems.append(f"{after} ran without the {before} it depends on")
    return problems


def evaluate_case(
    case: TrajectoryCase, run: dict[str, Any], vocabulary: dict[str, set[str]]
) -> TrajectoryResult:
    records = run.get("tool_calls", [])
    sequence = [r["name"] for r in records]
    seen: list[tuple[str, str]] = []
    calls = [
        judge_call(case, r.get("seq", i + 1), r["name"], r.get("arguments", {}),
                   vocabulary, seen)
        for i, r in enumerate(records)
    ]

    failures: list[str] = []
    modes: set[str] = set()

    missing = [t for t in case.required_tools if t not in sequence]
    if missing:
        failures.append(f"required tool(s) never called: {missing}")
        modes.add("false_positive_bypass")

    forbidden = sorted({t for t in sequence if t in case.forbidden_tools})
    if forbidden:
        failures.append(f"forbidden tool(s) called: {forbidden}")
        modes.add("false_positive_bypass")

    outside = sorted({
        t for t in sequence if case.allowed_tools and t not in case.allowed_tools
    })
    if outside:
        failures.append(f"tool(s) outside the allowed set: {outside}")
        modes.add("false_positive_bypass")

    steps = len(sequence)
    if steps > case.max_steps:
        failures.append(f"{steps} steps exceeds max_steps {case.max_steps}")
        modes.add("step_inefficiency")
    if steps < case.min_steps:
        failures.append(f"{steps} steps below min_steps {case.min_steps}")
        modes.add("premature_refusal")

    order_problems = _order_ok(case, sequence)
    failures.extend(order_problems)
    if order_problems:
        modes.add("false_positive_bypass")

    bad_args = [c for c in calls if not c.args_ok]
    for call in bad_args:
        failures.append(f"step {call.seq} {call.tool}: {call.args_reason}")
        modes.add("argument_violation")

    redundant = sum(1 for c in calls if c.redundant)
    if redundant:
        failures.append(f"{redundant} redundant call(s) repeating an earlier query")
        modes.add("redundant_thrashing")

    termination = run.get("termination", "")
    budget_type = (run.get("budget_trip") or {}).get("budget", "") if run.get("budget_trip") else ""
    if termination == "budget":
        failures.append(f"terminated on budget: {budget_type}")
        modes.add("budget_overrun")

    efficiency = min(1.0, case.min_steps / steps) if steps else (1.0 if case.min_steps == 0 else 0.0)

    output = run.get("output", {})
    return TrajectoryResult(
        case_id=case.id,
        question_id=case.question_id,
        cls=case.cls,
        steps=steps,
        calls=calls,
        trajectory_pass=not failures,
        failures=failures,
        modes=sorted(modes),
        step_efficiency=efficiency,
        redundant_calls=redundant,
        llm_calls=len(run.get("laps", [])),
        total_tokens=run.get("total_tokens", 0),
        cost_usd=run.get("cost_usd", 0.0),
        latency_ms=run.get("latency_ms", 0.0),
        termination=termination,
        budget_type=budget_type,
        outcome_pass=bool(run.get("_outcome_pass", False)),
    )


# --------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------


class SuiteMetrics(BaseModel):
    label: str
    cases: int
    total_calls: int

    tool_choice_accuracy: float
    correct_tool_calls: int
    argument_validity: float
    valid_arg_calls: int

    step_efficiency_p50: float
    step_efficiency_mean: float

    cost_p50_usd: float
    cost_max_usd: float
    latency_p50_ms: float
    latency_max_ms: float
    tokens_total: int
    tokens_max: int
    llm_calls_total: int

    trajectory_pass_rate: float
    outcome_pass_rate: float
    gap_points: float

    mode_counts: dict[str, int] = Field(default_factory=dict)
    false_positives: list[str] = Field(default_factory=list)
    false_negatives: list[str] = Field(default_factory=list)


def summarise(results: list[TrajectoryResult], label: str) -> SuiteMetrics:
    calls = [c for r in results for c in r.calls]
    total = len(calls)
    correct_tool = sum(1 for c in calls if c.tool_ok)
    valid_args = sum(1 for c in calls if c.args_ok)
    efficiencies = [r.step_efficiency for r in results]
    costs = [r.cost_usd for r in results]
    latencies = [r.latency_ms for r in results]

    counts = {mode: 0 for mode in FAILURE_MODES}
    for result in results:
        for mode in result.modes:
            counts[mode] = counts.get(mode, 0) + 1

    trajectory_rate = sum(1 for r in results if r.trajectory_pass) / len(results)
    outcome_rate = sum(1 for r in results if r.outcome_pass) / len(results)

    return SuiteMetrics(
        label=label,
        cases=len(results),
        total_calls=total,
        tool_choice_accuracy=correct_tool / total if total else 1.0,
        correct_tool_calls=correct_tool,
        argument_validity=valid_args / total if total else 1.0,
        valid_arg_calls=valid_args,
        step_efficiency_p50=median(efficiencies),
        step_efficiency_mean=statistics.fmean(efficiencies) if efficiencies else 0.0,
        cost_p50_usd=median(costs),
        cost_max_usd=max(costs) if costs else 0.0,
        latency_p50_ms=median(latencies),
        latency_max_ms=max(latencies) if latencies else 0.0,
        tokens_total=sum(r.total_tokens for r in results),
        tokens_max=max((r.total_tokens for r in results), default=0),
        llm_calls_total=sum(r.llm_calls for r in results),
        trajectory_pass_rate=trajectory_rate,
        outcome_pass_rate=outcome_rate,
        gap_points=(outcome_rate - trajectory_rate) * 100,
        mode_counts=counts,
        false_positives=[r.case_id for r in results if r.false_positive],
        false_negatives=[r.case_id for r in results if r.false_negative],
    )


def load_runs(path: Path) -> dict[str, dict[str, Any]]:
    """AgentRun dumps, keyed by question id, with outcome scored in."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload if isinstance(payload, list) else payload.get("runs", [])
    return {r["question_id"]: r for r in runs}


def score_outcomes(runs: dict[str, dict[str, Any]]) -> None:
    """Attach Week 7's own scorer verdict. Same scorer, unchanged."""
    from app.rag.week6_assertions import chunk_texts
    from app.rag.week7_contract import TaskOutput, score_output
    from app.rag.week7_questions import load_questions
    from app.rag.week7_tools import ToolCallRecord

    texts = chunk_texts()
    questions = {q.id: q for q in load_questions()}
    for qid, run in runs.items():
        question = questions.get(qid)
        if question is None:
            run["_outcome_pass"] = False
            continue
        output = TaskOutput(**run["output"])
        records = [ToolCallRecord(**r) for r in run.get("tool_calls", [])]
        run["_outcome_pass"] = score_output(question, output, records, texts).passed


def evaluate_suite(
    cases: list[TrajectoryCase], runs: dict[str, dict[str, Any]]
) -> list[TrajectoryResult]:
    vocabulary = corpus_vocabulary()
    results = []
    for case in cases:
        run = runs.get(case.question_id)
        if run is None:
            continue
        results.append(evaluate_case(case, run, vocabulary))
    return results


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render(results: list[TrajectoryResult], metrics: SuiteMetrics) -> str:
    lines = [
        "=" * 84,
        f"TRAJECTORY EVALUATION — {metrics.label}",
        "=" * 84,
        "",
        f"  {'metric':28} {'value':>14}   built from",
        "  " + "-" * 78,
        f"  {'Tool-Choice Accuracy':28} {metrics.tool_choice_accuracy:>13.1%}   "
        f"{metrics.correct_tool_calls}/{metrics.total_calls} tool calls",
        f"  {'Argument Validity':28} {metrics.argument_validity:>13.1%}   "
        f"{metrics.valid_arg_calls}/{metrics.total_calls} calls",
        f"  {'Step Efficiency p50':28} {metrics.step_efficiency_p50:>13.2f}   "
        f"min_steps / actual, per case",
        f"  {'Step Efficiency mean':28} {metrics.step_efficiency_mean:>13.2f}",
        f"  {'Cost p50':28} {'$' + format(metrics.cost_p50_usd, '.6f'):>13}   "
        f"{metrics.cases} cases",
        f"  {'Cost Max':28} {'$' + format(metrics.cost_max_usd, '.6f'):>13}   "
        f"the tail this week targets",
        f"  {'Latency p50':28} {metrics.latency_p50_ms:>12,.0f}ms",
        f"  {'Latency Max':28} {metrics.latency_max_ms:>12,.0f}ms",
        f"  {'Tokens total':28} {metrics.tokens_total:>13,}",
        f"  {'Tokens max (one case)':28} {metrics.tokens_max:>13,}",
        f"  {'LLM calls total':28} {metrics.llm_calls_total:>13}",
        "  " + "-" * 78,
        f"  {'Outcome pass rate':28} {metrics.outcome_pass_rate:>13.1%}",
        f"  {'Trajectory pass rate':28} {metrics.trajectory_pass_rate:>13.1%}",
        f"  {'GAP (outcome - traj)':28} {metrics.gap_points:>+12.1f}pp   "
        f"false positives: {metrics.false_positives or 'none'}",
        "",
        "FAILURE MODES",
        "  " + "-" * 78,
    ]
    for mode in FAILURE_MODES:
        lines.append(f"  {mode:28} {metrics.mode_counts.get(mode, 0):>3} case(s)")
    lines += ["", "PER CASE", "  " + "-" * 78,
              f"  {'case':7} {'q':7} {'class':14} {'steps':>5} {'eff':>5} "
              f"{'traj':>5} {'outc':>5}  notes"]
    for r in results:
        flag = "FP" if r.false_positive else ("FN" if r.false_negative else "")
        lines.append(
            f"  {r.case_id:7} {r.question_id:7} {r.cls:14} {r.steps:>5} "
            f"{r.step_efficiency:>5.2f} {'PASS' if r.trajectory_pass else 'FAIL':>5} "
            f"{'PASS' if r.outcome_pass else 'FAIL':>5}  {flag} "
            f"{(r.failures[0][:44] if r.failures else '')}"
        )
    return "\n".join(lines)


def render_compare(before: SuiteMetrics, after: SuiteMetrics) -> str:
    def row(name: str, a: Any, b: Any, fmt: str = "") -> str:
        fa = format(a, fmt) if fmt else str(a)
        fb = format(b, fmt) if fmt else str(b)
        return f"  {name:28} {fa:>14} {fb:>14}"

    lines = [
        "=" * 76,
        f"BEFORE / AFTER — {before.label} vs {after.label}",
        "=" * 76,
        f"  {'metric':28} {before.label[:14]:>14} {after.label[:14]:>14}",
        "  " + "-" * 60,
        row("Tool-Choice Accuracy", before.tool_choice_accuracy, after.tool_choice_accuracy, ".1%"),
        row("Argument Validity", before.argument_validity, after.argument_validity, ".1%"),
        row("Step Efficiency p50", before.step_efficiency_p50, after.step_efficiency_p50, ".2f"),
        row("Cost p50 ($)", before.cost_p50_usd, after.cost_p50_usd, ".6f"),
        row("Cost Max ($)", before.cost_max_usd, after.cost_max_usd, ".6f"),
        row("Latency Max (ms)", before.latency_max_ms, after.latency_max_ms, ",.0f"),
        row("Tokens total", before.tokens_total, after.tokens_total, ","),
        row("LLM calls total", before.llm_calls_total, after.llm_calls_total, ","),
        row("Trajectory pass rate", before.trajectory_pass_rate, after.trajectory_pass_rate, ".1%"),
        row("Outcome pass rate", before.outcome_pass_rate, after.outcome_pass_rate, ".1%"),
        "  " + "-" * 60,
    ]
    for mode in FAILURE_MODES:
        lines.append(row(mode, before.mode_counts.get(mode, 0), after.mode_counts.get(mode, 0)))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# validation + self-test
# --------------------------------------------------------------------------


def validate(cases: list[TrajectoryCase]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    lines: list[str] = []
    vocabulary = corpus_vocabulary()
    from app.rag.week7_questions import load_questions

    questions = {q.id: q for q in load_questions()}

    if len(cases) != 10:
        failures.append(f"expected 10 cases, found {len(cases)}")
    lines.append(f"  cases                {len(cases)}")

    for case in cases:
        if case.question_id not in questions:
            failures.append(f"{case.id}: unknown question {case.question_id}")
        elif questions[case.question_id].question != case.query:
            failures.append(f"{case.id}: query text drifted from eval/week7_questions")
        for tool in case.required_tools + case.allowed_tools + case.forbidden_tools:
            if tool not in TOOL_NAMES:
                failures.append(f"{case.id}: unknown tool {tool!r}")
        overlap = set(case.required_tools) & set(case.forbidden_tools)
        if overlap:
            failures.append(f"{case.id}: {overlap} both required and forbidden")
        missing = set(case.required_tools) - set(case.allowed_tools)
        if missing:
            failures.append(f"{case.id}: required {missing} not in allowed_tools")
        if case.min_steps > case.max_steps:
            failures.append(f"{case.id}: min_steps > max_steps")
        if not case.rationale:
            failures.append(f"{case.id}: no rationale")
        for tool, spec in case.arguments.items():
            for name in ("api_version", "sdk_method", "path"):
                for value in getattr(spec, name):
                    if value not in vocabulary[name] and value not in case.query:
                        failures.append(
                            f"{case.id}: {tool}.{name}={value!r} is not in the corpus"
                        )
        if case.ordering and case.commutative:
            failures.append(f"{case.id}: ordering set on a commutative case")

    ordered = [c.id for c in cases if c.ordering]
    commutative = [c.id for c in cases if c.commutative]
    lines.append(f"  ordering enforced    {ordered}")
    lines.append(f"  commutative          {commutative}")
    lines.append(f"  vocabulary           {len(vocabulary['path'])} paths, "
                 f"{len(vocabulary['sdk_method'])} methods, "
                 f"{len(vocabulary['api_version'])} versions")
    if not ordered:
        failures.append("no case enforces ordering; dependencies would go unmeasured")
    if not commutative:
        failures.append("no case is marked commutative; version checks would be over-strict")
    return failures, lines


def self_test() -> tuple[list[str], list[str]]:
    """Synthetic trajectories with known verdicts. No network, no tokens."""
    failures: list[str] = []
    lines: list[str] = []
    cases = {c.id: c for c in load_cases()}
    vocabulary = corpus_vocabulary()

    def check(name: str, ok: bool, detail: str) -> None:
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name:44} {detail}")
        if not ok:
            failures.append(f"self-test {name}: {detail}")

    def run(calls, **extra):
        return {"tool_calls": [
            {"seq": i + 1, "name": n, "arguments": a} for i, (n, a) in enumerate(calls)
        ], "laps": [{}] * (len(calls) + 1), "output": {}, **extra}

    # 1. a clean single-hop trajectory
    r = evaluate_case(cases["w8-01"],
                      run([("search_docs", {"query": "4xx", "api_version": "v3"})]),
                      vocabulary)
    check("clean single-hop passes", r.trajectory_pass, f"eff={r.step_efficiency:.2f}")

    # 2. forbidden tool is caught
    r = evaluate_case(cases["w8-01"],
                      run([("check_deprecation", {"api_version": "v3",
                                                  "sdk_method": "Client.send"})]),
                      vocabulary)
    check("forbidden tool fails", not r.trajectory_pass and "false_positive_bypass" in r.modes,
          r.failures[0][:52])

    # 3. commutative version order is NOT penalised
    v3_first = run([("search_docs", {"query": "heartbeat_ms", "api_version": "v3"}),
                    ("search_docs", {"query": "heartbeat_ms", "api_version": "v2"})])
    v2_first = run([("search_docs", {"query": "heartbeat_ms", "api_version": "v2"}),
                    ("search_docs", {"query": "heartbeat_ms", "api_version": "v3"})])
    a = evaluate_case(cases["w8-04"], v3_first, vocabulary)
    b = evaluate_case(cases["w8-04"], v2_first, vocabulary)
    check("commutative order both pass", a.trajectory_pass and b.trajectory_pass,
          "v3-then-v2 and v2-then-v3 score identically")

    # 4. a real dependency IS ordered
    wrong = run([("search_docs", {"query": "max_retries", "api_version": "v3"}),
                 ("check_deprecation", {"api_version": "v2", "path": "/v2/messages"})])
    r = evaluate_case(cases["w8-06"], wrong, vocabulary)
    check("two-hop wrong order fails", not r.trajectory_pass, r.failures[0][:52])
    right = run([("check_deprecation", {"api_version": "v2", "path": "/v2/messages"}),
                 ("search_docs", {"query": "max_retries", "api_version": "v3"})])
    check("two-hop right order passes",
          evaluate_case(cases["w8-06"], right, vocabulary).trajectory_pass, "ordered")

    # 5. missing required argument
    r = evaluate_case(cases["w8-01"], run([("search_docs", {"query": "4xx"})]), vocabulary)
    check("missing required arg fails", "argument_violation" in r.modes, r.failures[0][:52])

    # 6. hallucinated vocabulary
    r = evaluate_case(cases["w8-06"],
                      run([("check_deprecation", {"api_version": "v2",
                                                  "path": "/v2/nonexistent"}),
                           ("search_docs", {"query": "x", "api_version": "v3"})]),
                      vocabulary)
    check("hallucinated path fails", "argument_violation" in r.modes, r.failures[0][:56])

    # 7. an argument the QUESTION supplies is not a hallucination
    r = evaluate_case(cases["w8-10"],
                      run([("get_openapi_spec", {"api_version": "v3",
                                                 "path": "/v1/embeddings"})]),
                      vocabulary)
    check("question-supplied arg is not hallucination", r.trajectory_pass,
          "/v1/embeddings probed because the user named it")

    # 8. redundancy + thrashing
    thrash = run([("search_docs", {"query": "embeddings", "api_version": "v3"})] * 5)
    r = evaluate_case(cases["w8-10"], thrash, vocabulary)
    check("thrashing caught", "redundant_thrashing" in r.modes and
          "step_inefficiency" in r.modes, f"{r.redundant_calls} redundant, {r.steps} steps")

    # 9. step efficiency arithmetic
    r = evaluate_case(cases["w8-06"],
                      run([("check_deprecation", {"api_version": "v2", "path": "/v2/messages"}),
                           ("search_docs", {"query": "a", "api_version": "v3"}),
                           ("search_docs", {"query": "b", "api_version": "v3"})]),
                      vocabulary)
    check("efficiency = min/actual", abs(r.step_efficiency - 2 / 3) < 1e-9,
          f"min 2 / actual 3 = {r.step_efficiency:.3f}")

    # 10. budget overrun is its own mode
    r = evaluate_case(cases["w8-10"], run(
        [("search_docs", {"query": "x", "api_version": "v3"})],
        termination="budget", budget_trip={"budget": "max_tokens"}), vocabulary)
    check("budget overrun flagged", "budget_overrun" in r.modes, r.budget_type)

    # 11. false positive is detectable
    bad = run([("check_deprecation", {"api_version": "v3", "sdk_method": "Client.send"})])
    bad["_outcome_pass"] = True
    r = evaluate_case(cases["w8-01"], bad, vocabulary)
    check("false positive detected", r.false_positive,
          "outcome PASS with a failing trajectory")
    return failures, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 8 trajectory evaluation.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gap", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)

    if args.validate:
        failures, lines = validate(cases)
        print("trajectory spec validation (offline)\n")
        for line in lines:
            print(line)
        print()
        for failure in failures:
            print(f"FAIL  {failure}")
        if failures:
            print(f"\nVALIDATION FAILED — {len(failures)} problem(s)")
            return 1
        print("VALIDATION PASSED")
        return 0

    if args.self_test:
        failures, lines = self_test()
        print("trajectory engine self-test (synthetic, no network)\n")
        for line in lines:
            print(line)
        print()
        if failures:
            print(f"SELF-TEST FAILED — {len(failures)} problem(s)")
            return 1
        print("SELF-TEST PASSED")
        return 0

    if not args.replay:
        parser.print_help()
        return 1

    runs = load_runs(args.replay)
    score_outcomes(runs)
    results = evaluate_suite(cases, runs)
    metrics = summarise(results, args.label or args.replay.stem)
    print(render(results, metrics))

    if args.compare:
        other = load_runs(args.compare)
        score_outcomes(other)
        other_results = evaluate_suite(cases, other)
        other_metrics = summarise(other_results, args.compare.stem)
        print()
        print(render_compare(metrics, other_metrics))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "metrics": metrics.model_dump(),
            "results": [r.model_dump() for r in results],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
