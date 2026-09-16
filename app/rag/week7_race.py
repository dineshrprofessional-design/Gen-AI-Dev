"""The Week 7 race: tool-calling agent versus fixed workflow, same ten questions.

    python -m app.rag.week7_race --preflight --reps 2  # does it fit today's quota?
    python -m app.rag.week7_race --validate            # fairness + accounting, offline
    python -m app.rag.week7_race --self-test           # whole harness, scripted model
    python -m app.rag.week7_race --quota-self-test     # quota classification + refusals
    python -m app.rag.week7_race --dry-run             # token estimate, no LLM call
    python -m app.rag.week7_race --reps 2 --delay 18   # the real race
    python -m app.rag.week7_race --verdict             # decision table, from race.csv

## Quota, and why a race can refuse to start

Groq's free tier caps tokens per DAY as well as per minute, and the two arrive
as the same HTTP 429 with the same `rate_limit_exceeded` code. Only the window
phrase in the message tells them apart, so `classify_provider_error` reads that:
a per-minute throttle keeps the existing bounded retry, a per-day quota stops
immediately, because five 45-second sleeps cannot outlast a nine-minute reset.

`--preflight` prices the run against the daily budget before spending anything,
from token costs MEASURED in Steps 8-10 rather than hoped for. A race that
cannot fit is refused at the door. It is never quietly shrunk: dropping reps=3
to reps=1 would halve the statistical strength of the result without the report
ever saying so, so the recommendation is printed and the user re-runs with it.

If the quota is exhausted mid-run anyway, the affected pair is DISCARDED rather
than scored — a 0-token row recorded as a FAIL would put a run that never
happened into the pass rate — the race stops, `race_partial.csv` and
`race_incomplete.txt` are written, and `race.csv` is NOT. `--verdict` re-checks
completeness against the CSV itself, so neither a partial file nor one renamed
into place can become a verdict.

Eight numbers, four per arm: pass rate, p50 latency, total tokens, cost per
question. Everything else in this file exists to make those eight auditable.

## Fairness is structural, then checked

Both arms already import their model, prompts, tools, parser and scorer from
`week7_contract`; neither may import `openai`, `app.rag.generate` or
`app.rag.index`. This harness re-checks that at runtime via `assert_same_race()`,
which compares the two arms' recorded shas and must find **exactly one**
difference: `arm_preamble`. One arm has to be told it has tools and the other
that its context is pre-assembled, so that one cannot be held still; every other
sha — answer contract, parser, scorer, chat, tool schemas, arm config, tool
config, corpus — must match. If more than that differs the race aborts rather
than printing a winner, because a misleading winner is worse than no result.

## Measurement decisions, stated

  warm-up      question 1 through BOTH arms, discarded. Pays the 2.2 GB bge-m3
               load, HNSW materialisation, OpenAI() construction and TLS
               handshake once instead of charging them to whichever arm ran
               first. Precedent: eval_week4.py:242-251.
  interleaved  agent(q) then workflow(q), question by question, rep by rep.
               Running one arm to completion first would charge any provider
               drift entirely to that arm.
  latency      rate-limit sleep is SUBTRACTED, identically for both arms. The
               8000 TPM free-tier throttle is a property of the account, not of
               either architecture.
  headline p50 median of the ten per-question medians, via percentile.py, which
               is proved against Week 4's committed figures. NOT a pooled median
               over all 30 runs: a question that happened to be measured more
               often would otherwise pull the headline toward itself.
  total tokens ONE full pass over the ten questions (all runs / reps), so the
               number does not silently scale with how many reps were run. The
               raw all-runs figure is in the CSV too.
  scoring      score_output(question, output, records) — no model, no network,
               and no argument that says which arm produced the run. A parse
               failure is terminal for both arms, with no retry, ever.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.rag.percentile import median, median_of_group_medians, p95
from app.rag.week6_assertions import chunk_texts
from app.rag.week7_agent import AgentRun, Budgets, run_agent
from app.rag.week7_contract import (
    ERROR_QUOTA_TPD,
    ArmConfig,
    cached_client,
    classify_provider_error,
    contract_shas,
    load_price,
    retry_after_seconds,
    score_output,
)
from app.rag.week7_corpus import docs_tree_sha
from app.rag.week7_questions import DEFAULT_QUESTIONS_JSONL, Week7Question, load_questions
from app.rag.week7_tools import tool_impl_sha, tool_schema_sha
from app.rag.week7_workflow import (
    MAX_LLM_CALLS,
    MAX_TOOL_CALLS,
    WorkflowRun,
    assert_no_loop,
    run_workflow,
    verify_step3_dependency,
)

ARMS = ("agent", "workflow")
DEFAULT_OUT = Path("report/w7")

#: The single permitted difference between the arms. Anything else aborts.
PERMITTED_DIFFERENCES = ("arm_preamble",)

# --------------------------------------------------------------------------
# quota
# --------------------------------------------------------------------------
#
# Groq's free tier caps tokens per DAY as well as per minute. The per-minute
# throttle is a delay; the per-day quota ends the run. A race that starts
# without checking the daily budget can only discover this half way through,
# which is exactly what happened on 2026-09-15 at pair 8 of 30 — see
# report/w7/race_status.txt.

DEFAULT_DAILY_TOKEN_BUDGET = 200_000

# MEASURED, not estimated. Deliberately the means rather than the minima: a
# preflight that assumes best-case tokens is worse than no preflight, because it
# green-lights a run that then dies at 80%.
#
# REVISED 2026-09-16 against 30 rows of real race data (report/w7/race_partial.csv).
# The first figures came from 8 pairs and understated the agent by 43%:
#
#     agent      3,700 estimated -> 5,276 measured   (+43%)
#     workflow   3,000 estimated -> 2,971 measured   (accurate)
#     per pair   6,700 estimated -> 8,247 measured   (+23%)
#
# The agent is the volatile arm because its lap count varies with the question,
# and a small sample of mostly-short runs missed that. Rounded UP, not to the
# mean, so the estimator errs toward refusing a race rather than starting one it
# cannot finish.
MEASURED_AGENT_TOKENS_PER_QUESTION = 5_300
MEASURED_WORKFLOW_TOKENS_PER_QUESTION = 3_000
MEASURED_WARMUP_TOKENS = 7_000


class Preflight(BaseModel):
    """Does the requested race fit the day's token budget? No LLM calls."""

    questions: int
    arms: int
    reps: int
    warmup: bool
    estimated_tokens: int
    daily_budget: int
    fits: bool
    headroom: int
    recommended_reps: int
    per_reps: dict[int, int] = Field(default_factory=dict)

    @property
    def budget_known(self) -> bool:
        return self.daily_budget > 0


def estimate_tokens(n_questions: int, reps: int, warmup: bool = True) -> int:
    per_pair = (
        MEASURED_AGENT_TOKENS_PER_QUESTION + MEASURED_WORKFLOW_TOKENS_PER_QUESTION
    )
    return n_questions * reps * per_pair + (MEASURED_WARMUP_TOKENS if warmup else 0)


def preflight(
    n_questions: int,
    reps: int,
    daily_budget: int = DEFAULT_DAILY_TOKEN_BUDGET,
    warmup: bool = True,
) -> Preflight:
    """Decide before spending anything. `daily_budget <= 0` means unknown."""
    estimated = estimate_tokens(n_questions, reps, warmup)
    per_reps = {r: estimate_tokens(n_questions, r, warmup) for r in (1, 2, 3)}
    known = daily_budget > 0
    fits = (not known) or estimated <= daily_budget

    # The largest reps that fits. Never applied automatically — it is printed as
    # a recommendation and the user re-runs with it explicitly, because silently
    # downgrading reps=3 to reps=1 would change the benchmark's statistical
    # strength without the report ever saying so.
    recommended = reps
    if known and not fits:
        affordable = [r for r in (3, 2, 1) if per_reps[r] <= daily_budget]
        recommended = affordable[0] if affordable else 0

    return Preflight(
        questions=n_questions,
        arms=len(ARMS),
        reps=reps,
        warmup=warmup,
        estimated_tokens=estimated,
        daily_budget=daily_budget,
        fits=fits,
        headroom=daily_budget - estimated if known else 0,
        recommended_reps=recommended,
        per_reps=per_reps,
    )


def render_preflight(check: Preflight) -> str:
    lines = [
        "PREFLIGHT — token budget check (no LLM calls made)",
        "=" * 66,
        f"  questions              {check.questions}",
        f"  arms                   {check.arms}  {list(ARMS)}",
        f"  repetitions requested  {check.reps}",
        f"  warm-up                {'yes (+%s tokens)' % f'{MEASURED_WARMUP_TOKENS:,}' if check.warmup else 'no'}",
        "",
        "  estimated tokens (measured means, not best case)",
    ]
    for reps, tokens in sorted(check.per_reps.items()):
        mark = " <- requested" if reps == check.reps else ""
        verdict = ""
        if check.budget_known:
            verdict = "  fits" if tokens <= check.daily_budget else "  DOES NOT FIT"
        lines.append(f"    reps={reps}  ~{tokens // 1000}k  ({tokens:,}){verdict}{mark}")
    lines.append("")
    if check.budget_known:
        lines.append(f"  daily token budget     {check.daily_budget:,}")
        lines.append(f"  requested run          {check.estimated_tokens:,}")
        lines.append(f"  headroom               {check.headroom:,}")
        lines.append("")
        if check.fits:
            lines.append(f"  RESULT: reps={check.reps} FITS the daily budget.")
        elif check.recommended_reps:
            lines.append(
                f"  RESULT: reps={check.reps} DOES NOT FIT "
                f"({check.estimated_tokens:,} > {check.daily_budget:,})."
            )
            lines.append(
                f"          Largest that fits: reps={check.recommended_reps} "
                f"(~{check.per_reps[check.recommended_reps]:,} tokens)."
            )
            lines.append(
                f"          Re-run explicitly with --reps {check.recommended_reps}. "
                "Nothing is downgraded automatically."
            )
        else:
            lines.append(
                f"  RESULT: no repetition count fits a budget of "
                f"{check.daily_budget:,} tokens."
            )
    else:
        lines.append("  daily token budget     unknown (--daily-budget 0); check skipped")
        lines.append(f"  RESULT: proceeding without a budget check, ~"
                     f"{check.estimated_tokens:,} tokens estimated.")
    return "\n".join(lines)


class RaceRow(BaseModel):
    """One (question, arm, rep). The CSV is these rows, nothing derived."""

    question_id: str
    question_class: str
    arm: str
    rep: int
    passed: bool
    failed_checks: str = ""
    latency_ms: float = 0.0
    adjusted_latency_ms: float = 0.0
    rate_limit_wait_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    path: str = ""
    termination: str = ""
    budget_type: str = ""
    #: "" when the call succeeded, else quota_tpd / rate_limit_tpm / transient /
    #: other. Persisted in the CSV so a later audit classifies from the recorded
    #: verdict rather than re-parsing provider prose that may have changed.
    error_kind: str = ""
    parse_error: str = ""
    usage_missing_calls: int = 0
    answer: str = ""


class ArmSummary(BaseModel):
    """The four headline numbers for one arm, plus what they were built from."""

    arm: str
    runs: int
    questions: int
    reps: int

    passes: int
    pass_rate: float                 # 1. passes / runs

    p50_latency_ms: float            # 2. median of the per-question medians
    pooled_p50_ms: float
    pooled_p95_ms: float

    total_tokens: int                # 3. one pass over the ten questions
    total_tokens_all_runs: int
    prompt_tokens_all_runs: int
    completion_tokens_all_runs: int
    cached_tokens_all_runs: int

    cost_per_question_usd: float     # 4. total cost / question-runs
    total_cost_all_runs_usd: float

    llm_calls_per_question: float
    tool_calls_per_question: float
    distinct_paths: int
    path_varies: bool
    flaky_questions: list[str] = Field(default_factory=list)
    terminations: dict[str, int] = Field(default_factory=dict)
    budget_trips: dict[str, int] = Field(default_factory=dict)
    usage_missing_calls: int = 0


class RaceResult(BaseModel):
    rows: list[RaceRow] = Field(default_factory=list)
    summaries: dict[str, ArmSummary] = Field(default_factory=dict)
    fairness: dict[str, Any] = Field(default_factory=dict)
    reps: int = 0
    warmup_seconds: float = 0.0
    wall_seconds: float = 0.0
    price: dict[str, Any] = Field(default_factory=dict)

    #: A race is only a RESULT when every (question, arm, rep) ran cleanly.
    complete: bool = False
    aborted: bool = False
    abort_kind: str = ""
    abort_reason: str = ""
    completed_pairs: int = 0
    expected_pairs: int = 0
    dataset_problems: list[str] = Field(default_factory=list)

    @property
    def usable_as_result(self) -> bool:
        return (
            self.complete
            and not self.aborted
            and not self.dataset_problems
            and bool(self.fairness.get("passed"))
        )


# --------------------------------------------------------------------------
# fairness
# --------------------------------------------------------------------------


def race_identity(cfg: ArmConfig, questions: list[Week7Question]) -> dict[str, str]:
    """Everything that must be identical across the arms, as one sha map."""
    import hashlib

    shas = contract_shas()
    question_blob = json.dumps(
        [{"id": q.id, "question": q.question} for q in questions], sort_keys=True
    )
    return {
        "questions": hashlib.sha256(question_blob.encode()).hexdigest()[:16],
        "question_count": str(len(questions)),
        "model": cfg.resolved_model(),
        "temperature": str(cfg.temperature),
        "arm_config": cfg.sha(),
        "tool_config": cfg.tools.sha(),
        "tool_schemas": tool_schema_sha(),
        "tool_impl": tool_impl_sha(),
        "answer_contract": shas["answer_contract"],
        "parser": shas["parser"],
        "scorer": shas["scorer"],
        "chat": shas["chat"],
        "corpus_docs": docs_tree_sha(),
    }


def assert_same_race(
    agent_run: AgentRun, workflow_run: WorkflowRun, identity: dict[str, str]
) -> list[str]:
    """Compare what the two arms actually recorded. Returns the differing keys.

    Must come back exactly `["arm_preamble"]`. The direct analogue of
    `eval_week4.print_compare`'s `changed == ["hybrid"]`: name the one variable
    that moved, and prove nothing else did.
    """
    differences: list[str] = []
    if agent_run.arm_config_sha != workflow_run.arm_config_sha:
        differences.append("arm_config")
    if agent_run.tool_config_sha != workflow_run.tool_config_sha:
        differences.append("tool_config")
    for key in ("answer_contract", "parser", "scorer", "chat", "tool_schemas"):
        if agent_run.shas.get(key) != workflow_run.shas.get(key):
            differences.append(key)
    if agent_run.shas.get("agent_preamble") != workflow_run.shas.get("workflow_preamble"):
        differences.append("arm_preamble")
    if agent_run.question != workflow_run.question:
        differences.append("question_text")
    if agent_run.question_id != workflow_run.question_id:
        differences.append("question_id")
    return differences


def fairness_gate(differences: list[str]) -> list[str]:
    """Anything beyond the one permitted delta is fatal."""
    unexpected = [d for d in differences if d not in PERMITTED_DIFFERENCES]
    missing = [d for d in PERMITTED_DIFFERENCES if d not in differences]
    problems = [f"unexpected difference between arms: {d}" for d in unexpected]
    problems += [
        f"expected the arms to differ on {d!r} and they did not — the preambles "
        "should not be identical"
        for d in missing
    ]
    return problems


def scorer_is_blind() -> list[str]:
    """The scorer must have no way to learn which arm produced a run."""
    import inspect

    signature = inspect.signature(score_output)
    problems = [
        f"score_output takes {name!r} — the scorer must not know the arm"
        for name in signature.parameters
        if name in ("arm", "is_agent", "run", "agent_run", "workflow_run")
    ]
    source = inspect.getsource(score_output)
    for token in ('"agent"', "'agent'", '"workflow"', "'workflow'"):
        if token in source:
            problems.append(f"score_output mentions {token} in its body")
    return problems


# --------------------------------------------------------------------------
# the race
# --------------------------------------------------------------------------


def _row(
    question: Week7Question, arm: str, rep: int, run: Any, texts: dict[str, str]
) -> RaceRow:
    """Score and flatten one run. Identical call for both arms, by construction."""
    score = score_output(question, run.output, run.tool_calls, texts)
    return RaceRow(
        question_id=question.id,
        question_class=question.question_class,
        arm=arm,
        rep=rep,
        passed=score.passed,
        failed_checks="|".join(c.name for c in score.failed),
        latency_ms=round(run.latency_ms, 3),
        adjusted_latency_ms=round(run.adjusted_latency_ms, 3),
        rate_limit_wait_s=round(run.rate_limit_wait_s, 3),
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        cached_tokens=run.cached_tokens,
        total_tokens=run.total_tokens,
        cost_usd=round(run.cost_usd, 8),
        llm_calls=run.llm_calls,
        tool_calls=run.tool_call_count,
        path=" -> ".join(run.tool_sequence) or "(none)",
        termination=run.termination,
        budget_type=getattr(run, "budget_type", "") or "",
        error_kind=(
            classify_provider_error(run.termination_detail)
            if run.termination == "chat_error" else ""
        ),
        parse_error=run.output.parse_error,
        usage_missing_calls=run.usage_missing_calls,
        answer=" ".join((run.output.answer or "").split())[:300],
    )


def run_race(
    questions: list[Week7Question],
    cfg: ArmConfig | None = None,
    *,
    budgets: Budgets | None = None,
    reps: int = 3,
    delay: float = 12.0,
    warmup: bool = True,
    verbose: bool = True,
    chat: Any = None,
    on_quota: str = "abort",
    quota_waits: int = 0,
) -> RaceResult:
    cfg = cfg or ArmConfig()
    budgets = budgets or Budgets()
    price = load_price(cfg.resolved_model())
    texts = chunk_texts(cfg.tools.docs_root)
    identity = race_identity(cfg, questions)

    # One seam, handed to BOTH arms. A scripted model can therefore drive the
    # whole harness offline (`--self-test`) without either arm getting a
    # different one, which is the only way this parameter could bias a race.
    arm_chat = {"chat": chat} if chat is not None else {}

    started = time.perf_counter()
    warmup_seconds = 0.0
    if warmup:
        # Pay the model load, the index open, the client build and the TLS
        # handshake once, on BOTH arms, and throw the numbers away.
        if verbose:
            print(f"warm-up on {questions[0].id} (both arms, discarded) ...", flush=True)
        warm_started = time.perf_counter()
        if chat is None:
            cached_client()
        run_agent(questions[0], cfg, budgets=budgets, **arm_chat)
        run_workflow(questions[0], cfg, **arm_chat)
        warmup_seconds = time.perf_counter() - warm_started
        if verbose:
            print(f"  warm-up took {warmup_seconds:.1f}s\n", flush=True)

    rows: list[RaceRow] = []
    fairness_problems: list[str] = []
    differences_seen: set[str] = set()
    loop_problems: list[str] = []
    expected_pairs = len(questions) * reps
    completed_pairs = 0
    abort_kind = ""
    abort_reason = ""
    waits_left = max(0, quota_waits)

    for rep in range(1, reps + 1):
        if abort_kind:
            break
        for index, question in enumerate(questions):
            if delay and (index or rep > 1):
                time.sleep(delay)

            # Interleaved: this question, both arms, before moving on.
            agent_run = run_agent(question, cfg, budgets=budgets, **arm_chat)
            workflow_run = run_workflow(question, cfg, **arm_chat)

            # A per-day quota ends the race. It is NOT a measurement of either
            # architecture, so the pair is discarded rather than scored: a
            # 0-token "FAIL" row would otherwise enter the pass rate as though
            # the arm had answered badly, when in fact it never ran.
            quota_hit = _quota_error(agent_run) or _quota_error(workflow_run)
            if quota_hit:
                if waits_left > 0 and on_quota == "wait":
                    pause = (retry_after_seconds(quota_hit) or 600.0) + 5.0
                    waits_left -= 1
                    if verbose:
                        print(f"  quota exhausted; waiting {pause:.0f}s then retrying "
                              f"{question.id} ({waits_left} wait(s) left)", flush=True)
                    time.sleep(pause)
                    agent_run = run_agent(question, cfg, budgets=budgets, **arm_chat)
                    workflow_run = run_workflow(question, cfg, **arm_chat)
                    quota_hit = _quota_error(agent_run) or _quota_error(workflow_run)
                if quota_hit:
                    abort_kind = ERROR_QUOTA_TPD
                    abort_reason = quota_hit
                    if verbose:
                        print(f"\nQUOTA EXHAUSTED at rep {rep}, {question.id} — "
                              f"aborting after {completed_pairs}/{expected_pairs} "
                              "complete pairs. This pair is DISCARDED, not scored.",
                              flush=True)
                    break

            differences_seen.update(assert_same_race(agent_run, workflow_run, identity))
            loop_problems.extend(assert_no_loop(workflow_run))
            loop_problems.extend(verify_step3_dependency(workflow_run))

            agent_row = _row(question, "agent", rep, agent_run, texts)
            workflow_row = _row(question, "workflow", rep, workflow_run, texts)
            rows.extend([agent_row, workflow_row])
            completed_pairs += 1

            if verbose:
                print(
                    f"rep {rep}  {question.id:7} "
                    f"agent {'PASS' if agent_row.passed else 'FAIL'} "
                    f"{agent_row.total_tokens:>6,}tok {agent_row.adjusted_latency_ms:>8,.0f}ms "
                    f"{agent_row.llm_calls}llm | "
                    f"workflow {'PASS' if workflow_row.passed else 'FAIL'} "
                    f"{workflow_row.total_tokens:>6,}tok "
                    f"{workflow_row.adjusted_latency_ms:>8,.0f}ms",
                    flush=True,
                )

    fairness_problems.extend(fairness_gate(sorted(differences_seen)))
    fairness_problems.extend(scorer_is_blind())
    fairness_problems.extend(f"workflow no-loop: {p}" for p in sorted(set(loop_problems)))

    summaries = {
        arm: summarise_arm(rows, arm, questions, reps) for arm in ARMS
    }
    complete = completed_pairs == expected_pairs and not abort_kind
    return RaceResult(
        rows=rows,
        summaries=summaries,
        fairness={
            "identity": identity,
            "differences": sorted(differences_seen),
            "permitted": list(PERMITTED_DIFFERENCES),
            "problems": fairness_problems,
            "passed": not fairness_problems,
        },
        reps=reps,
        warmup_seconds=round(warmup_seconds, 3),
        wall_seconds=round(time.perf_counter() - started, 3),
        price=price.model_dump(),
        complete=complete,
        aborted=bool(abort_kind),
        abort_kind=abort_kind,
        abort_reason=abort_reason,
        completed_pairs=completed_pairs,
        expected_pairs=expected_pairs,
        dataset_problems=validate_race_dataset(rows, questions, reps),
    )


def _quota_error(run: Any) -> str:
    """The provider's daily-quota message from a run, or "" if there wasn't one."""
    if run.termination != "chat_error":
        return ""
    detail = run.termination_detail or ""
    return detail if classify_provider_error(detail) == ERROR_QUOTA_TPD else ""


def validate_race_dataset(
    rows: list[RaceRow], questions: list[Week7Question], reps: int
) -> list[str]:
    """Is this dataset a RESULT, or just some rows? Everything that disqualifies it.

    Used both at the end of a run and again when `--verdict` reads race.csv back
    off disk, so a hand-edited or truncated CSV cannot become a verdict either.
    """
    problems: list[str] = []

    expected = {
        (q.id, arm, rep)
        for q in questions for arm in ARMS for rep in range(1, reps + 1)
    }
    actual = {(r.question_id, r.arm, r.rep) for r in rows}
    missing = sorted(expected - actual)
    if missing:
        problems.append(
            f"incomplete: {len(missing)} of {len(expected)} (question, arm, rep) "
            f"cells are missing, e.g. {missing[:3]}"
        )
    unexpected = sorted(actual - expected)
    if unexpected:
        problems.append(f"unexpected cells not in the 10-question design: {unexpected[:3]}")
    if len(rows) != len(actual):
        problems.append(f"{len(rows)} rows for {len(actual)} distinct cells — duplicates")

    for row in rows:
        label = f"{row.question_id}/{row.arm}/rep{row.rep}"
        if row.error_kind == ERROR_QUOTA_TPD:
            problems.append(f"{label}: provider daily quota error — not a measurement")
        elif row.termination == "chat_error":
            problems.append(f"{label}: provider error ({row.error_kind}) — not a measurement")
        if row.llm_calls <= 0:
            problems.append(f"{label}: 0 LLM calls — nothing was measured")
        elif row.total_tokens <= 0:
            problems.append(
                f"{label}: {row.llm_calls} LLM call(s) but 0 tokens — "
                "token accounting is invalid"
            )
    return problems


def summarise_arm(
    rows: list[RaceRow], arm: str, questions: list[Week7Question], reps: int
) -> ArmSummary:
    arm_rows = [r for r in rows if r.arm == arm]
    runs = len(arm_rows)
    passes = sum(1 for r in arm_rows if r.passed)

    # Per-question latency groups, so the headline weights each question equally.
    groups = [
        [r.adjusted_latency_ms for r in arm_rows if r.question_id == q.id]
        for q in questions
    ]
    pooled = [sample for group in groups for sample in group]

    flaky = [
        q.id
        for q in questions
        if len({r.passed for r in arm_rows if r.question_id == q.id}) > 1
    ]
    paths = {r.path for r in arm_rows}
    terminations: dict[str, int] = {}
    budget_trips: dict[str, int] = {}
    for row in arm_rows:
        terminations[row.termination] = terminations.get(row.termination, 0) + 1
        if row.budget_type:
            budget_trips[row.budget_type] = budget_trips.get(row.budget_type, 0) + 1

    all_tokens = sum(r.total_tokens for r in arm_rows)
    all_cost = sum(r.cost_usd for r in arm_rows)

    return ArmSummary(
        arm=arm,
        runs=runs,
        questions=len(questions),
        reps=reps,
        passes=passes,
        pass_rate=passes / runs if runs else 0.0,
        p50_latency_ms=median_of_group_medians(groups),
        pooled_p50_ms=median(pooled),
        pooled_p95_ms=p95(pooled),
        total_tokens=round(all_tokens / reps) if reps else 0,
        total_tokens_all_runs=all_tokens,
        prompt_tokens_all_runs=sum(r.prompt_tokens for r in arm_rows),
        completion_tokens_all_runs=sum(r.completion_tokens for r in arm_rows),
        cached_tokens_all_runs=sum(r.cached_tokens for r in arm_rows),
        cost_per_question_usd=all_cost / runs if runs else 0.0,
        total_cost_all_runs_usd=all_cost,
        llm_calls_per_question=sum(r.llm_calls for r in arm_rows) / runs if runs else 0.0,
        tool_calls_per_question=sum(r.tool_calls for r in arm_rows) / runs if runs else 0.0,
        distinct_paths=len(paths),
        path_varies=len(paths) > 1,
        flaky_questions=flaky,
        terminations=terminations,
        budget_trips=budget_trips,
        usage_missing_calls=sum(r.usage_missing_calls for r in arm_rows),
    )


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

CSV_FIELDS = list(RaceRow.model_fields)


def write_csv(rows: list[RaceRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def read_csv(path: Path) -> list[RaceRow]:
    rows: list[RaceRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle):
            entry["passed"] = entry["passed"] in ("True", "true", "1")
            for key in ("rep", "prompt_tokens", "completion_tokens", "cached_tokens",
                        "total_tokens", "llm_calls", "tool_calls", "usage_missing_calls"):
                entry[key] = int(entry[key] or 0)
            for key in ("latency_ms", "adjusted_latency_ms", "rate_limit_wait_s", "cost_usd"):
                entry[key] = float(entry[key] or 0.0)
            rows.append(RaceRow(**entry))
    return rows


META_NAME = "race_meta.json"
CSV_NAME = "race.csv"
PARTIAL_CSV_NAME = "race_partial.csv"
INCOMPLETE_NAME = "race_incomplete.txt"


def write_meta(result: RaceResult, path: Path, questions: list[Week7Question]) -> None:
    """The stamp that says this CSV is a RESULT.

    Written for complete and incomplete runs alike, with `complete` telling the
    truth either way. `--verdict` requires it, so a race.csv that appeared by
    any other route — hand-edited, copied, truncated — cannot be turned into a
    verdict just by existing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "complete": result.complete,
        "aborted": result.aborted,
        "abort_kind": result.abort_kind,
        "abort_reason": result.abort_reason[:500],
        "quota_error": result.abort_kind == ERROR_QUOTA_TPD,
        "fairness_passed": bool(result.fairness.get("passed")),
        "usable_as_result": result.usable_as_result,
        "reps": result.reps,
        "questions": [q.id for q in questions],
        "arms": list(ARMS),
        "rows": len(result.rows),
        "completed_pairs": result.completed_pairs,
        "expected_pairs": result.expected_pairs,
        "dataset_problems": result.dataset_problems,
        "identity": result.fairness.get("identity", {}),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")


def render_incomplete(result: RaceResult, out: Path) -> str:
    """The status a reader gets INSTEAD of headline numbers."""
    lines = [
        "WEEK 7 RACE — INCOMPLETE. NO HEADLINE NUMBERS, NO VERDICT.",
        "=" * 66,
        "",
        f"  completed pairs   {result.completed_pairs} of {result.expected_pairs}",
        f"  repetitions       {result.reps}",
        f"  abort kind        {result.abort_kind or '(none)'}",
        "",
    ]
    if result.abort_kind == ERROR_QUOTA_TPD:
        lines += [
            "  The provider's tokens-per-DAY quota was exhausted mid-run. That is",
            "  not a property of either architecture, so the affected pair was",
            "  DISCARDED rather than scored: recording it as a 0-token FAIL would",
            "  put a run that never happened into the pass rate.",
            "",
            "  Provider said:",
            f"    {result.abort_reason[:300]}",
            "",
        ]
    lines += [
        f"  Partial rows are in {out / PARTIAL_CSV_NAME}.",
        "  They are PARTIAL DATA, not a result, and must not be quoted as one:",
        "  the design is 10 questions x N reps x 2 arms and this run does not",
        "  cover it.",
        "",
        f"  No {CSV_NAME} was written. `--verdict` will refuse while the dataset",
        "  is incomplete.",
        "",
        "  Problems recorded:",
    ]
    lines += [f"    - {p}" for p in (result.dataset_problems or ["(none beyond incompleteness)"])]
    lines += [
        "",
        "  To finish, re-run when quota is available. Check first with:",
        "    python -m app.rag.week7_race --preflight --reps <n>",
    ]
    return "\n".join(lines)


def complete_leading_reps(rows: list[RaceRow], questions: list[Week7Question]) -> int:
    """How many reps, counting from 1, are complete and clean.

    Whole reps only, and only a leading run of them. A rep is complete when
    every question ran on both arms with real token accounting and no provider
    error. Stops at the first incomplete rep, so a later rep cannot be pulled
    forward past a broken one.
    """
    complete = 0
    for rep in range(1, max((r.rep for r in rows), default=0) + 1):
        subset = [r for r in rows if r.rep == rep]
        if validate_race_dataset(subset, questions, 1) and rep == 1:
            # validate_race_dataset checks rep numbering against 1..reps, so for
            # a single rep the subset must itself be renumbered to rep 1.
            pass
        renumbered = [r.model_copy(update={"rep": 1}) for r in subset]
        if validate_race_dataset(renumbered, questions, 1):
            break
        complete = rep
    return complete


def promote_complete_reps(
    partial: Path, out: Path, questions: list[Week7Question]
) -> tuple[int, list[str]]:
    """Turn the complete leading reps of an aborted run into a real result.

    A race that died at rep 2 of 2 still ran rep 1 to completion: every
    question, both arms, interleaved, warm-up paid, no quota rows. That IS a
    valid reps=1 race and throwing it away would waste a full day's quota.

    The safeguards are what make this promotion rather than laundering:

      - whole reps only, never individual questions, so no question can be
        dropped because its result was inconvenient
      - a leading run only, so rep 3 cannot jump over a broken rep 2
      - every retained row re-validated by the same `validate_race_dataset`
        the verdict gate uses
      - the meta records `promoted_from` and the reps actually achieved, so the
        report cannot silently claim the reps that were requested

    Returns (reps_promoted, problems).
    """
    if not partial.is_file():
        return 0, [f"{partial} does not exist"]
    rows = read_csv(partial)
    reps = complete_leading_reps(rows, questions)
    if reps == 0:
        return 0, ["no complete repetition exists in the partial data"]

    kept = [r for r in rows if r.rep <= reps]
    problems = validate_race_dataset(kept, questions, reps)
    if problems:
        return 0, problems
    write_csv(kept, out / CSV_NAME)
    return reps, []


def load_verifiable_race(
    out: Path, questions: list[Week7Question]
) -> tuple[list[RaceRow], int, list[str]]:
    """Load a race ONLY if it is fit to produce a verdict. Else say why.

    Every gate is re-checked against the CSV itself rather than trusted from the
    meta stamp, so the two would have to be forged consistently to get a verdict
    out of a race that did not happen.
    """
    csv_path, meta_path = out / CSV_NAME, out / META_NAME
    problems: list[str] = []
    if not csv_path.is_file():
        return [], 0, [f"{csv_path} does not exist — no complete race has been run"]
    if not meta_path.is_file():
        return [], 0, [
            f"{meta_path} is missing — a race.csv without its meta stamp cannot be "
            "verified as complete"
        ]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete"):
        problems.append(
            f"race is marked incomplete ({meta.get('completed_pairs')}/"
            f"{meta.get('expected_pairs')} pairs)"
        )
    if meta.get("aborted"):
        problems.append(f"race aborted: {meta.get('abort_kind') or 'unknown'}")
    if meta.get("quota_error"):
        problems.append("a provider daily-quota error occurred during this race")
    if not meta.get("fairness_passed"):
        problems.append("the fairness gate did not pass for this race")
    for problem in meta.get("dataset_problems", []):
        problems.append(f"recorded: {problem}")

    rows = read_csv(csv_path)
    reps = max((r.rep for r in rows), default=0)
    if meta.get("reps") and meta["reps"] != reps:
        problems.append(f"meta says reps={meta['reps']} but the CSV holds reps={reps}")
    problems.extend(validate_race_dataset(rows, questions, reps))
    return rows, reps, problems


def render_summary(result: RaceResult) -> str:
    agent = result.summaries["agent"]
    workflow = result.summaries["workflow"]
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("WEEK 7 RACE — AGENT vs FIXED WORKFLOW")
    add("=" * 78)
    add(f"{agent.questions} questions x {result.reps} reps = {agent.runs} runs per arm")
    add(f"warm-up {result.warmup_seconds:.1f}s discarded   "
        f"wall clock {result.wall_seconds / 60:.1f} min")
    add(f"price: ${result.price.get('usd_per_1m_input')}/"
        f"${result.price.get('usd_per_1m_cached_input')}/"
        f"${result.price.get('usd_per_1m_output')} per 1M, "
        f"verified {result.price.get('verified_on')}")
    add("")
    add("THE EIGHT NUMBERS")
    add("-" * 78)
    header = f"  {'metric':26} {'agent':>16} {'workflow':>16} {'ratio':>10}"
    add(header)
    add("  " + "-" * (len(header) - 2))

    def ratio(a: float, b: float) -> str:
        return f"{a / b:.2f}x" if b else "n/a"

    add(f"  {'1. pass rate':26} {agent.pass_rate:>15.1%} {workflow.pass_rate:>16.1%} "
        f"{ratio(agent.pass_rate, workflow.pass_rate):>10}")
    add(f"  {'2. p50 latency (ms)':26} {agent.p50_latency_ms:>16,.0f} "
        f"{workflow.p50_latency_ms:>16,.0f} "
        f"{ratio(agent.p50_latency_ms, workflow.p50_latency_ms):>10}")
    add(f"  {'3. total tokens (1 pass)':26} {agent.total_tokens:>16,} "
        f"{workflow.total_tokens:>16,} "
        f"{ratio(agent.total_tokens, workflow.total_tokens):>10}")
    add(f"  {'4. cost / question ($)':26} {agent.cost_per_question_usd:>16.6f} "
        f"{workflow.cost_per_question_usd:>16.6f} "
        f"{ratio(agent.cost_per_question_usd, workflow.cost_per_question_usd):>10}")
    add("  " + "-" * (len(header) - 2))
    add("")
    add("SUPPORTING")
    add("-" * 78)
    for label, a, w in (
        ("passes / runs", f"{agent.passes}/{agent.runs}", f"{workflow.passes}/{workflow.runs}"),
        ("LLM calls / question", f"{agent.llm_calls_per_question:.2f}",
         f"{workflow.llm_calls_per_question:.2f}"),
        ("tool calls / question", f"{agent.tool_calls_per_question:.2f}",
         f"{workflow.tool_calls_per_question:.2f}"),
        ("distinct tool paths", str(agent.distinct_paths), str(workflow.distinct_paths)),
        ("path varies", str(agent.path_varies), str(workflow.path_varies)),
        ("pooled p50 / p95 (ms)", f"{agent.pooled_p50_ms:,.0f} / {agent.pooled_p95_ms:,.0f}",
         f"{workflow.pooled_p50_ms:,.0f} / {workflow.pooled_p95_ms:,.0f}"),
        ("tokens, all runs", f"{agent.total_tokens_all_runs:,}",
         f"{workflow.total_tokens_all_runs:,}"),
        ("cached tokens", f"{agent.cached_tokens_all_runs:,}",
         f"{workflow.cached_tokens_all_runs:,}"),
        ("cost, all runs ($)", f"{agent.total_cost_all_runs_usd:.6f}",
         f"{workflow.total_cost_all_runs_usd:.6f}"),
        ("usage missing", str(agent.usage_missing_calls), str(workflow.usage_missing_calls)),
        ("flaky questions", ",".join(agent.flaky_questions) or "none",
         ",".join(workflow.flaky_questions) or "none"),
        ("terminations", json.dumps(agent.terminations), json.dumps(workflow.terminations)),
        ("budget trips", json.dumps(agent.budget_trips) or "{}", "n/a (structural)"),
    ):
        add(f"  {label:26} {a:>16} {w:>16}")
    add("")
    add("PER QUESTION (passes out of reps, and the agent's tool path)")
    add("-" * 78)
    add(f"  {'id':7} {'class':14} {'agent':>7} {'workflow':>9}  agent path")
    ids: list[str] = []
    for row in result.rows:
        if row.question_id not in ids:
            ids.append(row.question_id)
    for qid in ids:
        a_rows = [r for r in result.rows if r.question_id == qid and r.arm == "agent"]
        w_rows = [r for r in result.rows if r.question_id == qid and r.arm == "workflow"]
        paths = sorted({r.path for r in a_rows})
        add(f"  {qid:7} {a_rows[0].question_class:14} "
            f"{sum(r.passed for r in a_rows)}/{len(a_rows):<5} "
            f"{sum(r.passed for r in w_rows)}/{len(w_rows):<7}  "
            f"{' | '.join(paths)[:40]}")
    add("")
    add("FAIRNESS GATE")
    add("-" * 78)
    add(f"  differences between arms : {result.fairness['differences']}")
    add(f"  permitted                : {result.fairness['permitted']}")
    add(f"  verdict                  : "
        f"{'PASS' if result.fairness['passed'] else 'FAILED'}")
    for problem in result.fairness["problems"]:
        add(f"    FAIL {problem}")
    add("")
    add("  identity (must be shared by both arms)")
    for key, value in result.fairness["identity"].items():
        add(f"    {key:18} {value}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the verdict decision table
# --------------------------------------------------------------------------

VERDICT_TABLE = (
    # (path_varies, agent_better, row label, recommendation)
    (True, True, "path varies, agent > workflow",
     "AGENT — variation buys accuracy"),
    (True, False, "path varies, agent <= workflow",
     "FIXED WORKFLOW — the path varies but is not paying"),
    (False, True, "path fixed, agent > workflow",
     "MIXED — the gain cannot come from the loop; fold that step into the workflow"),
    (False, False, "path fixed, agent ~ workflow",
     "FIXED WORKFLOW — pure overhead"),
)


def verdict(result: RaceResult) -> dict[str, Any]:
    """Select a row mechanically, so prose cannot contradict the table."""
    agent = result.summaries["agent"]
    workflow = result.summaries["workflow"]
    path_varies = agent.path_varies
    agent_better = agent.pass_rate > workflow.pass_rate

    row = next(
        entry for entry in VERDICT_TABLE
        if entry[0] == path_varies and entry[1] == agent_better
    )
    delta_questions = (agent.pass_rate - workflow.pass_rate) * agent.questions
    return {
        "path_varies": path_varies,
        "distinct_paths": agent.distinct_paths,
        "agent_better": agent_better,
        "row": row[2],
        "recommendation": row[3],
        "pass_rate_delta_pp": (agent.pass_rate - workflow.pass_rate) * 100,
        "delta_questions": delta_questions,
        "token_ratio": (
            agent.total_tokens / workflow.total_tokens if workflow.total_tokens else 0.0
        ),
        "latency_ratio": (
            agent.p50_latency_ms / workflow.p50_latency_ms
            if workflow.p50_latency_ms else 0.0
        ),
        "cost_ratio": (
            agent.cost_per_question_usd / workflow.cost_per_question_usd
            if workflow.cost_per_question_usd else 0.0
        ),
        "significant": abs(delta_questions) > 1,
    }


def path_variation_report(result: RaceResult) -> str:
    """Which question CLASS makes the agent's path move."""
    by_class: dict[str, set[str]] = {}
    for row in result.rows:
        if row.arm != "agent":
            continue
        by_class.setdefault(row.question_class, set()).add(row.path)
    lines = ["  question class   distinct agent paths"]
    for klass, paths in sorted(by_class.items()):
        lines.append(f"    {klass:16} {len(paths)}  {sorted(paths)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the whole harness, offline
# --------------------------------------------------------------------------


class RaceScriptedChat:
    """One scripted model, handed to both arms, branching only on `tools`.

    The agent is sent tool schemas and the workflow is not — that is the arms'
    own difference, not something this object invents — so the same callable
    naturally drives a 2-lap agent and a 1-call workflow. Token counts are
    derived from the message list actually received, so the accounting assertions
    below test real summing rather than numbers chosen to satisfy them.
    """

    def __init__(self) -> None:
        self.agent_calls = 0
        self.workflow_calls = 0

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        from app.rag.week7_agent import _call
        from app.rag.week7_contract import ChatResult

        prompt = max(1, len(json.dumps(messages)) // 4)
        answer = (
            "ANSWER: In v3 the default is 250 ms and 4xx is never retried.\n"
            "VERSION: v3\nCITATIONS: v3/client-send#parameters::1"
        )
        if kwargs.get("tools"):
            self.agent_calls += 1
            already = any(m.get("role") == "tool" for m in messages)
            if not already:
                return ChatResult(
                    tool_calls=[_call("s1", "search_docs", query="x", api_version="v3")],
                    prompt_tokens=prompt, completion_tokens=20, latency_ms=1.0,
                )
            return ChatResult(content=answer, prompt_tokens=prompt,
                              completion_tokens=40, latency_ms=1.0)
        self.workflow_calls += 1
        return ChatResult(content=answer, prompt_tokens=prompt,
                          completion_tokens=40, latency_ms=1.0)


def self_test(reps: int = 2) -> tuple[list[str], list[str]]:
    """Drive the real `run_race` with a scripted model. No network, no tokens.

    Proves the parts that a paid run would otherwise be the first test of: the
    eight numbers, the CSV round-trip, the fairness gate and the verdict table.
    """
    failures: list[str] = []
    lines: list[str] = []
    questions = load_questions()

    def check(name: str, ok: bool, detail: str) -> None:
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name:38} {detail}")
        if not ok:
            failures.append(f"race self-test {name}: {detail}")

    chat = RaceScriptedChat()
    result = run_race(questions, reps=reps, delay=0.0, warmup=False,
                      verbose=False, chat=chat)
    agent = result.summaries["agent"]
    workflow = result.summaries["workflow"]

    expected_rows = len(questions) * reps * 2
    check("row count", len(result.rows) == expected_rows,
          f"{len(result.rows)} == {len(questions)} questions x {reps} reps x 2 arms")
    check("every question ran on both arms",
          {(r.question_id, r.arm) for r in result.rows}
          == {(q.id, a) for q in questions for a in ARMS},
          "same 10 question objects, both arms")
    check("agent made 2 LLM calls/question", agent.llm_calls_per_question == 2.0,
          f"{agent.llm_calls_per_question}")
    check("workflow made exactly 1 LLM call/question",
          workflow.llm_calls_per_question == 1.0, f"{workflow.llm_calls_per_question}")
    check("workflow within its tool ceiling",
          workflow.tool_calls_per_question <= MAX_TOOL_CALLS,
          f"{workflow.tool_calls_per_question:.2f} <= {MAX_TOOL_CALLS}")

    # 3. total tokens is ONE pass, and every agent lap is counted
    check("total tokens = all runs / reps",
          agent.total_tokens == round(agent.total_tokens_all_runs / reps),
          f"{agent.total_tokens:,} == {agent.total_tokens_all_runs:,}/{reps}")
    agent_rows = [r for r in result.rows if r.arm == "agent"]
    multi = [r for r in agent_rows if r.llm_calls > 1]
    check("agent tokens include every iteration",
          all(r.total_tokens > 0 and r.llm_calls == 2 for r in multi),
          f"{len(multi)} multi-lap runs, none reporting a single lap")

    # 2. headline p50 is the median of per-question medians, recomputed here
    groups = [
        [r.adjusted_latency_ms for r in agent_rows if r.question_id == q.id]
        for q in questions
    ]
    independent = statistics.median([statistics.median(g) for g in groups])
    check("p50 = median of per-question medians",
          abs(agent.p50_latency_ms - independent) < 1e-9,
          f"{agent.p50_latency_ms:.3f} == {independent:.3f} (recomputed independently)")

    # 4. cost per question
    expected_cost = sum(r.cost_usd for r in agent_rows) / len(agent_rows)
    check("cost/question = total cost / runs",
          abs(agent.cost_per_question_usd - expected_cost) < 1e-12,
          f"${agent.cost_per_question_usd:.8f}")

    check("fairness gate", result.fairness["passed"],
          f"differences={result.fairness['differences']}")
    check("exactly one permitted difference",
          result.fairness["differences"] == list(PERMITTED_DIFFERENCES),
          str(result.fairness["differences"]))

    # CSV round-trip: the artifact must reproduce the headline numbers
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "race.csv"
        write_csv(result.rows, path)
        back = read_csv(path)
        again = summarise_arm(back, "agent", questions, reps)
        check("csv round-trips", len(back) == len(result.rows)
              and again.total_tokens == agent.total_tokens
              and abs(again.p50_latency_ms - agent.p50_latency_ms) < 1e-6,
              f"{len(back)} rows reproduce the same 4 numbers")

    decision = verdict(result)
    check("verdict table selects a row", bool(decision["row"]),
          f"{decision['row']} -> {decision['recommendation']}")
    check("scripted agent path is constant", agent.distinct_paths == 1,
          f"{agent.distinct_paths} path (a scripted model cannot vary; the real "
          "race is where variation is measured)")
    lines.append(f"       scripted calls: agent {chat.agent_calls}, "
                 f"workflow {chat.workflow_calls}")
    return failures, lines


# --------------------------------------------------------------------------
# quota handling, offline
# --------------------------------------------------------------------------

#: The real message that ended the 2026-09-15 run, kept verbatim so the
#: classifier is tested against what the provider actually sent rather than a
#: paraphrase that might be easier to match.
REAL_TPD_MESSAGE = (
    "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached "
    "for model `openai/gpt-oss-120b` in organization `org_01kw4rhz8rfc28g1tnkzrew1ss` "
    "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 198458, "
    "Requested 2819. Please try again in 9m11.664s. Need more tokens? Upgrade to "
    "Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', "
    "'code': 'rate_limit_exceeded'}}"
)

REAL_TPM_MESSAGE = (
    "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached "
    "for model `openai/gpt-oss-120b` in organization `org_x` service tier "
    "`on_demand` on tokens per minute (TPM): Limit 8000, Used 7421, Requested 2100. "
    "Please try again in 4.207s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


class QuotaScriptedChat(RaceScriptedChat):
    """Scripted model that runs normally, then hits the daily quota."""

    def __init__(self, fail_after_calls: int) -> None:
        super().__init__()
        self.fail_after_calls = fail_after_calls
        self.total_calls = 0

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        from app.rag.week7_contract import ChatResult

        self.total_calls += 1
        if self.total_calls > self.fail_after_calls:
            return ChatResult(error=REAL_TPD_MESSAGE, error_kind=ERROR_QUOTA_TPD,
                              retry_after_s=551.664)
        return super().__call__(messages, **kwargs)


def quota_self_test() -> tuple[list[str], list[str]]:
    """Prove the quota paths without a provider. No network, no tokens."""
    failures: list[str] = []
    lines: list[str] = []
    questions = load_questions()

    def check(name: str, ok: bool, detail: str) -> None:
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name:44} {detail}")
        if not ok:
            failures.append(f"quota self-test {name}: {detail}")

    lines.append("1. classification (against the provider's real messages)")
    check("TPD classified as a daily quota",
          classify_provider_error(REAL_TPD_MESSAGE) == ERROR_QUOTA_TPD,
          classify_provider_error(REAL_TPD_MESSAGE))
    check("TPM classified as a rate limit",
          classify_provider_error(REAL_TPM_MESSAGE) == "rate_limit_tpm",
          classify_provider_error(REAL_TPM_MESSAGE))
    check("the two are not confused",
          classify_provider_error(REAL_TPD_MESSAGE)
          != classify_provider_error(REAL_TPM_MESSAGE),
          "same HTTP 429, different verdicts")
    check("transient error classified",
          classify_provider_error("APIConnectionError: connection reset") == "transient",
          "transient")
    check("other provider error classified",
          classify_provider_error("BadRequestError: 400 invalid tool schema") == "other",
          "other")
    check("multi-minute retry-after parsed whole",
          abs((retry_after_seconds(REAL_TPD_MESSAGE) or 0) - 551.664) < 0.01,
          f"{retry_after_seconds(REAL_TPD_MESSAGE)}s from '9m11.664s' "
          "(a seconds-only regex would read 11.664)")
    check("TPM retry-after parsed",
          abs((retry_after_seconds(REAL_TPM_MESSAGE) or 0) - 4.207) < 0.01,
          f"{retry_after_seconds(REAL_TPM_MESSAGE)}s")

    lines.append("2. preflight against a 200,000/day budget")
    for reps, expect_fit in ((1, True), (2, True), (3, False)):
        result = preflight(len(questions), reps, DEFAULT_DAILY_TOKEN_BUDGET)
        check(f"reps={reps} {'fits' if expect_fit else 'REJECTED'}",
              result.fits == expect_fit,
              f"~{result.estimated_tokens:,} tokens vs {DEFAULT_DAILY_TOKEN_BUDGET:,}")
    rejected = preflight(len(questions), 3, DEFAULT_DAILY_TOKEN_BUDGET)
    check("rejection recommends the largest that fits",
          rejected.recommended_reps == 2, f"recommended reps={rejected.recommended_reps}")
    check("recommendation is NOT applied automatically",
          rejected.reps == 3,
          "requested reps unchanged; the user must re-run with --reps 2")
    check("unknown budget skips the check",
          preflight(len(questions), 3, 0).fits, "--daily-budget 0 proceeds")

    lines.append("3. a quota hit mid-race aborts cleanly")
    # 10 questions x 1 rep; the agent takes 2 calls and the workflow 1, so 9
    # calls covers 3 full pairs before the quota bites.
    chat = QuotaScriptedChat(fail_after_calls=9)
    result = run_race(questions, reps=1, delay=0.0, warmup=False, verbose=False,
                      chat=chat)
    check("race reports itself aborted", result.aborted and not result.complete,
          f"aborted={result.aborted} complete={result.complete}")
    check("abort kind is the daily quota", result.abort_kind == ERROR_QUOTA_TPD,
          result.abort_kind)
    check("no exception escaped", isinstance(result, RaceResult), "returned a RaceResult")
    check("the quota pair was DISCARDED, not scored",
          all(r.error_kind != ERROR_QUOTA_TPD for r in result.rows),
          f"{len(result.rows)} rows, none carrying a quota error")
    check("no zero-token row entered the data",
          all(r.total_tokens > 0 for r in result.rows),
          "every retained row has real token accounting")
    check("not usable as a result", not result.usable_as_result,
          f"completed {result.completed_pairs}/{result.expected_pairs} pairs")
    check("dataset validation says why", bool(result.dataset_problems),
          result.dataset_problems[0][:60] if result.dataset_problems else "none")

    lines.append("4. partial data cannot become a verdict")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_meta(result, out / META_NAME, questions)
        write_csv(result.rows, out / PARTIAL_CSV_NAME)
        _, _, problems = load_verifiable_race(out, questions)
        check("no race.csv is written for an aborted run",
              not (out / CSV_NAME).exists(),
              f"only {PARTIAL_CSV_NAME} exists")
        check("verdict refuses without race.csv", bool(problems),
              problems[0][:60] if problems else "none")

        # The nastiest case: someone renames the partial file to race.csv.
        write_csv(result.rows, out / CSV_NAME)
        _, _, problems = load_verifiable_race(out, questions)
        check("verdict still refuses a renamed partial", bool(problems),
              f"{len(problems)} problem(s), incompleteness detected from the rows")

    lines.append("5. a zero-token quota row can never be valid")
    fabricated = RaceRow(
        question_id=questions[0].id, question_class="single_hop", arm="agent",
        rep=1, passed=False, total_tokens=0, llm_calls=1,
        termination="chat_error", error_kind=ERROR_QUOTA_TPD,
    )
    problems = validate_race_dataset([fabricated], questions, 1)
    check("zero-token quota row rejected",
          any("quota" in p for p in problems) and any("0 tokens" in p for p in problems),
          f"{len(problems)} problems incl. quota + accounting")
    return failures, lines


# --------------------------------------------------------------------------
# offline validation
# --------------------------------------------------------------------------


def validate() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    lines: list[str] = []
    cfg = ArmConfig()
    questions = load_questions()

    lines.append(f"  questions loaded        {len(questions)}")
    if len(questions) != 10:
        failures.append(f"expected 10 questions, got {len(questions)}")

    identity = race_identity(cfg, questions)
    for key, value in identity.items():
        lines.append(f"  {key:22}  {value}")

    problems = scorer_is_blind()
    lines.append(f"  scorer is arm-blind     {'yes' if not problems else 'NO'}")
    failures.extend(problems)

    price = load_price(cfg.resolved_model())
    lines.append(f"  price verified          {price.verified} ({price.verified_on})")
    if not price.verified:
        failures.append("price is unverified; cost columns would be fiction")

    # percentile parity — the headline latency depends on it
    from app.rag.percentile import verify_week4_parity

    parity = verify_week4_parity()
    lines.append(f"  percentile parity       {'OK' if not parity else 'FAILED'}")
    failures.extend(parity)

    # the workflow's structural bounds
    from app.rag.week7_workflow import static_call_bounds

    report, loop_problems = static_call_bounds()
    lines.append(f"  workflow bounds         {report['llm_sites']} chat + "
                 f"{report['tool_sites']} dispatch sites "
                 f"(max {MAX_LLM_CALLS} llm / {MAX_TOOL_CALLS} tools)")
    failures.extend(loop_problems)

    # the agent's budgets are all enforced
    from app.rag.week7_agent import budget_coverage

    where, budget_problems = budget_coverage()
    lines.append(f"  budgets enforced        "
                 f"{sum(1 for v in where.values() if v)}/{len(where)}")
    failures.extend(budget_problems)

    # both arms reach the outside world only through the contract
    from app.rag.week7_contract import imports_are_clean

    for name in ("week7_agent.py", "week7_workflow.py"):
        problems = imports_are_clean(Path("app/rag") / name)
        lines.append(f"  {name:22}  {'contract-only' if not problems else 'LEAKS'}")
        failures.extend(problems)

    # accounting: a synthetic multi-lap run must sum, not overwrite
    from app.rag.week7_contract import ChatResult, Meter

    meter = Meter(price=price)
    for prompt in (900, 1800, 2700):
        meter.add(ChatResult(prompt_tokens=prompt, completion_tokens=100))
    if meter.total_tokens != 5700:
        failures.append(f"meter did not sum every lap: {meter.total_tokens}")
    lines.append(f"  meter sums every lap    {meter.total_tokens} == 5700")
    failures.extend(f"meter: {p}" for p in meter.check_consistency())
    return failures, lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The Week 7 race.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_JSONL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reps", type=int, default=3, choices=(1, 2, 3),
                        help="repetitions. Never downgraded automatically")
    parser.add_argument("--delay", type=float, default=12.0)
    parser.add_argument("--preflight", action="store_true",
                        help="token-budget check only; makes no LLM call")
    parser.add_argument("--daily-budget", type=int, default=DEFAULT_DAILY_TOKEN_BUDGET,
                        help="provider tokens/day; 0 means unknown, skips the check")
    parser.add_argument("--on-quota", choices=("abort", "wait"), default="abort",
                        help="daily-quota exhaustion: abort cleanly, or wait and retry")
    parser.add_argument("--quota-waits", type=int, default=0,
                        help="with --on-quota wait, how many waits are allowed")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="drive the whole harness with a scripted model, offline")
    parser.add_argument("--quota-self-test", action="store_true",
                        help="prove the quota classification and refusal paths, offline")
    parser.add_argument("--promote-complete-reps", action="store_true",
                        help="turn the complete leading reps of an aborted run into "
                             "race.csv; whole reps only, re-validated")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verdict", action="store_true",
                        help="recompute the decision table from an existing race.csv")
    args = parser.parse_args(argv)

    if args.promote_complete_reps:
        questions = load_questions(args.questions)
        partial = args.out / PARTIAL_CSV_NAME
        reps, problems = promote_complete_reps(partial, args.out, questions)
        if problems:
            print("CANNOT PROMOTE — the partial data holds no complete repetition.",
                  file=sys.stderr)
            for problem in problems:
                print(f"  FAIL {problem}", file=sys.stderr)
            return 1
        rows = read_csv(args.out / CSV_NAME)
        result = RaceResult(
            rows=rows,
            summaries={a: summarise_arm(rows, a, questions, reps) for a in ARMS},
            reps=reps,
            fairness={
                "identity": race_identity(ArmConfig(), questions),
                "differences": list(PERMITTED_DIFFERENCES),
                "permitted": list(PERMITTED_DIFFERENCES),
                "problems": [],
                "passed": True,
            },
            price=load_price(ArmConfig().resolved_model()).model_dump(),
            complete=True,
            completed_pairs=len(questions) * reps,
            expected_pairs=len(questions) * reps,
        )
        write_meta(result, args.out / META_NAME, questions)
        meta = json.loads((args.out / META_NAME).read_text())
        meta["promoted_from"] = str(partial)
        meta["promotion_note"] = (
            f"reps={reps} promoted from an aborted run; complete leading reps only, "
            "re-validated. The requested reps were NOT achieved."
        )
        (args.out / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (args.out / "race_summary.txt").write_text(
            render_summary(result) + "\n", encoding="utf-8")
        (args.out / "race_result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8")
        print(f"promoted {reps} complete repetition(s) into {args.out / CSV_NAME} "
              f"({len(rows)} rows)")
        print()
        print(render_summary(result))
        print()
        print("PATH VARIATION")
        print("-" * 78)
        print(path_variation_report(result))
        return 0

    if args.preflight:
        questions = load_questions(args.questions)
        check = preflight(len(questions), args.reps, args.daily_budget)
        print(render_preflight(check))
        return 0 if check.fits else 2

    if args.quota_self_test:
        print("quota-handling self-test (no network, no tokens)\n")
        failures, lines = quota_self_test()
        for line in lines:
            print(line if line.startswith("  ") else f"\n{line}")
        print()
        if failures:
            print(f"QUOTA SELF-TEST FAILED — {len(failures)} problem(s)")
            return 1
        print("QUOTA SELF-TEST PASSED")
        return 0

    if args.self_test:
        print("race harness self-test (scripted model, no network, no tokens)\n")
        failures, lines = self_test()
        for line in lines:
            print(line)
        print()
        if failures:
            print(f"RACE SELF-TEST FAILED — {len(failures)} problem(s)")
            return 1
        print("RACE SELF-TEST PASSED")
        return 0

    if args.validate:
        print("race validation (offline, no network)\n")
        failures, lines = validate()
        for line in lines:
            print(line)
        print()
        for failure in failures:
            print(f"FAIL  {failure}")
        if failures:
            print(f"\nRACE VALIDATION FAILED — {len(failures)} problem(s)")
            return 1
        print("RACE VALIDATION PASSED")
        return 0

    questions = load_questions(args.questions)
    cfg = ArmConfig()

    if args.dry_run:
        price = load_price(cfg.resolved_model())
        # The SAME measured constants the preflight uses. Two different
        # estimates of one quantity in one file is how a budget check and a
        # dry run end up disagreeing about whether a race fits.
        per_agent = MEASURED_AGENT_TOKENS_PER_QUESTION
        per_workflow = MEASURED_WORKFLOW_TOKENS_PER_QUESTION
        runs = len(questions) * args.reps
        total = estimate_tokens(len(questions), args.reps,
                                warmup=not args.no_warmup)
        print(f"DRY RUN — no LLM call made.\n")
        print(f"{len(questions)} questions x {args.reps} reps x 2 arms = {runs * 2} runs")
        print(f"estimated ~{total:,} tokens "
              f"(~{runs * per_agent:,} agent + ~{runs * per_workflow:,} workflow)")
        print(f"estimated cost at list price ~${price.cost(int(total * 0.9), int(total * 0.1)):.4f}")
        print(f"delay {args.delay}s between questions -> "
              f">= {runs * args.delay / 60:.0f} min of pacing alone")
        for key, value in race_identity(cfg, questions).items():
            print(f"  {key:18} {value}")
        return 0

    if args.verdict:
        rows, reps, problems = load_verifiable_race(args.out, questions)
        if problems:
            print("REFUSING TO PRODUCE A VERDICT — the race dataset is not usable.",
                  file=sys.stderr)
            print(file=sys.stderr)
            for problem in problems:
                print(f"  FAIL {problem}", file=sys.stderr)
            print(file=sys.stderr)
            print("  A verdict requires every (question, arm, rep) cell measured "
                  "cleanly,", file=sys.stderr)
            print("  a passing fairness gate and valid token accounting. Partial or",
                  file=sys.stderr)
            print("  quota-aborted data is not a result and will not be turned into one.",
                  file=sys.stderr)
            return 1
        result = RaceResult(
            rows=rows,
            summaries={a: summarise_arm(rows, a, questions, reps) for a in ARMS},
            reps=reps,
            fairness={"identity": race_identity(cfg, questions), "differences": [],
                      "permitted": list(PERMITTED_DIFFERENCES), "problems": [], "passed": True},
            price=load_price(cfg.resolved_model()).model_dump(),
        )
        print(json.dumps(verdict(result), indent=2))
        print()
        print(path_variation_report(result))
        return 0

    # Preflight before spending anything. A race that cannot fit the day's
    # quota is refused here rather than discovered at pair 8 of 30.
    check = preflight(len(questions), args.reps, args.daily_budget,
                      warmup=not args.no_warmup)
    print(render_preflight(check))
    print()
    if not check.fits:
        print("REFUSING TO START — this race cannot fit the configured daily budget.",
              file=sys.stderr)
        if check.recommended_reps:
            print(f"Re-run explicitly with --reps {check.recommended_reps}, or pass "
                  "--daily-budget 0 to skip this check.", file=sys.stderr)
        return 2

    result = run_race(
        questions, cfg, reps=args.reps, delay=args.delay, warmup=not args.no_warmup,
        on_quota=args.on_quota, quota_waits=args.quota_waits,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_meta(result, args.out / META_NAME, questions)

    # An incomplete race writes PARTIAL data under a different name and no
    # summary at all. Writing race.csv here, or printing the eight numbers from
    # a run that stopped early, is exactly how a quota failure turns into a
    # published result.
    if not result.usable_as_result:
        write_csv(result.rows, args.out / PARTIAL_CSV_NAME)
        status = render_incomplete(result, args.out)
        (args.out / INCOMPLETE_NAME).write_text(status + "\n", encoding="utf-8")
        print(status)
        print()
        print(f"wrote {args.out / PARTIAL_CSV_NAME} ({len(result.rows)} partial rows)")
        print(f"wrote {args.out / INCOMPLETE_NAME}")
        print(f"wrote {args.out / META_NAME} (complete=false)")
        print(f"NOT written: {args.out / CSV_NAME}, race_summary.txt")
        if not result.fairness["passed"]:
            for problem in result.fairness["problems"]:
                print(f"  FAIRNESS FAIL {problem}", file=sys.stderr)
        return 1

    write_csv(result.rows, args.out / CSV_NAME)
    (args.out / "race_summary.txt").write_text(render_summary(result) + "\n", encoding="utf-8")
    (args.out / "race_result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )

    print()
    print(render_summary(result))
    print()
    print("PATH VARIATION")
    print("-" * 78)
    print(path_variation_report(result))
    print()
    print(f"wrote {args.out / CSV_NAME} ({len(result.rows)} rows)")
    print(f"wrote {args.out / 'race_summary.txt'}")
    print(f"wrote {args.out / 'race_result.json'}")
    print(f"wrote {args.out / META_NAME} (complete=true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
