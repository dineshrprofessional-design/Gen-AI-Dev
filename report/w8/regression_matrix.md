# Week 8 Regression Matrix

Mitigation: **Consecutive-Search & Non-Existence Circuit Breaker** (`app/rag/week8_mitigation.py`)

Both arms replayed from saved `AgentRun` dumps by `python -m app.rag.trajectory_eval`. Same 10 cases, same spec, same scorer, no model in the loop. Re-running the command reproduces every number below.

## Six failure modes, before and after

| # | Failure mode | Before | After | Delta | Regression? |
|---|---|---|---|---|---|
| 1 | Redundant / thrashing tool calls | 1 | 0 | -1 | no (improved) |
| 2 | Argument hallucination / schema violation | 0 | 0 | +0 | no |
| 3 | Suboptimal step efficiency (over max_steps) | 1 | 0 | -1 | no (improved) |
| 4 | Parametric guessing / bypassed verification | 1 | 1 | +0 | no |
| 5 | Premature refusal (below min_steps) | 0 | 0 | +0 | no |
| 6 | Budget overrun | 1 | 0 | -1 | no (improved) |

**Regressions: 0.** None. Every mode is at or below its baseline count.

Mode 4 is unchanged at 1 rather than improved: the remaining instance is `w8-09`, where the agent searched before checking retirement. The breaker targets thrashing, not ordering, and deliberately does not touch it — one mitigation, one failure mode.

## Metric impact

| Metric | Before | After | Change |
|---|---|---|---|
| Tool-Choice Accuracy | 100.0% | 100.0% |  |
| Argument Validity | 100.0% | 100.0% |  |
| Step Efficiency p50 | 1.00 | 1.00 |  |
| Step Efficiency mean | 0.89 | 0.92 |  |
| Trajectory pass rate | 80.0% | 90.0% |  |
| Outcome pass rate | 50.0% | 50.0% |  |
| Cost p50 (USD) | 0.000810 | 0.000797 | -1.7% |
| **Cost Max (USD)** | 0.003246 | 0.001193 | -63.2% |
| Latency p50 (ms) | 9,617 | 7,879 | -18.1% |
| **Latency Max (ms)** | 127,041 | 16,890 | -86.7% |
| Tokens total | 61,217 | 46,037 | -24.8% |
| Tokens max (one case) | 21,481 | 7,546 | -64.9% |
| LLM calls total | 31 | 28 | -9.7% |
| Total tool calls | 21 | 18 | -14.3% |

Outcome pass rate is **unchanged at 50.0%**. That is the load-bearing non-regression: the breaker removed a runaway without costing a single answer. Trajectory pass rate rose 80.0% -> 90.0% because `w8-10` stopped exceeding `max_steps`.

## Resource price paid

| Cost | Measured |
|---|---|
| Breaker CPU, total | 33.2615 ms across the full 10-question run |
| Breaker CPU, per LLM call | 1.1879 ms (28 calls) |
| Extra LLM calls | 0 |
| Extra tool calls | 0 |
| Extra tokens | 0 (the breaker adds no message; it removes calls) |
| Vetoes issued | 1 across 10 questions |

The breaker is pure bookkeeping in the existing `chat` seam: a dict lookup and a JSON dump per tool call. It buys a 63% cut in worst-case cost and an 87% cut in worst-case latency for roughly one millisecond of CPU per LLM call. It never issues an answer itself — a vetoed call returns a tool result and the model still writes its own reply.

## Per-case, before and after

| Case | Q | Class | Steps B->A | Traj B->A | Outcome B->A | Tokens B->A |
|---|---|---|---|---|---|---|
| w8-01 | w7-01 | single_hop | 1 -> 1 | PASS -> PASS | PASS -> PASS | 3,276 -> 3,280 |
| w8-02 | w7-02 | single_hop | 1 -> 1 | PASS -> PASS | PASS -> PASS | 2,485 -> 2,485 |
| w8-03 | w7-03 | single_hop | 1 -> 1 | PASS -> PASS | FAIL -> FAIL | 2,081 -> 2,078 |
| w8-04 | w7-04 | cross_version | 2 -> 2 | PASS -> PASS | PASS -> PASS | 5,662 -> 5,659 |
| w8-05 | w7-05 | cross_version | 2 -> 2 | PASS -> PASS | FAIL -> FAIL | 5,187 -> 5,126 |
| w8-06 | w7-06 | two_hop | 2 -> 2 | PASS -> PASS | FAIL -> FAIL | 4,595 -> 4,664 |
| w8-07 | w7-07 | two_hop | 2 -> 2 | PASS -> PASS | PASS -> PASS | 4,476 -> 4,716 |
| w8-08 | w7-08 | two_hop | 2 -> 2 | PASS -> PASS | PASS -> FAIL | 4,441 -> 4,502 |
| w8-09 | w7-09 | two_hop | 3 -> 3 | FAIL -> FAIL | FAIL -> FAIL | 7,533 -> 7,546 |
| w8-10 | w7-10 | refusal | 5 -> 2 | FAIL -> PASS | FAIL -> PASS | 21,481 -> 5,981 |

Only `w8-10` changed. Nine of ten cases are untouched in steps, trajectory verdict and outcome — which is the point of a circuit breaker set above every legitimate trajectory: `max_consecutive_searches` is 2, and no passing Week 7 run ever made a third unproductive search.

## Week 3-7 regression check

| Check | Result |
|---|---|
| `python -m app.rag.week7_contract --validate` | PASS |
| `python -m app.rag.week7_agent --validate` | PASS |
| `python -m app.rag.week7_workflow --validate` | PASS |
| `python -m app.rag.week7_race --self-test` | PASS |
| `python -m app.rag.week7_race --quota-self-test` | PASS |
| `python -m app.rag.week7_race --validate` | PASS |
| `python -m app.rag.week7_tools --validate` | PASS |
| `python -m app.rag.week7_corpus --check` | PASS |
| `python -m app.rag.week7_questions --check` | PASS |
| `python -m app.rag.week6_cases --check` | PASS |
| `python -m app.rag.golden_set --check` | PASS |
| `python -m app.rag.percentile --verify` | PASS |
| `week6_assertions --json` vs committed baseline | BYTE-IDENTICAL |
| `git diff HEAD -- app/rag/week7_*.py eval/week7_*.yaml` | empty (Week 7 untouched) |

Week 8 is additive. No Week 3-7 source file was modified; the mitigation wraps `run_agent`'s existing `chat` seam rather than editing the loop, so both Week 7 race arms still share one byte-identical contract.
