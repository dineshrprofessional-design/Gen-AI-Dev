# Week 10: injected worker failure

**In one line: it degraded, but to no answer rather than a partial one. After one
HTTP 500 the orchestrator neither retried nor lied. It refused the whole question
(`INSUFFICIENT_CONTEXT`) without ever calling the code-sample worker, which could
still have answered it from the docs.**

## What was injected

| | |
|---|---|
| case | `w6-019`, "did chunk_size change between v2 and v3" (pre-declared in `eval/week10_selection.yaml`, commit `cb9efad`) |
| failure | the version/deprecation worker returns **HTTP 500** on every attempt for this case; body `{"detail": "Internal Server Error"}`, FastAPI's own, so nothing marks it as injected |
| where | `FailurePlan` in `app/rag/week10_squad.py`; the HTTP 500 is real, served by the worker app over the in-process HTTP client |
| retry policy | none in code. Any retry would have to be the orchestrator model's own decision. |
| run | `report/w10/runs/failure/w6-019.squad.json` (`python -m app.rag.week10_race --failure`) |

## What the orchestrator actually did

| hop | edge | HTTP | tokens | what happened |
|---|---|---|---|---|
| 1 | question → orchestrator | | 712 | called `delegate_version_check` with the question |
| 2 | orchestrator → version/deprecation worker | **500** | 0 | the tool result was `{"error": "HTTP 500 from the version/deprecation worker: Internal Server Error"}` |
| 3 | version/deprecation worker → orchestrator | | 827 | final answer: `ANSWER: INSUFFICIENT_CONTEXT`, `VERSION: none`, `CITATIONS: none` |

- **Retried?** No. The worker was asked once (`http_500_attempts: 1`).
- **Partial answer?** No. It never delegated to the code-sample worker, although the
  question names both versions and that worker searches `search_docs` itself.
- **Lied?** No. It claimed nothing about either version.
- Cost: 1,539 tokens and 1.7 s, against 9,217 tokens for the clean squad run.

What a correct answer looks like, from the clean race on the same case: both arms
passed. The squad answered "the default `chunk_size` was 512 in v2 and was changed
to 1024 in v3".

## Two caveats

1. **The grader scores this run PASS, and that is misleading.** Judge v2 checks a
   refusal against the context the assistant was given. The context here was empty,
   so a refusal is "correct". From the developer's side, an answerable question went
   unanswered. The headline race uses only the clean runs; this run is reported
   here and is not in the race table.
2. **The race shows the same failure shape without any injection.** On `w6-026`
   ("what type is concurrency"), the version worker ran 4 searches, used up its
   4-step limit, and returned an empty report (`termination: max_laps`). The
   orchestrator again answered `INSUFFICIENT_CONTEXT` and never called the
   code-sample worker. The single agent answered `int` from two searches.
   **The version worker is a single point of failure for the whole answer.**

## A2A view (bonus)

See `a2a.md`: this case should end as an A2A `failed` task, not a completed answer
that says "insufficient context".
