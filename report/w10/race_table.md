# Week 10 race: single agent vs docs squad

**Status: COMPLETE.** Cases: the 10 Week-6 cases in `eval/week10_selection.yaml`, unchanged, rule-selected before any run. Model `openai/gpt-oss-120b`, temperature 0. Judge: Week 6 Judge v2 plus the Week 6 assertions, identical for both arms.

| metric | single agent | squad (orchestrator + 2 workers) |
|---|---|---|
| pass rate | 10/10 (100%) | 7/10 (70%) |
| p50 latency | 4.4 s | 9.7 s |
| p99 latency | 38.0 s | 51.1 s |
| total tokens | 78,201 | 119,243 |
| cost per question | $0.001188 | $0.002138 |
| LLM calls (all cases) | 29 | 66 |

**Context re-send multiplier: 119,243 squad tokens / 78,201 single-agent tokens = 1.5x. Largest share: "orchestrator -> version/deprecation worker", 63% of all squad tokens (75,635 tokens over 10 hand-offs).**

## Per case

| case | question | single | squad | single tokens | squad tokens | single s | squad s |
|---|---|---|---|---|---|---|---|
| w6-001 | what does the client do when the endpoint returns 5xx | PASS | FAIL | 12,319 | 14,166 | 25.1 | 51.1 |
| w6-002 | is chunk_size required | PASS | PASS | 7,686 | 9,287 | 32.0 | 33.3 |
| w6-007 | can I call close twice | PASS | PASS | 7,297 | 13,058 | 38.0 | 50.1 |
| w6-014 | what type is telemetry | PASS | FAIL | 8,114 | 13,421 | 16.0 | 33.6 |
| w6-015 | can I disable telemetry for GDPR compliance | PASS | PASS | 13,328 | 9,397 | 23.1 | 16.4 |
| w6-016 | What is the default value of retry_backoff_ms on Client.send() in v3? | PASS | PASS | 4,664 | 14,101 | 2.5 | 9.5 |
| w6-019 | did chunk_size change between v2 and v3 | PASS | PASS | 7,596 | 9,217 | 4.4 | 6.3 |
| w6-026 | what type is concurrency | PASS | FAIL | 7,653 | 13,166 | 4.0 | 7.8 |
| w6-029 | What type is the max_redirects parameter on Client.send()? | PASS | PASS | 7,800 | 9,377 | 3.3 | 6.5 |
| w6-031 | What is the rate limit on the /v1/embeddings endpoint? | PASS | PASS | 1,744 | 14,053 | 0.9 | 9.7 |

## How each number was computed

- **pass_rate**: cases passed / 10, by PASS_RULE
- **latency**: wall clock per question, minus time spent sleeping on provider 429s (Week 7 rule, both arms alike); p50 and p99 by nearest rank (app/rag/percentile.py). With 10 samples p99 is the slowest run.
- **total_tokens**: prompt + completion tokens of every LLM call the arm made, summed over the 10 cases: for the squad that is the orchestrator plus both workers. Judge tokens are evaluation overhead, not either arm's cost, and are excluded from both.
- **cost_per_question**: list price from eval/week7_prices.yaml (openai/gpt-oss-120b, cached input at its own rate) / 10. The free tier bills nothing; this is what the tokens would cost.
- **multiplier**: squad total tokens / single-agent total tokens, one decimal place
- **pass rule**: A case passes when Judge v2 (app/rag/week6_judge.py, the Week 6 prompt and model, temperature 0) returns PASS and no Week 6 assertion (app/rag/week6_assertions.py) returns fail; n/a and skipped do not fail. Both instruments see the arm's ANSWER text only, with the citations the arm captured and, as context, every chunk or tool result its tools returned during the run. The VERSION and CITATIONS lines are metadata and are withheld, because the VERSION line would satisfy the version_stated assertion on every answer. An answer with no parsable ANSWER line fails, as in Week 7.

Ten cases give 10-point granularity on pass rate, and each arm ran once (see report/w10/prediction.txt for why a 1-case difference is treated as noise).
