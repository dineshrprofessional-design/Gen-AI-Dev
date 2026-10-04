Sunk cost, named first: we built an orchestrator, two workers, an HTTP layer and a 20-run race, and that effort is what makes "keep it" tempting. It is not evidence.
Rule fixed before any run (report/w10/prediction.txt): KEEP only if the squad passes at least 2 more of the 10 Week-6 cases than the single agent.
Pass rate: single agent 10/10, squad 7/10. The squad lost by 3.
Even forgiving its two one-line-format failures (w6-001, w6-014), the squad scores 9/10. Still not better.
Tokens: squad 119,243 vs single 78,201 (1.5x); cost per question $0.002138 vs $0.001188 (1.8x).
Latency: p50 9.7 s vs 4.4 s; p99 51.1 s vs 38.0 s.
63% of squad tokens went to the orchestrator -> version/deprecation worker hand-off, mostly on questions where the version was never in doubt.
One HTTP 500 from that worker and the squad answered nothing: no retry, no fallback to the code-sample worker.
VERDICT: KILL. On the same 10 cases the squad is slower, costs more, and answers fewer questions.
Each run was recorded once and never re-run to rescue it (report/w10/race.log). "We tried it and it lost" is the result.
