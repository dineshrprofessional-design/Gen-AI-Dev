# Failure taxonomy — Week 5, from 20 seeded-random traces

Sample: seed `20260831`, 20 of 141 organic traces (curated demo questions excluded).
A trace can exhibit more than one mode, so counts sum past 16 failing traces.
4 of 20 traces were correct with clean citations (incl. 1 correct refusal).

| # | Failure mode | Count | % of 20 | Severity | Example trace_id |
|---|---|---|---|---|---|
| 1 | Correct answer, but the citation is written in 【】 brackets, `[ spaced ]` brackets, or with zero-width characters — so it is recorded as *no citation* or flagged *invalid* | 10 | 50% | merely annoys — the answer is right; cite links break and correct answers get "invalid citation" warnings | `bcc84d1515d9` |
| 2 | Version-less question answered from one version's page, without saying another version gives a different value | 3 | 15% | **ships broken code** — a v2 default pasted into a v3 project is a wrong number that runs | `d84a2cd4f913` |
| 3 | The score floor refuses although the table containing the answer was retrieved — the gate reads only the rank-1 cosine, which fusion can legally place below a higher-scoring rank-2 | 2 | 10% | merely annoys — a false "I don't know" | `aa983b01fd36` |
| 4 | "Did X change between v2 and v3" retrieves Notes chunks from pages that don't define X, never fetches the two tables to compare, and refuses | 2 | 10% | merely annoys | `179023eb0b6a` |
| 5 | Confident answer about a *different parameter* than the one asked, when the asked one was never retrieved (`fail fast` → answered about `force`) | 1 | 5% | **ships broken code** — true statements about the wrong knob | `7a7abc208979` |
| 6 | Example code synthesised by the model that appears in no retrieved chunk (a `proxy=` call the docs never show) | 1 | 5% | **ships broken code** — plausible, unverified usage | `cbaa3d073d15` |

Fix order by frequency × severity: mode 1 first (half of all traces), then mode 2
(rarer but ships wrong numbers). Modes 3–4 are retrieval/gate seams; 5–6 are
generation discipline. The dated prediction targeting mode 1 is committed —
see notes.md.
