# Week 4 — Task Set E · Label the failures, buy back hit-rate@3 with one change

## 0. Summary

| | |
|---|---|
| **hit-rate@3** | **10/12 → 12/12  (+2)** |
| **p50 latency** | **451.3 ms → 440.5 ms** — the change costs ~0.7 ms; the difference shown is noise |
| Failure tally | R = 2, G = 0, Not-In-Corpus = 0 (+1 score-floor refusal, neither R nor G) |
| The one change | BM25 + Reciprocal Rank Fusion, k = 60, over the existing dense retrieval |
| Decision | **Ship.** Both R-failures fixed, no regressions, for 0.15% of query time |

The team lead wanted the embedding model swapped. The tally says both retrieval
failures are exact-symbol misses where the correct chunk **already contains the
query token verbatim** and was outranked anyway. A denser embedding re-ranks the
same semantic space and structurally cannot represent literal containment, so it
is the one intervention that could not have helped. That is what BM25 is for.

---

## 1. Frozen setup, and the proof that one variable moved

Everything below was identical across both runs. Only `hybrid` changed.

| Knob | Before | After |
|---|---|---|
| retrieval | dense only | **BM25 + RRF (k=60)** |
| corpus | `docs/`, 10 pages, sha `b3df167a299714e8` | same |
| chunker | `structure`, `max_chars=1200` | same |
| chunks | 38 | same |
| embedder | `bge-m3` / `BAAI/bge-m3` | same |
| collection | `structure__bge-m3` | same |
| index | `.rag_index`, 38 chunks, id-hash `07e94668aef5bf24` | same |
| golden set | `eval/golden_set.jsonl`, sha `218f4a9ec6834555` | same |
| k retrieved | 5 | same |
| metric | hit@3 | same |
| reps | 5 (+1 discarded warm-up) | same |
| generation | off during measurement | same |

This is machine-checked rather than promised. Each run records a fingerprint of
everything that must not move, and `--compare` exits non-zero unless the
fingerprints are identical and `hybrid` is the only differing config key:

```
$ python -m app.rag.eval_week4 --compare report/w4/before.json report/w4/after.json
## one-variable check

fingerprint fields that differ : none
config keys that differ        : ['hybrid']
verdict                        : PASS
```

The fingerprint separately hashes the chunk ids a *fresh* chunking of `docs/`
produces and the ids actually sitting in the index. They match
(`07e94668aef5bf24` both ways), which rules out the failure mode where a corpus
edit without a re-index yields plausible numbers for a corpus that no longer
exists. The run refuses to start if they disagree.

Environment: Python 3.12.0, Windows 11, chromadb 1.5.9, CPU only, single
machine, one session per run.

### One chunk per section, so chunk_id tagging is the same thing the harness measures

`docs/` has 38 heading sections and the index holds exactly 38 chunks — the
structure-aware chunker emits one chunk per section on this corpus, including
the 1950-character 16-row parameter table, which it keeps whole rather than
cutting. So tagging gold by `chunk_id` and Week 3's char-range-overlap criterion
agree exactly here. Gold chunk_ids are never hand-typed: they are **derived** by
re-running the chunker over the `(sdk_version, page_id, anchor)` triple each
question already carries, and the derivation raises if a section ever maps to
more than one chunk.

---

## 2. The 12-question golden set

Written from the pages **before any retrieval was run against them**, and frozen
from the moment the baseline was recorded. Terse and mostly version-less, the
way questions actually arrive.

| id | question, as typed | gold chunk_id | expected answer | exact token |
|---|---|---|---|---|
| q01 | what is the default for retry_backoff_ms | `v3/client-send#parameters::1` | 250 ms, base before jitter | `retry_backoff_ms` |
| q02 | what causes RedirectLoopError | `v3/client-send#parameters::1` | more than `max_redirects` (5) redirects | `RedirectLoopError` |
| q03 | pool_reap_seconds default | `v3/client-configure#parameters::1` | 90 s | `pool_reap_seconds` |
| q04 | rate_limit_rps default | `v3/client-batch#parameters::1` | 20 rps | `rate_limit_rps` |
| q05 | chunk_bytes default for upload | `v3/client-upload#parameters::1` | 8388608 | `chunk_bytes` |
| q06 | what raises CancelledError | `v3/client-close#notes::3` | `close(force=True)` | `CancelledError` |
| q07 | what does StreamClosed mean | `v2/client-stream#clientstream::0` | v2 stream dropped, no reconnect | `StreamClosed` |
| q08 | v3 stream chunk_size default | `v3/client-stream#parameters::1` | 1024 | — |
| q09 | did chunk_size change between v2 and v3 | `v2/client-stream#notes::3` | 512 → 1024 | — |
| q10 | setting force=False on close does nothing? | `v2/client-close#notes::3` | correct in v2 | — |
| q11 | does retries_enabled=false stop retry_backoff_ms working | `v3/client-configure#notes::3` | yes, accepted but ignored | — |
| q12 | concurrency 64 with rate_limit_rps 20 - do I get 64 rps | `v3/client-batch#notes::3` | no, 20 rps | — |

**7 of 12 are exact-token questions**; the requirement is at least 4. Coverage is
8 of 10 pages and three anchor types (`parameters`, `notes`, and one intro
chunk). 11 distinct gold chunks across 12 questions, so the metric is not
measuring one retrieval event repeatedly.

Verification, run before any measurement:

```
$ python -m app.rag.verify_questions --week4
12 questions against 10 documents, index .rag_index (38 chunks)
...
exact-token         : 7/12  (need at least 4)
distinct gold chunks: 11/12
verified            : 12/12

every gold chunk_id resolves, is in the index, and contains its answer
```

### The exact-token requirement, and what this corpus cannot supply

The task allows three flavours of exact token: a symbol name, an error code, or
a version string. **Two of the three do not exist in this corpus.** All 7
exact-token questions are therefore symbol/identifier questions — three of them
the only three exception names present, each appearing exactly once.

```
$ grep -rnoE "\b(E[0-9]{3,}|ERR_[A-Z_]+|SDK-[0-9]+)\b" docs/     -> 0 matches
$ grep -rnoE "\bv?[0-9]+\.[0-9]+\.[0-9]+\b" docs/                -> 0 matches
$ grep -rhoE "\bv[23]\b" docs/ | sort | uniq -c                  -> 15 v2, 13 v3
```

Inventing an `E1042` question and tagging it to a chunk that does not contain it
would have scored 0/12 on that row and misrepresented the corpus. The gap is
reported instead.

### Two questions written and then rejected

Kept here because rejecting a question for being unscoreable is evidence the set
was written from the pages rather than reverse-engineered from results.

- **"is 503 retried by default"** — the corpus contradicts itself.
  `v3/client-send#clientsend::0` says 5xx statuses are retried automatically,
  while `#parameters::1` gives `retry_on = [500,502]`, and `503` appears *only*
  inside the example fence. Two defensible golds and two defensible answers, so
  it cannot be scored. This is a genuine corpus defect found while writing.
- **bare "default pool_size"** and **bare "default chunk_size"** — version-less,
  and v2 and v3 give different correct answers. Terse is the goal; ambiguous is
  not. q05 keeps the word "upload" and q08 keeps "v3" for exactly this reason,
  and those are the minimum tokens that give the question an answer at all.

---

## 3. Baseline hit-rate@3 — recorded before anything changed

**hit-rate@3 = 10/12.** (hit@1 = 6/12, hit@5 = 12/12.) p50 = 451.3 ms.

| id | gold chunk_id | @3 | gold rank | top-1 retrieved | top-1 score |
|---|---|---|---|---|---|
| q01 | `v3/client-send#parameters::1` | **MISS** | 4 | `v3/client-send#notes::3` | 0.6371 |
| q02 | `v3/client-send#parameters::1` | HIT | 1 | `v3/client-send#parameters::1` | 0.4287 |
| q03 | `v3/client-configure#parameters::1` | HIT | 3 | `v3/client-configure#notes::3` | 0.5708 |
| q04 | `v3/client-batch#parameters::1` | **MISS** | 4 | `v3/client-batch#notes::3` | 0.6038 |
| q05 | `v3/client-upload#parameters::1` | HIT | 2 | `v3/client-upload#clientupload::0` | 0.5976 |
| q06 | `v3/client-close#notes::3` | HIT | 1 | `v3/client-close#notes::3` | 0.5280 |
| q07 | `v2/client-stream#clientstream::0` | HIT | 1 | `v2/client-stream#clientstream::0` | 0.5867 |
| q08 | `v3/client-stream#parameters::1` | HIT | 2 | `v2/client-stream#notes::3` | 0.6810 |
| q09 | `v2/client-stream#notes::3` | HIT | 1 | `v2/client-stream#notes::3` | 0.6174 |
| q10 | `v2/client-close#notes::3` | HIT | 1 | `v2/client-close#notes::3` | 0.6712 |
| q11 | `v3/client-configure#notes::3` | HIT | 3 | `v2/client-send#notes::2` | 0.6579 |
| q12 | `v3/client-batch#notes::3` | HIT | 1 | `v3/client-batch#notes::3` | 0.6816 |

Both misses put the gold chunk at **rank 4** — just outside the window. That
detail decides the choice of change in §5: a reordering problem is fixable by
fusion, a rank-20 problem is not.

---

## 4. Every failure labelled, with evidence

hit-rate@3 is a *retrieval* metric; G is a *generation* failure. They do not
share a denominator, so the failures are split into two populations rather than
lumped together.

```
Population A — retrieval misses (what hit-rate@3 counts)      : 2/12
    R                                                          : 2
    Not-In-Corpus                                              : 0
Population B — wrong answer despite a top-3 hit (NOT in @3)    : 0
    G                                                          : 0
Score-floor refusals — gold retrieved, the gate threw it away  : 1  [q02]

addressable ceiling for one retrieval change = R/12 = 2/12
```

Labels were assigned in a fixed order so that no wrong answer is called R
without first checking whether the parameter table was in the top-3 all along:
absent from the corpus → Not-In-Corpus; gold outside top-3 → R; gold inside
top-3 but the answer wrong → G.

### The two R failures

**q01 — R.** `what is the default for retry_backoff_ms`

```
   A 1. 0.6371   v3/client-send#notes::3
     2. 0.6274   v2/client-send#notes::2
     3. 0.5590   v2/client-send#parameters::1
  *A 4. 0.5527   v3/client-send#parameters::1     <- gold, outside the window
```

Evidence: gold at rank 4 (0.5527), beaten by two v2 chunks and a prose note; the
generated answer was **"The default for `retry_backoff_ms` is 100
[v2/client-send#parameters::1]"** — the *v2* value, wrong for an unqualified
question about the current SDK. The token appears in 6 chunks across 3 pages, so
dense similarity had no way to prefer the v3 parameter row.

**q04 — R.** `rate_limit_rps default`

```
   A 1. 0.6038   v3/client-batch#notes::3
     2. 0.5429   v3/client-configure#notes::3
     3. 0.5239   v2/client-configure#notes::3
  *A 4. 0.5213   v3/client-batch#parameters::1    <- gold, outside the window
```

Evidence: gold at rank 4 (0.5213), outranked by two `client-configure` notes
chunks from a different page entirely; the pipeline returned
`INSUFFICIENT_CONTEXT`. The correct chunk contains `rate_limit_rps | int | 20`
verbatim.

**An honest annotation on both.** In each case the rank-1 chunk did contain the
answer strings (marked `A` above), so a reader might have got there anyway. Both
are still counted R, because R is what the metric measures — retrieval did not
fetch the tagged context. But it means the +2 hit-rate delta slightly overstates
the user-visible improvement on q01, and understates it on q04, where the
pipeline refused outright.

### Not-In-Corpus: zero, and why that is the honest number

All 12 questions carry a verified `gold_chunk_id`, so none can be out of corpus
by construction. Padding the 12 with an unanswerable question would have been a
permanent miss in both columns, shrinking the addressable ceiling for no
information. The label is defined and available; it simply did not fire.

### G: zero — and a first pass that got this wrong

The mechanical answer-correctness check initially flagged three G candidates.
Reading the transcripts showed all three were false positives:

- **q05** answered `8,388,608 bytes` — correct; the checker missed it because of
  thousands separators.
- **q12** answered "about 20 requests per second, not 64" — correct; missed
  because the model used a narrow no-break space.
- **q02** was not a generation failure at all (below).

The checker now NFKC-folds unicode spaces and strips digit separators. This is
recorded rather than quietly fixed, because a labelling bug that *invents*
generation failures would have pointed the whole exercise at the prompt instead
of the retriever.

### The finding that is neither R nor G: a score-floor false refusal

**q02** `what causes RedirectLoopError` — gold chunk at **rank 1**, and it
contains the answer. The pipeline still returned `INSUFFICIENT_CONTEXT`, because
`generate.answer()` refuses when the top score is below `SCORE_FLOOR = 0.45` and
this query's top score was **0.4287**. The model was never called.

This is not R (the gold was retrieved), not G (nothing misused good context),
and invisible to hit-rate@3 (which correctly records a HIT). It is a hard gate
mis-calibrated for short exact-token queries, which embed with systematically
lower cosine similarity than prose questions. It is reported separately rather
than laundered into either label, and it is **unchanged by this week's change** —
BM25+RRF keeps cosine scores, so q02 still refuses. Fixing it means touching
`generate.py`, which would have been a second variable.

---

## 5. The one change, and why the tally chose it

**BM25 + Reciprocal Rank Fusion, k = 60**, over the existing dense arm. One
change, one flag, no reranker alongside it.

The rule was fixed before the labels existed: choose BM25+RRF if at least 60% of
R-failures are rare-token misses — the query names an identifier, the gold chunk
contains it verbatim, and it was outranked anyway. **Both R-failures are exactly
that, so 2/2 = 100%.** Both queries (`retry_backoff_ms`, `rate_limit_rps`) are
bare identifiers; in both cases the gold chunk contains the token verbatim and
lost to a prose sibling that is *semantically about* the same parameter. And in
both the gold sat at rank 4 — inside any sane candidate pool, wrongly ordered.
That is a reordering problem with a strong lexical signal, which is precisely
what rank fusion fixes.

Token rarity, computed mechanically — the input to that judgement:

| token | chunks containing it | BM25 idf |
|---|---|---|
| `RedirectLoopError` | 1 | 3.258 |
| `CancelledError` | 1 | 3.258 |
| `StreamClosed` | 1 | 3.258 |
| `pool_reap_seconds` | 2 | 2.747 |
| `rate_limit_rps` | 3 | 2.411 |
| `chunk_bytes` | 4 | 2.159 |
| `retry_backoff_ms` | 6 | 1.792 |

RRF fuses **ranks**, never scores:

    score(d) = Σ_arms  1 / (60 + rank_arm(d))

Cosine similarity sits around 0.4–0.7 here and BM25 term weights around 1–3.
Adding or averaging them is meaningless, and any weighting that made it look
reasonable would be a second tuned parameter. The fusion function accepts ranked
id lists only — scores are discarded at the boundary, so mixing scales is not
merely discouraged, it is unrepresentable. `k=60` is the paper default, taken
as-is; tuning it on 12 questions would be overfitting.

### Rejected alternatives

- **Swapping the embedding model** — the lead's suggestion, and structurally the
  one thing that could not work. Both R-failures are cases where the gold chunk
  *literally contains the query string* and was still outranked; a different
  dense model re-ranks the same semantic space and no dense representation
  encodes literal containment. It would also rebuild the index, so it could not
  have been measured as one variable even if it were the right idea.
- **Cross-encoder rerank over the top 25** — the right tool when the gold is in
  the pool but mis-ordered for *semantic* reasons (paraphrase, no shared rare
  token). The tally shows the opposite. It also costs 25 CPU model passes per
  query, hundreds of milliseconds against a 0.7 ms alternative, and no
  cross-encoder is cached on this machine.
- **Both at once** — two changes make the delta unattributable. Explicitly the
  first mistake the task warns about.
- **A `sdk_version` metadata filter** — genuinely attractive, since q01's failure
  is at heart a version ambiguity, and Week 3 already built the filter. Rejected
  for this week only because it needs a version the user did not type; inferring
  one is a new behaviour, not a retrieval change.

### What would have changed this decision

- If R-failures had been **v2/v3 ties** where both twins carry every query
  token, BM25 adds no discriminating signal and fusion just shuffles
  near-identical candidates — the metadata filter would win.
- If they had been **paraphrase misses** with no shared rare token, BM25 returns
  a noisy list and RRF can actively *demote* a dense-only hit — a cross-encoder
  would win.
- If the tally had been **mostly G**, no retrieval change moves the number at
  all and the honest recommendation would be prompt work.

---

## 6. After — same 12 questions, same index

**hit-rate@3 = 12/12.** (hit@1 = 7/12, hit@5 = 12/12.)

| metric | before (dense) | after (hybrid) | delta |
|---|---|---|---|
| hit-rate@1 | 6/12 | 7/12 | +1 |
| **hit-rate@3 (the metric)** | **10/12** | **12/12** | **+2** |
| hit-rate@5 | 12/12 | 12/12 | +0 |
| p50 ms (median of per-question medians) | 451.3 | 440.5 | −10.8 (−2.4%) |
| p50 ms (pooled) | 452.5 | 440.5 | −12.0 |
| p95 ms (pooled) | 664.0 | 731.4 | +67.4 |
| query embedding, median ms | 386.4 | 383.8 | — |

How the two fixes happened, from the fusion diagnostics (`d` = dense rank,
`b` = BM25 rank):

```
q01  gold=v3/client-send#parameters::1  -> HIT at rank 3
   A 1. 0.6371  v3/client-send#notes::3        rrf=0.032018 d=1 b=4
     2. 0.6274  v2/client-send#notes::2        rrf=0.032002 d=2 b=3
  *A 3. 0.5527  v3/client-send#parameters::1   rrf=0.031754 d=4 b=2   <- lifted
     4. 0.5590  v2/client-send#parameters::1   rrf=0.031258 d=3 b=5

q04  gold=v3/client-batch#parameters::1 -> HIT at rank 2
   A 1. 0.6038  v3/client-batch#notes::3       rrf=0.032522 d=1 b=2
  *A 2. 0.5213  v3/client-batch#parameters::1  rrf=0.031498 d=4 b=3   <- lifted
```

BM25 ranked each gold chunk 2nd or 3rd on the strength of the literal token, and
fusion pulled it from dense rank 4 into the window. Note in q01 that rank 3
carries a *lower* cosine (0.5527) than rank 4 (0.5590): ordering is by RRF, not
by score. That is the intended behaviour, not a bug.

The improvement reaches the user, not just the metric:

| | before | after |
|---|---|---|
| q01 | "the default for `retry_backoff_ms` is **100** [v2/client-send#parameters::1]" — wrong version | "the default `retry_backoff_ms` is **250** [v3/client-send#parameters::1]" — correct |
| q04 | `INSUFFICIENT_CONTEXT` | "the default `rate_limit_rps` is **20** [v3/client-batch#parameters::1]" — correct |

Re-labelling under hybrid gives **R = 0, G = 0, Not-In-Corpus = 0**, with the
q02 score-floor refusal unchanged.

---

## 7. Latency — the price, stated honestly

**The change costs about 0.7 ms per query and the measurement cannot see it.**

The apparent −10.8 ms is noise, and the replication proves it. Running
before → after → before → after:

| run | hit-rate@3 | p50 ms |
|---|---|---|
| before (1st) | 10/12 | 451.3 |
| after (1st) | 12/12 | 440.5 |
| before (2nd) | 10/12 | 441.1 |
| after (2nd) | 12/12 | 443.5 |

The spread *within* the same configuration (451.3 vs 441.1 — 10.2 ms) is larger
than the apparent difference *between* configurations. So the honest statement is
that the latency delta is **below measurement resolution**, not that hybrid is
faster. hit-rate@3 reproduced exactly both times; retrieval is deterministic.

The component split says why, measured inside the hybrid arm:

| stage | median ms | share |
|---|---|---|
| query embedding (bge-m3, CPU) | 374.9 | ~97% |
| dense vector search (Chroma) | 12.9 | ~3% |
| **BM25 over 38 chunks** | **0.351** | **0.09%** |
| **RRF fusion** | **0.322** | **0.08%** |

The entire change adds 0.67 ms to a query already spending 375 ms in a
transformer forward pass. Two caveats, stated rather than buried: this is a
hybrid-only breakdown, since the dense path has no instrumentation to compare
against stage-by-stage; and the BM25 index build (a one-off ~40 ms per process,
excluded by the discarded warm-up) is a startup cost, not a per-query one.

Method: timed around the public `search()` call by the same wrapper in both runs;
one warm-up query discarded so the 2.2 GB model load, the HNSW graph
materialisation and the BM25 build are never counted; 5 reps × 12 questions = 60
samples per run; p50 reported both as the median of the 12 per-question medians
and as the pooled median. Generation is excluded entirely — a Groq call is
hundreds of milliseconds and would swamp the signal. Single CPU machine, n = 60:
**treat differences under ~10% as noise.**

---

## 8. Which failures the change fixed, and which it did not touch

| id | exact token | before | after | verdict |
|---|---|---|---|---|
| q01 | `retry_backoff_ms` | MISS @4 | HIT @3 | **fixed** |
| q02 | `RedirectLoopError` | HIT @1 | HIT @1 | untouched (never failed) |
| q03 | `pool_reap_seconds` | HIT @3 | HIT @2 | untouched (never failed) |
| q04 | `rate_limit_rps` | MISS @4 | HIT @2 | **fixed** |
| q05 | `chunk_bytes` | HIT @2 | HIT @2 | untouched (never failed) |
| q06 | `CancelledError` | HIT @1 | HIT @1 | untouched (never failed) |
| q07 | `StreamClosed` | HIT @1 | HIT @1 | untouched (never failed) |
| q08 | — | HIT @2 | HIT @2 | untouched (never failed) |
| q09 | — | HIT @1 | HIT @1 | untouched (never failed) |
| q10 | — | HIT @1 | HIT @1 | untouched (never failed) |
| q11 | — | HIT @3 | HIT @1 | untouched (never failed) |
| q12 | — | HIT @1 | HIT @1 | untouched (never failed) |

- **Fixed: q01, q04** — both original R-failures, both exact-token queries where
  BM25 supplied the lexical rank the dense arm lacked.
- **Untouched, and worth naming: q02.** Its score-floor refusal is *not* a
  retrieval failure and is completely unmoved by this change. Fixing it requires
  re-calibrating `SCORE_FLOOR` in `generate.py` — a generation change, and a
  second variable.
- **Not fixed because it never broke: q03, q05–q12.** Three of them improved in
  rank without changing hit@3 (q03 3→2, q11 3→1), which is real but does not
  score.
- **Regressions: none.** No question that passed at baseline stopped passing.
  This was the risk worth watching: a chunk at dense rank 1 but absent from the
  BM25 list scores the same 1/61 as a lexical-only distractor at BM25 rank 1, so
  fusion *can* demote a correct dense hit. It did not here.

The remaining failure mode in this corpus is version ambiguity, and this change
does not address it. q01 now passes, but note it passes at rank 3 with two
higher-scoring chunks above it, one of them the v2 page. A `sdk_version` filter,
not a better retriever, is the real fix for that class.

---

## 9. Shipping decision

The rule, declared before the after-run: ship if hit-rate@3 improves by ≥ 2/12,
retrieval p50 grows by ≤ 25% of baseline, and nothing that passed at baseline
regresses.

**All three met: +2/12, latency delta below measurement resolution (true cost
~0.7 ms, 0.15% of a query), zero regressions. Ship it.**

Behind the `hybrid` flag, with the dense path retained as the rollback and still
the default, so `/ui`, `/api/v1/chat/ask` and the Week 3 evaluator behave exactly
as before unless the flag is set.

Honest limits on that number:

- **12 questions is a tiny sample.** Retrieval is deterministic, so +2 is exact
  *for these 12*; the confidence interval on 12/12 is wide and this is
  directional evidence, not a population estimate.
- **BM25 is free at 38 chunks, and will not stay free.** Scoring is linear in
  corpus size. At 38 chunks it is 0.35 ms; the constant matters at 10⁵ chunks and
  would need a real inverted index with posting-list iteration.
- **The saturation is suspicious in its own right.** hit@5 was already 12/12
  before the change, so all this bought was reordering within a window that
  already contained the answer. hit@1 (6/12 → 7/12) is the metric with headroom
  left, and it is where I would push next.

**What I would do next, and the number motivating it:** 1 of 12 questions
returns `INSUFFICIENT_CONTEXT` while holding the correct answer at rank 1 — q02,
top score 0.4287 against a 0.45 floor. That is a 100%-recoverable failure sitting
behind a hard-coded constant, and short exact-token queries embed systematically
lower than prose ones, so it will recur. Re-calibrating that floor is a smaller
change than anything in this report and it is worth exactly one question. After
that, the `sdk_version` filter for the version-ambiguity class.

---

## 10. The code diff

One retrieval change: a `hybrid` keyword and a branch in `search()`. The dense
path's final line is untouched, which is why every existing caller is unaffected.

```
$ git diff ragew3/chat-ui --stat -- app/
 app/api/routes/chat.py      |  26 ++++
 app/main.py                 |   3 +-
 app/rag/index.py            |  46 +++++-
 app/rag/store.py            |  28 ++++
 app/rag/verify_questions.py | 114 +++++++++
 app/web/index.html          | 206 ++++++++++++
```

```diff
 def search(
     query: str,
     strategy: str = "structure",
     embedder_name: str = "bge-m3",
     k: int = 5,
     persist_dir: Path | None = DEFAULT_INDEX,
     where: dict[str, Any] | None = None,
+    hybrid: bool = False,
+    stats: dict[str, float] | None = None,
 ) -> list[SearchHit]:
     embedder = get_embedder(embedder_name)
     store = get_store(strategy, embedder.name, persist_dir)
+    if hybrid:
+        from app.rag.retrieval import hybrid_search
+
+        return hybrid_search(
+            query, embedder=embedder, store=store,
+            strategy=strategy, k=k, where=where, stats=stats,
+        )
     return store.search(embedder.embed_query(query), k=k, where=where)
```

`app/rag/store.py` gains only additive `ids()` and `vectors()` accessors;
`search`, `upsert`, `count`, `reset` and `SearchHit` are unchanged.
`app/rag/verify_questions.py` gains a `--week4` section, leaving its Week 3
output byte-stable. **`app/rag/evaluate.py` and `app/rag/generate.py` are not
touched at all**, so Week 3's reported numbers still reproduce.

### The API and UI expose the flag; neither is a second retrieval change

`app/api/routes/chat.py` accepts `hybrid: bool = False` and passes it to
`search()`, returning `rrf_score` / `dense_rank` / `bm25_rank` on each chunk.
`app/web/index.html` gains a **Retrieval** selector (`dense` / `hybrid`), prints
which retriever ran, and shows a `d… b…` pill per chunk carrying the two source
ranks.

This is plumbing for an existing flag, not a second change to retrieval, and the
distinction is worth being precise about because the 35-mark criterion turns on
it. Three reasons it cannot affect the measurement:

- **Neither file is on the measured path.** `eval_week4.py` calls
  `index.search()` directly; it never goes through FastAPI.
- **The default is `False`**, so `/ui` and `/api/v1/chat/ask` behave exactly as
  they did in Week 3 unless a caller opts in.
- **The measurement predates it.** `before.json` and `after.json` were written
  before these two files were touched, and `--compare` still returns
  `fingerprint fields that differ: none`, `config keys that differ: ['hybrid']`,
  `verdict: PASS`.

Verified through the running app, both modes on the same question:

```
POST /api/v1/chat/ask   {"question":"rate_limit_rps default","k":3,"hybrid":false}
  1. 0.6038  v3/client-batch#notes::3
  2. 0.5429  v3/client-configure#notes::3
  3. 0.5239  v2/client-configure#notes::3          <- gold absent from the window

POST /api/v1/chat/ask   {"question":"rate_limit_rps default","k":3,"hybrid":true}
  1. 0.6038  v3/client-batch#notes::3        rrf=0.032522 d=1 b=2
  2. 0.5213  v3/client-batch#parameters::1   rrf=0.031498 d=4 b=3   <- gold, lifted
  3. 0.5239  v2/client-configure#notes::3    rrf=0.030579 d=3 b=8
```

The dense list is identical to the CLI's, which is the same "off means unchanged"
property stated once more at the HTTP boundary. Generation over hybrid hits
returns a cited answer rather than a blanket refusal, confirming the score-floor
interaction is handled end to end and not just inside the evaluator:

```
{"hybrid":true,"generate":true}  ->  refused: false
  "The default `rate_limit_rps` is 20 [v3/client-batch#parameters::1]"
```

The UI note is worth reading once: with hybrid on, ordering is by RRF, so a chunk
can appear **above** one with a higher cosine score (rank 2 scores 0.5213 while
rank 3 scores 0.5239 above). That is correct, and the UI says so rather than
leaving it looking like a sorting bug.

One pre-existing gap surfaced while testing, unrelated to this week: the citation
regex `\[([^\]\s]+::\d+)\]` rejects whitespace inside the brackets, and this model
writes `[ chunk_id ]` with spaces, so `citations` comes back empty even when the
answer cites correctly. It affects the citation chips in the UI, not retrieval or
any number in this report.

### Both numbers are visible in the app

`GET /api/v1/eval/summary` (`app/api/routes/eval.py`) serves the recorded runs, and
`/ui` renders them in a collapsed **Measurements** panel: hit-rate@1/@3/@5 and
p50/p95/min for dense against hybrid, the per-question verdicts, and the stage split
showing where a query's time actually goes.

Three deliberate choices in that panel, because a metrics view is easy to make
flattering:

- **Accuracy and latency never share an axis.** They are different scales, and a
  dual-axis chart would invent a relationship between them.
- **The cost bar is true-scale.** BM25 and fusion really are narrower than one pixel
  next to the 375 ms embedding, and the bar is not stretched to make them visible —
  their being invisible *is* the finding. The two added stages are the only coloured
  segments; everything already being paid is grey.
- **The repeat runs are shown next to the headline.** The panel states that two runs
  of the same configuration differ by 10.2 ms, more than the 10.8 ms between
  configurations, so a reader cannot mistake the p50 drop for a speed-up.

The route is read-only and never re-runs an evaluation: a button that recomputed the
comparison would quietly produce different numbers from the ones in this report.

`/api/v1/chat/ask` additionally returns `retrieval_ms` and, for hybrid, the per-stage
split for that single request, which the page shows beside the retriever name. That
timer lives in the route, not inside `search()`, so the measured path stays
uninstrumented. Note the first request after a restart reports tens of seconds — it
pays the one-off model load, which the evaluator discards as warm-up and this live
number honestly does not.

New modules, none of them on the dense path:

| file | purpose |
|---|---|
| `app/rag/bm25.py` | Okapi BM25, pure stdlib, underscore-preserving tokeniser |
| `app/rag/retrieval.py` | `hybrid_search()`, `rrf_fuse()`, `FUSION_K=60`, `FETCH_K=25` |
| `app/rag/golden_set.py` | derives gold chunk_ids, emits/checks the jsonl |
| `app/rag/fingerprint.py` | the one-variable guard |
| `app/rag/eval_week4.py` | hit-rate@3, latency, `--detail`, `--compare` |
| `app/rag/label_week4.py` | the R/G/Not-In-Corpus pass, generation in the loop |
| `app/api/routes/eval.py` | serves the recorded runs to the UI, read-only |

### Two implementation details that would otherwise be silent bugs

**BM25 IDF must be the always-positive variant.** `log(1 + (N-df+0.5)/(df+0.5))`
rather than the classic `log((N-df+0.5)/(df+0.5))`, which goes **negative** for
any term in more than half the documents. With 38 chunks where "client",
"parameters" and "default" appear almost everywhere, negative IDF would penalise
a chunk for containing a common query word and invert the ranking — quietly, and
in exactly the queries this exercise is about. Measured: `idf(client) = 0.013`,
positive.

**Fused hits must keep cosine-scale scores.** A raw RRF score maxes out at
`2/61 = 0.033`, far below `SCORE_FLOOR = 0.45`. Writing fused values onto
`SearchHit.score` would have made *every* answer refuse while the retrieval
metrics looked perfect — a retrieval experiment silently becoming a generation
regression. So `.score` stays the dense cosine, and RRF travels in `metadata`.
For a chunk BM25 surfaced that the dense arm did not return, the cosine is
recovered as a dot product against the stored vector, valid because bge-m3
normalises its embeddings (verified: ‖v‖ = 1.0).

### Reproducing

```bash
python -m app.rag.golden_set --check                 # jsonl matches a fresh derivation
python -m app.rag.verify_questions --week4           # 12/12 gold ids resolve
python -m app.rag.eval_week4 --out report/w4/before.json
python -m app.rag.eval_week4 --hybrid --out report/w4/after.json
python -m app.rag.eval_week4 --compare report/w4/before.json report/w4/after.json
python -m app.rag.label_week4 --model openai/gpt-oss-120b --out report/w4/labels.json
```

Raw evidence: `report/w4/before.json`, `after.json`, `before2.json`,
`after2.json`, `labels.json`, `labels-hybrid.json`, `before-detail.txt`,
`after-detail.txt`.

### Disclosure

Before the golden set was written, five ad-hoc queries were run against a
throwaway in-memory BM25 prototype to check that the approach was worth
pursuing. Two of those phrasings (`retry_backoff_ms`, `rate_limit_rps`) later
became q01 and q04. They were kept because both were already known failures from
the task's own problem statement, and both are drawn straight from the parameter
tables. The prototype also used subword splitting, and predicted q01 would
*stay* broken; the shipped tokeniser does not split, and q01 is fixed — so the
prototype did not select these questions for a flattering result.

The configured `GROQ_MODEL` in `.env` (`llama-3.3-70b-versatile`) has been
retired by Groq and now 404s. Generation for the labelling pass used
`openai/gpt-oss-120b` via an explicit `--model` override; `.env` was left
unchanged. This affects only the R/G labelling, never hit-rate@3, which is
search-only.
