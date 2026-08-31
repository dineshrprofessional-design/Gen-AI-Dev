# Week 5 notes — trace sample, open-coding, replay, prediction

## 0. What had to be reconstructed, disclosed first

The task assumes a trace file "grown past a thousand lines". This repo had no
tracing at all, so it was built this week (`app/rag/trace.py`, commit `41e8301`)
and a population of **151 real traces** was produced by driving the actual
`/api/v1/chat/ask` route (hybrid retrieval — the config Week 4 shipped — real
generation via `openai/gpt-oss-120b`): 141 questions from a seeded mechanical
recipe in `app/rag/make_traces.py`, 10 curated demo questions traced separately.
Every trace field the replay requirement names was in the schema from birth
(prompt version, chunk_ids + scores, model + params, raw output), so nothing had
to be added mid-week — the honest note is that the *whole file* is one week old,
not a thousand organic lines.

## 1. Replay evidence (requirement 1)

Seeded pick: seed `20260831` → trace `b98918daf643` of 141 organic traces.
Replayed from the trace's own fields (`python -m app.rag.replay --pick --seed 20260831`):

```
trace b98918daf643  ts=2026-08-31T18:47:34+0530  git=41e8301  prompt=faf82290b6be
question : 'what type is telemetry'
config   : strategy=structure hybrid=True k=5 filter=None
NOTE: environment drifted since the trace: git_sha (trace 41e8301, now 111f87e)

retrieval, original vs replayed:
  #   recorded chunk_id                    score    replayed chunk_id                    score    match
  1   v3/client-configure#example::2       0.4167   v3/client-configure#example::2       0.4167   yes
  2   v3/client-configure#parameters::1    0.3709   v3/client-configure#parameters::1    0.3709   yes
  3   v3/client-upload#parameters::1       0.3766   v3/client-upload#parameters::1       0.3766   yes
  4   v3/client-stream#parameters::1       0.3409   v3/client-stream#parameters::1       0.3409   yes
  5   v2/client-close#parameters::1        0.3300   v2/client-close#parameters::1        0.3300   yes
  -> retrieval replay IDENTICAL

model    : openai/gpt-oss-120b  params={'temperature': 0}
original : 'INSUFFICIENT_CONTEXT'
replayed : 'INSUFFICIENT_CONTEXT'
  -> outputs IDENTICAL
```

Nothing was missing from the trace. **Not reconstructable, stated plainly:**
wall-clock latency and timestamp (properties of the original moment), and
token-level LLM determinism is not guaranteed by the provider — the replay
prints both outputs rather than asserting equality in general. The git-sha
drift NOTE is the replay tool catching that HEAD moved one commit (the data
commit) past the trace's code commit — the detector working, not a defect.

## 2. The seeded random sample (requirement 2)

`python -m app.rag.replay --sample 20 --seed 20260831` over the 141 organic
traces. **Seed: 20260831.** The demo set is excluded by construction.

```
b98918daf643  aa983b01fd36  bcc84d1515d9  44491c769398  187edede03a2
b226ff8a5996  4452ba533635  39fc857740fa  d1ed2b96d352  179023eb0b6a
d84a2cd4f913  c832c65e52a3  cbaa3d073d15  7a7abc208979  6659349bf2cb
53fe4190ba35  493b0e8f6b28  67babf113e70  9d88b87ad51d  da78a479f1e1
```

## 3. Open-coding: one observation per trace (requirement 3)

Written while reading each trace in full — question, all five retrieved chunks,
raw output, citations — against the corpus. Observations, not diagnoses; **zero
code changes were made between drawing the sample and finishing this list**
(`git status` showed only untracked .md files throughout).

1. `b98918daf643` — "what type is telemetry": the parameters table containing
   `telemetry | bool` was retrieved at rank 2, and the pipeline still returned
   INSUFFICIENT_CONTEXT because the rank-1 chunk's cosine (0.417) is under the
   0.45 floor.
2. `aa983b01fd36` — "can I disable telemetry for GDPR compliance": the configure
   table was at rank 2 scoring 0.464 — above the floor — but the gate reads only
   the rank-1 score (0.445), so the request was refused before the model ran.
3. `bcc84d1515d9` — "what does the client do when the endpoint returns 5xx":
   the answer is correct and grounded, but the citation is written in full-width
   【】 brackets and the recorded citations list is empty.
4. `44491c769398` — "is chunk_size required": correct "No, optional" citing the
   right chunk, but the citation has spaces inside the brackets and was not
   captured.
5. `187edede03a2` — "what type is concurrency": correct "int", citing the batch
   parameters table which sat at rank 4 — outside the top 3 but inside the k=5
   context — and the citation was captured cleanly.
6. `b226ff8a5996` — "what does Client.send() do": a faithful multi-sentence
   summary; every claim, including the thread-safety sentence I first suspected
   was invented, is present in the cited intro chunk.
7. `4452ba533635` — "when are idle connections reaped": correct (90 s via
   `pool_reap_seconds`), 【】-bracket citation, citations field empty.
8. `39fc857740fa` — "how long is an upload handle valid": correct 24-hour /
   `retain_hours` answer with two accurate spaced-bracket citations recorded as
   none.
9. `d1ed2b96d352` — "what is the default for timeout_ms": the question is
   genuinely ambiguous (four tables define `timeout_ms`); the model enumerated
   three v3 defaults and never mentioned the v2 page that was ranked 1st.
10. `179023eb0b6a` — "did default_timeout_ms change between v2 and v3": all
    five retrieved chunks are Notes sections from pages that do not define
    `default_timeout_ms`, neither configure table appeared, and the model
    refused.
11. `d84a2cd4f913` — "base url default": answered with configure's default
    citing the **v2** page for a version-less question; the v3 configure table
    never appeared in the five chunks, and rank 1 was send's `base_url` whose
    default is None, which the answer silently ignores.
12. `c832c65e52a3` — "show me an example of Client.configure()": returned the
    v3 example fence verbatim and correctly, with a 【】 citation recorded as
    none.
13. `cbaa3d073d15` — "how do I set proxy": the answer invents a concrete call
    `client.send(payload, proxy="http://my-proxy:8080")` that appears in no
    retrieved chunk — the docs define the parameter but show no proxy example.
14. `7a7abc208979` — "fail fast default": none of the five retrieved chunks
    contains `fail_fast`, and the model confidently answered about `force` on
    close(), describing that as "failing fast"; the asked parameter is never
    mentioned.
15. `6659349bf2cb` — "did log_level change between v2 and v3": every retrieved
    chunk came from v2 pages, none was a configure table (where `log_level`
    lives), and the model refused.
16. `53fe4190ba35` — "how is jitter applied to retries": correct full-jitter
    formula; 【】 citation, citations field empty.
17. `493b0e8f6b28` — "is there a changelog for v3.2.0": out-of-corpus question,
    refused by the model itself with scores above the floor — the designed
    behaviour, and I have nothing to add.
18. `67babf113e70` — "can I call close twice": correct "yes, no-ops" with an
    accurate spaced citation not captured by the parser.
19. `9d88b87ad51d` — "what does Client.close() do": correct two-version summary
    with two valid captured citations — one of the few answers whose citations
    survived intact.
20. `da78a479f1e1` — "how do I set chunk_bytes": the question is ambiguous
    between upload and send_batch; the model answered upload-only, stitched a
    code sample from the example chunk, and its spaced citations were not
    captured.

## 4. The tally

The clusters, counts, severities and example trace_ids are in
[`taxonomy.md`](taxonomy.md) — six modes; the largest (citation capture) touches
10 of 20 traces. 4 of 20 traces were fully clean (`187edede03a2`,
`b226ff8a5996`, `9d88b87ad51d`, and the correct refusal `493b0e8f6b28`).

## 5. The prediction (requirement 5) — dated, falsifiable, committed before any fix

> **2026-08-31.** Next week I attack mode 1 (citation capture). The change:
> normalise citation extraction in `app/rag/generate.py` — accept full-width
> 【】 brackets, strip zero-width characters, and tolerate spaces inside the
> brackets in `_CITATION`, applying the same normalisation before the
> known-context check. I predict this drops mode 1 from **50% of traces (10/20)
> to under 5%** on a fresh 20-trace sample drawn with **seed 20260901** from a
> newly generated population, and changes **zero** answers' text (it touches
> only post-processing). If mode 1 stays above 5%, the prediction is wrong and
> the residue will name what the regex alone couldn't fix.

Committed to git with today's date — commit hash pasted here after the commit
exists: `PREDICTION_COMMIT_HASH`.

## 6. Why a public benchmark would have missed the top 3 modes (requirement 6)

A public benchmark scores answers against generic corpora and gold labels, and
mode 1 is not an answer defect at all — it lives in this app's own citation
regex meeting one model's typographic habits (【】, zero-width spaces), a seam
no benchmark instruments. Mode 2 requires a corpus that documents the same
symbol twice with different values across versions — exactly the deliberate
v2/v3 twin structure of this corpus, which public datasets don't have, so
"answered from the wrong version" cannot even occur there. Mode 3 is an
interaction between our hard-coded 0.45 score floor and RRF's reordering — a
threshold and a fusion rule no benchmark shares, invisible to any end-to-end
public score.

## 7. Bonus — the demo set vs the random sample

The 10 demo-set traces (UI sample chips + Week 3's polished gold questions)
were open-coded the same way. Headline comparison for the top mode:

| | random sample (n=20) | demo set (n=10) |
|---|---|---|
| mode 1, citation capture lost/corrupted | 10/20 = **50%** | 6/10 = **60%** |

Demo observations, one line each: `4ed93f95eefa` correct 250 with a 【】 cite
(recorded none); `0f716df1f40c` correct "payload" but cites the v2 page for a
Client.send() question; `759633170b5a` a model answer covering *both* versions
correctly, whose two citations were captured with a leading zero-width
character and therefore both flagged *invalid*; `a3316371230a` correct close()
guidance, 【】 cite; `4fb06208fdab` and `29e1a872fee7` correct refusals of
out-of-corpus questions; `655dbfdf558b` — **the flagship demo question**
("…retry_backoff_ms on Client.send() in v3?") **refused**, with the answer
present in its own rank-2 chunk; `706860c6d48e` perfect, clean citation;
`3051c2f6b87d` and `dcdf464065a0` correct with spaced citations recorded as
none.

The paragraph the team owes itself: the demo set has not been flattering us on
the mode we'd expect — citation capture is *worse* there (60% vs 50%) and
nobody noticed because the answers read fine on screen. What the demo set has
been hiding is sharper: under the hybrid configuration we shipped this week,
the exact question we open every DX review with now returns
INSUFFICIENT_CONTEXT while the correct chunk sits second in its own context —
and no one had asked it since the change, because demo questions are only ever
run at demos. A curated set doesn't just inflate scores; it goes stale the
moment the system underneath it moves, which is precisely when it's needed.

## 8. Where everything lives

| artifact | path |
|---|---|
| trace population (151, JSONL) | `traces/traces.jsonl` |
| pool recipe + trace_id manifest | `app/rag/make_traces.py`, `traces/pool_manifest.json` |
| replay tool / sampler | `app/rag/replay.py` |
| replay + sample transcripts | `report/w5-replay.txt`, `report/w5-sample.txt` |
| taxonomy | `taxonomy.md` |
