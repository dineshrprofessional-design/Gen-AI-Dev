# `api/` — the HTTP layer the Week 7 agent reads

## 0. What is invented, disclosed first

`docs/` documents a Python SDK. It names **no HTTP path, no verb and no operation
id anywhere**. Everything in this directory that describes a wire protocol was
**authored for Week 7 Task Set E**. No real API was consulted and none is
described.

The reason the fiction is necessary is itself a finding, and it is recorded
mechanically rather than claimed: `api/changelog.md` ends with **"Removed:
none"** — computed, not asserted. v3 is a strict superset of v2, with 19 added
parameters, 2 added methods, and **zero** removed parameters or methods. There is
no genuine deprecation in this corpus to find. The brief requires a deprecation
two-hop, so one had to be manufactured.

### Everything invented, in one table

Nine operation bindings — path, verb and operationId are authored; every
parameter on them is derived from the real tables in `docs/`:

| operationId | verb + path | SDK method | version |
|---|---|---|---|
| `sendMessageV2` | `POST /v2/messages` | `Client.send` | v2 |
| `sendMessageV3` | `POST /v3/messages` | `Client.send` | v3 |
| `streamMessageV2` | `POST /v2/messages:stream` | `Client.stream` | v2 |
| `streamMessageV3` | `POST /v3/messages:stream` | `Client.stream` | v3 |
| `closeConnectionsV2` | `POST /v2/connections:close` | `Client.close` | v2 |
| `drainConnectionsV3` | `POST /v3/connections:drain` | `Client.close` | v3 |
| `sendMultiV2` | `POST /v2/messages:multi` | *none* | v2 |
| `sendBatchV3` | `POST /v3/batches` | `Client.send_batch` | v3 |
| `uploadFileV3` | `POST /v3/files` | `Client.upload` | v3 |

Four deprecations, each pinned to a verbatim sentence in `docs/`:

| deprecated | replaced by | anchored to |
|---|---|---|
| `sendMessageV2` | `sendMessageV3` | `docs/v2/client-send.md` — "Automatic retries arrived in v3." |
| `streamMessageV2` | `streamMessageV3` | `docs/v2/client-stream.md` — "Resumable streams arrived in v3." |
| `closeConnectionsV2` | `drainConnectionsV3` | `docs/v2/client-close.md` — "Graceful draining arrived in v3." |
| `sendMultiV2` | `sendBatchV3` | `docs/v3/client-send.md` — "`Client.send_batch()`, which amortises connection setup across the whole batch" |

### The largest liberty taken

**`sendMultiV2`.** It is the only operation with no SDK binding at all — v2 has no
batch method, so nothing in `docs/` corresponds to it. It exists because it is the
only way to construct a deprecation whose replacement is a *different SDK method
on a page that exists only in v3*, which is what makes the hardest race question
un-shortcuttable. It is the one record a sceptical reader should discount first.

### What is deliberately NOT invented

No new parameters, defaults, types or behaviour. No removed SDK symbols. No error
codes. No throughput ceilings. No release versions or dates. No second
authentication scheme — the one security scheme is **derived** from the real
`api_key` row, which is documented as required in both versions.

`Client.configure()` is given **no endpoint**. It sets client-side defaults and
never crosses the wire, which `docs/v3/client-configure.md` states outright.
Inventing an endpoint for it would have been a liberty taken for no reason.

## 1. What is generated

| file | status | source |
|---|---|---|
| `endpoints.yaml` | **AUTHORED** — the only invented content | written by hand, ~70 lines |
| `openapi.json` | GENERATED | `docs/` tables + `endpoints.yaml` |
| `changelog.md` | GENERATED | `docs/` tables only |
| `deprecations.json` | GENERATED | projection of `openapi.json` |

```bash
python -m app.rag.week7_corpus              # regenerate
python -m app.rag.week7_corpus --check      # byte-diff against the committed files
python -m app.rag.week7_corpus --validate   # the regression gates below
python -m app.rag.week7_corpus --reconcile  # the 8-vs-9 divergence table
```

Every operation in `openapi.json` carries its own `x-provenance` string, so this
disclosure survives being read one operation at a time through a tool. An agent
that never opens this file still sees that the path was authored.

## 2. Why this lives outside `docs/`

`DEFAULT_DOCS = Path("docs")` is the only corpus root in the codebase. Every
consumer — the index builder, the chunker, the Week 6 assertions — takes a
`docs_root` argument and defaults to it. A sibling directory is therefore
invisible to the Week 3–6 pipeline **by construction**, not by care.

That matters because three committed Week 6 negative controls, all currently
passing in `report/judge_v1.txt` and `report/judge_v2.txt`, would flip to false
negatives if the indexed corpus grew an API surface:

| case | question | what would break it |
|---|---|---|
| `w6-030` | is there a changelog for a specific release | a changelog inside `docs/`, or any release-version string here |
| `w6-031` | throughput ceiling on a version-1 vector endpoint | a spec defining that path or a documented ceiling |
| `w6-032` | how do I authenticate with a federated scheme | a spec with a second security scheme |

`--validate` enforces all three as gates:

- **G1** — `docs/` is byte-identical to the tree Week 7 started from, checked by
  content hash *and* `git diff --quiet HEAD -- docs/`.
- **G2** — no byte anywhere under `api/` matches the forbidden patterns. There are
  no per-file carve-outs, which is why this README paraphrases rather than quotes
  them. The single documented exemption is the OpenAPI document-format version
  field, which is a different namespace from a product release.
- **G3** — `week6_cases --check` exits 0 and `week6_assertions --json` is
  byte-identical to the snapshot taken before Week 7 began.

`--validate` additionally re-runs the two greps that `app/rag/week6_assertions.py`
lines 19–27 cite as its reason for refusing two of the brief's criteria, so that
committed docstring cannot quietly become untrue.

## 3. The reconciliation worth knowing about

`week6_assertions.divergent_defaults()` is **name-scoped** and reports 9
version-divergent parameters. Only **8** are genuine. `base_url` is not one of
them: v3 `Client.send()`'s per-call override defaults to `None` while v3
`Client.configure()`'s defaults to the server URL — two different parameters that
happen to share a name.

`changelog.md` is generated from a method-scoped computation keyed on
`(page_id, parameter)` instead. Generating it from the name-scoped function would
have put a flat contradiction of `docs/` into the changelog. `--reconcile` prints
the 8-vs-9 table with the exclusion and its reason, and `--validate` asserts the
8 are a subset of the 9 — so if the corpus ever moves, the build stops rather than
the changelog quietly diverging.
