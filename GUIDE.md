# Ingestion — a guide

This explains the code currently in `app/rag/`. Read it alongside
`notebooks/01_ingestion.ipynb`, which runs every stage with real output.

---

## 1. What this project is

A RAG (Retrieval-Augmented Generation) docs-assistant, built one phase at a time.

A full RAG pipeline has five stages. **Only the first is built:**

```
  ingest  →  chunk  →  embed  →  store  →  retrieve  →  generate
    ▲
  you are here
```

Everything downstream inherits ingestion's decisions. If a chunk loses track of
which page it came from, no amount of clever retrieval gets it back — which is
why this phase is about metadata discipline, not text processing.

---

## 2. What ingestion does

**Read a folder of documentation files, and turn each one into a `Document`
carrying four required metadata fields. If a file can't be read, record why and
keep going.**

The four required fields — a document without all four is a failed ingest:

| Field | Example | Why it exists |
|---|---|---|
| `source_file` | `v3/reference/client-send.md` | Where this came from. Without it a citation can't be traced |
| `page_id` | `client-send` | Which page, independent of version |
| `sdk_version` | `v3` | Lets a search be scoped to one version |
| `page_type` | `reference` | `reference` \| `guide` \| `changelog` |

---

## 3. The pipeline

Every file goes through the same four steps:

```
file on disk
   │
   ├─ 1. pick an extractor        .md → MarkdownExtractor
   │     (by file extension)      no match → SKIPPED, done
   │
   ├─ 2. extract                  bytes → text + front_matter + title_hint
   │                              fails → FAILED (stage = extract)
   │
   ├─ 3. resolve metadata         fill the gaps from the folder path
   │                              bad declared value → FAILED (stage = metadata)
   │
   └─ 4. check identity           (sdk_version, page_id) seen before?
                                  yes → FAILED (stage = validate)
                                  no  → Document ✓
```

### Stage 1 — pick an extractor

`app/rag/extractors/__init__.py` is a dictionary from extension to handler:

| Extensions | Extractor | Notes |
|---|---|---|
| `.md` `.markdown` | markdown | parses front matter |
| `.txt` `.rst` | text | reads as-is |
| `.html` `.htm` | html | stdlib parser, drops `<script>`/`<style>` |
| `.pdf` | pdf | needs `pypdf` — **lazy imported** |
| `.docx` | docx | needs `python-docx` — **lazy imported** |

Anything else returns `None`, which means **record it as skipped**. Never
silently ignore a file — silence is how you end up with half a corpus indexed
and nobody noticing.

*Lazy imports* mean a missing optional package quarantines that one format
rather than crashing the program at startup.

### Stage 2 — extract

Every extractor returns the same shape:

```python
Extracted(
    text: str,                    # the content
    front_matter: dict[str, str], # what the file DECLARED about itself
    title_hint: str | None,       # a title found INSIDE the content
)
```

Only markdown can populate `front_matter` — a PDF has no way to declare
anything. That asymmetry is exactly why stage 3 exists.

### Stage 3 — resolve metadata

**Front matter is optional.** Real documentation isn't authored for your
pipeline, so each field walks a chain until something answers:

| Field | Precedence |
|---|---|
| `sdk_version` | front matter → a folder named like `v3` → `"unknown"` |
| `page_type` | front matter → a folder named `reference`/`guide`/`changelog` → `reference` |
| `page_id` | front matter → slugified filename |
| `title` | front matter → first heading in the content → humanised filename |

So `sample-corpus/v3/reference/client-send.md`, which has **no front matter at
all**, still resolves completely:

```
sdk_version  = v3               <- path
page_type    = reference        <- path
page_id      = client-send      <- filename
title        = Client.send()    <- content
```

`resolve()` also returns **which link answered**, stored on the document as
`metadata_sources`. This matters: "the author declared it" and "I guessed it
from a folder name" are different levels of trust, and only one is worth
investigating when a value looks wrong.

**One thing raises here:** an *invalid declared* value. If a file says
`page_type: tutorial`, that is an error, not something to guess around — the
author stated an intent and it's wrong. A *missing* value is derived; a *wrong*
value is quarantined.

### Stage 4 — check identity

`(sdk_version, page_id)` is a page's identity and must be unique. Two files
claiming the same identity would later collide in the index and one would
silently overwrite the other.

The **first** file wins; the second is quarantined with a message naming both.
First means first alphabetically by path — arbitrary, but deterministic.

---

## 4. Quarantine, not abort

This is the most important design decision in the phase.

`load_documents()` returns an **`IngestReport`** instead of raising:

```python
IngestReport
    documents : list[Document]     # made it through
    failures  : list[FailedFile]   # source_file, stage, reason
    skipped   : list[SkippedFile]  # source_file, reason
```

Each file sits inside its own `try/except`. A failure becomes a **record**, not
an exception. One malformed page in a corpus of fifty thousand must not stop the
other 49,999.

**Only one thing is fatal:** the docs root not existing. That's not a bad file,
it's a bad invocation.

Against `sample-corpus/` this produces:

```
5 loaded · 4 failed · 1 skipped        exit code 0

FAILED (quarantined — the run continued)
  [metadata] misc/bad-page-type.md
             page_type 'tutorial' must be one of reference, guide, changelog
  [extract ] misc/corrupt.pdf
             unreadable PDF: Stream has ended unexpectedly
  [extract ] misc/empty.md
             no text content
  [validate] v3/reference/zz-duplicate-of-client-send.md
             duplicate page v3/client-send — already ingested from v3/reference/client-send.md

SKIPPED (not read at all)
  misc/diagram.png — unsupported file type .png
```

Note the **exit code 0**. Bad files did not fail the run. `--strict` flips that
contract to exit 1 — use it in CI, where you *do* want a broken corpus to fail
the build.

---

## 5. The files

Start with **`app/rag/loader.py`**. The `for path in sorted(...)` loop in
`load_documents` *is* the program — about 40 lines. Everything else is a helper
it calls.

| File | Job |
|---|---|
| `app/rag/models.py` | The shapes: `Document`, `IngestReport`, `FailedFile`, `SkippedFile`, and the `PageType` / `MetadataSource` / `Stage` enums |
| `app/rag/extractors/base.py` | The `Extractor` protocol and `Extracted` |
| `app/rag/extractors/*.py` | One module per format — all expose `extract(path)` |
| `app/rag/extractors/__init__.py` | The registry — extension → extractor |
| `app/rag/metadata.py` | Stage 3: the fallback chains |
| `app/rag/loader.py` | **The conductor** — loops, catches, builds the report |
| `app/rag/ingest.py` | CLI: argument parsing and printing |
| `app/rag/service.py` | Holds the last report in memory for the API |
| `app/api/routes/ingest.py` | HTTP wrapper around the same functions |

Suggested reading order: `loader.py` → `models.py` → `extractors/text.py` (10
lines) → `metadata.py`.

---

## 6. The corpora

| Folder | Purpose |
|---|---|
| `docs/` | **Clean.** Every file has front matter. Nothing is quarantined |
| `sample-corpus/` | **Deliberately broken.** Proves quarantine works |

`sample-corpus/` contains, on purpose: markdown with no front matter (must
succeed by derivation), an HTML page, a plain-text guide, a PDF, a DOCX, a
declared-invalid `page_type`, a corrupt PDF, an empty file, a duplicate
identity, and a `.png` that must be skipped.

> ⚠️ **Anything you put in `docs/` becomes corpus.** Don't drop notes or READMEs
> there — they'll be ingested as documentation. That's why this guide lives at
> the repo root.

---

## 7. Running it

### Command line

```bash
python -m app.rag.ingest                                 # clean corpus
python -m app.rag.ingest --docs sample-corpus            # messy corpus → exit 0
python -m app.rag.ingest --docs sample-corpus --strict   # same corpus  → exit 1
python -m app.rag.ingest --version v3                    # one SDK version only
python -m app.rag.ingest --show                          # print every body
python -m app.rag.ingest --json                          # machine-readable report
```

| Exit code | Meaning |
|---|---|
| 0 | At least one document loaded (quarantined files are fine) |
| 1 | Nothing loaded, docs root missing, or `--strict` with failures |

### HTTP API

```bash
uvicorn app.main:app --reload --port 8000
```

| Route | Purpose |
|---|---|
| `POST /api/v1/ingest/run` | Run ingestion. Body: `{"docs_root": "...", "sdk_version": "..."}` |
| `GET /api/v1/ingest/report` | The last report — 404 before any run |
| `GET /api/v1/ingest/documents` | List, filter by `sdk_version` / `page_type` |
| `GET /api/v1/ingest/documents/{sdk_version}/{page_id}` | One document with body |

Swagger UI at http://127.0.0.1:8000/docs.

State is **in-process only** — restarting the server means running the ingest
again. That's honest while there's no index to be stale.

### Notebook

```bash
jupyter lab notebooks/01_ingestion.ipynb
```

Already executed, so it reads top to bottom without running anything.

---

## 8. A lesson worth carrying into the next phase

The same parameter table, ingested from two formats:

**From DOCX** — cells are real objects, so rows survive:

```
| name | type | default | required |
| timeout_ms | int | 5000 | no |
| force | bool | False | no |
```

**From PDF** — PDF stores glyph positions, not structure:

```
name type default required
payloads list - yes
retry_backoff_ms int 250 no
```

The row boundaries are gone. **No chunker downstream can rebuild them** — the
information was destroyed at ingest time. When you choose what goes into a
corpus, you're setting a ceiling on what every later stage can do.

---

## 9. What is deliberately *not* built

- **Chunking** — splitting documents into retrievable pieces
- **Embedding** — no model chosen yet
- **Vector store** — nothing is indexed
- **Retrieval / generation** — no search, no LLM
- **Tests** — removed by request
- **Frontend** — `web/` holds `node_modules` but no source

Ingestion is complete on its own terms: it reads a corpus, resolves metadata,
and reports honestly.

---

## 10. Next phase — chunking

A reference page looks like this:

````
## Client.send()

Sends a request...                          ← prose

| name             | type | default |
|------------------|------|---------|
| retry_backoff_ms | int  | 250     |       ← parameter table

```python
client.send(retry_backoff_ms=500)           ← code fence
```
````

Split that every 1000 characters and you can land mid-table — producing a chunk
that says `250` but no longer says *which parameter* or *which method*. That
chunk can't be retrieved and can't be used.

The next phase is learning to prevent it, and **measuring** the difference
rather than eyeballing it.
