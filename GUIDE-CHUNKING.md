# Chunking — a guide

Phase two. Ingestion produced whole `Document` pages; chunking splits them into
retrievable pieces. Read `GUIDE.md` first — this builds directly on it.

---

## 1. The problem

Retrieval works on pieces, not pages. But a reference page is not prose you can
cut anywhere:

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

Cut at 1000 characters and you can land mid-table. The resulting chunk contains
`250` but no longer says **which parameter** or **which method**. It cannot be
retrieved for "what is the default of `retry_backoff_ms`", and if it somehow
were, it could not answer the question.

The phase is about preventing that — and, more importantly, **measuring**
whether you did.

---

## 2. Two strategies

Deliberately comparable: same documents in, same `Chunk` shape out. Only the
cut points differ.

| | **A — `fixed`** | **B — `structure`** |
|---|---|---|
| Method | recursive character splitting | markdown heading splits |
| Chunk size | 1000 chars | 1200 chars (**soft**) |
| **Overlap** | **150 chars** | **none** |
| Separators | `\n\n` → `\n` → space | — |
| Tables | may be cut | **atomic** |
| Code fences | may be cut | **atomic** |
| Context carried | none | heading breadcrumb |

### Why A exists

A is the baseline. **It is deliberately not improved** — its failures are the
measurement. If you "fix" A, you lose the number that shows B is worth having.

### Why B has no overlap

Overlap is a patch for not knowing where the boundaries are. A fixed-size
cutter repeats 150 characters so that a severed sentence survives on one side of
the cut. B cuts at headings, which are real boundaries, so overlap would only
duplicate content, inflate the index, and let one section's words pollute
another section's match.

B carries the **heading breadcrumb** instead — `Client.send() > Parameters`
prepended to each chunk. Cheaper than repetition and more precise: a chunk
holding nothing but a parameter table still names the method it belongs to.

---

## 3. How B actually works

The trick is that **B barely does anything**. The work happens in the parser.

`app/rag/markdown.py` turns a document into typed blocks — `heading`,
`paragraph`, `table`, `code_fence`, `list` — with two rules:

- A ```` ``` ```` fence swallows everything up to its closing fence, including
  any `#` or `|` inside it.
- A table captures **header row + separator + every data row** as one unit.

Both are marked `is_atomic`. B then only ever places *whole blocks*, so it
**cannot** split a table or a fence even if it wanted to. Structure-awareness is
a parsing problem, not a chunking problem.

Two rules in B are worth knowing because they look like bugs and aren't:

**The size budget is soft.** B's largest chunk on the sample corpus is 1950
characters against a 1200 budget. An oversized table becomes one big chunk
rather than a cut one — a half-table is worth nothing regardless of how neatly
it fits.

**A heading never flushes alone.** Without that rule, `## Parameters` becomes a
14-character chunk with no content, which still matches on the word
"parameters" and outranks the chunk actually holding the answer. This was a real
bug, found by looking at ranked results.

---

## 4. The measurement

Two numbers, not opinions:

| | fixed | structure |
|---|---|---|
| chunks (docs/) | 7 | 10 |
| **split code fences** | **1** | **0** |
| **orphaned table rows** | **2** | **0** |

An *orphaned table row* is a chunk containing `| a | b |` rows with no
`|---|---|` header separator — a data row that lost the header naming its
columns. A *split code fence* is an odd number of ` ``` ` markers, meaning a
code block was cut in half.

### Same page, both strategies

`docs/v3/client-send.md`, 3383 characters:

**A — 5 chunks, 3 damaged**

| chunk | range | starts with | |
|---|---|---|---|
| `::0` | 0:720 | `# Client.send()` | |
| `::1` | 570:1568 | `urfaced to the caller…` | mid-word |
| `::2` | 1418:2416 | `s retries server-side…` | **orphan row** |
| `::3` | 2266:2942 | `s that trigger the automatic…` | **orphan row** |
| `::4` | 2792:3383 | `max_retries=5,` | **split fence** |

Ranges overlap (720 → 570 is the 150-char back-step). No chunk knows its
heading.

**B — 4 chunks, 0 damaged**

| chunk | range | breadcrumb |
|---|---|---|
| `#clientsend::0` | 1:704 | `Client.send()` |
| `#parameters::1` | 705:2628 | `Client.send() > Parameters` |
| `#example::2` | 2629:2931 | `Client.send() > Example` |
| `#notes::3` | 2932:3383 | `Client.send() > Notes` |

Ranges are contiguous, never overlapping — `704 → 705`, `2628 → 2629`.

> ⚠️ **The corpus has to be able to fail.** The first run of this comparison
> showed `0 vs 0` — every page was under 1000 characters, so A emitted one
> chunk per page and never had to cut anything. The parameter table was grown
> to sixteen rows (ordinary for an SDK page) before the damage appeared. A
> fixture that cannot expose a failure is worse than no fixture.

---

## 5. Chunk identity

```
v3/client-send#parameters::1
│  │           │          └── ordinal within the page
│  │           └───────────── heading anchor (B only; A has no idea)
│  └───────────────────────── page_id
└──────────────────────────── sdk_version
```

Readable on purpose — it tells you version, page, section and position without
a lookup. Every chunk also carries `char_start`/`char_end`, so you can slice the
source file and confirm the chunk really is what it claims.

All four required ingestion fields — `source_file`, `page_id`, `sdk_version`,
`page_type` — are copied onto every chunk. A chunk that can't say which page it
came from is useless no matter how well it matches a query.

---

## 6. The files

| File | Job |
|---|---|
| `app/rag/markdown.py` | **The load-bearing file.** Block parser; makes tables and fences atomic |
| `app/rag/chunkers/base.py` | Chunk building + damage detection |
| `app/rag/chunkers/fixed.py` | Strategy A |
| `app/rag/chunkers/structure.py` | Strategy B |
| `app/rag/chunk.py` | CLI |
| `app/rag/models.py` | The `Chunk` model |

Read `markdown.py` first — once atomic blocks make sense, `structure.py` is
obvious.

**Damage detection lives in `base.py`, not `markdown.py`, on purpose.**
*Measuring* whether a chunk got cut badly is a different job from *parsing*
structure, and a structure-blind chunker still has to be measurable.

---

## 7. Running it

```bash
python -m app.rag.chunk --compare              # both strategies side by side
python -m app.rag.chunk --strategy fixed --show # every chunk, in full
python -m app.rag.chunk --docs sample-corpus   # the messy corpus
python -m app.rag.chunk --version v3           # one SDK version
```

`--show` is the one that teaches. Look at `v3/client-send::3` under `fixed`:
it starts mid-row with no header and no method name.

### Branches

| Branch | Contents |
|---|---|
| `ragew3/chunk-a-fixed` | strategy A alone — 5 files, no `markdown.py` |
| `ragew3/chunk-b-structure` | strategy B alone — 6 files |
| `ragew3/chunk-ui` | both strategies together |

A and B live on separate branches so each can be tested in isolation. A has no
`markdown.py` at all — "A knows nothing about markdown" is true in the code,
not just in the comments.

---

## 8. What chunking cannot fix

Strategy B splits four of the five formats:

| Format | fixed | structure | |
|---|---|---|---|
| markdown | 5 | 4 | ✓ |
| html | 1 | 4 | ✓ |
| docx | 1 | 3 | ✓ |
| pdf | 1 | 1 | no headings exist |
| txt | 1 | 1 | no markup exists |

HTML and DOCX only work because **three ingestion bugs were fixed during this
phase**:

1. **HTML discarded heading levels** — `<h2>Parameters</h2>` arrived as the
   plain line `Parameters`. B saw no heading and had nothing to split on.
2. **DOCX discarded heading levels** — Word stores them as paragraph *styles*.
3. **DOCX read tables out of document order** — `document.paragraphs` and
   `document.tables` are two separate lists, so every table landed at the *end*
   of the text, detached from the heading it belonged under. Ingestion was
   orphaning the parameter table before any chunker touched it.

Bug 3 is the lesson of the phase. **No chunking strategy could have fixed it.**

PDF stays at one chunk and always will — a PDF stores positioned glyphs, not
headings. There is nothing to recover. The ceiling from the ingestion phase
still holds: *a chunker can only preserve structure that ingestion managed to
keep.*

---

## 9. Not built

- **Embedding** — no model chosen
- **Vector store** — nothing indexed
- **Retrieval / generation** — no search, no LLM
- **Tests** — removed by request; verification is running the CLI and reading it
- **UI** — a chunk viewer was built, used to find the three bugs above, then
  removed on request. The extractor fixes it exposed were kept.

---

## 10. Next phase — the question set

The chunkers are comparable but not yet *evaluated*. `split fences` and
`orphaned rows` measure structural damage, not whether retrieval finds the right
answer.

Next comes 8 questions with known answers, at least 3 of which depend on a
parameter-table row or a code fence — then hit-in-top-5 for each strategy over
the same questions. That's the number the assignment actually wants.

Two rules to carry in:

1. **Write the questions from the pages first**, before running any search. Do
   it the other way round and the hit-rate measures your question-writing.
2. **Freeze the embedding model across both runs.** Change the chunker and the
   embedder together and neither number means anything.
