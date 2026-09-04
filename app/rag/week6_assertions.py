"""Deterministic assertions over the Week 6 case set — no model in the loop.

    python -m app.rag.week6_assertions
    python -m app.rag.week6_assertions --detail
    python -m app.rag.week6_assertions --by-mode
    python -m app.rag.week6_assertions --json

Week 6 asks that criteria which can be decided by code are moved OUT of the
judge, so the judge is left with as little subjective surface as possible. Each
assertion here answers a yes/no question about a recorded answer using only the
answer text, the chunk ids that were in its context, and the corpus. No API key,
no network, no model, and no dependence on the judge that does not exist yet.

## Why these five

The Week 6 brief offers four candidate criteria. Two of them cannot be
implemented against this project without inventing data, and are NOT
implemented here:

- *every endpoint path mentioned exists in the OpenAPI spec* — the corpus
  documents a Python SDK (`Client.send()`), not an HTTP API. There is no
  endpoint path anywhere in docs/, and the FastAPI app's own /openapi.json
  describes this RAG service's routes, which are a different system from the one
  the answers talk about. Asserting one against the other is a category error.
- *no deprecated symbol without a migration note* — the corpus contains no
  deprecated symbols. `grep -rni "deprecat|superseded|removed in|migration"
  docs/` returns nothing, so there is no ground truth to assert against.

`code_sample_parses` and `version_stated` come from the brief's list. The other
three are derived from taxonomy.md, so that the assertions cover the modes this
system actually fails at rather than the ones a generic checklist expects:

    A1 citation_captured   -> mode 1 (50% of the Week 5 sample)
    A2 citation_resolves   -> mode 1 / hallucinated citations
    A3 code_sample_parses  -> brief's list
    A4 code_sample_grounded-> mode 6
    A5 version_stated      -> mode 2, brief's list

## The one thing the judge keeps

Everything above is mechanical. What survives is a single binary subjective
question — *is this answer correct and useful for the question asked, given the
context it was given* — which no amount of string matching decides. Phase 3
builds that; this module deliberately does not.

## Scope

Assertions run against `recorded.raw_output` and `recorded.retrieved` — the
verbatim transcript Phase 1 copied out of traces.jsonl. Cases with no recorded
transcript (the six golden-set probes) are reported as `skipped`, not as passes:
a probe has no answer to assert against until something generates one.
"""

import argparse
import ast
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.rag.index import DEFAULT_DOCS
from app.rag.week6_cases import DEFAULT_CASES_JSONL, Case, load_cases

# Characters that let a correct citation slip past a naive parser. Every one of
# these was observed in a real Week 5 trace, not imagined.
ZERO_WIDTH = "​‌‍﻿"
DASHES = "‐‑‒–—―−"

CHUNK_ID = re.compile(r"^v\d+/[a-z0-9-]+#[a-z0-9-]+::\d+$")
BRACKETED = re.compile(r"[\[【]([^\]】\n]{3,160})[\]】]")
FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.S)
INLINE_CODE = re.compile(r"`([^`\n]+)`")
# A call with at least one keyword argument. `Client.send()` in prose is not a
# code specimen; `client.send(payload, proxy="...")` is.
KWARG_CALL = re.compile(r"[A-Za-z_][\w.]*\s*\([^)]*=[^)]*\)")
VERSION_TOKEN = re.compile(r"\bv([23])\b|\bversion\s*([23])\b", re.I)
TABLE_ROW = re.compile(r"^\|\s*([a-z_][a-z0-9_]*)\s*\|\s*([a-z]+)\s*\|\s*([^|]*?)\s*\|", re.I)

PYTHON_LANGS = {"python", "py", ""}


def normalise(text: str) -> str:
    """Fold the ways a model can write the same citation and still be right.

    NFKC folds the full-width and narrow-no-break forms; zero-width characters
    are dropped; the seven unicode dashes become ASCII hyphen. Without this a
    correct citation reads as absent, which is exactly Week 5's mode 1.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = "".join(c for c in folded if c not in ZERO_WIDTH)
    for dash in DASHES:
        folded = folded.replace(dash, "-")
    return folded


def scan_citations(text: str) -> list[str]:
    """Every chunk id a human would read as a citation, however it is written.

    Deliberately more tolerant than `generate._CITATION`, which matches only
    ASCII brackets with no inner whitespace. The gap between the two is the
    measurement: what the model wrote versus what the pipeline captured.
    """
    found: list[str] = []
    for match in BRACKETED.finditer(normalise(text)):
        candidate = match.group(1).strip()
        if CHUNK_ID.match(candidate) and candidate not in found:
            found.append(candidate)
    return found


def prose_only(text: str) -> str:
    """The answer with code fences and bracketed citations removed.

    A version that appears only inside a chunk id is not a version the reader
    was told. `[v2/client-configure#parameters::1]` does not say "this is the v2
    value" to anyone who is copying the number out.
    """
    return BRACKETED.sub(" ", FENCE.sub(" ", normalise(text)))


def code_specimens(text: str) -> list[tuple[str, str]]:
    """(kind, body) for each block of code the answer presents as usage."""
    specimens: list[tuple[str, str]] = [
        ("fence", body) for _lang, body in FENCE.findall(text)
    ]
    for span in INLINE_CODE.findall(FENCE.sub("", text)):
        if KWARG_CALL.search(span):
            specimens.append(("inline", span))
    return specimens


def squash(code: str) -> str:
    """Whitespace- and comment-insensitive form, for grounding comparisons."""
    folded = unicodedata.normalize("NFKC", code)
    folded = re.sub(r"#.*$", "", folded, flags=re.M)
    return re.sub(r"\s+", "", folded)


def loose(text: str) -> str:
    """`retry_backoff_ms` and `retry backoff ms` are the same symbol to a reader."""
    return re.sub(r"[\s_]+", " ", text.lower())


# --------------------------------------------------------------------------
# corpus facts, derived — never typed
# --------------------------------------------------------------------------


def chunk_texts(docs_root: Path = DEFAULT_DOCS) -> dict[str, str]:
    from app.rag.chunkers.structure import StructureAwareChunker
    from app.rag.loader import load_documents

    chunker = StructureAwareChunker()
    return {
        chunk.chunk_id: chunk.text
        for doc in load_documents(docs_root).documents
        for chunk in chunker.chunk(doc)
    }


def divergent_defaults(docs_root: Path = DEFAULT_DOCS) -> dict[str, dict[str, list[str]]]:
    """Parameters documented with a DIFFERENT default in v2 than in v3.

    Parsed out of the corpus tables, so it tracks the docs rather than a list
    someone maintained by hand. These nine symbols are the ones for which an
    unversioned answer can ship a wrong number that still runs — which is
    exactly the harm taxonomy.md records for mode 2.
    """
    values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in sorted(docs_root.rglob("*.md")):
        version = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            match = TABLE_ROW.match(line)
            if not match or match.group(1).lower() == "name":
                continue
            default = match.group(3).strip()
            if default in ("—", "-", ""):
                continue
            values[match.group(1).lower()][version].add(default)

    divergent: dict[str, dict[str, list[str]]] = {}
    for name, by_version in values.items():
        flat = {v: sorted(s) for v, s in by_version.items()}
        if len(flat) > 1 and flat.get("v2") != flat.get("v3"):
            divergent[name] = flat
    return divergent


# --------------------------------------------------------------------------
# assertion results
# --------------------------------------------------------------------------

PASS, FAIL, SKIP, NA = "pass", "fail", "skipped", "n/a"


class AssertionResult(BaseModel):
    name: str
    status: str  # pass | fail | skipped | n/a
    detail: str = ""
    evidence: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    case_id: str
    mode: str
    mode_slug: str
    case_kind: str
    trace_id: str | None = None
    question: str
    results: list[AssertionResult]

    @property
    def failed(self) -> list[AssertionResult]:
        return [r for r in self.results if r.status == FAIL]


# --------------------------------------------------------------------------
# the assertions
# --------------------------------------------------------------------------


def assert_citation_captured(case: Case) -> AssertionResult:
    """A1 — every citation the model wrote was captured by the pipeline.

    PASS  every chunk id a tolerant scan finds in the answer text also appears
          in the recorded `citations` list (compared after normalisation).
          Vacuously passes when the answer cites nothing.
    FAIL  the answer contains at least one well-formed chunk id that the
          pipeline did not record — the citation exists for a reader and does
          not exist for the system. This is Week 5 mode 1.
    """
    name = "citation_captured"
    recorded = case.recorded
    if recorded is None:
        return AssertionResult(name=name, status=SKIP, detail="no recorded answer")

    written = scan_citations(recorded.raw_output or "")
    captured = {normalise(c).strip() for c in recorded.citations}
    lost = [c for c in written if c not in captured]
    if not written:
        return AssertionResult(name=name, status=PASS, detail="answer cites nothing")
    if lost:
        return AssertionResult(
            name=name,
            status=FAIL,
            detail=f"{len(lost)} of {len(written)} citation(s) written but not captured",
            evidence=lost,
        )
    return AssertionResult(
        name=name, status=PASS, detail=f"all {len(written)} citation(s) captured"
    )


def assert_citation_resolves(case: Case) -> AssertionResult:
    """A2 — every citation names a chunk that was actually in the context.

    PASS  each chunk id found in the answer is one of the retrieved chunk ids,
          AND no citation that does resolve is sitting in `invalid_citations`.
    FAIL  either a cited chunk was never in the context (a fabricated citation,
          which is worse than none because it looks verifiable), or a citation
          that does resolve was wrongly flagged invalid because of the
          characters it was written with.
    """
    name = "citation_resolves"
    recorded = case.recorded
    if recorded is None:
        return AssertionResult(name=name, status=SKIP, detail="no recorded answer")

    written = scan_citations(recorded.raw_output or "")
    context = {h["chunk_id"] for h in recorded.retrieved}
    unresolved = [c for c in written if c not in context]

    resolved = {c for c in written if c in context}
    false_invalid = [
        c for c in recorded.invalid_citations if normalise(c).strip() in resolved
    ]

    if unresolved:
        return AssertionResult(
            name=name,
            status=FAIL,
            detail=f"{len(unresolved)} citation(s) name a chunk not in the context",
            evidence=unresolved,
        )
    if false_invalid:
        return AssertionResult(
            name=name,
            status=FAIL,
            detail=f"{len(false_invalid)} valid citation(s) flagged invalid",
            evidence=false_invalid,
        )
    if not written:
        return AssertionResult(name=name, status=PASS, detail="answer cites nothing")
    return AssertionResult(
        name=name, status=PASS, detail=f"all {len(written)} citation(s) resolve"
    )


def assert_code_sample_parses(case: Case) -> AssertionResult:
    """A3 — every Python code block in the answer is syntactically valid.

    PASS  `ast.parse` accepts every fenced python block. Vacuously passes when
          the answer shows no fenced code.
    FAIL  any fenced python block raises SyntaxError — the answer hands the
          reader something that cannot run.

    Non-python fences are not asserted; there is no parser for an arbitrary
    language and guessing one would produce false failures.
    """
    name = "code_sample_parses"
    recorded = case.recorded
    if recorded is None:
        return AssertionResult(name=name, status=SKIP, detail="no recorded answer")

    fences = FENCE.findall(recorded.raw_output or "")
    python = [(lang, body) for lang, body in fences if lang.lower() in PYTHON_LANGS]
    if not python:
        return AssertionResult(name=name, status=NA, detail="no python fence in answer")

    broken: list[str] = []
    for _lang, body in python:
        try:
            ast.parse(body)
        except SyntaxError as exc:
            broken.append(f"line {exc.lineno}: {exc.msg}")
    if broken:
        return AssertionResult(
            name=name,
            status=FAIL,
            detail=f"{len(broken)} of {len(python)} python fence(s) do not parse",
            evidence=broken,
        )
    return AssertionResult(
        name=name, status=PASS, detail=f"{len(python)} python fence(s) parse"
    )


def assert_code_sample_grounded(
    case: Case, texts: dict[str, str]
) -> AssertionResult:
    """A4 — code the answer shows appears in a chunk that was retrieved.

    PASS  every non-comment line of every code specimen appears, ignoring
          whitespace, in the concatenated text of the retrieved chunks.
    FAIL  a specimen contains a line present in no retrieved chunk — the model
          composed usage the docs never show. Week 5 mode 6.

    A specimen is a fenced block, or an inline span holding a call with at least
    one keyword argument. Bare `Client.send()` mentions in prose are not
    specimens, so ordinary writing does not trip this.
    """
    name = "code_sample_grounded"
    recorded = case.recorded
    if recorded is None:
        return AssertionResult(name=name, status=SKIP, detail="no recorded answer")

    specimens = code_specimens(recorded.raw_output or "")
    if not specimens:
        return AssertionResult(name=name, status=NA, detail="answer shows no code")

    context = squash(" ".join(texts.get(h["chunk_id"], "") for h in recorded.retrieved))
    ungrounded: list[str] = []
    for kind, body in specimens:
        lines = (
            [l for l in body.splitlines() if l.strip() and not l.strip().startswith("#")]
            if kind == "fence"
            else [body]
        )
        for line in lines:
            squashed = squash(line)
            if squashed and squashed not in context:
                ungrounded.append(line.strip())
    if ungrounded:
        return AssertionResult(
            name=name,
            status=FAIL,
            detail=f"{len(ungrounded)} code line(s) appear in no retrieved chunk",
            evidence=ungrounded,
        )
    return AssertionResult(
        name=name,
        status=PASS,
        detail=f"{len(specimens)} specimen(s) fully grounded in the context",
    )


def assert_version_stated(
    case: Case, divergent: dict[str, dict[str, list[str]]]
) -> AssertionResult:
    """A5 — an answer quoting a version-divergent value says which version.

    PASS  the answer names a version (v2 / v3 / "version 2" / "version 3") in
          its prose — outside citation brackets and outside code, because a
          version buried in a chunk id tells the reader nothing.
    FAIL  the answer states a documented default for a parameter whose default
          differs between v2 and v3, and never names a version. That is a number
          which runs and is wrong — the harm taxonomy.md assigns to mode 2.
    n/a   the answer refused, or quotes no divergent value at all.

    The nine divergent symbols are parsed from the corpus tables at runtime, so
    this assertion follows the docs rather than a hand-kept list.
    """
    name = "version_stated"
    recorded = case.recorded
    if recorded is None:
        return AssertionResult(name=name, status=SKIP, detail="no recorded answer")
    if recorded.refused:
        return AssertionResult(name=name, status=NA, detail="answer refused")

    text = prose_only(recorded.raw_output or "")
    loose_text = loose(text)
    triggered: list[str] = []
    for symbol, by_version in divergent.items():
        if loose(symbol) not in loose_text:
            continue
        values = [v for vs in by_version.values() for v in vs]
        if any(
            re.search(rf"(?<![\w.]){re.escape(v)}(?![\w.])", text, re.I) for v in values
        ):
            triggered.append(symbol)

    if not triggered:
        return AssertionResult(
            name=name, status=NA, detail="answer quotes no version-divergent value"
        )
    if VERSION_TOKEN.search(text):
        return AssertionResult(
            name=name,
            status=PASS,
            detail=f"names a version while quoting {', '.join(sorted(triggered))}",
        )
    return AssertionResult(
        name=name,
        status=FAIL,
        detail=(
            f"quotes a v2/v3-divergent default for "
            f"{', '.join(sorted(triggered))} without naming a version"
        ),
        evidence=[
            f"{s}: v2={divergent[s].get('v2')} v3={divergent[s].get('v3')}"
            for s in sorted(triggered)
        ],
    )


ASSERTION_NAMES = [
    "citation_captured",
    "citation_resolves",
    "code_sample_parses",
    "code_sample_grounded",
    "version_stated",
]


def run_case(
    case: Case, texts: dict[str, str], divergent: dict[str, dict[str, list[str]]]
) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        mode=case.mode,
        mode_slug=case.mode_slug,
        case_kind=case.case_kind,
        trace_id=case.trace_id,
        question=case.question,
        results=[
            assert_citation_captured(case),
            assert_citation_resolves(case),
            assert_code_sample_parses(case),
            assert_code_sample_grounded(case, texts),
            assert_version_stated(case, divergent),
        ],
    )


def run_all(cases: list[Case], docs_root: Path = DEFAULT_DOCS) -> list[CaseResult]:
    texts = chunk_texts(docs_root)
    divergent = divergent_defaults(docs_root)
    return [run_case(c, texts, divergent) for c in cases]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_matrix(results: list[CaseResult]) -> None:
    glyph = {PASS: ".", FAIL: "F", SKIP: "-", NA: " "}
    short = ["capt", "resv", "prse", "grnd", "vers"]
    header = f"{'case':8} {'mode':6} {'kind':17} " + " ".join(f"{s:4}" for s in short)
    print(header)
    print("-" * len(header))
    for result in results:
        row = f"{result.case_id:8} {result.mode:6} {result.case_kind:17} "
        row += " ".join(f"{glyph[r.status]:4}" for r in result.results)
        print(row)
    print("-" * len(header))
    print("  . pass    F fail    - skipped (no recorded answer)    (blank) n/a")


def print_totals(results: list[CaseResult]) -> None:
    print()
    header = f"{'assertion':24} {'pass':>5} {'fail':>5} {'n/a':>5} {'skip':>5} {'ran':>5}"
    print(header)
    print("-" * len(header))
    for index, name in enumerate(ASSERTION_NAMES):
        counts = defaultdict(int)
        for result in results:
            counts[result.results[index].status] += 1
        ran = counts[PASS] + counts[FAIL]
        print(
            f"{name:24} {counts[PASS]:5} {counts[FAIL]:5} "
            f"{counts[NA]:5} {counts[SKIP]:5} {ran:5}"
        )
    print("-" * len(header))
    total_fail = sum(len(r.failed) for r in results)
    cases_with_fail = sum(1 for r in results if r.failed)
    print(f"{'TOTAL':24} {'':5} {total_fail:5}")
    print()
    print(f"cases evaluated        : {len(results)}")
    print(f"cases with >=1 failure : {cases_with_fail}")
    print(f"assertion failures     : {total_fail}")


def print_by_mode(results: list[CaseResult]) -> None:
    print()
    modes = sorted({r.mode for r in results}, key=lambda m: (m == "clean", m))
    header = f"{'mode':6} {'slug':26} {'cases':>5} {'ran':>5} {'fail':>5} {'clean':>6}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        rows = [r for r in results if r.mode == mode]
        slug = rows[0].mode_slug
        ran = sum(
            1 for r in rows for a in r.results if a.status in (PASS, FAIL)
        )
        failed = sum(len(r.failed) for r in rows)
        clean_cases = sum(1 for r in rows if not r.failed)
        print(
            f"{mode:6} {slug:26} {len(rows):5} {ran:5} {failed:5} "
            f"{clean_cases:5}/{len(rows)}"
        )


def print_detail(results: list[CaseResult]) -> None:
    print()
    print("=== failures, with evidence ===")
    for result in results:
        if not result.failed:
            continue
        print()
        trace = f"  trace {result.trace_id}" if result.trace_id else ""
        print(f"{result.case_id}  mode {result.mode} ({result.mode_slug}){trace}")
        print(f"  question: {result.question!r}")
        for failure in result.failed:
            print(f"  FAIL {failure.name}: {failure.detail}")
            for item in failure.evidence:
                print(f"       - {item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic assertions over the Week 6 case set."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--detail", action="store_true", help="per-failure evidence")
    parser.add_argument("--by-mode", action="store_true", help="failures per taxonomy mode")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any assertion fails (for CI; the default reports and exits 0, "
        "because on this dataset failures are the expected finding)",
    )
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases)
        results = run_all(cases, args.docs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"assertions failed to run: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.model_dump() for r in results], indent=2))
        return 0

    divergent = divergent_defaults(args.docs)
    print(f"{len(cases)} cases from {args.cases}")
    print(f"{len(ASSERTION_NAMES)} deterministic assertions, 0 model calls")
    print(f"version-divergent symbols derived from corpus: {len(divergent)}")
    print()
    print_matrix(results)
    print_totals(results)
    if args.by_mode:
        print_by_mode(results)
    if args.detail:
        print_detail(results)

    total_fail = sum(len(r.failed) for r in results)
    if args.strict and total_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
