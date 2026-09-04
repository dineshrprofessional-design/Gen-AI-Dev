"""Build the Week 6 evaluation set from Week 5 evidence.

    python -m app.rag.week6_cases --out eval/week6_cases.jsonl
    python -m app.rag.week6_cases --check       # re-derive and diff the committed file
    python -m app.rag.week6_cases --validate    # sanity checks only
    python -m app.rag.week6_cases --by-mode     # coverage table

`eval/week6_cases.yaml` is the hand-authored source: which cases exist, which
Week 5 taxonomy mode each belongs to, and what a correct answer must contain.
`eval/week6_cases.jsonl` is the *generated* artifact.

## Why the transcripts are copied, not typed

Week 6 asks for regression cases taken verbatim from real traces, and forbids
inventing trace ids. Both are enforced structurally rather than by care:

- the YAML names a `trace_id` and nothing else about the trace. Question,
  retrieval config, retrieved chunk_ids with scores, raw output, refusal state
  and citations are read out of `traces/traces.jsonl` at build time;
- an id that is not in the trace file raises `ValueError` here, so a fabricated
  provenance cannot reach the dataset;
- `trace_kind` (organic | demo) is cross-checked against
  `traces/pool_manifest.json` rather than asserted in the YAML.

The same reasoning as `app/rag/golden_set.py`, which derives `gold_chunk_id` by
re-running the chunker instead of letting anyone type one: a dataset whose
provenance fields are hand-maintained drifts silently, and a drifted provenance
field is worse than none because downstream analysis trusts it.

## What this module deliberately does not do

No judging, no labelling, no scoring. It emits cases and checks their integrity.
Every field an LLM judge would later need is present (`mode`, `case_kind`,
`expected_behavior`, `must_contain`, `must_not_contain`, `expected_citations`),
but nothing here reads a model's output and forms an opinion about it.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.rag.golden_set import DEFAULT_GOLDEN_JSONL, load_jsonl
from app.rag.index import DEFAULT_DOCS
from app.rag.trace import TRACE_FILE, load_all

DEFAULT_CASES_YAML = Path("eval/week6_cases.yaml")
DEFAULT_CASES_JSONL = Path("eval/week6_cases.jsonl")
DEFAULT_MANIFEST = Path("traces/pool_manifest.json")

MIN_CASES = 25
MIN_REGRESSIONS = 2
REQUIRED_MODES = {"1", "2", "3", "4", "5", "6"}


class ModeSpec(BaseModel):
    slug: str
    name: str
    severity: str
    week5_count: int
    week5_pct: float
    week5_example: str


class CaseSpec(BaseModel):
    """One hand-authored case. Mirrors a `cases:` entry in the YAML."""

    case_id: str
    source: str  # trace | golden_set
    mode: str
    case_kind: str  # regression | regression-clean | probe
    expected_behavior: str  # answer | refuse

    trace_id: str | None = None
    golden_id: str | None = None

    sdk_version: str = "none"
    depends_on: str = ""
    expected_answer: str = ""
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    gold_chunk_id: str = ""
    observed_failure: str = ""
    rationale: str = ""


class TaxonomySpec(BaseModel):
    source: str
    sample: str
    modes: dict[str, ModeSpec]


class CaseFile(BaseModel):
    taxonomy: TaxonomySpec
    cases: list[CaseSpec]


class RecordedTrace(BaseModel):
    """The verbatim transcript of a regression case's original request.

    Copied field-for-field out of traces.jsonl. Nothing here is authored.
    """

    trace_id: str
    ts: str
    git_sha: str
    prompt_version: str
    model: str
    params: dict[str, Any]
    retrieval: dict[str, Any]
    retrieved: list[dict[str, Any]]
    raw_output: str | None
    refused: bool
    refusal_reason: str
    citations: list[str]
    invalid_citations: list[str]
    generation_ran: bool
    retrieval_ms: float


class Case(BaseModel):
    """One row of eval/week6_cases.jsonl."""

    case_id: str
    question: str
    mode: str
    mode_slug: str
    mode_name: str
    severity: str
    case_kind: str
    is_regression: bool
    source: str
    provenance: str
    trace_id: str | None = None
    trace_kind: str | None = None
    golden_id: str | None = None

    sdk_version: str
    depends_on: str
    expected_behavior: str
    expected_answer: str
    must_contain: list[str]
    must_not_contain: list[str]
    expected_citations: list[str]
    gold_chunk_id: str

    observed_failure: str = ""
    rationale: str = ""
    recorded: RecordedTrace | None = None


def load_yaml(path: Path = DEFAULT_CASES_YAML) -> CaseFile:
    if not path.is_file():
        raise FileNotFoundError(f"case specs not found: {path}")
    return CaseFile(**yaml.safe_load(path.read_text(encoding="utf-8")))


def trace_kinds(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, str]:
    """trace_id -> organic | demo, from the Week 5 pool manifest."""
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kinds: dict[str, str] = {}
    for bucket in ("organic", "demo"):
        for row in manifest.get(bucket, []):
            if row.get("trace_id"):
                kinds[row["trace_id"]] = bucket
    return kinds


def build(
    spec_file: CaseFile,
    trace_path: Path = TRACE_FILE,
    golden_path: Path = DEFAULT_GOLDEN_JSONL,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> list[Case]:
    traces = {r["trace_id"]: r for r in load_all(trace_path)}
    golden = {row.id: row for row in load_jsonl(golden_path)}
    kinds = trace_kinds(manifest_path)

    cases: list[Case] = []
    for spec in spec_file.cases:
        mode = spec_file.taxonomy.modes.get(spec.mode)
        if mode is None:
            raise ValueError(
                f"{spec.case_id}: mode {spec.mode!r} is not in the taxonomy block. "
                f"Known: {sorted(spec_file.taxonomy.modes)}"
            )

        common = dict(
            case_id=spec.case_id,
            mode=spec.mode,
            mode_slug=mode.slug,
            mode_name=" ".join(mode.name.split()),
            severity=mode.severity,
            case_kind=spec.case_kind,
            is_regression=spec.case_kind.startswith("regression"),
            source=spec.source,
            sdk_version=spec.sdk_version,
            expected_behavior=spec.expected_behavior,
            expected_answer=" ".join(spec.expected_answer.split()),
            must_contain=spec.must_contain,
            must_not_contain=spec.must_not_contain,
            expected_citations=spec.expected_citations,
            observed_failure=" ".join(spec.observed_failure.split()),
            rationale=" ".join(spec.rationale.split()),
        )

        if spec.source == "trace":
            if not spec.trace_id:
                raise ValueError(f"{spec.case_id}: source is 'trace' but no trace_id")
            record = traces.get(spec.trace_id)
            if record is None:
                # The guard the Week 6 brief asks for: no invented provenance.
                raise ValueError(
                    f"{spec.case_id}: trace_id {spec.trace_id!r} is not in "
                    f"{trace_path}. A case may not claim a trace that does not exist."
                )
            cases.append(
                Case(
                    **common,
                    question=record["question"],  # verbatim, never typed
                    provenance=f"{trace_path}#{spec.trace_id}",
                    trace_id=spec.trace_id,
                    trace_kind=kinds.get(spec.trace_id),
                    depends_on=spec.depends_on,
                    gold_chunk_id=spec.gold_chunk_id,
                    recorded=RecordedTrace(
                        trace_id=record["trace_id"],
                        ts=record["ts"],
                        git_sha=record["git_sha"],
                        prompt_version=record["prompt_version"],
                        model=record["model"],
                        params=record["params"],
                        retrieval=record["retrieval"],
                        retrieved=record["retrieved"],
                        raw_output=record["raw_output"],
                        refused=record["refused"],
                        refusal_reason=record["refusal_reason"],
                        citations=record["citations"],
                        invalid_citations=record["invalid_citations"],
                        generation_ran=record["generation_ran"],
                        retrieval_ms=record["retrieval_ms"],
                    ),
                )
            )

        elif spec.source == "golden_set":
            if not spec.golden_id:
                raise ValueError(
                    f"{spec.case_id}: source is 'golden_set' but no golden_id"
                )
            row = golden.get(spec.golden_id)
            if row is None:
                raise ValueError(
                    f"{spec.case_id}: golden_id {spec.golden_id!r} is not in "
                    f"{golden_path}. Known: {sorted(golden)}"
                )
            fields = dict(common)
            # The golden set's own answer is authoritative; keep it when the
            # spec did not write one of its own.
            if not spec.expected_answer:
                fields["expected_answer"] = row.answer
            cases.append(
                Case(
                    **fields,
                    question=row.question,  # verbatim
                    provenance=f"{golden_path}#{spec.golden_id}",
                    golden_id=spec.golden_id,
                    depends_on=spec.depends_on or row.depends_on,
                    gold_chunk_id=spec.gold_chunk_id or row.gold_chunk_id,
                )
            )
        else:
            raise ValueError(
                f"{spec.case_id}: unknown source {spec.source!r} "
                "(expected 'trace' or 'golden_set')"
            )

    return cases


def dump_jsonl(cases: list[Case]) -> str:
    return "".join(
        json.dumps(c.model_dump(), ensure_ascii=False) + "\n" for c in cases
    )


def load_cases(path: Path = DEFAULT_CASES_JSONL) -> list[Case]:
    if not path.is_file():
        raise FileNotFoundError(f"case set not found: {path}")
    text = path.read_text(encoding="utf-8")
    return [Case(**json.loads(line)) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def corpus_chunk_ids(docs_root: Path = DEFAULT_DOCS) -> set[str]:
    """Every chunk_id the structure chunker emits, for citation resolution."""
    from app.rag.chunkers.structure import StructureAwareChunker
    from app.rag.loader import load_documents

    chunker = StructureAwareChunker()
    documents = load_documents(docs_root).documents
    return {c.chunk_id for d in documents for c in chunker.chunk(d)}


def validate(
    cases: list[Case],
    spec_file: CaseFile,
    trace_path: Path = TRACE_FILE,
    docs_root: Path = DEFAULT_DOCS,
) -> tuple[list[str], list[str]]:
    """Return (failures, warnings). Failures make the command exit 1."""
    failures: list[str] = []
    warnings: list[str] = []

    # 1. size
    if len(cases) < MIN_CASES:
        failures.append(f"only {len(cases)} cases; Week 6 requires >= {MIN_CASES}")

    # 2. unique ids
    dupes = [i for i, n in Counter(c.case_id for c in cases).items() if n > 1]
    if dupes:
        failures.append(f"duplicate case_ids: {dupes}")

    # 3. every case carries a taxonomy tag
    untagged = [c.case_id for c in cases if not c.mode or not c.mode_slug]
    if untagged:
        failures.append(f"cases with no taxonomy mode: {untagged}")

    # 4. the six failure modes are all covered
    present = {c.mode for c in cases}
    missing = REQUIRED_MODES - present
    if missing:
        failures.append(f"taxonomy modes with no case: {sorted(missing)}")
    if "clean" not in present:
        warnings.append(
            "no `clean` control cases — a pass rate with only failures in it "
            "cannot distinguish a good judge from one that flags everything"
        )

    # 5. enough regression cases, and every trace id is real
    regressions = [c for c in cases if c.is_regression and c.trace_id]
    if len(regressions) < MIN_REGRESSIONS:
        failures.append(
            f"only {len(regressions)} trace-backed regression cases; "
            f"Week 6 requires >= {MIN_REGRESSIONS}"
        )

    traces = {r["trace_id"]: r for r in load_all(trace_path)}
    for case in cases:
        if case.trace_id and case.trace_id not in traces:
            failures.append(
                f"{case.case_id}: trace_id {case.trace_id!r} not in {trace_path}"
            )

    # 6. the question and transcript really are verbatim
    for case in cases:
        if not case.trace_id or case.trace_id not in traces:
            continue
        record = traces[case.trace_id]
        if case.question != record["question"]:
            failures.append(
                f"{case.case_id}: question differs from trace {case.trace_id}"
            )
        if case.recorded is None:
            failures.append(f"{case.case_id}: regression case has no recorded block")
            continue
        if case.recorded.raw_output != record["raw_output"]:
            failures.append(
                f"{case.case_id}: recorded raw_output differs from trace"
            )
        if [h["chunk_id"] for h in case.recorded.retrieved] != [
            h["chunk_id"] for h in record["retrieved"]
        ]:
            failures.append(
                f"{case.case_id}: recorded context differs from trace"
            )

    # 7. citations and gold chunks resolve against the real corpus
    try:
        known = corpus_chunk_ids(docs_root)
    except Exception as exc:  # noqa: BLE001 - corpus is optional for --validate
        known = set()
        warnings.append(f"could not chunk the corpus to resolve citations: {exc}")
    if known:
        for case in cases:
            for cid in case.expected_citations:
                if cid not in known:
                    failures.append(
                        f"{case.case_id}: expected citation {cid!r} is not a chunk "
                        "the structure chunker emits"
                    )
            if case.gold_chunk_id and case.gold_chunk_id not in known:
                failures.append(
                    f"{case.case_id}: gold_chunk_id {case.gold_chunk_id!r} not in corpus"
                )

    # 8. shape rules
    for case in cases:
        if case.expected_behavior not in ("answer", "refuse"):
            failures.append(
                f"{case.case_id}: expected_behavior {case.expected_behavior!r}"
            )
        if case.case_kind not in ("regression", "regression-clean", "probe"):
            failures.append(f"{case.case_id}: case_kind {case.case_kind!r}")
        if case.case_kind == "regression" and not case.observed_failure:
            failures.append(
                f"{case.case_id}: a regression case must say what went wrong"
            )
        if case.case_kind == "probe" and case.trace_id:
            failures.append(
                f"{case.case_id}: a probe must not claim a trace_id "
                "(it has no observed failure behind it)"
            )

    # 9. does the recorded transcript actually exhibit the failure it claims?
    #    A warning, not a failure: some modes (4, 5) show up as an absence.
    for case in cases:
        if case.case_kind != "regression" or case.recorded is None:
            continue
        body = case.recorded.raw_output or ""
        hit = any(n and n in body for n in case.must_not_contain)
        lost = case.mode == "1" and not case.recorded.citations
        flagged = case.mode == "1" and case.recorded.invalid_citations
        refused = case.recorded.refused
        if not (hit or lost or flagged or refused):
            warnings.append(
                f"{case.case_id}: mode {case.mode} claimed, but the recorded "
                "transcript shows no refusal, no lost/invalid citation and no "
                "must_not_contain hit — check the tag by hand"
            )

    # 10. trace_kind agrees with the manifest
    kinds = trace_kinds()
    for case in cases:
        if case.trace_id and kinds and case.trace_kind != kinds.get(case.trace_id):
            failures.append(
                f"{case.case_id}: trace_kind {case.trace_kind!r} disagrees with "
                f"the manifest ({kinds.get(case.trace_id)!r})"
            )

    # 11. the taxonomy block matches taxonomy.md's headline counts
    declared = sum(
        m.week5_count for k, m in spec_file.taxonomy.modes.items() if k != "clean"
    )
    if declared != 19:
        warnings.append(
            f"taxonomy mode counts sum to {declared}; taxonomy.md records 19 "
            "mode-instances across 20 traces (a trace can carry more than one)"
        )

    return failures, warnings


def print_by_mode(cases: list[Case], spec_file: CaseFile) -> None:
    counts = Counter(c.mode for c in cases)
    kinds = Counter((c.mode, c.case_kind) for c in cases)
    order = ["1", "2", "3", "4", "5", "6", "clean"]

    header = (
        f"{'mode':6} {'slug':26} {'sev':18} {'cases':>5} "
        f"{'regr':>5} {'clean':>5} {'probe':>5}"
    )
    print(header)
    print("-" * len(header))
    for mode in order:
        spec = spec_file.taxonomy.modes.get(mode)
        if spec is None:
            continue
        print(
            f"{mode:6} {spec.slug:26} {spec.severity:18} {counts.get(mode, 0):5} "
            f"{kinds.get((mode, 'regression'), 0):5} "
            f"{kinds.get((mode, 'regression-clean'), 0):5} "
            f"{kinds.get((mode, 'probe'), 0):5}"
        )
    print("-" * len(header))
    total_kinds = Counter(c.case_kind for c in cases)
    print(
        f"{'TOTAL':6} {'':26} {'':18} {len(cases):5} "
        f"{total_kinds.get('regression', 0):5} "
        f"{total_kinds.get('regression-clean', 0):5} "
        f"{total_kinds.get('probe', 0):5}"
    )
    print()
    traced = [c for c in cases if c.trace_id]
    print(f"trace-backed cases : {len(traced)}  "
          f"(organic {sum(c.trace_kind == 'organic' for c in traced)}, "
          f"demo {sum(c.trace_kind == 'demo' for c in traced)})")
    print(f"golden-set reuse   : {sum(c.source == 'golden_set' for c in cases)}")
    print(f"expected refusals  : {sum(c.expected_behavior == 'refuse' for c in cases)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Week 6 evaluation set.")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_CASES_YAML)
    parser.add_argument("--out", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--traces", type=Path, default=TRACE_FILE)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_JSONL)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive and diff against the committed jsonl instead of writing",
    )
    parser.add_argument("--validate", action="store_true", help="sanity checks only")
    parser.add_argument("--by-mode", action="store_true", help="coverage table")
    args = parser.parse_args(argv)

    try:
        spec_file = load_yaml(args.yaml)
        cases = build(spec_file, args.traces, args.golden)
    except (FileNotFoundError, ValueError) as exc:
        print(f"week6 case build failed: {exc}", file=sys.stderr)
        return 1

    if args.by_mode:
        print_by_mode(cases, spec_file)
        return 0

    if args.validate:
        failures, warnings = validate(cases, spec_file, args.traces, args.docs)
        print(f"{len(cases)} cases from {args.yaml}")
        print()
        print_by_mode(cases, spec_file)
        print()
        for warning in warnings:
            print(f"WARN  {warning}")
        for failure in failures:
            print(f"FAIL  {failure}")
        print()
        if failures:
            print(f"VALIDATION FAILED — {len(failures)} problem(s)")
            return 1
        print(f"VALIDATION PASSED — {len(cases)} cases, {len(warnings)} warning(s)")
        return 0

    rendered = dump_jsonl(cases)

    if args.check:
        if not args.out.is_file():
            print(f"{args.out} does not exist; run without --check", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != rendered:
            print(
                f"{args.out} is stale — a fresh build from {args.yaml} differs. "
                "Re-run without --check.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} matches a fresh build ({len(cases)} cases)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}  ({len(cases)} cases)")
    print()
    print_by_mode(cases, spec_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
