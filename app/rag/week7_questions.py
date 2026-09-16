"""The Week 7 question set, and the T2 answerability matrix.

    python -m app.rag.week7_questions --out eval/week7_questions.jsonl
    python -m app.rag.week7_questions --check       # re-derive and byte-diff
    python -m app.rag.week7_questions --validate    # class mix, metadata, corpus refs, T2
    python -m app.rag.week7_questions --matrix      # the T2 table
    python -m app.rag.week7_questions --deps        # the proved dependency chains

`eval/week7_questions.yaml` is the hand-authored source; the `.jsonl` is derived,
following the repo norm set in `app/rag/golden_set.py` — the YAML carries what a
human decided, and anything resolvable from the corpus is computed so it cannot
drift from what the tools actually hold.

## What T2 is, and why it is computed rather than declared

T2 asks, for each question and each tool *alone*: giving that tool its best shot
over the whole argument space, does its output contain every literal the question
needs? The matrix is produced by **really calling the tools** and string-matching
their real payloads. Nothing in it is asserted by hand.

That matters because the claim being tested — "four questions cannot be answered
by any single tool" — is the justification for running an agent at all. A
hand-written matrix would make that claim unfalsifiable, which is exactly the
failure the Week 7 brief warns about when it says a verdict contradicting your own
numbers scores zero.

"Best shot" is defined generously on purpose, so a NO is meaningful:

  search_docs        the question text, plus every required_fact used as its own
                     query, against BOTH api_versions — 2 x (1 + n) calls
  get_openapi_spec   all 6 SDK methods x 2 versions, plus all 9 authored paths
                     x 2 versions
  check_deprecation  the same enumeration

If a tool cannot surface a literal under that, it genuinely cannot answer.

## The two literal fields

`required_facts` are matched against tool OUTPUT and drive T2. `must_contain` are
matched against the final ANSWER and drive scoring. They are deliberately
different: `"retired": false` is a fact only one tool reports and no answer text
would contain verbatim, while an answer must name `send_batch` in prose, which is
not the discriminating fact a tool returns. See the YAML header.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.rag.index import DEFAULT_DOCS
from app.rag.week7_corpus import API_DIR, OPENAPI_JSON, load_endpoints, load_pages
from app.rag.week7_tools import (
    API_VERSIONS,
    IMPLEMENTATIONS,
    SDK_METHODS,
    TOOL_NAMES,
    ToolConfig,
    ToolContext,
    _openapi,
)

DEFAULT_QUESTIONS_YAML = Path("eval/week7_questions.yaml")
DEFAULT_QUESTIONS_JSONL = Path("eval/week7_questions.jsonl")

REQUIRED_MIX = {
    "single_hop": 3,
    "cross_version": 2,
    "two_hop": 4,
    "refusal": 1,
}
MIN_MULTI_TOOL = 4
MIN_DEPENDENT = 3


class Week7Question(BaseModel):
    id: str
    question: str
    question_class: str
    sdk_version: str
    expect_refusal: bool = False
    required_facts: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    expected_order: list[str] = Field(default_factory=list)
    dependency: str = ""
    rationale: str = ""

    @property
    def is_dependent(self) -> bool:
        """Later step's ARGUMENTS come from an earlier step's RESULT."""
        return self.question_class == "two_hop" and bool(self.dependency.strip())


def _squash(text: str) -> str:
    return " ".join(text.split())


def load_yaml(path: Path = DEFAULT_QUESTIONS_YAML) -> list[Week7Question]:
    if not path.is_file():
        raise FileNotFoundError(f"question set not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: list[Week7Question] = []
    for entry in payload["questions"]:
        entry = dict(entry)
        entry["dependency"] = _squash(entry.get("dependency", ""))
        entry["rationale"] = _squash(entry.get("rationale", ""))
        rows.append(Week7Question(**entry))
    return rows


def dump_jsonl(rows: list[Week7Question]) -> str:
    return "".join(
        json.dumps(row.model_dump(), ensure_ascii=False) + "\n" for row in rows
    )


def load_questions(path: Path = DEFAULT_QUESTIONS_JSONL) -> list[Week7Question]:
    if not path.is_file():
        raise FileNotFoundError(f"question set not found: {path}")
    return [
        Week7Question(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# T2 — computed from real tool output
# --------------------------------------------------------------------------


def _strip_echo(payload: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is just an argument handed back.

    Without this, T2 is trivially satisfiable: `check_deprecation`'s not-found
    branch echoes the `path` it was given, so enumerating every authored path
    made the tool appear to "supply" any path literal a question mentions. A
    tool repeating its own input has told you nothing, and counting it would let
    any tool answer any question whose literals appear in its arguments.
    """
    echoed = {value for value in arguments.values() if isinstance(value, str)}
    return {
        key: value
        for key, value in payload.items()
        if not (isinstance(value, str) and value in echoed)
    }


def _tool_corpus(name: str, question: Week7Question, cfg: ToolConfig) -> str:
    """Everything this tool can produce for this question, JSON-serialised.

    Each tool gets its best shot over the argument space it accepts, minus any
    value it merely echoed back. A NO from this is a real capability statement,
    not a sampling artefact.
    """
    ctx = ToolContext(cfg=cfg)
    payloads: list[dict[str, Any]] = []

    def run(**arguments: Any) -> None:
        payloads.append(_strip_echo(IMPLEMENTATIONS[name](ctx, **arguments), arguments))

    if name == "search_docs":
        queries = [question.question] + [f for f in question.required_facts if f.strip()]
        for query in queries:
            for version in API_VERSIONS:
                run(query=query, api_version=version)
    else:
        spec_paths = list(_openapi(str(OPENAPI_JSON))["paths"])
        for version in API_VERSIONS:
            for method in SDK_METHODS:
                run(sdk_method=method, api_version=version)
            for path in spec_paths:
                run(path=path, api_version=version)

    return json.dumps(payloads, ensure_ascii=False)


class ToolVerdict(BaseModel):
    tool: str
    answers_alone: bool
    found: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class MatrixRow(BaseModel):
    id: str
    question: str
    question_class: str
    required_facts: list[str]
    verdicts: list[ToolVerdict]
    expected_order: list[str]
    dependency: str

    @property
    def single_tool_answers(self) -> list[str]:
        return [v.tool for v in self.verdicts if v.answers_alone]

    @property
    def min_tools(self) -> int:
        """1 if some tool answers alone; else the authored expectation."""
        if self.single_tool_answers:
            return 1
        return max(len(set(self.expected_order)), 2) if self.expected_order else 0

    @property
    def needs_multiple(self) -> bool:
        return not self.single_tool_answers and self.question_class != "refusal"


def build_matrix(
    rows: list[Week7Question], cfg: ToolConfig | None = None
) -> list[MatrixRow]:
    cfg = cfg or ToolConfig()
    matrix: list[MatrixRow] = []
    for question in rows:
        verdicts: list[ToolVerdict] = []
        for name in TOOL_NAMES:
            blob = _tool_corpus(name, question, cfg)
            found = [fact for fact in question.required_facts if fact in blob]
            missing = [fact for fact in question.required_facts if fact not in blob]
            verdicts.append(
                ToolVerdict(
                    tool=name,
                    # A question with no required facts (the refusal control) is
                    # not "answerable" by anything - there is nothing to retrieve.
                    answers_alone=bool(question.required_facts) and not missing,
                    found=found,
                    missing=missing,
                )
            )
        matrix.append(
            MatrixRow(
                id=question.id,
                question=question.question,
                question_class=question.question_class,
                required_facts=question.required_facts,
                verdicts=verdicts,
                expected_order=question.expected_order,
                dependency=question.dependency,
            )
        )
    return matrix


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate(
    rows: list[Week7Question], matrix: list[MatrixRow], docs_root: Path = DEFAULT_DOCS
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    # 1. exactly ten
    if len(rows) != 10:
        failures.append(f"expected exactly 10 questions, found {len(rows)}")

    # 2. class distribution
    counts = Counter(row.question_class for row in rows)
    for klass, expected in REQUIRED_MIX.items():
        if counts.get(klass, 0) != expected:
            failures.append(
                f"class {klass}: expected {expected}, found {counts.get(klass, 0)}"
            )
    unknown = set(counts) - set(REQUIRED_MIX)
    if unknown:
        failures.append(f"unknown question_class values: {sorted(unknown)}")

    # 3. metadata completeness
    ids = [row.id for row in rows]
    if len(set(ids)) != len(ids):
        failures.append("duplicate question ids")
    for row in rows:
        if not row.question.strip():
            failures.append(f"{row.id}: empty question")
        if not row.rationale:
            failures.append(f"{row.id}: no rationale")
        if row.sdk_version not in ("v2", "v3", "both", "none"):
            failures.append(f"{row.id}: bad sdk_version {row.sdk_version!r}")
        if row.expect_refusal and row.required_facts:
            failures.append(f"{row.id}: a refusal question must have no required_facts")
        if not row.expect_refusal and not row.required_facts:
            failures.append(f"{row.id}: non-refusal question has no required_facts")
        if not row.expect_refusal and not row.must_contain:
            failures.append(f"{row.id}: non-refusal question has no must_contain")
        for tool in row.expected_tools + row.expected_order:
            if tool not in TOOL_NAMES:
                failures.append(f"{row.id}: unknown tool {tool!r}")

    # 4. at least four questions answerable by NO single tool
    multi = [m.id for m in matrix if m.needs_multiple]
    if len(multi) < MIN_MULTI_TOOL:
        failures.append(
            f"only {len(multi)} questions need multiple tools; "
            f"at least {MIN_MULTI_TOOL} required — {multi}"
        )

    # 5. at least three with an explicit dependency
    dependent = [row.id for row in rows if row.is_dependent]
    if len(dependent) < MIN_DEPENDENT:
        failures.append(
            f"only {len(dependent)} questions declare a step dependency; "
            f"at least {MIN_DEPENDENT} required"
        )

    # the authored expectation must agree with the measured matrix
    by_id = {m.id: m for m in matrix}
    for row in rows:
        measured = by_id[row.id]
        if row.question_class == "two_hop" and not measured.needs_multiple:
            failures.append(
                f"{row.id}: declared two_hop but "
                f"{measured.single_tool_answers} answers it alone — the claim is false"
            )
        if row.question_class == "single_hop":
            if len(measured.single_tool_answers) == 0:
                failures.append(
                    f"{row.id}: declared single_hop but no tool answers it alone"
                )
            elif len(measured.single_tool_answers) > 1:
                warnings.append(
                    f"{row.id}: {measured.single_tool_answers} all answer alone; "
                    "the question does not discriminate between tools"
                )
            elif measured.single_tool_answers != row.expected_tools:
                failures.append(
                    f"{row.id}: expected {row.expected_tools} to answer alone, "
                    f"measured {measured.single_tool_answers}"
                )

    # 6. every referenced method / path / version exists in the authored corpus
    endpoints = load_endpoints()
    known_paths = {op["path"] for op in endpoints["operations"]}
    known_ops = {op["operation_id"] for op in endpoints["operations"]}
    pages = load_pages(docs_root)
    known_methods = {page.sdk_method for page in pages.values()}
    corpus_blob = " ".join(
        p.read_text(encoding="utf-8") for p in sorted(docs_root.rglob("*.md"))
    ) + " ".join(p.read_text(encoding="utf-8") for p in sorted(API_DIR.rglob("*")) if p.is_file())

    import re

    for row in rows:
        if row.expect_refusal:
            continue  # its whole point is naming things that do not exist
        for path in re.findall(r"/v\d+/[A-Za-z0-9:_-]+", row.question):
            if path not in known_paths:
                failures.append(f"{row.id}: question names path {path} which is not in api/")
        for method in re.findall(r"Client\.[a-z_]+", row.question):
            if method not in known_methods:
                failures.append(f"{row.id}: question names {method} which is not in docs/")
        for operation in row.required_facts:
            if operation in known_ops:
                continue
        # every literal must exist somewhere, or it tests nothing
        for literal in row.required_facts + row.must_contain:
            probe = literal.strip('"').split('"')[0] if literal.startswith('"') else literal
            if probe and probe not in corpus_blob and probe not in json.dumps(endpoints):
                if literal.startswith('"'):
                    continue  # structural JSON fact, checked by T2 directly
                warnings.append(
                    f"{row.id}: literal {literal!r} appears in neither docs/ nor api/"
                )
        # must_not_contain is NOT checked for existence in the corpus, and that is
        # deliberate. Its job is to catch a wrong answer, and a wrong answer is
        # often something the corpus does not contain: "/v2/batches" on w7-07 is a
        # plausible fabrication (mixing the v2 prefix with the batch noun) that
        # exists nowhere precisely because it is wrong. Warning that such a literal
        # "tests nothing" inverts its purpose.
        #
        # What IS worth failing on is self-contradiction: a forbidden literal that
        # a correct answer would have to contain makes the question unpassable.
        for literal in row.must_not_contain:
            if literal in row.must_contain:
                failures.append(
                    f"{row.id}: {literal!r} is in both must_contain and must_not_contain"
                )
            for wanted in row.must_contain:
                if literal and literal in wanted:
                    failures.append(
                        f"{row.id}: must_not_contain {literal!r} is a substring of "
                        f"required {wanted!r} — the question cannot be passed"
                    )
            for fact in row.required_facts:
                if literal and literal in fact:
                    warnings.append(
                        f"{row.id}: must_not_contain {literal!r} is a substring of "
                        f"required_fact {fact!r}; an answer quoting the fact would fail"
                    )

    return failures, warnings


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_matrix(matrix: list[MatrixRow]) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 100)
    add("T2 — ANSWERABILITY MATRIX")
    add("=" * 100)
    add("")
    add("Computed by calling each tool alone over its whole argument space and")
    add("string-matching required_facts against the real JSON payloads. Nothing below")
    add("is hand-asserted. Best shot per tool:")
    add("  search_docs       question text + each required_fact as a query, both versions")
    add("  get_openapi_spec  6 SDK methods x 2 versions + 9 authored paths x 2 versions")
    add("  check_deprecation same enumeration")
    add("")
    header = (
        f"{'id':7} {'class':14} {'SD':4} {'OS':4} {'CD':4} {'min':4} {'expected order'}"
    )
    add(header)
    add("-" * len(header))
    for row in matrix:
        verdicts = {v.tool: v for v in row.verdicts}
        cells = [
            "YES" if verdicts[name].answers_alone else "NO" for name in TOOL_NAMES
        ]
        order = " -> ".join(row.expected_order) if row.expected_order else "(refuse)"
        add(
            f"{row.id:7} {row.question_class:14} {cells[0]:4} {cells[1]:4} {cells[2]:4} "
            f"{row.min_tools:<4} {order}"
        )
    add("-" * len(header))
    multi = [r.id for r in matrix if r.needs_multiple]
    add(f"answerable by NO single tool: {len(multi)}  {multi}")
    add("")
    add("REQUIRED INFORMATION, and what each tool could not supply")
    add("-" * 100)
    for row in matrix:
        add("")
        add(f"{row.id}  [{row.question_class}]  {row.question}")
        add(f"  required facts : {row.required_facts}")
        for verdict in row.verdicts:
            mark = "YES" if verdict.answers_alone else "NO "
            detail = (
                "has every required fact"
                if verdict.answers_alone
                else (
                    f"cannot supply {verdict.missing}"
                    if verdict.missing
                    else "nothing to supply (refusal control)"
                )
            )
            add(f"  {mark} {verdict.tool:20} {detail}")
    return "\n".join(lines) + "\n"


def render_dependencies(rows: list[Week7Question], matrix: list[MatrixRow]) -> str:
    by_id = {m.id: m for m in matrix}
    lines: list[str] = []
    add = lines.append
    add("=" * 100)
    add("PROVED TOOL DEPENDENCIES")
    add("=" * 100)
    add("")
    add("For each dependent question: what step 1 returns, why step 2 cannot be")
    add("issued without it, and the measured fact that step 1's tool alone is")
    add("insufficient.")
    for row in rows:
        if not row.is_dependent:
            continue
        measured = by_id[row.id]
        first, second = row.expected_order[0], row.expected_order[1]
        first_verdict = next(v for v in measured.verdicts if v.tool == first)
        second_verdict = next(v for v in measured.verdicts if v.tool == second)
        add("")
        add("-" * 100)
        add(f"{row.id}  {row.question}")
        add("")
        add(f"  Step 1  {first}")
        add(f"          supplies : {first_verdict.found}")
        add(f"          missing  : {first_verdict.missing}   <- why step 2 is needed")
        add(f"  Step 2  {second}")
        add(f"          supplies : {second_verdict.found}")
        add(f"          missing  : {second_verdict.missing}   <- why step 1 is needed")
        add("")
        add(f"  Dependency: {row.dependency}")
    return "\n".join(lines) + "\n"


def print_summary(rows: list[Week7Question], matrix: list[MatrixRow]) -> None:
    counts = Counter(row.question_class for row in rows)
    print(f"{len(rows)} questions")
    for klass, expected in REQUIRED_MIX.items():
        mark = "OK " if counts.get(klass, 0) == expected else "BAD"
        print(f"  {mark} {klass:14} {counts.get(klass, 0)}/{expected}")
    multi = [m.id for m in matrix if m.needs_multiple]
    dependent = [r.id for r in rows if r.is_dependent]
    print(f"  need >1 tool  : {len(multi)}/{MIN_MULTI_TOOL} minimum  {multi}")
    print(f"  with dependency: {len(dependent)}/{MIN_DEPENDENT} minimum  {dependent}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 7 question set and T2 matrix.")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_QUESTIONS_YAML)
    parser.add_argument("--out", type=Path, default=DEFAULT_QUESTIONS_JSONL)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--deps", action="store_true")
    parser.add_argument("--write", type=Path, default=None, help="write the artifact to PATH")
    args = parser.parse_args(argv)

    try:
        rows = load_yaml(args.yaml)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"week7 questions failed: {exc}", file=sys.stderr)
        return 1

    if args.matrix or args.deps or args.validate:
        matrix = build_matrix(rows)

    if args.matrix:
        rendered = render_matrix(matrix)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(rendered, encoding="utf-8")
            print(f"wrote {args.write}")
        else:
            print(rendered)
        return 0

    if args.deps:
        rendered = render_dependencies(rows, matrix)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(rendered, encoding="utf-8")
            print(f"wrote {args.write}")
        else:
            print(rendered)
        return 0

    if args.validate:
        failures, warnings = validate(rows, matrix, args.docs)
        print_summary(rows, matrix)
        print()
        for warning in warnings:
            print(f"WARN  {warning}")
        for failure in failures:
            print(f"FAIL  {failure}")
        print()
        if failures:
            print(f"VALIDATION FAILED — {len(failures)} problem(s)")
            return 1
        print(f"VALIDATION PASSED — {len(warnings)} warning(s)")
        return 0

    rendered = dump_jsonl(rows)
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != rendered:
            print(f"{args.out} is stale or missing; re-run without --check", file=sys.stderr)
            return 1
        print(f"{args.out} matches a fresh derivation ({len(rows)} questions)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}  ({len(rows)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
