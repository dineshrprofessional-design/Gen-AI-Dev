"""Judge v1 — one binary criterion, measured against the committed labels.

    python -m app.rag.week6_judge --out report/judge_v1.txt
    python -m app.rag.week6_judge --dry-run      # print the prompt, call nothing

The judge answers exactly one question per case and nothing else. Everything a
machine can decide was moved into `app/rag/week6_assertions.py` in Phase 2 and
is deliberately ABSENT from the prompt below: citation bracket style, whether a
citation was captured, code syntax, code grounding, and whether a version label
is present. The prompt says so explicitly, because a judge told to weigh five
things silently weighs them at whatever ratio it feels like, and the resulting
number cannot be attributed to any one of them.

## What is being measured

Not "is the judge right". The judge's verdicts are compared against
`eval/labels_25.json`, which was committed in fb8c06d BEFORE this module
existed. The agreement percentage is a property of the pair, and the
disagreements are the useful output: each one is either a judge defect or a
label defect, and telling those apart is what Phase 4 is for.

## No iteration

This is v1 and stays v1. The prompt is written once and run once. Tuning it
against the agreement number would make the number meaningless - it would be
reporting how well a prompt was fitted to 26 known answers.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from app.rag.index import DEFAULT_DOCS
from app.rag.week6_assertions import chunk_texts
from app.rag.week6_cases import DEFAULT_CASES_JSONL, Case, load_cases

DEFAULT_LABELS = Path("eval/labels_25.json")
DEFAULT_OUT = Path("report/judge_v1.txt")

JUDGE_VERSION = "v1"

# ---------------------------------------------------------------------------
# The prompt. One criterion. The excluded list is not decoration - it is the
# Phase 2 requirement made literal.
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You evaluate a documentation assistant's answer about a Python SDK.

You are given the question a user asked, the numbered context chunks the \
assistant was given, and the answer it produced.

Judge exactly ONE thing:

  Is this answer correct and useful - does it answer the question that was \
asked, from the context it was given, without asserting anything that context \
does not support?

PASS when the answer resolves the question that was asked and every claim in it \
is supported by one of the context chunks. A refusal ("INSUFFICIENT_CONTEXT") \
is PASS only when the context genuinely does not contain the answer.

FAIL when the answer is wrong; or is about a different parameter than the one \
asked; or silently answers only one side of a question the context shows has \
two different answers; or asserts something no context chunk supports; or \
refuses even though the answer was present in the context it was given.

Do NOT consider any of the following. They are checked separately by \
deterministic code and are not your concern:
  - how citations are bracketed, or whether citations were captured at all
  - whether code samples are syntactically valid
  - whether an explicit version label appears in the text

Reply with exactly two lines and nothing else:
VERDICT: PASS
REASON: <one sentence>

or

VERDICT: FAIL
REASON: <one sentence>
"""

VERDICT_LINE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.I)
REASON_LINE = re.compile(r"REASON:\s*(.+)", re.I)

# ---------------------------------------------------------------------------
# Judge v2 — the SAME criterion text plus two worked examples.
#
# One variable moves between v1 and v2. The criterion wording, the excluded
# list, the reply format, the model, the temperature, the cases and the labels
# are all identical; the only difference is the FEW-SHOT block appended below.
# That is the whole point: if agreement changes, the examples are the reason,
# because nothing else was allowed to change.
#
# Both examples are real v1 disagreements, and both are false PASSes — the
# error that accounts for 7 of v1's 8 misses. Their texts are pulled from the
# case set at runtime rather than pasted here, so a few-shot example cannot
# silently drift from the trace it claims to quote.
#
# Deliberately NOT used as an example: any refusal case. w6-017 and w6-018 are
# label defects (verified: the asked symbol is in none of their retrieved
# chunks), so teaching the judge to fail them would raise the agreement number
# by making the judge reason wrongly. See report/prediction.txt.
# ---------------------------------------------------------------------------

FEW_SHOT_CASE_IDS = ["w6-020", "w6-023"]

FEW_SHOT_VERDICTS = {
    "w6-020": (
        "FAIL",
        "Every sentence is true and supported by the context, but the question "
        "asked about `fail_fast` and the answer is about `force` — a different "
        "parameter, which the context never connects to the one asked.",
    ),
    "w6-023": (
        "FAIL",
        "The context documents that a `proxy` parameter exists, but no chunk "
        "shows a call using it; the concrete invocation in the answer is "
        "presented as documented usage and is not.",
    ),
}

FEW_SHOT_PREAMBLE = """\

Before you judge, study these two worked examples. Both look correct on a \
sentence-by-sentence reading and both are FAIL. Grounding every sentence is \
necessary but it is not sufficient.
"""


def build_few_shot_block(cases: list[Case]) -> str:
    """Assemble the worked examples from the case set, verbatim."""
    by_id = {c.case_id: c for c in cases}
    parts = [FEW_SHOT_PREAMBLE]
    for index, case_id in enumerate(FEW_SHOT_CASE_IDS, start=1):
        case = by_id.get(case_id)
        if case is None or case.recorded is None:
            raise ValueError(f"few-shot case {case_id} missing from the case set")
        verdict, reason = FEW_SHOT_VERDICTS[case_id]
        context_ids = ", ".join(h["chunk_id"] for h in case.recorded.retrieved)
        parts.append(
            f"\nEXAMPLE {index}\n"
            f"Question: {case.question}\n"
            f"Chunk ids in the context: {context_ids}\n"
            f"Answer produced:\n{case.recorded.raw_output}\n"
            f"Correct judgement:\n"
            f"VERDICT: {verdict}\n"
            f"REASON: {reason}\n"
        )
    parts.append(
        "\nNow judge the case below by the same standard. Reply with the two "
        "lines and nothing else.\n"
    )
    return "".join(parts)


def system_prompt(version: str, cases: list[Case]) -> str:
    if version == "v1":
        return JUDGE_SYSTEM_PROMPT
    if version == "v2":
        return JUDGE_SYSTEM_PROMPT + build_few_shot_block(cases)
    raise ValueError(f"unknown judge version {version!r}")


class Verdict(BaseModel):
    case_id: str
    trace_id: str | None = None
    mode: str
    question: str
    verdict: str  # PASS | FAIL | ERROR
    reason: str = ""
    raw: str = ""
    error: str = ""


class Comparison(BaseModel):
    case_id: str
    mode: str
    question: str
    human: str
    judge: str
    agree: bool
    judge_reason: str = ""
    human_basis: str = ""
    needs_signoff: bool = False


class JudgeRun(BaseModel):
    version: str
    model: str
    prompt: str = ""
    labels_commit: str = ""
    verdicts: list[Verdict] = Field(default_factory=list)
    comparisons: list[Comparison] = Field(default_factory=list)

    @property
    def agreed(self) -> int:
        return sum(c.agree for c in self.comparisons)

    @property
    def compared(self) -> int:
        return len(self.comparisons)

    @property
    def agreement_pct(self) -> float:
        return (self.agreed / self.compared * 100) if self.compared else 0.0


def build_user_prompt(case: Case, texts: dict[str, str]) -> str:
    """Question, the context that was actually retrieved, and the answer."""
    recorded = case.recorded
    assert recorded is not None
    blocks = []
    for hit in recorded.retrieved:
        cid = hit["chunk_id"]
        blocks.append(f"[{cid}]\n{texts.get(cid, '(chunk text unavailable)')}")
    context = "\n\n---\n\n".join(blocks)
    return (
        f"Question the user asked:\n{case.question}\n\n"
        f"Context chunks the assistant was given:\n\n{context}\n\n"
        f"The answer the assistant produced:\n{recorded.raw_output}\n"
    )


def judge_one(
    case: Case,
    texts: dict[str, str],
    model: str | None,
    prompt: str = JUDGE_SYSTEM_PROMPT,
    retries: int = 5,
) -> Verdict:
    """One verdict. Retries transport failures; never retries a verdict.

    The free Groq tier caps this model at 8000 tokens per minute and each case
    is ~1.7k tokens, so 429s are expected rather than exceptional. A 429 is a
    transport failure, not an opinion — retrying it completes the run, and
    changes nothing about what is being asked. The prompt is identical on every
    attempt.
    """
    from app.rag.generate import _client, _model

    model = model or _model()
    base = Verdict(
        case_id=case.case_id,
        trace_id=case.trace_id,
        mode=case.mode,
        question=case.question,
        verdict="ERROR",
    )
    text = ""
    for attempt in range(retries):
        try:
            completion = _client().chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": build_user_prompt(case, texts)},
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            base.error = ""
            break
        except Exception as exc:  # noqa: BLE001 - one bad call must not lose the run
            base.error = f"{type(exc).__name__}: {exc}"
            if "rate_limit" not in str(exc) and "429" not in str(exc):
                return base
            wait = float(re.search(r"try again in ([\d.]+)s", str(exc)).group(1)) + 1.0 \
                if re.search(r"try again in ([\d.]+)s", str(exc)) else 8.0 * (attempt + 1)
            time.sleep(min(wait, 45.0))
    if base.error:
        return base

    base.raw = text
    match = VERDICT_LINE.search(text)
    if not match:
        base.error = "no VERDICT line in the judge's reply"
        return base
    base.verdict = match.group(1).upper()
    reason = REASON_LINE.search(text)
    base.reason = reason.group(1).strip() if reason else ""
    return base


def load_labels(path: Path = DEFAULT_LABELS) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"labels not found: {path}. The labels must exist and be committed "
            "before the judge runs."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def compare(verdicts: list[Verdict], labels: dict) -> list[Comparison]:
    by_id = {row["case_id"]: row for row in labels["labels"]}
    out: list[Comparison] = []
    for verdict in verdicts:
        row = by_id.get(verdict.case_id)
        if row is None:
            continue
        out.append(
            Comparison(
                case_id=verdict.case_id,
                mode=verdict.mode,
                question=verdict.question,
                human=row["label"],
                judge=verdict.verdict,
                agree=row["label"] == verdict.verdict,
                judge_reason=verdict.reason,
                human_basis=row.get("human_basis", ""),
                needs_signoff=row.get("needs_signoff", False),
            )
        )
    return out


def render(run: JudgeRun) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"WEEK 6 — JUDGE {run.version.upper()}")
    add("=" * 78)
    add(f"model                : {run.model}")
    add(f"temperature          : 0")
    add(f"criteria judged      : 1 (binary PASS/FAIL)")
    add(f"criteria excluded    : 4 (moved to deterministic assertions in Phase 2)")
    add(f"labels commit        : {run.labels_commit}")
    add("")
    add("-" * 78)
    add("JUDGE PROMPT (verbatim, as sent)")
    add("-" * 78)
    add(run.prompt.rstrip())
    add("")
    add("-" * 78)
    add("PER-CASE VERDICTS")
    add("-" * 78)
    add(f"{'case':8} {'mode':6} {'human':6} {'judge':6} {'agree':6} question")
    for c in run.comparisons:
        mark = "yes" if c.agree else "NO"
        add(
            f"{c.case_id:8} {c.mode:6} {c.human:6} {c.judge:6} {mark:6} "
            f"{c.question[:44]!r}"
        )
    add("")
    add("-" * 78)
    add("AGREEMENT")
    add("-" * 78)
    add(f"cases compared       : {run.compared}")
    add(f"agreed               : {run.agreed}")
    add(f"disagreed            : {run.compared - run.agreed}")
    add(f"agreement            : {run.agreement_pct:.1f}%")
    add("")

    confusion = Counter((c.human, c.judge) for c in run.comparisons)
    add("confusion matrix (human x judge):")
    add(f"                 judge PASS   judge FAIL")
    for human in ("PASS", "FAIL"):
        add(
            f"  human {human:4}   {confusion[(human, 'PASS')]:10} "
            f"{confusion[(human, 'FAIL')]:12}"
        )
    add("")

    by_mode: dict[str, list[Comparison]] = {}
    for c in run.comparisons:
        by_mode.setdefault(c.mode, []).append(c)
    add("agreement by taxonomy mode:")
    add(f"  {'mode':6} {'n':>3} {'agree':>6} {'pct':>7}")
    for mode in sorted(by_mode, key=lambda m: (m == "clean", m)):
        rows = by_mode[mode]
        agreed = sum(r.agree for r in rows)
        add(f"  {mode:6} {len(rows):3} {agreed:6} {agreed / len(rows) * 100:6.0f}%")
    add("")

    add("-" * 78)
    add("DISAGREEMENTS")
    add("-" * 78)
    disagreements = [c for c in run.comparisons if not c.agree]
    if not disagreements:
        add("none")
    for c in disagreements:
        add("")
        add(f"{c.case_id}  mode {c.mode}   human={c.human}  judge={c.judge}"
            f"{'   [label flagged needs_signoff]' if c.needs_signoff else ''}")
        add(f"  question    : {c.question!r}")
        add(f"  human basis : {c.human_basis}")
        add(f"  judge reason: {c.judge_reason}")
    add("")

    errors = [v for v in run.verdicts if v.verdict == "ERROR"]
    if errors:
        add("-" * 78)
        add("ERRORS")
        add("-" * 78)
        for v in errors:
            add(f"{v.case_id}: {v.error}")
        add("")

    add("=" * 78)
    add("NOTE ON ORDERING")
    add("=" * 78)
    add("eval/labels_25.json was committed in " + (run.labels_commit or "(unknown)"))
    add("before this module existed. The labels were not edited afterwards; any")
    add("change to them must be a new commit with its reason recorded.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Judge v1 over the Week 6 cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--delay",
        type=float,
        default=14.0,
        help="seconds between calls; the free tier caps this model at 8000 TPM",
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v1",
        help="v1 = criterion only; v2 = the same criterion plus two worked examples",
    )
    parser.add_argument("--dry-run", action="store_true", help="print a prompt, call nothing")
    args = parser.parse_args(argv)

    from app.rag.generate import _model, is_configured

    try:
        cases = load_cases(args.cases)
        labels = load_labels(args.labels)
    except (FileNotFoundError, ValueError) as exc:
        print(f"judge failed to start: {exc}", file=sys.stderr)
        return 1

    judgeable = [c for c in cases if c.recorded is not None]
    texts = chunk_texts(args.docs)

    prompt = system_prompt(args.version, cases)

    if args.dry_run:
        print(prompt)
        print("-" * 78)
        print(build_user_prompt(judgeable[0], texts)[:1800])
        print(f"\n({len(judgeable)} cases would be judged; nothing was called)")
        return 0

    if not is_configured():
        print("no GROQ_API_KEY — the judge cannot run", file=sys.stderr)
        return 1

    model = args.model or _model()
    run = JudgeRun(version=args.version, model=model, prompt=prompt)

    import subprocess

    try:
        out = subprocess.run(
            ["git", "log", "--format=%h %ai", "-1", "--", str(args.labels)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        run.labels_commit = out.stdout.strip()
    except Exception:  # noqa: BLE001
        run.labels_commit = "unknown"

    print(f"judging {len(judgeable)} cases with {model} ...")
    for index, case in enumerate(judgeable, start=1):
        verdict = judge_one(case, texts, model, prompt=prompt)
        run.verdicts.append(verdict)
        flag = verdict.verdict if not verdict.error else f"ERROR ({verdict.error})"
        print(f"  {index:2}/{len(judgeable)}  {case.case_id}  {flag}", flush=True)
        time.sleep(args.delay)  # stay under the tokens-per-minute cap

    run.comparisons = compare(run.verdicts, labels)

    rendered = render(run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")

    print()
    print(f"agreement: {run.agreed}/{run.compared} = {run.agreement_pct:.1f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
