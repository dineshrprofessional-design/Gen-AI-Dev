"""Week 10: the 10 Week-6 cases both arms race on, chosen by a rule fixed before any run.

    python -m app.rag.week10_cases            # print the selection and why
    python -m app.rag.week10_cases --write    # write eval/week10_selection.yaml
    python -m app.rag.week10_cases --check    # the committed file still matches the rule and the case set

No case is written here. These are 10 of the 32 committed Week 6 cases, used
unchanged: question, expected behaviour, and the Week 6 judge and assertions.
The brief's warning is the reason for the rule. Picking the cases that "suit the
orchestrator" changes the ruler mid-measurement, and so does picking them after
seeing a run.

## The rule

1. Drop the cases Judge v2 uses as worked examples (`FEW_SHOT_CASE_IDS`). The
   judge would be grading a case whose answer it has already been shown.
2. Split what is left by `expected_behavior` and keep the Week 6 mix: 28 answer
   to 4 refuse is 9 : 1.25, so 9 answer cases and 1 refusal case.
3. Draw each pool with `random.Random(SEED).sample` over its sorted ids.
   `SEED = 10` is the week number, fixed in advance. It was not searched for.

Two more cases are fixed by rule at the same time, before any run:

- **failure case**: the first selected case (by id) whose `sdk_version` is
  `both`, because that is where the version/deprecation worker matters most.
  It falls back to the first one that names a version.
- **smoke case**: the first answer case, by id, that is *not* selected. The
  1-case calibration runs on it, so calibrating re-runs no raced case.
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

import yaml

from app.rag.week6_cases import DEFAULT_CASES_JSONL, Case, load_cases
from app.rag.week6_judge import FEW_SHOT_CASE_IDS

SEED = 10
N_ANSWER = 9
N_REFUSE = 1
DEFAULT_SELECTION = Path("eval/week10_selection.yaml")

PASS_RULE = (
    "A case passes when Judge v2 (app/rag/week6_judge.py, the Week 6 prompt and model, temperature 0) "
    "returns PASS and no Week 6 assertion (app/rag/week6_assertions.py) returns fail; n/a and skipped "
    "do not fail. Both instruments see the arm's ANSWER text only, with the citations the arm captured "
    "and, as context, every chunk or tool result its tools returned during the run. The VERSION and "
    "CITATIONS lines are metadata and are withheld, because the VERSION line would satisfy the "
    "version_stated assertion on every answer. An answer with no parsable ANSWER line fails, as in Week 7."
)

METRICS = {
    "pass_rate": "cases passed / 10, by PASS_RULE",
    "latency": (
        "wall clock per question, minus time spent sleeping on provider 429s (Week 7 rule, both arms "
        "alike); p50 and p99 by nearest rank (app/rag/percentile.py). With 10 samples p99 is the slowest run."
    ),
    "total_tokens": (
        "prompt + completion tokens of every LLM call the arm made, summed over the 10 cases: for the squad "
        "that is the orchestrator plus both workers. Judge tokens are evaluation overhead, not either arm's "
        "cost, and are excluded from both."
    ),
    "cost_per_question": (
        "list price from eval/week7_prices.yaml (openai/gpt-oss-120b, cached input at its own rate) / 10. "
        "The free tier bills nothing; this is what the tokens would cost."
    ),
    "multiplier": "squad total tokens / single-agent total tokens, one decimal place",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(cases: list[Case]) -> dict:
    excluded = sorted(set(FEW_SHOT_CASE_IDS))
    pool = [c for c in cases if c.case_id not in excluded]
    answer = sorted(c.case_id for c in pool if c.expected_behavior == "answer")
    refuse = sorted(c.case_id for c in pool if c.expected_behavior == "refuse")
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(answer, N_ANSWER) + rng.sample(refuse, N_REFUSE))

    by_id = {c.case_id: c for c in cases}
    failure = next((i for i in chosen if by_id[i].sdk_version == "both"), None) or next(
        i for i in chosen if by_id[i].sdk_version != "none"
    )
    smoke = next(i for i in answer if i not in chosen)
    return {
        "excluded_judge_examples": excluded,
        "pool_sizes": {"answer": len(answer), "refuse": len(refuse)},
        "selected": [
            {
                "case_id": i,
                "expected_behavior": by_id[i].expected_behavior,
                "sdk_version": by_id[i].sdk_version,
                "mode": by_id[i].mode,
                "question": by_id[i].question,
            }
            for i in chosen
        ],
        "failure_case": failure,
        "smoke_case": smoke,
    }


def build_document(cases_path: Path = DEFAULT_CASES_JSONL) -> dict:
    picked = select(load_cases(cases_path))
    return {
        "generated_by": "python -m app.rag.week10_cases --write",
        "week6_cases": str(cases_path),
        "week6_cases_sha256": file_sha256(cases_path),
        "seed": SEED,
        "rule": (
            f"drop Judge v2 worked examples {sorted(FEW_SHOT_CASE_IDS)}; then random.Random({SEED}).sample "
            f"{N_ANSWER} of the sorted answer cases and {N_REFUSE} of the sorted refusal cases"
        ),
        **picked,
        "pass_rule": PASS_RULE,
        "metrics": METRICS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select the 10 Week-6 cases for the Week 10 race.")
    parser.add_argument("--write", action="store_true", help=f"write {DEFAULT_SELECTION}")
    parser.add_argument("--check", action="store_true", help="verify the committed selection")
    args = parser.parse_args(argv)

    doc = build_document()
    if args.check:
        committed = yaml.safe_load(DEFAULT_SELECTION.read_text(encoding="utf-8"))
        problems = [k for k in doc if committed.get(k) != doc[k]]
        if problems:
            print(f"selection does not match the rule or the case set: {problems}", file=sys.stderr)
            return 1
        print(f"{DEFAULT_SELECTION} matches the rule and {doc['week6_cases']} (sha256 {doc['week6_cases_sha256'][:12]})")
        return 0
    if args.write:
        DEFAULT_SELECTION.write_text(
            "# Week 10 race: which Week 6 cases, and how they were chosen. Generated, do not edit:\n"
            "#   python -m app.rag.week10_cases --write   /   --check\n"
            + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"wrote {DEFAULT_SELECTION}")
    for row in doc["selected"]:
        print(f"  {row['case_id']}  {row['expected_behavior']:6} {row['sdk_version']:4} mode {row['mode']:5}  {row['question']}")
    print(f"failure case: {doc['failure_case']}   smoke case: {doc['smoke_case']}   excluded: {doc['excluded_judge_examples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
