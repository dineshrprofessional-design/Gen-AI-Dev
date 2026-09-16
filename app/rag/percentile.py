"""Percentile helpers, factored out of `eval_week4` so Week 7 can reuse them.

    python -m app.rag.percentile --verify     # prove parity with the Week 4 numbers

Week 4 computes p50/p95 inline as properties on `Week4Run` (`eval_week4.py:120-141`).
Week 7's race needs the same statistics for two arms, and copying five lines into a
third place is how two "p50"s quietly stop meaning the same thing.

`eval_week4.py` is deliberately **not** refactored to delegate here. Its numbers are
already reported and committed in `report/w4/*.json` and `results.md`; editing the
file that produced them to satisfy a later week is how a published figure moves
without anyone noticing. Instead `--verify` recomputes Week 4's committed samples
through these functions and asserts the results are identical. That earns the reuse
without touching the artifact.

## Definitions, pinned

`percentile` uses **nearest-rank on the sorted samples**, matching
`eval_week4.py:130-132` exactly:

    index = min(len(samples) - 1, int(round(q * (len(samples) - 1))))

This is not the only defensible definition — numpy's default linear interpolation
gives different answers on small samples — so it is stated rather than assumed. A
race reporting p95 over 30 samples would differ by a whole sample position between
the two conventions.

## Small samples

Deterministic by construction, no special cases that could drift:

    n == 0   -> 0.0, matching `Week4Outcome.p50_ms`'s `if self.latencies_ms else 0.0`
    n == 1   -> that sample, for every q
    n == 2   -> median averages the pair; percentile picks by nearest rank

`median` is `statistics.median` (averages the middle pair on even n), because that
is what Week 4 used. `percentile(samples, 0.5)` is therefore NOT always equal to
`median(samples)` on even-length input — the first interpolates, the second picks.
Both are exposed, and `p50` is an alias of `median` so callers get Week 4's meaning
by default.
"""

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "median",
    "p50",
    "percentile",
    "p95",
    "median_of_group_medians",
    "summarise",
]


def median(samples: Sequence[float]) -> float:
    """Week 4's p50: `statistics.median`, averaging the middle pair on even n."""
    return float(statistics.median(samples)) if samples else 0.0


#: Week 4's meaning of p50. Alias, not a second implementation.
p50 = median


def percentile(samples: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, identical to `eval_week4.pooled_p95_ms`.

    `q` is a fraction in [0, 1].
    """
    if not samples:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return float(ordered[index])


def p95(samples: Sequence[float]) -> float:
    return percentile(samples, 0.95)


def median_of_group_medians(groups: Sequence[Sequence[float]]) -> float:
    """Week 4's headline p50: median of the per-question medians.

    Distinct from pooling every sample. A question measured more often would
    otherwise pull the pooled figure toward itself; this weights each question
    equally. `eval_week4.py:25-34` documents why both are reported.
    """
    medians = [median(group) for group in groups if group]
    return median(medians)


def summarise(groups: Sequence[Sequence[float]]) -> dict[str, float]:
    """Every figure the race reports, from one pass over the samples."""
    pooled = [sample for group in groups for sample in group]
    return {
        "per_group_p50": median_of_group_medians(groups),
        "pooled_p50": median(pooled),
        "pooled_p95": p95(pooled),
        "pooled_min": float(min(pooled)) if pooled else 0.0,
        "pooled_max": float(max(pooled)) if pooled else 0.0,
        "samples": float(len(pooled)),
        "groups": float(len([g for g in groups if g])),
    }


# --------------------------------------------------------------------------
# parity with the committed Week 4 run
# --------------------------------------------------------------------------

DEFAULT_WEEK4_RUN = Path("report/w4/after.json")


def verify_week4_parity(path: Path = DEFAULT_WEEK4_RUN) -> list[str]:
    """Recompute Week 4's committed samples here; report any disagreement.

    Reads the JSON directly rather than importing `Week4Run`, so this check
    cannot be satisfied by a shared bug in the model layer.
    """
    if not path.is_file():
        return [f"{path} not found; cannot verify parity"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = [outcome["latencies_ms"] for outcome in payload["outcomes"]]
    pooled = [sample for group in groups for sample in group]

    # The same three formulas as eval_week4.py:121-132, inlined for comparison.
    expected_per_question = statistics.median([statistics.median(g) for g in groups])
    expected_pooled_p50 = statistics.median(pooled)
    ordered = sorted(pooled)
    expected_pooled_p95 = ordered[
        min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    ]

    problems: list[str] = []
    checks = [
        ("per-question p50", median_of_group_medians(groups), expected_per_question),
        ("pooled p50", median(pooled), expected_pooled_p50),
        ("pooled p95", p95(pooled), expected_pooled_p95),
        ("pooled min", min(pooled), min(pooled)),
    ]
    for name, got, want in checks:
        if got != want:
            problems.append(f"{name}: percentile.py gives {got}, eval_week4 gives {want}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Percentile helpers for the Week 7 race.")
    parser.add_argument("--verify", action="store_true", help="parity with report/w4")
    parser.add_argument("--run", type=Path, default=DEFAULT_WEEK4_RUN)
    args = parser.parse_args(argv)

    if not args.verify:
        parser.print_help()
        return 1

    problems = verify_week4_parity(args.run)
    payload = json.loads(args.run.read_text(encoding="utf-8")) if args.run.is_file() else None
    if payload:
        groups = [o["latencies_ms"] for o in payload["outcomes"]]
        stats = summarise(groups)
        print(f"recomputed from {args.run} ({int(stats['groups'])} questions, "
              f"{int(stats['samples'])} samples)")
        for key in ("per_group_p50", "pooled_p50", "pooled_p95", "pooled_min"):
            print(f"  {key:16} {stats[key]:10.3f} ms")
        print()

    # small-sample behaviour, stated as executable examples
    print("small-sample behaviour")
    for name, samples in (("empty", []), ("one", [7.0]), ("two", [2.0, 4.0])):
        print(
            f"  {name:6} median={median(samples):<6} "
            f"p50={percentile(samples, 0.5):<6} p95={p95(samples)}"
        )
    print()

    for problem in problems:
        print(f"FAIL  {problem}", file=sys.stderr)
    if problems:
        print(f"PARITY FAILED — {len(problems)} disagreement(s)")
        return 1
    print("PARITY OK — percentile.py reproduces eval_week4's committed figures exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
