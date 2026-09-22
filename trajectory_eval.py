#!/usr/bin/env python3
"""Repo-root entry point for the Week 8 trajectory evaluator.

The implementation lives in `app/rag/trajectory_eval.py`, alongside the rest of
the pipeline it reads. This shim exists because the Week 8 brief names
`trajectory_eval.py` at the root as a deliverable, and because a trajectory run
is something you want to launch without remembering the package path:

    python trajectory_eval.py --self-test
    python trajectory_eval.py --replay report/w8/baseline_runs.json

A real module rather than a symlink, so a Windows checkout and a zip export
both still work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.rag.trajectory_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
