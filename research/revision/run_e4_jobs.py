"""E4 job list for the resumable runner (``run_llm_jobs.run_job``).

Dynamic few-shot: per-row retrieved examples (``e4_retrieve_examples.py``) in the
otherwise unchanged 1-example/class template, Qwen2.5:14B, same options.
Set ``PILOT_ROWS`` to an int for the cost pilot; rows done in the pilot are
kept and skipped by the full run (same JSONL, same settings).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_llm_jobs as R  # noqa: E402

# ==================================================
# CONFIG
# ==================================================

SEEDS = [42, 1, 2, 3, 4]
PILOT_ROWS = None  # e.g. 20 for the pilot; None = full 700 rows

JOBS = [{"exp": "e4", "model": "qwen2.5:14B", "mode": "dynamic", "seed": s} for s in SEEDS]

if __name__ == "__main__":
    R.MAX_ROWS = PILOT_ROWS
    with open(R.REFS_FILE) as f:
        refs = json.load(f)
    for job in JOBS:
        R.run_job(job, refs)
    print("\nAll E4 jobs done.")
