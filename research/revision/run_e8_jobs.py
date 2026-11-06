"""E8 job list for the resumable runner (``run_llm_jobs.run_job``).

3 examples/class (paper Table VI) on the 5 revision subsets, Qwen2.5:14B,
temperature=0, num_predict=60, num_ctx=8192 (as the original Table VI runs).
Reference blocks from ``e8_threeshot_examples.py``.
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
JOBS = [{"exp": "e8", "model": "qwen2.5:14B", "mode": "three", "seed": s} for s in SEEDS]

if __name__ == "__main__":
    with open(R.REFS_FILE) as f:
        refs = json.load(f)
    for job in JOBS:
        R.run_job(job, refs)
    print("\nAll E8 jobs done.")
