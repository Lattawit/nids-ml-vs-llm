"""v4 pipeline, Phase 2 validation: cross-split duplicate check on split_data_v4.

Reuses ``check_duplicates_v3.check_cross_split_duplicates()`` unmodified,
pointed at ``results/split_data_v4`` instead of ``results/split_data_v3``,
writing its own output dir so ``results/duplicate_check_v3/`` (the historical
record of the leakage bug this whole v4 rebuild is fixing) stays untouched.

Confirms whether deduplicating before the split (Phase 1) actually eliminated
train/test row overlap. Expected result: zero.
"""

import os

import check_duplicates_v3 as dup_check

dup_check.SPLIT_DIR = "results/split_data_v4"
dup_check.OUTPUT_DIR = "results/duplicate_check_v4"
os.makedirs(dup_check.OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    dup_check.check_cross_split_duplicates()
