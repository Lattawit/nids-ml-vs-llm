"""v4 pipeline, Phase 1: deduplicate the v3 coalesced merged dataset.

``check_duplicates_v3.py`` found that 21.1% of
``final_merged_dataset_coalesced.csv`` are exact duplicate rows (all 70
features + label identical), and that duplicates straddling the v3 train/test
split boundary caused ~24-49%-per-class exact-duplicate leakage into the
"held-out" test set. This script removes those duplicates BEFORE the split
(keep-first), so the v4 split in ``train_test_split_pipeline_v4.py`` starts
from a deduplicated pool and duplicate-driven leakage becomes structurally
impossible.

Does not touch ``final_merged_dataset_coalesced.csv`` or anything from v2/v3
-- writes a new file, per the versioned-pipeline-copies convention.

Streamed in chunks (same CHUNK_SIZE and hashing method --
``pandas.util.hash_pandas_object``, one hash per row -- as
``check_duplicates_v3.py``, on the same input file) so the per-class removed
counts below should reproduce ``results/duplicate_check_v3/
merged_dataset_duplicate_report.csv``'s ``Excess_Duplicate_Rows`` column
exactly; this script asserts that at the end as a validation check.
"""

import os
import time
from collections import defaultdict

import pandas as pd

# ==================================================
# CONFIG
# ==================================================

INPUT_FILE = "results/final_merged_dataset_coalesced.csv"
OUTPUT_FILE = "results/final_merged_dataset_deduped.csv"
CHUNK_SIZE = 1_000_000

PRIOR_DUP_REPORT = "results/duplicate_check_v3/merged_dataset_duplicate_report.csv"

OUTPUT_DIR = "results/dedup_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def deduplicate():
    print(f"=== Deduplicating {INPUT_FILE} -> {OUTPUT_FILE} (keep-first) ===")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    seen_hashes = set()
    class_before = defaultdict(int)
    class_after = defaultdict(int)

    t0 = time.time()
    total_before = 0
    total_after = 0
    wrote_header = False

    for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
        total_before += len(chunk)
        for label, cnt in chunk["Label"].value_counts().items():
            class_before[label] += int(cnt)

        row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        keep_mask = []
        for h in row_hashes:
            if h in seen_hashes:
                keep_mask.append(False)
            else:
                seen_hashes.add(h)
                keep_mask.append(True)

        kept = chunk.loc[keep_mask]
        total_after += len(kept)
        for label, cnt in kept["Label"].value_counts().items():
            class_after[label] += int(cnt)

        kept.to_csv(OUTPUT_FILE, mode="a", header=not wrote_header, index=False)
        wrote_header = True

        if (i + 1) % 5 == 0:
            print(f"  ...{total_before:,} rows read, {total_after:,} kept so far ({time.time() - t0:.0f}s)")

    print(f"Done in {time.time() - t0:.0f}s")
    print(f"Rows before: {total_before:,}")
    print(f"Rows after:  {total_after:,}")
    removed = total_before - total_after
    print(f"Removed:     {removed:,} ({100 * removed / total_before:.4f}% of dataset)")

    print("\nPer-class breakdown:")
    rows = []
    for lbl, before in sorted(class_before.items(), key=lambda kv: -kv[1]):
        after = class_after.get(lbl, 0)
        removed_c = before - after
        pct = 100 * removed_c / before if before else 0.0
        rows.append(
            {
                "Class": lbl,
                "Rows_Before": before,
                "Rows_After": after,
                "Rows_Removed": removed_c,
                "Pct_Removed": round(pct, 4),
            }
        )
        print(f"  {lbl:15s} before={before:>10,}  after={after:>10,}  removed={removed_c:>10,}  ({pct:.3f}%)")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "dedup_report.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    _validate_against_prior_report(rows)

    return rows


def _validate_against_prior_report(rows):
    """Cross-check removed-row counts per class against the
    already-reported excess-duplicate counts from check_duplicates_v3.py's
    Part 1. They should match exactly (same file, same chunking, same
    hash_pandas_object-based hashing)."""
    if not os.path.exists(PRIOR_DUP_REPORT):
        print(f"\nWARNING: {PRIOR_DUP_REPORT} not found -- skipping cross-check.")
        return

    prior = pd.read_csv(PRIOR_DUP_REPORT).set_index("Class")["Excess_Duplicate_Rows"].to_dict()
    current = {r["Class"]: r["Rows_Removed"] for r in rows}

    print(f"\nCross-checking against {PRIOR_DUP_REPORT}...")
    all_match = True
    for cls in sorted(set(prior) | set(current)):
        p = prior.get(cls)
        c = current.get(cls)
        ok = p == c
        all_match &= bool(ok)
        status = "OK" if ok else "MISMATCH"
        print(f"  {cls:15s} prior_excess={p!s:>10}  removed_here={c!s:>10}  [{status}]")

    if all_match:
        print("  All classes match exactly.")
    else:
        print("  WARNING: mismatch(es) found -- investigate before proceeding.")


if __name__ == "__main__":
    deduplicate()
