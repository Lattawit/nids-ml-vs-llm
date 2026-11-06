"""Duplicate / near-duplicate row check on the v3 pipeline (report only).

One-off diagnostic requested to address reviewer concerns R1#2 and R4#2
("duplicate/near-duplicate rows were flagged but never actually checked").
Does not modify or remove anything -- only reports counts, so the removal
strategy (if any) can be decided separately.

Two checks, both on exact row equality (all 70 features + label identical):

1. Exact duplicate rows in the full v3 coalesced dataset
   (``results/final_merged_dataset_coalesced.csv``, 18.3M rows), broken down
   by class.
2. Cross-split duplicates in ``results/split_data_v3/``: does the same exact
   row (features + encoded label) appear in both the train split
   (X_train.csv + y_train.csv) and the test split (X_test.csv + y_test.csv)?
   This is the more serious case -- literal test-set leakage, not just
   redundant training data.

Both checks are hash-based (``pandas.util.hash_pandas_object``, 64-bit,
streamed in chunks so the 8GB+ files never need to fit in memory at once).
Collision probability at this row count is negligible (~1e-5 birthday bound
for 18M rows over a 64-bit space), and Part 1 additionally spot-verifies a
sample of the largest duplicate groups against raw row content.
"""

import os
import time
from collections import defaultdict

import pandas as pd

# ==================================================
# CONFIG
# ==================================================

MERGED_FILE = "results/final_merged_dataset_coalesced.csv"
SPLIT_DIR = "results/split_data_v3"
CHUNK_SIZE = 1_000_000
VERIFY_TOP_N_GROUPS = 20

OUTPUT_DIR = "results/duplicate_check_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# PART 1: exact duplicates in the merged coalesced dataset
# ==================================================

def check_merged_duplicates():
    print(f"=== Part 1: exact duplicates in {MERGED_FILE} ===")
    hash_counts = defaultdict(int)
    hash_label = {}
    class_totals = defaultdict(int)

    t0 = time.time()
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(MERGED_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
        total_rows += len(chunk)

        for label, cnt in chunk["Label"].value_counts().items():
            class_totals[label] += int(cnt)

        row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        for h, lbl in zip(row_hashes, chunk["Label"].to_numpy()):
            hash_counts[h] += 1
            if h not in hash_label:
                hash_label[h] = lbl

        if (i + 1) % 5 == 0:
            print(f"  ...{total_rows:,} rows processed ({time.time() - t0:.0f}s)")

    print(f"Total rows: {total_rows:,}  (elapsed {time.time() - t0:.0f}s)")

    dup_hashes = {h: c for h, c in hash_counts.items() if c > 1}
    total_rows_involved = sum(dup_hashes.values())
    total_excess_rows = sum(c - 1 for c in dup_hashes.values())

    print(f"Unique hash values: {len(hash_counts):,}")
    print(f"Duplicate groups (row content seen >1x): {len(dup_hashes):,}")
    print(
        f"Rows involved in duplication (incl. first occurrence): "
        f"{total_rows_involved:,} ({100 * total_rows_involved / total_rows:.3f}% of dataset)"
    )
    print(
        f"'Excess' duplicate rows (what keep-first dedup would drop): "
        f"{total_excess_rows:,} ({100 * total_excess_rows / total_rows:.3f}% of dataset)"
    )

    class_dup_excess = defaultdict(int)
    class_dup_groups = defaultdict(int)
    for h, c in dup_hashes.items():
        lbl = hash_label[h]
        class_dup_excess[lbl] += c - 1
        class_dup_groups[lbl] += 1

    print("\nPer-class breakdown:")
    rows = []
    for lbl, total in sorted(class_totals.items(), key=lambda kv: -kv[1]):
        excess = class_dup_excess.get(lbl, 0)
        groups = class_dup_groups.get(lbl, 0)
        pct = 100 * excess / total if total else 0.0
        rows.append(
            {
                "Class": lbl,
                "Total_Rows": total,
                "Duplicate_Groups": groups,
                "Excess_Duplicate_Rows": excess,
                "Pct_Excess_Duplicate": round(pct, 4),
            }
        )
        print(f"  {lbl:15s} total={total:>10,}  dup_groups={groups:>8,}  excess_dup={excess:>10,}  ({pct:.3f}%)")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "merged_dataset_duplicate_report.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    _verify_duplicate_groups(dup_hashes)

    return dup_hashes


def _verify_duplicate_groups(dup_hashes):
    """Spot-check the largest duplicate groups against raw row content, to
    confirm they are true duplicates and not hash collisions."""
    if not dup_hashes:
        print("\nNo duplicate groups to verify.")
        return

    targets = set(sorted(dup_hashes, key=lambda h: -dup_hashes[h])[:VERIFY_TOP_N_GROUPS])
    print(f"\nVerifying top {len(targets)} duplicate groups against raw row content...")

    collected = defaultdict(list)
    for chunk in pd.read_csv(MERGED_FILE, chunksize=CHUNK_SIZE, low_memory=False):
        row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        mask = pd.Series(row_hashes).isin(targets).to_numpy()
        if mask.any():
            hit_hashes = row_hashes[mask]
            hit_rows = chunk.loc[mask]
            for h, (_, row) in zip(hit_hashes, hit_rows.iterrows()):
                collected[h].append(tuple(row.tolist()))
        if all(len(collected.get(h, [])) >= dup_hashes[h] for h in targets):
            break

    all_match = True
    for h in targets:
        rows_for_hash = collected.get(h, [])
        distinct = set(rows_for_hash)
        ok = len(distinct) == 1
        all_match &= ok
        if not ok:
            print(f"  MISMATCH for hash {h}: {len(distinct)} distinct row contents among {len(rows_for_hash)} rows (possible hash collision)")

    if all_match:
        print(f"  OK: all {len(targets)} verified groups have byte-identical row content.")
    else:
        print("  WARNING: some verified groups were NOT byte-identical -- see above.")


# ==================================================
# PART 2: cross-split duplicates (train vs. test leakage)
# ==================================================

def _hash_split(x_path, y_path):
    hash_counts = defaultdict(int)
    hash_label = {}
    total = 0
    t0 = time.time()
    x_iter = pd.read_csv(x_path, chunksize=CHUNK_SIZE, low_memory=False)
    y_iter = pd.read_csv(y_path, chunksize=CHUNK_SIZE, low_memory=False)
    for xc, yc in zip(x_iter, y_iter):
        assert len(xc) == len(yc), f"chunk length mismatch between {x_path} and {y_path}: {len(xc)} vs {len(yc)}"
        combined = xc.copy()
        combined["Label"] = yc["Label"].to_numpy()
        row_hashes = pd.util.hash_pandas_object(combined, index=False).to_numpy()
        for h, lbl in zip(row_hashes, combined["Label"].to_numpy()):
            hash_counts[h] += 1
            if h not in hash_label:
                hash_label[h] = lbl
        total += len(xc)
    print(f"  {x_path}: {total:,} rows hashed in {time.time() - t0:.0f}s, {len(hash_counts):,} unique row-hashes")
    return hash_counts, hash_label, total


def check_cross_split_duplicates():
    print(f"\n=== Part 2: cross-split duplicates in {SPLIT_DIR} (train vs. test) ===")
    label_map = pd.read_csv(os.path.join(SPLIT_DIR, "label_mapping.csv")).set_index("Encoded")["Class"].to_dict()

    train_counts, train_label, n_train = _hash_split(
        os.path.join(SPLIT_DIR, "X_train.csv"), os.path.join(SPLIT_DIR, "y_train.csv")
    )
    test_counts, test_label, n_test = _hash_split(
        os.path.join(SPLIT_DIR, "X_test.csv"), os.path.join(SPLIT_DIR, "y_test.csv")
    )

    overlap = set(train_counts) & set(test_counts)
    leaked_train_rows = sum(train_counts[h] for h in overlap)
    leaked_test_rows = sum(test_counts[h] for h in overlap)

    print(f"\nTrain rows: {n_train:,}  Test rows: {n_test:,}")
    print(f"Distinct row contents appearing in BOTH train and test: {len(overlap):,}")
    if n_train:
        print(f"  -> {leaked_train_rows:,} train rows match a test row ({100 * leaked_train_rows / n_train:.4f}% of train)")
    if n_test:
        print(f"  -> {leaked_test_rows:,} test rows match a train row ({100 * leaked_test_rows / n_test:.4f}% of test)")

    class_test_totals = defaultdict(int)
    class_leak = defaultdict(int)
    for h, c in test_counts.items():
        lbl = label_map.get(test_label[h], test_label[h])
        class_test_totals[lbl] += c
    for h in overlap:
        lbl = label_map.get(test_label[h], test_label[h])
        class_leak[lbl] += test_counts[h]

    print("\nPer-class breakdown (test rows that also appear in train):")
    rows = []
    for lbl, total in sorted(class_test_totals.items(), key=lambda kv: -kv[1]):
        leaked = class_leak.get(lbl, 0)
        pct = 100 * leaked / total if total else 0.0
        rows.append(
            {
                "Class": lbl,
                "Test_Rows": total,
                "Leaked_Test_Rows_Also_In_Train": leaked,
                "Pct_Leaked": round(pct, 4),
            }
        )
        print(f"  {lbl:15s} test_total={total:>10,}  leaked={leaked:>10,}  ({pct:.3f}%)")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "cross_split_leakage_report.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    check_merged_duplicates()
    check_cross_split_duplicates()
