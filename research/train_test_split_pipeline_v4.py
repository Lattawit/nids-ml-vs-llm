"""v4 train/test split.

Serves the same role as ``train_test_split_pipeline_v3.py``, on the
deduplicated dataset instead of the raw coalesced one. Differences from v3:

- ``INPUT_FILE`` is ``results/final_merged_dataset_deduped.csv`` (produced by
  ``deduplicate_dataset_v4.py`` -- exact duplicate rows removed, keep-first)
  instead of ``results/final_merged_dataset_coalesced.csv``. This is the fix
  for the cross-split leakage ``check_duplicates_v3.py`` found: ~24-49% of the
  v3 test set (depending on class) was made up of rows that were exact
  duplicates of a training row. Splitting a deduplicated pool makes that
  leakage mode structurally impossible -- there is nothing left to duplicate
  across the boundary.
- ``OUTPUT_DIR`` is ``results/split_data_v4`` so ``split_data_v3`` (and v2)
  are left intact for the before/after comparison.

Everything else -- the two-pass streaming strategy, stratified 70/30 split,
``random_state=42``, ``LabelEncoder`` semantics -- is identical to v3, with one
exception: v3's Pass 2 force-cast every feature column to ``float32`` when
reading the input file before re-writing X_train.csv/X_test.csv. That downcast
turned out to be a second, independent leakage mechanism on top of the
pre-split exact duplicates this whole v4 rebuild exists to fix -- on the
deduped (zero-duplicate-at-native-precision) input file, forcing float32
collapsed 405,342 previously-distinct rows into 493,686 "excess" duplicates
purely from precision loss, and when a collapsed pair landed on opposite sides
of the train/test boundary, that was cross-split leakage indistinguishable
from the original bug. v4 reads and writes features at native precision (no
forced downcast) so the split's output stays exactly as duplicate-free as the
deduped input file guarantees. The v3 script is untouched.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==================================================
# CONFIG
# ==================================================

INPUT_FILE = "results/final_merged_dataset_deduped.csv"

OUTPUT_DIR = "results/split_data_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.30
RANDOM_STATE = 42
CHUNK_SIZE = 1_000_000

# ==================================================
# PASS 1: LABELS -> STRATIFIED INDEX SPLIT
# ==================================================

print("Pass 1: reading labels...")

label_parts = []
for chunk in pd.read_csv(INPUT_FILE, usecols=["Label"], chunksize=CHUNK_SIZE):
    label_parts.append(chunk["Label"].to_numpy())

y_all = np.concatenate(label_parts)
n_rows = len(y_all)
print(f"Total rows: {n_rows:,}")

le = LabelEncoder()
y_encoded = le.fit_transform(y_all)

label_map = pd.DataFrame({
    "Class": le.classes_,
    "Encoded": range(len(le.classes_)),
})
label_map.to_csv(f"{OUTPUT_DIR}/label_mapping.csv", index=False)

print("\nLabel mapping")
for i, cls in enumerate(le.classes_):
    print(f"{cls} -> {i}")

print("\nFull class distribution BEFORE split:")
full_dist = pd.Series(y_all).value_counts().sort_index()
print(full_dist)

idx = np.arange(n_rows)
idx_train, idx_test = train_test_split(
    idx,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)

train_mask = np.zeros(n_rows, dtype=bool)
train_mask[idx_train] = True

# y written in FILE order (matches the streamed X rows), not shuffled order
y_train = y_encoded[train_mask]
y_test = y_encoded[~train_mask]

pd.DataFrame(y_train, columns=["Label"]).to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["Label"]).to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

# ==================================================
# PASS 2: STREAM FEATURES -> X_train.csv / X_test.csv
# ==================================================

print("\nPass 2: streaming features...")

# No forced float32 downcast here (unlike v3) -- see module docstring. Read at
# native precision so rows that Phase 1 (deduplicate_dataset_v4.py) guaranteed
# distinct don't get rounded into look-alikes that could straddle the split.
x_train_path = f"{OUTPUT_DIR}/X_train.csv"
x_test_path = f"{OUTPUT_DIR}/X_test.csv"

row0 = 0
first = True
for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    n = len(chunk)
    m = train_mask[row0:row0 + n]

    feats = chunk.drop(columns=["Label"])
    feats[m].to_csv(x_train_path, index=False, mode="w" if first else "a", header=first)
    feats[~m].to_csv(x_test_path, index=False, mode="w" if first else "a", header=first)

    row0 += n
    first = False
    print(f"  {row0:,} / {n_rows:,} rows routed")

# ==================================================
# DISTRIBUTION REPORT
# ==================================================

train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist = pd.Series(y_test).value_counts().sort_index()

report = pd.DataFrame({
    "Class": le.classes_,
    "Full": full_dist.values,
    "Train": train_dist.values,
    "Test": test_dist.values,
})
report.to_csv(f"{OUTPUT_DIR}/split_distribution.csv", index=False)

print("\nClass distribution (full / train / test):")
print(report.to_string(index=False))

print(f"\nTrain rows: {len(y_train):,}")
print(f"Test rows : {len(y_test):,}")
print(f"\nSaved to: {OUTPUT_DIR}")
