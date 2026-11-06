"""v3 train/test split.

Serves the same role as ``train_test_split_pipeline_v2.py`` for the paper
revision. Differences from v2:

- ``INPUT_FILE`` is ``results/final_merged_dataset_coalesced.csv`` (long/short
  column pairs coalesced, dead columns dropped, NO pre-split Benign
  downsampling) instead of ``results/final_balanced_dataset.csv``.
- ``OUTPUT_DIR`` is ``results/split_data_v3`` so ``split_data_v2`` is left
  intact for the before/after comparison.
- The split is done in two streaming passes instead of one in-memory
  ``train_test_split`` call: the coalesced dataset is 18.3M rows with no
  pre-split downsampling, and loading the whole frame exceeds RAM. Pass 1
  reads only ``Label`` and computes a stratified 70/30 index split
  (``random_state=42``, identical semantics to v2). Pass 2 streams the CSV in
  chunks and routes each row to ``X_train.csv`` / ``X_test.csv`` by that mask.

``LabelEncoder`` ordering and the 70/30 stratified split are otherwise the same
as v2. The v2 script is untouched.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==================================================
# CONFIG
# ==================================================

INPUT_FILE = "results/final_merged_dataset_coalesced.csv"

OUTPUT_DIR = "results/split_data_v3"
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

_header = [c.strip() for c in pd.read_csv(INPUT_FILE, nrows=0).columns]
_dtypes = {c: "float32" for c in _header if c != "Label"}

x_train_path = f"{OUTPUT_DIR}/X_train.csv"
x_test_path = f"{OUTPUT_DIR}/X_test.csv"

row0 = 0
first = True
for chunk in pd.read_csv(INPUT_FILE, dtype=_dtypes, chunksize=CHUNK_SIZE, low_memory=False):
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
