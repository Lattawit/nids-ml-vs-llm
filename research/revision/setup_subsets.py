"""Revision setup: E1 subsets, per-seed few-shot examples, E4 retrieval matrix.

1. Five balanced 700-row subsets (100/class) from the v4 test split.
   Seed 42 uses exactly the original ``create_llm_subset_v4.py`` draw
   (per-class ``RandomState(42)``) and must come out byte-identical to
   ``results/llm/llm_subset_v4.csv``. Seeds 1-4 draw per class from test rows
   not used by any earlier seed, so all five subsets are disjoint.
2. One real pre-SMOTE training row per class per seed, drawn with the original
   few-shot scheme (per-class ``RandomState(seed)`` over ``y_train``) and
   rendered through the original code path (chunked ``usecols`` read,
   ``chunk.iloc`` row). Seed 42 reproduces the original examples.
3. The 22 LLM features of every real training row (pre-undersampling, pre-SMOTE)
   as float64 + labels, cached for the E4 retrieval index.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from prompts_v4 import render_example, selected_features  # noqa: E402

# ==================================================
# CONFIG
# ==================================================

SEEDS = [42, 1, 2, 3, 4]
ORIGINAL_SEED = 42
SAMPLES_PER_CLASS = 100

SPLIT_DIR = "results/split_data_v4"
ORIGINAL_SUBSET = "results/llm/llm_subset_v4.csv"
OUT_DIR = "results/revision/setup"
CACHE_DIR = "results/revision/cache"
CHUNK_SIZE = 1_000_000

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================================================
# 1. PICK SUBSET POSITIONS (disjoint across seeds)
# ==================================================

print("Loading y_test...")
y_test = pd.read_csv(f"{SPLIT_DIR}/y_test.csv")["Label"].to_numpy()

picks = {s: {} for s in SEEDS}
for label in sorted(np.unique(y_test)):
    pos = np.flatnonzero(y_test == label)
    used = np.array([], dtype=pos.dtype)
    for s in SEEDS:
        pool = pos if s == ORIGINAL_SEED else np.setdiff1d(pos, used)
        rng = np.random.RandomState(s)
        chosen = np.sort(rng.choice(pool, size=SAMPLES_PER_CLASS, replace=False))
        picks[s][int(label)] = chosen
        used = np.union1d(used, chosen)
    print(f"  class {label}: {len(pos):,} test rows, {len(used)} used across seeds")

all_pos = np.concatenate([p for s in SEEDS for p in picks[s].values()])
assert len(all_pos) == len(set(all_pos.tolist())) == len(SEEDS) * 7 * SAMPLES_PER_CLASS, \
    "subsets are not disjoint"

manifest = pd.DataFrame(
    [(s, lab, int(p)) for s in SEEDS for lab in picks[s] for p in picks[s][lab]],
    columns=["seed", "label", "test_row_position"],
)
manifest.to_csv(f"{OUT_DIR}/subset_manifest_seeds42-1-2-3-4.csv", index=False)

# ==================================================
# 2. STREAM X_test -> subset rows
# ==================================================

wanted = np.sort(all_pos)
print("\nStreaming X_test...")
rows_by_pos = {}
row0 = 0
for chunk in pd.read_csv(f"{SPLIT_DIR}/X_test.csv", chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    lo, hi = row0, row0 + len(chunk)
    for p in wanted[(wanted >= lo) & (wanted < hi)]:
        rows_by_pos[int(p)] = chunk.iloc[p - lo]
    row0 = hi
    print(f"  {row0:,} / {len(y_test):,} scanned", flush=True)

for s in SEEDS:
    frames = []
    for label in sorted(picks[s]):
        block = pd.DataFrame([rows_by_pos[int(p)] for p in picks[s][label]]).reset_index(drop=True)
        block["Label"] = label
        frames.append(block)
    out = f"{OUT_DIR}/llm_subset_v4_seed{s}.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print("Saved:", out)

with open(ORIGINAL_SUBSET, "rb") as a, open(f"{OUT_DIR}/llm_subset_v4_seed{ORIGINAL_SEED}.csv", "rb") as b:
    identical = a.read() == b.read()
print(f"\nSeed {ORIGINAL_SEED} subset byte-identical to {ORIGINAL_SUBSET}: {identical}")
assert identical

# ==================================================
# 3. FEW-SHOT REFERENCE ROWS + E4 MATRIX (one pass over X_train)
# ==================================================

print("\nLoading y_train...")
y_train = pd.read_csv(f"{SPLIT_DIR}/y_train.csv")["Label"].to_numpy()
n_train = len(y_train)

ref_pos = {s: {} for s in SEEDS}  # seed -> {label: train row position}
for s in SEEDS:
    for label in sorted(np.unique(y_train)):
        pos = np.flatnonzero(y_train == label)
        rng = np.random.RandomState(s)
        ref_pos[s][int(label)] = int(rng.choice(pos, size=1)[0])

needed = {p for s in SEEDS for p in ref_pos[s].values()}
ref_text = {}  # train position -> rendered example
X_real = np.empty((n_train, len(selected_features)), dtype=np.float64)

print("Streaming X_train (22 features)...")
row0 = 0
for chunk in pd.read_csv(
    f"{SPLIT_DIR}/X_train.csv", usecols=selected_features, chunksize=CHUNK_SIZE, low_memory=False
):
    chunk.columns = chunk.columns.str.strip()
    n = len(chunk)
    for p in needed:
        if row0 <= p < row0 + n:
            ref_text[p] = render_example(chunk.iloc[p - row0])
    X_real[row0:row0 + n] = chunk[selected_features].to_numpy(dtype=np.float64)
    row0 += n
    print(f"  {row0:,} / {n_train:,} scanned", flush=True)
assert row0 == n_train

refs = {
    str(s): {
        "train_row_position": {str(lab): p for lab, p in ref_pos[s].items()},
        "examples": {str(lab): ref_text[p] for lab, p in ref_pos[s].items()},
    }
    for s in SEEDS
}
with open(f"{OUT_DIR}/fewshot_reference_examples_seeds42-1-2-3-4.json", "w") as f:
    json.dump(refs, f, indent=1)

# rendering from the cached float64 matrix must match the original code path,
# because E4 renders retrieved examples from this matrix
for p, txt in ref_text.items():
    alt = render_example(pd.Series(X_real[p], index=selected_features))
    assert alt == txt, f"matrix rendering differs at train row {p}:\n{alt}\nvs\n{txt}"
print("Matrix-rendered examples match the original rendering path.")

np.save(f"{CACHE_DIR}/X_train_real_22feat_float64.npy", X_real)
np.save(f"{CACHE_DIR}/y_train_real.npy", y_train)
print("Saved E4 cache:", X_real.shape)
print("\nDone.")
