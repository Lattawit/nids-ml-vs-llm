"""v4 LLM evaluation subset.

Serves the same role as ``create_llm_subset_v3.py``: stratified 100 rows/class
(700 total) drawn from the held-out test split, saved as the LLM benchmark +
like-for-like ML evaluation fixture. Differences from v3:

- Source is ``results/split_data_v4`` (deduplicated before the split, zero
  cross-split row overlap -- see ``deduplicate_dataset_v4.py`` /
  ``train_test_split_pipeline_v4.py``) instead of ``results/split_data_v3``.
- Output is ``results/llm/llm_subset_v4.csv`` so ``llm_subset_v3.csv`` (and
  the app's ``llm_subset.csv``) are untouched.

Same ``SAMPLES_PER_CLASS=100``, ``random_state=42``, per-class-RNG picking,
and streamed X_test extraction as v3. The v3 script is untouched.
"""

import os

import numpy as np
import pandas as pd

SAMPLES_PER_CLASS = 100
RANDOM_STATE = 42

INPUT_DIR = "results/split_data_v4"
OUTPUT_FILE = "results/llm/llm_subset_v4.csv"
CHUNK_SIZE = 1_000_000

os.makedirs("results/llm", exist_ok=True)

# ==================================================
# PICK ROW POSITIONS PER CLASS (from y_test)
# ==================================================

print("Loading y_test...")
y_test = pd.read_csv(f"{INPUT_DIR}/y_test.csv")["Label"].to_numpy()
print(f"Test rows: {len(y_test):,}")

picks: dict[int, np.ndarray] = {}
for label in sorted(np.unique(y_test)):
    pos = np.flatnonzero(y_test == label)
    if len(pos) < SAMPLES_PER_CLASS:
        raise SystemExit(
            f"class {label} has only {len(pos)} test rows (< {SAMPLES_PER_CLASS})"
        )
    rng = np.random.RandomState(RANDOM_STATE)
    picks[label] = np.sort(rng.choice(pos, size=SAMPLES_PER_CLASS, replace=False))
    print(f"  class {label}: picked {SAMPLES_PER_CLASS} of {len(pos):,}")

wanted = np.concatenate(list(picks.values()))
wanted_set = set(wanted.tolist())

# ==================================================
# STREAM X_test AND EXTRACT THE PICKED ROWS
# ==================================================

print("\nStreaming X_test...")
rows_by_pos: dict[int, pd.Series] = {}
row0 = 0
for chunk in pd.read_csv(f"{INPUT_DIR}/X_test.csv", chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    lo, hi = row0, row0 + len(chunk)
    hits = [p for p in wanted if lo <= p < hi]
    for p in hits:
        rows_by_pos[p] = chunk.iloc[p - lo]
    row0 = hi
    print(f"  {row0:,} / {len(y_test):,} scanned")

# ==================================================
# ASSEMBLE (class-grouped, matching v2/v3 order)
# ==================================================

frames = []
for label in sorted(picks):
    block = pd.DataFrame([rows_by_pos[p] for p in picks[label]]).reset_index(drop=True)
    block["Label"] = label
    frames.append(block)

df_subset = pd.concat(frames, ignore_index=True)

df_subset.to_csv(OUTPUT_FILE, index=False)

print(f"\nSubset shape: {df_subset.shape}")
print("Class counts:", df_subset["Label"].value_counts().sort_index().to_dict())
print("Saved:", OUTPUT_FILE)
