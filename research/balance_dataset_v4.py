"""v4 training-split balancing.

Serves the same role as ``balance_dataset_v3.py``: undersample any class above
``TARGET_SIZE`` down to it, then SMOTE every class up to ``TARGET_SIZE`` --
applied to the TRAINING SPLIT ONLY. Same two-pass undersample strategy,
``TARGET_SIZE=200_000``, ``random_state=42``, and SMOTE setup as v3 --
identical logic, just pointed at ``results/split_data_v4`` (the deduplicated,
leakage-free split) instead of ``results/split_data_v3``.

Because Phase 1 (``deduplicate_dataset_v4.py``) removed a large fraction of
Bot (48.5%) and DoS/DDoS (37.6%) rows as exact duplicates before the split,
the v4 training split starts with far fewer REAL rows for those two classes
than v3 did -- both now start below ``TARGET_SIZE`` (v3's Bot/DoS-DDoS train
counts were partly inflated by the same duplicate rows this pipeline removed),
so SMOTE is doing more synthetic-minority work for them here than in v3. See
the printed before/after report and ``training_balancing_report.csv`` for the
exact numbers.

The v3 script (and its float32-downcast-on-read Pass 2, which is safe here --
it operates purely within the training split, not across the train/test
boundary, so it isn't the leakage bug Phase 2 fixed) is left untouched, kept
identical to v3 per "same logic" above.
"""

import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# ==================================================
# CONFIG
# ==================================================

INPUT_DIR = "results/split_data_v4"
OUTPUT_DIR = "results/split_data_v4"

TARGET_SIZE = 200_000
RANDOM_STATE = 42
CHUNK_SIZE = 1_000_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# PASS 1: y_train -> per-class keep mask
# ==================================================

print("Pass 1: loading y_train...")

y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv")["Label"].to_numpy()
n_rows = len(y_train)
print(f"Training rows: {n_rows:,}")

print("\nTraining distribution BEFORE balancing:")
before = pd.Series(y_train).value_counts().sort_index()
print(before)

keep_mask = np.zeros(n_rows, dtype=bool)

print("\nApplying Random Undersampling...")
for label in sorted(np.unique(y_train)):
    pos = np.flatnonzero(y_train == label)
    if len(pos) > TARGET_SIZE:
        pos = resample(
            pos,
            replace=False,
            n_samples=TARGET_SIZE,
            random_state=RANDOM_STATE,
        )
        print(f"  {label}: {len(np.flatnonzero(y_train == label)):,} -> undersampled to {TARGET_SIZE:,}")
    else:
        print(f"  {label}: {len(pos):,} (kept)")
    keep_mask[pos] = True

# ==================================================
# PASS 2: stream X_train -> undersampled frame
# ==================================================

print("\nPass 2: streaming X_train...")

_header = [c.strip() for c in pd.read_csv(f"{INPUT_DIR}/X_train.csv", nrows=0).columns]
_dtypes = {c: "float32" for c in _header}

parts = []
row0 = 0
for chunk in pd.read_csv(
    f"{INPUT_DIR}/X_train.csv", dtype=_dtypes, chunksize=CHUNK_SIZE, low_memory=False
):
    chunk.columns = chunk.columns.str.strip()
    n = len(chunk)
    parts.append(chunk[keep_mask[row0:row0 + n]])
    row0 += n
    print(f"  {row0:,} / {n_rows:,} rows scanned")

X_under = pd.concat(parts, ignore_index=True)
y_under = y_train[keep_mask]
del parts

print("\nDistribution AFTER undersampling:")
print(pd.Series(y_under).value_counts().sort_index())

# ==================================================
# REMOVE NON-NUMERIC + CLEAN (parity with v2/v3)
# ==================================================

non_numeric = [c for c in X_under.columns if not pd.api.types.is_numeric_dtype(X_under[c])]
if non_numeric:
    print("\nDropping non-numeric columns:", non_numeric)
    X_under = X_under.drop(columns=non_numeric)

X_under = X_under.apply(pd.to_numeric, errors="coerce")
X_under = X_under.replace([np.inf, -np.inf], 0)
X_under = X_under.fillna(0)
print("\nRemaining NaN:", int(X_under.isnull().sum().sum()))

# ==================================================
# SMOTE
# ==================================================

print("\nApplying SMOTE...")

sampling_strategy = {cls: TARGET_SIZE for cls in sorted(np.unique(y_under))}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
X_balanced, y_balanced = smote.fit_resample(X_under, y_under)

print("\nDistribution AFTER SMOTE:")
print(pd.Series(y_balanced).value_counts().sort_index())

# ==================================================
# SAVE
# ==================================================

X_balanced.to_csv(f"{OUTPUT_DIR}/X_train_smote.csv", index=False)
pd.DataFrame(y_balanced, columns=["Label"]).to_csv(
    f"{OUTPUT_DIR}/y_train_smote.csv", index=False
)

report = pd.DataFrame({
    "Class": sorted(pd.Series(y_balanced).unique()),
    "Before": before.reindex(sorted(pd.Series(y_balanced).unique())).fillna(0).astype(int).values,
    "After_Undersample": pd.Series(y_under).value_counts().sort_index().values,
    "After_SMOTE": pd.Series(y_balanced).value_counts().sort_index().values,
})
report.to_csv(f"{OUTPUT_DIR}/training_balancing_report.csv", index=False)

print("\nSaved:")
print(f"  {OUTPUT_DIR}/X_train_smote.csv")
print(f"  {OUTPUT_DIR}/y_train_smote.csv")
print(f"  {OUTPUT_DIR}/training_balancing_report.csv")
print("\n" + report.to_string(index=False))
