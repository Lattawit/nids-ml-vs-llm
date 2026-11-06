"""E5: SMOTE alternative -- undersample majority classes, NO SMOTE, class weights.

Training rows = exactly the real rows ``balance_dataset_v4.py`` keeps after its
undersampling pass (same ``resample(..., random_state=42)`` call per class
>200k, same float32 streaming read), but the minority classes are NOT
oversampled. Imbalance is handled with inverse-frequency weights instead:
- Random Forest: ``class_weight="balanced"`` (also what the original uses).
- XGBoost: ``sample_weight = n / (n_classes * count[class])`` (no class_weight
  parameter in XGBoost).
Hyper-parameters otherwise identical to ``train_models_v4.py``. MLP skipped
(optional per plan). Models -> ``results/revision/e5/models_nosmote_weighted/``.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

# ==================================================
# CONFIG
# ==================================================

INPUT_DIR = "results/split_data_v4"
MODEL_DIR = "results/revision/e5/models_nosmote_weighted"
TARGET_SIZE = 200_000
RANDOM_STATE = 42
CHUNK_SIZE = 1_000_000
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# UNDERSAMPLE (identical to balance_dataset_v4.py pass 1 + 2)
# ==================================================

y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv")["Label"].to_numpy()
keep_mask = np.zeros(len(y_train), dtype=bool)
for label in sorted(np.unique(y_train)):
    pos = np.flatnonzero(y_train == label)
    if len(pos) > TARGET_SIZE:
        pos = resample(pos, replace=False, n_samples=TARGET_SIZE, random_state=RANDOM_STATE)
    keep_mask[pos] = True

_header = [c.strip() for c in pd.read_csv(f"{INPUT_DIR}/X_train.csv", nrows=0).columns]
parts, row0 = [], 0
for chunk in pd.read_csv(f"{INPUT_DIR}/X_train.csv", dtype={c: "float32" for c in _header},
                         chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    parts.append(chunk[keep_mask[row0:row0 + len(chunk)]])
    row0 += len(chunk)
    print(f"  {row0:,} / {len(y_train):,} scanned", flush=True)
X = pd.concat(parts, ignore_index=True)
y = y_train[keep_mask]
del parts

X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
counts = pd.Series(y).value_counts().sort_index()
print("Training rows per class (real only):\n", counts.to_string(), flush=True)

# cross-check against the committed balancing report
rep = pd.read_csv(f"{INPUT_DIR}/training_balancing_report.csv")
assert (rep["After_Undersample"].to_numpy() == counts.to_numpy()).all(), "undersample mismatch"

weights_per_class = len(y) / (len(counts) * counts)
sample_weight = weights_per_class.loc[y].to_numpy()
pd.DataFrame({"class": counts.index, "rows": counts.values,
              "weight": weights_per_class.values}).to_csv(
    f"{MODEL_DIR}/../e5_class_counts_weights.csv", index=False)

timings = {}
print("\nTraining Random Forest", flush=True)
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(X, y)
timings["random_forest"] = time.time() - t0
joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")
del rf

print("Training XGBoost", flush=True)
t0 = time.time()
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, subsample=0.8,
                    colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss", n_jobs=-1)
xgb.fit(X, y, sample_weight=sample_weight)
timings["xgboost"] = time.time() - t0
joblib.dump(xgb, f"{MODEL_DIR}/xgboost.pkl")

pd.DataFrame([{"model": k, "train_seconds": round(v, 2)} for k, v in timings.items()]).to_csv(
    f"{MODEL_DIR}/../e5_train_times.csv", index=False)
print(timings, "\nDone.")
