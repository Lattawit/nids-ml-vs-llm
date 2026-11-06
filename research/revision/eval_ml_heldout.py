"""Evaluate model sets on the full v4 held-out test split (natural distribution).

Same loading/cleaning/batched-predict as ``research/evaluate_models_v4.py``
(untouched): drop non-numeric, coerce, inf/NaN -> 0, float32, no clipping,
200k-row predict batches, MLP on ``scaler.transform``. ``v4_70feat`` is
re-evaluated as a reproduction check against
``results/model_evaluation_v4/overall_results_v4.csv``. Model sets missing
on disk are skipped.
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(__file__))
from prompts_v4 import label_names, selected_features  # noqa: E402

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# ==================================================
# CONFIG
# ==================================================

TEST_DIR = "results/split_data_v4"
OUT_DIR = "results/revision/heldout"
MODEL_SETS = [
    ("v4_70feat", "models/v4", "all", ["random_forest", "xgboost", "mlp"]),
    ("e2_22feat", "results/revision/e2/models_22feat", "22", ["random_forest", "xgboost", "mlp"]),
    ("e5_nosmote_weighted", "results/revision/e5/models_nosmote_weighted", "all",
     ["random_forest", "xgboost"]),
]
PREDICT_BATCH = 200_000

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading held-out test set...", flush=True)
X = pd.read_csv(f"{TEST_DIR}/X_test.csv", low_memory=False)
X.columns = X.columns.str.strip()
y = pd.read_csv(f"{TEST_DIR}/y_test.csv")["Label"].to_numpy()
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
X = X.drop(columns=non_numeric)
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
print("Test shape:", X.shape, flush=True)


def batched_predict(model, Xv, scaler=None):
    out = []
    for i in range(0, len(Xv), PREDICT_BATCH):
        xb = Xv[i:i + PREDICT_BATCH]
        if scaler is not None:
            xb = scaler.transform(xb)
        out.append(np.asarray(model.predict(xb)).ravel())
    return np.concatenate(out).astype(int)


overall, per_class = [], []
for set_name, model_dir, feats, models in MODEL_SETS:
    if not all(os.path.exists(f"{model_dir}/{m}.pkl") for m in models):
        print(f"skip {set_name}: models not found in {model_dir}")
        continue
    if os.path.exists(f"{OUT_DIR}/heldout_overall_{set_name}.csv"):
        print(f"skip {set_name}: already evaluated")
        continue
    Xv = (X[selected_features] if feats == "22" else X).to_numpy()
    scaler = joblib.load(f"{model_dir}/scaler.pkl") if "mlp" in models else None
    for m in models:
        model = joblib.load(f"{model_dir}/{m}.pkl")
        p = batched_predict(model, Xv, scaler if m == "mlp" else None)
        del model
        acc = accuracy_score(y, p)
        _, _, wf, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
        mp, mr, mf, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        overall.append({"model_set": set_name, "model": m, "accuracy": acc, "weighted_f1": wf,
                        "macro_precision": mp, "macro_recall": mr, "macro_f1": mf, "n": len(y)})
        rep = classification_report(y, p, labels=list(label_names),
                                    target_names=list(label_names.values()),
                                    output_dict=True, zero_division=0)
        for cls in label_names.values():
            per_class.append({"model_set": set_name, "model": m, "class": cls,
                              "precision": rep[cls]["precision"], "recall": rep[cls]["recall"],
                              "f1": rep[cls]["f1-score"], "support": int(rep[cls]["support"])})
        print(f"{set_name:<20} {m:<14} acc={acc:.4f} wF1={wf:.4f} macroF1={mf:.4f}", flush=True)

    # append-safe: one file per model set
    pd.DataFrame([r for r in overall if r["model_set"] == set_name]).to_csv(
        f"{OUT_DIR}/heldout_overall_{set_name}.csv", index=False)
    pd.DataFrame([r for r in per_class if r["model_set"] == set_name]).to_csv(
        f"{OUT_DIR}/heldout_per_class_{set_name}.csv", index=False)

print("Saved:", OUT_DIR)
