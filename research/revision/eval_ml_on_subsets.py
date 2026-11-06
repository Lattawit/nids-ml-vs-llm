"""Evaluate trained ML model sets on the five revision subsets (E1, E2, E5).

Same numeric cleaning as ``evaluate_models_v4_on_llm_subset.py`` (coerce,
inf->0, NaN->0, clip +-1e6); MLP gets ``scaler.transform``. Inference time is
the median of 5 ``.predict()`` calls on the 700 rows, as in
``measure_inference_time_v4.py``. Writes per-row predictions (for McNemar) and
per-seed overall + per-class metrics. Model sets that are missing on disk are
skipped with a message, so this can be re-run as E2/E5 models appear.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# ==================================================
# CONFIG
# ==================================================

SEEDS = [42, 1, 2, 3, 4]
SUBSET_TMPL = "results/revision/setup/llm_subset_v4_seed{seed}.csv"
FEATURES_22 = None  # filled from prompts_v4 below

MODEL_SETS = [
    # name, model dir, feature set, which models
    ("v4_70feat", "models/v4", "all", ["random_forest", "xgboost", "mlp"]),
    ("e2_22feat", "results/revision/e2/models_22feat", "22", ["random_forest", "xgboost", "mlp"]),
    ("e5_nosmote_weighted", "results/revision/e5/models_nosmote_weighted", "all",
     ["random_forest", "xgboost"]),
]
OUT_DIR = "results/revision/ml_subsets"
N_REPEATS = 5

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from prompts_v4 import label_names, selected_features  # noqa: E402

FEATURES_22 = selected_features
os.makedirs(OUT_DIR, exist_ok=True)


def clean(X):
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    return X.clip(-1e6, 1e6)


overall, per_class, timing = [], [], []

for set_name, model_dir, feats, models in MODEL_SETS:
    if not all(os.path.exists(f"{model_dir}/{m}.pkl") for m in models):
        print(f"skip {set_name}: models not found in {model_dir}")
        continue
    loaded = {m: joblib.load(f"{model_dir}/{m}.pkl") for m in models}
    scaler = joblib.load(f"{model_dir}/scaler.pkl") if "mlp" in models else None

    for seed in SEEDS:
        df = pd.read_csv(SUBSET_TMPL.format(seed=seed))
        y = df["Label"].values
        X = clean(df.drop(columns=["Label"]))
        if feats == "22":
            X = X[FEATURES_22]
        preds = pd.DataFrame({"row": np.arange(len(y)), "true": y})

        for m, model in loaded.items():
            X_in = scaler.transform(X) if m == "mlp" else X
            times = []
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                p = model.predict(X_in)
                times.append(time.perf_counter() - t0)
            p = np.asarray(p).ravel().astype(int)
            preds[m] = p

            acc = accuracy_score(y, p)
            _, _, wf, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
            mp, mr, mf, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
            overall.append({"model_set": set_name, "model": m, "seed": seed, "accuracy": acc,
                            "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
                            "weighted_f1": wf, "median_predict_s": float(np.median(times))})
            rep = classification_report(y, p, labels=list(label_names),
                                        target_names=list(label_names.values()),
                                        output_dict=True, zero_division=0)
            for cls in label_names.values():
                per_class.append({"model_set": set_name, "model": m, "seed": seed, "class": cls,
                                  "precision": rep[cls]["precision"], "recall": rep[cls]["recall"],
                                  "f1": rep[cls]["f1-score"]})
            print(f"{set_name:<20} {m:<14} seed={seed:<3} acc={acc:.4f} macroF1={mf:.4f} "
                  f"t={np.median(times):.4f}s", flush=True)

        preds.to_csv(f"{OUT_DIR}/predictions_{set_name}_seed{seed}.csv", index=False)

pd.DataFrame(overall).to_csv(f"{OUT_DIR}/overall_by_seed.csv", index=False)
pd.DataFrame(per_class).to_csv(f"{OUT_DIR}/per_class_by_seed.csv", index=False)
print("Saved:", OUT_DIR)
