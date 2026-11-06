"""E6: 5-fold stratified cross-validation of the ML models (R1#2).

Data: the deduplicated, coalesced dataset (``results/final_merged_dataset_deduped.csv``,
14,439,966 rows, 70 features, 7 classes) -- the same data the v4 70:30 split was
drawn from. ``StratifiedKFold(n_splits=5, shuffle=True, random_state=42)``.

Inside each fold, on the training part ONLY (test fold keeps the natural
distribution), the v4 recipe is applied unchanged:
- undersample every class > 200k to 200k (``resample(..., random_state=42)``),
- SMOTE every class up to 200k (``SMOTE(random_state=42)``),
- RF / XGBoost on raw features, MLP on ``StandardScaler`` fit on the balanced fold.
Hyper-parameters identical to ``train_models_v4.py``. Cleaning as the held-out
evaluation: float32, inf/NaN -> 0, no clip.

Resumable: one row per (fold, model) in ``e6_cv5_overall_rs42.csv``; finished
pairs are skipped on restart. Set ``FOLDS`` / ``MODELS`` to a subset to time a
single fold first.
"""

import os
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

# ==================================================
# CONFIG
# ==================================================

DEDUPED = "results/final_merged_dataset_deduped.csv"
OUT_DIR = "results/revision/e6"
CHUNK_SIZE = 1_000_000
CLASSES = ["Benign", "Bot", "Brute Force", "DoS/DDoS", "Infiltration", "PortScan", "Web Attack"]
TARGET_SIZE = 200_000
RS = 42
FOLDS = [0, 1, 2, 3, 4]
MODELS = ["random_forest", "xgboost", "mlp"]
os.makedirs(OUT_DIR, exist_ok=True)
OVERALL = f"{OUT_DIR}/e6_cv5_overall_rs42.csv"
PER_CLASS = f"{OUT_DIR}/e6_cv5_per_class_rs42.csv"

done = set()
if os.path.exists(OVERALL):
    prev = pd.read_csv(OVERALL)
    done = set(zip(prev["fold"], prev["model"]))
todo = [(f, m) for f in FOLDS for m in MODELS if (f, m) not in done]
if not todo:
    print("Nothing to do.")
    raise SystemExit

# ==================================================
# LOAD (once; ~4 GB float32)
# ==================================================

t_load = time.time()
header = [c.strip() for c in pd.read_csv(DEDUPED, nrows=0).columns]
feat = [c for c in header if c != "Label"]
assert len(feat) == 70, len(feat)
cls_id = {c: i for i, c in enumerate(CLASSES)}
Xs, ys, n = [], [], 0
for chunk in pd.read_csv(DEDUPED, dtype={c: "float32" for c in feat}, chunksize=CHUNK_SIZE,
                         low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    X = chunk[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    Xs.append(X.to_numpy(dtype=np.float32))
    ys.append(chunk["Label"].map(cls_id).to_numpy())
    n += len(chunk)
    print(f"  loaded {n:,}", flush=True)
X_all = np.concatenate(Xs)
y_all = np.concatenate(ys).astype(np.int64)
del Xs, ys
assert len(y_all) == 14_439_966 and not np.isnan(y_all.astype(float)).any()
print(f"Loaded {X_all.shape} in {time.time() - t_load:.0f}s", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
splits = list(skf.split(np.zeros(len(y_all)), y_all))


def make(name):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1,
                                      random_state=42)
    if name == "xgboost":
        return XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
                             n_jobs=-1)
    return MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
                         batch_size=2048, learning_rate="adaptive", learning_rate_init=0.001,
                         max_iter=50, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=5, random_state=42)


def batched_predict(model, X, scaler=None, batch=200_000):
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        out.append(np.asarray(model.predict(scaler.transform(xb) if scaler is not None else xb)).ravel())
    return np.concatenate(out).astype(int)


def append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


for fold in sorted({f for f, _ in todo}):
    t_fold = time.time()
    tr, te = splits[fold]
    y_tr = y_all[tr]
    keep = []
    for c in sorted(np.unique(y_tr)):
        pos = tr[y_tr == c]
        if len(pos) > TARGET_SIZE:
            pos = resample(pos, replace=False, n_samples=TARGET_SIZE, random_state=RS)
        keep.append(pos)
    keep = np.sort(np.concatenate(keep))
    sm = SMOTE(sampling_strategy={c: TARGET_SIZE for c in range(len(CLASSES))}, random_state=RS)
    Xb, yb = sm.fit_resample(X_all[keep], y_all[keep])
    real = pd.Series(y_all[keep]).value_counts().sort_index()
    append(f"{OUT_DIR}/e6_cv5_fold_composition_rs42.csv",
           [{"fold": fold, "class": CLASSES[c], "train_real_available": int((y_tr == c).sum()),
             "real_in_balanced": int(real.get(c, 0)), "balanced_rows": int((yb == c).sum()),
             "test_rows": int((y_all[te] == c).sum())} for c in range(len(CLASSES))])
    Xte, yte = X_all[te], y_all[te]
    print(f"\nFold {fold}: balanced {Xb.shape}, test {Xte.shape} ({time.time() - t_fold:.0f}s)",
          flush=True)
    scaler = None
    for name in [m for f, m in todo if f == fold]:
        t0 = time.time()
        model = make(name)
        if name == "mlp":
            scaler = StandardScaler().fit(Xb)
            model.fit(scaler.transform(Xb), yb)
        else:
            model.fit(Xb, yb)
        fit_s = time.time() - t0
        p = batched_predict(model, Xte, scaler if name == "mlp" else None)
        acc = accuracy_score(yte, p)
        _, _, wf, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
        _, _, mf, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
        rep = classification_report(yte, p, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        append(PER_CLASS, [{"fold": fold, "model": name, "class": c,
                            "precision": rep[c]["precision"], "recall": rep[c]["recall"],
                            "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])}
                           for c in CLASSES])
        append(OVERALL, [{"fold": fold, "model": name, "accuracy": acc, "weighted_f1": wf,
                          "macro_f1": mf, "n_test": len(yte), "fit_s": round(fit_s, 1),
                          "total_s": round(time.time() - t0, 1)}])
        print(f"fold {fold} {name:<14} acc={acc:.4f} wF1={wf:.4f} macroF1={mf:.4f} "
              f"fit={fit_s:.0f}s total={time.time() - t0:.0f}s", flush=True)
        del model
    print(f"Fold {fold} done in {time.time() - t_fold:.0f}s", flush=True)

res = pd.read_csv(OVERALL)
summ = res.groupby("model")[["accuracy", "weighted_f1", "macro_f1"]].agg(["mean", "std", "min", "max"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ["folds"] = res.groupby("model").size()
summ.to_csv(f"{OUT_DIR}/e6_cv5_summary_rs42.csv")
print(summ.round(4).to_string())
