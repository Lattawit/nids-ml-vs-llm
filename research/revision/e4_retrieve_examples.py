"""E4: dynamic few-shot retrieval + retrieval-only baselines.

For every test row of the five revision subsets, find the nearest REAL
(pre-SMOTE, pre-undersampling) training row of EACH class, by Euclidean
distance on the 22 prompt features after the subset cleaning (inf/NaN -> 0,
clip +-1e6) and standardisation (StandardScaler fit on all real training
rows). The 7 retrieved rows replace the fixed few-shot examples; they are
rendered with their raw values exactly like the original examples.

Baselines written alongside (per seed):
  (a) 1-NN over the 7 retrieved rows = the class of the closest one. This is
      identical to a global 1-NN over all real training rows.
  (b) k-NN (k=5, majority vote, sklearn default tie-break) on the same real
      training rows / same standardisation.
  (b') k-NN (k=5) fit on the balanced SMOTE training set, standardised with
      a scaler fit on that set (optional variant, as requested).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from prompts_v4 import label_names, render_example, selected_features  # noqa: E402

# ==================================================
# CONFIG
# ==================================================

SEEDS = [42, 1, 2, 3, 4]
K = 5
SUBSET_TMPL = "results/revision/setup/llm_subset_v4_seed{seed}.csv"
CACHE_DIR = "results/revision/cache"
SMOTE_X = "results/split_data_v4/X_train_smote.csv"
SMOTE_Y = "results/split_data_v4/y_train_smote.csv"
OUT_DIR = "results/revision/e4"
os.makedirs(OUT_DIR, exist_ok=True)


def clean(a):
    a = np.where(np.isfinite(a), a, 0.0)
    return np.clip(a, -1e6, 1e6)


t_all = time.time()
X_raw = np.load(f"{CACHE_DIR}/X_train_real_22feat_float64.npy")
y_real = np.load(f"{CACHE_DIR}/y_train_real.npy")
print("Real training rows:", X_raw.shape, flush=True)

scaler = StandardScaler().fit(clean(X_raw))
Z = scaler.transform(clean(X_raw)).astype(np.float64)

subsets = {s: pd.read_csv(SUBSET_TMPL.format(seed=s)) for s in SEEDS}
Q = {s: scaler.transform(clean(df[selected_features].to_numpy(dtype=np.float64)))
     for s, df in subsets.items()}

# ---- per-class nearest real row ------------------------------------------
nearest_pos = {s: np.zeros((700, 7), dtype=np.int64) for s in SEEDS}
nearest_dist = {s: np.zeros((700, 7)) for s in SEEDS}
for c in range(7):
    idx = np.flatnonzero(y_real == c)
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Z[idx])
    for s in SEEDS:
        d, j = nn.kneighbors(Q[s])
        nearest_pos[s][:, c] = idx[j[:, 0]]
        nearest_dist[s][:, c] = d[:, 0]
    print(f"class {c} ({label_names[c]}): {len(idx):,} rows, {time.time() - t0:.1f}s", flush=True)

# ---- baseline (b): k-NN on real rows -------------------------------------
t0 = time.time()
knn_real = KNeighborsClassifier(n_neighbors=K, algorithm="brute", n_jobs=-1).fit(Z, y_real)
pred_knn_real = {s: knn_real.predict(Q[s]) for s in SEEDS}
print(f"k-NN (real rows) done in {time.time() - t0:.1f}s", flush=True)
del knn_real

# ---- baseline (b'): k-NN on the SMOTE-balanced set -----------------------
t0 = time.time()
Xs = pd.read_csv(SMOTE_X, usecols=selected_features)[selected_features].to_numpy(dtype=np.float64)
ys = pd.read_csv(SMOTE_Y)["Label"].to_numpy()
sc_s = StandardScaler().fit(clean(Xs))
knn_s = KNeighborsClassifier(n_neighbors=K, algorithm="brute", n_jobs=-1).fit(sc_s.transform(clean(Xs)), ys)
pred_knn_smote = {
    s: knn_s.predict(sc_s.transform(clean(subsets[s][selected_features].to_numpy(dtype=np.float64))))
    for s in SEEDS}
print(f"k-NN (SMOTE set) done in {time.time() - t0:.1f}s", flush=True)
del knn_s, Xs

# ---- write retrieved examples + baselines --------------------------------
metrics = []
for s in SEEDS:
    y = subsets[s]["Label"].to_numpy()
    rows = {}
    for i in range(700):
        rows[str(i)] = {
            "examples": {str(c): render_example(pd.Series(X_raw[nearest_pos[s][i, c]],
                                                          index=selected_features))
                         for c in range(7)},
            "train_row_position": {str(c): int(nearest_pos[s][i, c]) for c in range(7)},
            "distance": {str(c): float(nearest_dist[s][i, c]) for c in range(7)},
            "nearest_class": int(np.argmin(nearest_dist[s][i])),
        }
    with open(f"{OUT_DIR}/e4_retrieved_examples_seed{s}.json", "w") as f:
        json.dump({"seed": s, "scaler": "StandardScaler on all real train rows, cleaned",
                   "rows": rows}, f)

    pred_1nn = nearest_dist[s].argmin(axis=1)
    preds = pd.DataFrame({"row": np.arange(700), "true": y, "retrieval_1nn": pred_1nn,
                          f"knn{K}_real": pred_knn_real[s], f"knn{K}_smote": pred_knn_smote[s]})
    preds.to_csv(f"{OUT_DIR}/e4_baseline_predictions_seed{s}.csv", index=False)

    for name in ["retrieval_1nn", f"knn{K}_real", f"knn{K}_smote"]:
        p = preds[name].to_numpy()
        mp, mr, mf, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        rep = classification_report(y, p, labels=list(label_names), output_dict=True, zero_division=0)
        row = {"baseline": name, "seed": s, "accuracy": accuracy_score(y, p), "macro_f1": mf}
        row.update({f"f1_{label_names[c]}": rep[str(c)]["f1-score"] for c in label_names})
        metrics.append(row)
        print(f"seed {s:<3} {name:<14} acc={row['accuracy']:.4f} macroF1={mf:.4f}", flush=True)

pd.DataFrame(metrics).to_csv(f"{OUT_DIR}/e4_baseline_metrics_by_seed.csv", index=False)
print(f"\nDone in {time.time() - t_all:.1f}s")
