"""E7: source-separated generalisation -- train on CSE-CIC-IDS2018, test on CIC-IDS2017.

Source tagging (no new pipeline outputs needed):
  1. ``merge_datasets.py`` concatenates the CIC-IDS2017 folder, then the
     CSE-CIC-IDS2018 folder; ``coalesce_features.py`` and
     ``deduplicate_dataset_v4.py`` stream rows in order (keep-first). So the
     deduplicated file is a 2017 block followed by a 2018 block.
  2. Verify (1): in ``final_merged_dataset.csv`` the 2017-only column
     ``Destination Port`` is non-null for exactly the first N17 rows.
  3. Replay the dedup keep-first decisions (same file, chunk size and
     ``hash_pandas_object`` hashing as ``deduplicate_dataset_v4.py``) to get
     B = number of the first N17 coalesced rows that survive dedup; rows [0, B)
     of the deduplicated file are 2017-origin, rows [B, end) 2018-origin.
     Cross-check the per-class kept totals against ``results/dedup_v4/dedup_report.csv``.
  Because dedup is keep-first and 2017 comes first, no 2018 training row is an
  exact (features+label) duplicate of a 2017 test row.

Experiment:
  - Classes present in both sources: Benign, Bot, Brute Force, DoS/DDoS,
    Web Attack (PortScan absent from 2018; Infiltration has 36 rows in 2017).
  - ``Protocol`` dropped (0 for every 2017 row, so it encodes source) -> 69 features.
  - Train on ALL 2018 rows of those classes with the same recipe as v4:
    undersample classes >200k to 200k (``resample``, rs=42), SMOTE every class
    to 200k (rs=42), same model hyper-parameters (RF, XGB, MLP).
  - Test on ALL 2017 rows of those classes (natural distribution).
  - Control (RF/XGB): stratified 70:30 split of the same 2018 data, same recipe
    on the 70%, test on the 30% -- measures in-source performance, so the
    cross-source drop is attributable to the source change.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

# ==================================================
# CONFIG
# ==================================================

MERGED = "results/final_merged_dataset.csv"
COALESCED = "results/final_merged_dataset_coalesced.csv"
DEDUPED = "results/final_merged_dataset_deduped.csv"
DEDUP_REPORT = "results/dedup_v4/dedup_report.csv"
CHUNK_SIZE = 1_000_000
OUT_DIR = "results/revision/e7"
BOUNDARY_FILE = f"{OUT_DIR}/e7_source_boundary.json"
CLASSES = ["Benign", "Bot", "Brute Force", "DoS/DDoS", "Web Attack"]
DROP = ["Protocol"]
TARGET_SIZE = 200_000
RS = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ==================================================
# 1-3. SOURCE BOUNDARY
# ==================================================

if os.path.exists(BOUNDARY_FILE):
    boundary = json.load(open(BOUNDARY_FILE))
    print("Loaded boundary:", boundary)
else:
    t0 = time.time()
    print("Verifying 2017-first ordering in the merged file...", flush=True)
    n17, n_total, first_2018, after_2018_with_dport = 0, 0, None, 0
    for chunk in pd.read_csv(MERGED, usecols=["Destination Port"], chunksize=CHUNK_SIZE,
                             low_memory=False):
        has = chunk["Destination Port"].notna().to_numpy()
        if first_2018 is None:
            if has.all():
                n17 += len(has)
            else:
                k = int(np.argmin(has))
                n17 += k
                first_2018 = n_total + k
                after_2018_with_dport += int(has[k:].sum())
        else:
            after_2018_with_dport += int(has.sum())
        n_total += len(has)
    assert after_2018_with_dport == 0, "2017 rows are not a contiguous prefix"
    print(f"  N17 = {n17:,} of {n_total:,} merged rows ({time.time() - t0:.0f}s)", flush=True)

    print("Replaying keep-first dedup on the coalesced file...", flush=True)
    seen, kept17, kept_total, kept_by_class = set(), 0, 0, {}
    row0 = 0
    for chunk in pd.read_csv(COALESCED, chunksize=CHUNK_SIZE, low_memory=False):
        hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        keep = np.zeros(len(hashes), dtype=bool)
        for j, h in enumerate(hashes):
            if h not in seen:
                seen.add(h)
                keep[j] = True
        is17 = (np.arange(row0, row0 + len(chunk)) < n17)
        kept17 += int((keep & is17).sum())
        kept_total += int(keep.sum())
        for lab, c in chunk.loc[keep, "Label"].value_counts().items():
            kept_by_class[lab] = kept_by_class.get(lab, 0) + int(c)
        row0 += len(chunk)
    del seen
    assert row0 == n_total, "coalesced row count differs from merged"
    rep = pd.read_csv(DEDUP_REPORT).set_index("Class")["Rows_After"].to_dict()
    assert kept_by_class == rep, f"replayed dedup differs from report: {kept_by_class} vs {rep}"
    boundary = {"n17_merged": n17, "n_merged": n_total, "B_deduped_2017_rows": kept17,
                "n_deduped": kept_total}
    json.dump(boundary, open(BOUNDARY_FILE, "w"), indent=1)
    print("Boundary:", boundary, f"({time.time() - t0:.0f}s)", flush=True)

B = boundary["B_deduped_2017_rows"]

# ==================================================
# 4. LOAD BY SOURCE (two passes: labels, then rows)
# ==================================================

labels = pd.read_csv(DEDUPED, usecols=["Label"])["Label"].to_numpy()
pos = np.arange(len(labels))
in_cls = np.isin(labels, CLASSES)
src17 = pos < B

counts = pd.crosstab(pd.Series(labels[in_cls | ~in_cls], name="Label"),
                     pd.Series(np.where(src17, "CIC-IDS2017", "CSE-CIC-IDS2018"), name="Source"))
counts.to_csv(f"{OUT_DIR}/e7_deduped_counts_by_source.csv")
print(counts, flush=True)

# 2018 rows of the 5 classes, undersampled to 200k/class (same resample call as v4)
idx18 = np.flatnonzero(in_cls & ~src17)
keep18 = []
for c in CLASSES:
    p = idx18[labels[idx18] == c]
    if len(p) > TARGET_SIZE:
        p = resample(p, replace=False, n_samples=TARGET_SIZE, random_state=RS)
    keep18.append(np.sort(p))
keep18 = np.sort(np.concatenate(keep18))
idx17 = np.flatnonzero(in_cls & src17)

# control split on ALL 2018 rows of the 5 classes (before undersampling)
ctl_tr, ctl_te = train_test_split(idx18, test_size=0.30, stratify=labels[idx18], random_state=RS)
ctl_keep = []
for c in CLASSES:
    p = ctl_tr[labels[ctl_tr] == c]
    if len(p) > TARGET_SIZE:
        p = resample(p, replace=False, n_samples=TARGET_SIZE, random_state=RS)
    ctl_keep.append(p)
ctl_keep = np.sort(np.concatenate(ctl_keep))
# the control test set is large (Benign); evaluate on all of it
ctl_te = np.sort(ctl_te)

need = np.zeros(len(labels), dtype=np.uint8)
need[keep18] |= 1
need[idx17] |= 2
need[ctl_keep] |= 4
need[ctl_te] |= 8

header = [c.strip() for c in pd.read_csv(DEDUPED, nrows=0).columns]
feat = [c for c in header if c != "Label" and c not in DROP]
parts = {1: [], 2: [], 4: [], 8: []}
row0 = 0
for chunk in pd.read_csv(DEDUPED, dtype={c: "float32" for c in feat}, usecols=feat + ["Label"],
                         chunksize=CHUNK_SIZE, low_memory=False):
    m = need[row0:row0 + len(chunk)]
    for bit in parts:
        sel = (m & bit) > 0
        if sel.any():
            parts[bit].append(chunk.loc[sel])
    row0 += len(chunk)
    print(f"  {row0:,} / {len(labels):,}", flush=True)
data = {bit: pd.concat(v, ignore_index=True) for bit, v in parts.items()}
del parts

cls_id = {c: i for i, c in enumerate(CLASSES)}


def xy(df):
    X = df[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    return X.astype("float32"), df["Label"].map(cls_id).to_numpy()


def balance(X, y):
    sm = SMOTE(sampling_strategy={i: TARGET_SIZE for i in np.unique(y)}, random_state=RS)
    return sm.fit_resample(X, y)


def models(include_mlp):
    out = {
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
        "xgboost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist",
                                 eval_metric="mlogloss", n_jobs=-1),
    }
    if include_mlp:
        out["mlp"] = MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
                                   batch_size=2048, learning_rate="adaptive",
                                   learning_rate_init=0.001, max_iter=50, early_stopping=True,
                                   validation_fraction=0.1, n_iter_no_change=5, random_state=42)
    return out


def batched_predict(model, X, scaler=None, batch=200_000):
    Xv = X.to_numpy()
    out = []
    for i in range(0, len(Xv), batch):
        xb = Xv[i:i + batch]
        out.append(np.asarray(model.predict(scaler.transform(xb) if scaler is not None else xb)).ravel())
    return np.concatenate(out).astype(int)


overall, per_class, train_info = [], [], []


def run(setting, train_df, test_df, include_mlp):
    Xtr, ytr = xy(train_df)
    real = pd.Series(ytr).value_counts().sort_index()
    Xb, yb = balance(Xtr, ytr)
    for i, c in enumerate(CLASSES):
        train_info.append({"setting": setting, "class": c, "real_rows": int(real.get(i, 0)),
                           "balanced_rows": int((yb == i).sum()),
                           "pct_synthetic": 100 * (1 - real.get(i, 0) / (yb == i).sum())})
    Xte, yte = xy(test_df)
    scaler = StandardScaler().fit(Xb) if include_mlp else None
    for name, model in models(include_mlp).items():
        t0 = time.time()
        model.fit(scaler.transform(Xb) if name == "mlp" else Xb, yb)
        p = batched_predict(model, Xte, scaler if name == "mlp" else None)
        acc = accuracy_score(yte, p)
        _, _, wf, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
        _, _, mf, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
        overall.append({"setting": setting, "model": name, "accuracy": acc, "weighted_f1": wf,
                        "macro_f1": mf, "n_test": len(yte), "fit_s": round(time.time() - t0, 1)})
        rep = classification_report(yte, p, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        for c in CLASSES:
            per_class.append({"setting": setting, "model": name, "class": c,
                              "precision": rep[c]["precision"], "recall": rep[c]["recall"],
                              "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])})
        print(f"{setting:<26} {name:<14} acc={acc:.4f} wF1={wf:.4f} macroF1={mf:.4f}", flush=True)
        pd.DataFrame(overall).to_csv(f"{OUT_DIR}/e7_overall_noProtocol_rs42.csv", index=False)
        pd.DataFrame(per_class).to_csv(f"{OUT_DIR}/e7_per_class_noProtocol_rs42.csv", index=False)
        del model


run("train2018_test2017", data[1], data[2], include_mlp=True)
run("control_2018_70_30", data[4], data[8], include_mlp=False)
pd.DataFrame(train_info).to_csv(f"{OUT_DIR}/e7_training_composition.csv", index=False)
print("Done.")
