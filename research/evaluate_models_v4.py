"""v4 held-out test-set evaluation.

Extends ``evaluate_models_v3.py``'s pattern with a third model/test config so
the paper has the full v2 -> v3 -> v4 comparison in one place: how much of
v2's inflated performance was the dataset-origin leakage + balanced-test-set
bug (fixed in v3), and how much of v3's still-inflated leakage was the
cross-split exact-duplicate bug (fixed in v4, see ``deduplicate_dataset_v4.py``
/ ``train_test_split_pipeline_v4.py``).

Three model sets, scored identically here (not read back from the v3 run) so
methodology is guaranteed consistent across all three rows:

- **v4**: ``models/v4`` on ``results/split_data_v4`` (deduplicated before the
  split -- zero cross-split row overlap, confirmed by
  ``check_cross_split_duplicates_v4.py``).
- **v3**: ``models/v3`` on ``results/split_data_v3`` (coalesced features, no
  pre-split balancing, but train/test still had duplicate-driven overlap).
- **v2**: ``models/v2`` on ``results/split_data_v2`` (original 130-feature
  half-zeroed schema, artificially 1:1-balanced test set) -- the original
  baseline row.

Because the v3/v4 test sets are natural-distribution (~85%/88% Benign),
MACRO-averaged precision/recall/F1 are reported alongside weighted-average and
accuracy: weighted/accuracy alone hide minority-class (Web Attack, PortScan,
Infiltration) performance.

Outputs -> ``results/model_evaluation_v4/``:
    overall_results_{v4,v3,v2}.csv
    class_level_results_{v4,v3,v2}.csv
    predictions_{v4,v3,v2}.csv
    overall_results_comparison.csv   (all three, side by side)
    confusion_matrices/*.png
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# models were fitted on DataFrames; we predict on ndarray batches for memory
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

RESULT_DIR = "results/model_evaluation_v4"
CM_DIR = f"{RESULT_DIR}/confusion_matrices"
os.makedirs(CM_DIR, exist_ok=True)

# LabelEncoder order is identical in split_data_v2, split_data_v3, split_data_v4
LABEL_NAMES = {
    0: "Benign",
    1: "Bot",
    2: "Brute Force",
    3: "DoS/DDoS",
    4: "Infiltration",
    5: "PortScan",
    6: "Web Attack",
}
TARGET_NAMES = list(LABEL_NAMES.values())

CONFIGS = [
    {"tag": "v2", "model_dir": "models/v2", "test_dir": "results/split_data_v2"},
    {"tag": "v3", "model_dir": "models/v3", "test_dir": "results/split_data_v3"},
    {"tag": "v4", "model_dir": "models/v4", "test_dir": "results/split_data_v4"},
]


def load_clean_test(test_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load X_test/y_test and apply the same numeric cleaning used in training
    (drop non-numeric identifier columns, coerce, inf/NaN -> 0)."""
    X = pd.read_csv(f"{test_dir}/X_test.csv", low_memory=False)
    X.columns = X.columns.str.strip()
    y = pd.read_csv(f"{test_dir}/y_test.csv")["Label"].to_numpy()

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X = X.drop(columns=non_numeric)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    return X.astype("float32"), y


PREDICT_BATCH = 200_000


def _batched_predict(model, X, scaled, scaler):
    """Predict in row batches. The test set has millions of rows and the MLP
    forward pass materialises (n_rows x 512) float64 -- full-array predict OOMs."""
    Xv = X.to_numpy()
    out = []
    for i in range(0, len(Xv), PREDICT_BATCH):
        xb = Xv[i:i + PREDICT_BATCH]
        if scaled:
            xb = scaler.transform(xb)
        out.append(model.predict(xb))
    return np.concatenate(out)


def evaluate_one(name: str, model, X, y_true, scaled: bool, scaler, overall, per_class, preds):
    y_pred = _batched_predict(model, X, scaled, scaler)
    preds[name] = y_pred

    acc = accuracy_score(y_true, y_pred)
    w_p, w_r, w_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    m_p, m_r, m_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"\n--- {name} ---")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Weighted P/R/F1     : {w_p:.4f} / {w_r:.4f} / {w_f:.4f}")
    print(f"Macro    P/R/F1     : {m_p:.4f} / {m_r:.4f} / {m_f:.4f}")

    overall.append({
        "Model": name,
        "Accuracy": acc,
        "Weighted_Precision": w_p,
        "Weighted_Recall": w_r,
        "Weighted_F1": w_f,
        "Macro_Precision": m_p,
        "Macro_Recall": m_r,
        "Macro_F1": m_f,
    })

    report = classification_report(
        y_true, y_pred, labels=list(LABEL_NAMES),
        target_names=TARGET_NAMES, output_dict=True, zero_division=0,
    )
    for cls in TARGET_NAMES:
        m = report[cls]
        per_class.append({
            "Model": name, "Class": cls,
            "Precision": m["precision"], "Recall": m["recall"],
            "F1-score": m["f1-score"], "Support": int(m["support"]),
        })

    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_NAMES))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CM_DIR}/{name.replace(' ', '_').replace('(', '').replace(')', '')}_cm.png",
                dpi=200, bbox_inches="tight")
    plt.close()


all_overall_by_tag = {}

for cfg in CONFIGS:
    tag, model_dir, test_dir = cfg["tag"], cfg["model_dir"], cfg["test_dir"]
    print("\n" + "=" * 60)
    print(f"{tag.upper()}  |  models: {model_dir}  |  test: {test_dir}")
    print("=" * 60)

    if not all(os.path.exists(f"{model_dir}/{f}.pkl")
               for f in ["random_forest", "xgboost", "mlp", "scaler"]):
        print(f"  SKIP: {model_dir} is missing pickles")
        continue

    X, y_true = load_clean_test(test_dir)
    print(f"Test shape: {X.shape}   class counts: {pd.Series(y_true).value_counts().sort_index().to_dict()}")

    rf = joblib.load(f"{model_dir}/random_forest.pkl")
    xgb = joblib.load(f"{model_dir}/xgboost.pkl")
    mlp = joblib.load(f"{model_dir}/mlp.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")

    overall: list[dict] = []
    per_class: list[dict] = []
    preds = pd.DataFrame({"True Label": y_true})

    evaluate_one(f"Random Forest ({tag})", rf, X, y_true, False, scaler, overall, per_class, preds)
    evaluate_one(f"XGBoost ({tag})", xgb, X, y_true, False, scaler, overall, per_class, preds)
    evaluate_one(f"MLP ({tag})", mlp, X, y_true, True, scaler, overall, per_class, preds)

    pd.DataFrame(overall).to_csv(f"{RESULT_DIR}/overall_results_{tag}.csv", index=False)
    pd.DataFrame(per_class).to_csv(f"{RESULT_DIR}/class_level_results_{tag}.csv", index=False)
    preds.to_csv(f"{RESULT_DIR}/predictions_{tag}.csv", index=False)
    print(f"\nSaved {RESULT_DIR}/overall_results_{tag}.csv, class_level_results_{tag}.csv, predictions_{tag}.csv")

    all_overall_by_tag[tag] = overall

    del X, rf, xgb, mlp

# ==================================================
# SIDE-BY-SIDE COMPARISON (v2 -> v3 -> v4)
# ==================================================

comparison_rows = []
for tag, overall in all_overall_by_tag.items():
    for row in overall:
        model_type = row["Model"].split(" (")[0]
        comparison_rows.append({
            "Version": tag,
            "Model": model_type,
            "Accuracy": row["Accuracy"],
            "Weighted_Precision": row["Weighted_Precision"],
            "Weighted_Recall": row["Weighted_Recall"],
            "Weighted_F1": row["Weighted_F1"],
            "Macro_Precision": row["Macro_Precision"],
            "Macro_Recall": row["Macro_Recall"],
            "Macro_F1": row["Macro_F1"],
        })

comparison = pd.DataFrame(comparison_rows)
model_order = ["Random Forest", "XGBoost", "MLP"]
version_order = ["v2", "v3", "v4"]
comparison["Model"] = pd.Categorical(comparison["Model"], categories=model_order, ordered=True)
comparison["Version"] = pd.Categorical(comparison["Version"], categories=version_order, ordered=True)
comparison = comparison.sort_values(["Model", "Version"]).reset_index(drop=True)

comparison.to_csv(f"{RESULT_DIR}/overall_results_comparison.csv", index=False)

print("\n" + "=" * 60)
print("SIDE-BY-SIDE: v2 -> v3 -> v4")
print("=" * 60)
print(comparison.to_string(index=False))
print(f"\nSaved {RESULT_DIR}/overall_results_comparison.csv")

print("\nDone.")
