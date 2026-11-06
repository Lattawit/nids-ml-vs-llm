"""v4 ensemble evaluation on the LLM subset.

v4 copy of ``evaluate_models_v3_on_llm_subset.py``: runs the trained models on
the SAME 700-row subset the v4 LLM scripts use, for a like-for-like ML-vs-LLM
comparison. Differences from v3:

- ``DATA_FILE`` is ``results/llm/llm_subset_v4.csv`` (sampled from the
  deduplicated, cross-split-leakage-free v4 test split).
- ``MODEL_DIR`` is ``models/v4``.
- ``RESULT_DIR`` is ``results/llm_evaluation_v4``.

No cross-lineage schema alignment is needed here either: ``llm_subset_v4.csv``
and ``models/v4`` share the same 70 coalesced feature columns. The v3 script
is untouched.
"""

import os

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

DATA_FILE = "results/llm/llm_subset_v4.csv"
MODEL_DIR = "models/v4"
RESULT_DIR = "results/llm_evaluation_v4"
CM_DIR = f"{RESULT_DIR}/confusion_matrices"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

label_names = {
    0: "Benign",
    1: "Bot",
    2: "Brute Force",
    3: "DoS/DDoS",
    4: "Infiltration",
    5: "PortScan",
    6: "Web Attack",
}
target_names = list(label_names.values())

# ==================================================
# LOAD + CLEAN DATA  (same numeric cleaning as training)
# ==================================================

print("Loading LLM evaluation dataset...")

df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)

y_true = df["Label"].values
X = df.drop(columns=["Label"])

non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
if non_numeric:
    print("Dropping non-numeric columns:", non_numeric)
    X = X.drop(columns=non_numeric)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], 0)
X = X.fillna(0)
X = X.clip(-1e6, 1e6)

# ==================================================
# LOAD MODELS
# ==================================================

print("\nLoading models from", MODEL_DIR)

rf = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
xgb = joblib.load(f"{MODEL_DIR}/xgboost.pkl")
mlp = joblib.load(f"{MODEL_DIR}/mlp.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

X_scaled = scaler.transform(X)

# ==================================================
# EVALUATE
# ==================================================

overall_results = []
class_results = []
prediction_table = pd.DataFrame({"True Label": y_true})


def evaluate_model(name, model, X_in, y_true):
    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    y_pred = model.predict(X_in)
    prediction_table[name] = y_pred

    acc = accuracy_score(y_true, y_pred)
    w_p, w_r, w_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    m_p, m_r, m_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"Accuracy       : {acc:.4f}")
    print(f"Weighted P/R/F1: {w_p:.4f} / {w_r:.4f} / {w_f:.4f}")
    print(f"Macro    P/R/F1: {m_p:.4f} / {m_r:.4f} / {m_f:.4f}")

    overall_results.append({
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
        y_true, y_pred, labels=list(label_names),
        target_names=target_names, output_dict=True, zero_division=0,
    )
    for cls in target_names:
        m = report[cls]
        class_results.append({
            "Model": name, "Class": cls,
            "Precision": m["precision"], "Recall": m["recall"],
            "F1-score": m["f1-score"], "Support": int(m["support"]),
        })

    cm = confusion_matrix(y_true, y_pred, labels=list(label_names))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CM_DIR}/{name.replace(' ', '_')}_cm.png", dpi=200, bbox_inches="tight")
    plt.close()


evaluate_model("Random Forest", rf, X, y_true)
evaluate_model("XGBoost", xgb, X, y_true)
evaluate_model("MLP", mlp, X_scaled, y_true)

# ==================================================
# SAVE
# ==================================================

pd.DataFrame(overall_results).to_csv(f"{RESULT_DIR}/model_results_v4.csv", index=False)
pd.DataFrame(class_results).to_csv(f"{RESULT_DIR}/class_level_results_v4.csv", index=False)
prediction_table.to_csv(f"{RESULT_DIR}/predictions_v4.csv", index=False)

print("\n===================================")
print("Evaluation Complete")
print("===================================")
print(f"Saved: {RESULT_DIR}/model_results_v4.csv")
print(f"Saved: {RESULT_DIR}/class_level_results_v4.csv")
print(f"Saved: {RESULT_DIR}/predictions_v4.csv")
print(f"Saved: {CM_DIR}/")
