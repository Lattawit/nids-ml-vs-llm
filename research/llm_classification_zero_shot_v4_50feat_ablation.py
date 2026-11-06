"""Feature-count ablation of the v4 zero-shot LLM classification.

Re-run of ``llm_classification_zero_shot_v3_50feat_ablation.py`` on the v4
(deduplicated, cross-split-leakage-free) pipeline, for internal consistency
with the other v4 headline numbers -- the original 50-feature ablation ran
against ``llm_subset_v3.csv``, before the cross-split-duplicate leakage fix.

- Same ``selected_features`` (22-feature baseline + 28 more, real names from
  the 70-column coalesced schema; confirmed present in both
  ``results/split_data_v4/X_train.csv`` and ``results/llm/llm_subset_v4.csv``
  since v4 keeps the same coalesced column set as v3, just deduplicated rows).
- ``INPUT_FILE`` is ``results/llm/llm_subset_v4.csv`` instead of
  ``llm_subset_v3.csv``.
- Outputs go to ``results/llm_results_v4_ablation/<model>_zeroshot_50feat/``
  so nothing v3 or earlier is overwritten.

Edit ``MODEL_NAME`` between runs. The Ollama host is resolved by
``ollama_host.get_ollama_client()``: ``OLLAMA_HOST`` env var if set, else
localhost, else the auto-detected WSL2 gateway IP.

Decoding: ``ollama.chat`` is called with ``options={"temperature": 0,
"num_predict": 60}``, matching every other v3/v4 LLM script. ``num_predict``
was added after a v4 3-shot ablation run stalled for ~1h49m on a single row:
with no cap, a row that never reaches a natural stop token lets the model
generate all the way to the ~4096-token context limit (see CLAUDE.md, "v4
3-shot ablation stall"). 60 tokens is generous for a bare class-name answer.
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ollama
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from ollama_host import get_ollama_client

# ==================================================
# CONFIG
# ==================================================

INPUT_FILE = "results/llm/llm_subset_v4.csv"
MODEL_NAME = "qwen2.5:14B"

_model_slug = MODEL_NAME.replace(":", "_").replace("/", "_")
OUTPUT_DIR = f"results/llm_results_v4_ablation/{_model_slug}_zeroshot_50feat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_ollama_client = get_ollama_client()

# ==================================================
# LABELS
# ==================================================

label_names = {
    0: "Benign",
    1: "Bot",
    2: "Brute Force",
    3: "DoS/DDoS",
    4: "Infiltration",
    5: "PortScan",
    6: "Web Attack",
}

reverse_labels = {v: k for k, v in label_names.items()}

# ==================================================
# FEATURE-COUNT ABLATION: 22-feature baseline + 28 more (50 total)
# ==================================================

selected_features = [
    # --- 22-feature baseline (unchanged) ---
    "Protocol",
    "Dst Port",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Size Avg",
    "ACK Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Active Mean",
    "Idle Mean",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Pkt Len Max",
    "Pkt Len Min",
    "Down/Up Ratio",
    # --- +28: inter-arrival timing ---
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    # --- backward packet-length spread ---
    "Bwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    # --- remaining TCP flags ---
    "FIN Flag Cnt",
    "PSH Flag Cnt",
    "URG Flag Cnt",
    # --- header/segment info ---
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    # --- subflow stats ---
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    # --- active/idle spread ---
    "Active Std",
    "Active Max",
    "Idle Std",
    "Idle Max",
]

# ==================================================
# LOAD DATA
# ==================================================

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

missing = [f for f in selected_features if f not in df.columns]
if missing:
    raise SystemExit(f"features not in {INPUT_FILE}: {missing}")

X = df[selected_features]
y_true = df["Label"].values

print("Shape:", X.shape)

predictions = []

# ==================================================
# CLASSIFICATION LOOP (ZERO SHOT)
# ==================================================

print(f"\nRunning ZERO-SHOT LLM classification with {MODEL_NAME}...")

start = time.time()

for i, row in X.iterrows():

    traffic = "\n".join(
        [f"{col}: {row[col]}" for col in selected_features]
    )

    prompt = f"""
You are a Network Intrusion Detection System.

Classify this network traffic into exactly ONE class:

Benign
Bot
Brute Force
DoS/DDoS
Infiltration
PortScan
Web Attack

Traffic:
{traffic}

Respond with ONLY ONE class name.
"""

    response = _ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 60},
    )

    prediction = (
        response["message"]["content"]
        .strip()
        .replace('"', "")
        .split("\n")[0]
        .strip()
    )

    # Map free text back to a class; unparseable -> Benign (documented quirk,
    # inflates Benign recall -- kept for continuity with the v2/v3 results)
    found = False
    for label in reverse_labels:
        if label.lower() in prediction.lower():
            prediction = label
            found = True
            break
    if not found:
        prediction = "Benign"

    predictions.append(reverse_labels[prediction])

    print(f"[{i + 1}/{len(X)}] True: {label_names[y_true[i]]} Pred: {prediction}")

elapsed = time.time() - start

# ==================================================
# SAVE PREDICTIONS
# ==================================================

prediction_df = pd.DataFrame({
    "True Label": [label_names[x] for x in y_true],
    "Predicted Label": [label_names[x] for x in predictions],
})

prediction_file = f"{OUTPUT_DIR}/llm_zero_predictions.csv"
prediction_df.to_csv(prediction_file, index=False)
print(f"\nSaved: {prediction_file}")

# ==================================================
# OVERALL METRICS
# ==================================================

acc = accuracy_score(y_true, predictions)

w_p, w_r, w_f, _ = precision_recall_fscore_support(
    y_true, predictions, average="weighted", zero_division=0
)
m_p, m_r, m_f, _ = precision_recall_fscore_support(
    y_true, predictions, average="macro", zero_division=0
)

print("\n===================================")
print(f"LLM RESULTS  ({MODEL_NAME}, zero-shot, 50-feature ablation, v4)")
print("===================================")
print(f"Accuracy       : {acc:.4f}")
print(f"Weighted P/R/F1: {w_p:.4f} / {w_r:.4f} / {w_f:.4f}")
print(f"Macro    P/R/F1: {m_p:.4f} / {m_r:.4f} / {m_f:.4f}")
print(f"Inference Time : {elapsed:.2f} s")

results_df = pd.DataFrame([{
    "Model": f"LLM {MODEL_NAME} (zero-shot, 50feat, v4)",
    "Accuracy": acc,
    "Weighted_Precision": w_p,
    "Weighted_Recall": w_r,
    "Weighted_F1": w_f,
    "Macro_Precision": m_p,
    "Macro_Recall": m_r,
    "Macro_F1": m_f,
    "Inference Time (s)": round(elapsed, 2),
}])

results_file = f"{OUTPUT_DIR}/llm_zero_results.csv"
results_df.to_csv(results_file, index=False)
print(f"Saved: {results_file}")

# ==================================================
# CLASSIFICATION REPORT
# ==================================================

report = classification_report(
    y_true, predictions,
    labels=list(label_names),
    target_names=list(label_names.values()),
    output_dict=True, zero_division=0,
)

print("\nClassification Report:")
print(classification_report(
    y_true, predictions,
    labels=list(label_names),
    target_names=list(label_names.values()),
    zero_division=0,
))

class_results = []
for cls in label_names.values():
    m = report[cls]
    class_results.append({
        "Model": f"LLM {MODEL_NAME} (zero-shot, 50feat, v4)",
        "Class": cls,
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1-score": m["f1-score"],
        "Support": int(m["support"]),
    })

class_file = f"{OUTPUT_DIR}/llm_zero_class_results.csv"
pd.DataFrame(class_results).to_csv(class_file, index=False)
print(f"Saved: {class_file}")

# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(y_true, predictions, labels=list(label_names))

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=list(label_names.values()),
    yticklabels=list(label_names.values()),
)
plt.title(f"{MODEL_NAME} zero-shot (50feat, v4) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

cm_file = f"{OUTPUT_DIR}/llm_zero_confusion_matrix.png"
plt.savefig(cm_file, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {cm_file}")

print("\nDone.")
