"""Example-count ablation of the v4 few-shot LLM classification.

Re-run of ``llm_classification_few_shot_v3_3shot_ablation.py`` on the v4
(deduplicated, cross-split-leakage-free) pipeline, for internal consistency
with the other v4 headline numbers -- the original 3-shot ablation (3 runs)
drew its reference examples from ``split_data_v3`` and classified
``llm_subset_v3.csv``, before the cross-split-duplicate leakage fix.

- ``EXAMPLES_PER_CLASS = 3`` reference examples per class (vs. 1 in the v4
  reference script, ``llm_classification_few_shot_v4.py``). The 1st example
  per class is always the SAME row that script picks (``RandomState(42)``,
  size=1), so every repeat here is a strict superset of the v4 1-shot
  baseline; only the remaining ``EXAMPLES_PER_CLASS - 1`` "extra" examples
  are drawn per repeat, seeded by ``EXTRA_SEED`` (sampled without
  replacement, excluding the base row). All drawn from the same source as
  the v4 baseline: the v4 coalesced, deduplicated **pre-SMOTE training
  split** (``results/split_data_v4/X_train.csv`` + ``y_train.csv``), real
  rows only.
- Same 22-feature ``selected_features`` set as the v3 ablation.
  ``INPUT_FILE`` is ``results/llm/llm_subset_v4.csv``.
- Outputs go to
  ``results/llm_results_v4_ablation/<model>_fewshot_3shot[_<RUN_LABEL>]/``.
  Edit ``EXTRA_SEED`` and ``RUN_LABEL`` between repeat runs. The v3 script's
  own per-run seeds were never committed between edits (each run overwrote
  the same file in-place, so only the final run3 state survives in git) --
  this v4 run uses its own fresh seeds (101 / 202 / 303) rather than trying
  to reproduce the unrecoverable v3 values; what matters for the ablation is
  the *direction* of the effect (more examples -> collapse), not bit-for-bit
  seed parity with v3.

Edit ``MODEL_NAME`` between runs. The Ollama host is resolved by
``ollama_host.get_ollama_client()``: ``OLLAMA_HOST`` env var if set, else
localhost, else the auto-detected WSL2 gateway IP.

Decoding: ``ollama.chat`` is called with ``options={"temperature": 0,
"num_predict": 60}``. The ``num_predict`` cap was added after run 3 of this
exact ablation stalled for ~1h49m on a single row (out of 700) before being
manually killed -- with ``temperature=0`` and no cap, a row that never
reaches a natural stop token lets the model generate all the way to the
~4096-token context limit, extremely slow given this GPU's partial
CPU-offload (see CLAUDE.md, "v4 3-shot ablation stall" note). 60 tokens is
generous for a bare class-name answer.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
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

MODEL_NAME = "qwen2.5:14B"

INPUT_FILE = "results/llm/llm_subset_v4.csv"
TRAIN_DIR = "results/split_data_v4"
CHUNK_SIZE = 1_000_000
EXAMPLES_PER_CLASS = 3

# The 1st example per class is always the SAME row the v4 1-shot reference
# script uses (RandomState(42), size=1) -- kept fixed so the 3-shot runs are
# a strict superset of the 1-shot baseline, not an unrelated resample.
# EXTRA_SEED controls only the (EXAMPLES_PER_CLASS - 1) additional examples;
# edit it between repeat runs (along with RUN_LABEL) so each repeat draws
# different extras rather than the identical prompt.
EXTRA_SEED = int(os.environ.get("EXTRA_SEED", "303"))
RUN_LABEL = os.environ.get("RUN_LABEL", "run3")
NUM_CTX = 8192  # audit follow-up override; v4 original script leaves this unset (default 4096)

_model_slug = MODEL_NAME.replace(":", "_").replace("/", "_")
_suffix = f"_{RUN_LABEL}" if RUN_LABEL else ""
OUTPUT_DIR = f"results/audit_followup/step2_ctx8192_{_model_slug}_fewshot_3shot{_suffix}"
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
# FEATURES (22 -- coalesced schema)
# ==================================================

selected_features = [
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
]

# ==================================================
# LOAD TEST SUBSET
# ==================================================

print("Loading subset...")

test_df = pd.read_csv(INPUT_FILE)

missing = [f for f in selected_features if f not in test_df.columns]
if missing:
    raise SystemExit(f"features not in {INPUT_FILE}: {missing}")

X = test_df[selected_features]
y_true = test_df["Label"].values

print("Subset shape:", X.shape)

# ==================================================
# BUILD REFERENCE EXAMPLES: 1 base row/class (same as the v4 1-shot
# baseline, RandomState(42)) + (EXAMPLES_PER_CLASS - 1) extra rows/class
# seeded by EXTRA_SEED, from the v4 TRAIN split, no-replacement sample per
# class.
# ==================================================

print(f"Selecting {EXAMPLES_PER_CLASS} reference examples/class "
      f"(1 base + {EXAMPLES_PER_CLASS - 1} extra, EXTRA_SEED={EXTRA_SEED}) "
      f"from the v4 training split...")

y_train = pd.read_csv(f"{TRAIN_DIR}/y_train.csv")["Label"].to_numpy()

ref_pos = {}
for label in sorted(np.unique(y_train)):
    pos = np.flatnonzero(y_train == label)

    base_rng = np.random.RandomState(42)
    base_idx = int(base_rng.choice(pos, size=1)[0])
    ref_pos[base_idx] = int(label)

    remaining = pos[pos != base_idx]
    extra_rng = np.random.RandomState(EXTRA_SEED)
    extra_idx = extra_rng.choice(remaining, size=EXAMPLES_PER_CLASS - 1, replace=False)
    for p in extra_idx:
        ref_pos[int(p)] = int(label)

ref_rows = {}
row0 = 0
for chunk in pd.read_csv(
    f"{TRAIN_DIR}/X_train.csv", usecols=selected_features, chunksize=CHUNK_SIZE, low_memory=False
):
    chunk.columns = chunk.columns.str.strip()
    for p, label in ref_pos.items():
        if row0 <= p < row0 + len(chunk):
            ref_rows.setdefault(label, []).append(chunk.iloc[p - row0])
    row0 += len(chunk)

reference_examples = {}
for label in sorted(ref_rows):
    reference_examples[label] = [
        "".join(f"{feature}: {row[feature]}\n" for feature in selected_features)
        for row in ref_rows[label]
    ]

# ==================================================
# KNOWLEDGE BASE
# ==================================================

knowledge = """
You are an expert cybersecurity analyst specializing in Network Intrusion Detection.

Attack Categories

Benign
- Normal user traffic.
- Balanced communication.
- Typical web or application usage.

Bot
- Malware-controlled host.
- Automated repetitive communication.
- Often shows periodic behaviour.

Brute Force
- Multiple authentication attempts.
- Often targets login services.
- Numerous short connections.

DoS/DDoS
- Extremely high traffic volume.
- Large packet counts.
- High packet rate.
- Victim receives overwhelming traffic.

Infiltration
- Unauthorized internal activity.
- Usually subtle.
- Longer communication sessions.

PortScan
- Sequential probing of many ports.
- Small packets.
- Numerous connection attempts.

Web Attack
- HTTP/HTTPS based attacks.
- SQL Injection.
- XSS.
- Command Injection.

Always compare the traffic against ALL categories before deciding.

Return ONLY ONE class name.

Valid outputs:

Benign
Bot
Brute Force
DoS/DDoS
Infiltration
PortScan
Web Attack
"""

reference_block = ""
for label in sorted(reference_examples):
    for idx, example in enumerate(reference_examples[label], start=1):
        reference_block += f"\nReference Example {idx} ({label_names[label]})\n"
        reference_block += example
        reference_block += "\n"

# ==================================================
# INFERENCE
# ==================================================

predictions = []
raw_responses = []
fallback_flags = []
per_row_seconds = []

print(f"\nRunning FEW-SHOT LLM classification with {MODEL_NAME}... (audit follow-up: num_ctx={NUM_CTX})")

start = time.time()

for i, row in X.iterrows():

    traffic = "".join(f"{feature}: {row[feature]}\n" for feature in selected_features)

    prompt = f"""
{knowledge}

==========================
Representative Examples
==========================

{reference_block}

==========================
Traffic to Classify
==========================

{traffic}

Compare the traffic characteristics with the attack definitions and representative examples.

Determine the single best matching class.

Output ONLY the class name.
"""

    _t0 = time.time()
    response = _ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 60, "num_ctx": NUM_CTX},
    )
    per_row_seconds.append(time.time() - _t0)

    answer = response["message"]["content"].strip()
    raw_responses.append(answer)

    prediction = "Benign"
    fallback = True
    for cls in reverse_labels:
        if cls.lower() in answer.lower():
            prediction = cls
            fallback = False
            break
    fallback_flags.append(1 if fallback else 0)

    predictions.append(reverse_labels[prediction])

    print(f"[{i + 1}/{len(X)}] True={label_names[y_true[i]]} Pred={prediction} Fallback={fallback} ({per_row_seconds[-1]:.2f}s)")

elapsed = time.time() - start

# ==================================================
# METRICS
# ==================================================

acc = accuracy_score(y_true, predictions)
w_p, w_r, w_f, _ = precision_recall_fscore_support(
    y_true, predictions, average="weighted", zero_division=0
)
m_p, m_r, m_f, _ = precision_recall_fscore_support(
    y_true, predictions, average="macro", zero_division=0
)

_run_desc = RUN_LABEL if RUN_LABEL else "run1"
print("\n===================================")
print(f"LLM RESULTS  ({MODEL_NAME}, few-shot, {EXAMPLES_PER_CLASS}-shot ablation, v4, {_run_desc})")
print("===================================")
print(f"Accuracy       : {acc:.4f}")
print(f"Weighted P/R/F1: {w_p:.4f} / {w_r:.4f} / {w_f:.4f}")
print(f"Macro    P/R/F1: {m_p:.4f} / {m_r:.4f} / {m_f:.4f}")
print(f"Inference Time : {elapsed:.2f} s")

n_fallback = int(sum(fallback_flags))
avg_sec_per_row = float(np.mean(per_row_seconds))

results_df = pd.DataFrame([{
    "Model": f"LLM {MODEL_NAME} (few-shot, {EXAMPLES_PER_CLASS}shot, v4, {_run_desc}, num_ctx={NUM_CTX})",
    "Accuracy": acc,
    "Weighted_Precision": w_p,
    "Weighted_Recall": w_r,
    "Weighted_F1": w_f,
    "Macro_Precision": m_p,
    "Macro_Recall": m_r,
    "Macro_F1": m_f,
    "Inference Time (s)": round(elapsed, 2),
    "Avg_Seconds_Per_Row": round(avg_sec_per_row, 4),
    "Num_Fallback_Rows": n_fallback,
    "Num_Ctx": NUM_CTX,
    "Extra_Seed": EXTRA_SEED,
}])
results_df.to_csv(f"{OUTPUT_DIR}/llm_few_results.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/llm_few_results.csv")

pd.DataFrame({
    "True Label": [label_names[x] for x in y_true],
    "Predicted Label": [label_names[x] for x in predictions],
    "Raw Response": raw_responses,
    "Fallback": fallback_flags,
    "Seconds": per_row_seconds,
}).to_csv(f"{OUTPUT_DIR}/llm_few_predictions.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/llm_few_predictions.csv")

# ==================================================
# CLASS RESULTS
# ==================================================

report = classification_report(
    y_true, predictions,
    labels=list(label_names),
    target_names=list(label_names.values()),
    output_dict=True, zero_division=0,
)

print("\n" + classification_report(
    y_true, predictions,
    labels=list(label_names),
    target_names=list(label_names.values()),
    zero_division=0,
))

rows = []
for cls in label_names.values():
    m = report[cls]
    rows.append({
        "Model": f"LLM {MODEL_NAME} (few-shot, {EXAMPLES_PER_CLASS}shot, v4, {_run_desc})",
        "Class": cls,
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1-score": m["f1-score"],
        "Support": int(m["support"]),
    })
pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/llm_few_class_results.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/llm_few_class_results.csv")

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
plt.title(f"{MODEL_NAME} few-shot ({EXAMPLES_PER_CLASS}shot, v4, {_run_desc}) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/llm_few_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/llm_few_confusion_matrix.png")

print("\nDone.")
