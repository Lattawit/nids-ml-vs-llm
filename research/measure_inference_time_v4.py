"""Inference-time-only benchmark for the v4 models, for paper Table II.

Standalone script (not a copy of evaluate_models_v4_on_llm_subset.py) because
that script's job is accuracy metrics; this one isolates just the
``.predict()`` call so it can be compared like-for-like against the LLM
configs' inference time on the same 700-row subset. Data loading, numeric
cleaning, and scaler.transform are done once, outside the timed region, for
all three models.

Resolves a reviewer concern that the original Table II mixed ML training
time with LLM inference time.
"""

import time

import joblib
import numpy as np
import pandas as pd

DATA_FILE = "results/llm/llm_subset_v4.csv"
MODEL_DIR = "models/v4"
RESULT_DIR = "results/llm_evaluation_v4"
N_REPEATS = 5  # median of repeated predict() calls, to smooth OS/cache noise

# ==================================================
# LOAD + CLEAN DATA  (same numeric cleaning as training)
# ==================================================

print("Loading LLM evaluation dataset...")

df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)

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
# TIME .predict() ONLY
# ==================================================


def time_predict(name, model, X_in):
    times = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        model.predict(X_in)
        times.append(time.perf_counter() - start)
    median = float(np.median(times))
    print(f"{name:<15} median={median:.4f}s  runs={[f'{t:.4f}' for t in times]}")
    return median, times


print(f"\nTiming .predict() on {len(X)} rows, {N_REPEATS} repeats each (median reported)...\n")

rf_median, rf_runs = time_predict("Random Forest", rf, X)
xgb_median, xgb_runs = time_predict("XGBoost", xgb, X)
mlp_median, mlp_runs = time_predict("MLP", mlp, X_scaled)

# ==================================================
# SAVE
# ==================================================

results = pd.DataFrame([
    {"Model": "Random Forest", "N_Samples": len(X), "Median_Predict_Seconds": rf_median,
     "All_Runs_Seconds": rf_runs},
    {"Model": "XGBoost", "N_Samples": len(X), "Median_Predict_Seconds": xgb_median,
     "All_Runs_Seconds": xgb_runs},
    {"Model": "MLP", "N_Samples": len(X), "Median_Predict_Seconds": mlp_median,
     "All_Runs_Seconds": mlp_runs},
])
out_path = f"{RESULT_DIR}/inference_time_v4.csv"
results.to_csv(out_path, index=False)

print("\n===================================")
print("Inference Timing Complete")
print("===================================")
for _, row in results.iterrows():
    print(f"{row['Model']:<15}: {row['Median_Predict_Seconds']:.4f}s for {row['N_Samples']} rows")
print(f"\nSaved: {out_path}")
