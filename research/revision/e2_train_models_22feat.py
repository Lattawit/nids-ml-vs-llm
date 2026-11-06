"""E2: retrain RF / XGBoost / MLP on only the 22 LLM-prompt features.

Copy of ``research/train_models_v4.py`` (untouched) with one change: the
balanced training set ``X_train_smote.csv`` is restricted to the 22
``selected_features`` used in the LLM prompts, in the same order. Same
cleaning, same hyper-parameters, same seeds (RF/MLP random_state=42; XGBoost
left at its default, as in the original). Models go to
``results/revision/e2/models_22feat/`` -- ``models/v4`` is not touched.
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from prompts_v4 import selected_features  # noqa: E402

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = "results/split_data_v4"
MODEL_DIR = "results/revision/e2/models_22feat"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# LOAD + CLEAN (as train_models_v4.py)
# ==================================================

print("Loading balanced training data (22 features)...")
X_train = pd.read_csv(f"{DATA_PATH}/X_train_smote.csv", usecols=selected_features)
X_train = X_train[selected_features]
y_train = pd.read_csv(f"{DATA_PATH}/y_train_smote.csv").values.ravel()
print("Training shape:", X_train.shape)

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_train = X_train.replace([np.inf, -np.inf], 0)
X_train = X_train.fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

timings = {}

# ==================================================
# RANDOM FOREST
# ==================================================

print("\nTraining Random Forest", flush=True)
start = time.time()
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
timings["random_forest"] = time.time() - start
print(f"Finished in {timings['random_forest']:.2f} sec", flush=True)
joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")
del rf

# ==================================================
# XGBOOST
# ==================================================

print("\nTraining XGBoost", flush=True)
start = time.time()
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, subsample=0.8,
                    colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss", n_jobs=-1)
xgb.fit(X_train, y_train)
timings["xgboost"] = time.time() - start
print(f"Finished in {timings['xgboost']:.2f} sec", flush=True)
joblib.dump(xgb, f"{MODEL_DIR}/xgboost.pkl")
del xgb

# ==================================================
# MLP
# ==================================================

print("\nTraining MLP", flush=True)
start = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
                    batch_size=2048, learning_rate="adaptive", learning_rate_init=0.001,
                    max_iter=50, early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=5, random_state=42, verbose=True)
mlp.fit(X_train_scaled, y_train)
timings["mlp"] = time.time() - start
print(f"Finished in {timings['mlp']:.2f} sec", flush=True)
joblib.dump(mlp, f"{MODEL_DIR}/mlp.pkl")

pd.DataFrame([{"model": k, "train_seconds": round(v, 2)} for k, v in timings.items()]).to_csv(
    f"{MODEL_DIR}/../e2_train_times.csv", index=False)
print("\nDone.")
