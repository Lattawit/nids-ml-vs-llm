"""v4 model training.

Copy of ``train_models_v3.py`` for the leakage-fix rebuild. Only the paths
differ:

- ``DATA_PATH`` is ``results/split_data_v4`` (deduplicated before the split,
  zero cross-split row overlap -- see ``deduplicate_dataset_v4.py`` and
  ``train_test_split_pipeline_v4.py``).
- ``MODEL_DIR`` is ``models/v4`` so ``models/v3`` and ``models/v2`` are left
  untouched.

Model definitions and hyper-parameters (RF 100 trees / class_weight balanced;
XGBoost 200 est, depth 8; MLP 512-256, 50 epochs, early stopping) are identical
to v2/v3. RF + XGBoost train on the raw SMOTE'd features; MLP trains on
``StandardScaler``'d features and the scaler is saved. The v3 script is
untouched.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = "results/split_data_v4"

MODEL_DIR = "models/v4"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

print("Loading balanced training data...")

X_train = pd.read_csv(f"{DATA_PATH}/X_train_smote.csv")

y_train = pd.read_csv(f"{DATA_PATH}/y_train_smote.csv").values.ravel()

print("Training shape:", X_train.shape)

# ==================================================
# REMOVE NON NUMERIC
# ==================================================

drop_cols = []

for col in X_train.columns:

    if not pd.api.types.is_numeric_dtype(X_train[col]):

        drop_cols.append(col)

if len(drop_cols):

    print("\nDropping columns:")

    print(drop_cols)

    X_train.drop(
        columns=drop_cols,
        inplace=True
    )

# ==================================================
# CLEAN DATA
# ==================================================

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_train = X_train.replace(
    [np.inf, -np.inf],
    0
)

X_train = X_train.fillna(0)

# ==================================================
# SCALE FOR MLP
# ==================================================

print("\nPreparing scaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

joblib.dump(
    scaler,
    f"{MODEL_DIR}/scaler.pkl"
)

print("Scaler saved.")

# ==================================================
# RANDOM FOREST
# ==================================================

print("\n======================")
print("Training Random Forest")
print("======================")

start = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

print(
    f"Finished in {time.time()-start:.2f} sec"
)

joblib.dump(
    rf,
    f"{MODEL_DIR}/random_forest.pkl"
)

# ==================================================
# XGBOOST
# ==================================================

print("\n======================")
print("Training XGBoost")
print("======================")

start = time.time()

xgb = XGBClassifier(

    n_estimators=200,

    learning_rate=0.1,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    tree_method="hist",

    eval_metric="mlogloss",

    n_jobs=-1

)

xgb.fit(
    X_train,
    y_train
)

print(
    f"Finished in {time.time()-start:.2f} sec"
)

joblib.dump(
    xgb,
    f"{MODEL_DIR}/xgboost.pkl"
)

# ==================================================
# MLP
# ==================================================

print("\n======================")
print("Training MLP")
print("======================")

start = time.time()

mlp = MLPClassifier(

    hidden_layer_sizes=(512,256),

    activation="relu",

    solver="adam",

    batch_size=2048,

    learning_rate="adaptive",

    learning_rate_init=0.001,

    max_iter=50,

    early_stopping=True,

    validation_fraction=0.1,

    n_iter_no_change=5,

    random_state=42,

    verbose=True

)

mlp.fit(
    X_train_scaled,
    y_train
)

print(
    f"Finished in {time.time()-start:.2f} sec"
)

joblib.dump(
    mlp,
    f"{MODEL_DIR}/mlp.pkl"
)

print("\n======================")
print("Training Complete")
print("======================")

print("\nSaved models:")

print("Random Forest")

print("XGBoost")

print("MLP")

print("Scaler")
