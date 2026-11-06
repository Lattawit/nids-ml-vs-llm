import pandas as pd
import glob
import os

# ==================================================
# CONFIG
# ==================================================

DATASETS = [
    "mapped/CIC-IDS2017",
    "mapped/CSE-CIC-IDS2018"
]

OUTPUT_FILE = "results/final_merged_dataset.csv"

os.makedirs("results", exist_ok=True)

# ==================================================
# STORAGE
# ==================================================

all_data = []

# ==================================================
# LOAD + CLEAN + MERGE
# ==================================================

for dataset_folder in DATASETS:

    csv_files = glob.glob(os.path.join(dataset_folder, "*.csv"))

    print(f"\nProcessing dataset: {dataset_folder}")
    print(f"Files found: {len(csv_files)}")

    for file in csv_files:

        print(f"Loading: {os.path.basename(file)}")

        df = pd.read_csv(file, low_memory=False)

        # Strip column names
        df.columns = df.columns.str.strip()

        # Ensure Label exists
        if "Label" not in df.columns:
            print("Skipping (no Label column)")
            continue

        # Drop Heartbleed BEFORE merge
        df = df[df["Label"] != "Heartbleed"]

        # Safety: remove any NaN labels
        df = df.dropna(subset=["Label"])

        all_data.append(df)

# ==================================================
# CONCATENATE ALL DATA
# ==================================================

print("\nMerging datasets...")

final_df = pd.concat(all_data, ignore_index=True)

# ==================================================
# FINAL CHECK
# ==================================================

print("\nFinal dataset shape:", final_df.shape)

print("\nClass distribution:")
print(final_df["Label"].value_counts())

# ==================================================
# SAVE FINAL DATASET
# ==================================================

final_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved merged dataset to:")
print(OUTPUT_FILE)