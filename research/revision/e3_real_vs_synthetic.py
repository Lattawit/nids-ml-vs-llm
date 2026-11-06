"""E3: real vs. SMOTE-synthetic rows per class in the v4 balanced training set.

Pure computation on ``results/split_data_v4/training_balancing_report.csv``
(written by ``balance_dataset_v4.py``): Before = real pre-SMOTE training rows,
After_Undersample = real rows kept, After_SMOTE = final count. Synthetic =
After_SMOTE - After_Undersample.
"""

import os

import pandas as pd

REPORT = "results/split_data_v4/training_balancing_report.csv"
LABELS = "results/split_data_v4/label_mapping.csv"
OUT = "results/revision/e3/e3_real_vs_synthetic_v4_target200k_rs42.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

rep = pd.read_csv(REPORT)
names = {0: "Benign", 1: "Bot", 2: "Brute Force", 3: "DoS/DDoS",
         4: "Infiltration", 5: "PortScan", 6: "Web Attack"}

out = pd.DataFrame({
    "Class": rep["Class"].map(names),
    "Train_Real_Available": rep["Before"],
    "Real_In_Balanced": rep["After_Undersample"],
    "Synthetic_SMOTE": rep["After_SMOTE"] - rep["After_Undersample"],
    "Balanced_Total": rep["After_SMOTE"],
})
out["Pct_Synthetic"] = (100 * out["Synthetic_SMOTE"] / out["Balanced_Total"]).round(2)
out["Undersampled"] = rep["Before"] > rep["After_Undersample"]

total = out[["Train_Real_Available", "Real_In_Balanced", "Synthetic_SMOTE", "Balanced_Total"]].sum()
print(pd.read_csv(LABELS).to_string(index=False))
print()
print(out.to_string(index=False))
print(f"\nOverall synthetic: {total['Synthetic_SMOTE']:,} / {total['Balanced_Total']:,} "
      f"= {100 * total['Synthetic_SMOTE'] / total['Balanced_Total']:.2f}%")
out.to_csv(OUT, index=False)
print("Saved:", OUT)
