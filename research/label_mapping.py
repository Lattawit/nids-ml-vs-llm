import pandas as pd
import os
import glob

# ==================================================
# CONFIGURATION
# ==================================================

DATASETS = [
    "cleaned/CIC-IDS2017",
    "cleaned/CSE-CIC-IDS2018"
]

OUTPUT_FOLDER = "mapped"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================================================
# LABEL MAPPING
# ==================================================

label_mapping = {

    # Benign
    "BENIGN": "Benign",
    "Benign": "Benign",

    # Brute Force
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",

    # Web Attacks
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack � Brute Force": "Web Attack",

    "Web Attack - XSS": "Web Attack",
    "Web Attack � XSS": "Web Attack",
    "Brute Force -XSS": "Web Attack",

    "Web Attack - Sql Injection": "Web Attack",
    "Web Attack � Sql Injection": "Web Attack",
    "SQL Injection": "Web Attack",

    "Brute Force -Web": "Web Attack",

    # DoS / DDoS
    "DoS Hulk": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DDoS": "DoS/DDoS",

    "DoS attacks-Hulk": "DoS/DDoS",
    "DoS attacks-GoldenEye": "DoS/DDoS",
    "DoS attacks-SlowHTTPTest": "DoS/DDoS",
    "DoS attacks-Slowloris": "DoS/DDoS",

    "DDOS attack-HOIC": "DoS/DDoS",
    "DDoS attacks-LOIC-HTTP": "DoS/DDoS",
    "DDOS attack-LOIC-UDP": "DoS/DDoS",

    # Bot
    "Bot": "Bot",

    # PortScan
    "PortScan": "PortScan",

    # Infiltration
    "Infiltration": "Infiltration",
    "Infilteration": "Infiltration",

    # Heartbleed
    "Heartbleed": "Heartbleed"

}

# ==================================================
# FINAL CLASS COUNTS
# ==================================================

all_class_counts = {}

# ==================================================
# PROCESS FILES
# ==================================================

for dataset_folder in DATASETS:

    dataset_name = os.path.basename(dataset_folder)

    output_dataset_folder = os.path.join(
        OUTPUT_FOLDER,
        dataset_name
    )

    os.makedirs(
        output_dataset_folder,
        exist_ok=True
    )

    csv_files = glob.glob(
        os.path.join(dataset_folder, "*.csv")
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Files Found: {len(csv_files)}")

    for file_path in csv_files:

        print(
            f"Processing: "
            f"{os.path.basename(file_path)}"
        )

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        df.columns = df.columns.str.strip()

        # Apply label mapping
        df["Label"] = (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.replace("�", "-", regex=False)
            .replace(label_mapping)
        )

        # Count classes
        counts = df["Label"].value_counts()

        for label, count in counts.items():

            all_class_counts[label] = (
                all_class_counts.get(label, 0)
                + count
            )

        # Save mapped file
        output_file = os.path.join(
            output_dataset_folder,
            os.path.basename(file_path)
        )

        df.to_csv(
            output_file,
            index=False
        )

# ==================================================
# CREATE CLASS REPORT
# ==================================================

report_df = pd.DataFrame(
    all_class_counts.items(),
    columns=[
        "Class",
        "Number of Records"
    ]
)

report_df["Percentage (%)"] = (
    report_df["Number of Records"]
    /
    report_df["Number of Records"].sum()
    * 100
).round(2)

report_df = report_df.sort_values(
    by="Number of Records",
    ascending=False
)

report_df.to_csv(
    "results/final_class_distribution.csv",
    index=False
)

print("\n================================")
print("FINAL CLASS DISTRIBUTION")
print("================================")

print(report_df)

print(
    "\nSaved:"
    " results/final_class_distribution.csv"
)