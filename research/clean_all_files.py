import pandas as pd
import numpy as np
import os
import glob
import gc

# ==================================================
# CHANGE THIS TO THE DATASET YOU WANT TO CLEAN
# ==================================================
DATASET_FOLDER = r"datasets/CSE-CIC-IDS2018"

# ==================================================
# Dataset Name
# ==================================================
dataset_name = os.path.basename(DATASET_FOLDER)

# ==================================================
# Create Output Folders
# ==================================================
cleaned_folder = os.path.join(
    "cleaned",
    dataset_name
)

os.makedirs(
    cleaned_folder,
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)

# ==================================================
# Dataset-specific Report File
# ==================================================
report_file = os.path.join(
    "results",
    f"{dataset_name}_cleaning_report.csv"
)

print(f"\nDataset: {dataset_name}")
print(f"Output Folder: {cleaned_folder}")
print(f"Report File: {report_file}")

# ==================================================
# Get all CSV files
# ==================================================
csv_files = glob.glob(
    os.path.join(DATASET_FOLDER, "*.csv")
)

print(f"\nFound {len(csv_files)} CSV files.\n")

# ==================================================
# Load existing report if available
# ==================================================
if os.path.exists(report_file):
    report_df = pd.read_csv(report_file)
else:
    report_df = pd.DataFrame()

# ==================================================
# Process files one-by-one
# ==================================================
for index, file_path in enumerate(csv_files, start=1):

    try:

        print("=" * 60)
        print(
            f"[{index}/{len(csv_files)}] "
            f"Processing: "
            f"{os.path.basename(file_path)}"
        )

        # --------------------------
        # Read CSV
        # --------------------------
        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        # --------------------------
        # Before stats
        # --------------------------
        rows_before = len(df)

        missing_before = (
            df.isnull().sum().sum()
        )

        infinite_before = np.isinf(
            df.select_dtypes(include=[np.number])
        ).sum().sum()

        duplicates_before = (
            df.duplicated().sum()
        )

        memory_before = (
            df.memory_usage(deep=True)
            .sum() / 1024**2
        )

        # --------------------------
        # Cleaning
        # --------------------------
        df.columns = (
            df.columns.str.strip()
        )

        df = df[df["Label"] != "Label"]

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        df.dropna(inplace=True)

        df.drop_duplicates(inplace=True)

        # --------------------------
        # Label summary
        # --------------------------
        label_counts = (
            df["Label"]
            .value_counts()
            .to_dict()
        )

        # --------------------------
        # After stats
        # --------------------------
        rows_after = len(df)

        missing_after = (
            df.isnull().sum().sum()
        )

        infinite_after = np.isinf(
            df.select_dtypes(include=[np.number])
        ).sum().sum()

        memory_after = (
            df.memory_usage(deep=True)
            .sum() / 1024**2
        )

        rows_removed = (
            rows_before - rows_after
        )

        classes = (
            df["Label"].nunique()
        )

        # --------------------------
        # Save cleaned file
        # --------------------------
        cleaned_path = os.path.join(
            cleaned_folder,
            os.path.basename(file_path)
        )

        df.to_csv(
            cleaned_path,
            index=False
        )

        # --------------------------
        # Print summary
        # --------------------------
        print(
            f"Rows: "
            f"{rows_before:,} -> "
            f"{rows_after:,}"
        )

        print(
            f"Removed: "
            f"{rows_removed:,}"
        )

        print(
            f"Classes: "
            f"{classes}"
        )

        print(
            f"Labels: "
            f"{label_counts}"
        )

        # --------------------------
        # Update report
        # --------------------------
        report_row = pd.DataFrame([{
            "File": os.path.basename(file_path),
            "Rows Before": rows_before,
            "Rows After": rows_after,
            "Rows Removed": rows_removed,
            "Missing Before": missing_before,
            "Infinite Before": infinite_before,
            "Missing After": missing_after,
            "Infinite After": infinite_after,
            "Duplicate Rows Found": duplicates_before,
            "Memory Before (MB)": round(memory_before, 2),
            "Memory After (MB)": round(memory_after, 2),
            "Classes": classes
        }])

        report_df = report_df[
            report_df["File"]
            != os.path.basename(file_path)
        ] if not report_df.empty else report_df

        report_df = pd.concat(
            [report_df, report_row],
            ignore_index=True
        )

        report_df.to_csv(
            report_file,
            index=False
        )

        # --------------------------
        # Free memory
        # --------------------------
        del df
        gc.collect()

    except Exception as e:

        print(
            f"ERROR: "
            f"{os.path.basename(file_path)}"
        )

        print(e)

print("\nCleaning completed.")