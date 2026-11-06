import os

import numpy as np
import pandas as pd

# ==================================================
# CONFIG
# ==================================================

INPUT_FILE = "results/final_merged_dataset.csv"
OUTPUT_FILE = "results/final_merged_dataset_coalesced.csv"

CHUNK_SIZE = 1_000_000

os.makedirs("results", exist_ok=True)

# ==================================================
# COLUMN MAP
# ==================================================
#
# merge_datasets.py concatenates CIC-IDS2017 (long column names) and
# CSE-CIC-IDS2018 (short column names). pd.concat unions the columns, so every
# row has NaN in the ~half of the columns its source dataset never had. The
# downstream numeric cleaning then turns those NaNs into 0, producing a
# near-perfect dataset-origin fingerprint entangled with the class labels.
#
# This script collapses every long/short name pair into ONE canonical column
# (short-name convention), taking the populated value from whichever side the
# row actually carries, and drops both originals. Result: ~half the columns,
# no origin fingerprint.

# canonical (short / 2018 name) -> long (2017 name)
PAIRS = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

# present in BOTH source datasets under the identical name -> keep as-is
SHARED = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd URG Flags",
    "CWE Flag Count", "Down/Up Ratio",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# only exists in CSE-CIC-IDS2018 (no 2017 equivalent) -> keep, but it will be 0
# for every CIC-IDS2017-origin row (that export has no Protocol column)
UNPAIRED_KEEP = ["Protocol"]

# identifiers / junk -> drop entirely (not features; Fwd Header Length.1 is the
# known duplicate-header artifact, Src Port is a high-cardinality identifier)
DROP = ["Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port", "Fwd Header Length.1"]

# columns that are 100% zero across all 18.3M rows in BOTH source datasets
# (verified): the 6 bulk-rate features CICFlowMeter never implements, and the
# backward PSH/URG flag counts it never emits. Dropped as zero-information.
# Listed under both naming conventions since the merge carries both.
DEAD = [
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Bwd PSH Flags", "Bwd URG Flags",
]

FINAL_ORDER = list(PAIRS.keys()) + SHARED + UNPAIRED_KEEP + ["Label"]


# ==================================================
# VALIDATE MAP AGAINST HEADER
# ==================================================

header = [c.strip() for c in pd.read_csv(INPUT_FILE, nrows=0).columns]

expected = (
    set(PAIRS) | set(PAIRS.values()) | set(SHARED) | set(UNPAIRED_KEEP)
    | set(DROP) | set(DEAD) | {"Label"}
)
missing = sorted(expected - set(header))
unaccounted = sorted(set(header) - expected)
if missing:
    raise SystemExit(f"columns named in the map are absent from {INPUT_FILE}: {missing}")
if unaccounted:
    raise SystemExit(f"columns in {INPUT_FILE} not handled by the map: {unaccounted}")

print(f"Map validated against {INPUT_FILE} header ({len(header)} columns).")
print(f"  {len(PAIRS)} long/short pairs -> coalesced")
print(f"  {len(SHARED)} shared columns -> kept as-is")
print(f"  {len(UNPAIRED_KEEP)} unpaired kept: {UNPAIRED_KEEP}")
print(f"  {len(DROP)} identifier/junk columns dropped: {DROP}")
print(f"  {len(DEAD)} all-zero columns dropped (both namings): {sorted(set(DEAD))}")
print(f"  final: {len(FINAL_ORDER)} columns (incl. Label)")


# ==================================================
# COALESCE (CHUNKED)
# ==================================================

def coalesce_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = chunk.columns.str.strip()

    for short, long in PAIRS.items():
        s = pd.to_numeric(chunk[short], errors="coerce")
        l = pd.to_numeric(chunk[long], errors="coerce")
        # take the short value where it is a real reading, else the long value.
        # each row carries exactly one origin's columns, so exactly one side is
        # populated; ties (both populated) prefer the short/2018 value.
        combined = s.where(s.notna() & (s != 0), l)
        chunk[short] = combined

    for col in SHARED + UNPAIRED_KEEP:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    out = chunk.reindex(columns=FINAL_ORDER)
    num_cols = [c for c in FINAL_ORDER if c != "Label"]
    out[num_cols] = (
        out[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    )
    return out


total_rows = 0
origin_2017 = 0  # rows carrying the 2017-only 'Destination Port' column
zero_counts = pd.Series(0, index=[c for c in FINAL_ORDER if c != "Label"], dtype="int64")
label_counts: dict[str, int] = {}
first = True

reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)
for i, chunk in enumerate(reader, start=1):
    origin_2017 += int(chunk["Destination Port"].notna().sum())

    out = coalesce_chunk(chunk)

    total_rows += len(out)
    for c in zero_counts.index:
        zero_counts[c] += int((out[c] == 0).sum())
    for k, v in out["Label"].value_counts().items():
        label_counts[k] = label_counts.get(k, 0) + int(v)

    out.to_csv(OUTPUT_FILE, index=False, mode="w" if first else "a", header=first)
    first = False
    print(f"  chunk {i}: {total_rows:,} rows written")


# ==================================================
# VALIDATION REPORT
# ==================================================

print("\n" + "=" * 55)
print("COALESCE COMPLETE")
print("=" * 55)
print(f"Output           : {OUTPUT_FILE}")
print(f"Rows             : {total_rows:,}")
print(f"Columns          : {len(FINAL_ORDER)} ({len(FINAL_ORDER) - 1} features + Label)")
print(f"CIC-IDS2017 origin  : {origin_2017:,}")
print(f"CSE-CIC-IDS2018 origin: {total_rows - origin_2017:,}")

print("\nLabel distribution:")
for k in sorted(label_counts):
    print(f"  {k:<14} {label_counts[k]:,}")

all_zero = zero_counts[zero_counts == total_rows]
print(f"\nColumns that are 100% zero across all rows: {len(all_zero)}")
for c in all_zero.index:
    print(f"  {c}")
if len(all_zero) == 0:
    print("  (none)")

print("\nTop 12 columns by zero fraction:")
frac = (zero_counts / total_rows).sort_values(ascending=False)
for c, f in frac.head(12).items():
    print(f"  {f * 100:6.2f}%  {c}")
