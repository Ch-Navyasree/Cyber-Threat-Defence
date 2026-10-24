from pathlib import Path
import polars as pl


TRAIN_FILE = Path(
    "data/processed/standard/train.parquet"
)

df = pl.read_parquet(TRAIN_FILE)


# ============================================================
# HELPER
# ============================================================

def inspect_feature(feature, condition):
    print("\n" + "=" * 70)
    print(feature)
    print("=" * 70)

    columns = [
        feature,
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Fwd Header Length",
        "Bwd Header Length",
        "min_seg_size_forward",
        "Attack_Type",
        "Source_File"
    ]

    # Remove duplicate column names while preserving order
    columns = list(dict.fromkeys(columns))

    result = (
        df
        .filter(condition)
        .select(columns)
        .head(20)
    )

    print(result)


# ============================================================
# 1. EXTREME NEGATIVE VALUES
# ============================================================

inspect_feature(
    "Fwd Header Length",
    pl.col("Fwd Header Length") < -1_000_000
)

inspect_feature(
    "Bwd Header Length",
    pl.col("Bwd Header Length") < -1_000_000
)

inspect_feature(
    "min_seg_size_forward",
    pl.col("min_seg_size_forward") < -1_000_000
)


# ============================================================
# 2. ALL NEGATIVE FLOW/IAT VALUES
# ============================================================

inspect_feature(
    "Flow IAT Min",
    pl.col("Flow IAT Min") < 0
)

inspect_feature(
    "Fwd IAT Min",
    pl.col("Fwd IAT Min") < 0
)


# ============================================================
# 3. EXTREME WINDOW VALUES
# ============================================================

inspect_feature(
    "Init_Win_bytes_forward",
    pl.col("Init_Win_bytes_forward") < 0
)

inspect_feature(
    "Init_Win_bytes_backward",
    pl.col("Init_Win_bytes_backward") < 0
)


# ============================================================
# 4. EXACT MINIMUM VALUES
# ============================================================

print("\n" + "=" * 70)
print("EXTREME MINIMUM VALUES")
print("=" * 70)

features = [
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

for feature in features:

    minimum = df[feature].min()

    print("\n" + "-" * 70)
    print(feature)
    print("-" * 70)

    print(f"Minimum: {minimum}")

    columns = [
        feature,
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Fwd Header Length",
        "Bwd Header Length",
        "min_seg_size_forward",
        "Attack_Type",
        "Source_File"
    ]

    columns = list(dict.fromkeys(columns))

    result = (
        df
        .filter(pl.col(feature) == minimum)
        .select(columns)
        .head(10)
    )

    print(result)


print("\n" + "=" * 70)
print("ANOMALY INVESTIGATION COMPLETE")
print("=" * 70)