from pathlib import Path
import polars as pl


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = Path(
    "data/processed/standard/train.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

df = pl.read_parquet(TRAIN_FILE)

print(f"Rows: {df.height:,}")
print(f"Columns: {df.width}")


# ============================================================
# COLUMN TYPE CHECK
# ============================================================

print("\n" + "=" * 70)
print("COLUMN TYPES")
print("=" * 70)

for name, dtype in zip(df.columns, df.dtypes):
    print(f"{name:<40} {dtype}")


# ============================================================
# SELECT TRUE NUMERICAL FEATURES
# ============================================================

exclude_columns = [
    "Attack_Type",
    "Source_File"
]

numeric_types = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64
}

feature_columns = [
    col
    for col, dtype in zip(df.columns, df.dtypes)
    if col not in exclude_columns
    and dtype in numeric_types
]

non_numeric = [
    col
    for col, dtype in zip(df.columns, df.dtypes)
    if col not in exclude_columns
    and dtype not in numeric_types
]

print("\n" + "=" * 70)
print("FEATURE TYPE SUMMARY")
print("=" * 70)

print(f"Expected features: 69")
print(f"Numeric features:  {len(feature_columns)}")
print(f"Non-numeric:       {len(non_numeric)}")

if non_numeric:
    print("\nNon-numeric feature columns:")
    for col in non_numeric:
        print(f"  - {col}")


# ============================================================
# FEATURE AUDIT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE AUDIT")
print("=" * 70)

results = []

for feature in feature_columns:

    col = df[feature]

    results.append({
        "Feature": feature,
        "Type": str(col.dtype),
        "Min": col.min(),
        "Max": col.max(),
        "Mean": col.mean(),
        "Std": col.std(),
        "Zeros": col.eq(0).sum(),
        "Negative": col.lt(0).sum(),
        "Null": col.null_count()
    })


audit = pl.DataFrame(results)

print(audit)


# ============================================================
# NEGATIVE VALUES
# ============================================================

print("\n" + "=" * 70)
print("FEATURES WITH NEGATIVE VALUES")
print("=" * 70)

negative_features = (
    audit
    .filter(pl.col("Negative") > 0)
    .sort("Negative", descending=True)
)

print(negative_features)


# ============================================================
# ZERO-DOMINATED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ZERO-DOMINATED FEATURES (>95% ZERO)")
print("=" * 70)

zero_dominated = (
    audit
    .with_columns(
        (
            pl.col("Zeros") /
            df.height *
            100
        )
        .round(4)
        .alias("Zero_Percentage")
    )
    .filter(
        pl.col("Zero_Percentage") > 95
    )
    .sort("Zero_Percentage", descending=True)
)

print(zero_dominated)


# ============================================================
# RANGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FEATURE RANGE CHECK")
print("=" * 70)

range_check = (
    audit
    .with_columns(
        (
            pl.col("Max") - pl.col("Min")
        ).alias("Range")
    )
    .sort("Range", descending=True)
)

print(
    range_check.select([
        "Feature",
        "Min",
        "Max",
        "Range",
        "Mean",
        "Std"
    ])
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FEATURE AUDIT COMPLETE")
print("=" * 70)