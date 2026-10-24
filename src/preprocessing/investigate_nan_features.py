from pathlib import Path
import polars as pl


# ============================================================
# PATH
# ============================================================

DATA_FILE = Path(
    "data/processed/final/train.parquet"
)

LABEL_COLUMN = "Attack_Type"
METADATA_COLUMN = "Source_File"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("INVESTIGATING HIDDEN NaN VALUES")
print("=" * 70)

df = pl.read_parquet(DATA_FILE)

feature_columns = [
    column
    for column in df.columns
    if column not in {LABEL_COLUMN, METADATA_COLUMN}
]

print(f"\nRows    : {df.height:,}")
print(f"Features: {len(feature_columns)}")


# ============================================================
# FIND NaNs
# ============================================================

print("\n" + "=" * 70)
print("NaN COUNTS BY FEATURE")
print("=" * 70)

nan_results = []

for feature in feature_columns:

    column = df[feature]

    if column.dtype in {pl.Float32, pl.Float64}:

        nan_count = column.is_nan().sum()

        if nan_count > 0:
            nan_results.append(
                (feature, nan_count)
            )


if len(nan_results) == 0:

    print("\nNo NaN values found.")

else:

    nan_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for feature, count in nan_results:
        print(
            f"{feature:40} : {count:,}"
        )


# ============================================================
# TOTAL
# ============================================================

total_nan = sum(
    count for _, count in nan_results
)

print("\n" + "=" * 70)
print("TOTAL")
print("=" * 70)

print(f"\nTotal NaN values: {total_nan:,}")


# ============================================================
# SHOW AFFECTED ROWS
# ============================================================

if total_nan > 0:

    affected_features = [
        feature
        for feature, _ in nan_results
    ]

    print("\n" + "=" * 70)
    print("SAMPLE AFFECTED ROWS")
    print("=" * 70)

    for feature in affected_features:

        print(f"\n--- {feature} ---")

        affected = (
            df
            .filter(pl.col(feature).is_nan())
            .select(
                [
                    feature,
                    LABEL_COLUMN,
                    METADATA_COLUMN,
                ]
            )
            .head(10)
        )

        print(affected)


# ============================================================
# FINITE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINITE VALUE CHECK")
print("=" * 70)

for feature in feature_columns:

    column = df[feature]

    if column.dtype in {pl.Float32, pl.Float64}:

        nan_count = column.is_nan().sum()
        inf_count = column.is_infinite().sum()

        if nan_count > 0 or inf_count > 0:

            print(
                f"{feature}: "
                f"NaN={nan_count:,}, "
                f"Infinity={inf_count:,}"
            )


print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)