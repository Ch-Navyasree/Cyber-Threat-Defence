from pathlib import Path

import numpy as np
import polars as pl


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "data/processed/transformed/train.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Attack_Type"
METADATA_COLUMN = "Source_File"

TOP_N = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TRANSFORMED FEATURE OUTLIER ANALYSIS")
print("=" * 80)

print("\nLoading transformed training data...")

df = pl.read_parquet(TRAIN_FILE)

print(f"Rows    : {df.height}")
print(f"Columns : {df.width}")


# ============================================================
# IDENTIFY MODEL FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in {
        LABEL_COLUMN,
        METADATA_COLUMN,
    }
]

print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)

print(f"\nModel features: {len(feature_columns)}")

if len(feature_columns) != 69:
    raise ValueError(
        f"Expected 69 model features, found {len(feature_columns)}"
    )


# ============================================================
# NUMERIC SAFETY CHECK
# ============================================================

print("\n" + "=" * 80)
print("NUMERIC SAFETY CHECK")
print("=" * 80)

total_nan = 0
total_inf = 0

for column in feature_columns:

    values = df[column]

    nan_count = values.is_nan().sum()
    inf_count = values.is_infinite().sum()

    total_nan += nan_count
    total_inf += inf_count

print(f"\nTotal NaN      : {total_nan}")
print(f"Total Infinity : {total_inf}")

if total_nan == 0 and total_inf == 0:
    print("PASS: No NaN or infinity values.")
else:
    print("WARNING: NaN or infinity values detected.")


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("TRANSFORMED FEATURE STATISTICS")
print("=" * 80)

statistics = []

for column in feature_columns:

    series = df[column]

    minimum = series.min()
    maximum = series.max()
    median = series.median()
    mean = series.mean()
    std = series.std()

    # RobustScaler uses:
    # transformed = (x - median) / IQR
    #
    # Therefore values far from 0 can indicate extreme observations.

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    statistics.append(
        {
            "Feature": column,
            "Min": minimum,
            "Q1": q1,
            "Median": median,
            "Q3": q3,
            "Max": maximum,
            "Mean": mean,
            "Std": std,
        }
    )


stats_df = pl.DataFrame(statistics)


# ============================================================
# DISPLAY FEATURES WITH LARGEST ABSOLUTE MAXIMUM
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WITH LARGEST ABSOLUTE VALUES")
print("=" * 80)

stats_df = stats_df.with_columns(
    pl.max_horizontal(
        pl.col("Min").abs(),
        pl.col("Max").abs(),
    ).alias("Max_Absolute_Value")
)

largest_features = (
    stats_df
    .sort(
        "Max_Absolute_Value",
        descending=True,
    )
    .head(TOP_N)
)

print(largest_features)


# ============================================================
# COUNT EXTREME VALUES
# ============================================================

print("\n" + "=" * 80)
print("EXTREME VALUE COUNTS")
print("=" * 80)

thresholds = [
    10,
    100,
    1000,
    10000,
]

for threshold in thresholds:

    print("\n" + "-" * 80)
    print(f"Values with |x| > {threshold}")
    print("-" * 80)

    results = []

    for column in feature_columns:

        count = (
            df
            .filter(
                pl.col(column).abs() > threshold
            )
            .height
        )

        if count > 0:
            results.append(
                {
                    "Feature": column,
                    "Count": count,
                    "Percentage": (
                        count / df.height
                    ) * 100,
                }
            )

    if results:

        result_df = (
            pl.DataFrame(results)
            .sort("Count", descending=True)
        )

        print(result_df)

    else:
        print("No values exceed this threshold.")


# ============================================================
# MOST EXTREME ROWS
# ============================================================

print("\n" + "=" * 80)
print("MOST EXTREME VALUES BY FEATURE")
print("=" * 80)

for column in feature_columns:

    max_value = df[column].max()
    min_value = df[column].min()

    max_abs = max(
        abs(max_value),
        abs(min_value),
    )

    if max_abs > 1000:

        print("\n" + "-" * 80)
        print(column)
        print("-" * 80)

        print(f"Minimum : {min_value}")
        print(f"Maximum : {max_value}")

        extreme_rows = (
            df
            .with_columns(
                pl.col(column)
                .abs()
                .alias("__absolute_value")
            )
            .sort(
                "__absolute_value",
                descending=True,
            )
            .select(
                [
                    column,
                    "Attack_Type",
                    "Source_File",
                ]
            )
            .head(5)
        )

        print(extreme_rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OUTLIER ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nRows analyzed    : {df.height}")
print(f"Features analyzed: {len(feature_columns)}")
print(f"Total NaN        : {total_nan}")
print(f"Total Infinity   : {total_inf}")

print("\nNo values were modified.")
print("This script is analysis-only.")

print("\n" + "=" * 80)