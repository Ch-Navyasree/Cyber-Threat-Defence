from pathlib import Path
import numpy as np
import polars as pl


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "data/processed/final/train.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Attack_Type"
METADATA_COLUMN = "Source_File"

PERCENTILES = [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
    0.999,
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FEATURE DISTRIBUTION AND SKEW ANALYSIS")
print("=" * 80)

print("\nLoading final training data...")

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

print("PASS: Exactly 69 model features found.")


# ============================================================
# NUMERIC SAFETY CHECK
# ============================================================

print("\n" + "=" * 80)
print("NUMERIC SAFETY CHECK")
print("=" * 80)

total_nan = 0
total_inf = 0

for feature in feature_columns:

    column = df[feature]

    total_nan += column.null_count()

    if column.dtype in [pl.Float32, pl.Float64]:
        total_inf += column.is_infinite().sum()

print(f"\nTotal NaN      : {total_nan}")
print(f"Total Infinity : {total_inf}")

if total_nan == 0 and total_inf == 0:
    print("PASS: No NaN or infinity values.")
else:
    print("WARNING: NaN or infinity values detected.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_skew(values):
    """
    Calculate Pearson moment coefficient of skewness.

    Returns NaN for constant or insufficient data.
    """

    values = np.asarray(values, dtype=np.float64)

    if len(values) < 3:
        return np.nan

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return 0.0

    centered = values - mean

    return np.mean(centered ** 3) / (std ** 3)


def get_feature_statistics(feature):
    """
    Calculate distribution statistics for one feature.
    """

    values = (
        df
        .select(feature)
        .to_series()
        .drop_nulls()
        .to_numpy()
    )

    values = values.astype(np.float64)

    if len(values) == 0:
        return None

    # Remove infinity defensively for analysis.
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    percentile_values = np.percentile(
        values,
        [p * 100 for p in PERCENTILES]
    )

    q1 = percentile_values[2]
    median = percentile_values[3]
    q3 = percentile_values[4]

    iqr = q3 - q1

    maximum = np.max(values)
    minimum = np.min(values)

    mean = np.mean(values)
    std = np.std(values)

    skewness = calculate_skew(values)

    # Percentage above the 99th percentile.
    p99 = percentile_values[6]

    above_p99 = np.sum(values > p99)

    percentage_above_p99 = (
        above_p99 / len(values)
    ) * 100

    # Percentage of zero values.
    zero_count = np.sum(values == 0)

    percentage_zero = (
        zero_count / len(values)
    ) * 100

    return {
        "Feature": feature,
        "Min": minimum,
        "Q1": q1,
        "Median": median,
        "Q3": q3,
        "P95": percentile_values[5],
        "P99": percentile_values[6],
        "P99.9": percentile_values[7],
        "Max": maximum,
        "Mean": mean,
        "Std": std,
        "IQR": iqr,
        "Skewness": skewness,
        "Above_P99_%": percentage_above_p99,
        "Zero_%": percentage_zero,
    }


# ============================================================
# FEATURE DISTRIBUTION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING FEATURE DISTRIBUTIONS")
print("=" * 80)

statistics = []

for index, feature in enumerate(feature_columns, start=1):

    print(
        f"Analyzing {index:02d}/{len(feature_columns)}: {feature}"
    )

    result = get_feature_statistics(feature)

    if result is not None:
        statistics.append(result)


stats_df = pl.DataFrame(statistics)


# ============================================================
# FORMAT DISPLAY
# ============================================================

display_columns = [
    "Feature",
    "Min",
    "Q1",
    "Median",
    "Q3",
    "P95",
    "P99",
    "P99.9",
    "Max",
    "Mean",
    "Std",
    "IQR",
    "Skewness",
    "Above_P99_%",
    "Zero_%",
]


# ============================================================
# TOP FEATURES BY SKEWNESS
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WITH HIGHEST SKEWNESS")
print("=" * 80)

print(
    stats_df
    .select(display_columns)
    .sort(
        "Skewness",
        descending=True,
        nulls_last=True,
    )
    .head(20)
)


# ============================================================
# TOP FEATURES BY MAXIMUM VALUE
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WITH LARGEST MAXIMUM VALUES")
print("=" * 80)

print(
    stats_df
    .select(display_columns)
    .sort(
        "Max",
        descending=True,
        nulls_last=True,
    )
    .head(20)
)


# ============================================================
# TOP FEATURES BY P99
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WITH LARGEST P99 VALUES")
print("=" * 80)

print(
    stats_df
    .select(display_columns)
    .sort(
        "P99",
        descending=True,
        nulls_last=True,
    )
    .head(20)
)


# ============================================================
# FEATURES WITH EXTREME SKEW
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WITH SKEWNESS > 10")
print("=" * 80)

extreme_skew = (
    stats_df
    .filter(
        pl.col("Skewness") > 10
    )
    .select(display_columns)
    .sort(
        "Skewness",
        descending=True,
    )
)

print(extreme_skew)

print(
    f"\nNumber of features with skewness > 10: "
    f"{extreme_skew.height}"
)


# ============================================================
# FEATURES WITH SKEWNESS > 100
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WITH SKEWNESS > 100")
print("=" * 80)

very_extreme_skew = (
    stats_df
    .filter(
        pl.col("Skewness") > 100
    )
    .select(display_columns)
    .sort(
        "Skewness",
        descending=True,
    )
)

print(very_extreme_skew)

print(
    f"\nNumber of features with skewness > 100: "
    f"{very_extreme_skew.height}"
)


# ============================================================
# ZERO-DOMINATED FEATURES
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WITH HIGH ZERO PERCENTAGE")
print("=" * 80)

print(
    stats_df
    .select(
        [
            "Feature",
            "Zero_%",
            "Median",
            "P95",
            "P99",
            "P99.9",
            "Max",
        ]
    )
    .sort(
        "Zero_%",
        descending=True,
    )
    .head(20)
)


# ============================================================
# P99 TO MAX RATIO
# ============================================================

print("\n" + "=" * 80)
print("FEATURES WHERE MAXIMUM IS FAR ABOVE P99")
print("=" * 80)

ratio_df = (
    stats_df
    .with_columns(
        pl.when(
            pl.col("P99") != 0
        )
        .then(
            pl.col("Max").abs()
            / pl.col("P99").abs()
        )
        .otherwise(None)
        .alias("Max_to_P99_Ratio")
    )
)

print(
    ratio_df
    .select(
        [
            "Feature",
            "P99",
            "P99.9",
            "Max",
            "Max_to_P99_Ratio",
        ]
    )
    .sort(
        "Max_to_P99_Ratio",
        descending=True,
        nulls_last=True,
    )
    .head(20)
)


# ============================================================
# ATTACK-TYPE DISTRIBUTION FOR EXTREME FEATURES
# ============================================================

print("\n" + "=" * 80)
print("EXTREME VALUE DISTRIBUTION BY ATTACK TYPE")
print("=" * 80)

# Select the most skewed features for investigation.
top_skew_features = (
    stats_df
    .sort(
        "Skewness",
        descending=True,
        nulls_last=True,
    )
    .head(10)
    .get_column("Feature")
    .to_list()
)

print(
    "\nAnalyzing the following highly skewed features:"
)

for feature in top_skew_features:
    print(f"  - {feature}")


for feature in top_skew_features:

    print("\n" + "-" * 80)
    print(feature)
    print("-" * 80)

    feature_stats = (
        df
        .group_by(LABEL_COLUMN)
        .agg(
            [
                pl.col(feature).count().alias("Count"),

                pl.col(feature)
                .median()
                .alias("Median"),

                pl.col(feature)
                .quantile(0.95)
                .alias("P95"),

                pl.col(feature)
                .quantile(0.99)
                .alias("P99"),

                pl.col(feature)
                .max()
                .alias("Maximum"),
            ]
        )
        .sort(LABEL_COLUMN)
    )

    print(feature_stats)


# ============================================================
# SOURCE FILE ANALYSIS FOR EXTREME VALUES
# ============================================================

print("\n" + "=" * 80)
print("SOURCE FILE ANALYSIS")
print("=" * 80)

# Investigate the top 5 most skewed features.
source_features = top_skew_features[:5]

for feature in source_features:

    threshold = (
        df[feature]
        .quantile(0.99)
    )

    print("\n" + "-" * 80)
    print(feature)
    print(f"P99 threshold: {threshold}")
    print("-" * 80)

    source_summary = (
        df
        .filter(
            pl.col(feature) > threshold
        )
        .group_by(METADATA_COLUMN)
        .agg(
            [
                pl.len().alias("Extreme_Count"),
                pl.col(feature).max().alias("Maximum"),
            ]
        )
        .sort(
            "Extreme_Count",
            descending=True,
        )
        .head(15)
    )

    print(source_summary)


# ============================================================
# SAVE COMPLETE REPORT
# ============================================================

OUTPUT_FILE = Path(
    "data/processed/transformed/"
    "feature_distribution_analysis.parquet"
)

stats_df.write_parquet(OUTPUT_FILE)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nRows analyzed    : {df.height}")
print(f"Features analyzed: {len(feature_columns)}")
print(f"Total NaN        : {total_nan}")
print(f"Total Infinity   : {total_inf}")

print(f"\nFull feature statistics saved to:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("This script is analysis-only.")
print("No values were modified.")
print("No rows were removed.")
print("No features were clipped.")
print("No transformations were applied.")

print("\nNext step:")
print(
    "Use the output to determine whether extreme/skewed "
    "features require a justified transformation."
)

print("=" * 80)