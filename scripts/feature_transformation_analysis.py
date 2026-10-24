# =============================================================================
# FEATURE QUALITY, REDUNDANCY & CORRELATION ANALYSIS
# =============================================================================
#
# PURPOSE:
#   Analysis-only examination of the 69 model features.
#
# THIS SCRIPT DOES NOT:
#   - modify values
#   - remove rows
#   - remove features
#   - clip outliers
#   - transform skewed features
#   - scale features
#
# OUTPUTS:
#   1. feature_quality_report.parquet
#   2. feature_quality_report.csv
#   3. exact_duplicate_features.csv
#   4. constant_features.csv
#   5. near_zero_variance_features.csv
#   6. high_correlation_pairs.csv
#   7. feature_unique_values.csv
# =============================================================================

import os
import math
import polars as pl

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATH = "data/processed/transformed/train.parquet"

OUTPUT_DIR = r"data\processed\transformed\feature_quality"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Your expected model features
TARGET_COLUMN = "Attack_Type"

# Correlation threshold
HIGH_CORRELATION_THRESHOLD = 0.95

# Near-zero variance threshold:
# A feature is flagged when its most common value represents >= 99.9%
# of all rows.
NEAR_ZERO_VARIANCE_THRESHOLD = 99.9


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("FEATURE QUALITY, REDUNDANCY & CORRELATION ANALYSIS")
print("=" * 80)

print("\nLoading final training data...")

df = pl.read_parquet(INPUT_PATH)

print(f"Rows    : {df.height:,}")
print(f"Columns : {df.width}")


# =============================================================================
# IDENTIFY MODEL FEATURES
# =============================================================================

model_features = [
    c for c in df.columns
    if c != TARGET_COLUMN
]

print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)

print(f"\nTarget column : {TARGET_COLUMN}")
print(f"Model features: {len(model_features)}")

if len(model_features) != 69:
    print(
        f"WARNING: Expected 69 model features, "
        f"but found {len(model_features)}."
    )
else:
    print("PASS: Exactly 69 model features found.")


# =============================================================================
# NUMERIC SAFETY CHECK
# =============================================================================

print("\n" + "=" * 80)
print("NUMERIC SAFETY CHECK")
print("=" * 80)

numeric_features = []

for feature in model_features:
    dtype = df.schema[feature]

    if dtype.is_numeric():
        numeric_features.append(feature)

print(f"\nNumeric model features: {len(numeric_features)}")

nan_count = 0
null_count = 0
inf_count = 0

for feature in numeric_features:

    null_count += df.select(
        pl.col(feature).null_count()
    ).item()

    # NaN check only works on floating-point columns
    dtype = df.schema[feature]

    if dtype in [
        pl.Float32,
        pl.Float64
    ]:
        nan_count += df.select(
            pl.col(feature).is_nan().sum()
        ).item()

        inf_count += df.select(
            pl.col(feature).is_infinite().sum()
        ).item()

print(f"Total Null     : {null_count:,}")
print(f"Total NaN      : {nan_count:,}")
print(f"Total Infinity : {inf_count:,}")

if null_count == 0 and nan_count == 0 and inf_count == 0:
    print("PASS: No null, NaN or infinity values.")


# =============================================================================
# FEATURE QUALITY ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING FEATURE QUALITY")
print("=" * 80)

quality_records = []

for i, feature in enumerate(numeric_features, start=1):

    print(
        f"Analyzing {i:02d}/{len(numeric_features)}: {feature}"
    )

    col = df.get_column(feature)

    total = df.height

    nulls = col.null_count()

    # -------------------------------------------------------------------------
    # Unique values
    # -------------------------------------------------------------------------

    unique_count = col.n_unique()

    # -------------------------------------------------------------------------
    # Zero percentage
    # -------------------------------------------------------------------------

    zero_count = df.select(
        (pl.col(feature) == 0).sum()
    ).item()

    zero_pct = (
        zero_count / total * 100
        if total > 0
        else 0
    )

    # -------------------------------------------------------------------------
    # Most common value percentage
    # -------------------------------------------------------------------------

    value_counts = (
        df
        .select(pl.col(feature))
        .group_by(feature)
        .len()
        .sort("len", descending=True)
    )

    if value_counts.height > 0:

        most_common_count = value_counts["len"][0]

        most_common_pct = (
            most_common_count / total * 100
        )

    else:
        most_common_count = 0
        most_common_pct = 0

    # -------------------------------------------------------------------------
    # Basic statistics
    # -------------------------------------------------------------------------

    stats = df.select(
        [
            pl.col(feature).min().alias("min"),
            pl.col(feature).mean().alias("mean"),
            pl.col(feature).median().alias("median"),
            pl.col(feature).max().alias("max"),
            pl.col(feature).quantile(0.95).alias("p95"),
            pl.col(feature).quantile(0.99).alias("p99"),
            pl.col(feature).quantile(0.999).alias("p99_9"),
            pl.col(feature).std().alias("std"),
        ]
    ).row(0)

    (
        min_value,
        mean_value,
        median_value,
        max_value,
        p95,
        p99,
        p99_9,
        std_value
    ) = stats

    # -------------------------------------------------------------------------
    # IQR
    # -------------------------------------------------------------------------

    q1 = df.select(
        pl.col(feature).quantile(0.25)
    ).item()

    q3 = df.select(
        pl.col(feature).quantile(0.75)
    ).item()

    iqr = q3 - q1

    # -------------------------------------------------------------------------
    # Skewness
    # -------------------------------------------------------------------------

    skewness = df.select(
        pl.col(feature).skew()
    ).item()

    # -------------------------------------------------------------------------
    # Max / P99 ratio
    # -------------------------------------------------------------------------

    if p99 is not None and p99 != 0:

        max_to_p99 = max_value / p99

    else:

        max_to_p99 = None

    # -------------------------------------------------------------------------
    # Flags
    # -------------------------------------------------------------------------

    is_constant = unique_count <= 1

    is_near_zero_variance = (
        most_common_pct >= NEAR_ZERO_VARIANCE_THRESHOLD
    )

    is_high_zero = zero_pct >= 95.0

    is_highly_skewed = (
        skewness is not None
        and abs(skewness) > 10
    )

    quality_records.append(
        {
            "Feature": feature,
            "Dtype": str(df.schema[feature]),
            "Unique_Count": unique_count,
            "Null_Count": nulls,
            "Zero_Count": zero_count,
            "Zero_%": zero_pct,
            "Most_Common_%": most_common_pct,
            "Min": min_value,
            "Q1": q1,
            "Median": median_value,
            "Q3": q3,
            "IQR": iqr,
            "Mean": mean_value,
            "Std": std_value,
            "P95": p95,
            "P99": p99,
            "P99_9": p99_9,
            "Max": max_value,
            "Max_to_P99": max_to_p99,
            "Skewness": skewness,
            "Is_Constant": is_constant,
            "Is_Near_Zero_Variance": is_near_zero_variance,
            "Is_High_Zero": is_high_zero,
            "Is_Highly_Skewed": is_highly_skewed,
        }
    )


quality_df = pl.DataFrame(quality_records)


# =============================================================================
# SAVE COMPLETE QUALITY REPORT
# =============================================================================

quality_parquet = os.path.join(
    OUTPUT_DIR,
    "feature_quality_report.parquet"
)

quality_csv = os.path.join(
    OUTPUT_DIR,
    "feature_quality_report.csv"
)

quality_df.write_parquet(quality_parquet)
quality_df.write_csv(quality_csv)

print("\nPASS: Feature quality report generated.")


# =============================================================================
# CONSTANT FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("CONSTANT FEATURES")
print("=" * 80)

constant_df = (
    quality_df
    .filter(pl.col("Is_Constant"))
    .select(
        [
            "Feature",
            "Unique_Count",
            "Zero_%",
            "Min",
            "Max"
        ]
    )
)

if constant_df.height == 0:

    print("None found.")

else:

    print(constant_df)

constant_df.write_csv(
    os.path.join(
        OUTPUT_DIR,
        "constant_features.csv"
    )
)


# =============================================================================
# NEAR-ZERO VARIANCE FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("NEAR-ZERO-VARIANCE FEATURES")
print("=" * 80)

nzv_df = (
    quality_df
    .filter(pl.col("Is_Near_Zero_Variance"))
    .sort("Most_Common_%", descending=True)
)

if nzv_df.height == 0:

    print("None found.")

else:

    print(nzv_df.select(
        [
            "Feature",
            "Unique_Count",
            "Most_Common_%",
            "Zero_%",
            "Min",
            "Median",
            "Max"
        ]
    ))

nzv_df.write_csv(
    os.path.join(
        OUTPUT_DIR,
        "near_zero_variance_features.csv"
    )
)


# =============================================================================
# HIGH-ZERO FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("FEATURES WITH >=95% ZERO VALUES")
print("=" * 80)

high_zero_df = (
    quality_df
    .filter(pl.col("Zero_%") >= 95)
    .sort("Zero_%", descending=True)
)

if high_zero_df.height == 0:

    print("None found.")

else:

    print(
        high_zero_df.select(
            [
                "Feature",
                "Zero_%",
                "Unique_Count",
                "Median",
                "P95",
                "P99",
                "Max"
            ]
        )
    )


# =============================================================================
# HIGHLY SKEWED FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("FEATURES WITH |SKEWNESS| > 10")
print("=" * 80)

skewed_df = (
    quality_df
    .filter(
        pl.col("Skewness").abs() > 10
    )
    .sort(
        "Skewness",
        descending=True
    )
)

print(
    skewed_df.select(
        [
            "Feature",
            "Skewness",
            "Zero_%",
            "Median",
            "P99",
            "Max",
            "Max_to_P99"
        ]
    )
)


# =============================================================================
# EXACT DUPLICATE FEATURE DETECTION
# =============================================================================

print("\n" + "=" * 80)
print("CHECKING EXACT DUPLICATE FEATURES")
print("=" * 80)

duplicate_pairs = []

for i in range(len(numeric_features)):

    feature_a = numeric_features[i]

    for j in range(i + 1, len(numeric_features)):

        feature_b = numeric_features[j]

        # Compare columns exactly
        identical = df.select(
            (
                pl.col(feature_a)
                == pl.col(feature_b)
            ).all()
        ).item()

        if identical:

            duplicate_pairs.append(
                {
                    "Feature_A": feature_a,
                    "Feature_B": feature_b,
                    "Relationship": "EXACT_DUPLICATE"
                }
            )


duplicate_df = pl.DataFrame(
    duplicate_pairs,
    schema={
        "Feature_A": pl.String,
        "Feature_B": pl.String,
        "Relationship": pl.String
    }
)

if duplicate_df.height == 0:

    print("No exact duplicate feature pairs found.")

else:

    print(
        f"Found {duplicate_df.height} exact duplicate pairs:"
    )

    print(duplicate_df)

duplicate_df.write_csv(
    os.path.join(
        OUTPUT_DIR,
        "exact_duplicate_features.csv"
    )
)


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING FEATURE CORRELATIONS")
print("=" * 80)

print(
    f"\nThreshold: |correlation| >= "
    f"{HIGH_CORRELATION_THRESHOLD}"
)

# Polars correlation matrix
corr_df = df.select(
    numeric_features
).corr()

high_corr_pairs = []

for i in range(len(numeric_features)):

    feature_a = numeric_features[i]

    for j in range(i + 1, len(numeric_features)):

        feature_b = numeric_features[j]

        correlation = corr_df[feature_b][i]

        if correlation is None:
            continue

        if abs(correlation) >= HIGH_CORRELATION_THRESHOLD:

            high_corr_pairs.append(
                {
                    "Feature_A": feature_a,
                    "Feature_B": feature_b,
                    "Correlation": correlation,
                    "Absolute_Correlation": abs(correlation)
                }
            )


high_corr_df = pl.DataFrame(
    high_corr_pairs,
    schema={
        "Feature_A": pl.String,
        "Feature_B": pl.String,
        "Correlation": pl.Float64,
        "Absolute_Correlation": pl.Float64
    }
).sort(
    "Absolute_Correlation",
    descending=True
)

if high_corr_df.height == 0:

    print("No highly correlated pairs found.")

else:

    print(
        f"Found {high_corr_df.height} highly correlated pairs:"
    )

    print(high_corr_df)


high_corr_df.write_csv(
    os.path.join(
        OUTPUT_DIR,
        "high_correlation_pairs.csv"
    )
)


# =============================================================================
# FEATURE SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL FEATURE QUALITY SUMMARY")
print("=" * 80)

print(f"\nTotal model features        : {len(model_features)}")
print(f"Numeric features            : {len(numeric_features)}")

print(
    f"Constant features           : "
    f"{quality_df.filter(pl.col('Is_Constant')).height}"
)

print(
    f"Near-zero variance features : "
    f"{quality_df.filter(pl.col('Is_Near_Zero_Variance')).height}"
)

print(
    f">=95% zero features         : "
    f"{quality_df.filter(pl.col('Zero_%') >= 95).height}"
)

print(
    f"|Skewness| > 10 features    : "
    f"{quality_df.filter(pl.col('Skewness').abs() > 10).height}"
)

print(
    f"Exact duplicate pairs       : "
    f"{duplicate_df.height}"
)

print(
    f"High correlation pairs      : "
    f"{high_corr_df.height}"
)


# =============================================================================
# OUTPUT LOCATIONS
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print(
    "1.",
    quality_parquet
)

print(
    "2.",
    quality_csv
)

print(
    "3.",
    os.path.join(
        OUTPUT_DIR,
        "exact_duplicate_features.csv"
    )
)

print(
    "4.",
    os.path.join(
        OUTPUT_DIR,
        "constant_features.csv"
    )
)

print(
    "5.",
    os.path.join(
        OUTPUT_DIR,
        "near_zero_variance_features.csv"
    )
)

print(
    "6.",
    os.path.join(
        OUTPUT_DIR,
        "high_correlation_pairs.csv"
    )
)

print("\nIMPORTANT:")
print("No values were modified.")
print("No rows were removed.")
print("No features were removed.")
print("No transformations were applied.")
print("No clipping was applied.")

print("\nNext step:")
print(
    "Review the duplicate, constant, near-zero-variance, "
    "and correlation reports before deciding on feature removal "
    "or transformation."
)

print("=" * 80)