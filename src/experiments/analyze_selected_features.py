from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# 1. PATHS
# =========================================================

project_root = Path(__file__).resolve().parents[2]

input_file = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "train.parquet"
)

output_dir = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("\nLoading selected training dataset...")

df = pd.read_parquet(input_file)

print("Dataset shape:", df.shape)


# =========================================================
# 3. IDENTIFY FEATURES
# =========================================================

target = "Attack_Type"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found."
    )

features = [
    column
    for column in df.columns
    if column != target
]


print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Target:", target)
print("Model features:", len(features))


# =========================================================
# 4. NUMERIC SAFETY CHECK
# =========================================================

print("\n========================================")
print("NUMERIC SAFETY CHECK")
print("========================================")

numeric_features = df[features].select_dtypes(
    include=np.number
).columns.tolist()

non_numeric_features = [
    column
    for column in features
    if column not in numeric_features
]

print("Numeric features:", len(numeric_features))

if non_numeric_features:
    print("\nNon-numeric features:")
    for feature in non_numeric_features:
        print(" -", feature)
else:
    print("All model features are numeric.")


# ---------------------------------------------------------
# Nulls
# ---------------------------------------------------------

null_counts = df[numeric_features].isnull().sum()

total_nulls = int(null_counts.sum())

print("\nTotal null values:", total_nulls)


# ---------------------------------------------------------
# NaNs
# ---------------------------------------------------------

nan_count = int(
    df[numeric_features]
    .isna()
    .sum()
    .sum()
)

print("Total NaN values:", nan_count)


# ---------------------------------------------------------
# Infinite values
# ---------------------------------------------------------

numeric_values = df[numeric_features].to_numpy()

infinite_count = int(
    np.isinf(numeric_values).sum()
)

print("Total infinite values:", infinite_count)


# =========================================================
# 5. FEATURE QUALITY ANALYSIS
# =========================================================

print("\n========================================")
print("FEATURE QUALITY ANALYSIS")
print("========================================")

rows = []

for feature in numeric_features:

    series = df[feature]

    zero_count = int((series == 0).sum())
    unique_count = int(series.nunique(dropna=False))

    zero_percentage = (
        zero_count / len(series) * 100
    )

    most_common_percentage = (
        series.value_counts(
            normalize=True,
            dropna=False
        ).iloc[0] * 100
    )

    is_constant = unique_count <= 1

    # Near-zero variance:
    # 99.99% or more of observations have the same value
    is_near_zero_variance = (
        most_common_percentage >= 99.99
    )

    # High zero:
    # 95% or more observations are zero
    is_high_zero = (
        zero_percentage >= 95
    )

    skewness = series.skew()

    is_highly_skewed = (
        abs(skewness) >= 3
    )

    rows.append({
        "Feature": feature,
        "Dtype": str(series.dtype),
        "Unique_Count": unique_count,
        "Null_Count": int(series.isnull().sum()),
        "Zero_Count": zero_count,
        "Zero_%": zero_percentage,
        "Most_Common_%": most_common_percentage,
        "Min": series.min(),
        "Q1": series.quantile(0.25),
        "Median": series.median(),
        "Q3": series.quantile(0.75),
        "IQR": series.quantile(0.75)
        - series.quantile(0.25),
        "Mean": series.mean(),
        "Std": series.std(),
        "P95": series.quantile(0.95),
        "P99": series.quantile(0.99),
        "P99_9": series.quantile(0.999),
        "Max": series.max(),
        "Max_to_P99": (
            series.max() / series.quantile(0.99)
            if series.quantile(0.99) != 0
            else np.nan
        ),
        "Skewness": skewness,
        "Is_Constant": is_constant,
        "Is_Near_Zero_Variance": is_near_zero_variance,
        "Is_High_Zero": is_high_zero,
        "Is_Highly_Skewed": is_highly_skewed,
    })


quality_df = pd.DataFrame(rows)


# =========================================================
# 6. SAVE QUALITY REPORT
# =========================================================

quality_file = (
    output_dir
    / "feature_quality_report.parquet"
)

quality_df.to_parquet(
    quality_file,
    index=False
)

print("\nQuality report saved to:")
print(quality_file)


# =========================================================
# 7. SUMMARY
# =========================================================

print("\n========================================")
print("QUALITY SUMMARY")
print("========================================")

print(
    "Constant features:",
    int(quality_df["Is_Constant"].sum())
)

print(
    "Near-zero-variance features:",
    int(
        quality_df[
            "Is_Near_Zero_Variance"
        ].sum()
    )
)

print(
    "High-zero features:",
    int(
        quality_df[
            "Is_High_Zero"
        ].sum()
    )
)

print(
    "Highly skewed features:",
    int(
        quality_df[
            "Is_Highly_Skewed"
        ].sum()
    )
)


# =========================================================
# 8. DISPLAY CANDIDATES
# =========================================================

print("\n========================================")
print("NEAR-ZERO-VARIANCE FEATURES")
print("========================================")

nzv = quality_df[
    quality_df["Is_Near_Zero_Variance"]
]

if len(nzv) == 0:
    print("None")
else:
    print(
        nzv[
            [
                "Feature",
                "Unique_Count",
                "Zero_%",
                "Most_Common_%",
            ]
        ].to_string(index=False)
    )


print("\n========================================")
print("HIGH-ZERO FEATURES")
print("========================================")

high_zero = quality_df[
    quality_df["Is_High_Zero"]
]

if len(high_zero) == 0:
    print("None")
else:
    print(
        high_zero[
            [
                "Feature",
                "Zero_%",
                "Unique_Count",
            ]
        ].to_string(index=False)
    )


print("\n========================================")
print("HIGHLY SKEWED FEATURES")
print("========================================")

skewed = quality_df[
    quality_df["Is_Highly_Skewed"]
]

if len(skewed) == 0:
    print("None")
else:
    print(
        skewed[
            [
                "Feature",
                "Skewness",
            ]
        ]
        .sort_values(
            "Skewness",
            key=lambda x: abs(x),
            ascending=False
        )
        .to_string(index=False)
    )


# =========================================================
# 9. EXACT DUPLICATE FEATURES
# =========================================================

print("\n========================================")
print("EXACT DUPLICATE FEATURES")
print("========================================")

duplicate_pairs = []

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        feature_a = features[i]
        feature_b = features[j]

        if df[feature_a].equals(df[feature_b]):

            duplicate_pairs.append(
                (feature_a, feature_b)
            )


if duplicate_pairs:

    for feature_a, feature_b in duplicate_pairs:
        print(
            f"{feature_a} == {feature_b}"
        )

else:
    print("None")


# =========================================================
# 10. CORRELATION ANALYSIS
# =========================================================

print("\n========================================")
print("CORRELATION ANALYSIS")
print("========================================")

correlation_matrix = df[numeric_features].corr()

correlation_pairs = []

threshold = 0.95

for i in range(len(numeric_features)):

    for j in range(i + 1, len(numeric_features)):

        feature_a = numeric_features[i]
        feature_b = numeric_features[j]

        correlation = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if abs(correlation) >= threshold:

            correlation_pairs.append({
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Correlation": correlation,
                "Absolute_Correlation": abs(correlation),
            })


correlation_df = pd.DataFrame(
    correlation_pairs
)

if len(correlation_df) > 0:

    correlation_df = correlation_df.sort_values(
        "Absolute_Correlation",
        ascending=False
    )

    print(
        "\nHighly correlated pairs "
        f"(|correlation| >= {threshold}): "
        f"{len(correlation_df)}"
    )

    print(
        correlation_df.to_string(
            index=False
        )
    )

else:

    print(
        f"No pairs found with "
        f"|correlation| >= {threshold}"
    )


# =========================================================
# 11. SAVE CORRELATION REPORT
# =========================================================

correlation_file = (
    output_dir
    / "high_correlation_pairs.parquet"
)

correlation_df.to_parquet(
    correlation_file,
    index=False
)

print("\nCorrelation report saved to:")
print(correlation_file)


# =========================================================
# 12. FINAL CHECKPOINT
# =========================================================

print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print("Dataset:", input_file)
print("Rows:", len(df))
print("Model features:", len(features))
print("Target:", target)

print("\nOriginal data was NOT modified.")

print("\nSelected feature-analysis reports:")
print(output_dir)