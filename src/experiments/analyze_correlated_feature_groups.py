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

quality_dir = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
)

mi_file = quality_dir / "feature_target_relevance.parquet"

output_file = (
    quality_dir
    / "correlated_feature_analysis.parquet"
)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("\nLoading selected training data...")

df = pd.read_parquet(input_file)

target = "Attack_Type"

features = [
    column
    for column in df.columns
    if column != target
]

print("Dataset shape:", df.shape)
print("Features:", len(features))


# =========================================================
# 3. LOAD MUTUAL INFORMATION RESULTS
# =========================================================

print("\nLoading mutual-information results...")

mi_df = pd.read_parquet(mi_file)

mi_df = mi_df[
    [
        "Feature",
        "Mutual_Information",
        "Unique_Count",
        "Zero_%",
        "Skewness",
        "Is_Near_Zero_Variance",
        "Is_High_Zero",
    ]
].copy()


# =========================================================
# 4. CORRELATION MATRIX
# =========================================================

print("\nCalculating feature correlations...")

X = df[features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

correlation_matrix = X.corr()


# =========================================================
# 5. EXTRACT HIGHLY CORRELATED PAIRS
# =========================================================

threshold = 0.95

pairs = []

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        feature_a = features[i]
        feature_b = features[j]

        correlation = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if abs(correlation) >= threshold:

            pairs.append({
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Correlation": correlation,
                "Absolute_Correlation": abs(correlation),
            })


correlation_df = pd.DataFrame(pairs)

correlation_df = correlation_df.sort_values(
    "Absolute_Correlation",
    ascending=False
).reset_index(drop=True)


# =========================================================
# 6. ADD FEATURE INFORMATION
# =========================================================

correlation_df = correlation_df.merge(
    mi_df[
        [
            "Feature",
            "Mutual_Information",
            "Unique_Count",
            "Zero_%",
            "Skewness",
        ]
    ].rename(
        columns={
            "Feature": "Feature_A",
            "Mutual_Information": "MI_A",
            "Unique_Count": "Unique_A",
            "Zero_%": "Zero_A_%",
            "Skewness": "Skewness_A",
        }
    ),
    on="Feature_A",
    how="left",
)

correlation_df = correlation_df.merge(
    mi_df[
        [
            "Feature",
            "Mutual_Information",
            "Unique_Count",
            "Zero_%",
            "Skewness",
        ]
    ].rename(
        columns={
            "Feature": "Feature_B",
            "Mutual_Information": "MI_B",
            "Unique_Count": "Unique_B",
            "Zero_%": "Zero_B_%",
            "Skewness": "Skewness_B",
        }
    ),
    on="Feature_B",
    how="left",
)


# =========================================================
# 7. COMPARE MUTUAL INFORMATION
# =========================================================

correlation_df["MI_Difference"] = (
    correlation_df["MI_A"]
    - correlation_df["MI_B"]
).abs()

correlation_df["Higher_MI_Feature"] = np.where(
    correlation_df["MI_A"] >= correlation_df["MI_B"],
    correlation_df["Feature_A"],
    correlation_df["Feature_B"],
)

correlation_df["Lower_MI_Feature"] = np.where(
    correlation_df["MI_A"] < correlation_df["MI_B"],
    correlation_df["Feature_A"],
    correlation_df["Feature_B"],
)


# =========================================================
# 8. COMPARE DATA QUALITY
# =========================================================

correlation_df["Lower_Zero_Feature"] = np.where(
    correlation_df["Zero_A_%"] <= correlation_df["Zero_B_%"],
    correlation_df["Feature_A"],
    correlation_df["Feature_B"],
)

correlation_df["Higher_Unique_Feature"] = np.where(
    correlation_df["Unique_A"] >= correlation_df["Unique_B"],
    correlation_df["Feature_A"],
    correlation_df["Feature_B"],
)


# =========================================================
# 9. DISPLAY RESULTS
# =========================================================

print("\n========================================")
print("HIGHLY CORRELATED FEATURE PAIRS")
print("========================================")

print(
    correlation_df[
        [
            "Feature_A",
            "Feature_B",
            "Correlation",
            "MI_A",
            "MI_B",
            "Higher_MI_Feature",
            "Lower_MI_Feature",
            "Zero_A_%",
            "Zero_B_%",
        ]
    ].to_string(index=False)
)


# =========================================================
# 10. SAVE REPORT
# =========================================================

correlation_df.to_parquet(
    output_file,
    index=False
)

print("\n========================================")
print("REPORT SAVED")
print("========================================")

print(output_file)

print("\nNumber of highly correlated pairs:", len(correlation_df))

print("\nOriginal dataset was NOT modified.")
print("Selected dataset was NOT modified.")