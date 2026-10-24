from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


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
print("Target:", target)


# =========================================================
# 3. PREPARE TARGET
# =========================================================

print("\nEncoding target labels...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df[target].astype(str)
)

print("Number of classes:", len(label_encoder.classes_))

print("\nAttack classes:")

for index, class_name in enumerate(
    label_encoder.classes_
):
    print(f"{index}: {class_name}")


# =========================================================
# 4. PREPARE FEATURES
# =========================================================

X = df[features].copy()

# Replace problematic values just in case
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

if X.isnull().any().any():

    print("\nMissing values detected after safety check.")

    X = X.fillna(
        X.median(numeric_only=True)
    )

else:

    print("\nNo missing values detected.")


# =========================================================
# 5. MUTUAL INFORMATION
# =========================================================

print("\n========================================")
print("MUTUAL INFORMATION ANALYSIS")
print("========================================")

print(
    "\nCalculating mutual information..."
)

mi_scores = mutual_info_classif(
    X,
    y,
    discrete_features=False,
    random_state=42
)


# =========================================================
# 6. CREATE RESULT TABLE
# =========================================================

results = pd.DataFrame({
    "Feature": features,
    "Mutual_Information": mi_scores,
})


# =========================================================
# 7. ADD BASIC FEATURE STATISTICS
# =========================================================

quality_file = (
    output_dir
    / "feature_quality_report.parquet"
)

quality_df = pd.read_parquet(
    quality_file
)

results = results.merge(
    quality_df[
        [
            "Feature",
            "Unique_Count",
            "Zero_%",
            "Skewness",
            "Is_Near_Zero_Variance",
            "Is_High_Zero",
        ]
    ],
    on="Feature",
    how="left"
)


# =========================================================
# 8. SORT BY MUTUAL INFORMATION
# =========================================================

results = results.sort_values(
    "Mutual_Information",
    ascending=False
)


# =========================================================
# 9. DISPLAY RESULTS
# =========================================================

print("\n========================================")
print("FEATURES RANKED BY MUTUAL INFORMATION")
print("========================================")

print(
    results[
        [
            "Feature",
            "Mutual_Information",
            "Unique_Count",
            "Zero_%",
            "Skewness",
        ]
    ].to_string(index=False)
)


# =========================================================
# 10. SPARSE FLAG ANALYSIS
# =========================================================

flag_features = [
    feature
    for feature in features
    if "Flag" in feature
]

print("\n========================================")
print("TCP FLAG FEATURE ANALYSIS")
print("========================================")

for feature in flag_features:

    print("\n----------------------------------------")
    print(feature)
    print("----------------------------------------")

    print(
        "Value counts:"
    )

    print(
        df[feature]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nAttack-type mean:"
    )

    flag_by_attack = (
        df.groupby(target)[feature]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        flag_by_attack.to_string()
    )


# =========================================================
# 11. SAVE REPORT
# =========================================================

output_file = (
    output_dir
    / "feature_target_relevance.parquet"
)

results.to_parquet(
    output_file,
    index=False
)

print("\n========================================")
print("REPORT SAVED")
print("========================================")

print(output_file)


# =========================================================
# 12. FINAL CHECKPOINT
# =========================================================

print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print("Rows:", len(df))
print("Features analyzed:", len(features))

print(
    "\nOriginal dataset was NOT modified."
)