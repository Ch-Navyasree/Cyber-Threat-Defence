from pathlib import Path

import pandas as pd


# =========================================================
# 1. PATHS
# =========================================================

project_root = Path(__file__).resolve().parents[2]

analysis_file = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
    / "correlated_feature_analysis.parquet"
)

output_file = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
    / "feature_groups.parquet"
)


# =========================================================
# 2. LOAD CORRELATION RESULTS
# =========================================================

print("\nLoading correlation analysis...")

df = pd.read_parquet(analysis_file)

print("Correlated pairs:", len(df))


# =========================================================
# 3. BUILD CONNECTED FEATURE GROUPS
# =========================================================

print("\nBuilding correlated feature groups...")

features = set(df["Feature_A"]) | set(df["Feature_B"])

parent = {
    feature: feature
    for feature in features
}


def find(feature):
    while parent[feature] != feature:
        parent[feature] = parent[parent[feature]]
        feature = parent[feature]

    return feature


def union(feature_a, feature_b):

    root_a = find(feature_a)
    root_b = find(feature_b)

    if root_a != root_b:
        parent[root_b] = root_a


for _, row in df.iterrows():

    union(
        row["Feature_A"],
        row["Feature_B"]
    )


# =========================================================
# 4. CREATE GROUPS
# =========================================================

groups = {}

for feature in features:

    root = find(feature)

    groups.setdefault(
        root,
        []
    ).append(feature)


# Sort groups by size
groups = sorted(
    groups.values(),
    key=len,
    reverse=True
)


# =========================================================
# 5. LOAD MUTUAL INFORMATION
# =========================================================

mi_file = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
    / "feature_target_relevance.parquet"
)

mi_df = pd.read_parquet(mi_file)


# =========================================================
# 6. DISPLAY GROUPS
# =========================================================

print("\n========================================")
print("CORRELATED FEATURE GROUPS")
print("========================================")

group_records = []

for group_number, group in enumerate(
    groups,
    start=1
):

    group_mi = mi_df[
        mi_df["Feature"].isin(group)
    ].copy()

    group_mi = group_mi.sort_values(
        "Mutual_Information",
        ascending=False
    )

    representative = group_mi.iloc[0]["Feature"]

    print("\n----------------------------------------")
    print(f"GROUP {group_number}")
    print("----------------------------------------")

    print("Number of features:", len(group))

    print("\nFeatures:")

    for _, row in group_mi.iterrows():

        print(
            f"  {row['Feature']}"
            f" | MI={row['Mutual_Information']:.6f}"
            f" | Zero={row['Zero_%']:.2f}%"
            f" | Unique={int(row['Unique_Count'])}"
        )

    print(
        "\nHighest-MI candidate:",
        representative
    )

    # Store group information
    for _, row in group_mi.iterrows():

        group_records.append({
            "Group_ID": group_number,
            "Feature": row["Feature"],
            "Mutual_Information": row[
                "Mutual_Information"
            ],
            "Unique_Count": row[
                "Unique_Count"
            ],
            "Zero_%": row[
                "Zero_%"
            ],
            "Skewness": row[
                "Skewness"
            ],
            "Group_Representative": representative,
        })


# =========================================================
# 7. SAVE GROUP REPORT
# =========================================================

group_df = pd.DataFrame(
    group_records
)

group_df.to_parquet(
    output_file,
    index=False
)


# =========================================================
# 8. SUMMARY
# =========================================================

print("\n========================================")
print("GROUPING COMPLETE")
print("========================================")

print(
    "Number of correlated groups:",
    len(groups)
)

print(
    "Features involved in groups:",
    len(features)
)

print(
    "\nReport saved to:"
)

print(output_file)

print(
    "\nNo dataset was modified."
)