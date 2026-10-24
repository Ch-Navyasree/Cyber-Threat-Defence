from pathlib import Path

import pandas as pd
import numpy as np


# =========================================================
# 1. PATHS
# =========================================================

project_root = Path(__file__).resolve().parents[2]

group_file = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
    / "feature_groups.parquet"
)

mi_file = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
    / "feature_target_relevance.parquet"
)

output_file = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "selected"
    / "feature_quality"
    / "group_redundancy_analysis.parquet"
)


# =========================================================
# 2. LOAD REPORTS
# =========================================================

print("\nLoading feature groups...")

groups_df = pd.read_parquet(group_file)

print(
    "Features in correlated groups:",
    len(groups_df)
)

print("\nLoading mutual-information results...")

mi_df = pd.read_parquet(mi_file)


# =========================================================
# 3. BUILD GROUP SUMMARY
# =========================================================

group_summary = []

for group_id, group in groups_df.groupby(
    "Group_ID"
):

    group = group.sort_values(
        "Mutual_Information",
        ascending=False
    ).copy()

    features = group["Feature"].tolist()

    representative = features[0]

    representative_mi = group.iloc[0][
        "Mutual_Information"
    ]

    lowest_mi = group.iloc[-1][
        "Mutual_Information"
    ]

    mi_range = (
        representative_mi
        - lowest_mi
    )

    group_summary.append({
        "Group_ID": group_id,
        "Feature_Count": len(features),
        "Features": " | ".join(features),
        "Representative": representative,
        "Representative_MI": representative_mi,
        "Lowest_MI": lowest_mi,
        "MI_Range": mi_range,
    })


summary_df = pd.DataFrame(
    group_summary
)


# =========================================================
# 4. DISPLAY
# =========================================================

print("\n========================================")
print("GROUP REDUNDANCY SUMMARY")
print("========================================")

for _, row in summary_df.iterrows():

    print("\n----------------------------------------")

    print(
        f"GROUP {int(row['Group_ID'])}"
    )

    print("----------------------------------------")

    print(
        "Feature count:",
        int(row["Feature_Count"])
    )

    print(
        "\nFeatures:"
    )

    for feature in row["Features"].split(" | "):

        feature_info = mi_df[
            mi_df["Feature"] == feature
        ]

        if len(feature_info) == 0:
            continue

        feature_info = feature_info.iloc[0]

        print(
            f"  {feature}"
            f" | MI={feature_info['Mutual_Information']:.6f}"
            f" | Zero={feature_info['Zero_%']:.2f}%"
            f" | Unique={int(feature_info['Unique_Count'])}"
        )

    print(
        "\nHighest-MI representative:",
        row["Representative"]
    )

    print(
        "Representative MI:",
        f"{row['Representative_MI']:.6f}"
    )

    print(
        "MI range:",
        f"{row['MI_Range']:.6f}"
    )


# =========================================================
# 5. SAVE
# =========================================================

summary_df.to_parquet(
    output_file,
    index=False
)

print("\n========================================")
print("REDUNDANCY ANALYSIS COMPLETE")
print("========================================")

print(
    "Groups analyzed:",
    len(summary_df)
)

print(
    "\nReport saved to:"
)

print(output_file)

print(
    "\nNo dataset was modified."
)