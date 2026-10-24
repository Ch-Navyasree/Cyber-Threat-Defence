import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = (
    "data/processed/transformed/selected/train.parquet"
)

MI_PATH = (
    "data/processed/transformed/selected/"
    "feature_quality/feature_target_relevance.parquet"
)

OUTPUT_DIR = (
    "data/processed/transformed/selected/feature_quality"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "removal_candidate_analysis.parquet"
)


# ============================================================
# FEATURES TO INVESTIGATE
# ============================================================

REMOVAL_CANDIDATES = [
    "Subflow Bwd Bytes",
    "Bwd Packet Length Mean",
    "Avg Fwd Segment Size",
    "Fwd URG Flags",
    "RST Flag Count",
    "ECE Flag Count",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("REMOVAL CANDIDATE ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load selected training dataset
    # --------------------------------------------------------

    print()
    print("Loading selected training dataset...")

    df = pd.read_parquet(TRAIN_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Load MI results
    # --------------------------------------------------------

    print()
    print("Loading mutual-information results...")

    mi_df = pd.read_parquet(MI_PATH)

    # --------------------------------------------------------
    # Detect MI column names
    # --------------------------------------------------------

    print()
    print("MI report columns:")
    print(list(mi_df.columns))

    # Expected structure is:
    # Feature + Mutual_Information
    #
    # The code below also handles common alternative names.

    feature_col = None
    mi_col = None

    for col in mi_df.columns:
        if col.lower() in [
            "feature",
            "feature_name",
            "features"
        ]:
            feature_col = col

        if col.lower() in [
            "mutual_information",
            "mutual information",
            "mi",
            "mutual_info"
        ]:
            mi_col = col

    if feature_col is None:
        raise ValueError(
            "Could not identify feature column in MI report."
        )

    if mi_col is None:
        raise ValueError(
            "Could not identify mutual-information column "
            "in MI report."
        )

    # --------------------------------------------------------
    # Validate candidates
    # --------------------------------------------------------

    available_features = set(df.columns)

    valid_candidates = []

    for feature in REMOVAL_CANDIDATES:

        if feature in available_features:
            valid_candidates.append(feature)
        else:
            print(
                f"WARNING: Feature not found: {feature}"
            )

    print()
    print(
        f"Valid removal candidates: "
        f"{len(valid_candidates)}"
    )

    # --------------------------------------------------------
    # Target classes
    # --------------------------------------------------------

    target = "Attack_Type"

    if target not in df.columns:
        raise ValueError(
            "Attack_Type column not found."
        )

    classes = sorted(
        df[target].dropna().unique()
    )

    print()
    print("Attack classes:")
    for cls in classes:
        print(f"  {cls}")

    # --------------------------------------------------------
    # Analyze each candidate
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 60)
    print("CANDIDATE ANALYSIS")
    print("=" * 60)

    for feature in valid_candidates:

        print()
        print("-" * 60)
        print(feature)
        print("-" * 60)

        series = df[feature]

        # ----------------------------------------------------
        # Overall statistics
        # ----------------------------------------------------

        zero_percentage = (
            (series == 0).mean() * 100
        )

        unique_count = series.nunique()

        # ----------------------------------------------------
        # MI
        # ----------------------------------------------------

        mi_match = mi_df[
            mi_df[feature_col] == feature
        ]

        if len(mi_match) > 0:
            mi_value = float(
                mi_match.iloc[0][mi_col]
            )
        else:
            mi_value = float("nan")

        print(
            f"Mutual Information: {mi_value:.6f}"
        )

        print(
            f"Overall zero %: "
            f"{zero_percentage:.6f}%"
        )

        print(
            f"Overall unique values: "
            f"{unique_count}"
        )

        # ----------------------------------------------------
        # Class-wise analysis
        # ----------------------------------------------------

        for cls in classes:

            class_data = df[
                df[target] == cls
            ][feature]

            class_zero_percentage = (
                (class_data == 0).mean() * 100
            )

            class_nonzero_percentage = (
                (class_data != 0).mean() * 100
            )

            class_unique = class_data.nunique()

            class_mean = class_data.mean()

            results.append({
                "Feature": feature,
                "Attack_Type": cls,
                "Mutual_Information": mi_value,
                "Overall_Zero_Percentage":
                    zero_percentage,
                "Overall_Unique_Count":
                    unique_count,
                "Class_Mean":
                    float(class_mean),
                "Class_Zero_Percentage":
                    float(class_zero_percentage),
                "Class_Nonzero_Percentage":
                    float(class_nonzero_percentage),
                "Class_Unique_Count":
                    int(class_unique),
                "Class_Sample_Count":
                    int(len(class_data)),
            })

            print(
                f"  {cls:<15} "
                f"Mean={class_mean:.6f} | "
                f"Zero={class_zero_percentage:.4f}% | "
                f"Nonzero={class_nonzero_percentage:.4f}% | "
                f"Unique={class_unique}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("REMOVAL ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Candidates analyzed: "
        f"{len(valid_candidates)}"
    )

    print()
    print("Report saved to:")
    print(
        os.path.abspath(OUTPUT_PATH)
    )

    print()
    print("No dataset was modified.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()