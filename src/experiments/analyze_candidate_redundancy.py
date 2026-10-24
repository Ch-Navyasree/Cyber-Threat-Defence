import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = (
    "data/processed/transformed/selected/train.parquet"
)

OUTPUT_DIR = (
    "data/processed/transformed/selected/feature_quality"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_redundancy_analysis.parquet"
)


# ============================================================
# CANDIDATE PAIRS
# ============================================================

CANDIDATE_PAIRS = [
    (
        "Total Length of Bwd Packets",
        "Subflow Bwd Bytes"
    ),
    (
        "Avg Bwd Segment Size",
        "Bwd Packet Length Mean"
    ),
    (
        "Fwd Packet Length Mean",
        "Avg Fwd Segment Size"
    ),
    (
        "Average Packet Size",
        "Packet Length Mean"
    ),
    (
        "Fwd IAT Total",
        "Flow Duration"
    ),
    (
        "Fwd Packets/s",
        "Flow Packets/s"
    ),
    (
        "Fwd Packet Length Max",
        "Fwd Packet Length Std"
    ),
    (
        "Packet Length Std",
        "Max Packet Length"
    ),
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def analyze_pair(df, feature_a, feature_b):
    """
    Analyze whether two highly correlated features
    are actually redundant.
    """

    a = df[feature_a].astype(float)
    b = df[feature_b].astype(float)

    # --------------------------------------------------------
    # Exact equality
    # --------------------------------------------------------

    exact_equal = bool(a.equals(b))

    # --------------------------------------------------------
    # Difference analysis
    # --------------------------------------------------------

    difference = (a - b).abs()

    max_abs_difference = float(difference.max())
    mean_abs_difference = float(difference.mean())

    nonzero_difference_count = int(
        (difference != 0).sum()
    )

    nonzero_difference_percentage = (
        nonzero_difference_count / len(df)
    ) * 100

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation = float(
        a.corr(b)
    )

    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    unique_a = int(a.nunique())
    unique_b = int(b.nunique())

    # --------------------------------------------------------
    # Zero percentages
    # --------------------------------------------------------

    zero_a = float(
        (a == 0).mean() * 100
    )

    zero_b = float(
        (b == 0).mean() * 100
    )

    # --------------------------------------------------------
    # Scale information
    # --------------------------------------------------------

    mean_a = float(a.mean())
    mean_b = float(b.mean())

    std_a = float(a.std())
    std_b = float(b.std())

    min_a = float(a.min())
    min_b = float(b.min())

    max_a = float(a.max())
    max_b = float(b.max())

    # --------------------------------------------------------
    # Ratio analysis
    # --------------------------------------------------------
    # Useful for detecting whether one feature is approximately
    # a scaled version of the other.

    valid_ratio = b != 0

    if valid_ratio.any():
        ratio = a[valid_ratio] / b[valid_ratio]

        ratio_median = float(
            ratio.median()
        )

        ratio_mean = float(
            ratio.mean()
        )

    else:
        ratio_median = np.nan
        ratio_mean = np.nan

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "Feature_A": feature_a,
        "Feature_B": feature_b,

        "Correlation": correlation,

        "Exact_Equal": exact_equal,

        "Max_Absolute_Difference": max_abs_difference,
        "Mean_Absolute_Difference": mean_abs_difference,

        "Nonzero_Difference_Count":
            nonzero_difference_count,

        "Nonzero_Difference_Percentage":
            nonzero_difference_percentage,

        "Unique_A": unique_a,
        "Unique_B": unique_b,

        "Zero_Percentage_A": zero_a,
        "Zero_Percentage_B": zero_b,

        "Mean_A": mean_a,
        "Mean_B": mean_b,

        "Std_A": std_a,
        "Std_B": std_b,

        "Min_A": min_a,
        "Min_B": min_b,

        "Max_A": max_a,
        "Max_B": max_b,

        "Median_A_to_B_Ratio":
            ratio_median,

        "Mean_A_to_B_Ratio":
            ratio_mean,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("CANDIDATE FEATURE REDUNDANCY ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print("Loading selected training dataset...")

    df = pd.read_parquet(TRAIN_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Check candidate features
    # --------------------------------------------------------

    available_features = set(df.columns)

    print()
    print("Checking candidate feature pairs...")

    valid_pairs = []

    for feature_a, feature_b in CANDIDATE_PAIRS:

        if feature_a not in available_features:
            print(f"WARNING: Missing feature: {feature_a}")
            continue

        if feature_b not in available_features:
            print(f"WARNING: Missing feature: {feature_b}")
            continue

        valid_pairs.append(
            (feature_a, feature_b)
        )

    print(
        f"Valid candidate pairs: {len(valid_pairs)}"
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 60)
    print("PAIR ANALYSIS")
    print("=" * 60)

    for index, (feature_a, feature_b) in enumerate(
        valid_pairs,
        start=1
    ):

        print()
        print("-" * 60)
        print(f"PAIR {index}")
        print("-" * 60)

        print(f"A: {feature_a}")
        print(f"B: {feature_b}")

        result = analyze_pair(
            df,
            feature_a,
            feature_b
        )

        results.append(result)

        print(
            f"Correlation: "
            f"{result['Correlation']:.6f}"
        )

        print(
            f"Exact equality: "
            f"{result['Exact_Equal']}"
        )

        print(
            f"Max absolute difference: "
            f"{result['Max_Absolute_Difference']:.6f}"
        )

        print(
            f"Mean absolute difference: "
            f"{result['Mean_Absolute_Difference']:.6f}"
        )

        print(
            f"Nonzero differences: "
            f"{result['Nonzero_Difference_Count']}"
        )

        print(
            f"Nonzero difference %: "
            f"{result['Nonzero_Difference_Percentage']:.4f}%"
        )

        print(
            f"Unique values: "
            f"{result['Unique_A']} vs "
            f"{result['Unique_B']}"
        )

        print(
            f"Zero %: "
            f"{result['Zero_Percentage_A']:.2f}% vs "
            f"{result['Zero_Percentage_B']:.2f}%"
        )

        print(
            f"Median A/B ratio: "
            f"{result['Median_A_to_B_Ratio']:.6f}"
        )

    # --------------------------------------------------------
    # Save report
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
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Pairs analyzed: {len(results_df)}"
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