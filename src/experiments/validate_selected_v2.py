import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed/transformed/selected_v2"

SPLITS = [
    "train",
    "validation",
    "test",
]

TARGET = "Attack_Type"

REMOVED_FEATURES = [
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
    print("=" * 70)
    print("SELECTED_V2 DATASET VALIDATION")
    print("=" * 70)

    datasets = {}

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    for split in SPLITS:

        path = os.path.join(
            DATA_DIR,
            f"{split}.parquet"
        )

        print()
        print(f"Loading {split}...")

        df = pd.read_parquet(path)

        datasets[split] = df

        print(
            f"{split} shape: {df.shape}"
        )

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STRUCTURE CHECK")
    print("=" * 70)

    train_columns = list(
        datasets["train"].columns
    )

    print(
        f"Total columns: {len(train_columns)}"
    )

    print(
        f"Model features: "
        f"{len(train_columns) - 1}"
    )

    print(
        f"Target column: {TARGET}"
    )

    # --------------------------------------------------------
    # Target check
    # --------------------------------------------------------

    print()
    print("Target column check:")

    for split, df in datasets.items():

        exists = TARGET in df.columns

        print(
            f"  {split}: "
            f"{'PASS' if exists else 'FAIL'}"
        )

    # --------------------------------------------------------
    # Column consistency
    # --------------------------------------------------------

    print()
    print("Column consistency:")

    for split in ["validation", "test"]:

        same_columns = (
            list(datasets[split].columns)
            == train_columns
        )

        print(
            f"  train vs {split}: "
            f"{'PASS' if same_columns else 'FAIL'}"
        )

    # --------------------------------------------------------
    # Removed feature check
    # --------------------------------------------------------

    print()
    print("Removed feature check:")

    for feature in REMOVED_FEATURES:

        present_in = [
            split
            for split, df in datasets.items()
            if feature in df.columns
        ]

        if present_in:
            print(
                f"  FAIL: {feature} "
                f"still present in {present_in}"
            )
        else:
            print(
                f"  PASS: {feature}"
            )

    # --------------------------------------------------------
    # Missing values / NaN / infinity
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    for split, df in datasets.items():

        numeric_df = df.drop(
            columns=[TARGET]
        )

        null_count = int(
            numeric_df.isnull().sum().sum()
        )

        nan_count = int(
            np.isnan(
                numeric_df.to_numpy()
            ).sum()
        )

        inf_count = int(
            np.isinf(
                numeric_df.to_numpy()
            ).sum()
        )

        duplicate_columns = (
            numeric_df.columns.duplicated().sum()
        )

        print()
        print(split)

        print(
            f"  Null values: {null_count}"
        )

        print(
            f"  NaN values: {nan_count}"
        )

        print(
            f"  Infinite values: {inf_count}"
        )

        print(
            f"  Duplicate columns: "
            f"{duplicate_columns}"
        )

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    print()
    print("Constant feature check:")

    train_features = datasets["train"].drop(
        columns=[TARGET]
    )

    constant_features = [
        column
        for column in train_features.columns
        if train_features[column].nunique() <= 1
    ]

    if constant_features:

        print("  Constant features found:")

        for feature in constant_features:
            print(
                f"    - {feature}"
            )

    else:

        print(
            "  PASS: No constant features"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for split, df in datasets.items():

        print()
        print(split)

        counts = (
            df[TARGET]
            .value_counts()
            .sort_index()
        )

        percentages = (
            df[TARGET]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )

        for cls in counts.index:

            print(
                f"  {cls:<15} "
                f"Count={counts[cls]:>8} "
                f"Percentage={percentages[cls]:>8.4f}%"
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "If all checks show PASS and there are no "
        "unexpected data-quality issues, selected_v2 "
        "is ready for the next pipeline stage."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()