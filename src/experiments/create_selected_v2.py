import os
import pandas as pd


# ============================================================
# INPUT DATA
# ============================================================

INPUT_DIR = "data/processed/transformed/selected"


# ============================================================
# OUTPUT DATA
# ============================================================

OUTPUT_DIR = "data/processed/transformed/selected_v2"
QUALITY_DIR = os.path.join(
    OUTPUT_DIR,
    "feature_quality"
)


# ============================================================
# FEATURES TO REMOVE
# ============================================================

REMOVED_FEATURES = [
    "Subflow Bwd Bytes",
    "Bwd Packet Length Mean",
    "Avg Fwd Segment Size",
    "Fwd URG Flags",
    "RST Flag Count",
    "ECE Flag Count",
]


# ============================================================
# SPLITS
# ============================================================

SPLITS = [
    "train",
    "validation",
    "test",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CREATING SELECTED_V2 DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        QUALITY_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process each split
    # --------------------------------------------------------

    for split in SPLITS:

        input_path = os.path.join(
            INPUT_DIR,
            f"{split}.parquet"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{split}.parquet"
        )

        print()
        print("-" * 70)
        print(f"Processing: {split}")
        print("-" * 70)

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        print(
            f"Loading: {input_path}"
        )

        df = pd.read_parquet(
            input_path
        )

        print(
            f"Original shape: {df.shape}"
        )

        # ----------------------------------------------------
        # Validate removal columns
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in REMOVED_FEATURES
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing expected features in {split}: "
                f"{missing_features}"
            )

        # ----------------------------------------------------
        # Remove selected features
        # ----------------------------------------------------

        df_v2 = df.drop(
            columns=REMOVED_FEATURES
        )

        print(
            f"New shape: {df_v2.shape}"
        )

        print()
        print("Removed features:")

        for feature in REMOVED_FEATURES:
            print(
                f"  - {feature}"
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        df_v2.to_parquet(
            output_path,
            index=False
        )

        print()
        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Create feature-selection report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING FEATURE SELECTION REPORT")
    print("=" * 70)

    # Load train only to obtain final feature list
    train_v2 = pd.read_parquet(
        os.path.join(
            OUTPUT_DIR,
            "train.parquet"
        )
    )

    all_columns = list(
        train_v2.columns
    )

    feature_columns = [
        column
        for column in all_columns
        if column != "Attack_Type"
    ]

    removed_df = pd.DataFrame({
        "Removed_Feature": REMOVED_FEATURES
    })

    retained_df = pd.DataFrame({
        "Retained_Feature": feature_columns
    })

    removed_path = os.path.join(
        QUALITY_DIR,
        "removed_features.parquet"
    )

    retained_path = os.path.join(
        QUALITY_DIR,
        "retained_features.parquet"
    )

    removed_df.to_parquet(
        removed_path,
        index=False
    )

    retained_df.to_parquet(
        retained_path,
        index=False
    )

    # --------------------------------------------------------
    # Save feature-selection summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "Original_Model_Features": 64,
            "Removed_Features": len(
                REMOVED_FEATURES
            ),
            "Final_Model_Features": len(
                feature_columns
            ),
            "Target_Column": "Attack_Type",
            "Final_Total_Columns": len(
                all_columns
            ),
        }
    ])

    summary_path = os.path.join(
        QUALITY_DIR,
        "feature_selection_summary.parquet"
    )

    summary.to_parquet(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTED_V2 CREATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Original model features: 64"
    )

    print(
        f"Removed features: "
        f"{len(REMOVED_FEATURES)}"
    )

    print(
        f"Final model features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Final total columns: "
        f"{len(all_columns)}"
    )

    print()
    print("Output directory:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )

    print()
    print("Feature-selection reports saved.")

    print()
    print("IMPORTANT:")
    print(
        "Original datasets were NOT modified."
    )

    print(
        "The previous selected dataset was NOT modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()