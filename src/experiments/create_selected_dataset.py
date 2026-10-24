from pathlib import Path
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

project_root = Path(__file__).resolve().parents[2]

input_dir = (
    project_root
    / "data"
    / "processed"
    / "transformed"
)

output_dir = input_dir / "selected"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. COLUMNS TO REMOVE
# =========================================================

columns_to_remove = [
    "Source_File",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "SYN Flag Count",
    "CWE Flag Count",
]


# =========================================================
# 3. DATASETS
# =========================================================

datasets = [
    "train",
    "validation",
    "test",
]


# =========================================================
# 4. PROCESS EACH DATASET
# =========================================================

for dataset_name in datasets:

    input_file = input_dir / f"{dataset_name}.parquet"
    output_file = output_dir / f"{dataset_name}.parquet"

    print("\n" + "=" * 60)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 60)

    # Load dataset
    df = pd.read_parquet(input_file)

    print("Original shape:", df.shape)

    # -----------------------------------------------------
    # Verify target
    # -----------------------------------------------------

    if "Attack_Type" not in df.columns:
        raise ValueError(
            f"'Attack_Type' not found in {dataset_name}.parquet"
        )

    # -----------------------------------------------------
    # Verify removable columns
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in columns_to_remove
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns in {dataset_name}: "
            f"{missing_columns}"
        )

    # -----------------------------------------------------
    # Remove confirmed columns
    # -----------------------------------------------------

    selected_df = df.drop(
        columns=columns_to_remove
    )

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    if "Attack_Type" not in selected_df.columns:
        raise ValueError(
            "Target column was accidentally removed."
        )

    if selected_df.isnull().any().any():
        raise ValueError(
            f"Null values detected after processing {dataset_name}."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    selected_df.to_parquet(
        output_file,
        index=False
    )

    print("Selected shape:", selected_df.shape)
    print("Saved to:", output_file)


# =========================================================
# 5. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETE")
print("=" * 60)

print("\nRemoved columns:")

for column in columns_to_remove:
    print(" -", column)

print("\nNumber of columns removed:", len(columns_to_remove))

print("\nOriginal:")
print("70 model features + Attack_Type")

print("\nCurrent candidate:")
print("64 model features + Attack_Type")

print("\nOriginal datasets were NOT modified.")
print("Cleaned candidates were saved under:")
print(output_dir)