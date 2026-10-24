from pathlib import Path
import polars as pl
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/cicids2017_clean.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/standard"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("LOADING CLEAN DATASET")
print("=" * 70)

df = pl.read_parquet(INPUT_FILE)

print(f"Rows: {df.height:,}")
print(f"Columns: {df.width}")


# ============================================================
# STRATIFIED SPLIT
# ============================================================

print("\nPreparing stratified split...")

# Use NumPy directly instead of converting Polars → Pandas.
labels = df["Attack_Type"].to_numpy()

indices = list(range(df.height))


# ============================================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ============================================================

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=labels
)


# ============================================================
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# ============================================================

temp_labels = labels[temp_idx]

validation_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_df = df[train_idx]
validation_df = df[validation_idx]
test_df = df[test_idx]


# ============================================================
# SAVE
# ============================================================

train_file = OUTPUT_DIR / "train.parquet"
validation_file = OUTPUT_DIR / "validation.parquet"
test_file = OUTPUT_DIR / "test.parquet"

print("\nSaving datasets...")

train_df.write_parquet(
    train_file,
    compression="zstd"
)

validation_df.write_parquet(
    validation_file,
    compression="zstd"
)

test_df.write_parquet(
    test_file,
    compression="zstd"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STANDARD SPLIT SUMMARY")
print("=" * 70)

print(
    f"Training:   {train_df.height:,} "
    f"({train_df.height / df.height * 100:.2f}%)"
)

print(
    f"Validation: {validation_df.height:,} "
    f"({validation_df.height / df.height * 100:.2f}%)"
)

print(
    f"Test:       {test_df.height:,} "
    f"({test_df.height / df.height * 100:.2f}%)"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

for name, dataset in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df)
]:

    print("\n" + "-" * 70)
    print(f"{name} CLASS DISTRIBUTION")
    print("-" * 70)

    distribution = (
        dataset
        .group_by("Attack_Type")
        .len()
        .with_columns(
            (
                pl.col("len") /
                dataset.height *
                100
            )
            .round(4)
            .alias("percentage")
        )
        .sort("len", descending=True)
    )

    print(distribution)


print("\n" + "=" * 70)
print("STANDARD SPLIT COMPLETE")
print("=" * 70)

print(f"Train:      {train_file}")
print(f"Validation: {validation_file}")
print(f"Test:       {test_file}")