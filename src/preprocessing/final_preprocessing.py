from pathlib import Path
import json
import polars as pl


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("data/processed/standard")
OUTPUT_DIR = Path("data/processed/final")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = BASE_DIR / "train.parquet"
VAL_FILE = BASE_DIR / "validation.parquet"
TEST_FILE = BASE_DIR / "test.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Attack_Type"

RATE_COLUMNS = [
    "Flow Bytes/s",
    "Flow Packets/s",
]

NEGATIVE_AS_MISSING = [
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Flow IAT Min",
    "Fwd IAT Min",
]

# Sentinel -1 values in these fields are intentionally preserved.
SENTINEL_COLUMNS = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL LEAKAGE-SAFE PREPROCESSING")
print("=" * 70)

print("\nLoading datasets...")

train = pl.read_parquet(TRAIN_FILE)
validation = pl.read_parquet(VAL_FILE)
test = pl.read_parquet(TEST_FILE)

print(f"Train      : {train.shape}")
print(f"Validation : {validation.shape}")
print(f"Test       : {test.shape}")


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

feature_columns = [
    c for c in train.columns
    if c != LABEL_COLUMN
]

print(f"\nNumber of features: {len(feature_columns)}")


# ============================================================
# STEP 1 — CONVERT STRING NUMERIC FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: STRING → NUMERIC")
print("=" * 70)

for column in RATE_COLUMNS:

    if train[column].dtype == pl.String:
        print(f"Converting: {column}")

        train = train.with_columns(
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .alias(column)
        )

        validation = validation.with_columns(
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .alias(column)
        )

        test = test.with_columns(
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .alias(column)
        )


# ============================================================
# STEP 2 — INFINITY → NULL
# ============================================================

# ============================================================
# STEP 2 — HANDLING NaN AND INFINITY
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: HANDLING NaN AND INFINITY")
print("=" * 70)

for column in RATE_COLUMNS:

    print(f"\n{column}")

    train_nan = train[column].is_nan().sum()
    val_nan = validation[column].is_nan().sum()
    test_nan = test[column].is_nan().sum()

    train_inf = train[column].is_infinite().sum()
    val_inf = validation[column].is_infinite().sum()
    test_inf = test[column].is_infinite().sum()

    print(f"Train NaN           : {train_nan}")
    print(f"Validation NaN      : {val_nan}")
    print(f"Test NaN            : {test_nan}")

    print(f"Train infinity      : {train_inf}")
    print(f"Validation infinity : {val_inf}")
    print(f"Test infinity       : {test_inf}")

    # NaN → NULL
    train = train.with_columns(
        pl.col(column)
        .fill_nan(None)
        .alias(column)
    )

    validation = validation.with_columns(
        pl.col(column)
        .fill_nan(None)
        .alias(column)
    )

    test = test.with_columns(
        pl.col(column)
        .fill_nan(None)
        .alias(column)
    )

    # Infinity → NULL
    train = train.with_columns(
        pl.when(pl.col(column).is_infinite())
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    )

    validation = validation.with_columns(
        pl.when(pl.col(column).is_infinite())
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    )

    test = test.with_columns(
        pl.when(pl.col(column).is_infinite())
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    )


# ============================================================
# STEP 3 — INVALID NEGATIVE VALUES → NULL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: INVALID NEGATIVE VALUES")
print("=" * 70)

for column in NEGATIVE_AS_MISSING:

    train_negative = (train[column] < 0).sum()
    val_negative = (validation[column] < 0).sum()
    test_negative = (test[column] < 0).sum()

    print(f"\n{column}")
    print(f"Train negative      : {train_negative}")
    print(f"Validation negative : {val_negative}")
    print(f"Test negative      : {test_negative}")

    train = train.with_columns(
        pl.when(pl.col(column) < 0)
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    )

    validation = validation.with_columns(
        pl.when(pl.col(column) < 0)
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    )

    test = test.with_columns(
        pl.when(pl.col(column) < 0)
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    )


# ============================================================
# STEP 4 — TRAINING-ONLY MEDIAN IMPUTATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: TRAINING-ONLY MEDIAN IMPUTATION")
print("=" * 70)

medians = {}

imputation_columns = RATE_COLUMNS + NEGATIVE_AS_MISSING

for column in imputation_columns:

    median_value = train[column].median()

    medians[column] = median_value

    print(f"{column}: {median_value}")

    train = train.with_columns(
        pl.col(column)
        .fill_null(median_value)
        .alias(column)
    )

    validation = validation.with_columns(
        pl.col(column)
        .fill_null(median_value)
        .alias(column)
    )

    test = test.with_columns(
        pl.col(column)
        .fill_null(median_value)
        .alias(column)
    )


# ============================================================
# STEP 5 — FINAL NUMERIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: FINAL DATA QUALITY CHECK")
print("=" * 70)


def count_nulls(df):
    return sum(
        df[column].null_count()
        for column in feature_columns
    )


def count_infinity(df):
    total = 0

    for column in feature_columns:

        if df[column].dtype in [
            pl.Float32,
            pl.Float64,
        ]:
            total += df[column].is_infinite().sum()

    return total


print("\nNull values:")
print(f"Train      : {count_nulls(train)}")
print(f"Validation : {count_nulls(validation)}")
print(f"Test       : {count_nulls(test)}")

print("\nInfinity values:")
print(f"Train      : {count_infinity(train)}")
print(f"Validation : {count_infinity(validation)}")
print(f"Test       : {count_infinity(test)}")


# ============================================================
# STEP 6 — VERIFY SENTINEL VALUES WERE PRESERVED
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: SENTINEL VALUE CHECK")
print("=" * 70)

for column in SENTINEL_COLUMNS:

    train_minus_one = (train[column] == -1).sum()
    val_minus_one = (validation[column] == -1).sum()
    test_minus_one = (test[column] == -1).sum()

    print(f"\n{column}")
    print(f"Train -1      : {train_minus_one}")
    print(f"Validation -1 : {val_minus_one}")
    print(f"Test -1       : {test_minus_one}")


# ============================================================
# STEP 7 — SAVE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: SAVING FINAL DATASETS")
print("=" * 70)

train_output = OUTPUT_DIR / "train.parquet"
validation_output = OUTPUT_DIR / "validation.parquet"
test_output = OUTPUT_DIR / "test.parquet"

train.write_parquet(train_output)
validation.write_parquet(validation_output)
test.write_parquet(test_output)


# ============================================================
# STEP 8 — SAVE PREPROCESSING METADATA
# ============================================================

metadata = {
    "label_column": LABEL_COLUMN,
    "feature_count": len(feature_columns),
    "features": feature_columns,
    "rate_columns": RATE_COLUMNS,
    "negative_values_treated_as_missing": NEGATIVE_AS_MISSING,
    "sentinel_values_preserved": SENTINEL_COLUMNS,
    "imputation_strategy": "training_median",
    "training_medians": medians,
}

metadata_file = OUTPUT_DIR / "preprocessing_metadata.json"

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING COMPLETE")
print("=" * 70)

print(f"\nTrain      : {train_output}")
print(f"Validation : {validation_output}")
print(f"Test       : {test_output}")
print(f"Metadata   : {metadata_file}")

print("\nFinal shapes:")
print(f"Train      : {train.shape}")
print(f"Validation : {validation.shape}")
print(f"Test       : {test.shape}")

print("\nClass distributions:")

for name, dataset in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    print(f"\n{name}")
    print(
        dataset
        .group_by(LABEL_COLUMN)
        .len()
        .sort(LABEL_COLUMN)
    )

print("\n" + "=" * 70)
print("READY FOR MODEL DEVELOPMENT")
print("=" * 70)