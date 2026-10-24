from pathlib import Path
import polars as pl


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("data/processed/final")

TRAIN_FILE = BASE_DIR / "train.parquet"
VALIDATION_FILE = BASE_DIR / "validation.parquet"
TEST_FILE = BASE_DIR / "test.parquet"


# ============================================================
# SPECIAL COLUMNS
# ============================================================

LABEL_COLUMN = "Attack_Type"
METADATA_COLUMN = "Source_File"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL FEATURE VERIFICATION")
print("=" * 70)

print("\nLoading datasets...")

train = pl.read_parquet(TRAIN_FILE)
validation = pl.read_parquet(VALIDATION_FILE)
test = pl.read_parquet(TEST_FILE)

print(f"Train      : {train.shape}")
print(f"Validation : {validation.shape}")
print(f"Test       : {test.shape}")


# ============================================================
# IDENTIFY MODEL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: IDENTIFYING MODEL FEATURES")
print("=" * 70)

excluded_columns = {
    LABEL_COLUMN,
    METADATA_COLUMN,
}

model_features = [
    column
    for column in train.columns
    if column not in excluded_columns
]

print(f"\nLabel column excluded   : {LABEL_COLUMN}")
print(f"Metadata excluded       : {METADATA_COLUMN}")
print(f"Model features         : {len(model_features)}")


# ============================================================
# DISPLAY FEATURES
# ============================================================

print("\nModel feature list:")

for index, feature in enumerate(model_features, start=1):
    print(f"{index:02d}. {feature}")


# ============================================================
# STEP 2: CHECK SPECIAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: SPECIAL COLUMN CHECK")
print("=" * 70)

print(f"\n{LABEL_COLUMN} present   : {LABEL_COLUMN in train.columns}")
print(f"{METADATA_COLUMN} present : {METADATA_COLUMN in train.columns}")

if LABEL_COLUMN in model_features:
    print("ERROR: Label column is included in model features!")

if METADATA_COLUMN in model_features:
    print("ERROR: Source_File is included in model features!")

if (
    LABEL_COLUMN not in model_features
    and METADATA_COLUMN not in model_features
):
    print("\nPASS: Label and metadata are excluded from model input.")


# ============================================================
# STEP 3: NUMERIC TYPE CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: NUMERIC FEATURE CHECK")
print("=" * 70)

numeric_types = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}

non_numeric = []

for feature in model_features:

    dtype = train[feature].dtype

    if dtype not in numeric_types:
        non_numeric.append((feature, dtype))

if len(non_numeric) == 0:

    print("\nPASS: All model features are numeric.")

else:

    print("\nERROR: Non-numeric model features found:")

    for feature, dtype in non_numeric:
        print(f"  - {feature}: {dtype}")


# ============================================================
# STEP 4: FEATURE CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: FEATURE CONSISTENCY")
print("=" * 70)

train_features = [
    column for column in train.columns
    if column not in excluded_columns
]

validation_features = [
    column for column in validation.columns
    if column not in excluded_columns
]

test_features = [
    column for column in test.columns
    if column not in excluded_columns
]


print(f"\nTrain features      : {len(train_features)}")
print(f"Validation features : {len(validation_features)}")
print(f"Test features       : {len(test_features)}")


if (
    train_features == validation_features
    and train_features == test_features
):

    print("\nPASS: Feature columns are identical across all splits.")

else:

    print("\nERROR: Feature mismatch detected.")

    train_only = set(train_features) - set(validation_features)
    validation_only = set(validation_features) - set(train_features)

    test_only = set(test_features) - set(train_features)

    if train_only:
        print("\nTrain-only features:")
        for feature in sorted(train_only):
            print(f"  - {feature}")

    if validation_only:
        print("\nValidation-only features:")
        for feature in sorted(validation_only):
            print(f"  - {feature}")

    if test_only:
        print("\nTest-only features:")
        for feature in sorted(test_only):
            print(f"  - {feature}")


# ============================================================
# STEP 5: NULL CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: NULL VALUE CHECK")
print("=" * 70)


def count_feature_nulls(df):

    total = 0

    for feature in model_features:
        total += df[feature].null_count()

    return total


train_nulls = count_feature_nulls(train)
validation_nulls = count_feature_nulls(validation)
test_nulls = count_feature_nulls(test)

print(f"\nTrain null values      : {train_nulls}")
print(f"Validation null values : {validation_nulls}")
print(f"Test null values       : {test_nulls}")


if train_nulls == 0 and validation_nulls == 0 and test_nulls == 0:

    print("\nPASS: No null values in model features.")

else:

    print("\nWARNING: Null values detected.")


# ============================================================
# STEP 6: INFINITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: INFINITY CHECK")
print("=" * 70)


def count_infinity(df):

    total = 0

    for feature in model_features:

        if df[feature].dtype in {pl.Float32, pl.Float64}:

            total += df[feature].is_infinite().sum()

    return total


train_inf = count_infinity(train)
validation_inf = count_infinity(validation)
test_inf = count_infinity(test)

print(f"\nTrain infinity      : {train_inf}")
print(f"Validation infinity : {validation_inf}")
print(f"Test infinity       : {test_inf}")


if train_inf == 0 and validation_inf == 0 and test_inf == 0:

    print("\nPASS: No infinity values in model features.")

else:

    print("\nWARNING: Infinity values detected.")


# ============================================================
# STEP 7: TARGET CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: TARGET CHECK")
print("=" * 70)

print("\nTraining classes:")

print(
    train
    .group_by(LABEL_COLUMN)
    .len()
    .sort(LABEL_COLUMN)
)

print("\nValidation classes:")

print(
    validation
    .group_by(LABEL_COLUMN)
    .len()
    .sort(LABEL_COLUMN)
)

print("\nTest classes:")

print(
    test
    .group_by(LABEL_COLUMN)
    .len()
    .sort(LABEL_COLUMN)
)


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

checks = {
    "Exactly 69 model features": len(model_features) == 69,
    "Label excluded": LABEL_COLUMN not in model_features,
    "Source_File excluded": METADATA_COLUMN not in model_features,
    "All features numeric": len(non_numeric) == 0,
    "Feature consistency": (
        train_features == validation_features
        and train_features == test_features
    ),
    "No train nulls": train_nulls == 0,
    "No validation nulls": validation_nulls == 0,
    "No test nulls": test_nulls == 0,
    "No train infinity": train_inf == 0,
    "No validation infinity": validation_inf == 0,
    "No test infinity": test_inf == 0,
}

all_passed = True

for check, result in checks.items():

    status = "PASS" if result else "FAIL"

    print(f"{status:6} | {check}")

    if not result:
        all_passed = False


print("\n" + "=" * 70)

if all_passed:

    print("ALL MODEL FEATURE CHECKS PASSED")
    print("The dataset is ready for feature transformation.")

else:

    print("SOME CHECKS FAILED")
    print("Do NOT start model training yet.")

print("=" * 70)