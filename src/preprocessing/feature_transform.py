from pathlib import Path
import json

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/processed/final")
OUTPUT_DIR = Path("data/processed/transformed")

TRAIN_FILE = INPUT_DIR / "train.parquet"
VALIDATION_FILE = INPUT_DIR / "validation.parquet"
TEST_FILE = INPUT_DIR / "test.parquet"

TRAIN_OUTPUT = OUTPUT_DIR / "train.parquet"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation.parquet"
TEST_OUTPUT = OUTPUT_DIR / "test.parquet"

METADATA_OUTPUT = OUTPUT_DIR / "transformation_metadata.json"


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Attack_Type"
METADATA_COLUMN = "Source_File"

RANDOM_STATE = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LEAKAGE-SAFE FEATURE TRANSFORMATION")
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

excluded_columns = {
    LABEL_COLUMN,
    METADATA_COLUMN,
}

feature_columns = [
    column
    for column in train.columns
    if column not in excluded_columns
]

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print(f"\nNumber of model features: {len(feature_columns)}")

if len(feature_columns) != 69:
    raise ValueError(
        f"Expected 69 model features, found {len(feature_columns)}"
    )

print("\nExcluded columns:")
print(f"Label    : {LABEL_COLUMN}")
print(f"Metadata : {METADATA_COLUMN}")


# ============================================================
# VERIFY FEATURE CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CONSISTENCY CHECK")
print("=" * 70)

validation_features = [
    column
    for column in validation.columns
    if column not in excluded_columns
]

test_features = [
    column
    for column in test.columns
    if column not in excluded_columns
]

if feature_columns != validation_features:
    raise ValueError(
        "Validation feature columns do not match training features."
    )

if feature_columns != test_features:
    raise ValueError(
        "Test feature columns do not match training features."
    )

print("PASS: Train, validation and test feature columns are identical.")


# ============================================================
# CONVERT TO NUMPY
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING FEATURES")
print("=" * 70)

X_train = train.select(feature_columns).to_numpy()
X_validation = validation.select(feature_columns).to_numpy()
X_test = test.select(feature_columns).to_numpy()

print(f"\nX_train shape      : {X_train.shape}")
print(f"X_validation shape : {X_validation.shape}")
print(f"X_test shape       : {X_test.shape}")


# ============================================================
# NUMERIC TYPE CHECK
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC TYPE CHECK")
print("=" * 70)

if not np.issubdtype(X_train.dtype, np.number):
    raise TypeError(
        f"Training feature array is not numeric. "
        f"Dtype: {X_train.dtype}"
    )

if not np.issubdtype(X_validation.dtype, np.number):
    raise TypeError(
        f"Validation feature array is not numeric. "
        f"Dtype: {X_validation.dtype}"
    )

if not np.issubdtype(X_test.dtype, np.number):
    raise TypeError(
        f"Test feature array is not numeric. "
        f"Dtype: {X_test.dtype}"
    )

print(f"Train dtype      : {X_train.dtype}")
print(f"Validation dtype : {X_validation.dtype}")
print(f"Test dtype       : {X_test.dtype}")

print("PASS: All model feature arrays are numeric.")


# ============================================================
# FINAL NUMERIC SAFETY CHECK
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC SAFETY CHECK")
print("=" * 70)


def check_array(name, array):
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())

    print(f"\n{name}")
    print(f"NaN      : {nan_count}")
    print(f"Infinity : {inf_count}")

    if nan_count > 0 or inf_count > 0:

        print("\nINVALID VALUES DETECTED")

        # Find affected feature columns
        invalid_columns = []

        for index, feature in enumerate(feature_columns):

            column = array[:, index]

            column_nan = int(np.isnan(column).sum())
            column_inf = int(np.isinf(column).sum())

            if column_nan > 0 or column_inf > 0:
                invalid_columns.append(
                    (
                        feature,
                        column_nan,
                        column_inf
                    )
                )

        print("\nAffected features:")

        for feature, nan_count_col, inf_count_col in invalid_columns:
            print(
                f"{feature:<35} "
                f"NaN={nan_count_col:<8} "
                f"Infinity={inf_count_col}"
            )

        raise ValueError(
            f"{name} contains NaN or infinity values. "
            f"Fix final preprocessing before transformation."
        )

    print(f"PASS: {name} contains no NaN or infinity.")


check_array(
    "Train",
    X_train
)

check_array(
    "Validation",
    X_validation
)

check_array(
    "Test",
    X_test
)


# ============================================================
# ROBUST SCALING
# ============================================================

print("\n" + "=" * 70)
print("ROBUST SCALING")
print("=" * 70)

print("\nFitting RobustScaler on TRAINING DATA ONLY...")

scaler = RobustScaler(
    with_centering=True,
    with_scaling=True,
)

X_train_scaled = scaler.fit_transform(X_train)

print("Training scaler fitted.")

print("\nApplying training transformation to validation data...")

X_validation_scaled = scaler.transform(
    X_validation
)

print("Applying training transformation to test data...")

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# TRANSFORMED DATA SAFETY CHECK
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMED DATA SAFETY CHECK")
print("=" * 70)


def check_transformed_array(name, array):

    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())

    print(f"\n{name}")
    print(f"NaN      : {nan_count}")
    print(f"Infinity : {inf_count}")

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"{name} contains invalid values after RobustScaler."
        )

    print("PASS: No NaN or infinity after transformation.")


check_transformed_array(
    "Train",
    X_train_scaled
)

check_transformed_array(
    "Validation",
    X_validation_scaled
)

check_transformed_array(
    "Test",
    X_test_scaled
)


# ============================================================
# TRANSFORMED STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMED DATA CHECK")
print("=" * 70)

for name, array in [
    ("Train", X_train_scaled),
    ("Validation", X_validation_scaled),
    ("Test", X_test_scaled),
]:

    print(f"\n{name}")

    print(f"Minimum : {np.min(array):.6f}")
    print(f"Maximum : {np.max(array):.6f}")
    print(f"Mean    : {np.mean(array):.6f}")
    print(f"Std     : {np.std(array):.6f}")

    print(f"NaN     : {np.isnan(array).sum()}")
    print(f"Inf     : {np.isinf(array).sum()}")


# ============================================================
# REBUILD DATAFRAMES
# ============================================================

print("\n" + "=" * 70)
print("REBUILDING DATASETS")
print("=" * 70)

train_features_df = pl.DataFrame(
    X_train_scaled,
    schema=feature_columns,
)

validation_features_df = pl.DataFrame(
    X_validation_scaled,
    schema=feature_columns,
)

test_features_df = pl.DataFrame(
    X_test_scaled,
    schema=feature_columns,
)


# ============================================================
# RESTORE LABEL + METADATA
# ============================================================

train_transformed = train_features_df.with_columns(
    train[LABEL_COLUMN],
    train[METADATA_COLUMN],
)

validation_transformed = validation_features_df.with_columns(
    validation[LABEL_COLUMN],
    validation[METADATA_COLUMN],
)

test_transformed = test_features_df.with_columns(
    test[LABEL_COLUMN],
    test[METADATA_COLUMN],
)


# ============================================================
# REORDER COLUMNS
# ============================================================

final_columns = (
    feature_columns
    + [
        LABEL_COLUMN,
        METADATA_COLUMN,
    ]
)

train_transformed = train_transformed.select(
    final_columns
)

validation_transformed = validation_transformed.select(
    final_columns
)

test_transformed = test_transformed.select(
    final_columns
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("SAVING TRANSFORMED DATASETS")
print("=" * 70)

train_transformed.write_parquet(
    TRAIN_OUTPUT
)

validation_transformed.write_parquet(
    VALIDATION_OUTPUT
)

test_transformed.write_parquet(
    TEST_OUTPUT
)

print(f"\nTrain      : {TRAIN_OUTPUT}")
print(f"Validation : {VALIDATION_OUTPUT}")
print(f"Test       : {TEST_OUTPUT}")


# ============================================================
# SAVE TRANSFORMATION METADATA
# ============================================================

metadata = {
    "transformation": "RobustScaler",

    "scaler_parameters": {
        "with_centering": True,
        "with_scaling": True,
    },

    "fit_on": "training_data_only",

    "random_state": RANDOM_STATE,

    "number_of_features": len(feature_columns),

    "feature_columns": feature_columns,

    "excluded_columns": [
        LABEL_COLUMN,
        METADATA_COLUMN,
    ],

    "train_shape": list(
        train_transformed.shape
    ),

    "validation_shape": list(
        validation_transformed.shape
    ),

    "test_shape": list(
        test_transformed.shape
    ),
}

with open(
    METADATA_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print(f"Metadata   : {METADATA_OUTPUT}")


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRANSFORMATION CHECK")
print("=" * 70)

print(f"\nTrain      : {train_transformed.shape}")
print(f"Validation : {validation_transformed.shape}")
print(f"Test       : {test_transformed.shape}")

print("\nFeature count:", len(feature_columns))

print("\nPASS   | Exactly 69 model features")
print("PASS   | Label excluded from scaling")
print("PASS   | Source_File excluded from scaling")
print("PASS   | RobustScaler fitted on training data only")
print("PASS   | Validation transformed without refitting")
print("PASS   | Test transformed without refitting")
print("PASS   | No NaN after transformation")
print("PASS   | No infinity after transformation")

print("\nTransformation complete.")
print("=" * 70)