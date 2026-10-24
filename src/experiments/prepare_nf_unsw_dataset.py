
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "NF-UNSW-NB15-v2"
    / "NF-UNSW-NB15-v2.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "nf_unsw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42

IP_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
TARGET_COLUMN = "Label"
ANALYSIS_COLUMN = "Attack"


# --------------------------------------------------
# 2. Load and validate the source dataset
# --------------------------------------------------
def main():
    for path in [INPUT_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading NF-UNSW-NB15-v2 dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Original shape: {df.shape}")

    required = set(IP_COLUMNS + [TARGET_COLUMN, ANALYSIS_COLUMN])
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Label contains missing values.")

    labels = set(df[TARGET_COLUMN].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Unexpected binary labels: {labels}")

    # Keep Attack for descriptive analysis only.
    attack_summary = (
        df.groupby(ANALYSIS_COLUMN, dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # Exclude IP strings and both label columns from model features.
    excluded = IP_COLUMNS + [TARGET_COLUMN, ANALYSIS_COLUMN]
    feature_columns = [c for c in df.columns if c not in excluded]

    # Require numeric features for this initial benchmark.
    non_numeric = [
        c for c in feature_columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise TypeError(
            f"Non-numeric candidate features found: {non_numeric}"
        )

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COLUMN].astype("int8")

    # Convert missing values to a consistent numeric representation.
    X = X.apply(pd.to_numeric, errors="coerce")

    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            "Features entirely missing; inspect before continuing: "
            f"{all_missing}"
        )

    print(f"Numeric candidate features: {len(feature_columns)}")
    print("Class distribution:")
    print(y.value_counts().sort_index().to_string())

    # --------------------------------------------------
    # 3. Split before fitting imputation/scaling
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # --------------------------------------------------
    # 4. Fit preprocessing on training data only
    # --------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_train_array = imputer.fit_transform(X_train)
    X_val_array = imputer.transform(X_val)
    X_test_array = imputer.transform(X_test)

    X_train_array = scaler.fit_transform(X_train_array)
    X_val_array = scaler.transform(X_val_array)
    X_test_array = scaler.transform(X_test_array)

    # float32 reduces disk and training memory requirements.
    X_train_array = X_train_array.astype(np.float32)
    X_val_array = X_val_array.astype(np.float32)
    X_test_array = X_test_array.astype(np.float32)

    # --------------------------------------------------
    # 5. Save processed splits
    # --------------------------------------------------
    split_data = {
        "train": (X_train_array, y_train),
        "validation": (X_val_array, y_val),
        "test": (X_test_array, y_test),
    }

    for split_name, (features, targets) in split_data.items():
        output = pd.DataFrame(features, columns=feature_columns)
        output[TARGET_COLUMN] = targets.to_numpy()

        output_path = OUTPUT_DIR / f"{split_name}.parquet"
        output.to_parquet(output_path, index=False)

        print(
            f"Saved {split_name}: shape={output.shape}, "
            f"path={output_path}"
        )

    # --------------------------------------------------
    # 6. Save preprocessing artifacts and metadata
    # --------------------------------------------------
    joblib.dump(
        {
            "feature_columns": feature_columns,
            "imputer": imputer,
            "scaler": scaler,
            "target_column": TARGET_COLUMN,
            "label_mapping": {"Benign": 0, "Attack": 1},
        },
        MODEL_DIR / "nf_unsw_binary_preprocessor.joblib",
    )

    metadata = {
        "source_file": str(INPUT_FILE),
        "original_rows": int(len(df)),
        "original_columns": int(len(df.columns)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "excluded_columns": excluded,
        "target_column": TARGET_COLUMN,
        "label_mapping": {"0": "Benign", "1": "Attack"},
        "random_state": RANDOM_STATE,
        "split_ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "preprocessing": {
            "missing_value_strategy": "training median",
            "scaling": "RobustScaler",
            "fit_on": "training split only",
            "output_dtype": "float32",
        },
        "attack_category_counts": {
            str(k): int(v) for k, v in attack_summary.items()
        },
        "split_rows": {
            name: int(len(targets))
            for name, (_, targets) in split_data.items()
        },
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\nDataset preparation completed.")
    print(f"Metadata: {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
