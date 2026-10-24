import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]

FINAL_DIR = ROOT / "data" / "processed" / "final"
TRANSFORMED_DIR = ROOT / "data" / "processed" / "transformed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

METADATA_PATH = TRANSFORMED_DIR / "transformation_metadata.json"

SCALER_PATH = MODEL_DIR / "cicids2017_robust_scaler_with_features.joblib"
REPORT_PATH = REPORT_DIR / "scaler_reconstruction_verification.json"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]

    if len(feature_columns) != 69:
        raise ValueError(
            f"Expected 69 original features, found {len(feature_columns)}."
        )

    train_path = FINAL_DIR / "train.parquet"

    train_df = pl.read_parquet(train_path)
    missing = set(feature_columns) - set(train_df.columns)

    if missing:
        raise ValueError(f"Missing training features: {sorted(missing)}")

    X_train = train_df.select(feature_columns).to_numpy()

    if not np.isfinite(X_train).all():
        raise ValueError("Training data contains NaN or infinite values.")

    # Fit only on the original training split.
    scaler = RobustScaler()
    scaler.fit(X_train)

    verification = {}

    for split in ("train", "validation", "test"):
        original_path = FINAL_DIR / f"{split}.parquet"
        transformed_path = TRANSFORMED_DIR / f"{split}.parquet"

        original_df = pl.read_parquet(original_path)
        transformed_df = pl.read_parquet(transformed_path)

        X_original = original_df.select(feature_columns).to_numpy()
        X_expected = transformed_df.select(feature_columns).to_numpy()

        X_actual = scaler.transform(X_original)

        if X_actual.shape != X_expected.shape:
            raise ValueError(
                f"{split}: shape mismatch "
                f"{X_actual.shape} != {X_expected.shape}"
            )

        max_difference = float(
            np.max(np.abs(X_actual - X_expected))
        )

        matches = bool(
            np.allclose(
                X_actual,
                X_expected,
                rtol=1e-7,
                atol=1e-9,
            )
        )

        verification[split] = {
            "rows": int(X_actual.shape[0]),
            "features": int(X_actual.shape[1]),
            "max_absolute_difference": max_difference,
            "matches_existing_transformation": matches,
        }

        print(
            f"{split}: rows={X_actual.shape[0]:,}, "
            f"max_difference={max_difference:.12g}, "
            f"matches={matches}"
        )

        if not matches:
            raise ValueError(
                f"{split} verification failed. Scaler was not saved."
            )

    # Save the scaler together with its exact feature order.
    artifact = {
        "scaler": scaler,
        "feature_columns": feature_columns,
        "scaler_type": "sklearn.preprocessing.RobustScaler",
        "with_centering": scaler.with_centering,
        "with_scaling": scaler.with_scaling,
        "source_training_file": str(train_path),
        "verification": verification,
    }

    joblib.dump(artifact, SCALER_PATH)

    report = {
        "status": "PASS",
        "scaler_path": str(SCALER_PATH),
        "feature_count": len(feature_columns),
        "fit_split": "train",
        "verification": verification,
        "note": (
            "Reconstructed from the current final training split. "
            "Existing transformed splits were reproduced within tolerance. "
            "This does not prove the original historical scaler artifact "
            "was identical."
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nSUCCESS: scaler saved and all split checks passed.")
    print(f"Scaler: {SCALER_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()