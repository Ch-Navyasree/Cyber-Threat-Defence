
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "transformed" / "selected_v2"
RESULT_DIR = PROJECT_ROOT / "reports" / "baseline_comparison"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Attack_Type"


def load_split(name):
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column in {path}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if not X.columns.is_unique:
        raise ValueError(f"Duplicate feature columns in {path}")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Non-finite feature values in {path}")

    return X, y


def main():
    print("Loading dataset splits...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Train and validation features do not match.")

    if set(y_val.unique()) - set(y_train.unique()):
        raise ValueError("Validation contains unseen target classes.")

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Classes:", sorted(y_train.unique()))

    # Scale features using training data only.
    # Balanced weights help account for class imbalance.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            solver="lbfgs",
            random_state=42,
        ),
    )

    print("\nTraining Logistic Regression...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    print(f"Training time: {training_seconds:.2f} seconds")

    print("\nEvaluating on validation set...")
    start = time.perf_counter()
    predictions = model.predict(X_val)
    inference_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )

    print(f"Validation accuracy: {accuracy:.6f}")
    print(f"Validation macro-F1: {macro_f1:.6f}")
    print(f"Validation inference time: {inference_seconds:.2f} seconds")
    print(f"Average time per sample: "
          f"{inference_seconds / len(X_val) * 1000:.4f} ms")

    report = classification_report(
        y_val,
        predictions,
        labels=sorted(y_train.unique()),
        zero_division=0,
        output_dict=True,
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=sorted(y_train.unique()),
            zero_division=0,
        )
    )

    labels = sorted(y_train.unique())
    matrix = confusion_matrix(y_val, predictions, labels=labels)

    report_data = {
        "model": "Logistic Regression",
        "target": TARGET,
        "features": int(X_train.shape[1]),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "training_seconds": float(training_seconds),
        "validation_inference_seconds": float(inference_seconds),
        "average_inference_ms_per_sample": float(
            inference_seconds / len(X_val) * 1000
        ),
        "classification_report": report,
        "class_labels": labels,
        "confusion_matrix": matrix.tolist(),
    }

    output_path = RESULT_DIR / "logistic_regression_validation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
