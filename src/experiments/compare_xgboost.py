
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

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

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    unknown_classes = set(y_val.unique()) - set(encoder.classes_)
    if unknown_classes:
        raise ValueError(f"Unknown validation classes: {unknown_classes}")

    y_val_encoded = encoder.transform(y_val)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Classes:", list(encoder.classes_))

    # Balanced sample weights reduce the dominance of majority classes.
    class_counts = y_train.value_counts()
    sample_weights = y_train.map(
        lambda label: len(y_train) / (len(class_counts) * class_counts[label])
    ).to_numpy()

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
    )

    print("\nTraining XGBoost...")
    start = time.perf_counter()
    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    training_seconds = time.perf_counter() - start

    print(f"Training time: {training_seconds:.2f} seconds")

    print("\nEvaluating on validation set...")
    start = time.perf_counter()
    predictions_encoded = model.predict(X_val)
    inference_seconds = time.perf_counter() - start

    predictions = encoder.inverse_transform(
        predictions_encoded.astype(int)
    )

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )

    print(f"Validation accuracy: {accuracy:.6f}")
    print(f"Validation macro-F1: {macro_f1:.6f}")
    print(f"Validation inference time: {inference_seconds:.2f} seconds")
    print(
        "Average inference time per sample: "
        f"{inference_seconds / len(X_val) * 1000:.4f} ms"
    )

    report = classification_report(
        y_val,
        predictions,
        labels=list(encoder.classes_),
        zero_division=0,
        output_dict=True,
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=list(encoder.classes_),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_val, predictions, labels=list(encoder.classes_)
    )

    output = {
        "model": "XGBoost",
        "target": TARGET,
        "features": int(X_train.shape[1]),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "classes": list(encoder.classes_),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "training_seconds": float(training_seconds),
        "validation_inference_seconds": float(inference_seconds),
        "average_inference_ms_per_sample": float(
            inference_seconds / len(X_val) * 1000
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    output_path = RESULT_DIR / "xgboost_validation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
