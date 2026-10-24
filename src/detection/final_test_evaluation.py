
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.detection.predictor import ThreatPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "selected_v2"
    / "test.parquet"
)

# These thresholds were selected using validation data.
# Keep them fixed during final test evaluation.
THRESHOLDS = {
    "BOT": 0.80,
    "WEB_ATTACK": 0.80,
}

BATCH_SIZE = 2048


def get_probabilities(predictor, data):
    """Generate class probabilities in batches."""

    feature_names = predictor.feature_names
    probabilities = []

    for start in range(0, len(data), BATCH_SIZE):
        batch = data.iloc[
            start:start + BATCH_SIZE
        ].loc[:, feature_names]

        batch = batch.apply(pd.to_numeric, errors="raise")
        values = batch.to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                "Test features contain NaN or infinite values."
            )

        features = torch.tensor(
            values,
            dtype=torch.float32,
            device="cpu",
        )

        with torch.no_grad():
            logits = predictor.model(features)
            batch_probabilities = (
                F.softmax(logits, dim=1)
                .cpu()
                .numpy()
            )

        probabilities.append(batch_probabilities)

        processed = min(start + BATCH_SIZE, len(data))
        print(f"Processed {processed:,}/{len(data):,}")

    return np.concatenate(probabilities, axis=0)


def predict_with_thresholds(probabilities, class_names):
    """
    Reject low-confidence BOT and WEB_ATTACK predictions.

    When a target prediction is below its threshold,
    fall back to the strongest non-target class.
    """

    predictions = probabilities.argmax(axis=1)

    target_indices = {
        class_names.index(target)
        for target in THRESHOLDS
    }

    for target, threshold in THRESHOLDS.items():
        target_idx = class_names.index(target)

        low_confidence = (
            (predictions == target_idx)
            & (probabilities[:, target_idx] < threshold)
        )

        rows = np.where(low_confidence)[0]

        if len(rows) == 0:
            continue

        alternative_scores = probabilities[rows].copy()

        # Avoid transferring rejected predictions to
        # another threshold-controlled class.
        alternative_scores[:, list(target_indices)] = -1.0

        predictions[rows] = alternative_scores.argmax(axis=1)

    return np.asarray(class_names)[predictions]


def report_results(title, y_true, predictions, class_names):
    """Print evaluation metrics and confusion matrix."""

    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)

    print(f"Accuracy: {accuracy_score(y_true, predictions):.6f}")

    print(
        "Macro F1: "
        f"{f1_score(y_true, predictions, average='macro', zero_division=0):.6f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            predictions,
            labels=class_names,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=class_names,
        )
    )


def main():
    print("=" * 72)
    print("FINAL HELD-OUT TEST EVALUATION")
    print("=" * 72)

    print("Model checkpoint will not be changed.")
    print(f"Fixed thresholds: {THRESHOLDS}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    test_data = pd.read_parquet(TEST_PATH)

    if "Attack_Type" not in test_data.columns:
        raise ValueError(
            "Test dataset does not contain Attack_Type."
        )

    predictor = ThreatPredictor()

    feature_names = predictor.feature_names
    class_names = predictor.class_names

    missing_features = [
        feature
        for feature in feature_names
        if feature not in test_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features[:10]}"
        )

    y_true = test_data["Attack_Type"].astype(str).to_numpy()

    unknown_labels = sorted(set(y_true) - set(class_names))

    if unknown_labels:
        raise ValueError(
            f"Unknown test labels: {unknown_labels}"
        )

    print(f"\nTest records: {len(test_data):,}")
    print("Generating class probabilities...")

    probabilities = get_probabilities(
        predictor,
        test_data,
    )

    baseline_predictions = np.asarray(class_names)[
        probabilities.argmax(axis=1)
    ]

    policy_predictions = predict_with_thresholds(
        probabilities,
        class_names,
    )

    changed = int(
        np.sum(baseline_predictions != policy_predictions)
    )

    print(f"\nPredictions changed by policy: {changed:,}")

    report_results(
        "BASELINE TRANSFORMER",
        y_true,
        baseline_predictions,
        class_names,
    )

    report_results(
        "TRANSFORMER + THRESHOLD POLICY",
        y_true,
        policy_predictions,
        class_names,
    )

    print("\n" + "=" * 72)
    print("FINAL TEST EVALUATION COMPLETED")
    print("No model weights or thresholds were changed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
