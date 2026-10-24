
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.detection.predictor import ThreatPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "selected_v2"
    / "validation.parquet"
)

TARGET_COLUMN = "Attack_Type"
BATCH_SIZE = 2048

# Initial candidate thresholds selected from validation analysis.
THRESHOLDS = {
    "BOT": 0.80,
    "WEB_ATTACK": 0.80,
}


def get_probabilities(df, predictor):
    """Calculate class probabilities without modifying the model."""
    feature_names = predictor.feature_names
    all_probabilities = []

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        missing = [
            name for name in feature_names
            if name not in batch.columns
        ]

        if missing:
            raise ValueError(f"Missing features: {missing[:5]}")

        features = batch.loc[:, feature_names].apply(
            pd.to_numeric,
            errors="raise",
        )

        values = features.to_numpy(dtype=np.float32, copy=True)

        if not np.isfinite(values).all():
            raise ValueError("Invalid numerical values found.")

        tensor = torch.from_numpy(values)

        with torch.no_grad():
            logits = predictor.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        all_probabilities.append(probabilities.cpu().numpy())

        processed = min(start + len(batch), len(df))
        if processed % 20000 < BATCH_SIZE or processed == len(df):
            print(f"Processed {processed:,}/{len(df):,}")

    return np.concatenate(all_probabilities, axis=0)




def predict_with_thresholds(probabilities, class_names):
    """
    Reject low-confidence BOT and WEB_ATTACK predictions.

    If either target class is predicted below its threshold,
    fall back to the highest-probability non-target class.
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

        # Do not reassign a rejected prediction to the other
        # threshold-controlled class.
        alternative_scores[:, list(target_indices)] = -1.0

        predictions[rows] = alternative_scores.argmax(axis=1)

    return np.asarray(class_names)[predictions]


def evaluate_split(df, predictor, split_name):
    print(f"\n{'=' * 72}")
    print(f"{split_name.upper()} SET EVALUATION")
    print("=" * 72)

    actual = df[TARGET_COLUMN].astype(str).to_numpy()
    class_names = predictor.class_names

    if not set(actual).issubset(set(class_names)):
        raise ValueError("Unknown target labels found.")

    probabilities = get_probabilities(df, predictor)

    # Original seven-class argmax baseline.
    baseline_predictions = np.asarray(class_names)[
        probabilities.argmax(axis=1)
    ]

    # Candidate threshold-policy predictions.
    policy_predictions = predict_with_thresholds(
        probabilities,
        class_names,
    )
    changed_predictions = np.sum(baseline_predictions != policy_predictions)

    print(f"\nPredictions changed by policy: {changed_predictions:,}")

    baseline_accuracy = accuracy_score(
        actual,
        baseline_predictions,
    )
    baseline_macro_f1 = f1_score(
        actual,
        baseline_predictions,
        labels=class_names,
        average="macro",
        zero_division=0,
    )

    policy_accuracy = accuracy_score(
        actual,
        policy_predictions,
    )
    policy_macro_f1 = f1_score(
        actual,
        policy_predictions,
        labels=class_names,
        average="macro",
        zero_division=0,
    )

    print("\nBaseline metrics:")
    print(f"Accuracy: {baseline_accuracy:.6f}")
    print(f"Macro F1: {baseline_macro_f1:.6f}")

    print("\nThreshold-policy metrics:")
    print(f"Accuracy: {policy_accuracy:.6f}")
    print(f"Macro F1: {policy_macro_f1:.6f}")

    print("\nThreshold-policy classification report:")
    print(
        classification_report(
            actual,
            policy_predictions,
            labels=class_names,
            zero_division=0,
            digits=4,
        )
    )

    print("Threshold-policy confusion matrix:")
    print(
        confusion_matrix(
            actual,
            policy_predictions,
            labels=class_names,
        )
    )

    print("\nClass order:")
    print(class_names)


def main():
    print("=" * 72)
    print("THRESHOLD-POLICY EXPERIMENT")
    print("=" * 72)
    print("Model checkpoint will not be changed.")
    print("Candidate thresholds:", THRESHOLDS)

    predictor = ThreatPredictor()

    validation_df = pd.read_parquet(VALIDATION_PATH)
    print(f"\nValidation records: {len(validation_df):,}")

    evaluate_split(
        validation_df,
        predictor,
        "validation",
    )

    print("\nValidation evaluation completed.")
    print(
        "Review the validation results before deciding whether "
        "to run the final test evaluation."
    )


if __name__ == "__main__":
    main()
