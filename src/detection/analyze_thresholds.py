
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score

from src.detection.predictor import ThreatPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "selected_v2"
    / "validation.parquet"
)
TARGET_COLUMN = "Attack_Type"
FOCUS_CLASSES = ["BOT", "WEB_ATTACK"]
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BATCH_SIZE = 2048


def main():
    print("=" * 76)
    print("CONFIDENCE-THRESHOLD ANALYSIS")
    print("=" * 76)

    df = pd.read_parquet(VALIDATION_DATA_PATH)
    predictor = ThreatPredictor()

    actual = df[TARGET_COLUMN].astype(str).to_numpy()
    true_labels = set(predictor.class_names)

    if not set(actual).issubset(true_labels):
        raise ValueError("Test dataset contains unknown class labels.")

    # Store softmax probabilities for all records.
    all_probabilities = []

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        features = batch.drop(columns=[TARGET_COLUMN])

        # Use the predictor's exact feature ordering and validation.
        missing = [
            name for name in predictor.feature_names
            if name not in features.columns
        ]
        if missing:
            raise ValueError(f"Missing required features: {missing[:5]}")

        features = features.loc[:, predictor.feature_names]
        features = features.apply(pd.to_numeric, errors="raise")

        values = features.to_numpy(dtype="float32")
        if not np.isfinite(values).all():
            raise ValueError("Test features contain invalid values.")

        tensor = torch.from_numpy(values.copy())

        with torch.no_grad():
            logits = predictor.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        all_probabilities.append(probabilities.cpu().numpy())

        processed = min(start + len(batch), len(df))
        if processed % 20000 < BATCH_SIZE or processed == len(df):
            print(f"Processed {processed:,}/{len(df):,}")

    probabilities = np.concatenate(all_probabilities, axis=0)
    class_names = predictor.class_names

    print("\nImportant: thresholds below are analysis-only.")
    print("They do not change the detector or its saved checkpoint.")

    for target in FOCUS_CLASSES:
        target_index = class_names.index(target)

        # One-vs-rest evaluation: does this class's score exceed threshold?
        scores = probabilities[:, target_index]
        actual_binary = (actual == target)

        print(f"\n{'=' * 76}")
        print(f"TARGET CLASS: {target}")
        print(f"{'Threshold':>10} {'Precision':>12} {'Recall':>12} "
              f"{'Predicted':>12} {'TP':>8} {'FP':>8} {'FN':>8}")
        print("-" * 76)

        for threshold in THRESHOLDS:
            predicted_binary = scores >= threshold

            precision = precision_score(
                actual_binary,
                predicted_binary,
                zero_division=0,
            )
            recall = recall_score(
                actual_binary,
                predicted_binary,
                zero_division=0,
            )

            tp = int(np.sum(predicted_binary & actual_binary))
            fp = int(np.sum(predicted_binary & ~actual_binary))
            fn = int(np.sum(~predicted_binary & actual_binary))

            print(
                f"{threshold:>10.2f} "
                f"{precision:>11.4f} "
                f"{recall:>11.4f} "
                f"{int(predicted_binary.sum()):>12,} "
                f"{tp:>8,} {fp:>8,} {fn:>8,}"
            )

    print("\nThreshold analysis completed.")
    print("No retraining or checkpoint changes were performed.")


if __name__ == "__main__":
    main()
