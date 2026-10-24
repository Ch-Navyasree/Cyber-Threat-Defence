
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.detection.predictor import ThreatPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "selected_v2"
    / "test.parquet"
)

TARGET_COLUMN = "Attack_Type"


def main():
    print("=" * 70)
    print("TRANSFORMER DETECTOR - HELD-OUT TEST EVALUATION")
    print("=" * 70)

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    # Load the held-out test dataset.
    df = pd.read_parquet(TEST_DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    print(f"Test records: {len(df):,}")

    # Initialize the saved model; no training occurs.
    predictor = ThreatPredictor()

    # Predict in batches to limit memory usage.
    batch_size = 2048
    predictions = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]

        results = predictor.predict(
            batch.drop(columns=[TARGET_COLUMN])
        )

        predictions.extend(
            results["predicted_attack"].tolist()
        )

        processed = min(start + len(batch), len(df))

        if processed % 20000 < batch_size or processed == len(df):
            print(f"Processed {processed:,}/{len(df):,} records")

    actual = df[TARGET_COLUMN].astype(str).tolist()
    class_names = predictor.class_names

    # Compute metrics using the fixed seven-class label order.
    accuracy = accuracy_score(actual, predictions)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual,
            predictions,
            labels=class_names,
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predictions,
            labels=class_names,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=class_names,
    )

    print("\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy:         {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:  {macro_precision:.6f}")
    print(f"Macro Recall:     {macro_recall:.6f}")
    print(f"Macro F1-score:   {macro_f1:.6f}")

    print("\nPER-CLASS RESULTS")
    print("-" * 90)
    print(
        f"{'Class':<16}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>12}"
    )

    for i, name in enumerate(class_names):
        print(
            f"{name:<16}"
            f"{precision[i]:>12.4f}"
            f"{recall[i]:>12.4f}"
            f"{f1[i]:>12.4f}"
            f"{support[i]:>12,}"
        )

    print("\nCONFUSION MATRIX")
    print("Rows = actual classes; columns = predicted classes")
    print("Class order:", class_names)
    print(matrix)

    print("\nTest evaluation completed.")
    print("The model checkpoint was not retrained or modified.")


if __name__ == "__main__":
    main()
