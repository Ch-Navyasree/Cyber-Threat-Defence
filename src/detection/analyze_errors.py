
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

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
FOCUS_CLASSES = ["BOT", "WEB_ATTACK"]
BATCH_SIZE = 2048


def main():
    print("=" * 72)
    print("MINORITY-CLASS ERROR ANALYSIS")
    print("=" * 72)

    df = pd.read_parquet(TEST_DATA_PATH)
    predictor = ThreatPredictor()

    actual = df[TARGET_COLUMN].astype(str).tolist()
    predicted = []

    # Predict in batches to keep memory usage manageable.
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        result = predictor.predict(
            batch.drop(columns=[TARGET_COLUMN])
        )
        predicted.extend(result["predicted_attack"].tolist())

        processed = min(start + len(batch), len(df))
        if processed % 20000 < BATCH_SIZE or processed == len(df):
            print(f"Processed {processed:,}/{len(df):,}")

    classes = predictor.class_names
    matrix = confusion_matrix(actual, predicted, labels=classes)

    print("\n1. CONFUSION MATRIX")
    print("Rows = actual; columns = predicted")
    print("Class order:", classes)
    print(matrix)

    print("\n2. FOCUS-CLASS ERROR COUNTS")

    for target in FOCUS_CLASSES:
        tp = sum(
            a == target and p == target
            for a, p in zip(actual, predicted)
        )
        fp = sum(
            a != target and p == target
            for a, p in zip(actual, predicted)
        )
        fn = sum(
            a == target and p != target
            for a, p in zip(actual, predicted)
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        print(f"\n{target}")
        print(f"  True positives:  {tp:,}")
        print(f"  False positives: {fp:,}")
        print(f"  False negatives: {fn:,}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  Recall:          {recall:.4f}")

    # Identify which actual classes generate false positives
    # for BOT and WEB_ATTACK.
    print("\n3. SOURCES OF FALSE POSITIVES")

    for target in FOCUS_CLASSES:
        errors = pd.Series(
            [
                a for a, p in zip(actual, predicted)
                if p == target and a != target
            ],
            name="actual_class",
        )

        print(f"\nPredicted as {target}, but actually:")
        if errors.empty:
            print("  No false positives.")
        else:
            print(errors.value_counts().to_string())

    # Save record-level errors for later inspection.
    error_mask = [
        a != p for a, p in zip(actual, predicted)
    ]

    error_df = df.loc[error_mask].copy()
    error_df["Predicted_Attack"] = [
        p for a, p in zip(actual, predicted) if a != p
    ]

    output_path = (
        PROJECT_ROOT / "data" / "processed"
        / "transformed" / "minority_class_errors.csv"
    )
    error_df.to_csv(output_path, index=False)

    print(f"\nMisclassified records saved to: {output_path}")
    print(f"Total misclassified records: {len(error_df):,}")
    print("Analysis complete. The model was not modified.")


if __name__ == "__main__":
    main()
