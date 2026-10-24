
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from src.detection.predictor import ThreatPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "selected_v2"
    / "validation.parquet"
)


def main():
    print("Loading validation data...")
    data = pd.read_parquet(DATA_PATH)

    predictor = ThreatPredictor()

    X = data[predictor.feature_names].to_numpy(dtype="float32")
    y_true = data["Attack_Type"].astype(str).to_numpy()

    print(f"Validation records: {len(data):,}")
    print("Running model predictions...")

    all_probabilities = []

    # Process batches to limit memory use.
    batch_size = 4096

    for start in range(0, len(X), batch_size):
        batch = torch.tensor(
            X[start:start + batch_size],
            dtype=torch.float32,
        )

        with torch.no_grad():
            logits = predictor.model(batch)
            probabilities = F.softmax(logits, dim=1)

        all_probabilities.append(probabilities.cpu().numpy())

    probabilities = __import__("numpy").concatenate(
        all_probabilities,
        axis=0,
    )

    predicted_indices = predictor.apply_threshold_policy(
        probabilities
    )
    y_pred = [
        predictor.class_names[int(index)]
        for index in predicted_indices
    ]

    print("\nOverall validation classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=predictor.class_names,
            zero_division=0,
            digits=4,
        )
    )

    print("\nConfusion matrix")
    print("Rows = actual classes; columns = predicted classes")
    print(predictor.class_names)
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=predictor.class_names,
        )
    )

    bot_mask = y_true == "BOT"

    print(f"\nActual BOT records: {bot_mask.sum():,}")
    print("Predictions for actual BOT records:")

    bot_predictions = pd.Series(y_pred)[bot_mask]
    print(bot_predictions.value_counts().to_string())

    bot_probabilities = probabilities[bot_mask]
    bot_index = predictor.class_names.index("BOT")

    print(
        "\nMean raw BOT probability for actual BOT records:",
        round(float(bot_probabilities[:, bot_index].mean()), 6),
    )


if __name__ == "__main__":
    main()
