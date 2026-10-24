
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
    data = pd.read_parquet(DATA_PATH)
    predictor = ThreatPredictor()

    X = data[predictor.feature_names].to_numpy(dtype=np.float32)
    y_true = data["Attack_Type"].astype(str).to_numpy()

    batches = []

    for start in range(0, len(X), 4096):
        batch = torch.tensor(
            X[start:start + 4096],
            dtype=torch.float32,
        )

        with torch.no_grad():
            probs = F.softmax(predictor.model(batch), dim=1)

        batches.append(probs.cpu().numpy())

    probabilities = np.concatenate(batches, axis=0)

    bot_idx = predictor.class_names.index("BOT")
    benign_idx = predictor.class_names.index("BENIGN")

    raw_indices = probabilities.argmax(axis=1)
    raw_predictions = np.array([
        predictor.class_names[i] for i in raw_indices
    ])

    policy_indices = predictor.apply_threshold_policy(probabilities)
    policy_predictions = np.array([
        predictor.class_names[int(i)] for i in policy_indices
    ])

    actual_bot = y_true == "BOT"

    print("Actual BOT records:", int(actual_bot.sum()))
    print(
        "Raw model predicts BOT:",
        int(np.sum(actual_bot & (raw_predictions == "BOT"))),
    )
    print(
        "Raw model predicts BENIGN:",
        int(np.sum(actual_bot & (raw_predictions == "BENIGN"))),
    )
    print(
        "Raw model predicts other classes:",
        int(np.sum(actual_bot & (raw_predictions != "BOT")
                   & (raw_predictions != "BENIGN"))),
    )
    print(
        "Threshold policy predicts BOT:",
        int(np.sum(actual_bot & (policy_predictions == "BOT"))),
    )
    print(
        "Threshold policy predicts BENIGN:",
        int(np.sum(actual_bot & (policy_predictions == "BENIGN"))),
    )
    print(
        "Threshold policy predicts other classes:",
        int(np.sum(actual_bot & (policy_predictions != "BOT")
                   & (policy_predictions != "BENIGN"))),
    )

    print(
        "\nRaw BOT probability mean:",
        round(float(probabilities[actual_bot, bot_idx].mean()), 6),
    )
    print(
        "Raw BENIGN probability mean:",
        round(float(probabilities[actual_bot, benign_idx].mean()), 6),
    )


if __name__ == "__main__":
    main()
