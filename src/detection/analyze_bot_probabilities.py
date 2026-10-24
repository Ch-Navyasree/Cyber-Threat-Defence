
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
    y = data["Attack_Type"].astype(str).to_numpy()

    all_probabilities = []

    for start in range(0, len(X), 4096):
        batch = torch.tensor(
            X[start:start + 4096],
            dtype=torch.float32,
        )

        with torch.no_grad():
            probabilities = F.softmax(
                predictor.model(batch), dim=1
            )

        all_probabilities.append(probabilities.cpu().numpy())

    probabilities = np.concatenate(all_probabilities)
    bot_idx = predictor.class_names.index("BOT")
    benign_idx = predictor.class_names.index("BENIGN")

    actual_bot = y == "BOT"
    predicted_idx = probabilities.argmax(axis=1)
    missed_bot = actual_bot & (predicted_idx != bot_idx)

    bot_scores = probabilities[actual_bot, bot_idx]
    missed_scores = probabilities[missed_bot, bot_idx]
    missed_benign_scores = probabilities[missed_bot, benign_idx]

    print("Actual BOT records:", int(actual_bot.sum()))
    print("Missed BOT records:", int(missed_bot.sum()))

    print("\nBOT probability distribution — all actual BOT records:")
    print(pd.Series(bot_scores).describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    ).to_string())

    print("\nBOT probability distribution — missed BOT records:")
    print(pd.Series(missed_scores).describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    ).to_string())

    print("\nBENIGN probability distribution — missed BOT records:")
    print(pd.Series(missed_benign_scores).describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    ).to_string())


if __name__ == "__main__":
    main()
