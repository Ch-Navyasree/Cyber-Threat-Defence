
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

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

BOT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def main():
    print("Loading validation data...")
    data = pd.read_parquet(DATA_PATH)

    predictor = ThreatPredictor()

    X = data[predictor.feature_names].to_numpy(dtype=np.float32)
    y_true = data["Attack_Type"].astype(str).to_numpy()

    print(f"Validation records: {len(data):,}")
    print("Calculating raw model probabilities...")

    probability_batches = []

    for start in range(0, len(X), 4096):
        batch = torch.tensor(
            X[start:start + 4096],
            dtype=torch.float32,
        )

        with torch.no_grad():
            probabilities = F.softmax(
                predictor.model(batch),
                dim=1,
            )

        probability_batches.append(probabilities.cpu().numpy())

    probabilities = np.concatenate(probability_batches, axis=0)

    class_names = predictor.class_names
    bot_idx = class_names.index("BOT")
    benign_idx = class_names.index("BENIGN")

    # Start from the existing policy, then test an experimental BOT override.
    baseline_indices = predictor.apply_threshold_policy(probabilities)
    results = []

    for threshold in BOT_THRESHOLDS:
        predicted_indices = baseline_indices.copy()

        # Experimental rule:
        # If BENIGN is the raw top prediction but BOT probability reaches
        # this threshold, predict BOT instead.
        raw_top_indices = probabilities.argmax(axis=1)

        eligible = (
            (raw_top_indices == benign_idx)
            & (probabilities[:, bot_idx] >= threshold)
        )

        predicted_indices[eligible] = bot_idx

        y_pred = np.array([
            class_names[int(index)] for index in predicted_indices
        ])

        actual_bot = y_true == "BOT"
        predicted_bot = y_pred == "BOT"

        bot_precision = precision_score(
            actual_bot, predicted_bot, zero_division=0
        )
        bot_recall = recall_score(
            actual_bot, predicted_bot, zero_division=0
        )
        bot_f1 = f1_score(
            actual_bot, predicted_bot, zero_division=0
        )

        benign_mask = y_true == "BENIGN"
        benign_false_alarms = int(
            np.sum(benign_mask & predicted_bot)
        )

        results.append({
            "BOT_threshold": threshold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Macro_F1": f1_score(
                y_true,
                y_pred,
                labels=class_names,
                average="macro",
                zero_division=0,
            ),
            "BOT_Precision": bot_precision,
            "BOT_Recall": bot_recall,
            "BOT_F1": bot_f1,
            "Benign_records_flagged_as_BOT": benign_false_alarms,
            "Records_changed": int(
                np.sum(predicted_indices != baseline_indices)
            ),
        })

    result_df = pd.DataFrame(results)

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "transformed"
        / "bot_threshold_experiment.csv"
    )

    result_df.to_csv(output_path, index=False)

    print("\nValidation-only BOT threshold experiment:")
    print(result_df.round(4).to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    print("\nNo model weights or production thresholds were changed.")


if __name__ == "__main__":
    main()
