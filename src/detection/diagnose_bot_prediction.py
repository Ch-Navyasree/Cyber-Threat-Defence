
from pathlib import Path
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
    row = data[data["Attack_Type"] == "BOT"].iloc[[0]]

    features = row.drop(columns=["Attack_Type"])

    predictor = ThreatPredictor()

    # Reorder features exactly as expected by the trained model.
    features = features[predictor.feature_names]
    values = features.to_numpy(dtype="float32")

    tensor = torch.tensor(values, dtype=torch.float32)

    with torch.no_grad():
        logits = predictor.model(tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

    print("Actual label: BOT")
    print("\nRaw model probabilities:")

    for name, probability in sorted(
        zip(predictor.class_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:15s} {probability:.6f}")

    raw_index = int(probabilities.argmax())
    raw_label = predictor.class_names[raw_index]

    final_prediction = predictor.apply_threshold_policy(
        probabilities.reshape(1, -1)
    )[0]
    final_label = predictor.class_names[int(final_prediction)]

    print("\nRaw model prediction:", raw_label)
    print("Prediction after threshold policy:", final_label)


if __name__ == "__main__":
    main()
