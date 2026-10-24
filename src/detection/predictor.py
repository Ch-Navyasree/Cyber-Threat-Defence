
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd

from src.detection.baseline_detector import ThreatDetectionTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "transformer_detector_2000_best_with_features.pt"
)

DEVICE = torch.device("cpu")
THRESHOLDS = {
    "BOT": 0.80,
    "WEB_ATTACK": 0.80,
}


class ThreatPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        # Load the checkpoint safely.
        # The new checkpoint contains only standard Python metadata
        # and the model's tensor state.
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )

        self.feature_names = checkpoint["feature_names"]
        self.class_names = checkpoint["class_names"]

        self.num_features = checkpoint["num_features"]
        self.num_classes = checkpoint["num_classes"]

        if len(self.feature_names) != self.num_features:
            raise ValueError("Checkpoint feature count mismatch.")

        if len(self.class_names) != self.num_classes:
            raise ValueError("Checkpoint class count mismatch.")

        # Re-create the same architecture used during training.
        self.model = ThreatDetectionTransformer(
            num_features=self.num_features,
            num_classes=self.num_classes,
        ).to(DEVICE)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print("Threat Predictor loaded successfully.")
        print(f"Features: {self.num_features}")
        print(f"Classes: {self.num_classes}")

        
    def apply_threshold_policy(self, probabilities):
        """
        Reject low-confidence BOT and WEB_ATTACK predictions.
        Fall back to the strongest non-target class.
        """
        import numpy as np

        predictions = probabilities.argmax(axis=1)

        target_indices = {
            self.class_names.index(target)
            for target in THRESHOLDS
        }

        for target, threshold in THRESHOLDS.items():
            target_idx = self.class_names.index(target)

            low_confidence = (
                (predictions == target_idx)
                & (probabilities[:, target_idx] < threshold)
            )

            rows = np.where(low_confidence)[0]

            if len(rows) == 0:
                continue

            alternative_scores = probabilities[rows].copy()
            alternative_scores[:, list(target_indices)] = -1.0

            predictions[rows] = alternative_scores.argmax(axis=1)

        return predictions


    def predict(self, data):
        """
        Predict attack categories from a DataFrame or dictionary.

        A dictionary represents one network-flow record.
        A DataFrame represents one or more records.
        """

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame or dictionary."
            )

        missing = [
            name for name in self.feature_names
            if name not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing {len(missing)} required features: "
                f"{missing[:5]}"
            )

        # Match the exact feature order used during training.
        data = data.loc[:, self.feature_names]

        # Reject invalid values instead of silently altering them.
        data = data.apply(pd.to_numeric, errors="raise")

        if data.isna().any().any():
            raise ValueError("Input contains missing values.")

        values = data.to_numpy(dtype="float32")

        if not torch.isfinite(torch.tensor(values)).all():
            raise ValueError(
                "Input contains infinite or invalid values."
            )

        features = torch.tensor(
            values,
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():
            logits = self.model(features)
            probabilities = F.softmax(logits, dim=1)

            probabilities = probabilities.cpu().numpy()

            predicted_indices = self.apply_threshold_policy(
                probabilities
        )

        results = []

        for index, row in enumerate(predicted_indices):
            class_index = row.item()

            results.append({
                "predicted_attack": self.class_names[class_index],
                "confidence": round(
    float(probabilities[index, class_index]),
    6,
),
            })

        return pd.DataFrame(results)
