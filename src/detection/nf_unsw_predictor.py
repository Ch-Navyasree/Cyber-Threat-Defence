from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from nf_unsw_binary_detector import NFUNSWBinaryTransformer


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = ROOT / "models" / "nf_unsw_binary_transformer.pt"
PREPROCESSOR_PATH = ROOT / "models" / "nf_unsw_binary_preprocessor.joblib"

ATTACK_THRESHOLD = 0.5


class NFUNSWPredictor:
    """Read-only prediction interface for the NF-UNSW binary detector."""

    def __init__(self):
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(CHECKPOINT_PATH)

        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(PREPROCESSOR_PATH)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.preprocessing = joblib.load(PREPROCESSOR_PATH)
        self.feature_names = self.checkpoint["feature_names"]

        if self.feature_names != self.preprocessing["feature_columns"]:
            raise ValueError(
                "Checkpoint and preprocessing feature order do not match."
            )

        self.model = NFUNSWBinaryTransformer(
            num_features=self.checkpoint["num_features"],
            num_classes=self.checkpoint["num_classes"],
        ).to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )
        self.model.eval()

    def predict_raw(self, raw_features):
        """
        Predict one flow from raw numeric feature values.

        raw_features must be a dictionary containing exactly the
        41 expected feature names and their raw numeric values.
        """

        if not isinstance(raw_features, dict):
            raise TypeError("raw_features must be a dictionary.")

        missing = [
            name for name in self.feature_names
            if name not in raw_features
        ]

        extra = [
            name for name in raw_features
            if name not in self.feature_names
        ]

        if missing:
            raise ValueError(f"Missing required features: {missing}")

        if extra:
            raise ValueError(f"Unexpected features: {extra}")

        row = pd.DataFrame(
            [[raw_features[name] for name in self.feature_names]],
            columns=self.feature_names,
        )

        # Convert non-finite values to missing values before imputation.
        row = row.replace([np.inf, -np.inf], np.nan)

        # Use the fitted training preprocessing; do not fit again.
        imputed = self.preprocessing["imputer"].transform(row)
        scaled = self.preprocessing["scaler"].transform(imputed)

        if not np.isfinite(scaled).all():
            raise ValueError(
                "Preprocessing produced NaN or infinite values."
            )

        return self.predict(scaled[0])

    def predict(self, features):
        """
        Predict one already-preprocessed flow.

        features must contain all 41 values in checkpoint feature order.
        """

        values = np.asarray(features, dtype=np.float32)

        if values.shape != (len(self.feature_names),):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"received shape {values.shape}."
            )

        if not np.isfinite(values).all():
            raise ValueError("Features contain NaN or infinite values.")

        tensor = torch.from_numpy(values.copy()).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        attack_probability = float(probabilities[1].item())
        benign_probability = float(probabilities[0].item())

        predicted_class = (
            "Attack"
            if attack_probability >= ATTACK_THRESHOLD
            else "Benign"
        )

        return {
            "predicted_class": predicted_class,
            "attack_probability": attack_probability,
            "benign_probability": benign_probability,
            "threshold": ATTACK_THRESHOLD,
            "checkpoint_epoch": int(self.checkpoint["epoch"]),
            "model": "NFUNSWBinaryTransformer",
            "dataset": "NF-UNSW-NB15-v2",
        }