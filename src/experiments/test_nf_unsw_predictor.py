from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "detection"))

from nf_unsw_predictor import NFUNSWPredictor


def main():
    predictor = NFUNSWPredictor()
    preprocessor = joblib.load(
        ROOT / "models" / "nf_unsw_binary_preprocessor.joblib"
    )

    train_path = ROOT / "data" / "processed" / "nf_unsw" / "train.parquet"
    df = pd.read_parquet(train_path)

    feature_names = preprocessor["feature_columns"]

    # Reverse the column order to verify name-based ordering.
    raw_row = df[feature_names].iloc[0]
    raw_features = raw_row.to_dict()

    result = predictor.predict_raw(raw_features)

    print("Raw-input prediction test")
    print("=" * 35)
    print("Input feature count:", len(raw_features))
    print("Prediction:", result)

    assert result["predicted_class"] in {"Benign", "Attack"}
    assert 0.0 <= result["attack_probability"] <= 1.0
    assert 0.0 <= result["benign_probability"] <= 1.0
    assert abs(
        result["attack_probability"]
        + result["benign_probability"] - 1.0
    ) < 1e-5

    print("\nRAW-INPUT TEST: PASS")


if __name__ == "__main__":
    main()