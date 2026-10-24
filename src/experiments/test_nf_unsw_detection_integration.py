from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's module search path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intelligence.nf_unsw_detection_integration import (
    NFUNSWDetectionIntegration,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    print("Loading NF-UNSW training data...")

    preprocessor = joblib.load(
        ROOT / "models" / "nf_unsw_binary_preprocessor.joblib"
    )

    data = pd.read_parquet(
        ROOT / "data" / "processed" / "nf_unsw" / "train.parquet"
    )

    feature_names = preprocessor["feature_columns"]
    raw_features = data[feature_names].iloc[0].to_dict()

    integration = NFUNSWDetectionIntegration()
    result = integration.analyze(raw_features)

    print("\nNF-UNSW Detection Integration Test")
    print("=" * 45)
    print("Dataset:", result["dataset"])
    print("Finding:", result["detection"]["finding"])
    print(
        "Attack probability:",
        result["detection"]["attack_probability"],
    )
    print(
        "Investigation status:",
        result["investigation"]["investigation_status"],
    )
    print("Mapping status:", result["investigation"].get("mapping_status"))
    print("Action taken:", result["action_taken"])

    assert result["dataset"] == "NF-UNSW-NB15-v2"
    assert result["detection"]["finding"] in {"Benign", "Attack"}
    assert result["investigation"]["investigation_status"] in {
        "benign_prediction",
        "requires_investigation",
        "knowledge_not_found",
    }
    assert result["action_taken"] is False

    if result["detection"]["finding"] == "Attack":
        assert result["investigation"]["category"] == "unknown"
        assert result["investigation"]["mitre_technique_ids"] == []

    print("\nINTEGRATION TEST: PASS")


if __name__ == "__main__":
    main()