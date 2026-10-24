
from pathlib import Path
import pandas as pd

from src.intelligence.detection_integration import DetectionIntegration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "selected_v2"
    / "validation.parquet"
)

ATTACK_CATEGORIES = [
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DDoS",
    "DoS",
    "PORT_SCAN",
    "WEB_ATTACK",
]


def main():
    print("Loading validation data...")
    data = pd.read_parquet(DATA_PATH)

    integration = DetectionIntegration()
    print("\nTesting detection and threat intelligence...\n")

    for category in ATTACK_CATEGORIES:
        matching_rows = data[data["Attack_Type"] == category]

        if matching_rows.empty:
            print(f"{category}: No validation records found.")
            continue

        # Select one known record for this category.
        network_flow = matching_rows.drop(
            columns=["Attack_Type"]
        ).iloc[[0]]

        result = integration.analyze(network_flow)

        detection = result["detection"]
        investigation = result["investigation"]

        print(f"Actual category:      {category}")
        print(f"Predicted category:   {detection['predicted_attack']}")
        print(f"Confidence:           {detection['confidence']:.6f}")
        print(f"Investigation status: {investigation['investigation_status']}")
        print(f"Mapping status:       {investigation['mapping_status']}")
        print("-" * 55)


if __name__ == "__main__":
    main()
