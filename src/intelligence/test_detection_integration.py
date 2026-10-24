
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


def main():
    print("Loading validation data...")
    data = pd.read_parquet(DATA_PATH)

    # Use one row and remove the target column.
    network_flow = data.drop(columns=["Attack_Type"]).iloc[[0]]

    print("Initializing detection integration...")
    integration = DetectionIntegration()

    print("Analyzing network flow...")
    result = integration.analyze(network_flow)

    print("\nDetection result:")
    print(result["detection"])

    print("\nThreat-intelligence investigation:")
    print(result["investigation"])


if __name__ == "__main__":
    main()
