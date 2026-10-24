from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src" / "agents"))

from nf_unsw_investigation_tool import NFUNSWInvestigationTool


def main():
    tool = NFUNSWInvestigationTool()

    # Load one row from the existing training split.
    preprocessor = joblib.load(
        ROOT / "models" / "nf_unsw_binary_preprocessor.joblib"
    )

    train_path = (
        ROOT / "data" / "processed" / "nf_unsw" / "train.parquet"
    )
    df = pd.read_parquet(train_path)

    feature_names = preprocessor["feature_columns"]
    raw_features = df[feature_names].iloc[0].to_dict()

    result = tool.investigate(raw_features)

    print("Investigation Tool Test")
    print("=" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")

    assert result["status"] == "completed"
    assert result["finding"] in {"Benign", "Attack"}
    assert result["action_taken"] is False
    assert 0.0 <= result["attack_probability"] <= 1.0

    print("\nINVESTIGATION TOOL TEST: PASS")


if __name__ == "__main__":
    main()