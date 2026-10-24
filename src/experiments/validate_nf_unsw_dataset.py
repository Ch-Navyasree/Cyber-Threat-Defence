
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "nf_unsw"

SPLITS = ["train", "validation", "test"]
TARGET = "Label"


def main():
    with open(DATA_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    expected_features = metadata["feature_columns"]
    expected_columns = expected_features + [TARGET]

    total_rows = 0

    for split in SPLITS:
        path = DATA_DIR / f"{split}.parquet"
        df = pd.read_parquet(path)

        assert list(df.columns) == expected_columns, (
            f"{split}: column names/order mismatch"
        )

        labels = set(df[TARGET].unique())
        assert labels.issubset({0, 1}), (
            f"{split}: unexpected labels {labels}"
        )

        X = df[expected_features].to_numpy(dtype=np.float32)
        assert np.isfinite(X).all(), (
            f"{split}: NaN or infinite feature values found"
        )

        counts = df[TARGET].value_counts().sort_index()
        total_rows += len(df)

        print(f"\n{split.upper()}")
        print(f"Rows: {len(df):,}")
        print(f"Features: {len(expected_features)}")
        print(f"Label counts:\n{counts.to_string()}")
        print(
            "Attack percentage: "
            f"{100 * (df[TARGET] == 1).mean():.4f}%"
        )
        print("Column order: PASS")
        print("Finite feature values: PASS")

    assert total_rows == metadata["original_rows"], (
        f"Row count mismatch: {total_rows} "
        f"!= {metadata['original_rows']}"
    )

    print("\nSUCCESS: all external dataset splits passed validation.")


if __name__ == "__main__":
    main()
