
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "transformed" / "selected_v2"

TARGET = "Attack_Type"
SAMPLE_SIZE = 100


def main():
    print("=" * 65)
    print("VERIFYING CROSS-SPLIT FEATURE DUPLICATES")
    print("=" * 65)

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    feature_columns = [c for c in train.columns if c != TARGET]

    # Build a mapping from each train feature hash to its actual rows.
    train_hashes = pd.util.hash_pandas_object(
        train[feature_columns], index=False
    )

    train_groups = {}
    for position, value in enumerate(train_hashes):
        train_groups.setdefault(int(value), []).append(position)

    for split_name, other in [
        ("validation", validation),
        ("test", test),
    ]:
        print(f"\nChecking train vs {split_name}...")

        other_hashes = pd.util.hash_pandas_object(
            other[feature_columns], index=False
        )

        candidates = []
        for position, value in enumerate(other_hashes):
            key = int(value)
            if key in train_groups:
                candidates.append((position, key))

        print(f"Matching-hash rows found: {len(candidates):,}")

        checked = 0
        exact_matches = 0
        same_label = 0
        different_label = 0

        for other_pos, key in candidates[:SAMPLE_SIZE]:
            other_row = other.iloc[other_pos]

            for train_pos in train_groups[key]:
                train_row = train.iloc[train_pos]

                # Verify actual feature values to guard against hash collisions.
                equal_features = train_row[feature_columns].equals(
                    other_row[feature_columns]
                )

                if not equal_features:
                    continue

                exact_matches += 1
                checked += 1

                if train_row[TARGET] == other_row[TARGET]:
                    same_label += 1
                else:
                    different_label += 1

        print(f"Sampled matching-hash rows checked: {checked}")
        print(f"Verified exact feature matches: {exact_matches}")
        print(f"Exact matches with same label: {same_label}")
        print(f"Exact matches with different labels: {different_label}")

    print("\nVerification finished. No dataset files were modified.")


if __name__ == "__main__":
    main()
