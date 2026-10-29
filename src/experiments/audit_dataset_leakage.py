
from pathlib import Path
import json
import time

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "transformed" / "selected_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "leakage_audit"

SPLITS = {
    "train": DATA_DIR / "train.parquet",
    "validation": DATA_DIR / "validation.parquet",
    "test": DATA_DIR / "test.parquet",
}

TARGET = "Attack_Type"
CHUNK_SIZE = 100_000


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("DATASET LEAKAGE AUDIT - EXACT DUPLICATE FEATURE ROWS")
    print("=" * 65)

    split_hashes = {}
    split_labels = {}
    feature_columns = None
    summary = {}

    for split_name, file_path in SPLITS.items():
        print(f"\nLoading {split_name}: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        df = pd.read_parquet(file_path)

        if TARGET not in df.columns:
            raise ValueError(f"{TARGET} missing from {split_name}")

        current_features = [c for c in df.columns if c != TARGET]

        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError(
                f"Feature columns/order differ in {split_name}"
            )

        print(f"Rows: {len(df):,}")
        print(f"Features: {len(current_features)}")

        # Hash only the feature columns, excluding the target.
        # Chunking limits temporary memory use.
        hashes = []
        labels = []

        for start in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[start:start + CHUNK_SIZE]

            chunk_hashes = pd.util.hash_pandas_object(
                chunk[current_features],
                index=False,
            ).to_numpy(dtype="uint64")

            hashes.append(chunk_hashes)
            labels.extend(chunk[TARGET].astype(str).tolist())

            print(
                f"  Hashed {min(start + CHUNK_SIZE, len(df)):,}"
                f" / {len(df):,} rows",
                end="\r",
            )

        all_hashes = pd.Series(
            pd.array(
                [int(value) for part in hashes for value in part],
                dtype="UInt64",
            )
        )

        all_labels = pd.Series(labels, dtype="string")

        split_hashes[split_name] = all_hashes
        split_labels[split_name] = all_labels

        duplicated = all_hashes.duplicated(keep=False)

        summary[split_name] = {
            "rows": int(len(df)),
            "unique_feature_hashes": int(all_hashes.nunique()),
            "rows_with_duplicate_feature_hash": int(duplicated.sum()),
            "duplicate_hash_groups": int(
                all_hashes[duplicated].nunique()
            ),
        }

        print(f"\nWithin {split_name} duplicate check complete.")

        del df, hashes, labels, all_hashes, all_labels

    # Compare feature hashes across different splits.
    cross_split = {}

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        first_hashes = split_hashes[first]
        second_hashes = split_hashes[second]

        first_set = set(map(int, first_hashes.unique()))
        second_set = set(map(int, second_hashes.unique()))

        overlap = first_set.intersection(second_set)

        cross_split[f"{first}_vs_{second}"] = {
            "shared_unique_feature_hashes": len(overlap),
            "first_split_rows_matching_other_split": int(
                first_hashes.isin(overlap).sum()
            ),
            "second_split_rows_matching_other_split": int(
                second_hashes.isin(overlap).sum()
            ),
        }

        print(
            f"\n{first} vs {second}: "
            f"{len(overlap):,} shared unique feature hashes"
        )

    # Check whether repeated feature hashes within each split have
    # conflicting labels.
    label_conflicts = {}

    for split_name in SPLITS:
        pairs = pd.DataFrame({
            "feature_hash": split_hashes[split_name].astype("uint64"),
            "label": split_labels[split_name],
        })

        labels_per_hash = pairs.groupby("feature_hash")["label"].nunique()
        conflict_hashes = labels_per_hash[labels_per_hash > 1].index

        label_conflicts[split_name] = {
            "feature_hashes_with_conflicting_labels": int(
                len(conflict_hashes)
            )
        }

    report = {
        "audit_type": "hash-based exact feature-row duplicate screening",
        "target_column": TARGET,
        "feature_count": len(feature_columns),
        "within_split": summary,
        "cross_split": cross_split,
        "within_split_label_conflicts": label_conflicts,
        "important_note": (
            "Hash matches are a screening result, not proof of leakage. "
            "Matching rows should be checked against original feature values. "
            "This audit does not detect all forms of related-flow or "
            "temporal leakage."
        ),
    }

    output_path = REPORT_DIR / "dataset_leakage_audit.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("AUDIT FINISHED")
    print(f"Report saved to: {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
