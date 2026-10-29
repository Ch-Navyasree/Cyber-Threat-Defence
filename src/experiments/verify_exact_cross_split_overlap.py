from pathlib import Path
import json

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "transformed" / "selected_v2"
OUT_DIR = ROOT / "reports" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Attack_Type"
SPLITS = ["train", "validation", "test"]


def main():
    frames = {
        name: pl.read_parquet(SPLIT_DIR / f"{name}.parquet")
        for name in SPLITS
    }

    features = [c for c in frames["train"].columns if c != TARGET]
    results = {}

    for a, b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        print(f"\nChecking {a} vs {b}...")

        left = frames[a].select(features + [TARGET])
        right = frames[b].select(features + [TARGET])

        # Reduce each side to distinct feature/label combinations.
        left_unique = left.unique()
        right_unique = right.unique()

        # Join on every feature column, so matches are exact feature-value matches.
        matched = left_unique.join(
            right_unique,
            on=features,
            how="inner",
            suffix="_right",
        )

        # Each row represents a shared feature vector and a pair of labels.
        shared_vectors = matched.select(features).unique().height

        label_disagreements = matched.filter(
            pl.col(TARGET) != pl.col(f"{TARGET}_right")
        ).select(features).unique().height

        results[f"{a}_vs_{b}"] = {
            "shared_feature_vectors": shared_vectors,
            "shared_vectors_with_label_disagreement": label_disagreements,
        }

        print(f"Shared exact feature vectors: {shared_vectors:,}")
        print(f"Shared vectors with label disagreement: {label_disagreements:,}")

    report_path = OUT_DIR / "exact_cross_split_overlap.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()