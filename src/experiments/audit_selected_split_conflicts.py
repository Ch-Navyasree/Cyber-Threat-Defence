from pathlib import Path
import json

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "transformed" / "selected_v2"
OUT_DIR = ROOT / "reports" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]
TARGET = "Attack_Type"


def main():
    frames = {}
    feature_columns = None

    for name in SPLITS:
        df = pl.read_parquet(SPLIT_DIR / f"{name}.parquet")
        frames[name] = df

        current_features = [
            c for c in df.columns if c != TARGET
        ]

        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError(f"Feature mismatch in {name}")

        print(f"{name}: {df.height:,} rows, {len(current_features)} features")

    # Count exact feature-value groups with conflicting labels
    # inside each split.
    report = {"feature_count": len(feature_columns), "splits": {}}

    for name, df in frames.items():
        grouped = df.group_by(feature_columns).agg(
            pl.len().alias("row_count"),
            pl.col(TARGET).n_unique().alias("label_count"),
        )

        conflicts = grouped.filter(pl.col("label_count") > 1)

        report["splits"][name] = {
            "rows": df.height,
            "conflicting_feature_groups": conflicts.height,
            "rows_in_conflicting_groups": int(
                conflicts["row_count"].sum() or 0
            ),
        }

        print(f"\n{name} conflicting feature groups: {conflicts.height:,}")
        print(
            "Rows in those groups:",
            f"{int(conflicts['row_count'].sum() or 0):,}",
        )

    output = OUT_DIR / "selected_split_conflict_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()