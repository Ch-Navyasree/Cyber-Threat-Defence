from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.parquet"

TARGET = "Attack_Type"
SOURCE = "Source_File"

df = pl.read_parquet(INPUT_FILE)

features = [
    column for column in df.columns
    if column not in {TARGET, SOURCE}
]

print("Checking repeated feature vectors with conflicting labels...")

# Find exact feature-value groups, not hash-only matches.
conflicts = (
    df.group_by(features)
    .agg(
        pl.len().alias("row_count"),
        pl.col(TARGET).n_unique().alias("label_count"),
        pl.col(TARGET).unique().alias("labels"),
        pl.col(SOURCE).n_unique().alias("source_count"),
        pl.col(SOURCE).unique().alias("source_files"),
    )
    .filter(pl.col("label_count") > 1)
    .sort("row_count", descending=True)
)

print(f"\nExact conflicting feature groups: {conflicts.height:,}")

if conflicts.height:
    print("\nTop 20 conflicting groups:")
    print(conflicts.head(20).select(
        "row_count",
        "label_count",
        "labels",
        "source_count",
        "source_files",
    ))

    output = (
        PROJECT_ROOT
        / "reports"
        / "leakage_audit"
        / "verified_clean_label_conflicts.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    conflicts.write_parquet(output)

    print(f"\nFull results saved to: {output}")
else:
    print("No exact feature-value groups with conflicting labels were found.")