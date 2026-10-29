from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "leakage_audit"
    / "verified_clean_label_conflicts.parquet"
)

df = pl.read_parquet(INPUT_FILE)

print("CONFLICTING FEATURE GROUP SUMMARY")
print("=" * 60)
print(f"Conflicting groups: {df.height:,}")

print(
    "Rows represented by conflicting groups:",
    f"{df['row_count'].sum():,}",
)

print("\nConflicting label combinations:")
print(
    df.group_by("labels")
    .agg(
        pl.len().alias("group_count"),
        pl.col("row_count").sum().alias("affected_rows"),
    )
    .sort("affected_rows", descending=True)
)

print("\nGroups by number of source files:")
print(
    df.group_by("source_count")
    .agg(
        pl.len().alias("group_count"),
        pl.col("row_count").sum().alias("affected_rows"),
    )
    .sort("source_count")
)

print("\nLargest conflicting groups:")
print(
    df.sort("row_count", descending=True)
    .select(
        "row_count",
        "labels",
        "source_count",
        "source_files",
    )
    .head(10)
)