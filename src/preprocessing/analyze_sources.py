from pathlib import Path
import polars as pl

DATA_FILE = Path(
    "data/processed/cicids2017_clean.parquet"
)

df = pl.read_parquet(DATA_FILE)

print("=" * 70)
print("SOURCE FILE DISTRIBUTION")
print("=" * 70)

source_summary = (
    df
    .group_by(["Source_File", "Attack_Type"])
    .len()
    .sort(["Source_File", "len"], descending=[False, True])
)

print(source_summary)


print("\n" + "=" * 70)
print("SOURCE FILE TOTALS")
print("=" * 70)

source_totals = (
    df
    .group_by("Source_File")
    .len()
    .sort("len", descending=True)
)

print(source_totals)


print("\n" + "=" * 70)
print("ATTACK CLASSES BY SOURCE FILE")
print("=" * 70)

for source in source_totals["Source_File"].to_list():

    print("\n" + "-" * 70)
    print(source)
    print("-" * 70)

    result = (
        df
        .filter(pl.col("Source_File") == source)
        .group_by("Attack_Type")
        .len()
        .sort("len", descending=True)
    )

    print(result)


print("\n" + "=" * 70)
print("SOURCE ANALYSIS COMPLETE")
print("=" * 70)