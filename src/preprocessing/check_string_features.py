from pathlib import Path
import polars as pl


TRAIN_FILE = Path(
    "data/processed/standard/train.parquet"
)

df = pl.read_parquet(TRAIN_FILE)

print("=" * 70)
print("STRING FEATURE ANALYSIS")
print("=" * 70)

for feature in ["Flow Bytes/s", "Flow Packets/s"]:

    print("\n" + "-" * 70)
    print(feature)
    print("-" * 70)

    col = df[feature]

    print(f"Data type: {col.dtype}")
    print(f"Null values: {col.null_count()}")

    print("\nTop 20 values:")
    print(
        col
        .value_counts()
        .sort("count", descending=True)
        .head(20)
    )

    numeric = (
        col
        .cast(pl.Float64, strict=False)
    )

    failed = (
        col.is_not_null() &
        numeric.is_null()
    ).sum()

    print(f"\nValues that fail numeric conversion: {failed}")

    print("\nConverted numeric statistics:")

    print(
        numeric
        .describe()
    )

print("\n" + "=" * 70)
print("STRING FEATURE CHECK COMPLETE")
print("=" * 70)