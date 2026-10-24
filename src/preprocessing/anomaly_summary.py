from pathlib import Path
import polars as pl

FILE = Path("data/processed/standard/train.parquet")

df = pl.read_parquet(FILE)

features = [
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

print("=" * 80)
print("ANOMALY SUMMARY")
print("=" * 80)

for feature in features:

    col = df[feature]

    print(f"\n{feature}")
    print("-" * 60)
    print(f"Minimum : {col.min()}")
    print(f"Maximum : {col.max()}")
    print(f"Negative: {col.lt(0).sum()}")
    print(f"Zero    : {col.eq(0).sum()}")

print("\n" + "=" * 80)
print("INFINITY SUMMARY")
print("=" * 80)

for feature in ["Flow Bytes/s", "Flow Packets/s"]:

    col = df[feature].cast(pl.Float64, strict=False)

    print(f"\n{feature}")
    print(f"+Infinity: {col.is_infinite().sum()}")
    print(f"Null: {col.is_null().sum()}")

print("\n" + "=" * 80)
print("SUMMARY COMPLETE")
print("=" * 80)