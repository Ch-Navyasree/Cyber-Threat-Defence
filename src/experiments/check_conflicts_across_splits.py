from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]

conflicts_file = (
    ROOT / "reports" / "leakage_audit"
    / "verified_clean_label_conflicts.parquet"
)

split_dir = ROOT / "data" / "processed" / "transformed" / "selected_v2"

conflicts = pl.read_parquet(conflicts_file)
train = pl.read_parquet(split_dir / "train.parquet")
validation = pl.read_parquet(split_dir / "validation.parquet")
test = pl.read_parquet(split_dir / "test.parquet")

# Match on the feature columns common to the cleaned and selected datasets.
# Use the selected dataset's actual feature names.
features = [
    c for c in train.columns
    if c != "Attack_Type"
]

print("Selected feature count:", len(features))
print("Conflict groups in cleaned data:", conflicts.height)
print(
    "Note: selected_v2 has fewer features than the cleaned dataset, "
    "so exact matching requires a consistent feature transformation."
)
print("Train shape:", train.shape)
print("Validation shape:", validation.shape)
print("Test shape:", test.shape)