from pathlib import Path
import json

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2017_clean.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "leakage_audit"
OUTPUT_FILE = OUTPUT_DIR / "clean_dataset_duplicate_audit.json"

TARGET = "Attack_Type"
SOURCE = "Source_File"
CHUNK_SIZE = 100_000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CLEAN DATASET DUPLICATE AUDIT")
    print("=" * 65)

    df = pl.read_parquet(INPUT_FILE)

    print(f"Rows: {df.height:,}")
    print(f"Columns: {df.width}")

    features = [
        c for c in df.columns
        if c not in {TARGET, SOURCE}
    ]

    print(f"Feature columns: {len(features)}")

    # Exact duplicate rows, including labels and source metadata.
    exact_duplicate_rows = (
        df.height - df.unique().height
    )

    print(
        "\nExact duplicate rows "
        f"(including source and label): {exact_duplicate_rows:,}"
    )

    # Hash feature values only, excluding label and source.
    # These hashes are a screening mechanism, not proof of equality.
    hash_parts = []

    for start in range(0, df.height, CHUNK_SIZE):
        chunk = df.slice(start, CHUNK_SIZE)

        hashes = chunk.select(
            pl.struct(features)
            .hash(seed=42)
            .alias("feature_hash")
        )["feature_hash"]

        hash_parts.append(hashes)

        print(
            f"Hashed {min(start + CHUNK_SIZE, df.height):,}"
            f" / {df.height:,} rows",
            end="\r",
        )

    all_hashes = pl.concat(hash_parts).alias("feature_hash")

    audit_df = pl.DataFrame({
        "feature_hash": all_hashes,
        TARGET: df[TARGET],
        SOURCE: df[SOURCE],
    })

    # Repeated feature hashes and conflicting labels.
    grouped = audit_df.group_by("feature_hash").agg(
        pl.len().alias("row_count"),
        pl.col(TARGET).n_unique().alias("label_count"),
        pl.col(SOURCE).n_unique().alias("source_count"),
    )

    repeated_groups = grouped.filter(
        pl.col("row_count") > 1
    )

    conflicting_groups = grouped.filter(
        pl.col("label_count") > 1
    )

    cross_source_groups = grouped.filter(
        (pl.col("row_count") > 1)
        & (pl.col("source_count") > 1)
    )

    report = {
        "dataset": str(INPUT_FILE),
        "rows": df.height,
        "columns": df.width,
        "feature_count": len(features),
        "exact_duplicate_rows_including_metadata": exact_duplicate_rows,
        "unique_feature_hashes": grouped.height,
        "repeated_feature_hash_groups": repeated_groups.height,
        "rows_in_repeated_feature_groups": int(
            repeated_groups["row_count"].sum() or 0
        ),
        "feature_hash_groups_with_conflicting_labels": (
            conflicting_groups.height
        ),
        "repeated_feature_groups_across_source_files": (
            cross_source_groups.height
        ),
        "note": (
            "Hash-based screening. Hash matches should be verified "
            "against actual feature values before drawing conclusions."
        ),
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()