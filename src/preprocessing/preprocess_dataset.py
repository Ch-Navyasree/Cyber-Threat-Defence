from pathlib import Path
import polars as pl


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("data/raw/CICIDS2017")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cicids2017_clean.parquet"
REPORT_FILE = OUTPUT_DIR / "preprocessing_report.txt"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 1: LOADING RAW DATA")
print("=" * 70)

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DIR.resolve()}"
    )

dataframes = []

for file in csv_files:

    print(f"Reading: {file.name}")

    df = pl.read_csv(
        file,
        infer_schema_length=10000,
        ignore_errors=True
    )
    df = df.with_columns(
    pl.lit(file.name).alias("Source_File")
)

    # Remove accidental whitespace from column names
    df = df.rename({
        column: column.strip()
        for column in df.columns
    })

    dataframes.append(df)


df = pl.concat(
    dataframes,
    how="vertical_relaxed"
)

original_rows = df.height
original_columns = df.width

print(f"\nOriginal rows: {original_rows:,}")
print(f"Original columns: {original_columns}")


# ============================================================
# 2. REMOVE INVALID FLOW DURATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: REMOVING INVALID FLOW DURATION")
print("=" * 70)

invalid_duration = df.filter(
    pl.col("Flow Duration") < 0
).height

print(
    f"Rows with Flow Duration < 0: "
    f"{invalid_duration:,}"
)

df = df.filter(
    pl.col("Flow Duration") >= 0
)

print(f"Remaining rows: {df.height:,}")


# ============================================================
# 3. REMOVE EXACT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: REMOVING EXACT DUPLICATES")
print("=" * 70)

before_duplicates = df.height

df = df.unique(
    maintain_order=False
)

duplicates_removed = (
    before_duplicates - df.height
)

print(
    f"Duplicate rows removed: "
    f"{duplicates_removed:,}"
)

print(f"Remaining rows: {df.height:,}")


# ============================================================
# 4. REMOVE CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: REMOVING CONSTANT FEATURES")
print("=" * 70)

constant_columns = []

for column in df.columns:

    if column == "Label":
        continue

    if df[column].n_unique() <= 1:
        constant_columns.append(column)

print("Constant columns:")

for column in constant_columns:
    print(f"  - {column}")

df = df.drop(constant_columns)

print(
    f"\nRemoved {len(constant_columns)} "
    f"constant features."
)


# ============================================================
# 5. REMOVE DUPLICATED FEATURE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: REMOVING DUPLICATED FEATURE")
print("=" * 70)

duplicate_feature = "Fwd Header Length_duplicated_0"

if duplicate_feature in df.columns:

    df = df.drop(duplicate_feature)

    print(
        f"Removed: {duplicate_feature}"
    )

else:

    print(
        f"{duplicate_feature} not found."
    )


# ============================================================
# 6. MAP ORIGINAL LABELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: MAPPING ATTACK LABELS")
print("=" * 70)

df = df.with_columns(

    pl.when(
        pl.col("Label") == "BENIGN"
    )
    .then(pl.lit("BENIGN"))

    .when(
        pl.col("Label") == "DDoS"
    )
    .then(pl.lit("DDoS"))

    .when(
        pl.col("Label").is_in([
            "DoS Hulk",
            "DoS GoldenEye",
            "DoS slowloris",
            "DoS Slowhttptest"
        ])
    )
    .then(pl.lit("DoS"))

    .when(
        pl.col("Label") == "PortScan"
    )
    .then(pl.lit("PORT_SCAN"))

    .when(
        pl.col("Label").is_in([
            "FTP-Patator",
            "SSH-Patator"
        ])
    )
    .then(pl.lit("BRUTE_FORCE"))

    .when(
        pl.col("Label") == "Bot"
    )
    .then(pl.lit("BOT"))

    # Handle the actual Web Attack labels robustly.
    .when(
        pl.col("Label").str.contains(
            "Web Attack"
        )
    )
    .then(pl.lit("WEB_ATTACK"))

    .otherwise(
        pl.lit("EXCLUDED")
    )

    .alias("Attack_Type")
)


# ============================================================
# 7. SHOW MAPPING RESULTS
# ============================================================

print("\nMapped labels:")

mapped_distribution = (
    df
    .group_by("Attack_Type")
    .len()
    .sort("len", descending=True)
)

print(mapped_distribution)


# ============================================================
# 8. REMOVE EXCLUDED CLASSES
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: REMOVING EXCLUDED CLASSES")
print("=" * 70)

excluded_rows = df.filter(
    pl.col("Attack_Type") == "EXCLUDED"
).height

print(
    f"Excluded rows: {excluded_rows:,}"
)

df = df.filter(
    pl.col("Attack_Type") != "EXCLUDED"
)

print(
    f"Remaining rows: {df.height:,}"
)


# ============================================================
# 9. REMOVE ORIGINAL LABEL
# ============================================================

df = df.drop("Label")


# ============================================================
# 10. FINAL COLUMN ORDER
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in ["Attack_Type", "Source_File"]
]

df = df.select(
    feature_columns + ["Attack_Type", "Source_File"]
)


# ============================================================
# 11. FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"Rows: {df.height:,}"
)

print(
    f"Features: {len(feature_columns)}"
)

print(
    f"Total columns: {df.width}"
)

print("\nFinal class distribution:")

final_distribution = (
    df
    .group_by("Attack_Type")
    .len()
    .with_columns(
        (
            pl.col("len") /
            df.height *
            100
        )
        .round(4)
        .alias("percentage")
    )
    .sort("len", descending=True)
)

print(final_distribution)


# ============================================================
# 12. SAVE AS PARQUET
# ============================================================

print("\n" + "=" * 70)
print("SAVING CLEAN DATASET")
print("=" * 70)

df.write_parquet(
    OUTPUT_FILE,
    compression="zstd"
)

print(
    f"Saved to:\n{OUTPUT_FILE.resolve()}"
)


# ============================================================
# 13. SAVE REPORT
# ============================================================

report_lines = []

report_lines.append(
    "CICIDS2017 PREPROCESSING REPORT"
)

report_lines.append(
    "=" * 60
)

report_lines.append(
    f"Original rows: {original_rows:,}"
)

report_lines.append(
    f"Original columns: {original_columns}"
)

report_lines.append(
    f"Invalid Flow Duration rows removed: "
    f"{invalid_duration:,}"
)

report_lines.append(
    f"Duplicate rows removed: "
    f"{duplicates_removed:,}"
)

report_lines.append(
    "Constant features removed:"
)

for column in constant_columns:
    report_lines.append(
        f"  - {column}"
    )

report_lines.append(
    f"Duplicated feature removed: "
    f"{duplicate_feature}"
)

report_lines.append(
    f"Excluded rows: {excluded_rows:,}"
)

report_lines.append(
    f"Final rows: {df.height:,}"
)

report_lines.append(
    f"Final features: {len(feature_columns)}"
)

report_lines.append(
    "\nFinal class distribution:"
)

for row in final_distribution.iter_rows():
    label, count, percentage = row

    report_lines.append(
        f"{label}: {count:,} "
        f"({percentage}%)"
    )

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(
    f"Report saved to:\n{REPORT_FILE.resolve()}"
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)