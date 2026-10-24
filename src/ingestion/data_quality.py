from pathlib import Path
import polars as pl

DATA_DIR = Path("data/raw/CICIDS2017")

# --------------------------------------------------
# 1. Load all CSV files
# --------------------------------------------------

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV files found.")
    raise SystemExit(1)

dataframes = []

for file in csv_files:
    print(f"Reading: {file.name}")

    df = pl.read_csv(
        file,
        infer_schema_length=10000,
        ignore_errors=True
    )

    df = df.rename({
        column: column.strip()
        for column in df.columns
    })

    dataframes.append(df)

df = pl.concat(dataframes, how="vertical_relaxed")

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 2. Exact label counts
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXACT LABEL COUNTS")
print("=" * 70)

labels = (
    df
    .group_by("Label")
    .len()
    .sort("len", descending=True)
)

print(labels)


# --------------------------------------------------
# 3. Negative values
# --------------------------------------------------

print("\n" + "=" * 70)
print("NEGATIVE VALUES")
print("=" * 70)

numeric_columns = [
    column
    for column, dtype in zip(df.columns, df.dtypes)
    if dtype.is_numeric()
]

negative_results = []

for column in numeric_columns:

    count = (
        df
        .select(
            (pl.col(column) < 0).sum()
        )
        .item()
    )

    if count > 0:
        negative_results.append((column, count))

if negative_results:
    for column, count in negative_results:
        print(f"{column}: {count}")
else:
    print("No negative numeric values found.")


# --------------------------------------------------
# 4. Flow Duration specifically
# --------------------------------------------------

print("\n" + "=" * 70)
print("FLOW DURATION CHECK")
print("=" * 70)

negative_duration = (
    df
    .filter(pl.col("Flow Duration") < 0)
    .select([
        "Flow Duration",
        "Label"
    ])
)

print("Negative Flow Duration rows:", negative_duration.height)

if negative_duration.height > 0:
    print("\nNegative Flow Duration by label:")
    print(
        negative_duration
        .group_by("Label")
        .len()
        .sort("len", descending=True)
    )

    print("\nExample rows:")
    print(negative_duration.head(10))


# --------------------------------------------------
# 5. Duplicate feature comparison
# --------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE FEATURE CHECK")
print("=" * 70)

col1 = "Fwd Header Length"
col2 = "Fwd Header Length_duplicated_0"

if col1 in df.columns and col2 in df.columns:

    different_count = (
        df
        .select(
            (pl.col(col1) != pl.col(col2)).sum()
        )
        .item()
    )

    print(f"{col1} vs {col2}")
    print("Different rows:", different_count)

    if different_count == 0:
        print("RESULT: Columns are identical.")
    else:
        print("RESULT: Columns are NOT identical.")

else:
    print("One or both columns were not found.")


# --------------------------------------------------
# 6. Proposed label mapping
# --------------------------------------------------

print("\n" + "=" * 70)
print("PROPOSED 7-CLASS MAPPING")
print("=" * 70)

df = df.with_columns(
    pl.when(pl.col("Label") == "BENIGN")
      .then(pl.lit("BENIGN"))

    .when(pl.col("Label") == "DDoS")
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

    .when(pl.col("Label") == "PortScan")
      .then(pl.lit("PORT_SCAN"))

    .when(
        pl.col("Label").is_in([
            "FTP-Patator",
            "SSH-Patator"
        ])
    )
      .then(pl.lit("BRUTE_FORCE"))

    .when(pl.col("Label") == "Bot")
      .then(pl.lit("BOT"))

    .when(
        pl.col("Label").is_in([
            "Web Attack � Brute Force",
            "Web Attack � XSS",
            "Web Attack � Sql Injection"
        ])
    )
      .then(pl.lit("WEB_ATTACK"))

    .otherwise(pl.lit("EXCLUDED"))
    .alias("Mapped_Label")
)


# --------------------------------------------------
# 7. Mapped distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("MAPPED LABEL DISTRIBUTION")
print("=" * 70)

mapped = (
    df
    .group_by("Mapped_Label")
    .len()
    .sort("len", descending=True)
)

print(mapped)


# --------------------------------------------------
# 8. Number of usable samples
# --------------------------------------------------

usable = df.filter(
    pl.col("Mapped_Label") != "EXCLUDED"
)

print("\nUsable samples:", usable.height)
print("Excluded samples:", df.height - usable.height)


# --------------------------------------------------
# 9. Class percentages
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLASS PERCENTAGES")
print("=" * 70)

total = usable.height

percentages = (
    usable
    .group_by("Mapped_Label")
    .len()
    .with_columns(
        (
            pl.col("len") / total * 100
        ).round(4).alias("percentage")
    )
    .sort("len", descending=True)
)

print(percentages)


print("\n" + "=" * 70)
print("DATA QUALITY CHECK COMPLETE")
print("=" * 70)