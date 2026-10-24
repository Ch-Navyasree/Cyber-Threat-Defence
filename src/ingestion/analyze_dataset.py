from pathlib import Path
import polars as pl

DATA_DIR = Path("data/raw/CICIDS2017")

# --------------------------------------------------
# 1. Find CSV files
# --------------------------------------------------

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV files found.")
    print("Expected location:", DATA_DIR.resolve())
    raise SystemExit(1)

print(f"Found {len(csv_files)} CSV files.\n")


# --------------------------------------------------
# 2. Read and combine datasets
# --------------------------------------------------

dataframes = []

for file in csv_files:
    print(f"Reading: {file.name}")

    df = pl.read_csv(
        file,
        infer_schema_length=10000,
        ignore_errors=True
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

print("\n" + "=" * 70)
print("COMBINED DATASET")
print("=" * 70)

print("Rows:", df.height)
print("Columns:", df.width)


# --------------------------------------------------
# 3. Show columns
# --------------------------------------------------

print("\nColumns:")
for i, column in enumerate(df.columns):
    print(f"{i:02d}. {column}")


# --------------------------------------------------
# 4. Original label distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("ORIGINAL LABEL DISTRIBUTION")
print("=" * 70)

label_distribution = (
    df
    .group_by("Label")
    .len()
    .sort("len", descending=True)
)

print(label_distribution)


# --------------------------------------------------
# 5. Missing values
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = []

for column in df.columns:
    count = df[column].null_count()

    if count > 0:
        missing.append((column, count))

if missing:
    for column, count in missing:
        print(f"{column}: {count}")
else:
    print("No null values detected.")


# --------------------------------------------------
# 6. Numeric infinity check
# --------------------------------------------------

print("\n" + "=" * 70)
print("INFINITY CHECK")
print("=" * 70)

numeric_columns = [
    column
    for column, dtype in zip(df.columns, df.dtypes)
    if dtype.is_numeric()
]

infinity_found = False

for column in numeric_columns:
    try:
        positive_inf = df.select(
            pl.col(column).is_infinite().sum()
        ).item()

        negative_inf = df.select(
            pl.col(column).is_infinite().sum()
        ).item()

        total_inf = positive_inf + negative_inf

        if total_inf > 0:
            infinity_found = True
            print(f"{column}: {total_inf}")
    except Exception:
        pass

if not infinity_found:
    print("No infinity values detected.")


# --------------------------------------------------
# 7. Constant columns
# --------------------------------------------------

print("\n" + "=" * 70)
print("CONSTANT COLUMNS")
print("=" * 70)

constant_columns = []

for column in df.columns:
    unique_count = df[column].n_unique()

    if unique_count <= 1:
        constant_columns.append(column)

if constant_columns:
    for column in constant_columns:
        print(column)
else:
    print("No constant columns found.")


# --------------------------------------------------
# 8. Duplicate rows
# --------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicate_count = df.height - df.unique().height

print("Duplicate rows:", duplicate_count)


# --------------------------------------------------
# 9. Basic numeric statistics
# --------------------------------------------------

print("\n" + "=" * 70)
print("NUMERIC SUMMARY")
print("=" * 70)

print(
    df
    .select(numeric_columns)
    .describe()
)


# --------------------------------------------------
# 10. Dataset memory estimate
# --------------------------------------------------

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("Rows:", df.height)
print("Columns:", df.width)
print("Numeric columns:", len(numeric_columns))