from pathlib import Path
import polars as pl

DATA_DIR = Path("data/raw/CICIDS2017")

csv_files = list(DATA_DIR.glob("*.csv"))

print(f"Number of CSV files: {len(csv_files)}")

for file in csv_files:
    print("\n" + "=" * 70)
    print(f"FILE: {file.name}")
    print("=" * 70)

    df = pl.read_csv(
        file,
        infer_schema_length=10000,
        ignore_errors=True
    )

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns)

    # Dataset versions sometimes contain whitespace in column names
    label_columns = [
        column
        for column in df.columns
        if column.strip().lower() == "label"
    ]

    if label_columns:
        label_column = label_columns[0]

        print("\nLabel distribution:")
        print(
            df.group_by(label_column)
              .len()
              .sort("len", descending=True)
        )
    else:
        print("Label column not detected.")