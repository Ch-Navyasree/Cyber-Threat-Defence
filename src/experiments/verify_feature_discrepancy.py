from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# 1. Locate transformed training data
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

transformed_dir = project_root / "data" / "processed" / "transformed"

print("\nProject root:")
print(project_root)

print("\nTransformed directory:")
print(transformed_dir)


# ---------------------------------------------------------
# 2. Find parquet files
# ---------------------------------------------------------

parquet_files = list(transformed_dir.rglob("*.parquet"))

print("\nParquet files found:")

for file in parquet_files:
    print(" -", file.relative_to(project_root))


if not parquet_files:
    raise FileNotFoundError(
        "No Parquet files were found under data/processed/transformed/"
    )


# ---------------------------------------------------------
# 3. Display available files
# ---------------------------------------------------------

print("\nNumber of Parquet files:", len(parquet_files))


# ---------------------------------------------------------
# 4. Look for training file
# ---------------------------------------------------------

train_candidates = [
    f for f in parquet_files
    if "train" in f.name.lower()
]

if not train_candidates:
    raise FileNotFoundError(
        "Could not automatically identify the training Parquet file."
    )

print("\nTraining candidates:")

for file in train_candidates:
    print(" -", file.relative_to(project_root))


# Use the first training candidate for now
train_file = train_candidates[0]

print("\nUsing training file:")
print(train_file.relative_to(project_root))


# ---------------------------------------------------------
# 5. Load only the schema
# ---------------------------------------------------------

df = pd.read_parquet(train_file)


# ---------------------------------------------------------
# 6. Inspect dimensions
# ---------------------------------------------------------

print("\n========================================")
print("DATASET SHAPE")
print("========================================")

print("Rows   :", df.shape[0])
print("Columns:", df.shape[1])


# ---------------------------------------------------------
# 7. Identify target
# ---------------------------------------------------------

target = "Attack_Type"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' was not found."
    )


features = [
    column for column in df.columns
    if column != target
]


print("\n========================================")
print("FEATURE COUNT")
print("========================================")

print("Total columns :", len(df.columns))
print("Target column :", target)
print("Model features:", len(features))


# ---------------------------------------------------------
# 8. Print every feature
# ---------------------------------------------------------

print("\n========================================")
print("MODEL FEATURES")
print("========================================")

for i, feature in enumerate(features, start=1):
    print(f"{i:02d}. {feature}")

# ---------------------------------------------------------
# 7B. Compare training features with feature-quality report
# ---------------------------------------------------------

quality_report = (
    project_root
    / "data"
    / "processed"
    / "transformed"
    / "feature_quality"
    / "feature_quality_report.parquet"
)

print("\n========================================")
print("FEATURE QUALITY REPORT COMPARISON")
print("========================================")

if not quality_report.exists():
    print("Feature quality report not found:")
    print(quality_report)

else:
    quality_df = pd.read_parquet(quality_report)

    print("Report shape:", quality_df.shape)

    print("\nReport columns:")
    print(list(quality_df.columns))

    # Try to identify the feature-name column
    possible_name_columns = [
        "Feature",
        "feature",
        "Feature_Name",
        "feature_name",
        "Column",
        "column"
    ]

    feature_name_column = None

    for column in possible_name_columns:
        if column in quality_df.columns:
            feature_name_column = column
            break

    if feature_name_column is None:
        print("\nCould not automatically identify feature-name column.")
    else:
        report_features = set(
            quality_df[feature_name_column].astype(str)
        )

        train_features = set(features)

        missing_from_report = sorted(
            train_features - report_features
        )

        extra_in_report = sorted(
            report_features - train_features
        )

        print("\nFeature-name column:", feature_name_column)

        print("\nTraining features:", len(train_features))
        print("Report features:", len(report_features))

        print("\nFeatures in training data but NOT in report:")
        for feature in missing_from_report:
            print(" -", feature)

        print("\nFeatures in report but NOT in training data:")
        for feature in extra_in_report:
            print(" -", feature)

# ---------------------------------------------------------
# 9. Check duplicate column names
# ---------------------------------------------------------

duplicate_column_names = df.columns[
    df.columns.duplicated()
].tolist()

print("\n========================================")
print("DUPLICATE COLUMN NAMES")
print("========================================")

if duplicate_column_names:
    print(duplicate_column_names)
else:
    print("None")


# ---------------------------------------------------------
# 10. Check exact duplicate feature CONTENT
# ---------------------------------------------------------

print("\n========================================")
print("EXACT DUPLICATE FEATURE CONTENT")
print("========================================")

duplicate_pairs = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        col_a = features[i]
        col_b = features[j]

        if df[col_a].equals(df[col_b]):
            duplicate_pairs.append((col_a, col_b))


if duplicate_pairs:
    for col_a, col_b in duplicate_pairs:
        print(f"{col_a}  ==  {col_b}")
else:
    print("No exact duplicate feature contents found.")


# ---------------------------------------------------------
# 11. Final checkpoint
# ---------------------------------------------------------

print("\n========================================")
print("CHECKPOINT")
print("========================================")

print(f"Dataset columns : {df.shape[1]}")
print(f"Model features  : {len(features)}")
print(f"Target          : {target}")
print(
    f"Exact duplicate pairs found: "
    f"{len(duplicate_pairs)}"
)

print("\nVerification completed.")