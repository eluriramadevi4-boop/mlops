import pandas as pd
import os
import sys

DATA_PATH = "data/raw/stroke.csv"

print("=" * 60)
print("LAB 5 - DATA VALIDATION")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print("[ERROR] Dataset not found:", DATA_PATH)
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

print("\n[INFO] Dataset loaded successfully")
print("Shape:", df.shape)

required_columns = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]

print("\n[INFO] Checking required columns...")

missing_columns = []

for column in required_columns:
    if column not in df.columns:
        missing_columns.append(column)

if len(missing_columns) > 0:
    print("[ERROR] Missing columns:", missing_columns)
    sys.exit(1)

print("[SUCCESS] All required columns are present.")

print("\n[INFO] Checking duplicate rows...")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

print("\n[INFO] Checking target column...")

if "stroke" not in df.columns:
    print("[ERROR] Target column 'stroke' not found.")
    sys.exit(1)

print("Target column: stroke")
print("Target values:", df["stroke"].unique().tolist())

print("\n[INFO] Checking missing values...")

missing_values = df.isnull().sum()

print(missing_values)

print("\n[SUCCESS] Data validation completed successfully.")