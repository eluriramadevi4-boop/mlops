import pandas as pd
import os
import sys

PROCESSED_PATH = "data/processed"

print("=" * 60)
print("LAB 5 - OUTPUT VALIDATION")
print("=" * 60)

required_files = [
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv"
]

print("\n[INFO] Checking processed files...")

for file in required_files:

    file_path = os.path.join(PROCESSED_PATH, file)

    if not os.path.exists(file_path):
        print("[ERROR] Missing:", file)
        sys.exit(1)

    print("[OK]", file)

# Load files
X_train = pd.read_csv(
    os.path.join(PROCESSED_PATH, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(PROCESSED_PATH, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(PROCESSED_PATH, "y_train.csv")
)

y_test = pd.read_csv(
    os.path.join(PROCESSED_PATH, "y_test.csv")
)

print("\n[INFO] Output shapes")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n[INFO] Checking missing values...")

if X_train.isnull().sum().sum() > 0:
    print("[ERROR] Missing values found in X_train.")
    sys.exit(1)

if X_test.isnull().sum().sum() > 0:
    print("[ERROR] Missing values found in X_test.")
    sys.exit(1)

print("[SUCCESS] No missing values found.")

print("\n[INFO] Checking feature consistency...")

if list(X_train.columns) != list(X_test.columns):
    print("[ERROR] X_train and X_test features do not match.")
    sys.exit(1)

print("[SUCCESS] X_train and X_test features match.")

print("\n" + "=" * 60)
print("[SUCCESS] OUTPUT VALIDATION COMPLETED")
print("=" * 60)