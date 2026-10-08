import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==============================
# File paths
# ==============================

DATA_PATH = "data/raw/stroke.csv"

X_TRAIN_PATH = "data/processed/X_train.csv"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

SCALER_PATH = "models/scaler.pkl"


# ==============================
# Load dataset
# ==============================

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ==============================
# Remove duplicate rows
# ==============================

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

df = df.drop_duplicates()


# ==============================
# Target column
# ==============================

TARGET = "stroke"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ==============================
# Remove ID column
# ==============================

if "id" in X.columns:
    X = X.drop(columns=["id"])


# ==============================
# Identify columns
# ==============================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ==============================
# Handle missing values
# ==============================

for column in numerical_columns:
    X[column] = X[column].fillna(X[column].median())

for column in categorical_columns:
    X[column] = X[column].fillna(X[column].mode()[0])


# ==============================
# One-hot encoding
# ==============================

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# ==============================
# Train-test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# Feature scaling
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# Convert back to DataFrame
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)


# ==============================
# Save processed data
# ==============================

X_train_scaled.to_csv(X_TRAIN_PATH, index=False)
X_test_scaled.to_csv(X_TEST_PATH, index=False)

y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)


# ==============================
# Save scaler
# ==============================

joblib.dump(scaler, SCALER_PATH)


# ==============================
# Save feature information
# ==============================

feature_info = {
    "target": TARGET,
    "number_of_features": len(X_train.columns),
    "features": X_train.columns.tolist(),
    "categorical_features": categorical_columns,
    "numerical_features": numerical_columns
}

with open(
    "artifacts/feature_information.json",
    "w"
) as file:
    json.dump(feature_info, file, indent=4)


# ==============================
# Output
# ==============================

print("\nPreprocessing completed successfully.")

print("Training data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

print("\nSaved files:")
print("X_train:", X_TRAIN_PATH)
print("X_test:", X_TEST_PATH)
print("y_train:", Y_TRAIN_PATH)
print("y_test:", Y_TEST_PATH)
print("Scaler:", SCALER_PATH)
print("Feature information: artifacts/feature_information.json")