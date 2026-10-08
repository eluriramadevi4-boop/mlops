import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/stroke.csv"
PROCESSED_PATH = "data/processed"
MODEL_PATH = "models"
ARTIFACT_PATH = "artifacts"

print("=" * 60)
print("LAB 5 - PRODUCTION PREPROCESSING")
print("=" * 60)

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(ARTIFACT_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n[INFO] Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove ID because it is not useful for prediction
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Separate target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Fill numerical missing values
for column in numerical_columns:
    X[column] = X[column].fillna(X[column].median())

# Fill categorical missing values
for column in categorical_columns:
    X[column] = X[column].fillna(X[column].mode()[0])

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Convert boolean columns to integers
X = X.astype(int)

print("[INFO] Features after encoding:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

# Save processed data
X_train_scaled.to_csv(
    os.path.join(PROCESSED_PATH, "X_train.csv"),
    index=False
)

X_test_scaled.to_csv(
    os.path.join(PROCESSED_PATH, "X_test.csv"),
    index=False
)

y_train.to_csv(
    os.path.join(PROCESSED_PATH, "y_train.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(PROCESSED_PATH, "y_test.csv"),
    index=False
)

# Save scaler
joblib.dump(
    scaler,
    os.path.join(MODEL_PATH, "scaler.pkl")
)

print("\n[INFO] Training data shape:", X_train_scaled.shape)
print("[INFO] Testing data shape :", X_test_scaled.shape)

print("\n[SUCCESS] Preprocessing completed.")
print("[SUCCESS] Processed files saved.")