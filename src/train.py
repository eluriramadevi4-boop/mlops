import pandas as pd
import joblib
import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# File paths
# ==========================================

X_TRAIN_PATH = "data/processed/X_train.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"

MODEL_PATH = "models/stroke_model.pkl"
RESULT_PATH = "outputs/model_comparison.csv"


# ==========================================
# Create required folders
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================
# Load processed data
# ==========================================

X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

print("Training data shape:", X_train.shape)
print("Target shape:", y_train.shape)


# ==========================================
# Define models
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}


# ==========================================
# Train models
# ==========================================

trained_models = []

for model_name, model in models.items():

    print("\nTraining:", model_name)

    model.fit(X_train, y_train)

    trained_models.append({
        "Model": model_name,
        "Model_Object": model
    })

    print(model_name, "training completed.")


# ==========================================
# Save models temporarily
# ==========================================

for item in trained_models:

    model_name = item["Model"]
    model = item["Model_Object"]

    safe_name = model_name.lower().replace(" ", "_")

    path = "models/" + safe_name + ".pkl"

    joblib.dump(model, path)

    print("Saved:", path)


# ==========================================
# Save training information
# ==========================================

training_info = {
    "dataset": "Stroke Prediction Dataset",
    "target": "stroke",
    "random_state": 42,
    "models": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]
}

with open(
    "artifacts/training_information.json",
    "w"
) as file:

    json.dump(
        training_info,
        file,
        indent=4
    )


# ==========================================
# Message
# ==========================================

print("\nModel training completed successfully.")
print("Models have been saved in the models folder.")