import pandas as pd
import joblib
import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ==========================================
# File paths
# ==========================================

X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

MODEL_PATH = "models/random_forest.pkl"

METRICS_PATH = "outputs/metrics.json"
CONFUSION_PATH = "outputs/confusion_matrix.png"
PREDICTIONS_PATH = "outputs/predictions.csv"


# ==========================================
# Create output folder
# ==========================================

os.makedirs("outputs", exist_ok=True)


# ==========================================
# Load test data
# ==========================================

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print("Testing data shape:", X_test.shape)


# ==========================================
# Load trained Random Forest
# ==========================================

model = joblib.load(MODEL_PATH)

print("Loaded model:", MODEL_PATH)


# ==========================================
# Make predictions
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# Calculate metrics
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# Display results
# ==========================================

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)


# ==========================================
# Classification report
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# Save confusion matrix
# ==========================================

plt.figure()

plt.imshow(cm)

plt.title("Stroke Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.xticks(
    [0, 1],
    ["No Stroke", "Stroke"]
)

plt.yticks(
    [0, 1],
    ["No Stroke", "Stroke"]
)

plt.savefig(
    CONFUSION_PATH,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# Save metrics
# ==========================================

metrics = {

    "model": "Random Forest",

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "roc_auc": roc_auc
}

with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================
# Save predictions
# ==========================================

predictions = pd.DataFrame({

    "actual_stroke": y_test,

    "predicted_stroke": y_pred,

    "stroke_probability": y_probability

})

predictions.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ==========================================
# Final message
# ==========================================

print("\nEvaluation completed successfully.")

print("\nSaved files:")
print("Metrics:", METRICS_PATH)
print("Confusion Matrix:", CONFUSION_PATH)
print("Predictions:", PREDICTIONS_PATH)
