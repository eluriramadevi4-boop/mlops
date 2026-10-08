import os
import json
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)

TRACKING_URI = "sqlite:///" + DATABASE_PATH.replace("\\", "/")

MODEL_NAME = "StrokePredictionModel"

X_TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_train.csv"
)

X_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_test.csv"
)

Y_TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "y_train.csv"
)

Y_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "y_test.csv"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "dataset_metadata.json"
)


print("=" * 60)
print("LAB 6 - TRAIN AND REGISTER MODEL")
print("=" * 60)


# ------------------------------------------------------------
# MLflow setup
# ------------------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment("Stroke Prediction")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("\n[INFO] Loading processed data...")

X_train = __import__("pandas").read_csv(X_TRAIN_PATH)
X_test = __import__("pandas").read_csv(X_TEST_PATH)

y_train = __import__("pandas").read_csv(Y_TRAIN_PATH).iloc[:, 0]
y_test = __import__("pandas").read_csv(Y_TEST_PATH).iloc[:, 0]

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

print("\n[INFO] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("[SUCCESS] Model training completed.")


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

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


print("\n[INFO] Model Metrics")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ------------------------------------------------------------
# MLflow registration
# ------------------------------------------------------------

print("\n[INFO] Registering model...")

with mlflow.start_run() as run:

    mlflow.log_param(
        "model",
        "Random Forest"
    )

    mlflow.log_param(
        "n_estimators",
        200
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "dataset",
        "Stroke Prediction Dataset"
    )

    mlflow.log_param(
        "target",
        "stroke"
    )

    mlflow.log_param(
        "preprocessing",
        "median + mode + one-hot + StandardScaler"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    # Register model
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=MODEL_NAME,
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    run_id = run.info.run_id


print("\n[SUCCESS] Model registered.")

print("Model name:", MODEL_NAME)
print("Run ID    :", run_id)


# ------------------------------------------------------------
# Save training information
# ------------------------------------------------------------

os.makedirs(
    os.path.join(PROJECT_ROOT, "artifacts"),
    exist_ok=True
)

training_information = {

    "model_name": MODEL_NAME,

    "model_type": "Random Forest",

    "run_id": run_id,

    "n_estimators": 200,

    "random_state": 42,

    "metrics": {

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc

    },

    "dataset": "Stroke Prediction Dataset",

    "target": "stroke",

    "preprocessing": (
        "median + mode + one-hot encoding + StandardScaler"
    )

}


output_path = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "latest_registry_training.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        training_information,
        file,
        indent=4
    )


print("\n[SUCCESS] Training information saved.")