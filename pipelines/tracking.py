import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# Project root
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

os.chdir(PROJECT_ROOT)


# ==========================================
# File paths
# ==========================================

X_TRAIN_PATH = "data/processed/X_train.csv"
X_TEST_PATH = "data/processed/X_test.csv"

Y_TRAIN_PATH = "data/processed/y_train.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

METADATA_PATH = "data/processed/dataset_metadata.json"


# ==========================================
# MLflow SQLite tracking
# ==========================================

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)

TRACKING_URI = "sqlite:///" + DATABASE_PATH.replace("\\", "/")

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(
    "Stroke Prediction"
)


# ==========================================
# Load processed data
# ==========================================

X_train = pd.read_csv(
    X_TRAIN_PATH
)

X_test = pd.read_csv(
    X_TEST_PATH
)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze()

y_test = pd.read_csv(
    Y_TEST_PATH
).squeeze()


# ==========================================
# Display dataset information
# ==========================================

print("=" * 60)
print("LAB 4 - MLFLOW EXPERIMENT TRACKING")
print("STROKE PREDICTION PROJECT")
print("=" * 60)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)


# ==========================================
# Load metadata
# ==========================================

with open(
    METADATA_PATH,
    "r"
) as file:

    metadata = json.load(file)


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
# Run MLflow experiments
# ==========================================

for model_name, model in models.items():

    print("\n")
    print("=" * 60)
    print("Running:", model_name)
    print("=" * 60)

    with mlflow.start_run(
        run_name=model_name
    ):

        # ==================================
        # Log common parameters
        # ==================================

        mlflow.log_param(
            "dataset",
            "Stroke Prediction Dataset"
        )

        mlflow.log_param(
            "target_column",
            "stroke"
        )

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "scaling",
            "StandardScaler"
        )

        mlflow.log_param(
            "categorical_encoding",
            "One-Hot Encoding"
        )

        mlflow.log_param(
            "missing_value_strategy",
            "Median/Mode"
        )

        mlflow.log_param(
            "number_of_features",
            X_train.shape[1]
        )

        # ==================================
        # Log model-specific parameters
        # ==================================

        if model_name == "Logistic Regression":

            mlflow.log_param(
                "max_iter",
                1000
            )

        elif model_name == "Decision Tree":

            mlflow.log_param(
                "max_depth",
                5
            )

        elif model_name == "Random Forest":

            mlflow.log_param(
                "n_estimators",
                200
            )

            mlflow.log_param(
                "class_weight",
                "balanced"
            )

        # ==================================
        # Train model
        # ==================================

        model.fit(
            X_train,
            y_train
        )

        # ==================================
        # Predictions
        # ==================================

        y_pred = model.predict(
            X_test
        )

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        # ==================================
        # Calculate metrics
        # ==================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

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

        # ==================================
        # Log metrics
        # ==================================

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

        # ==================================
        # Log dataset metadata
        # ==================================

        mlflow.log_artifact(
            METADATA_PATH
        )

        # ==================================
        # Log trained model
        # ==================================

        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
        # ==================================
        # Display results
        # ==================================

        print("\nMetrics")

        print(
            "Accuracy :",
            round(accuracy, 4)
        )

        print(
            "Precision:",
            round(precision, 4)
        )

        print(
            "Recall   :",
            round(recall, 4)
        )

        print(
            "F1 Score :",
            round(f1, 4)
        )

        print(
            "ROC-AUC  :",
            round(roc_auc, 4)
        )

        print("\nMLflow run completed successfully.")


# ==========================================
# Completion
# ==========================================

print("\n")
print("=" * 60)
print("LAB 4 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nExperiment:")
print("Stroke Prediction")

print("\nTracking database:")
print("mlflow.db")

print("\nThree experiments were tracked:")
print("1. Logistic Regression")
print("2. Decision Tree")
print("3. Random Forest")