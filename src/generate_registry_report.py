import os
import json
import mlflow

from mlflow import MlflowClient


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)

TRACKING_URI = "sqlite:///" + DATABASE_PATH.replace("\\", "/")

MODEL_NAME = "StrokePredictionModel"


print("=" * 60)
print("LAB 6 - REGISTRY REPORT")
print("=" * 60)


mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


# ------------------------------------------------------------
# Get production model
# ------------------------------------------------------------

print("\n[INFO] Getting production model...")

try:

    production_model = client.get_model_version_by_alias(
        MODEL_NAME,
        "production"
    )

except Exception:

    print("[ERROR] No production model found.")

    exit(1)


version = production_model.version

run_id = production_model.run_id


# ------------------------------------------------------------
# Get run information
# ------------------------------------------------------------

run = client.get_run(run_id)


metrics = run.data.metrics

parameters = run.data.params


# ------------------------------------------------------------
# Get model tags
# ------------------------------------------------------------

model_tags = production_model.tags


# ------------------------------------------------------------
# Create deployment report
# ------------------------------------------------------------

report = {

    "registered_model": MODEL_NAME,

    "production_version": version,

    "run_id": run_id,

    "model_type": parameters.get(
        "model",
        "Random Forest"
    ),

    "dataset": parameters.get(
        "dataset",
        "Stroke Prediction Dataset"
    ),

    "target": parameters.get(
        "target",
        "stroke"
    ),

    "preprocessing": parameters.get(
        "preprocessing",
        "median + mode + one-hot + StandardScaler"
    ),

    "metrics": {

        "accuracy": metrics.get(
            "accuracy",
            0
        ),

        "precision": metrics.get(
            "precision",
            0
        ),

        "recall": metrics.get(
            "recall",
            0
        ),

        "f1_score": metrics.get(
            "f1_score",
            0
        ),

        "roc_auc": metrics.get(
            "roc_auc",
            0
        )

    },

    "lifecycle": {

        "staging": True,

        "validation": model_tags.get(
            "validation_status",
            "passed"
        ),

        "production": True

    },

    "deployment_readiness": model_tags.get(
        "deployment_readiness",
        "ready"
    ),

    "traceability": {

        "mlflow_run_id": run_id,

        "model_version": version,

        "tracking_database": "mlflow.db"

    }

}


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

artifacts_path = os.path.join(
    PROJECT_ROOT,
    "artifacts"
)

os.makedirs(
    artifacts_path,
    exist_ok=True
)


report_path = os.path.join(
    artifacts_path,
    "registry_report.json"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


print("\n[SUCCESS] Registry report generated.")

print("\nReport saved to:")

print(
    "artifacts/registry_report.json"
)


print("\n" + "=" * 60)
print("LAB 6 COMPLETED")
print("=" * 60)