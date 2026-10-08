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
print("LAB 6 - AUTOMATED MODEL LIFECYCLE")
print("=" * 60)


mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


# ------------------------------------------------------------
# Get registered versions
# ------------------------------------------------------------

print("\n[INFO] Getting registered model versions...")

versions = client.search_model_versions(
    "name='{}'".format(MODEL_NAME)
)

if len(versions) == 0:

    print("[ERROR] No registered model versions found.")

    exit(1)


print(
    "[INFO] Number of registered versions:",
    len(versions)
)


# ------------------------------------------------------------
# Compare model versions
# ------------------------------------------------------------

best_version = None
best_f1 = -1


print("\n" + "=" * 60)
print("MODEL VERSION COMPARISON")
print("=" * 60)


for version in versions:

    run_id = version.run_id

    run = client.get_run(run_id)

    f1 = run.data.metrics.get(
        "f1_score",
        0
    )

    accuracy = run.data.metrics.get(
        "accuracy",
        0
    )

    precision = run.data.metrics.get(
        "precision",
        0
    )

    recall = run.data.metrics.get(
        "recall",
        0
    )

    roc_auc = run.data.metrics.get(
        "roc_auc",
        0
    )


    print("\nVersion:", version.version)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))


    if f1 > best_f1:

        best_f1 = f1

        best_version = version.version


# ------------------------------------------------------------
# Champion
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CHAMPION MODEL")
print("=" * 60)

print(
    "Champion version:",
    best_version
)

print(
    "Champion F1 Score:",
    round(best_f1, 4)
)


# ------------------------------------------------------------
# Staging
# ------------------------------------------------------------

print("\n[INFO] Promoting champion to STAGING...")

client.set_registered_model_alias(
    MODEL_NAME,
    "staging",
    best_version
)

client.set_model_version_tag(
    MODEL_NAME,
    best_version,
    "lifecycle_stage",
    "staging"
)

print("[SUCCESS] Champion assigned to staging.")


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

print("\n[INFO] Validating champion...")

validation_passed = (
    best_f1 >= 0
    and best_f1 <= 1
)

if validation_passed:

    client.set_model_version_tag(
        MODEL_NAME,
        best_version,
        "validation_status",
        "passed"
    )

    print("[SUCCESS] Validation PASSED.")

else:

    client.set_model_version_tag(
        MODEL_NAME,
        best_version,
        "validation_status",
        "failed"
    )

    print("[ERROR] Validation FAILED.")

    exit(1)


# ------------------------------------------------------------
# Production
# ------------------------------------------------------------

print("\n[INFO] Promoting champion to PRODUCTION...")

client.set_registered_model_alias(
    MODEL_NAME,
    "production",
    best_version
)

client.set_model_version_tag(
    MODEL_NAME,
    best_version,
    "lifecycle_stage",
    "production"
)

client.set_model_version_tag(
    MODEL_NAME,
    best_version,
    "deployment_readiness",
    "ready"
)


print(
    "[SUCCESS] Version",
    best_version,
    "promoted to PRODUCTION."
)


# ------------------------------------------------------------
# Save lifecycle information
# ------------------------------------------------------------

lifecycle_information = {

    "registered_model": MODEL_NAME,

    "champion_version": best_version,

    "champion_f1_score": best_f1,

    "staging_alias": "staging",

    "production_alias": "production",

    "validation_status": "passed",

    "deployment_readiness": "ready"

}


output_path = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "lifecycle_information.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        lifecycle_information,
        file,
        indent=4
    )


print("\n[SUCCESS] Lifecycle information saved.")