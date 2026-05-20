import os
from pathlib import PurePosixPath

import mlflow
from mlflow import MlflowClient
from kubernetes import client, config


MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-prediction-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "best")

MLFLOW_ARTIFACT_BUCKET = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow")

KSERVE_NAMESPACE = os.getenv("KSERVE_NAMESPACE", "diabet")
INFERENCE_SERVICE_NAME = os.getenv("INFERENCE_SERVICE_NAME", "diabetes-predictor")


def load_kubernetes_config():
    try:
        config.load_incluster_config()
        print("Loaded Kubernetes in-cluster config.")
    except config.ConfigException:
        config.load_kube_config()
        print("Loaded local kubeconfig.")


def mlflow_source_to_s3_uri(source: str) -> str:
    """
    Example input:
      mlflow-artifacts:/4/models/m-xxxx/artifacts/model.pkl

    KServe sklearn runtime wants the model directory, not the exact file:
      s3://mlflow/4/models/m-xxxx/artifacts
    """

    if source.startswith("s3://"):
        s3_uri = source
    elif source.startswith("mlflow-artifacts:/"):
        path = source.replace("mlflow-artifacts:/", "", 1).lstrip("/")
        s3_uri = f"s3://{MLFLOW_ARTIFACT_BUCKET}/{path}"
    else:
        raise ValueError(f"Unsupported MLflow model source URI: {source}")

    parsed = PurePosixPath(s3_uri)

    if parsed.name in {"model.pkl", "model.joblib", "model.pickle"}:
        s3_uri = str(parsed.parent)

    return s3_uri


def patch_kserve_storage_uri(storage_uri: str):
    load_kubernetes_config()

    api = client.CustomObjectsApi()

    patch_body = {
        "spec": {
            "predictor": {
                "model": {
                    "storageUri": storage_uri
                }
            }
        }
    }

    api.patch_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=KSERVE_NAMESPACE,
        plural="inferenceservices",
        name=INFERENCE_SERVICE_NAME,
        body=patch_body,
    )

    print(
        f"KServe InferenceService patched: "
        f"{KSERVE_NAMESPACE}/{INFERENCE_SERVICE_NAME}"
    )
    print(f"New storageUri: {storage_uri}")


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client_mlflow = MlflowClient()

    best_version = client_mlflow.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    print(f"Resolved alias: {MODEL_NAME}@{MODEL_ALIAS}")
    print(f"Version: {best_version.version}")
    print(f"Source: {best_version.source}")

    storage_uri = mlflow_source_to_s3_uri(best_version.source)

    print(f"Resolved KServe storageUri: {storage_uri}")

    patch_kserve_storage_uri(storage_uri)


if __name__ == "__main__":
    main()
