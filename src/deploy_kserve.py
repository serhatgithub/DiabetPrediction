import os
from typing import Optional

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


def strip_model_file_if_needed(uri: str) -> str:
    for filename in ("model.pkl", "model.joblib", "model.pickle"):
        suffix = f"/{filename}"
        if uri.endswith(suffix):
            return uri[: -len(suffix)]

    return uri


def mlflow_source_to_s3_uri(
    source: str,
    experiment_id: Optional[str] = None,
) -> str:
    """
    Supported examples:

    1)
    mlflow-artifacts:/4/models/m-xxxx/artifacts/model.pkl
    ->
    s3://mlflow/4/models/m-xxxx/artifacts

    2)
    models:/m-xxxx
    ->
    s3://mlflow/<experiment_id>/models/m-xxxx/artifacts

    3)
    s3://mlflow/4/models/m-xxxx/artifacts/model.pkl
    ->
    s3://mlflow/4/models/m-xxxx/artifacts
    """

    if source.startswith("s3://"):
        return strip_model_file_if_needed(source)

    if source.startswith("mlflow-artifacts:/"):
        path = source.replace("mlflow-artifacts:/", "", 1).lstrip("/")
        s3_uri = f"s3://{MLFLOW_ARTIFACT_BUCKET}/{path}"
        return strip_model_file_if_needed(s3_uri)

    if source.startswith("models:/"):
        if not experiment_id:
            raise ValueError(
                "source is models:/... but experiment_id is missing. "
                "Cannot build MinIO S3 path."
            )

        model_id = source.replace("models:/", "", 1).strip("/")

        if not model_id.startswith("m-"):
            raise ValueError(f"Unexpected logged model id: {model_id}")

        return (
            f"s3://{MLFLOW_ARTIFACT_BUCKET}/"
            f"{experiment_id}/models/{model_id}/artifacts"
        )

    raise ValueError(f"Unsupported MLflow model source URI: {source}")


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

    mlflow_client = MlflowClient()

    best_version = mlflow_client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    print(f"Resolved alias: {MODEL_NAME}@{MODEL_ALIAS}")
    print(f"Version: {best_version.version}")
    print(f"Source: {best_version.source}")
    print(f"Run ID: {best_version.run_id}")

    experiment_id = None

    if best_version.run_id:
        run = mlflow_client.get_run(best_version.run_id)
        experiment_id = run.info.experiment_id
        print(f"Experiment ID: {experiment_id}")

    storage_uri = mlflow_source_to_s3_uri(
        source=best_version.source,
        experiment_id=experiment_id,
    )

    print(f"Resolved KServe storageUri: {storage_uri}")

    patch_kserve_storage_uri(storage_uri)


if __name__ == "__main__":
    main()
