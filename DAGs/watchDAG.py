from datetime import datetime
import json
import urllib.request

try:
    from airflow.sdk import DAG
except ImportError:
    from airflow import DAG

try:
    from airflow.sdk import Variable
except ImportError:
    from airflow.models import Variable

try:
    from airflow.providers.standard.operators.python import (
        ShortCircuitOperator,
        PythonOperator,
    )
except ImportError:
    from airflow.operators.python import (
        ShortCircuitOperator,
        PythonOperator,
    )

from airflow.providers.cncf.kubernetes.operators.job import KubernetesJobOperator
from kubernetes.client import models as k8s


NAMESPACE = "diabet"

IMAGE = "serhatsdocker/diabetprediction-training:latest"

DOCKER_HUB_TAG_URL = (
    "https://hub.docker.com/v2/repositories/"
    "serhatsdocker/diabetprediction-training/tags/latest"
)

VARIABLE_KEY = "diabetes_last_training_image_marker"

MLFLOW_TRACKING_URI = "http://mlflow-tracking.mlflow.svc.cluster.local"

MODEL_NAME = "diabetes-prediction-model"
MODEL_ALIAS = "best"
PROMOTION_METRIC = "f1"

KSERVE_NAMESPACE = "diabet"
INFERENCE_SERVICE_NAME = "diabetes-predictor"
MLFLOW_ARTIFACT_BUCKET = "mlflow"


def get_variable(key, default=None):
    try:
        return Variable.get(key, default=default)
    except TypeError:
        return Variable.get(key, default_var=default)
    except KeyError:
        return default


def fetch_latest_image_marker():
    with urllib.request.urlopen(DOCKER_HUB_TAG_URL, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    image_digests = sorted(
        {
            image.get("digest")
            for image in data.get("images", [])
            if image.get("digest")
        }
    )

    marker = (
        data.get("digest")
        or data.get("tag_last_pushed")
        or data.get("last_updated")
        or "|".join(image_digests)
    )

    if not marker:
        raise ValueError(
            f"Docker Hub image marker bulunamadı. Gelen keys: {list(data.keys())}"
        )

    return marker


def check_new_image(**context):
    latest_marker = fetch_latest_image_marker()
    previous_marker = get_variable(VARIABLE_KEY, default=None)

    print(f"Latest image marker: {latest_marker}")
    print(f"Previous image marker: {previous_marker}")

    if previous_marker is None:
        Variable.set(VARIABLE_KEY, latest_marker)
        print("İlk image baseline olarak kaydedildi. Training Job çalışmayacak.")
        return False

    if previous_marker == latest_marker:
        print("Yeni image yok. Training Job çalışmayacak.")
        return False

    context["ti"].xcom_push(key="latest_marker", value=latest_marker)
    print("Yeni image bulundu. Training pipeline çalışacak.")
    return True


def mark_image_processed(**context):
    latest_marker = context["ti"].xcom_pull(
        task_ids="check_new_image",
        key="latest_marker",
    )

    if not latest_marker:
        raise ValueError("latest_marker XCom içinde bulunamadı.")

    Variable.set(VARIABLE_KEY, latest_marker)
    print(f"Image işlendi olarak kaydedildi: {latest_marker}")


with DAG(
    dag_id="diabetes_image_watcher",
    start_date=datetime(2025, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "diabetes", "dockerhub", "training", "kserve"],
) as dag:

    check_new_image_task = ShortCircuitOperator(
        task_id="check_new_image",
        python_callable=check_new_image,
    )

    run_training_job = KubernetesJobOperator(
        task_id="run_diabetes_training_job",
        name="diabetes-training-{{ ts_nodash | lower }}",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        in_cluster=True,
        backoff_limit=1,
        wait_until_job_complete=True,
        get_logs=False,
        labels={
            "app": "diabetes-training",
            "created-by": "airflow",
            "pipeline-step": "train",
        },
        env_vars=[
            k8s.V1EnvVar(
                name="MLFLOW_TRACKING_URI",
                value=MLFLOW_TRACKING_URI,
            ),
            k8s.V1EnvVar(
                name="MODEL_NAME",
                value=MODEL_NAME,
            ),
            k8s.V1EnvVar(
                name="IMAGE_TAG",
                value=IMAGE,
            ),
        ],
        env_from=[
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(
                    name="mlflow-credentials"
                )
            )
        ],
    )

    promote_best_model = KubernetesJobOperator(
        task_id="promote_best_model",
        name="diabetes-promote-{{ ts_nodash | lower }}",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        in_cluster=True,
        backoff_limit=1,
        wait_until_job_complete=True,
        get_logs=False,
        cmds=["/bin/sh", "-c"],
        arguments=[
            "/opt/venv/bin/python src/promote_model.py"
        ],
        labels={
            "app": "diabetes-training",
            "created-by": "airflow",
            "pipeline-step": "promote",
        },
        env_vars=[
            k8s.V1EnvVar(
                name="MLFLOW_TRACKING_URI",
                value=MLFLOW_TRACKING_URI,
            ),
            k8s.V1EnvVar(
                name="MODEL_NAME",
                value=MODEL_NAME,
            ),
            k8s.V1EnvVar(
                name="MODEL_ALIAS",
                value=MODEL_ALIAS,
            ),
            k8s.V1EnvVar(
                name="PROMOTION_METRIC",
                value=PROMOTION_METRIC,
            ),
        ],
        env_from=[
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(
                    name="mlflow-credentials"
                )
            )
        ],
    )

    deploy_best_model_to_kserve = KubernetesJobOperator(
        task_id="deploy_best_model_to_kserve",
        name="diabetes-deploy-kserve-{{ ts_nodash | lower }}",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        in_cluster=True,
        service_account_name="diabetes-kserve-deployer",
        backoff_limit=1,
        wait_until_job_complete=True,
        get_logs=False,
        cmds=["/bin/sh", "-c"],
        arguments=[
            "/opt/venv/bin/python src/deploy_kserve.py"
        ],
        labels={
            "app": "diabetes-training",
            "created-by": "airflow",
            "pipeline-step": "deploy-kserve",
        },
        env_vars=[
            k8s.V1EnvVar(
                name="MLFLOW_TRACKING_URI",
                value=MLFLOW_TRACKING_URI,
            ),
            k8s.V1EnvVar(
                name="MODEL_NAME",
                value=MODEL_NAME,
            ),
            k8s.V1EnvVar(
                name="MODEL_ALIAS",
                value=MODEL_ALIAS,
            ),
            k8s.V1EnvVar(
                name="MLFLOW_ARTIFACT_BUCKET",
                value=MLFLOW_ARTIFACT_BUCKET,
            ),
            k8s.V1EnvVar(
                name="KSERVE_NAMESPACE",
                value=KSERVE_NAMESPACE,
            ),
            k8s.V1EnvVar(
                name="INFERENCE_SERVICE_NAME",
                value=INFERENCE_SERVICE_NAME,
            ),
        ],
        env_from=[
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(
                    name="mlflow-credentials"
                )
            )
        ],
    )

    mark_processed = PythonOperator(
        task_id="mark_image_processed",
        python_callable=mark_image_processed,
    )

    (
        check_new_image_task
        >> run_training_job
        >> promote_best_model
        >> deploy_best_model_to_kserve
        >> mark_processed
    )
