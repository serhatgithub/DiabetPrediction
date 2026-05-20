import os
from typing import Optional

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-prediction-model")
ALIAS_NAME = os.getenv("MODEL_ALIAS", "best")
METRIC_NAME = os.getenv("PROMOTION_METRIC", "f1")


def get_metric_for_model_version(client: MlflowClient, version) -> Optional[float]:
    if not version.run_id:
        return None

    run = client.get_run(version.run_id)
    metric_value = run.data.metrics.get(METRIC_NAME)

    if metric_value is None:
        return None

    return float(metric_value)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")

    if not versions:
        raise RuntimeError(f"No model versions found for model: {MODEL_NAME}")

    scored_versions = []

    for version in versions:
        metric_value = get_metric_for_model_version(client, version)

        if metric_value is None:
            print(
                f"Skipping version {version.version}: "
                f"metric '{METRIC_NAME}' not found"
            )
            continue

        scored_versions.append(
            {
                "version": str(version.version),
                "run_id": version.run_id,
                "source": version.source,
                "metric": metric_value,
            }
        )

    if not scored_versions:
        raise RuntimeError(
            f"No model versions have metric '{METRIC_NAME}'. "
            "Promotion cannot continue."
        )

    best_candidate = max(scored_versions, key=lambda item: item["metric"])

    try:
        current_best = client.get_model_version_by_alias(
            MODEL_NAME,
            ALIAS_NAME,
        )
        current_best_metric = get_metric_for_model_version(
            client,
            current_best,
        )
    except MlflowException:
        current_best = None
        current_best_metric = None

    print("Best candidate:")
    print(f"  version: {best_candidate['version']}")
    print(f"  {METRIC_NAME}: {best_candidate['metric']}")
    print(f"  source: {best_candidate['source']}")

    if current_best:
        print("Current best:")
        print(f"  version: {current_best.version}")
        print(f"  {METRIC_NAME}: {current_best_metric}")

    if (
        current_best is not None
        and current_best_metric is not None
        and current_best_metric >= best_candidate["metric"]
    ):
        print("Current best is equal or better. Alias will not be changed.")
        return

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS_NAME,
        version=best_candidate["version"],
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=best_candidate["version"],
        key="promotion_status",
        value="best",
    )

    print(
        f"Alias updated: {MODEL_NAME}@{ALIAS_NAME} "
        f"-> version {best_candidate['version']}"
    )


if __name__ == "__main__":
    main()
