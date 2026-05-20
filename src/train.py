from pathlib import Path
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("diabetes.csv")
TARGET_COLUMN = "Outcome"
EXPERIMENT_NAME = "Diabetes_Prediction"
REGISTERED_MODEL_NAME = "diabetes-prediction-model"


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", model),
        ]
    )


def calculate_metrics(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    test_size = 0.25
    random_state = 48

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    candidate_models = [
        {
            "model_name": "LogisticRegression",
            "model": LogisticRegression(
                max_iter=2500,
                C=0.5,
                class_weight=None,
            ),
            "params": {
                "max_iter": 2500,
                "C": 0.5,
                "class_weight": "None",
            },
        },
        {
            "model_name": "LogisticRegression",
            "model": LogisticRegression(
                max_iter=2500,
                C=1.0,
                class_weight=None,
            ),
            "params": {
                "max_iter": 2500,
                "C": 1.0,
                "class_weight": "None",
            },
        },
        {
            "model_name": "LogisticRegression",
            "model": LogisticRegression(
                max_iter=2500,
                C=2.0,
                class_weight=None,
            ),
            "params": {
                "max_iter": 2500,
                "C": 2.0,
                "class_weight": "None",
            },
        },
        {
            "model_name": "RandomForestClassifier",
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=4,
                random_state=random_state,
                class_weight=None,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 4,
                "min_samples_leaf": 4,
                "class_weight": "None",
            },
        },
        {
            "model_name": "GradientBoostingClassifier",
            "model": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=2,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 2,
            },
        },
        {
            "model_name": "GradientBoostingClassifier",
            "model": GradientBoostingClassifier(
                n_estimators=160,
                learning_rate=0.04,
                max_depth=2,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": 160,
                "learning_rate": 0.04,
                "max_depth": 2,
            },
        },
    ]

    thresholds = [
        0.35,
        0.38,
        0.40,
        0.42,
        0.45,
        0.48,
        0.50,
        0.52,
        0.55,
        0.58,
        0.60,
    ]

    best_result = None

    for candidate in candidate_models:
        model = build_pipeline(candidate["model"])
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            metrics = calculate_metrics(y_test, predictions)

            result = {
                "model": model,
                "model_name": candidate["model_name"],
                "model_params": candidate["params"],
                "threshold": threshold,
                "metrics": metrics,
                "predictions": predictions,
            }

            if best_result is None or metrics["f1"] > best_result["metrics"]["f1"]:
                best_result = result

    if best_result is None:
        raise RuntimeError("No model candidate produced a result.")

    model = best_result["model"]
    predictions = best_result["predictions"]
    metrics = best_result["metrics"]

    params = {
        "model_name": best_result["model_name"],
        "dataset": "diabetes.csv",
        "target_column": TARGET_COLUMN,
        "test_size": test_size,
        "random_state": random_state,
        "threshold": best_result["threshold"],
        "selection_metric": "f1",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **best_result["model_params"],
    }

    git_commit = os.getenv("GIT_COMMIT")
    image_tag = os.getenv("IMAGE_TAG")

    if git_commit:
        params["git_commit"] = git_commit

    if image_tag:
        params["image_tag"] = image_tag

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_test_for_signature = X_test.astype(float)
    input_example = X_test_for_signature.head(3)
    signature = infer_signature(X_test_for_signature, predictions)

    with mlflow.start_run(run_name="best_diabetes_model_search") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("MLflow run_id:", run.info.run_id)
        print("MLflow model artifact path: model")
        print("Registered model name:", REGISTERED_MODEL_NAME)
        print("Selected model:", best_result["model_name"])
        print("Selected threshold:", best_result["threshold"])
        print(json.dumps({**params, **metrics}, indent=2))


if __name__ == "__main__":
    main()
