from pathlib import Path
import json

import mlflow
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("diabetes.csv")
TARGET_COLUMN = "Outcome"
EXPERIMENT_NAME = "Diabetes_Prediction"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    test_size = 0.2
    random_state = 42
    max_iter = 1000
    model_name = "LogisticRegression"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=max_iter)),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }

    params = {
        "model_name": model_name,
        "dataset": "diabetes.csv",
        "target_column": TARGET_COLUMN,
        "test_size": test_size,
        "random_state": random_state,
        "max_iter": max_iter,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logistic_regression_diabetes") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        print("MLflow run_id:", run.info.run_id)
        print(json.dumps({**params, **metrics}, indent=2))


if __name__ == "__main__":
    main()
