from __future__ import annotations

import argparse
import os
from typing import Any

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def compute_classification_metrics(y_true, y_pred, y_probability) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probability)
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def compute_baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    majority_class = int(y_train.mode().iloc[0])
    predictions = [majority_class] * len(y_test)
    probability = [float(majority_class)] * len(y_test)
    return {
        f"baseline_{name}": value
        for name, value in compute_classification_metrics(y_test, predictions, probability).items()
    }


def find_best_run(config: dict[str, Any]) -> pd.Series:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if experiment is None:
        raise ValueError(f"Experiment not found: {config['mlflow']['experiment_name']}")

    primary_metric = config["training"]["primary_metric"]
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{primary_metric} DESC"],
    )
    if runs.empty:
        raise ValueError("No MLflow runs found.")
    return runs.iloc[0]


def print_best_run(config_path: str) -> None:
    config = load_config(config_path)
    best_run = find_best_run(config)
    columns = [
        "run_id",
        "tags.mlflow.runName",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.roc_auc",
        "metrics.baseline_accuracy",
    ]
    available_columns = [column for column in columns if column in best_run.index]
    print("Best MLflow run:")
    print(best_run[available_columns])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best AAPL MLflow experiment run.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print_best_run(args.config)
