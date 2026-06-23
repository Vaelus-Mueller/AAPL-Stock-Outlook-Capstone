from __future__ import annotations

import argparse
import json
import os
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.evaluate import compute_baseline_metrics, compute_classification_metrics
    from src.preprocess import create_features, get_feature_columns, load_config, load_stock_data, save_features
except ImportError:
    from evaluate import compute_baseline_metrics, compute_classification_metrics
    from preprocess import create_features, get_feature_columns, load_config, load_stock_data, save_features


def build_estimator(algorithm: str, params: dict[str, Any]):
    if algorithm == "logistic_regression":
        return LogisticRegression(**params)
    if algorithm == "random_forest":
        return RandomForestClassifier(**params)
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_pipeline(model_config: dict[str, Any]) -> Pipeline:
    algorithm = model_config["algorithm"]
    estimator = build_estimator(algorithm, model_config.get("params", {}))
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if algorithm == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def time_series_split(features: pd.DataFrame, feature_columns: list[str], test_size: float):
    split_index = int(len(features) * (1 - test_size))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("Not enough rows for the configured time-series split.")

    train_data = features.iloc[:split_index].copy()
    test_data = features.iloc[split_index:].copy()
    X_train = train_data[feature_columns]
    y_train = train_data["target_up"]
    X_test = test_data[feature_columns]
    y_test = test_data["target_up"]
    return X_train, X_test, y_train, y_test, train_data, test_data


def _predict_probability(model: Pipeline, X_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def _log_run(
    config: dict[str, Any],
    model_config: dict[str, Any],
    model: Pipeline,
    metrics: dict[str, float],
    feature_columns: list[str],
    data_rows: int,
) -> str:
    with mlflow.start_run(run_name=model_config["name"]) as run:
        mlflow.log_param("ticker", config["project"]["supported_ticker"])
        mlflow.log_param("forecast_horizon_days", config["project"]["forecast_horizon_days"])
        mlflow.log_param("data_source", config["data"]["data_source"])
        mlflow.log_param("data_rows", data_rows)
        mlflow.log_param("algorithm", model_config["algorithm"])
        for key, value in model_config.get("params", {}).items():
            mlflow.log_param(key, value)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_text(json.dumps(feature_columns, indent=2), artifact_file="feature_columns.json")
        return run.info.run_id


def train(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    raw_data = load_stock_data(config["data"]["raw_path"])
    features = create_features(raw_data, config)
    save_features(features, config["data"]["processed_path"])
    feature_columns = get_feature_columns(features)
    X_train, X_test, y_train, y_test, train_data, test_data = time_series_split(
        features,
        feature_columns,
        float(config["training"]["test_size"]),
    )

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    baseline_metrics = compute_baseline_metrics(y_train, y_test)
    run_results = []
    primary_metric = config["training"]["primary_metric"]

    for model_config in config["training"]["model_configs"]:
        model = build_pipeline(model_config)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = _predict_probability(model, X_test)
        metrics = compute_classification_metrics(y_test, predictions, probabilities)
        metrics.update(baseline_metrics)
        run_id = _log_run(config, model_config, model, metrics, feature_columns, len(features))
        run_results.append(
            {
                "run_id": run_id,
                "name": model_config["name"],
                "model": model,
                "metrics": metrics,
                "score": metrics[primary_metric],
            }
        )

    best = max(run_results, key=lambda result: result["score"])
    best_model = best["model"]
    best_probabilities = _predict_probability(best_model, X_test)
    probability_centered_returns = test_data["future_return"] - pd.Series(best_probabilities, index=test_data.index).sub(0.5) * 0.02
    residual_std = float(max(probability_centered_returns.std(), test_data["future_return"].std(), 0.0001))

    output_path = config["training"]["model_output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    artifact = {
        "model": best_model,
        "feature_columns": feature_columns,
        "metrics": best["metrics"],
        "run_id": best["run_id"],
        "ticker": config["project"]["supported_ticker"],
        "forecast_horizon_days": config["project"]["forecast_horizon_days"],
        "residual_std": residual_std,
        "last_training_date": str(features["Date"].max().date()),
    }
    joblib.dump(artifact, output_path)

    for metric_name, threshold in config["training"]["minimum_metrics"].items():
        if best["metrics"][metric_name] < threshold:
            raise RuntimeError(
                f"Best model {metric_name} below threshold: "
                f"{best['metrics'][metric_name]:.3f} < {threshold:.3f}"
            )

    print(f"Best run: {best['name']} ({best['run_id']})")
    for metric_name, metric_value in best["metrics"].items():
        print(f" - {metric_name}: {metric_value:.4f}")
    print(f"Saved best model artifact to {output_path}")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AAPL direction models with MLflow tracking.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
