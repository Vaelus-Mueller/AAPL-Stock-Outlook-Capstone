from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_stock_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stock data not found at {path}. Run src/download_data.py first.")
    return pd.read_csv(path)


def clean_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of daily OHLCV data without mutating the input."""
    cleaned = data.copy(deep=True)
    missing = REQUIRED_COLUMNS - set(cleaned.columns)
    if missing:
        raise ValueError(f"Missing required stock columns: {sorted(missing)}")

    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()
    cleaned = cleaned.dropna(subset=numeric_columns).reset_index(drop=True)
    cleaned = cleaned[cleaned["Volume"] >= 0].reset_index(drop=True)
    return cleaned


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)


def create_features(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create model-ready features and labels using only current/past values."""
    features = clean_stock_data(data)
    price_column = config["features"].get("price_column", "Adj Close")
    horizon = int(config["project"]["forecast_horizon_days"])

    features["daily_return"] = features[price_column].pct_change()
    features["intraday_range"] = (features["High"] - features["Low"]) / features["Open"]
    features["close_to_open"] = (features["Close"] - features["Open"]) / features["Open"]
    features["volume_change"] = features["Volume"].pct_change()

    for lag in config["features"]["lag_days"]:
        features[f"return_lag_{lag}"] = features["daily_return"].shift(lag)

    for window in config["features"]["moving_average_windows"]:
        moving_average = features[price_column].rolling(window=window, min_periods=window).mean()
        features[f"ma_ratio_{window}"] = features[price_column] / moving_average - 1
        features[f"volume_ratio_{window}"] = features["Volume"] / features["Volume"].rolling(window=window, min_periods=window).mean() - 1

    for window in config["features"]["volatility_windows"]:
        features[f"volatility_{window}"] = features["daily_return"].rolling(window=window, min_periods=window).std()

    features["rsi"] = calculate_rsi(features[price_column], int(config["features"]["rsi_window"]))
    features["future_return"] = features[price_column].shift(-horizon) / features[price_column] - 1
    features["target_up"] = (features["future_return"] > 0).astype(int)

    features = features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return features


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {
        "Date",
        "Ticker",
        "target_up",
        "future_return",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    }
    return [column for column in data.columns if column not in excluded]


def build_latest_feature_row(raw_data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    processed = create_features(raw_data, config)
    feature_columns = get_feature_columns(processed)
    return processed[feature_columns].tail(1)


def save_features(features: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)


def main(config_path: str) -> None:
    config = load_config(config_path)
    raw = load_stock_data(config["data"]["raw_path"])
    features = create_features(raw, config)
    save_features(features, config["data"]["processed_path"])
    print(f"Saved {len(features)} feature rows to {config['data']['processed_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AAPL stock data.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
