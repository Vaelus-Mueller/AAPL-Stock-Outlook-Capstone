from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Any

import pandas as pd
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_stock_data(config: dict[str, Any]) -> pd.DataFrame:
    """Download daily OHLCV data for the configured ticker."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to download stock data: pip install yfinance") from exc

    ticker = config["project"]["supported_ticker"]
    start_date = config["data"]["start_date"]
    end_date = config["data"].get("end_date") or date.today().isoformat()

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Ticker"] = ticker
    return data


def save_stock_data(data: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)


def main(config_path: str) -> None:
    config = load_config(config_path)
    data = download_stock_data(config)
    save_stock_data(data, config["data"]["raw_path"])
    print(f"Saved {len(data)} rows to {config['data']['raw_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download daily AAPL market data.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
