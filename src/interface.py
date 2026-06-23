from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

try:
    from src.download_data import download_stock_data
    from src.preprocess import build_latest_feature_row, load_config, load_stock_data
except ImportError:
    from download_data import download_stock_data
    from preprocess import build_latest_feature_row, load_config, load_stock_data


SUPPORTED_COMPANY_ALIASES = {
    "AAPL": "AAPL",
    "APPLE": "AAPL",
    "APPLE INC": "AAPL",
}

COMMON_UNSUPPORTED_TICKERS = {"TSLA", "NVDA", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "ARCC"}
UNSUPPORTED_COMPANY_ALIASES = {
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "ARES CAPITAL": "ARCC",
    "ARES CAPITOL": "ARCC",
}


@dataclass
class ParsedQuery:
    ticker: str | None
    horizon_days: int | None
    intent: str
    missing_fields: list[str]
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_fields and not self.errors


def fallback_parse_query(query: str, supported_ticker: str = "AAPL", default_horizon: int = 5) -> ParsedQuery:
    normalized = query.upper()
    ticker = None
    errors: list[str] = []

    for alias, canonical in SUPPORTED_COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            ticker = canonical
            break

    found_unsupported = sorted(symbol for symbol in COMMON_UNSUPPORTED_TICKERS if re.search(rf"\b{symbol}\b", normalized))
    for alias, symbol in UNSUPPORTED_COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            found_unsupported.append(symbol)
    if found_unsupported and ticker is None:
        unsupported = sorted(set(found_unsupported))[0]
        errors.append(f"Unsupported ticker: {unsupported}. This model is trained for {supported_ticker} only.")

    horizon_days = default_horizon
    explicit_day_match = re.search(r"NEXT\s+(\d+)\s+(TRADING\s+)?DAYS?", normalized)
    week_match = re.search(r"NEXT\s+WEEK|COMING\s+WEEK|5\s+TRADING\s+DAYS?", normalized)
    month_match = re.search(r"NEXT\s+MONTH|COMING\s+MONTH", normalized)

    if explicit_day_match:
        horizon_days = int(explicit_day_match.group(1))
    elif week_match:
        horizon_days = 5
    elif month_match:
        horizon_days = 21

    if horizon_days != default_horizon:
        errors.append(f"Unsupported horizon: {horizon_days} trading days. This model supports {default_horizon} trading days.")

    missing_fields = []
    if ticker is None and not errors:
        ticker = supported_ticker

    return ParsedQuery(
        ticker=ticker,
        horizon_days=horizon_days,
        intent="stock_outlook",
        missing_fields=missing_fields,
        errors=errors,
    )


def parse_query_with_llm(query: str, config: dict[str, Any]) -> ParsedQuery:
    api_key = os.getenv(config["llm"]["api_key_env"])
    if not api_key:
        return fallback_parse_query(
            query,
            supported_ticker=config["project"]["supported_ticker"],
            default_horizon=int(config["project"]["forecast_horizon_days"]),
        )

    system_prompt = (
        "Extract stock forecast request fields as JSON only. "
        "Supported ticker is AAPL. Required keys: ticker, horizon_days, intent. "
        "If ticker is missing, use null. If unsupported, return the requested ticker."
    )
    payload = {
        "model": config["llm"]["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
    }
    response = requests.post(
        config["llm"]["api_base_url"],
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return fallback_parse_query(
            query,
            supported_ticker=config["project"]["supported_ticker"],
            default_horizon=int(config["project"]["forecast_horizon_days"]),
        )

    ticker = parsed.get("ticker")
    horizon_days = parsed.get("horizon_days")
    fallback = fallback_parse_query(
        query,
        supported_ticker=config["project"]["supported_ticker"],
        default_horizon=int(config["project"]["forecast_horizon_days"]),
    )
    if ticker:
        fallback.ticker = str(ticker).upper()
    if horizon_days:
        fallback.horizon_days = int(horizon_days)
    if fallback.ticker != config["project"]["supported_ticker"]:
        fallback.errors.append(f"Unsupported ticker: {fallback.ticker}. This model is trained for AAPL only.")
    if fallback.horizon_days != int(config["project"]["forecast_horizon_days"]):
        fallback.errors.append(
            f"Unsupported horizon: {fallback.horizon_days} trading days. "
            f"This model supports {config['project']['forecast_horizon_days']} trading days."
        )
    return fallback


def load_model_artifact(model_path: str) -> dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run src/train.py first.")
    return joblib.load(model_path)


def estimate_expected_return(up_probability: float, recent_returns: pd.Series) -> float:
    recent_mean = float(recent_returns.tail(20).mean())
    recent_volatility = float(recent_returns.tail(20).std())
    directional_adjustment = (up_probability - 0.5) * max(recent_volatility, 0.0001) * math.sqrt(5)
    return recent_mean * 5 + directional_adjustment


def build_probability_curve(expected_return: float, std: float, points: int = 121) -> pd.DataFrame:
    std = max(float(std), 0.0001)
    x_values = np.linspace(expected_return - 3 * std, expected_return + 3 * std, points)
    density = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - expected_return) / std) ** 2)
    return pd.DataFrame({"return": x_values, "probability_density": density})


def make_prediction(query: str, config_path: str = "configs/config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    parsed = parse_query_with_llm(query, config)
    if not parsed.is_valid:
        return {
            "status": "needs_clarification",
            "parsed": parsed,
            "message": clarification_message(parsed, config),
        }

    artifact = load_model_artifact(config["training"]["model_output_path"])
    try:
        raw_data = load_stock_data(config["data"]["raw_path"])
    except FileNotFoundError:
        raw_data = download_stock_data(config)

    latest_features = build_latest_feature_row(raw_data, config)
    latest_features = latest_features[artifact["feature_columns"]]
    model = artifact["model"]
    up_probability = float(model.predict_proba(latest_features)[0, 1])
    prediction = int(up_probability >= 0.5)
    price_column = config["features"].get("price_column", "Adj Close")
    recent_returns = pd.to_numeric(raw_data[price_column], errors="coerce").pct_change().dropna()
    expected_return = estimate_expected_return(up_probability, recent_returns)
    residual_std = float(artifact.get("residual_std", recent_returns.tail(60).std() * math.sqrt(5)))
    curve = build_probability_curve(expected_return, residual_std)

    return {
        "status": "ok",
        "parsed": parsed,
        "ticker": parsed.ticker,
        "horizon_days": parsed.horizon_days,
        "up_probability": up_probability,
        "prediction": prediction,
        "expected_return": expected_return,
        "uncertainty_std": residual_std,
        "curve": curve,
        "metrics": artifact.get("metrics", {}),
        "run_id": artifact.get("run_id"),
        "message": generate_response(query, up_probability, expected_return, residual_std, config),
    }


def clarification_message(parsed: ParsedQuery, config: dict[str, Any]) -> str:
    if parsed.errors:
        return " ".join(parsed.errors)
    if "ticker" in parsed.missing_fields:
        return f"Please include AAPL or Apple in your question. This project is trained for {config['project']['supported_ticker']} only."
    return "I need a clearer AAPL stock outlook question before running the model."


def generate_response(
    query: str,
    up_probability: float,
    expected_return: float,
    uncertainty_std: float,
    config: dict[str, Any],
) -> str:
    api_key = os.getenv(config["llm"]["api_key_env"])
    fallback = (
        f"For AAPL over the next {config['project']['forecast_horizon_days']} trading days, "
        f"the model estimates a {up_probability:.1%} probability of closing higher. "
        f"The expected return estimate is {expected_return:.2%}, with a typical uncertainty band of "
        f"about +/- {uncertainty_std:.2%}. This is an educational probabilistic forecast, not financial advice."
    )
    if not api_key:
        return fallback

    payload = {
        "model": config["llm"]["model"],
        "messages": [
            {
                "role": "system",
                "content": "Explain model stock forecasts clearly, briefly, with caveats. Do not give financial advice.",
            },
            {
                "role": "user",
                "content": (
                    f"User query: {query}\n"
                    f"Probability AAPL closes higher: {up_probability:.3f}\n"
                    f"Expected 5-day return: {expected_return:.4f}\n"
                    f"Uncertainty std: {uncertainty_std:.4f}"
                ),
            },
        ],
        "temperature": 0.2,
    }
    try:
        response = requests.post(
            config["llm"]["api_base_url"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException:
        return fallback
