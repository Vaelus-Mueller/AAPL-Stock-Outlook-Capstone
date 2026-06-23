import pandas as pd

from src.preprocess import calculate_rsi, clean_stock_data, create_features, get_feature_columns


def sample_stock_data(rows=80):
    dates = pd.date_range("2023-01-01", periods=rows, freq="B")
    base = pd.Series(range(rows), dtype=float) * 0.2 + 150
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": base,
            "High": base + 2,
            "Low": base - 2,
            "Close": base + 0.5,
            "Adj Close": base + 0.5,
            "Volume": 1_000_000 + pd.Series(range(rows)) * 1000,
            "Ticker": "AAPL",
        }
    )


def sample_config():
    return {
        "project": {"forecast_horizon_days": 5},
        "features": {
            "price_column": "Adj Close",
            "lag_days": [1, 2, 3, 5, 10],
            "moving_average_windows": [5, 10, 20],
            "volatility_windows": [5, 10, 20],
            "rsi_window": 14,
        },
    }


def test_clean_stock_data_handles_missing_values():
    data = sample_stock_data()
    data.loc[3, "Close"] = None
    cleaned = clean_stock_data(data)
    assert cleaned["Close"].isna().sum() == 0


def test_create_features_adds_encoded_numeric_target():
    features = create_features(sample_stock_data(), sample_config())
    assert set(features["target_up"].unique()) <= {0, 1}
    assert "rsi" in features.columns
    assert "ma_ratio_20" in features.columns


def test_rsi_stays_in_expected_range():
    rsi = calculate_rsi(sample_stock_data()["Adj Close"], window=14)
    assert rsi.between(0, 100).all()


def test_preprocessing_does_not_modify_original_dataframe():
    data = sample_stock_data()
    original = data.copy(deep=True)
    create_features(data, sample_config())
    pd.testing.assert_frame_equal(data, original)


def test_feature_columns_exclude_target_and_prices():
    features = create_features(sample_stock_data(), sample_config())
    feature_columns = get_feature_columns(features)
    assert "target_up" not in feature_columns
    assert "future_return" not in feature_columns
    assert "Adj Close" not in feature_columns
    assert "volume_change" in feature_columns
