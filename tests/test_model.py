import pandas as pd

from src.preprocess import create_features, get_feature_columns
from src.train import build_pipeline, time_series_split


def sample_stock_data(rows=120):
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    values = []
    price = 150.0
    for index in range(rows):
        price += 0.4 if index % 7 not in (0, 1) else -0.8
        values.append(price)
    close = pd.Series(values)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + (pd.Series(range(rows)) % 10) * 5000,
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


def test_model_prediction_shape_and_type():
    features = create_features(sample_stock_data(), sample_config())
    columns = get_feature_columns(features)
    X_train, X_test, y_train, _, _, _ = time_series_split(features, columns, test_size=0.2)
    model = build_pipeline(
        {
            "algorithm": "logistic_regression",
            "params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        }
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert predictions.shape == (X_test.shape[0],)
    assert set(predictions) <= {0, 1}


def test_model_meets_minimum_sample_performance():
    features = create_features(sample_stock_data(), sample_config())
    columns = get_feature_columns(features)
    X_train, X_test, y_train, y_test, _, _ = time_series_split(features, columns, test_size=0.2)
    model = build_pipeline(
        {
            "algorithm": "random_forest",
            "params": {"n_estimators": 50, "max_depth": 4, "random_state": 42},
        }
    )
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) >= 0.5
