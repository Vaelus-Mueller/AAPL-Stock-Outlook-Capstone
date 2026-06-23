from src.interface import build_probability_curve, fallback_parse_query


def test_parser_extracts_aapl_and_week_horizon():
    parsed = fallback_parse_query("What is the outlook for Apple next week?")
    assert parsed.ticker == "AAPL"
    assert parsed.horizon_days == 5
    assert parsed.is_valid


def test_parser_defaults_missing_ticker_to_aapl():
    parsed = fallback_parse_query("Will it go up next week?")
    assert parsed.ticker == "AAPL"
    assert parsed.horizon_days == 5
    assert parsed.is_valid


def test_parser_rejects_unsupported_ticker():
    parsed = fallback_parse_query("What is the outlook for ARCC next week?")
    assert not parsed.is_valid
    assert parsed.errors
    assert "Unsupported ticker" in parsed.errors[0]


def test_parser_rejects_unsupported_company_name():
    parsed = fallback_parse_query("Should I buy Tesla stock?")
    assert not parsed.is_valid
    assert parsed.errors
    assert "TSLA" in parsed.errors[0]


def test_probability_curve_has_expected_shape():
    curve = build_probability_curve(expected_return=0.02, std=0.03, points=51)
    assert list(curve.columns) == ["return", "probability_density"]
    assert len(curve) == 51
    assert curve["probability_density"].max() > curve["probability_density"].min()
