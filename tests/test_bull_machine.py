import pandas as pd
from bull_machine.core.types import Bar, Series
from bull_machine.io.feeders import load_csv_to_series
from bull_machine.modules.wyckoff.analyzer import analyze
from bull_machine.signals.gating import _compute_dynamic_ttl_bars


def test_ttl_bounds_enforcement():
    series = Series(
        bars=[Bar(ts=i, open=100 + i, high=101 + i, low=99 + i, close=100 + i, volume=0) for i in range(50)],
        timeframe="1h",
        symbol="TEST",
    )
    config = {"risk": {"ttl_bars": 18, "ttl_dynamic": {"min": 5, "max": 25, "atr_period": 14}}}
    ttl = _compute_dynamic_ttl_bars(series, None, config)
    assert 5 <= ttl <= 25


def test_insufficient_data_wyckoff():
    series = Series(
        bars=[Bar(ts=i, open=100, high=101, low=99, close=100, volume=0) for i in range(5)],
        timeframe="1h",
        symbol="T",
    )
    config = {"wyckoff": {"lookback_bars": 50}}
    result = analyze(series, config, {})
    assert result.regime == "neutral"
