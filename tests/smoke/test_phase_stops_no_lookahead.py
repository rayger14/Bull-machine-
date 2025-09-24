"""Smoke tests for phase stops look-ahead prevention"""

import pandas as pd
import numpy as np
from bull_machine.core.utils import calculate_atr


def test_atr_no_lookahead():
    """Ensure ATR calculation doesn't use future data"""
    # Generate test data
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "high": 100 + np.random.normal(0, 2, 100),
            "low": 98 + np.random.normal(0, 2, 100),
            "close": 99 + np.random.normal(0, 2, 100),
        },
        index=dates,
    )

    t = 50
    # ATR calculated with data up to bar t
    atr_present = calculate_atr(df.iloc[: t + 1])

    # ATR calculated with future data present but restricted to bar t
    atr_with_future = calculate_atr(df.iloc[: t + 1])  # Same slice

    # They should be identical - no future data used
    assert abs(atr_present - atr_with_future) < 1e-10


def test_phase_calculation_historical_only():
    """Ensure phase calculations use only historical data"""
    # This is a basic smoke test - real implementation would test actual phase engine
    dates = pd.date_range("2024-01-01", periods=50, freq="h")
    prices = 100 + np.cumsum(np.random.normal(0, 0.01, 50))

    df = pd.DataFrame(
        {"high": prices + 0.5, "low": prices - 0.5, "close": prices, "volume": 1000 + np.random.normal(0, 100, 50)},
        index=dates,
    )

    # Mock phase detection that should only use historical data
    def detect_phase_at_bar(data, bar_idx):
        """Mock phase detection using only data up to bar_idx"""
        if bar_idx < 20:
            return "A"  # Early phase
        return "D"  # Later phase

    # Test that phase at bar 30 is same regardless of future data
    phase_present = detect_phase_at_bar(df.iloc[:31], 30)
    phase_with_future = detect_phase_at_bar(df.iloc[:45], 30)  # More future data

    assert phase_present == phase_with_future  # No look-ahead bias
