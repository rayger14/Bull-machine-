"""
Unit tests for v1.5.1 Core Trader ATR components
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import pytest
from bull_machine.core.numerics import safe_atr, safe_true_range
from bull_machine.strategy.position_sizing import atr_risk_size
from bull_machine.strategy.atr_exits import compute_exit_levels, maybe_trail_sl
from bull_machine.modules.regime_filter import regime_ok

def test_safe_atr_positive():
    """Test that ATR calculation returns positive values."""
    df = pd.DataFrame({
        "open": np.arange(100, 130),
        "high": np.arange(101, 131),
        "low": np.arange(99, 129),
        "close": np.arange(100, 130),
        "volume": [1000] * 30
    })

    atr = safe_atr(df, 14)
    assert float(atr.iloc[-1]) > 0
    assert not atr.isnull().any()

def test_safe_true_range():
    """Test True Range calculation."""
    df = pd.DataFrame({
        "open": [100, 102, 101],
        "high": [105, 106, 104],
        "low": [98, 100, 99],
        "close": [103, 101, 102],
        "volume": [1000] * 3
    })

    tr = safe_true_range(df)
    assert len(tr) == 3
    assert tr.iloc[0] == 7  # high - low = 105 - 98
    assert not tr.isnull().any()

def test_atr_risk_size_nonzero():
    """Test ATR-based position sizing returns positive values."""
    df = pd.DataFrame({
        "open": [100] * 30,
        "high": [101] * 30,
        "low": [99] * 30,
        "close": [100] * 30,
        "volume": [1000] * 30
    })

    risk_cfg = {
        "risk_pct": 0.005,
        "atr_window": 14,
        "sl_atr": 2.0
    }

    size = atr_risk_size(df, 10000.0, risk_cfg)
    assert size > 0

def test_atr_risk_size_scales_with_equity():
    """Test position sizing scales with equity."""
    df = pd.DataFrame({
        "open": [100] * 30,
        "high": [101] * 30,
        "low": [99] * 30,
        "close": [100] * 30,
        "volume": [1000] * 30
    })

    risk_cfg = {
        "risk_pct": 0.01,
        "atr_window": 14,
        "sl_atr": 2.0
    }

    size_10k = atr_risk_size(df, 10000.0, risk_cfg)
    size_20k = atr_risk_size(df, 20000.0, risk_cfg)

    assert size_20k > size_10k
    assert abs(size_20k / size_10k - 2.0) < 0.1  # Should be roughly 2x

def test_compute_exit_levels_long():
    """Test exit level calculation for long positions."""
    df = pd.DataFrame({
        "open": [100] * 30,
        "high": [101] * 30,
        "low": [99] * 30,
        "close": [100] * 30,
        "volume": [1000] * 30
    })

    risk_cfg = {
        "atr_window": 14,
        "sl_atr": 2.0,
        "tp_atr": 3.0
    }

    sl, tp = compute_exit_levels(df, "long", risk_cfg)

    assert tp > 100  # Take profit above current price
    assert sl < 100  # Stop loss below current price
    assert tp > sl   # TP should be above SL

def test_compute_exit_levels_short():
    """Test exit level calculation for short positions."""
    df = pd.DataFrame({
        "open": [100] * 30,
        "high": [101] * 30,
        "low": [99] * 30,
        "close": [100] * 30,
        "volume": [1000] * 30
    })

    risk_cfg = {
        "atr_window": 14,
        "sl_atr": 2.0,
        "tp_atr": 3.0
    }

    sl, tp = compute_exit_levels(df, "short", risk_cfg)

    assert tp < 100  # Take profit below current price (short)
    assert sl > 100  # Stop loss above current price (short)
    assert sl > tp   # SL should be above TP for shorts

def test_trailing_stop_long():
    """Test trailing stop for long positions."""
    df = pd.DataFrame({
        "open": [100] * 30,
        "high": [101] * 30,
        "low": [99] * 30,
        "close": [102] * 30,  # Price moved up
        "volume": [1000] * 30
    })

    risk_cfg = {
        "atr_window": 14,
        "trail_atr": 1.0
    }

    original_sl = 95.0
    new_sl = maybe_trail_sl(df, "long", original_sl, risk_cfg)

    assert new_sl >= original_sl  # Should only move stop up for longs

def test_trailing_stop_short():
    """Test trailing stop for short positions."""
    df = pd.DataFrame({
        "open": [100] * 30,
        "high": [101] * 30,
        "low": [99] * 30,
        "close": [98] * 30,  # Price moved down (favorable for short)
        "volume": [1000] * 30
    })

    risk_cfg = {
        "atr_window": 14,
        "trail_atr": 1.0
    }

    original_sl = 105.0
    new_sl = maybe_trail_sl(df, "short", original_sl, risk_cfg)

    assert new_sl <= original_sl  # Should only move stop down for shorts

def test_regime_ok_volume():
    """Test regime filter with volume conditions."""
    # Create data with good volume regime
    df = pd.DataFrame({
        "open": [100] * 120,
        "high": [101] * 120,
        "low": [99] * 120,
        "close": [100] * 120,
        "volume": [800] * 20 + [1200] * 100  # Recent volume higher
    })

    regime_cfg = {
        "vol_ratio_min": 1.2,
        "atr_pct_max": 0.10
    }

    assert regime_ok(df, "1D", regime_cfg) == True

def test_regime_ok_volatility():
    """Test regime filter with volatility conditions."""
    # Create data with high volatility (should fail)
    df = pd.DataFrame({
        "open": [100] * 120,
        "high": [150] * 120,  # Very high
        "low": [50] * 120,    # Very low
        "close": [100] * 120,
        "volume": [1200] * 120  # Good volume
    })

    regime_cfg = {
        "vol_ratio_min": 1.0,
        "atr_pct_max": 0.05  # 5% max volatility
    }

    assert regime_ok(df, "1D", regime_cfg) == False

def test_regime_ok_insufficient_data():
    """Test regime filter with insufficient data."""
    df = pd.DataFrame({
        "open": [100] * 50,  # Less than 100 bars
        "high": [101] * 50,
        "low": [99] * 50,
        "close": [100] * 50,
        "volume": [1000] * 50
    })

    regime_cfg = {
        "vol_ratio_min": 1.5,
        "atr_pct_max": 0.01
    }

    # Should return True (allow trading) when insufficient data
    assert regime_ok(df, "1D", regime_cfg) == True


if __name__ == "__main__":
    test_safe_atr_positive()
    print("✅ Safe ATR test passed")

    test_safe_true_range()
    print("✅ True Range test passed")

    test_atr_risk_size_nonzero()
    print("✅ ATR risk sizing test passed")

    test_atr_risk_size_scales_with_equity()
    print("✅ Risk sizing scaling test passed")

    test_compute_exit_levels_long()
    print("✅ Long exit levels test passed")

    test_compute_exit_levels_short()
    print("✅ Short exit levels test passed")

    test_trailing_stop_long()
    print("✅ Long trailing stop test passed")

    test_trailing_stop_short()
    print("✅ Short trailing stop test passed")

    test_regime_ok_volume()
    print("✅ Regime volume test passed")

    test_regime_ok_volatility()
    print("✅ Regime volatility test passed")

    test_regime_ok_insufficient_data()
    print("✅ Regime insufficient data test passed")

    print("\n🎯 All v1.5.1 Core Trader tests passed!")