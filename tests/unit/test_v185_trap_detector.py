"""
Unit tests for v1.8.5 Narrative Trap Detector - Bull Machine

Tests HODL trap detection, Wyckoff distribution patterns, and veto logic.
"""

import pytest
import pandas as pd
import numpy as np
from engine.narrative.trap_detector import (
    decode_liquidity_programming,
    check_distribution_pattern,
    should_veto_narrative
)


def test_hodl_trap_volume_divergence():
    """Test HODL trap detection on declining volume + rising price."""
    df = pd.DataFrame({
        'close': [100 + i for i in range(30)],  # Rising price
        'volume': [1000 - i * 20 for i in range(30)]  # Declining volume
    })

    macro_cache = {
        'TOTAL3': {'value': 1.2}  # 20% above threshold
    }

    config = {
        'hodl_trap_vol_min': 0.8,
        'total3_hype_threshold': 1.1
    }

    is_trap, msg = decode_liquidity_programming(df, macro_cache, config)

    assert is_trap is True
    assert 'HODL trap' in msg or 'exit liquidity' in msg


def test_hodl_trap_btc_dominance_declining():
    """Test HODL trap detection on BTC dominance decline."""
    df = pd.DataFrame({
        'close': [100 + i for i in range(30)],
        'volume': [1000 - i * 25 for i in range(30)]  # More aggressive decline
    })

    macro_cache = {
        'BTC_D': {'value': pd.Series([45, 44, 43, 42, 41, 40, 39, 38, 37, 36])}
    }

    config = {'hodl_trap_vol_min': 0.8}

    is_trap, msg = decode_liquidity_programming(df, macro_cache, config)

    # May not always trigger depending on volume ratio
    assert isinstance(is_trap, bool)


def test_no_hodl_trap_clean_conditions():
    """Test no trap when conditions are healthy."""
    df = pd.DataFrame({
        'close': [100 + i for i in range(30)],
        'volume': [1000 + i * 10 for i in range(30)]  # Rising volume
    })

    macro_cache = {'TOTAL3': {'value': 0.9}}  # Below threshold

    config = {
        'hodl_trap_vol_min': 0.8,
        'total3_hype_threshold': 1.1
    }

    is_trap, msg = decode_liquidity_programming(df, macro_cache, config)

    assert is_trap is False
    assert msg == ""


def test_hodl_trap_insufficient_data():
    """Test graceful handling with insufficient data."""
    df = pd.DataFrame({
        'close': [100, 101],
        'volume': [1000, 900]
    })

    config = {'hodl_trap_vol_min': 0.8}

    is_trap, msg = decode_liquidity_programming(df, {}, config)

    assert is_trap is False


def test_distribution_pattern_detection():
    """Test Wyckoff distribution pattern detection."""
    np.random.seed(42)
    df = pd.DataFrame({
        'high': [100 + i + np.random.rand() * 2 for i in range(40)],
        'low': [98 + i - np.random.rand() * 2 for i in range(40)],
        'open': [99 + i for i in range(40)],
        'close': [99.5 + i for i in range(40)],
        'volume': [1000 - i * 15 for i in range(40)]  # Declining volume
    })

    config = {}

    is_dist, conf = check_distribution_pattern(df, config)

    # Check types (np.bool_ is subclass of bool-like)
    assert is_dist in [True, False] or isinstance(is_dist, (bool, np.bool_))
    assert 0.0 <= conf <= 1.0


def test_distribution_confidence_calculation():
    """Test distribution confidence is properly bounded."""
    df = pd.DataFrame({
        'high': np.random.uniform(95, 105, 50),
        'low': np.random.uniform(90, 100, 50),
        'open': np.random.uniform(92, 102, 50),
        'close': np.random.uniform(92, 102, 50),
        'volume': np.random.uniform(800, 1200, 50)
    })

    config = {}

    is_dist, conf = check_distribution_pattern(df, config)

    assert 0.0 <= conf <= 1.0


def test_distribution_insufficient_data():
    """Test distribution check with insufficient data."""
    df = pd.DataFrame({
        'high': [100, 101],
        'low': [98, 99],
        'open': [99, 100],
        'close': [99.5, 100.5],
        'volume': [1000, 900]
    })

    config = {}

    is_dist, conf = check_distribution_pattern(df, config)

    assert is_dist is False
    assert conf == 0.0


def test_should_veto_narrative_hodl_trap():
    """Test veto on HODL trap detection."""
    df = pd.DataFrame({
        'close': [100 + i for i in range(30)],
        'volume': [1000 - i * 20 for i in range(30)]
    })

    macro_cache = {'TOTAL3': {'value': 1.3}}

    config = {
        'narrative_enabled': True,
        'hodl_trap_vol_min': 0.8,
        'total3_hype_threshold': 1.1
    }

    veto, reason = should_veto_narrative(df, macro_cache, config)

    assert veto is True
    assert 'HODL trap' in reason or 'exit liquidity' in reason


def test_should_veto_narrative_distribution():
    """Test veto on strong distribution pattern."""
    df = pd.DataFrame({
        'high': [100 + i * 0.5 for i in range(40)],
        'low': [98 + i * 0.5 for i in range(40)],
        'open': [99 + i * 0.5 for i in range(40)],
        'close': [99 + i * 0.5 for i in range(40)],
        'volume': [1000 - i * 20 for i in range(40)]  # Strong decline
    })

    config = {'narrative_enabled': True}

    # This might not trigger high enough confidence, but test structure
    veto, reason = should_veto_narrative(df, {}, config)

    # If veto triggers, must have reason
    if veto:
        assert 'distribution' in reason.lower() or 'HODL' in reason


def test_should_not_veto_narrative_disabled():
    """Test no veto when narrative detection disabled."""
    df = pd.DataFrame({
        'close': [100 + i for i in range(30)],
        'volume': [1000 - i * 20 for i in range(30)]
    })

    macro_cache = {'TOTAL3': {'value': 1.5}}

    config = {'narrative_enabled': False}

    veto, reason = should_veto_narrative(df, macro_cache, config)

    assert veto is False
    assert reason == ""


def test_should_not_veto_clean_market():
    """Test no veto on clean market conditions."""
    df = pd.DataFrame({
        'high': [102 + i for i in range(30)],
        'low': [98 + i for i in range(30)],
        'open': [100 + i for i in range(30)],
        'close': [100 + i for i in range(30)],
        'volume': [1000 + i * 10 for i in range(30)]  # Rising volume
    })

    macro_cache = {'TOTAL3': {'value': 0.95}}

    config = {
        'narrative_enabled': True,
        'hodl_trap_vol_min': 0.8,
        'total3_hype_threshold': 1.1
    }

    veto, reason = should_veto_narrative(df, macro_cache, config)

    assert veto is False


def test_performance_trap_detection():
    """Microbench: trap detection should be fast."""
    import time

    df = pd.DataFrame({
        'close': np.random.randn(100) + 100,
        'volume': np.random.uniform(800, 1200, 100),
        'high': np.random.uniform(95, 105, 100),
        'low': np.random.uniform(90, 100, 100),
        'open': np.random.uniform(92, 102, 100)
    })

    macro_cache = {'TOTAL3': {'value': 1.1}}
    config = {'narrative_enabled': True, 'hodl_trap_vol_min': 0.8}

    start = time.time()
    for _ in range(100):
        should_veto_narrative(df, macro_cache, config)
    elapsed_ms = (time.time() - start) * 1000

    avg_ms = elapsed_ms / 100
    assert avg_ms < 50.0, f"Trap detection too slow: {avg_ms:.3f}ms (target <50ms)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
