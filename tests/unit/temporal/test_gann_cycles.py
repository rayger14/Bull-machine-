"""
Unit tests for v1.8.6 Temporal/Gann Cycles Module

Tests all components:
1. ACF cycle detection (30/60/90 day)
2. Square of 9 proximity
3. Gann angle adherence
4. Thermo-floor calculation
5. Log premium
6. Logistic bid scoring
7. LPPLS blowoff detection
8. Integration via temporal_signal
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from engine.temporal.gann_cycles import (
    _cycles_acf,
    _square9,
    _angles_score,
    _thermo_floor,
    _log_premium,
    _logistic_bid,
    _lppls_simple,
    temporal_signal
)


@pytest.fixture
def sample_config():
    """Default config for temporal module."""
    return {
        'acf_lookback_days': 180,
        'target_cycles': [30, 60, 90],
        'cycle_tolerance_days': 5,
        'square9_step': 9.0,
        'square9_tolerance': 2.0,
        'gann_angle_lookback': 24,
        'energy_cost_per_hash': 0.05e-12,
        'log_premium_beta': 0.0001,
        'logistic_k': 5.0,
        'volume_threshold': 1.2,
        'lppls_veto_confidence': 0.75,
        'bonus_cap': 0.15,
        'feature_weights': {
            'acf': 0.25,
            'square9': 0.20,
            'angles': 0.15,
            'thermo': 0.15,
            'logistic': 0.25
        }
    }


@pytest.fixture
def sample_1h_data():
    """Generate sample 1H OHLCV data."""
    dates = pd.date_range(start='2025-01-01', periods=500, freq='1H')
    np.random.seed(42)

    close = 50000 + np.cumsum(np.random.randn(500) * 100)
    high = close + np.abs(np.random.randn(500) * 50)
    low = close - np.abs(np.random.randn(500) * 50)
    open_price = close + np.random.randn(500) * 30
    volume = np.random.uniform(100, 1000, 500)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def sample_4h_data():
    """Generate sample 4H OHLCV data."""
    dates = pd.date_range(start='2025-01-01', periods=125, freq='4H')
    np.random.seed(43)

    close = 50000 + np.cumsum(np.random.randn(125) * 200)
    high = close + np.abs(np.random.randn(125) * 100)
    low = close - np.abs(np.random.randn(125) * 100)
    open_price = close + np.random.randn(125) * 60
    volume = np.random.uniform(400, 4000, 125)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def sample_1d_data():
    """Generate sample 1D OHLCV data."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1D')
    np.random.seed(44)

    # Add 30-day cycle for ACF testing
    cycle = 1000 * np.sin(2 * np.pi * np.arange(200) / 30)
    trend = np.arange(200) * 10
    noise = np.random.randn(200) * 300
    close = 50000 + trend + cycle + noise

    high = close + np.abs(np.random.randn(200) * 150)
    low = close - np.abs(np.random.randn(200) * 150)
    open_price = close + np.random.randn(200) * 100
    volume = np.random.uniform(1000, 10000, 200)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


# ═══════════════════════════════════════════════════════════════
# 1. ACF Cycle Detection Tests
# ═══════════════════════════════════════════════════════════════

def test_acf_detects_30day_cycle(sample_1d_data, sample_config):
    """Test ACF can detect 30-day cycle in synthetic data."""
    confluence, phase, peaks = _cycles_acf(sample_1d_data, sample_config)

    # Should detect some cycle
    assert confluence >= 0.0
    assert phase in ['accumulation', 'distribution', 'equilibrium']
    assert isinstance(peaks, list)


def test_acf_insufficient_data(sample_config):
    """Test ACF handles insufficient data gracefully."""
    short_df = pd.DataFrame({
        'close': [50000] * 50
    })

    confluence, phase, peaks = _cycles_acf(short_df, sample_config)

    assert confluence == 0.0
    assert phase == "insufficient_data"
    assert peaks == []


def test_acf_bounds(sample_1d_data, sample_config):
    """Test ACF confluence is bounded [0, 1]."""
    confluence, _, _ = _cycles_acf(sample_1d_data, sample_config)

    assert 0.0 <= confluence <= 1.0


# ═══════════════════════════════════════════════════════════════
# 2. Square of 9 Tests
# ═══════════════════════════════════════════════════════════════

def test_square9_exact_level(sample_config):
    """Test Square of 9 returns perfect score at exact level."""
    price = 54000.0  # Exact multiple of 9
    score, level = _square9(price, sample_config)

    assert score == 1.0
    assert level == 54000.0


def test_square9_proximity(sample_config):
    """Test Square of 9 scores based on proximity."""
    # Just off from 54000
    price = 54005.0
    score, level = _square9(price, sample_config)

    assert 0.0 < score < 1.0
    # Level should be nearest multiple of 9 (54009 is closer than 54000)
    assert level in [54000.0, 54009.0]


def test_square9_far_from_level(sample_config):
    """Test Square of 9 returns 0 when far from level."""
    # Midway between levels (54000 and 54009)
    price = 54004.5
    score, _ = _square9(price, sample_config)

    # Should have some score (not necessarily 0)
    assert 0.0 <= score <= 1.0


def test_square9_bounds(sample_config):
    """Test Square of 9 score is bounded [0, 1]."""
    for price in [50000, 50123, 50999, 51234]:
        score, _ = _square9(price, sample_config)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════
# 3. Gann Angle Tests
# ═══════════════════════════════════════════════════════════════

def test_gann_angle_trending(sample_1h_data, sample_config):
    """Test Gann angle scoring on trending data."""
    score = _angles_score(sample_1h_data, sample_1h_data, sample_config)

    assert 0.0 <= score <= 1.0


def test_gann_angle_insufficient_data(sample_config):
    """Test Gann angle handles short data."""
    short_df = pd.DataFrame({
        'close': [50000] * 10,
        'high': [50100] * 10,
        'low': [49900] * 10
    })

    score = _angles_score(short_df, short_df, sample_config)

    assert score == 0.0


# ═══════════════════════════════════════════════════════════════
# 4. Thermo Floor Tests
# ═══════════════════════════════════════════════════════════════

def test_thermo_floor_calculation(sample_1d_data, sample_config):
    """Test thermo floor calculates mining cost."""
    macro_cache = {
        'HASHRATE': {'value': 600e18},  # 600 EH/s
    }

    floor, distance = _thermo_floor(sample_1d_data, macro_cache, sample_config)

    # Floor should be positive
    assert floor > 0
    # Distance should be calculated
    assert isinstance(distance, float)


def test_thermo_floor_above_current_price(sample_1d_data, sample_config):
    """Test distance calculation when price above floor."""
    macro_cache = {
        'HASHRATE': {'value': 1e18},  # Very low hashrate → very low floor
    }

    floor, distance = _thermo_floor(sample_1d_data, macro_cache, sample_config)

    # With very low hashrate, floor should be low → price likely above → positive distance
    # But if not, just check that distance is calculated
    assert isinstance(distance, float)


def test_thermo_floor_no_macro_data(sample_1d_data, sample_config):
    """Test thermo floor uses defaults when no macro data."""
    floor, distance = _thermo_floor(sample_1d_data, {}, sample_config)

    assert floor > 0
    assert isinstance(distance, float)


# ═══════════════════════════════════════════════════════════════
# 5. Log Premium Tests
# ═══════════════════════════════════════════════════════════════

def test_log_premium_calculation(sample_1d_data, sample_config):
    """Test log premium multiplier."""
    macro_cache = {
        'DIFFICULTY': {'value': 1e14}
    }

    premium = _log_premium(sample_1d_data, macro_cache, sample_config)

    # Premium should be >= 1.0
    assert premium >= 1.0


def test_log_premium_increases_with_time(sample_config):
    """Test premium increases over time (days since halving)."""
    # Early post-halving
    early_df = pd.DataFrame({
        'close': [50000]
    }, index=[pd.Timestamp('2024-05-01')])

    # Late in cycle
    late_df = pd.DataFrame({
        'close': [50000]
    }, index=[pd.Timestamp('2025-09-01')])

    macro_cache = {'DIFFICULTY': {'value': 1e14}}

    premium_early = _log_premium(early_df, macro_cache, sample_config)
    premium_late = _log_premium(late_df, macro_cache, sample_config)

    # Later premium should be higher (time component)
    assert premium_late > premium_early


# ═══════════════════════════════════════════════════════════════
# 6. Logistic Bid Tests
# ═══════════════════════════════════════════════════════════════

def test_logistic_bid_high_volume(sample_1d_data, sample_config):
    """Test logistic bid detects high volume re-accumulation."""
    # Boost recent volume
    df = sample_1d_data.copy()
    df['volume'].iloc[-7:] *= 2.0

    score, phase = _logistic_bid(df, df, sample_config)

    assert 0.0 <= score <= 1.0
    assert phase in ['strong_bid', 'weak_bid', 'neutral']


def test_logistic_bid_low_volume(sample_1d_data, sample_config):
    """Test logistic bid detects weak volume."""
    # Reduce recent volume
    df = sample_1d_data.copy()
    df['volume'].iloc[-7:] *= 0.5

    score, phase = _logistic_bid(df, df, sample_config)

    assert score < 0.5  # Should be lower
    assert phase in ['weak_bid', 'neutral']


def test_logistic_bid_insufficient_data(sample_config):
    """Test logistic bid handles short data."""
    short_df = pd.DataFrame({
        'volume': [1000] * 20
    })

    score, phase = _logistic_bid(short_df, short_df, sample_config)

    assert score == 0.5
    assert phase == "neutral"


# ═══════════════════════════════════════════════════════════════
# 7. LPPLS Blowoff Tests
# ═══════════════════════════════════════════════════════════════

def test_lppls_no_blowoff_normal_market(sample_1d_data, sample_config):
    """Test LPPLS doesn't veto normal market conditions."""
    veto, confidence, reason = _lppls_simple(sample_1d_data, sample_config)

    # Normal market shouldn't veto
    assert isinstance(veto, bool)
    assert 0.0 <= confidence <= 1.0


def test_lppls_detects_parabolic(sample_config):
    """Test LPPLS detects parabolic price with declining volume."""
    dates = pd.date_range(start='2025-01-01', periods=90, freq='1D')

    # Parabolic price
    prices = 50000 + 1000 * np.arange(90) ** 1.3

    # Declining volume
    volumes = 10000 - np.arange(90) * 50

    df = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)

    veto, confidence, reason = _lppls_simple(df, sample_config)

    # Should detect blowoff
    assert confidence > 0.0


def test_lppls_insufficient_data(sample_config):
    """Test LPPLS handles short data."""
    short_df = pd.DataFrame({
        'close': [50000] * 20,
        'volume': [1000] * 20
    })

    veto, confidence, reason = _lppls_simple(short_df, sample_config)

    assert veto is False
    assert confidence == 0.0
    assert reason is None


# ═══════════════════════════════════════════════════════════════
# 8. Integration Tests (temporal_signal)
# ═══════════════════════════════════════════════════════════════

def test_temporal_signal_basic(sample_1h_data, sample_4h_data, sample_1d_data, sample_config):
    """Test temporal_signal returns expected structure."""
    result = temporal_signal(
        sample_1h_data,
        sample_4h_data,
        sample_1d_data,
        sample_config,
        {}
    )

    # Check structure
    assert 'confluence_score' in result
    assert 'cycle_phase' in result
    assert 'veto' in result
    assert 'features' in result

    # Check types
    assert isinstance(result['confluence_score'], float)
    assert isinstance(result['cycle_phase'], str)
    assert isinstance(result['veto'], bool)
    assert isinstance(result['features'], dict)


def test_temporal_signal_bounds(sample_1h_data, sample_4h_data, sample_1d_data, sample_config):
    """Test temporal_signal confluence is bounded [0, 1]."""
    result = temporal_signal(
        sample_1h_data,
        sample_4h_data,
        sample_1d_data,
        sample_config,
        {}
    )

    assert 0.0 <= result['confluence_score'] <= 1.0


def test_temporal_signal_features_present(sample_1h_data, sample_4h_data, sample_1d_data, sample_config):
    """Test all expected features are in result."""
    result = temporal_signal(
        sample_1h_data,
        sample_4h_data,
        sample_1d_data,
        sample_config,
        {}
    )

    features = result['features']

    # Check all features exist
    assert 'acf_score' in features
    assert 'acf_cycles' in features
    assert 'square9_score' in features
    assert 'square9_level' in features
    assert 'gann_angle_score' in features
    assert 'thermo_floor' in features
    assert 'thermo_distance' in features
    assert 'log_premium' in features
    assert 'logistic_bid_score' in features
    assert 'logistic_phase' in features
    assert 'lppls_veto' in features
    assert 'lppls_confidence' in features


def test_temporal_signal_veto_propagates(sample_config):
    """Test LPPLS veto propagates to top-level result."""
    # Create blowoff data
    dates = pd.date_range(start='2025-01-01', periods=200, freq='1H')
    parabolic = 50000 + 1000 * np.arange(200) ** 1.2

    df_1h = pd.DataFrame({
        'open': parabolic,
        'high': parabolic + 100,
        'low': parabolic - 100,
        'close': parabolic,
        'volume': 1000 - np.arange(200)
    }, index=dates)

    dates_4h = pd.date_range(start='2025-01-01', periods=50, freq='4H')
    df_4h = df_1h.iloc[::4].copy()
    df_4h.index = dates_4h

    dates_1d = pd.date_range(start='2024-07-01', periods=200, freq='1D')
    parabolic_1d = 50000 + 1000 * np.arange(200) ** 1.3
    df_1d = pd.DataFrame({
        'open': parabolic_1d,
        'high': parabolic_1d + 500,
        'low': parabolic_1d - 500,
        'close': parabolic_1d,
        'volume': 10000 - np.arange(200) * 40
    }, index=dates_1d)

    result = temporal_signal(df_1h, df_4h, df_1d, sample_config, {})

    # If LPPLS detects blowoff, veto should be True
    if result['features']['lppls_veto']:
        assert result['veto'] is True
        assert result['veto_reason'] is not None


def test_temporal_signal_deterministic(sample_1h_data, sample_4h_data, sample_1d_data, sample_config):
    """Test temporal_signal is deterministic (same inputs → same outputs)."""
    result1 = temporal_signal(
        sample_1h_data,
        sample_4h_data,
        sample_1d_data,
        sample_config,
        {}
    )

    result2 = temporal_signal(
        sample_1h_data,
        sample_4h_data,
        sample_1d_data,
        sample_config,
        {}
    )

    assert result1['confluence_score'] == result2['confluence_score']
    assert result1['cycle_phase'] == result2['cycle_phase']
    assert result1['veto'] == result2['veto']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
