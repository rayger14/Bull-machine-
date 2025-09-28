"""
Unit tests for Fibonacci Clusters in Bull Machine v1.6.1
Tests temporal clusters, price clusters, and Oracle whisper system
"""

import pandas as pd
import pytest
import numpy as np
from bull_machine.strategy.temporal_fib_clusters import fib_time_clusters, detect_pivot_points
from bull_machine.strategy.hidden_fibs import fib_price_clusters, detect_price_time_confluence
from bull_machine.oracle import trigger_whisper, should_trigger_confluence_alert


class TestFibTimeClusters:
    """Test Fibonacci time cluster detection."""

    def test_basic_time_cluster(self):
        """Test basic time cluster detection with overlapping fibonacci projections."""
        pivots = pd.DataFrame({
            'bar_index': [100, 120],
            'price': [5000, 5100],
            'type': ['swing_low', 'swing_high']
        })

        # Test at bar 155 (34 bars from 120, 55 bars from 100)
        cluster = fib_time_clusters(pivots, current_bar=155, tf='1H', tolerance_bars=3)

        assert cluster is not None
        assert cluster['overlap_count'] >= 2
        assert cluster['strength'] >= 0.30
        assert 'cluster_bar' in cluster
        assert 'window' in cluster

    def test_no_cluster_insufficient_overlaps(self):
        """Test when no cluster forms due to insufficient overlaps."""
        pivots = pd.DataFrame({
            'bar_index': [100],
            'price': [5000],
            'type': ['swing_low']
        })

        # Single pivot cannot form cluster
        cluster = fib_time_clusters(pivots, current_bar=121, tf='1H')
        assert cluster is None

    def test_cluster_strength_calculation(self):
        """Test cluster strength calculation based on convergence and fibonacci levels."""
        pivots = pd.DataFrame({
            'bar_index': [50, 100, 150],
            'price': [4900, 5000, 5100],
            'type': ['swing_low', 'swing_high', 'swing_low']
        })

        # Multiple overlapping high fibonacci numbers should increase strength
        cluster = fib_time_clusters(pivots, current_bar=205, tf='1D', tolerance_bars=2)

        if cluster is not None:
            assert cluster['strength'] <= 0.80  # Maximum cap
            assert cluster['convergence_bars'] >= 0
            assert len(cluster['fib_numbers']) >= 2

    def test_timeframe_adjustment(self):
        """Test timeframe strength multiplier."""
        pivots = pd.DataFrame({
            'bar_index': [100, 140],
            'price': [5000, 5100],
            'type': ['swing_low', 'swing_high']
        })

        cluster_1h = fib_time_clusters(pivots, current_bar=155, tf='1H', tolerance_bars=3)
        cluster_1d = fib_time_clusters(pivots, current_bar=155, tf='1D', tolerance_bars=3)

        # Daily timeframe should have higher strength multiplier
        if cluster_1h and cluster_1d:
            assert cluster_1d['strength'] >= cluster_1h['strength']


class TestFibPriceClusters:
    """Test Fibonacci price cluster detection."""

    def test_price_cluster_overlap(self):
        """Test price cluster detection with overlapping fibonacci levels."""
        swings = pd.DataFrame({
            'high': [5100, 5200],
            'low': [4900, 5000],
            'close': [5000, 5150]
        })

        cluster = fib_price_clusters(swings, tolerance=0.02, volume_confirm=True)

        assert cluster is not None
        assert cluster['strength'] >= 0.60
        assert cluster['overlap_count'] >= 1
        assert 'zone' in cluster
        assert 'classification' in cluster

    def test_cluster_classification(self):
        """Test cluster classification as premium, discount, or target."""
        swings = pd.DataFrame({
            'high': [5000, 5100],
            'low': [4800, 4900],
            'close': [4900, 5000]
        })

        cluster = fib_price_clusters(swings, tolerance=0.03)

        if cluster is not None:
            assert cluster['classification'] in ['premium', 'discount', 'target']
            assert cluster['zone_type'] in ['retracement', 'extension']

    def test_volume_confirmation_boost(self):
        """Test volume confirmation increases cluster strength."""
        swings = pd.DataFrame({
            'high': [5100, 5150],
            'low': [4900, 4950],
            'close': [5000, 5100]
        })

        cluster_no_vol = fib_price_clusters(swings, tolerance=0.02, volume_confirm=False)
        cluster_with_vol = fib_price_clusters(swings, tolerance=0.02, volume_confirm=True)

        if cluster_no_vol and cluster_with_vol:
            assert cluster_with_vol['strength'] >= cluster_no_vol['strength']

    def test_key_fibonacci_levels_bonus(self):
        """Test that key fibonacci levels (0.382, 0.618, 1.618) get strength bonus."""
        swings = pd.DataFrame({
            'high': [5000],
            'low': [4800],
            'close': [4900]
        })

        cluster = fib_price_clusters(swings, tolerance=0.03)

        # Should detect clusters at key levels with bonus strength
        if cluster is not None:
            # Check if any key levels are present
            key_levels = [level for level in cluster['levels'] if level in [0.382, 0.618, 1.618]]
            if key_levels:
                assert cluster['strength'] >= 0.65  # Base + key level bonus


class TestPriceTimeConfluence:
    """Test price-time confluence detection."""

    def test_confluence_detection(self):
        """Test detection of price-time confluence."""
        # Create sample data with clear structure
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='1D')

        # Create trending data with clear swings
        base_price = 100
        price_trend = np.cumsum(np.random.normal(0.1, 0.5, 200))
        prices = base_price + price_trend

        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 3000, 200)
        }, index=dates)

        config = {
            'features': {'temporal_fib': True},
            'fib': {'tolerance': 0.02},
            'temporal': {'tolerance_bars': 3, 'pivot_window': 5},
            'timeframe': '1D'
        }

        confluence_data = detect_price_time_confluence(df, config, current_idx=150)

        assert isinstance(confluence_data, dict)
        assert 'confluence_detected' in confluence_data
        assert 'confluence_strength' in confluence_data
        assert 'tags' in confluence_data
        assert confluence_data['confluence_strength'] >= 0.0

    def test_confluence_strength_calculation(self):
        """Test confluence strength calculation logic."""
        # Mock data that should trigger both price and time clusters
        df = pd.DataFrame({
            'open': [100] * 100,
            'high': [102] * 100,
            'low': [98] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })

        config = {
            'features': {'temporal_fib': True},
            'fib': {'tolerance': 0.05},  # Higher tolerance to increase detection
            'temporal': {'tolerance_bars': 5, 'pivot_window': 3}
        }

        confluence_data = detect_price_time_confluence(df, config, current_idx=50)

        # Should have valid confluence data structure
        assert 'price_cluster' in confluence_data
        assert 'time_cluster' in confluence_data
        assert confluence_data['confluence_strength'] <= 0.90  # Maximum cap


class TestOracleWhispers:
    """Test Oracle whisper system."""

    def test_price_time_confluence_whisper(self):
        """Test whisper triggering for price-time confluence."""
        scores = {
            'cluster_tags': ['price_time_confluence'],
            'fib_retracement': 0.45,
            'wyckoff_phase': 'C',
            'confluence_strength': 0.75
        }

        whispers = trigger_whisper(scores, phase='C')

        assert whispers is not None
        assert any("Symmetry detected" in whisper for whisper in whispers)
        assert any("Time and price converge" in whisper for whisper in whispers)

    def test_premium_discount_whispers(self):
        """Test whispers for premium/discount zone detection."""
        # Test discount zone
        scores_discount = {
            'cluster_tags': ['price_discount'],
            'fib_retracement': 0.50,
            'confluence_strength': 0.60
        }

        whispers_discount = trigger_whisper(scores_discount, phase='C')
        if whispers_discount:
            assert any("Discount" in whisper for whisper in whispers_discount)

        # Test premium zone
        scores_premium = {
            'cluster_tags': ['price_premium'],
            'fib_retracement': 0.50,
            'confluence_strength': 0.60
        }

        whispers_premium = trigger_whisper(scores_premium, phase='D')
        if whispers_premium:
            assert any("Premium" in whisper for whisper in whispers_premium)

    def test_cvd_divergence_whispers(self):
        """Test whispers for CVD divergence patterns."""
        # Hidden bullish divergence
        scores = {
            'cvd_delta': -1000,
            'cvd_slope': 150,
            'confluence_strength': 0.55
        }

        whispers = trigger_whisper(scores, phase='C')
        if whispers:
            assert any("Hidden intent revealed" in whisper for whisper in whispers)

    def test_confluence_alert_threshold(self):
        """Test confluence alert threshold logic."""
        # High confluence should trigger alert
        high_confluence = {
            'confluence_strength': 0.80,
            'cluster_tags': ['price_time_confluence']
        }
        assert should_trigger_confluence_alert(high_confluence) is True

        # Low confluence should not trigger alert
        low_confluence = {
            'confluence_strength': 0.30,
            'cluster_tags': []
        }
        assert should_trigger_confluence_alert(low_confluence) is False

        # Moderate confluence with price-time confluence should trigger
        moderate_confluence = {
            'confluence_strength': 0.62,
            'cluster_tags': ['price_time_confluence']
        }
        assert should_trigger_confluence_alert(moderate_confluence) is True

    def test_whisper_suppression_for_low_confluence(self):
        """Test that whispers are suppressed for low confluence scenarios."""
        scores = {
            'cluster_tags': ['price_confluence'],
            'fib_retracement': 0.30,  # Below threshold
            'confluence_strength': 0.25  # Low confluence
        }

        whispers = trigger_whisper(scores, phase='C')
        assert whispers is None  # Should be suppressed


class TestPivotDetection:
    """Test pivot point detection for time cluster analysis."""

    def test_swing_high_detection(self):
        """Test detection of swing high pivot points."""
        # Create data with clear swing high
        df = pd.DataFrame({
            'high': [100, 101, 102, 105, 103, 102, 101],  # Peak at index 3
            'low': [98, 99, 100, 103, 101, 100, 99],
            'close': [99, 100, 101, 104, 102, 101, 100]
        })

        pivots = detect_pivot_points(df, window=2)

        # Should detect swing high
        assert len(pivots) > 0
        swing_highs = pivots[pivots['type'] == 'swing_high']
        assert len(swing_highs) > 0

    def test_swing_low_detection(self):
        """Test detection of swing low pivot points."""
        # Create data with clear swing low
        df = pd.DataFrame({
            'high': [102, 101, 100, 97, 99, 100, 101],  # Trough at index 3
            'low': [100, 99, 98, 95, 97, 98, 99],
            'close': [101, 100, 99, 96, 98, 99, 100]
        })

        pivots = detect_pivot_points(df, window=2)

        # Should detect swing low
        assert len(pivots) > 0
        swing_lows = pivots[pivots['type'] == 'swing_low']
        assert len(swing_lows) > 0

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for pivot detection."""
        # Not enough data
        df = pd.DataFrame({
            'high': [100, 101],
            'low': [98, 99],
            'close': [99, 100]
        })

        pivots = detect_pivot_points(df, window=5)

        # Should return empty DataFrame
        assert len(pivots) == 0


# Integration test
def test_fib_clusters_integration():
    """Integration test for fibonacci clusters system."""
    # Create realistic market data
    np.random.seed(42)

    # Simulate accumulation -> markup -> distribution cycle
    base_price = 100

    # Accumulation phase (sideways with slight upward bias)
    accumulation = base_price + np.cumsum(np.random.normal(0.02, 0.3, 50))

    # Markup phase (strong uptrend)
    markup = accumulation[-1] + np.cumsum(np.random.normal(0.3, 0.5, 30))

    # Distribution phase (sideways to slight down)
    distribution = markup[-1] + np.cumsum(np.random.normal(-0.1, 0.4, 40))

    all_prices = np.concatenate([accumulation, markup, distribution])

    df = pd.DataFrame({
        'open': all_prices * 0.999,
        'high': all_prices * 1.015,
        'low': all_prices * 0.985,
        'close': all_prices,
        'volume': np.random.uniform(800, 2500, len(all_prices))
    })

    config = {
        'features': {'temporal_fib': True, 'fib_clusters': True},
        'fib': {'tolerance': 0.03},
        'temporal': {'tolerance_bars': 4, 'pivot_window': 5},
        'timeframe': '1D'
    }

    # Test confluence detection at various points
    for test_idx in [60, 80, 100]:
        if test_idx < len(df):
            confluence_data = detect_price_time_confluence(df, config, test_idx)

            # Should have valid structure
            assert 'confluence_detected' in confluence_data
            assert 'confluence_strength' in confluence_data
            assert 'tags' in confluence_data

            # If confluence detected, should have reasonable strength
            if confluence_data['confluence_detected']:
                assert confluence_data['confluence_strength'] >= 0.30
                assert confluence_data['confluence_strength'] <= 0.90

                # Should have meaningful tags
                assert len(confluence_data['tags']) > 0

    # Test whisper system with realistic confluence
    test_scores = {
        'cluster_tags': ['price_time_confluence'],
        'fib_retracement': 0.65,
        'fib_extension': 0.40,
        'confluence_strength': 0.72,
        'wyckoff_phase': 'D'
    }

    whispers = trigger_whisper(test_scores, phase='D')

    # Should generate whispers for strong confluence
    assert whispers is not None
    assert len(whispers) > 0
    assert any("Symmetry detected" in whisper for whisper in whispers)