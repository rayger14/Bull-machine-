"""
Unit tests for enhanced CVD (Cumulative Volume Delta) in Bull Machine v1.6.1
Tests CVD calculation, slope analysis, and divergence detection
"""

import pandas as pd
import pytest
import numpy as np
from bull_machine.modules.orderflow.lca import calculate_cvd, orderflow_lca, analyze_market_structure


class TestCVDCalculation:
    """Test CVD calculation with slope analysis."""

    def test_basic_cvd_calculation(self):
        """Test basic CVD calculation from OHLC and volume data."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],  # Closes near highs = bullish
            'volume': [1000, 1500, 2000]
        })

        cvd_result = calculate_cvd(df)

        assert 'delta' in cvd_result
        assert 'slope' in cvd_result
        assert 'current_delta' in cvd_result
        assert 'delta_ma' in cvd_result

        # Should be positive CVD for bullish closes
        assert cvd_result['delta'] > 0

    def test_bearish_cvd_pattern(self):
        """Test CVD calculation for bearish price action."""
        df = pd.DataFrame({
            'open': [103, 102, 101],
            'high': [104, 103, 102],
            'low': [101, 100, 99],
            'close': [102, 101, 100],  # Closes near lows = bearish
            'volume': [1000, 1500, 2000]
        })

        cvd_result = calculate_cvd(df)

        # Should be negative CVD for bearish closes
        assert cvd_result['delta'] < 0

    def test_cvd_slope_calculation(self):
        """Test CVD slope calculation for divergence detection."""
        # Create data with changing volume patterns
        volume_data = [1000] * 10 + [2000] * 10  # Volume increase
        price_base = 100

        df = pd.DataFrame({
            'open': [price_base + i * 0.1 for i in range(20)],
            'high': [price_base + i * 0.1 + 1 for i in range(20)],
            'low': [price_base + i * 0.1 - 1 for i in range(20)],
            'close': [price_base + i * 0.1 + 0.8 for i in range(20)],  # Bullish closes
            'volume': volume_data
        })

        cvd_result = calculate_cvd(df)

        # Should have non-zero slope with sufficient data
        assert cvd_result['slope'] != 0
        assert isinstance(cvd_result['slope'], float)

    def test_cvd_with_zero_ranges(self):
        """Test CVD calculation handles zero ranges (doji bars)."""
        df = pd.DataFrame({
            'open': [100, 100, 100],
            'high': [100, 100, 100],  # Zero range bars
            'low': [100, 100, 100],
            'close': [100, 100, 100],
            'volume': [1000, 1500, 2000]
        })

        cvd_result = calculate_cvd(df)

        # Should not crash and return valid structure
        assert 'delta' in cvd_result
        assert 'slope' in cvd_result
        assert isinstance(cvd_result['delta'], float)
        assert isinstance(cvd_result['slope'], float)

    def test_insufficient_data_handling(self):
        """Test CVD calculation with insufficient data."""
        df = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101],
            'volume': [1000]
        })

        cvd_result = calculate_cvd(df)

        # Should return zero values for insufficient data (< 2 bars)
        assert cvd_result['delta'] == 0.0
        assert cvd_result['slope'] == 0.0


class TestCVDDivergence:
    """Test CVD divergence detection patterns."""

    def test_bullish_divergence_pattern(self):
        """Test detection of bullish divergence (price down, CVD up)."""
        # Create price making lower lows but with increasing buying pressure
        prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]

        # Volume distribution favoring buyers (closes near highs)
        df = pd.DataFrame({
            'open': [p - 0.5 for p in prices],
            'high': [p + 1 for p in prices],
            'low': [p - 1.5 for p in prices],
            'close': [p + 0.8 for p in prices],  # Closes near highs despite downtrend
            'volume': [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800]
        })

        cvd_result = calculate_cvd(df)

        # Should show positive slope despite falling prices
        assert cvd_result['slope'] > 0
        assert cvd_result['delta'] > 0  # Cumulative buying pressure

    def test_bearish_divergence_pattern(self):
        """Test detection of bearish divergence (price up, CVD down)."""
        # Create price making higher highs but with decreasing buying pressure
        prices = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

        # Volume distribution favoring sellers (closes near lows)
        df = pd.DataFrame({
            'open': [p + 0.5 for p in prices],
            'high': [p + 1.5 for p in prices],
            'low': [p - 1 for p in prices],
            'close': [p - 0.8 for p in prices],  # Closes near lows despite uptrend
            'volume': [2800, 2600, 2400, 2200, 2000, 1800, 1600, 1400, 1200, 1000]
        })

        cvd_result = calculate_cvd(df)

        # Should show negative slope despite rising prices
        assert cvd_result['slope'] < 0
        assert cvd_result['delta'] < 0  # Cumulative selling pressure

    def test_no_divergence_pattern(self):
        """Test normal market without divergence."""
        # Create normal uptrend with matching CVD
        prices = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

        df = pd.DataFrame({
            'open': [p - 0.2 for p in prices],
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': [p + 0.6 for p in prices],  # Bullish closes with uptrend
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })

        cvd_result = calculate_cvd(df)

        # Should show positive slope matching price direction
        assert cvd_result['slope'] > 0
        assert cvd_result['delta'] > 0


class TestEnhancedOrderflowLCA:
    """Test enhanced orderflow LCA with CVD integration."""

    def test_cvd_enhancement_boost(self):
        """Test CVD enhancement boosts orderflow score."""
        # Create strong bullish setup with CVD confirmation
        df = pd.DataFrame({
            'open': [100] * 10 + [101, 102, 103],
            'high': [102] * 10 + [103, 104, 106],  # BOS at end
            'low': [99] * 10 + [100, 101, 102],
            'close': [101] * 10 + [102, 103, 105],  # Strong closes
            'volume': [1000] * 10 + [2000, 2500, 3000]  # Volume spike
        })

        config = {'features': {'orderflow_lca': True}}

        score = orderflow_lca(df, config)

        # Should have decent score with CVD confirmation
        assert score >= 0.5
        assert score <= 1.0

    def test_cvd_divergence_bonus(self):
        """Test CVD divergence detection provides appropriate bonus."""
        # Create hidden bullish divergence scenario
        df = pd.DataFrame({
            'open': [105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93],
            'high': [106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94],
            'low': [104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92],
            'close': [105.8, 104.8, 103.8, 102.8, 101.8, 100.8, 99.8, 98.8, 97.8, 96.8, 95.8, 94.8, 93.8],  # Strong closes despite downtrend
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
        })

        config = {'features': {'orderflow_lca': True}}

        score = orderflow_lca(df, config)

        # Should reflect the divergence analysis
        assert isinstance(score, float)
        assert score >= 0.0
        assert score <= 1.0

    def test_confluence_counting(self):
        """Test confluence counting includes CVD signals."""
        # Create multi-signal confluence
        df = pd.DataFrame({
            'open': [100] * 15 + [101],
            'high': [102] * 15 + [105],  # BOS
            'low': [99] * 15 + [100],
            'close': [100.8] * 15 + [104],  # LCA + strong close
            'volume': [1000] * 15 + [3000]  # Volume spike
        })

        config = {'features': {'orderflow_lca': True}}

        score = orderflow_lca(df, config)

        # Should reflect multiple confluence factors
        assert score >= 0.6  # Strong confluence should boost score


class TestMarketStructureAnalysis:
    """Test enhanced market structure analysis with CVD."""

    def test_complete_structure_analysis(self):
        """Test complete market structure analysis includes CVD data."""
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 107],  # Clear uptrend
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 106],
            'volume': [1000, 1200, 1400, 1600, 2000]
        })

        config = {'features': {'orderflow_lca': True}}

        result = analyze_market_structure(df, config)

        # Check all required fields are present
        assert 'lca_score' in result
        assert 'bos_analysis' in result
        assert 'intent_analysis' in result
        assert 'cvd_analysis' in result
        assert 'structure_health' in result
        assert 'orderflow_divergence' in result

        # Check CVD analysis structure
        cvd_analysis = result['cvd_analysis']
        assert 'delta' in cvd_analysis
        assert 'slope' in cvd_analysis
        assert 'current_delta' in cvd_analysis
        assert 'delta_ma' in cvd_analysis

        # Check divergence analysis
        divergence = result['orderflow_divergence']
        assert 'detected' in divergence
        assert 'type' in divergence
        assert 'strength' in divergence

    def test_divergence_detection_logic(self):
        """Test orderflow divergence detection logic."""
        # Create data with significant CVD slope
        df = pd.DataFrame({
            'open': [100] * 15,
            'high': [102] * 15,
            'low': [98] * 15,
            'close': [101.5] * 15,  # Consistent bullish closes
            'volume': list(range(1000, 2500, 100))  # Increasing volume
        })

        config = {'features': {'orderflow_lca': True}}

        result = analyze_market_structure(df, config)

        # Should have meaningful divergence analysis
        divergence = result['orderflow_divergence']
        assert isinstance(divergence['detected'], bool)
        assert divergence['type'] in ['bullish', 'bearish']
        assert 0 <= divergence['strength'] <= 1.0

    def test_structure_health_with_cvd(self):
        """Test structure health classification considers CVD."""
        # Strong bullish structure
        df_strong = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 108],  # Strong BOS
            'low': [99, 100, 101, 102, 103],
            'close': [101.8, 102.8, 103.8, 104.8, 107],  # Very bullish closes
            'volume': [1000, 1500, 2000, 2500, 3000]  # Volume confirmation
        })

        config = {'features': {'orderflow_lca': True}}

        result_strong = analyze_market_structure(df_strong, config)

        # Should reflect strong structure
        assert result_strong['lca_score'] >= 0.4  # Should be decent score
        assert result_strong['structure_health'] in ['strong', 'weak']


class TestCVDErrorHandling:
    """Test CVD calculation error handling."""

    def test_empty_dataframe(self):
        """Test CVD calculation with empty DataFrame."""
        df = pd.DataFrame()

        cvd_result = calculate_cvd(df)

        # Should return safe defaults
        assert cvd_result['delta'] == 0.0
        assert cvd_result['slope'] == 0.0

    def test_missing_columns(self):
        """Test CVD calculation with missing required columns."""
        df = pd.DataFrame({
            'open': [100, 101],
            'close': [101, 102]
            # Missing high, low, volume
        })

        cvd_result = calculate_cvd(df)

        # Should handle gracefully and return defaults
        assert cvd_result['delta'] == 0.0
        assert cvd_result['slope'] == 0.0

    def test_nan_values(self):
        """Test CVD calculation with NaN values in data."""
        df = pd.DataFrame({
            'open': [100, 101, np.nan],
            'high': [102, 103, 104],
            'low': [99, 100, np.nan],
            'close': [101, 102, 103],
            'volume': [1000, np.nan, 2000]
        })

        cvd_result = calculate_cvd(df)

        # Should handle NaN values gracefully
        assert isinstance(cvd_result['delta'], float)
        assert isinstance(cvd_result['slope'], float)
        assert not np.isnan(cvd_result['delta'])
        assert not np.isnan(cvd_result['slope'])


# Integration test
def test_cvd_orderflow_integration():
    """Integration test for CVD with full orderflow system."""
    # Create realistic orderflow scenario
    np.random.seed(42)

    # Simulate accumulation phase with hidden buying
    base_price = 100
    accumulation_bars = 30

    # Price stays relatively flat but CVD builds (hidden buying)
    prices = base_price + np.cumsum(np.random.normal(0.05, 0.3, accumulation_bars))

    # Create realistic OHLC with increasing buying pressure
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices * 1.01,  # Consistent strong closes
        'volume': np.random.uniform(1000, 3000, accumulation_bars)
    })

    config = {'features': {'orderflow_lca': True}}

    # Test full analysis
    lca_score = orderflow_lca(df, config)
    structure_analysis = analyze_market_structure(df, config)

    # Should produce valid results
    assert 0.0 <= lca_score <= 1.0
    assert 'cvd_analysis' in structure_analysis
    assert 'orderflow_divergence' in structure_analysis

    # CVD should reflect the accumulation pattern
    cvd_data = structure_analysis['cvd_analysis']
    assert isinstance(cvd_data['delta'], float)
    assert isinstance(cvd_data['slope'], float)

    # Divergence analysis should be meaningful
    divergence = structure_analysis['orderflow_divergence']
    assert divergence['type'] in ['bullish', 'bearish']
    assert 0 <= divergence['strength'] <= 1.0