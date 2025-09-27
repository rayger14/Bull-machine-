"""
Unit tests for MTF enhancements in Bull Machine v1.5.0
"""

import pandas as pd
import pytest
import numpy as np
from bull_machine.modules.mtf.mtf_sync import (
    six_candle_structure,
    mtf_dl2_filter,
    enhanced_mtf_sync,
    calculate_mtf_alignment_score
)


class TestSixCandleStructure:
    """Test 6-candle leg rule implementation."""

    def test_valid_pattern(self):
        """Test detection of valid 6-candle pattern."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 101, 100, 101]  # Valid alternating pattern
        })
        assert six_candle_structure(df) == True

    def test_invalid_pattern(self):
        """Test rejection of invalid pattern."""
        df = pd.DataFrame({
            "close": [100, 99, 98, 97, 96, 95]  # Trending down, no alternation
        })
        assert six_candle_structure(df) == False

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({"close": [100, 101, 102]})  # Only 3 candles
        assert six_candle_structure(df) == False

    def test_edge_case_equal_values(self):
        """Test handling of equal close values."""
        df = pd.DataFrame({
            "close": [100, 100, 101, 101, 100, 100]
        })
        # Should fail because 100 == 100 violates alternating pattern
        assert six_candle_structure(df) == False


class TestMTFDL2Filter:
    """Test MTF DL2 filter implementation."""

    def test_normal_deviation(self):
        """Test acceptance of normal price deviation."""
        # Create data with low volatility
        np.random.seed(42)
        prices = 100 + np.random.normal(0, 1, 20)  # Low volatility around 100
        df = pd.DataFrame({"close": prices})
        assert mtf_dl2_filter(df, "4H") == True  # Use 4H timeframe (threshold=2.5)

    def test_extreme_deviation(self):
        """Test rejection of extreme deviation."""
        # Normal prices then extreme spike
        prices = [100] * 19 + [130]  # Extreme spike (z-score ~30)
        df = pd.DataFrame({"close": prices})
        assert mtf_dl2_filter(df, "1H") == False  # Use 1H timeframe (threshold=2.0)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        assert mtf_dl2_filter(df) is True  # Should default to True

    def test_zero_volatility(self):
        """Test handling of zero volatility (constant prices)."""
        df = pd.DataFrame({"close": [100] * 20})
        assert mtf_dl2_filter(df) is True

    def test_custom_threshold(self):
        """Test different timeframe thresholds."""
        prices = [100] * 19 + [110]  # Moderate spike (z-score ~10)
        df = pd.DataFrame({"close": prices})

        # Should fail with any timeframe since z-score is very high
        assert mtf_dl2_filter(df, "4H") == False  # threshold=2.5
        assert mtf_dl2_filter(df, "1D") == False  # threshold=2.2
        assert mtf_dl2_filter(df, "1H") == False  # threshold=2.0


class TestEnhancedMTFSync:
    """Test enhanced MTF synchronization."""

    def test_no_features_enabled(self):
        """Test behavior when no features are enabled."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 101, 100, 101],
            "high": [101, 102, 103, 102, 101, 102],
            "low": [99, 100, 101, 100, 99, 100]
        })
        config = {"features": {}}

        result = enhanced_mtf_sync(df, config)
        assert result["mtf_decision"] == "ALLOW"
        assert len(result["filters_applied"]) == 0

    def test_six_candle_leg_enabled_valid(self):
        """Test with 6-candle leg rule enabled and valid pattern."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 101, 100, 101]
        })
        config = {"features": {"six_candle_leg": True}}

        result = enhanced_mtf_sync(df, config)
        assert result["mtf_decision"] == "ALLOW"
        assert "six_candle_leg" in result["filters_applied"]
        assert result["filter_results"]["six_candle_leg"] == True

    def test_six_candle_leg_enabled_invalid(self):
        """Test with 6-candle leg rule enabled and invalid pattern."""
        df = pd.DataFrame({
            "close": [100, 99, 98, 97, 96, 95]
        })
        config = {"features": {"six_candle_leg": True}}

        result = enhanced_mtf_sync(df, config)
        assert result["mtf_decision"] == "VETO"
        assert result["veto_reason"] == "6-candle structure invalid"

    def test_mtf_dl2_enabled_extreme_deviation(self):
        """Test with MTF DL2 enabled and extreme deviation."""
        prices = [100] * 19 + [130]
        df = pd.DataFrame({"close": prices})
        config = {
            "features": {"mtf_dl2": True},
            "mtf_dl2_threshold": 2.0
        }

        result = enhanced_mtf_sync(df, config)
        assert result["mtf_decision"] == "VETO"
        assert result["veto_reason"] == "MTF DL2 deviation too high"

    def test_multiple_filters_enabled(self):
        """Test with multiple filters enabled."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 101, 100, 101] * 5  # 30 bars, valid pattern
        })
        config = {
            "features": {
                "six_candle_leg": True,
                "mtf_dl2": True
            },
            "mtf_dl2_threshold": 3.0
        }

        result = enhanced_mtf_sync(df, config)
        assert result["mtf_decision"] == "ALLOW"
        assert len(result["filters_applied"]) == 2


class TestMTFAlignmentScore:
    """Test MTF alignment score calculation."""

    def test_perfect_long_alignment(self):
        """Test perfect long alignment across all timeframes."""
        score = calculate_mtf_alignment_score("long", "long", "long")
        assert score == 1.0

    def test_perfect_short_alignment(self):
        """Test perfect short alignment across all timeframes."""
        score = calculate_mtf_alignment_score("short", "short", "short")
        assert score == 1.0

    def test_strong_alignment(self):
        """Test strong alignment (2/3 match)."""
        score = calculate_mtf_alignment_score("long", "long", "neutral")
        assert score == 0.7

    def test_mixed_alignment(self):
        """Test mixed alignment without neutrals."""
        score = calculate_mtf_alignment_score("long", "short", "long")
        assert score == 0.7  # 2/3 match = strong alignment

    def test_poor_alignment(self):
        """Test poor alignment with neutrals."""
        score = calculate_mtf_alignment_score("long", "neutral", "short")
        assert score == 0.2


# Integration test
def test_mtf_integration():
    """Integration test for MTF components."""
    # Create realistic price data
    np.random.seed(42)
    base_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 50))

    df = pd.DataFrame({
        "close": base_prices,
        "high": base_prices * 1.01,
        "low": base_prices * 0.99,
        "open": np.roll(base_prices, 1)
    })

    config = {
        "features": {
            "six_candle_leg": True,
            "mtf_dl2": True
        },
        "mtf_dl2_threshold": 2.0
    }

    result = enhanced_mtf_sync(df, config)

    # Should have valid structure
    assert "mtf_decision" in result
    assert "filters_applied" in result
    assert "filter_results" in result

    # Both filters should be applied
    assert len(result["filters_applied"]) == 2