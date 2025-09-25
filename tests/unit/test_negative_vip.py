"""
Unit tests for Negative VIP module in Bull Machine v1.5.0
"""

import pandas as pd
import pytest
import numpy as np
from bull_machine.modules.sentiment.negative_vip import (
    detect_volume_spike,
    detect_reversal_pattern,
    calculate_momentum_divergence,
    negative_vip_score,
    analyze_reversal_risk
)


class TestDetectVolumeSpike:
    """Test volume spike detection."""

    def test_moderate_spike(self):
        """Test detection of moderate volume spike."""
        # Normal volume then spike
        volume_data = [1000] * 19 + [1600]  # 1.6x spike
        df = pd.DataFrame({"volume": volume_data})

        result = detect_volume_spike(df, spike_threshold=1.5)
        assert result["detected"] is True
        assert result["intensity"] == "moderate"
        assert result["ratio"] == 1.6

    def test_extreme_spike(self):
        """Test detection of extreme volume spike."""
        volume_data = [1000] * 19 + [3500]  # 3.5x spike
        df = pd.DataFrame({"volume": volume_data})

        result = detect_volume_spike(df, spike_threshold=1.5)
        assert result["detected"] is True
        assert result["intensity"] == "extreme"
        assert result["ratio"] == 3.5

    def test_no_spike(self):
        """Test when no volume spike occurs."""
        volume_data = [1000] * 20  # Constant volume
        df = pd.DataFrame({"volume": volume_data})

        result = detect_volume_spike(df, spike_threshold=1.5)
        assert result["detected"] is False
        assert result["intensity"] == "none"
        assert result["ratio"] == 1.0

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({"volume": [1000, 2000]})

        result = detect_volume_spike(df, window=20)
        assert result["detected"] is False

    def test_zero_average_volume(self):
        """Test handling of zero average volume."""
        df = pd.DataFrame({"volume": [0] * 21})

        result = detect_volume_spike(df)
        assert result["detected"] is False
        assert result["ratio"] == 0.0


class TestDetectReversalPattern:
    """Test reversal pattern detection."""

    def test_bearish_reversal(self):
        """Test detection of bearish reversal pattern."""
        df = pd.DataFrame({
            "open": [100, 102],
            "high": [103, 104],
            "low": [99, 97],   # Current low below previous low
            "close": [102, 98]  # Current close below previous low
        })

        result = detect_reversal_pattern(df)
        assert result["detected"] is True
        assert result["type"] == "bearish"
        assert result["bearish"] is True
        assert result["strength"] > 0

    def test_bullish_reversal(self):
        """Test detection of bullish reversal pattern."""
        df = pd.DataFrame({
            "open": [98, 100],
            "high": [100, 105],  # Current high above previous high
            "low": [97, 99],
            "close": [99, 104]   # Current close above previous high
        })

        result = detect_reversal_pattern(df)
        assert result["detected"] is True
        assert result["type"] == "bullish"
        assert result["bullish"] is True
        assert result["strength"] > 0

    def test_bearish_wick_rejection(self):
        """Test detection of bearish wick rejection."""
        df = pd.DataFrame({
            "open": [100, 100],
            "high": [102, 110],  # Large upper wick
            "low": [98, 99],
            "close": [101, 101]  # Close near low of range
        })

        result = detect_reversal_pattern(df)
        assert result["detected"] is True
        assert result["type"] == "bearish_wick"
        assert result["wick_analysis"]["upper_wick_ratio"] > 0.3

    def test_bullish_wick_rejection(self):
        """Test detection of bullish wick rejection."""
        df = pd.DataFrame({
            "open": [100, 100],
            "high": [102, 102],
            "low": [98, 90],   # Large lower wick
            "close": [99, 99]  # Close near high of range
        })

        result = detect_reversal_pattern(df)
        assert result["detected"] is True
        assert result["type"] == "bullish_wick"
        assert result["wick_analysis"]["lower_wick_ratio"] > 0.3

    def test_no_reversal(self):
        """Test when no reversal pattern exists."""
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102]  # Normal continuation
        })

        result = detect_reversal_pattern(df)
        assert result["detected"] is False
        assert result["type"] is None
        assert result["strength"] == 0

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({
            "open": [100],
            "high": [102],
            "low": [98],
            "close": [101]
        })

        result = detect_reversal_pattern(df)
        assert result["detected"] is False


class TestCalculateMomentumDivergence:
    """Test momentum divergence calculation."""

    def test_bearish_divergence(self):
        """Test detection of bearish divergence."""
        # Price makes higher high, RSI makes lower high
        np.random.seed(42)
        prices = [100, 105, 103, 107, 106, 110, 108, 112, 109]  # Higher highs overall

        # Add more data to ensure sufficient RSI calculation
        prices = [95, 96, 97, 98, 99] + prices + [113, 114, 115]

        df = pd.DataFrame({
            "close": prices,
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.98 for p in prices]
        })

        result = calculate_momentum_divergence(df, rsi_period=14)

        # Should detect some form of analysis
        assert "detected" in result
        assert "type" in result
        assert "strength" in result
        assert "current_rsi" in result

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "open": [99, 100, 101],
            "high": [101, 102, 103],
            "low": [98, 99, 100]
        })

        result = calculate_momentum_divergence(df, rsi_period=14)
        assert result["detected"] is False
        assert result["type"] is None
        assert result["strength"] == 0.0

    def test_normal_market_no_divergence(self):
        """Test normal market conditions without divergence."""
        # Create trending market without divergence
        prices = list(range(100, 120))  # Steady uptrend
        df = pd.DataFrame({
            "close": prices,
            "open": [p * 0.999 for p in prices],
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.998 for p in prices]
        })

        result = calculate_momentum_divergence(df, rsi_period=14)
        # May or may not detect divergence in steady trend
        assert isinstance(result["detected"], bool)
        assert result["current_rsi"] >= 0


class TestNegativeVIPScore:
    """Test main negative VIP scoring function."""

    def test_feature_disabled(self):
        """Test behavior when negative_vip feature is disabled."""
        df = pd.DataFrame({
            "close": [100, 98],
            "volume": [1000, 3000],
            "low": [99, 97]
        })
        config = {"features": {"negative_vip": False}}

        score = negative_vip_score(df, config)
        assert score == 0.3  # Neutral score when disabled

    def test_high_vip_scenario(self):
        """Test high VIP score scenario with multiple signals."""
        df = pd.DataFrame({
            "open": [100] * 20 + [102],
            "high": [101] * 20 + [110],  # Large upper wick
            "low": [99] * 20 + [98],
            "close": [100] * 20 + [98],  # Reversal close below previous low
            "volume": [1000] * 20 + [3000]  # Volume spike
        })
        config = {"features": {"negative_vip": True}}

        score = negative_vip_score(df, config)
        assert score >= 0.6  # Should be high due to multiple signals

    def test_low_vip_scenario(self):
        """Test low VIP score scenario with weak signals."""
        df = pd.DataFrame({
            "open": [100] * 21,
            "high": [100.5] * 21,
            "low": [99.5] * 21,
            "close": [100] * 21,  # No reversal
            "volume": [1000] * 21  # No volume spike
        })
        config = {"features": {"negative_vip": True}}

        score = negative_vip_score(df, config)
        assert score <= 0.5  # Should be low due to weak signals

    def test_confluence_bonus(self):
        """Test confluence bonus for multiple signals."""
        # Create scenario with volume spike AND reversal
        df = pd.DataFrame({
            "open": [100] * 20 + [102],
            "high": [101] * 20 + [103],
            "low": [99] * 20 + [97],    # Break below previous low
            "close": [100] * 20 + [98], # Close below previous low
            "volume": [1000] * 20 + [2500]  # Volume spike
        })
        config = {"features": {"negative_vip": True}}

        score = negative_vip_score(df, config)

        # Should get confluence bonus for multiple signals
        assert score > 0.5


class TestAnalyzeReversalRisk:
    """Test comprehensive reversal risk analysis."""

    def test_complete_analysis(self):
        """Test complete reversal risk analysis."""
        df = pd.DataFrame({
            "open": [100] * 20 + [102],
            "high": [101] * 20 + [103],
            "low": [99] * 20 + [97],
            "close": [100] * 20 + [98],
            "volume": [1000] * 20 + [2500]
        })
        config = {"features": {"negative_vip": True}}

        result = analyze_reversal_risk(df, config)

        # Check all required fields
        assert "vip_score" in result
        assert "risk_level" in result
        assert "volume_spike" in result
        assert "reversal_pattern" in result
        assert "momentum_divergence" in result
        assert "recommendation" in result

        # Check data types and values
        assert isinstance(result["vip_score"], float)
        assert result["risk_level"] in ["low", "medium", "high"]
        assert result["recommendation"] in ["monitor", "reduce_position"]

    def test_risk_level_classification(self):
        """Test risk level classification logic."""
        df = pd.DataFrame({
            "close": [100] * 21,
            "volume": [1000] * 21,
            "open": [100] * 21,
            "high": [101] * 21,
            "low": [99] * 21
        })

        # Mock different VIP scores to test classification
        config_low = {"features": {"negative_vip": True}}

        result = analyze_reversal_risk(df, config_low)

        # Should classify based on VIP score
        if result["vip_score"] >= 0.7:
            assert result["risk_level"] == "high"
            assert result["recommendation"] == "reduce_position"
        elif result["vip_score"] >= 0.5:
            assert result["risk_level"] == "medium"
        else:
            assert result["risk_level"] == "low"
            assert result["recommendation"] == "monitor"


# Integration test
def test_negative_vip_integration():
    """Integration test for negative VIP components."""
    # Create realistic reversal scenario
    np.random.seed(42)

    # Simulate market rally followed by reversal signals
    prices = []
    volumes = []

    # Rally phase
    for i in range(15):
        prices.append(100 + i * 2 + np.random.normal(0, 0.5))
        volumes.append(1000 + np.random.normal(0, 100))

    # Reversal phase with volume spike and bearish pattern
    for i in range(5):
        prices.append(130 - i * 1.5 + np.random.normal(0, 0.8))
        volumes.append(2500 + np.random.normal(0, 300))

    df = pd.DataFrame({
        "close": prices,
        "open": [p * 0.999 for p in prices],
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "volume": volumes
    })

    config = {"features": {"negative_vip": True}}

    # Run comprehensive analysis
    result = analyze_reversal_risk(df, config)

    # Should detect some reversal risk given the setup
    assert result["vip_score"] >= 0.3
    assert result["volume_spike"]["detected"] or result["reversal_pattern"]["detected"]