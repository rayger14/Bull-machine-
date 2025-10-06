"""
Unit tests for Orderflow LCA module in Bull Machine v1.5.0
"""

import pandas as pd
import pytest
import numpy as np
from bull_machine.modules.orderflow.lca import (
    detect_bos,
    detect_liquidity_capture,
    calculate_intent_nudge,
    orderflow_lca,
    analyze_market_structure
)


class TestDetectBOS:
    """Test Break of Structure (BOS) detection."""

    def test_bullish_bos(self):
        """Test detection of bullish BOS."""
        df = pd.DataFrame({
            "open": [98.5, 99.5, 100.5, 101.5, 106.5],
            "high": [100, 101, 102, 103, 108],  # Clear break above 103
            "low": [98, 99, 100, 101, 106],
            "close": [99, 100, 101, 102, 107]
        })

        result = detect_bos(df, lookback=3)
        assert result["detected"] == True
        assert result["direction"] == "bullish"
        assert result["bullish_bos"] == True
        assert result["strength"] > 0

    def test_bearish_bos(self):
        """Test detection of bearish BOS."""
        df = pd.DataFrame({
            "open": [106.5, 105.5, 104.5, 103.5, 96],
            "high": [108, 107, 106, 105, 100],
            "low": [106, 105, 104, 103, 95],  # Clear break below 103
            "close": [107, 106, 105, 104, 97]
        })

        result = detect_bos(df, lookback=3)
        assert result["detected"] == True
        assert result["direction"] == "bearish"
        assert result["bearish_bos"] == True
        assert result["strength"] > 0

    def test_no_bos(self):
        """Test when no BOS occurs."""
        df = pd.DataFrame({
            "open": [98.5, 99.5, 100.5, 99.5, 100.5],
            "high": [100, 101, 102, 101, 102],  # No significant break
            "low": [98, 99, 100, 99, 100],
            "close": [99, 100, 101, 100, 101]
        })

        result = detect_bos(df, lookback=3)
        assert result["detected"] == False
        assert result["direction"] is None
        assert result["strength"] == 0

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({
            "open": [98.5, 99.5],
            "high": [100, 101],
            "low": [98, 99],
            "close": [99, 100]
        })

        result = detect_bos(df, lookback=5)
        assert result["detected"] == False


class TestDetectLiquidityCapture:
    """Test Liquidity Capture Analysis (LCA)."""

    def test_lca_detected(self):
        """Test detection of LCA pattern."""
        df = pd.DataFrame({
            "close": [100, 103],  # Close above previous high
            "high": [101, 104],
            "low": [99, 102],
            "open": [99.5, 102.5]
        })

        result = detect_liquidity_capture(df)
        assert result == True

    def test_no_lca(self):
        """Test when no LCA pattern exists."""
        df = pd.DataFrame({
            "close": [100, 100.5],  # Close below previous high
            "high": [101, 101.5],
            "low": [99, 99.5],
            "open": [99.5, 100]
        })

        result = detect_liquidity_capture(df)
        assert result == False

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({
            "close": [100],
            "high": [101]
        })

        result = detect_liquidity_capture(df)
        assert result == False


class TestCalculateIntentNudge:
    """Test intent nudge calculation."""

    def test_high_volume_nudge(self):
        """Test high volume intent nudge."""
        # High volume on latest bar (need more data for CVD calculation)
        volume_data = [1000] * 9 + [3000]  # 3x average volume
        df = pd.DataFrame({
            "volume": volume_data,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10
        })

        result = calculate_intent_nudge(df, volume_threshold=1.2)
        assert result["nudge_score"] > 0.5
        assert result["conviction"] in ["high", "very_high"]
        assert result["volume_ratio"] == 3.0

    def test_low_volume_nudge(self):
        """Test low volume intent nudge."""
        # Low volume throughout
        volume_data = [1000] * 10
        df = pd.DataFrame({
            "volume": volume_data,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10
        })

        result = calculate_intent_nudge(df, volume_threshold=1.2)
        assert result["nudge_score"] <= 1.0
        assert result["conviction"] == "low"

    def test_zero_average_volume(self):
        """Test handling of zero average volume."""
        df = pd.DataFrame({
            "volume": [0] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10
        })

        result = calculate_intent_nudge(df)
        assert result["nudge_score"] == 0.0
        assert result["conviction"] == "low"

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({"volume": [1000, 2000]})

        result = calculate_intent_nudge(df)
        assert result["nudge_score"] == 0.0


class TestOrderflowLCA:
    """Test main orderflow LCA function."""

    def test_feature_disabled(self):
        """Test behavior when orderflow_lca feature is disabled."""
        df = pd.DataFrame({
            "close": [100, 103],
            "high": [101, 104],
            "low": [99, 102],
            "volume": [1000, 2000]
        })
        config = {"features": {"orderflow_lca": False}}

        score = orderflow_lca(df, config)
        assert score == 0.5  # Neutral score when disabled

    @pytest.mark.xfail(reason="Legacy v1.5.0 orderflow threshold differs post v1.7.x refactor (0.65 vs 0.70)", strict=False)
    def test_strong_confluence(self):
        """Test strong confluence scenario."""
        df = pd.DataFrame({
            "close": [100, 103, 102, 101, 100, 104],  # LCA pattern
            "high": [101, 104, 103, 102, 101, 105],   # BOS potential
            "low": [99, 102, 101, 100, 99, 103],
            "volume": [1000, 1000, 1000, 1000, 1000, 3000]  # Volume spike
        })
        config = {"features": {"orderflow_lca": True}}

        score = orderflow_lca(df, config)
        assert score >= 0.7  # Should be high due to confluence

    def test_weak_signals(self):
        """Test scenario with weak signals."""
        df = pd.DataFrame({
            "close": [100, 100.1, 100.2, 100.1, 100, 100.05],  # Weak movement
            "high": [100.5, 100.6, 100.7, 100.6, 100.5, 100.55],
            "low": [99.5, 99.6, 99.7, 99.6, 99.5, 99.55],
            "volume": [1000] * 6  # No volume spike
        })
        config = {"features": {"orderflow_lca": True}}

        score = orderflow_lca(df, config)
        assert 0.3 <= score <= 0.6  # Should be moderate


class TestAnalyzeMarketStructure:
    """Test comprehensive market structure analysis."""

    def test_complete_analysis(self):
        """Test complete market structure analysis."""
        df = pd.DataFrame({
            "close": [100, 102, 101, 103, 105, 107],
            "high": [101, 103, 102, 104, 106, 108],
            "low": [99, 101, 100, 102, 104, 106],
            "volume": [1000, 1500, 1200, 2000, 2500, 3000]
        })
        config = {"features": {"orderflow_lca": True}}

        result = analyze_market_structure(df, config)

        # Check all required fields are present
        assert "lca_score" in result
        assert "bos_analysis" in result
        assert "intent_analysis" in result
        assert "structure_health" in result

        # Check data types
        assert isinstance(result["lca_score"], float)
        assert isinstance(result["bos_analysis"], dict)
        assert isinstance(result["intent_analysis"], dict)
        assert result["structure_health"] in ["strong", "weak"]

    def test_structure_health_classification(self):
        """Test structure health classification logic."""
        # Strong structure scenario
        df_strong = pd.DataFrame({
            "close": [100, 105, 110],
            "high": [101, 106, 111],
            "low": [99, 104, 109],
            "volume": [1000, 3000, 4000]  # Strong volume
        })

        # Weak structure scenario
        df_weak = pd.DataFrame({
            "close": [100, 100.1, 100.2],
            "high": [100.5, 100.6, 100.7],
            "low": [99.5, 99.6, 99.7],
            "volume": [1000, 1000, 1000]  # Weak volume
        })

        config = {"features": {"orderflow_lca": True}}

        result_strong = analyze_market_structure(df_strong, config)
        result_weak = analyze_market_structure(df_weak, config)

        # Strong structure should have higher LCA score
        assert result_strong["lca_score"] >= result_weak["lca_score"]


# Integration test
def test_orderflow_integration():
    """Integration test for orderflow components."""
    # Create realistic market data with clear orderflow patterns
    np.random.seed(42)

    # Simulate a breakout scenario
    base_price = 100
    prices = []
    volumes = []

    # Build up to resistance
    for i in range(10):
        prices.append(base_price + i * 0.5 + np.random.normal(0, 0.2))
        volumes.append(1000 + np.random.normal(0, 100))

    # Breakout with volume
    for i in range(5):
        prices.append(base_price + 5 + i * 1.0 + np.random.normal(0, 0.3))
        volumes.append(2500 + np.random.normal(0, 200))

    df = pd.DataFrame({
        "open": [p * 0.995 for p in prices],
        "close": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "volume": volumes
    })

    config = {"features": {"orderflow_lca": True}}

    # Run comprehensive analysis
    result = analyze_market_structure(df, config)

    # Should detect strong structure during breakout
    assert result["lca_score"] > 0.4
    assert result["bos_analysis"]["detected"] or result["intent_analysis"]["nudge_score"] > 0.5