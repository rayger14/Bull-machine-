"""
Test MTF synchronization and Wyckoff alignment
"""

import unittest
import pandas as pd
import numpy as np

from bull_machine.modules.wyckoff.mtf_sync import wyckoff_state, mtf_alignment, get_mtf_context


class TestWyckoffMTF(unittest.TestCase):
    """Test Wyckoff MTF synchronization."""

    def setUp(self):
        """Set up test data."""
        # Create accumulation pattern (low in range with volume expansion)
        self.accumulation_data = pd.DataFrame(
            {
                "open": [100] * 20,
                "high": [105] * 10 + [110] * 10,
                "low": [95] * 10 + [98] * 10,
                "close": [98] * 10 + [102] * 10,  # Moving up
                "volume": [1000] * 10 + [1500] * 10,  # Volume expansion
            }
        )

        # Create distribution pattern (high in range with volume expansion)
        self.distribution_data = pd.DataFrame(
            {
                "open": [200] * 20,
                "high": [205] * 10 + [202] * 10,
                "low": [195] * 10 + [190] * 10,
                "close": [202] * 10 + [198] * 10,  # Moving down
                "volume": [1000] * 10 + [1500] * 10,  # Volume expansion
            }
        )

        # Create neutral/ranging data
        self.neutral_data = pd.DataFrame(
            {
                "open": [300] * 20,
                "high": [305] * 20,
                "low": [295] * 20,
                "close": [300] * 20,
                "volume": [1000] * 20,
            }
        )

    def test_wyckoff_state_accumulation(self):
        """Test Wyckoff state detection for accumulation."""
        state = wyckoff_state(self.accumulation_data)

        self.assertEqual(state["bias"], "long")
        self.assertGreater(state["confidence"], 0.6)
        self.assertIn("accumulation", state["phase"])

    def test_wyckoff_state_distribution(self):
        """Test Wyckoff state detection for distribution."""
        state = wyckoff_state(self.distribution_data)

        self.assertEqual(state["bias"], "short")
        self.assertGreater(state["confidence"], 0.6)
        self.assertIn("distribution", state["phase"])

    def test_wyckoff_state_neutral(self):
        """Test Wyckoff state detection for neutral/ranging."""
        state = wyckoff_state(self.neutral_data)

        self.assertEqual(state["bias"], "neutral")
        self.assertLessEqual(state["confidence"], 0.6)

    def test_mtf_alignment_matching_bias(self):
        """Test MTF alignment when both timeframes agree."""
        # Both timeframes show accumulation
        aligned, details = mtf_alignment(
            self.accumulation_data, self.accumulation_data, liquidity_score=0.6
        )

        self.assertTrue(aligned)
        self.assertTrue(details["biases_match"])
        self.assertGreater(details["alignment_score"], 0.6)

    def test_mtf_alignment_desync_with_liquidity_override(self):
        """Test MTF desync overridden by high liquidity."""
        # Timeframes disagree but liquidity is high
        aligned, details = mtf_alignment(
            self.accumulation_data,
            self.distribution_data,
            liquidity_score=0.80,  # High liquidity
        )

        self.assertTrue(aligned)  # Should be aligned due to liquidity override
        self.assertFalse(details["biases_match"])
        self.assertTrue(details["liquidity_override"])

    def test_mtf_alignment_desync_without_override(self):
        """Test MTF desync without liquidity override."""
        # Timeframes disagree and liquidity is not high enough
        aligned, details = mtf_alignment(
            self.accumulation_data,
            self.distribution_data,
            liquidity_score=0.50,  # Low liquidity
        )

        self.assertFalse(aligned)
        self.assertFalse(details["biases_match"])
        self.assertFalse(details["liquidity_override"])

    def test_mtf_alignment_quality_checks(self):
        """Test that quality checks are enforced."""
        # Create low confidence data
        low_confidence_data = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "close": [100] * 5,
                "volume": [1000] * 5,
            }
        )

        aligned, details = mtf_alignment(
            low_confidence_data, low_confidence_data, liquidity_score=0.6
        )

        # Should fail quality checks due to insufficient data/confidence
        self.assertFalse(aligned)

    def test_get_mtf_context(self):
        """Test MTF context generation."""
        # Create different timeframe data
        df_1h = self.accumulation_data
        df_4h = self.accumulation_data.iloc[::4]  # Sample every 4th bar
        df_1d = self.accumulation_data.iloc[::24]  # Sample every 24th bar

        context = get_mtf_context(df_1h, df_4h, df_1d)

        # Check structure
        self.assertIn("1h", context)
        self.assertIn("4h", context)
        self.assertIn("1d", context)
        self.assertIn("htf_bias", context)
        self.assertIn("htf", context)

        # Check that close_4h is available for exit rules
        self.assertIsNotNone(context["htf"]["close_4h"])

    def test_mtf_context_bias_determination(self):
        """Test HTF bias determination in context."""
        # Create bullish daily pattern (high in range)
        bullish_daily = pd.DataFrame(
            {
                "open": [100],
                "high": [120],
                "low": [90],
                "close": [115],  # 83% of range (> 65%)
            }
        )

        context = get_mtf_context(self.accumulation_data, self.accumulation_data, bullish_daily)

        self.assertEqual(context["htf_bias"], "bullish")
        self.assertTrue(context["htf_resistance_near"])

        # Create bearish daily pattern (low in range)
        bearish_daily = pd.DataFrame(
            {
                "open": [100],
                "high": [120],
                "low": [90],
                "close": [95],  # 17% of range (< 35%)
            }
        )

        context = get_mtf_context(self.accumulation_data, self.accumulation_data, bearish_daily)

        self.assertEqual(context["htf_bias"], "bearish")
        self.assertTrue(context["htf_support_near"])


if __name__ == "__main__":
    unittest.main()
