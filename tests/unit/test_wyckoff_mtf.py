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
        # Position < 0.35 and vol_expansion > 1.2
        self.accumulation_data = pd.DataFrame(
            {
                "open": [100] * 20,
                "high": [105] * 10 + [110] * 10,
                "low": [95] * 10 + [95] * 10,  # Keep lows at range bottom
                "close": [97] * 10 + [98] * 10,  # Close near lows (position ~0.20)
                "volume": [1000] * 10 + [1600] * 10,  # Volume expansion > 1.2x
            }
        )

        # Create distribution pattern (high in range with volume expansion)
        # Position > 0.65 and vol_expansion > 1.2
        self.distribution_data = pd.DataFrame(
            {
                "open": [200] * 20,
                "high": [205] * 10 + [205] * 10,  # Keep highs at range top
                "low": [195] * 10 + [190] * 10,
                "close": [203] * 10 + [204] * 10,  # Close near highs (position ~0.93)
                "volume": [1000] * 10 + [1600] * 10,  # Volume expansion > 1.2x
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
        # Create bullish daily pattern (high in range, needs >=5 rows)
        bullish_daily = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [120] * 5,
                "low": [90] * 5,
                "close": [95, 100, 105, 110, 115],  # Last close at 83% of range (> 65%)
            }
        )

        context = get_mtf_context(self.accumulation_data, self.accumulation_data, bullish_daily)

        self.assertEqual(context["htf_bias"], "bullish")
        self.assertTrue(context["htf_resistance_near"])

        # Create bearish daily pattern (low in range, needs >=5 rows)
        bearish_daily = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [120] * 5,
                "low": [90] * 5,
                "close": [115, 110, 105, 100, 95],  # Last close at 17% of range (< 35%)
            }
        )

        context = get_mtf_context(self.accumulation_data, self.accumulation_data, bearish_daily)

        self.assertEqual(context["htf_bias"], "bearish")
        self.assertTrue(context["htf_support_near"])


if __name__ == "__main__":
    unittest.main()
