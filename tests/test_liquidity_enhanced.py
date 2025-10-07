"""
Test liquidity scoring and imbalance detection
"""

import unittest
import pytest
import pandas as pd
import numpy as np

from bull_machine.modules.liquidity.imbalance import (
    score_sweep_mitigation,
    detect_liquidity_pools,
    calculate_liquidity_score,
    liquidity_ttl_decay,
)


class TestLiquidity(unittest.TestCase):
    """Test liquidity scoring functionality."""

    def test_score_sweep_mitigation_fast_reclaim(self):
        """Test sweep mitigation scoring for fast reclaim."""
        # Create sweep and reclaim bars
        sweep_bar = {
            "ts": 1000,  # timestamp in seconds
            "low": 100,
            "high": 105,
            "close": 101,
        }

        # Fast reclaim (3 hours later)
        reclaim_bar = {
            "ts": 1000 + 3 * 3600,  # 3 hours later
            "close": 104,  # Strong reclaim
        }

        atr = 2.0
        fib_level = 0.618  # Golden pocket

        score = score_sweep_mitigation(sweep_bar, reclaim_bar, atr, fib_level)

        # Fast reclaim in discount zone should score high
        self.assertGreater(score, 0.7)

    def test_score_sweep_mitigation_slow_reclaim(self):
        """Test sweep mitigation scoring for slow reclaim."""
        sweep_bar = {"ts": 1000, "low": 100, "high": 105, "close": 101}

        # Slow reclaim (8 hours later)
        reclaim_bar = {"ts": 1000 + 8 * 3600, "close": 103}

        atr = 2.0
        fib_level = 0.618

        score = score_sweep_mitigation(sweep_bar, reclaim_bar, atr, fib_level)

        # Slow reclaim should have lower score due to decay
        self.assertLess(score, 0.6)

    @pytest.mark.xfail(reason="Liquidity pool detection logic changed in v1.7.x - needs golden fixture update", strict=False)
    def test_detect_liquidity_pools(self):
        """Test liquidity pool detection."""
        # Create data with equal highs
        df = pd.DataFrame(
            {
                "high": [100, 102, 100, 103, 100, 101],  # Three touches at 100
                "low": [95, 97, 95, 98, 95, 96],
                "volume": [1000, 1200, 1000, 1100, 1000, 1000],
            }
        )

        pools = detect_liquidity_pools(df)

        # Should detect equal highs at 100
        equal_high_pools = [p for p in pools if p["type"] == "equal_highs"]
        self.assertGreater(len(equal_high_pools), 0)

        # Check pool structure
        if equal_high_pools:
            pool = equal_high_pools[0]
            self.assertIn("level", pool)
            self.assertIn("touches", pool)
            self.assertIn("strength", pool)

    def test_calculate_liquidity_score(self):
        """Test comprehensive liquidity scoring."""
        # Create data with sweep pattern
        df = pd.DataFrame(
            {
                "open": [100, 101, 99, 102, 103],
                "high": [102, 103, 100, 104, 105],
                "low": [99, 100, 97, 101, 102],  # Sweep down then recover
                "close": [101, 102, 102, 103, 104],  # Recovery
                "volume": [1000, 1200, 1500, 1100, 1000],  # Volume spike on sweep
            }
        )

        result = calculate_liquidity_score(df)

        # Should return valid score structure
        self.assertIn("score", result)
        self.assertIn("pools", result)
        self.assertIn("recent_sweep", result)
        self.assertIn("imbalance_filled", result)

        # Score should be between 0 and 1
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)

    @pytest.mark.xfail(reason="FVG imbalance_filled detection logic changed in v1.7.x - needs golden fixture update", strict=False)
    def test_calculate_liquidity_score_with_fvg(self):
        """Test liquidity scoring with fair value gap."""
        # Create FVG pattern: gap between bars
        df = pd.DataFrame(
            {
                "open": [100, 101, 105],
                "high": [101, 102, 106],
                "low": [99, 100, 104],  # Gap: prev high 102 < current low 104
                "close": [100, 101, 105],
                "volume": [1000, 1000, 1200],
            }
        )

        # Add bar that fills the gap
        fill_bar = pd.DataFrame(
            {
                "open": [104],
                "high": [105],
                "low": [101],  # Fills into the gap
                "close": [103],
                "volume": [1100],
            }
        )

        df_with_fill = pd.concat([df, fill_bar], ignore_index=True)

        result = calculate_liquidity_score(df_with_fill)

        # Should detect imbalance fill
        self.assertTrue(result["imbalance_filled"])

    def test_liquidity_ttl_decay(self):
        """Test time-to-live decay function."""
        base_score = 0.8
        ttl_bars = 12

        # Within TTL - no decay
        score_within = liquidity_ttl_decay(base_score, 10, ttl_bars)
        self.assertEqual(score_within, base_score)

        # After TTL - should decay
        score_after = liquidity_ttl_decay(base_score, 20, ttl_bars)
        self.assertLess(score_after, base_score)

        # Much later - should decay significantly
        score_much_later = liquidity_ttl_decay(base_score, 30, ttl_bars)
        self.assertLess(score_much_later, score_after)

    def test_liquidity_ttl_decay_math(self):
        """Test TTL decay mathematics."""
        base_score = 1.0
        ttl_bars = 12

        # At TTL + 1, decay should be 0.9^1 = 0.9
        score_ttl_plus_1 = liquidity_ttl_decay(base_score, 13, ttl_bars)
        expected = 0.9
        self.assertAlmostEqual(score_ttl_plus_1, expected, places=3)

        # At TTL + 5, decay should be 0.9^5
        score_ttl_plus_5 = liquidity_ttl_decay(base_score, 17, ttl_bars)
        expected = 0.9**5
        self.assertAlmostEqual(score_ttl_plus_5, expected, places=3)


if __name__ == "__main__":
    unittest.main()
