"""
Test suite for advanced exit rules in Bull Machine v1.4.1
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bull_machine.strategy.exits.advanced_rules import (
    MarkupSOWUTWarning,
    MarkupUTADRejection,
    MarkupExhaustion,
    MarkdownSOSSpringFlip,
    MoneytaurTrailing,
    GlobalVeto,
    BojanExtremeProtection,
    ExitDecision,
)
from bull_machine.strategy.exits.advanced_evaluator import AdvancedExitEvaluator


class TestAdvancedExitRules(unittest.TestCase):
    """Test individual exit rules."""

    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        self.df = pd.DataFrame(
            {
                "open": 60000 + np.cumsum(np.random.randn(100) * 100),
                "high": 61000 + np.cumsum(np.random.randn(100) * 100),
                "low": 59000 + np.cumsum(np.random.randn(100) * 100),
                "close": 60000 + np.cumsum(np.random.randn(100) * 100),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        # Sample trade plan
        self.trade_plan_long = {
            "bias": "long",
            "entry_price": 60000,
            "sl": 58000,
            "tp": 65000,
            "initial_range_high": 62000,
            "initial_range_low": 58000,
            "wyckoff_phase": "D",
            "exits": {
                "history": [],
                "current_action": None,
                "cooldown_bars": 0,
                "current_sl": 58000,
            },
        }

        # Sample confluence scores
        self.scores = {
            "wyckoff": 0.7,
            "liquidity": 0.6,
            "structure": 0.5,
            "momentum": 0.5,
            "volume": 0.6,
            "context": 0.5,
            "mtf": 0.7,
        }

        # Shared config
        self.shared_cfg = {"atr_period": 14, "vol_sma": 10, "range_lookback": 20}

    def test_markup_sow_ut_warning(self):
        """Test SOW/UT warning detection."""
        rule_cfg = {
            "premium_floor": 0.70,
            "vol_divergence_ratio": 0.70,
            "wick_atr_mult": 1.5,
            "mtf_desync_floor": 0.6,
            "veto_needed": 3,
            "partial_pct": 0.25,
            "trail_atr_buffer_R": 0.5,
        }

        rule = MarkupSOWUTWarning(rule_cfg, self.shared_cfg)

        # Modify data to trigger conditions
        self.df.iloc[-1, self.df.columns.get_loc("close")] = 61500  # Premium position
        self.df.iloc[-1, self.df.columns.get_loc("volume")] = 500  # Volume divergence
        self.df.iloc[-1, self.df.columns.get_loc("high")] = 62500  # Big wick

        # Lower MTF score
        self.scores["mtf"] = 0.5

        decision = rule.evaluate(self.df, self.trade_plan_long, self.scores, 15, None)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "partial")
        self.assertEqual(decision.size_pct, 0.25)

    def test_markup_utad_rejection(self):
        """Test UTAD rejection detection."""
        rule_cfg = {
            "wick_close_frac": 0.5,
            "fib_retrace": 0.618,
            "partial_pct": 0.5,
            "trail_to_structure_minus_atr": True,
        }

        rule = MarkupUTADRejection(rule_cfg, self.shared_cfg)

        # Create false breakout conditions
        self.df.iloc[-1, self.df.columns.get_loc("high")] = 62500  # Above prior high
        self.df.iloc[-1, self.df.columns.get_loc("close")] = 61000  # Close in lower half
        self.scores["liquidity"] = 0.3  # Liquidity fail

        decision = rule.evaluate(self.df, self.trade_plan_long, self.scores, 10, None)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "partial")
        self.assertEqual(decision.size_pct, 0.5)

    def test_moneytaur_trailing(self):
        """Test Moneytaur trailing system."""
        rule_cfg = {
            "activate_after_R": 1.0,
            "trail_rule": "max(BE + 0.5R, structure_pivot - 1*ATR)",
            "update_every_bars": 3,
        }

        rule = MoneytaurTrailing(rule_cfg, self.shared_cfg)

        # Set current price for 1.5R profit
        self.df.iloc[-1, self.df.columns.get_loc("close")] = 63000  # 1.5R from entry

        decision = rule.evaluate(
            self.df, self.trade_plan_long, self.scores, 9, None
        )  # 9 bars = multiple of 3

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "trail")
        self.assertIsNotNone(decision.new_sl)
        self.assertGreater(decision.new_sl, self.trade_plan_long["sl"])

    def test_global_veto(self):
        """Test global veto system."""
        rule_cfg = {"aggregate_floor": 0.40, "context_floor": 0.30, "cooldown_bars": 8}

        rule = GlobalVeto(rule_cfg, self.shared_cfg)

        # Lower scores to trigger veto
        low_scores = {k: 0.3 for k in self.scores.keys()}

        decision = rule.evaluate(self.df, self.trade_plan_long, low_scores, 10, None)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "full")
        self.assertEqual(decision.size_pct, 1.0)
        self.assertEqual(decision.cooldown_bars, 8)

    def test_bojan_extreme_protection_disabled(self):
        """Test that Bojan protection is phase-gated."""
        rule_cfg = {
            "enabled": False,  # Phase-gated
            "wick_atr_mult": 2.0,
            "vol_under_sma_mult": 0.5,
            "exit_pct": 0.75,
            "require_htf_alignment": True,
        }

        rule = BojanExtremeProtection(rule_cfg, self.shared_cfg)

        # Create extreme wick conditions
        self.df.iloc[-1, self.df.columns.get_loc("high")] = 65000  # Extreme wick
        self.df.iloc[-1, self.df.columns.get_loc("close")] = 60500
        self.df.iloc[-1, self.df.columns.get_loc("volume")] = 300

        decision = rule.evaluate(self.df, self.trade_plan_long, self.scores, 10, None)

        # Should not trigger when disabled
        self.assertIsNone(decision)


class TestAdvancedExitEvaluator(unittest.TestCase):
    """Test the master evaluator."""

    def setUp(self):
        """Set up test environment."""
        # Create evaluator with default config
        self.evaluator = AdvancedExitEvaluator()

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=50, freq="H")
        self.df = pd.DataFrame(
            {
                "open": 60000 + np.cumsum(np.random.randn(50) * 100),
                "high": 61000 + np.cumsum(np.random.randn(50) * 100),
                "low": 59000 + np.cumsum(np.random.randn(50) * 100),
                "close": 60000 + np.cumsum(np.random.randn(50) * 100),
                "volume": np.random.uniform(1000, 5000, 50),
            },
            index=dates,
        )

        self.trade_plan = {
            "bias": "long",
            "entry_price": 60000,
            "sl": 58000,
            "tp": 65000,
            "wyckoff_phase": "D",
        }

        self.scores = {
            "wyckoff": 0.7,
            "liquidity": 0.6,
            "structure": 0.5,
            "momentum": 0.5,
            "volume": 0.6,
            "context": 0.5,
            "mtf": 0.7,
        }

    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        self.assertTrue(self.evaluator.cfg["enabled"])
        self.assertGreater(len(self.evaluator.rules), 0)

    def test_evaluate_exits(self):
        """Test main evaluation method."""
        updated_plan = self.evaluator.evaluate_exits(
            self.df, self.trade_plan, self.scores, 10, None
        )

        # Should have exits section
        self.assertIn("exits", updated_plan)
        self.assertIn("history", updated_plan["exits"])
        self.assertIn("current_sl", updated_plan["exits"])

    def test_telemetry_logging(self):
        """Test telemetry is captured."""
        # Force a veto by lowering scores
        low_scores = {k: 0.3 for k in self.scores.keys()}

        self.evaluator.evaluate_exits(self.df, self.trade_plan, low_scores, 10, None)

        # Should have telemetry
        self.assertGreater(len(self.evaluator.telemetry), 0)

        # Check telemetry structure
        entry = self.evaluator.telemetry[0]
        self.assertIn("rule", entry)
        self.assertIn("evaluated", entry)
        self.assertIn("triggered", entry)
        self.assertIn("scores", entry)

    def test_cooldown_mechanism(self):
        """Test cooldown prevents repeated exits."""
        # Set cooldown
        self.trade_plan["exits"] = {"cooldown_bars": 5, "history": []}

        # Lower scores to trigger veto
        low_scores = {k: 0.3 for k in self.scores.keys()}

        updated_plan = self.evaluator.evaluate_exits(self.df, self.trade_plan, low_scores, 10, None)

        # Should decrement cooldown
        self.assertEqual(updated_plan["exits"]["cooldown_bars"], 4)

        # Should not add new exits during cooldown
        self.assertEqual(len(updated_plan["exits"]["history"]), 0)


if __name__ == "__main__":
    unittest.main()
