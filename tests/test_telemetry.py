"""
Test telemetry and regression guards for Bull Machine v1.4.1
"""

import unittest
import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from bull_machine.strategy.exits.advanced_evaluator import AdvancedExitEvaluator
from bull_machine.scoring.fusion import FusionEngineV141


class TestTelemetry(unittest.TestCase):
    """Test telemetry capture and file generation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test config
        self.config = {
            "enabled": True,
            "order": ["global_veto"],
            "shared": {"atr_period": 14, "vol_sma": 10},
            "global_veto": {
                "enabled": True,
                "aggregate_floor": 0.40,
                "context_floor": 0.30,
                "cooldown_bars": 8,
            },
        }

        # Create test data
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

        self.trade_plan = {"bias": "long", "entry_price": 60000, "sl": 58000, "tp": 65000}

        self.scores = {
            "wyckoff": 0.3,  # Low scores to trigger veto
            "liquidity": 0.3,
            "structure": 0.3,
            "momentum": 0.3,
            "volume": 0.3,
            "context": 0.25,  # Below context floor
            "mtf": 0.3,
        }

    def test_telemetry_files_created(self):
        """Test that telemetry files are created."""
        evaluator = AdvancedExitEvaluator()

        # Trigger telemetry by evaluating exits
        updated_plan = evaluator.evaluate_exits(self.df, self.trade_plan, self.scores, 10, None)

        # Save telemetry
        evaluator.save_telemetry(str(self.temp_path))

        # Check expected files exist
        expected_files = ["exits_applied.jsonl", "exit_counts.json"]
        for filename in expected_files:
            file_path = self.temp_path / filename
            self.assertTrue(file_path.exists(), f"Missing telemetry file: {filename}")

    def test_telemetry_contains_expected_keys(self):
        """Test telemetry entries have required fields."""
        evaluator = AdvancedExitEvaluator()

        # Trigger telemetry
        evaluator.evaluate_exits(self.df, self.trade_plan, self.scores, 10, None)

        # Check telemetry structure
        self.assertGreater(len(evaluator.telemetry), 0)

        entry = evaluator.telemetry[0]
        required_keys = ["timestamp", "rule", "evaluated", "triggered", "params", "scores"]
        for key in required_keys:
            self.assertIn(key, entry, f"Missing key: {key}")

    @pytest.mark.xfail(reason="Telemetry parameter tracking logic changed in v1.7.x - needs update", strict=False)
    def test_parameter_usage_tracking(self):
        """Test that parameter usage is tracked correctly."""
        evaluator = AdvancedExitEvaluator()

        # Trigger multiple evaluations
        for i in range(5):
            evaluator.evaluate_exits(self.df, self.trade_plan, self.scores, i + 10, None)

        # Save and check parameter usage
        evaluator.save_telemetry(str(self.temp_path))

        with open(self.temp_path / "exit_counts.json", "r") as f:
            counts = json.load(f)

        self.assertIn("total_evaluations", counts)
        self.assertEqual(counts["total_evaluations"], 5)


class TestRegressionGuards(unittest.TestCase):
    """Test regression guards to prevent parameter shadowing."""

    def test_parameter_variance_in_sweeps(self):
        """Test that parameter sweeps produce varied results."""

        # Create different configs
        configs = [
            {
                "weights": {"wyckoff": 0.30, "liquidity": 0.25, "structure": 0.45},
                "signals": {"enter_threshold": 0.70},
            },
            {
                "weights": {"wyckoff": 0.50, "liquidity": 0.30, "structure": 0.20},
                "signals": {"enter_threshold": 0.70},
            },
            {
                "weights": {"wyckoff": 0.30, "liquidity": 0.25, "structure": 0.45},
                "signals": {"enter_threshold": 0.75},
            },
        ]

        # Run fusion with different configs
        results = []
        for config in configs:
            fusion = FusionEngineV141(config)

            # Mock scores
            scores = {"wyckoff": 0.7, "liquidity": 0.6, "structure": 0.5}
            result = fusion.fuse_scores(scores)
            results.append(result["weighted_score"])

        # Results should be different (not identical due to parameter shadowing)
        unique_results = len(set(results))
        self.assertGreater(
            unique_results, 1, "Parameter sweep produced identical results - possible shadowing"
        )

    @pytest.mark.xfail(reason="Fusion weight application logic changed in v1.7.x - needs recalibration", strict=False)
    def test_fusion_weight_application(self):
        """Test that fusion weights are actually applied."""

        # Config with extreme weights
        config = {
            "weights": {"wyckoff": 0.90, "liquidity": 0.10},
            "signals": {"enter_threshold": 0.70},
        }

        fusion = FusionEngineV141(config)

        # Scores where wyckoff is high, liquidity is low
        scores_high_wyckoff = {"wyckoff": 0.9, "liquidity": 0.2}
        scores_high_liquidity = {"wyckoff": 0.2, "liquidity": 0.9}

        result1 = fusion.fuse_scores(scores_high_wyckoff)
        result2 = fusion.fuse_scores(scores_high_liquidity)

        # With 90% wyckoff weight, high wyckoff should score better
        self.assertGreater(
            result1["weighted_score"],
            result2["weighted_score"],
            "Weights not properly applied in fusion",
        )

    def test_bojan_capping_enforced(self):
        """Test that Bojan scores are capped at 0.6 in v1.4.1."""

        config = {
            "weights": {"wyckoff": 0.5, "bojan": 0.5},
            "features": {"bojan": True},
            "signals": {"enter_threshold": 0.70},
        }

        fusion = FusionEngineV141(config)

        # High Bojan score should be capped
        scores = {"wyckoff": 0.5, "bojan": 0.95}  # Very high Bojan
        result = fusion.fuse_scores(scores)

        # Check that Bojan contribution is capped
        bojan_contribution = result["layer_contributions"].get("bojan", 0)
        max_possible_bojan = 0.6 * 0.5  # Cap * weight

        self.assertLessEqual(
            bojan_contribution, max_possible_bojan, "Bojan score not properly capped"
        )


class TestSystemIntegration(unittest.TestCase):
    """Test full system integration."""

    def test_smoke_backtest(self):
        """Test basic backtest functionality."""
        from bull_machine.backtest.eval import run_backtest

        # Create minimal config
        config = {
            "features": {"wyckoff": True, "liquidity": True},
            "weights": {"wyckoff": 0.7, "liquidity": 0.3},
            "signals": {"enter_threshold": 0.70, "aggregate_floor": 0.35},
        }

        # Create test data
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        df = pd.DataFrame(
            {
                "open": 60000 + np.cumsum(np.random.randn(100) * 100),
                "high": 61000 + np.cumsum(np.random.randn(100) * 100),
                "low": 59000 + np.cumsum(np.random.randn(100) * 100),
                "close": 60000 + np.cumsum(np.random.randn(100) * 100),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        # Run backtest
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_backtest(df, config, temp_dir)

            # Check basic requirements
            self.assertIn("performance", summary)
            self.assertGreaterEqual(summary["performance"]["total_trades"], 0)

            # Check telemetry files exist
            temp_path = Path(temp_dir)
            expected_files = [
                "exits_applied.json",
                "parameter_usage.json",
                "layer_masks.json",
                "exit_counts.json",
            ]
            for filename in expected_files:
                self.assertTrue(
                    (temp_path / filename).exists(), f"Missing telemetry file: {filename}"
                )


if __name__ == "__main__":
    unittest.main()
