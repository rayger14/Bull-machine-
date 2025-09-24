#!/usr/bin/env python3
"""
Bull Machine v1.4.1 - Smoke Tests
Essential production readiness validation
"""

import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import traceback
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bull_machine.scoring.fusion import FusionEngineV141
from bull_machine.modules.wyckoff.state_machine import WyckoffStateMachine
from bull_machine.modules.liquidity.imbalance import calculate_liquidity_score
from bull_machine.modules.risk.dynamic_risk import (
    calculate_dynamic_position_size,
    calculate_stop_loss,
)


class SmokeTestSuite:
    """Production readiness smoke tests."""

    def __init__(self):
        self.results = {}
        self.config_path = "configs/v141/profile_balanced.json"
        self.load_config()

    def load_config(self):
        """Load and merge configuration."""
        with open(self.config_path) as f:
            base_config = json.load(f)

        if "extends" in base_config:
            system_path = "configs/v141/system_config.json"
            with open(system_path) as f:
                system_config = json.load(f)

            # Merge configs
            self.config = {**system_config, **base_config}
            for key in ["signals", "quality_floors", "risk_management"]:
                if key in base_config:
                    self.config.setdefault(key, {})
                    self.config[key].update(base_config[key])
        else:
            self.config = base_config

    def test_configuration_loading(self) -> bool:
        """Test that all configurations load correctly."""
        try:
            # Test Balanced profile
            assert self.config["signals"]["enter_threshold"] == 0.69
            assert self.config["quality_floors"]["wyckoff"] == 0.37
            assert self.config["quality_floors"]["liquidity"] == 0.32

            # Test exits config path exists
            exits_path = Path(self.config["exits_config_path"])
            assert exits_path.exists()

            return True
        except Exception as e:
            self.results["config_error"] = str(e)
            return False

    def test_fusion_engine_initialization(self) -> bool:
        """Test fusion engine initializes correctly."""
        try:
            engine = FusionEngineV141(self.config)

            # Test weight normalization
            total_weight = sum(engine.weights.values())
            assert abs(total_weight - 1.0) < 0.01

            # Test required components
            assert "wyckoff" in engine.weights
            assert "liquidity" in engine.weights
            assert engine.weights["wyckoff"] > 0.25  # Should be ~0.26 after normalization

            return True
        except Exception as e:
            self.results["fusion_error"] = str(e)
            return False

    def test_layer_scoring(self) -> bool:
        """Test all layer scoring functions."""
        try:
            # Create synthetic data
            np.random.seed(42)
            dates = pd.date_range("2024-01-01", periods=100, freq="h")
            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": np.random.uniform(45000, 55000, 100),
                    "high": np.random.uniform(46000, 56000, 100),
                    "low": np.random.uniform(44000, 54000, 100),
                    "close": np.random.uniform(45000, 55000, 100),
                    "volume": np.random.lognormal(10, 0.5, 100),
                }
            )
            df.set_index("timestamp", inplace=True)

            # Fix OHLC consistency
            for i in range(len(df)):
                high = max(df.iloc[i][["open", "close"]].max(), df.iloc[i]["high"])
                low = min(df.iloc[i][["open", "close"]].min(), df.iloc[i]["low"])
                df.iloc[i, df.columns.get_loc("high")] = high
                df.iloc[i, df.columns.get_loc("low")] = low

            # Test Wyckoff
            wyckoff_machine = WyckoffStateMachine()
            wyckoff_result = wyckoff_machine.analyze_wyckoff_state(df)
            assert "confidence" in wyckoff_result
            assert 0 <= wyckoff_result["confidence"] <= 1

            # Test Liquidity
            liquidity_result = calculate_liquidity_score(df)
            assert "score" in liquidity_result
            assert 0 <= liquidity_result["score"] <= 1

            # Test Dynamic Risk
            liquidity_data = {"pools": [], "cluster_score": 0.1}
            risk_result = calculate_dynamic_position_size(0.01, df, liquidity_data)
            assert "adjusted_risk_pct" in risk_result
            assert risk_result["adjusted_risk_pct"] > 0

            # Test Stop Loss
            stop_price = calculate_stop_loss(df, "long", 50000, 0.5, 500)
            assert stop_price < 50000  # Should be below entry for long

            return True
        except Exception as e:
            self.results["layer_error"] = str(e)
            return False

    def test_fusion_scoring(self) -> bool:
        """Test complete fusion scoring workflow."""
        try:
            engine = FusionEngineV141(self.config)

            # Test with various score combinations
            test_cases = [
                # High quality trade
                {
                    "wyckoff": 0.80,
                    "liquidity": 0.75,
                    "structure": 0.65,
                    "momentum": 0.60,
                    "volume": 0.70,
                    "context": 0.50,
                    "mtf": 0.65,
                },
                # Marginal trade
                {
                    "wyckoff": 0.38,
                    "liquidity": 0.33,
                    "structure": 0.45,
                    "momentum": 0.40,
                    "volume": 0.50,
                    "context": 0.40,
                    "mtf": 0.55,
                },
                # Poor quality trade
                {
                    "wyckoff": 0.30,
                    "liquidity": 0.25,
                    "structure": 0.35,
                    "momentum": 0.30,
                    "volume": 0.30,
                    "context": 0.20,
                    "mtf": 0.40,
                },
            ]

            for i, scores in enumerate(test_cases):
                result = engine.fuse_scores(scores, quality_floors=self.config["quality_floors"])

                assert "weighted_score" in result
                assert 0 <= result["weighted_score"] <= 1
                assert "global_veto" in result
                assert isinstance(result["global_veto"], bool)

                # Check entry decision
                should_enter = engine.should_enter(result)
                assert isinstance(should_enter, bool)

            return True
        except Exception as e:
            self.results["fusion_scoring_error"] = str(e)
            return False

    def test_parameter_enforcement(self) -> bool:
        """Test that parameter enforcement works."""
        try:
            # This should fail due to missing required keys
            try:
                from bull_machine.strategy.exits.advanced_rules import MarkupSOWUTWarning

                incomplete_config = {"enabled": True}  # Missing required keys
                shared_config = {"atr_period": 14}

                # This should raise ValueError
                rule = MarkupSOWUTWarning(incomplete_config, shared_config)
                return False  # Should not reach here
            except ValueError:
                pass  # Expected behavior

            return True
        except Exception as e:
            self.results["param_enforcement_error"] = str(e)
            return False

    def test_telemetry_generation(self) -> bool:
        """Test that telemetry files can be generated."""
        try:
            # Create telemetry directory
            telemetry_dir = Path("reports/smoke_test_telemetry")
            telemetry_dir.mkdir(parents=True, exist_ok=True)

            # Generate sample telemetry
            sample_data = {
                "timestamp": "2024-01-01T12:00:00",
                "phase_c_trap_score": 0.15,
                "reclaim_speed_bonus": 0.08,
                "cluster_score": 0.12,
                "regime_veto": False,
                "mtf_override": True,
            }

            # Write telemetry files
            files_to_write = ["layer_masks.json", "parameter_usage.json", "exit_counts.json"]

            for filename in files_to_write:
                filepath = telemetry_dir / filename
                with open(filepath, "w") as f:
                    json.dump(sample_data, f, indent=2)

                assert filepath.exists()
                assert filepath.stat().st_size > 0

            return True
        except Exception as e:
            self.results["telemetry_error"] = str(e)
            return False

    def test_determinism(self) -> bool:
        """Test that identical inputs produce identical outputs."""
        try:
            engine1 = FusionEngineV141(self.config)
            engine2 = FusionEngineV141(self.config)

            test_scores = {
                "wyckoff": 0.70,
                "liquidity": 0.65,
                "structure": 0.60,
                "momentum": 0.55,
                "volume": 0.60,
                "context": 0.45,
                "mtf": 0.70,
            }

            result1 = engine1.fuse_scores(test_scores, self.config["quality_floors"])
            result2 = engine2.fuse_scores(test_scores, self.config["quality_floors"])

            # Check key results are identical
            assert abs(result1["weighted_score"] - result2["weighted_score"]) < 1e-10
            assert result1["global_veto"] == result2["global_veto"]
            assert result1["mtf_gate"] == result2["mtf_gate"]

            return True
        except Exception as e:
            self.results["determinism_error"] = str(e)
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete smoke test suite."""
        tests = [
            ("Configuration Loading", self.test_configuration_loading),
            ("Fusion Engine Init", self.test_fusion_engine_initialization),
            ("Layer Scoring", self.test_layer_scoring),
            ("Fusion Scoring", self.test_fusion_scoring),
            ("Parameter Enforcement", self.test_parameter_enforcement),
            ("Telemetry Generation", self.test_telemetry_generation),
            ("Determinism", self.test_determinism),
        ]

        results = {}
        passed = 0
        total = len(tests)

        print("🧪 Bull Machine v1.4.1 - Smoke Test Suite")
        print("=" * 50)

        for test_name, test_func in tests:
            try:
                success = test_func()
                status = "✅ PASS" if success else "❌ FAIL"
                results[test_name] = success
                if success:
                    passed += 1

                print(f"{test_name:<25} {status}")

                if not success and test_name.lower() in self.results:
                    print(f"   Error: {self.results[test_name.lower().replace(' ', '_') + '_error']}")

            except Exception as e:
                results[test_name] = False
                print(f"{test_name:<25} ❌ FAIL")
                print(f"   Error: {str(e)}")

        print("\n" + "=" * 50)
        print(f"Results: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

        if passed == total:
            print("🎉 ALL SMOKE TESTS PASSED - SYSTEM READY")
        else:
            print("⚠️  SMOKE TESTS FAILED - SYSTEM NOT READY")

        return results


def main():
    """Run smoke test suite."""
    suite = SmokeTestSuite()
    results = suite.run_all_tests()

    # Save results
    results_dir = Path("reports/smoke_test_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "smoke_test_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": pd.Timestamp.now().isoformat(),
                "results": results,
                "passed": sum(results.values()),
                "total": len(results),
                "pass_rate": sum(results.values()) / len(results) * 100,
            },
            f,
            indent=2,
        )

    print(f"\n📁 Results saved to: {results_file}")

    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
