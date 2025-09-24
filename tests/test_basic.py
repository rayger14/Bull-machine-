#!/usr/bin/env python3
"""Basic tests without pytest dependency"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bull_machine.modules.fusion.advanced import AdvancedFusionEngine
from bull_machine.core.types import Series, Bar, WyckoffResult


def test_config_loads():
    """Test config loading"""
    config_path = "bull_machine/config/config_v1_2_1.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print("✅ Config loads successfully")
        print(f"   Version: {config.get('version')}")
        print(f"   Enter threshold: {config['fusion']['enter_threshold']}")
        return True
    except Exception as e:
        print(f"❌ Config load failed: {e}")
        return False


def test_weights_sum():
    """Test weights sum to 1.0"""
    try:
        with open("bull_machine/config/config_v1_2_1.json", "r") as f:
            config = json.load(f)

        weights = config["fusion"]["weights"]
        total = sum(weights.values())

        if abs(total - 1.0) < 0.01:
            print(f"✅ Weights sum correctly: {total:.3f}")
            return True
        else:
            print(f"❌ Weights sum incorrect: {total:.3f}")
            return False
    except Exception as e:
        print(f"❌ Weight test failed: {e}")
        return False


def test_early_phase_veto():
    """Test early phase veto logic"""
    try:
        config = {
            "features": {"veto_system": True},
            "fusion": {"enter_threshold": 0.40, "weights": {"wyckoff": 0.5, "liquidity": 0.5}},
        }
        engine = AdvancedFusionEngine(config)

        # Create test data
        wy = WyckoffResult(
            regime="trending",
            phase="A",
            bias="long",
            phase_confidence=0.5,
            trend_confidence=0.7,
            range=None,
        )

        modules_data = {
            "wyckoff": wy,
            "liquidity": {"overall_score": 0.5},
            "series": None,  # Simplified
        }

        vetoes = engine._check_vetoes(modules_data)

        if "early_wyckoff_phase" in vetoes:
            print("✅ Early phase veto works (Phase A, conf=0.5 blocked)")
            return True
        else:
            print("❌ Early phase veto failed")
            return False
    except Exception as e:
        print(f"❌ Early phase veto test failed: {e}")
        return False


def test_import_modules():
    """Test that all new modules can be imported"""
    try:
        from bull_machine.modules.structure.advanced import AdvancedStructureAnalyzer
        from bull_machine.modules.momentum.advanced import AdvancedMomentumAnalyzer
        from bull_machine.modules.volume.advanced import AdvancedVolumeAnalyzer
        from bull_machine.modules.context.advanced import AdvancedContextAnalyzer

        print("✅ All new modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False


if __name__ == "__main__":
    print("Running basic v1.2.1 tests...")

    tests = [test_config_loads, test_weights_sum, test_early_phase_veto, test_import_modules]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests PASSED!")
        sys.exit(0)
    else:
        print("⚠️  Some tests FAILED")
        sys.exit(1)
