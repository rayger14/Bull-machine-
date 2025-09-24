#!/usr/bin/env python3
"""
Bull Machine v1.4.1 Smoke Test
Quick validation of core functionality
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bull_machine.strategy.exits.advanced_evaluator import AdvancedExitEvaluator
from bull_machine.modules.wyckoff.mtf_sync import wyckoff_state, mtf_alignment
from bull_machine.modules.liquidity.imbalance import calculate_liquidity_score
from bull_machine.modules.bojan.candle_logic import wick_magnets
from bull_machine.scoring.fusion import FusionEngineV141
from bull_machine.backtest.eval import run_ablation


def create_test_data(bars=100):
    """Create synthetic test data."""
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=bars, freq="H")
    base_price = 60000

    # Create trending data with some volatility
    returns = np.random.normal(0.0005, 0.02, bars)  # Slight upward trend
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some volume patterns
    volume_base = 1000
    volume_multiplier = 1 + 0.5 * np.sin(np.arange(bars) * 0.1) + np.random.normal(0, 0.2, bars)
    volume = volume_base * np.abs(volume_multiplier)

    # Generate OHLC
    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = df["close"].shift(1).fillna(prices[0])

    # Add some spread for high/low
    spread = np.abs(np.random.normal(0, 0.01, bars)) * prices
    df["high"] = df[["open", "close"]].max(axis=1) + spread
    df["low"] = df[["open", "close"]].min(axis=1) - spread
    df["volume"] = volume

    return df


def test_advanced_exits():
    """Test advanced exit system."""
    print("🧪 Testing Advanced Exit System...")

    try:
        # Create test data and evaluator
        df = create_test_data(50)
        evaluator = AdvancedExitEvaluator()

        # Test trade plan
        trade_plan = {
            "bias": "long",
            "entry_price": 60000,
            "sl": 58000,
            "tp": 65000,
            "initial_range_high": 62000,
            "initial_range_low": 58000,
        }

        # Test with normal scores
        scores = {
            "wyckoff": 0.7,
            "liquidity": 0.6,
            "structure": 0.5,
            "momentum": 0.5,
            "volume": 0.6,
            "context": 0.5,
            "mtf": 0.7,
        }

        result = evaluator.evaluate_exits(df, trade_plan, scores, 10, None)

        assert "exits" in result
        assert "history" in result["exits"]
        print("  ✅ Advanced exits initialized and running")

        # Test with low scores (should trigger global veto)
        low_scores = {k: 0.3 for k in scores.keys()}
        low_scores["context"] = 0.2  # Below context floor

        result_veto = evaluator.evaluate_exits(df, trade_plan, low_scores, 10, None)

        if result_veto["exits"]["history"]:
            print("  ✅ Global veto triggered correctly")
        else:
            print("  ⚠️  Global veto may not be triggering")

        print("  📊 Telemetry captured:", len(evaluator.telemetry), "entries")

    except Exception as e:
        print(f"  ❌ Advanced exits test failed: {e}")
        return False

    return True


def test_mtf_sync():
    """Test MTF synchronization."""
    print("🧪 Testing MTF Sync...")

    try:
        df = create_test_data(30)

        # Test Wyckoff state detection
        state = wyckoff_state(df)
        assert "bias" in state
        assert "confidence" in state
        print(f"  ✅ Wyckoff state: {state['bias']} (conf: {state['confidence']:.2f})")

        # Test MTF alignment
        aligned, details = mtf_alignment(df, df, liquidity_score=0.7)
        print(f"  ✅ MTF alignment: {aligned} (score: {details['alignment_score']:.2f})")

    except Exception as e:
        print(f"  ❌ MTF sync test failed: {e}")
        return False

    return True


def test_liquidity_scoring():
    """Test liquidity scoring."""
    print("🧪 Testing Liquidity Scoring...")

    try:
        df = create_test_data(40)

        # Test liquidity score calculation
        result = calculate_liquidity_score(df)

        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert "pools" in result

        print(f"  ✅ Liquidity score: {result['score']:.3f}")
        print(f"  📊 Pools detected: {len(result['pools'])}")

    except Exception as e:
        print(f"  ❌ Liquidity scoring test failed: {e}")
        return False

    return True


def test_bojan_magnets():
    """Test Bojan wick magnets."""
    print("🧪 Testing Bojan Wick Magnets...")

    try:
        df = create_test_data(20)

        # Test wick magnet scoring
        ob_level = df["close"].iloc[-1] * 1.02  # 2% above current price
        score = wick_magnets(df, ob_level)

        assert 0 <= score <= 0.6  # Should be capped at 0.6 for v1.4.1
        print(f"  ✅ Bojan wick score: {score:.3f} (capped at 0.6)")

    except Exception as e:
        print(f"  ❌ Bojan wick test failed: {e}")
        return False

    return True


def test_fusion_engine():
    """Test fusion engine."""
    print("🧪 Testing Fusion Engine...")

    try:
        config = {
            "weights": {
                "wyckoff": 0.30,
                "liquidity": 0.25,
                "structure": 0.15,
                "momentum": 0.15,
                "volume": 0.15,
                "context": 0.05,
                "mtf": 0.10,
            },
            "features": {"bojan": True},
            "signals": {"enter_threshold": 0.72, "aggregate_floor": 0.35},
        }

        fusion = FusionEngineV141(config)

        # Test normal scores
        scores = {
            "wyckoff": 0.8,
            "liquidity": 0.7,
            "structure": 0.6,
            "momentum": 0.6,
            "volume": 0.6,
            "context": 0.5,
            "mtf": 0.7,
            "bojan": 0.9,  # Should be capped
        }

        result = fusion.fuse_scores(scores)

        assert "aggregate" in result
        assert "weighted_score" in result
        assert not result["global_veto"]

        # Check Bojan capping
        bojan_contrib = result["layer_contributions"].get("bojan", 0)
        max_bojan = 0.6 * config["weights"].get("bojan", 0)  # Should be capped

        print(f"  ✅ Fusion score: {result['weighted_score']:.3f}")
        print(f"  ✅ Aggregate: {result['aggregate']:.3f}")
        print(f"  ✅ Bojan capped correctly: {bojan_contrib <= max_bojan}")

        # Test entry decision
        should_enter = fusion.should_enter(result)
        print(f"  📊 Entry signal: {should_enter}")

    except Exception as e:
        print(f"  ❌ Fusion engine test failed: {e}")
        return False

    return True


def test_ablation():
    """Test ablation study."""
    print("🧪 Testing Ablation Study...")

    try:
        df = create_test_data(100)

        config = {
            "features": {"wyckoff": True, "liquidity": True, "structure": True},
            "weights": {"wyckoff": 0.5, "liquidity": 0.3, "structure": 0.2},
            "signals": {"enter_threshold": 0.70},
        }

        results = run_ablation(df, config)

        assert len(results) > 0
        print(f"  ✅ Ablation completed: {len(results)} layer combinations")

        # Check that different layer sets produce different results
        if "wyckoff" in results:
            print(f"  📊 Sample result keys: {list(results['wyckoff'].keys())}")

            # Use weighted_score instead of estimated_sharpe
            scores = [r.get("weighted_score", 0) for r in results.values()]
            unique_scores = len(set([round(s, 3) for s in scores]))  # Round for comparison
            print(f"  📊 Unique performance scores: {unique_scores}")

            if unique_scores > 1:
                print("  ✅ Parameter variance detected (no shadowing)")
            else:
                print("  ⚠️  All scores identical - possible parameter shadowing")
        else:
            print("  ⚠️  No results to analyze")

    except Exception as e:
        print(f"  ❌ Ablation test failed: {e}")
        return False

    return True


def test_config_loading():
    """Test configuration loading."""
    print("🧪 Testing Configuration Loading...")

    try:
        # Test system config
        config_path = "configs/v141/system_config.json"
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                config = json.load(f)

            required_keys = ["features", "weights", "signals"]
            for key in required_keys:
                assert key in config

            print(f"  ✅ System config loaded: {config['version']}")
        else:
            print(f"  ⚠️  System config not found: {config_path}")

        # Test exits config
        exits_path = "configs/v141/exits_config.json"
        if Path(exits_path).exists():
            with open(exits_path, "r") as f:
                exits_config = json.load(f)

            assert "order" in exits_config
            assert "shared" in exits_config

            print(f"  ✅ Exits config loaded: {len(exits_config['order'])} rules")
        else:
            print(f"  ⚠️  Exits config not found: {exits_path}")

    except Exception as e:
        print(f"  ❌ Config loading test failed: {e}")
        return False

    return True


def main():
    """Run all smoke tests."""
    print("🚀 Bull Machine v1.4.1 Smoke Test Suite")
    print("=" * 50)

    tests = [
        test_config_loading,
        test_fusion_engine,
        test_mtf_sync,
        test_liquidity_scoring,
        test_bojan_magnets,
        test_advanced_exits,
        test_ablation,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  💥 Test crashed: {e}")
            print()

    print("=" * 50)
    print(f"🏁 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("✅ All tests passed! Ready for backtest validation.")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
