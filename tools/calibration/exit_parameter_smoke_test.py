#!/usr/bin/env python3
"""
Exit Parameter Smoke Test: Small 2×2×2 grid

Quick test to confirm CHoCH and Momentum parameters create variance
with broader parameter ranges that might trigger more conditions.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path


def create_base_config():
    """Base config for smoke test."""
    return {
        "run_id": "smoke_test",
        "data": {
            "sources": {"BTCUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv"},
            "timeframes": ["1H"],
            "schema": {
                "timestamp": {"name": "time", "unit": "s"},
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
        },
        "broker": {"fee_bps": 10, "slippage_bps": 5, "spread_bps": 2, "partial_fill": True},
        "portfolio": {"starting_cash": 100000, "exposure_cap_pct": 0.60, "max_positions": 4},
        "engine": {"lookback_bars": 50, "seed": 42},
        "strategy": {
            "version": "v1.4",
            "config": "bull_machine/configs/diagnostic_v14_step4_config.json",
        },
        "risk": {"base_risk_pct": 0.008, "max_risk_per_trade": 0.02, "min_stop_pct": 0.001},
        "exit_signals": {
            "enabled": True,
            "enabled_exits": ["choch_against", "momentum_fade", "time_stop"],
            "emit_exit_edge_logs": True,
            "choch_against": {
                "swing_lookback": 3,
                "bars_confirm": 2,  # Will vary
                "min_break_strength": 0.05,
            },
            "momentum_fade": {
                "lookback": 6,
                "drop_pct": 0.20,  # Will vary
                "min_bars_in_pos": 4,
            },
            "time_stop": {
                "max_bars_1h": 24,  # Will vary
                "max_bars_4h": 8,
                "max_bars_1d": 4,
            },
            "emit_exit_debug": True,
        },
        "logging": {"level": "INFO", "emit_fusion_debug": False, "emit_exit_debug": True},
    }


def run_test_config(test_name, config_overrides):
    """Run a single test configuration."""
    config = create_base_config()
    config["run_id"] = f"smoke_{test_name}"

    # Apply config overrides
    for path, value in config_overrides.items():
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    # Create temp config file
    temp_dir = tempfile.mkdtemp(prefix=f"smoke_{test_name}_")
    config_path = Path(temp_dir) / f"{test_name}.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    result_dir = Path(temp_dir) / "results"

    print(f"🧪 Running {test_name} smoke test...")
    try:
        result = subprocess.run(
            [
                "python3",
                "-m",
                "bull_machine.app.main_backtest",
                "--config",
                str(config_path),
                "--out",
                str(result_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            # Read exit counts
            exit_counts_path = result_dir / "exit_counts.json"
            if exit_counts_path.exists():
                with open(exit_counts_path, "r") as f:
                    exit_counts = json.load(f)

                print(f"✅ {test_name}: {exit_counts}")
                return exit_counts
            else:
                print(f"❌ {test_name}: No exit_counts.json found")
                return None
        else:
            print(f"❌ {test_name}: Failed with code {result.returncode}")
            print(f"Error: {result.stderr[:200]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"❌ {test_name}: Timeout")
        return None
    finally:
        # Cleanup
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    print("🔬 EXIT PARAMETER SMOKE TEST (2×2×2)")
    print("=" * 50)
    print("Testing broader parameter ranges for CHoCH/Momentum trigger conditions")
    print()

    # Smaller 2×2×2 grid with more aggressive settings
    tests = [
        # Test 1: Conservative (baseline)
        ("conservative", {}),
        # Test 2: Aggressive CHoCH (should trigger more)
        (
            "choch_aggressive",
            {
                "exit_signals.choch_against.bars_confirm": 1,  # More sensitive
                "exit_signals.choch_against.min_break_strength": 0.02,  # Lower threshold
            },
        ),
        # Test 3: Sensitive Momentum (should trigger more)
        (
            "momentum_sensitive",
            {
                "exit_signals.momentum_fade.drop_pct": 0.10,  # Lower threshold
                "exit_signals.momentum_fade.min_bars_in_pos": 2,  # Shorter requirement
            },
        ),
        # Test 4: Tight TimeStop (known to work)
        (
            "timestop_tight",
            {
                "exit_signals.time_stop.max_bars_1h": 12  # Tighter than baseline
            },
        ),
        # Test 5: All aggressive
        (
            "all_aggressive",
            {
                "exit_signals.choch_against.bars_confirm": 1,
                "exit_signals.choch_against.min_break_strength": 0.02,
                "exit_signals.momentum_fade.drop_pct": 0.10,
                "exit_signals.momentum_fade.min_bars_in_pos": 2,
                "exit_signals.time_stop.max_bars_1h": 12,
            },
        ),
        # Test 6: Very aggressive CHoCH
        (
            "choch_very_aggressive",
            {
                "exit_signals.choch_against.bars_confirm": 1,
                "exit_signals.choch_against.min_break_strength": 0.01,
                "exit_signals.choch_against.swing_lookback": 2,
            },
        ),
        # Test 7: Very sensitive Momentum
        (
            "momentum_very_sensitive",
            {
                "exit_signals.momentum_fade.drop_pct": 0.05,  # Very low threshold
                "exit_signals.momentum_fade.lookback": 4,
                "exit_signals.momentum_fade.min_bars_in_pos": 1,
            },
        ),
        # Test 8: Conservative TimeStop (baseline)
        (
            "timestop_loose",
            {
                "exit_signals.time_stop.max_bars_1h": 48  # Very loose
            },
        ),
    ]

    results = {}

    for test_name, overrides in tests:
        result = run_test_config(test_name, overrides)
        if result:
            results[test_name] = result

    print("\n" + "=" * 50)
    print("📊 SMOKE TEST RESULTS")
    print("=" * 50)

    if len(results) < 2:
        print("❌ Not enough successful runs to analyze")
        return

    # Print comparison table
    print(f"{'Test':<20} {'CHoCH':<8} {'Momentum':<10} {'TimeStop':<10} {'None':<8}")
    print("-" * 65)

    for test_name, counts in results.items():
        print(
            f"{test_name:<20} {counts.get('choch_against', 0):<8} "
            f"{counts.get('momentum_fade', 0):<10} {counts.get('time_stop', 0):<10} "
            f"{counts.get('none', 0):<8}"
        )

    # Analyze variance
    print(f"\n🎯 VARIANCE ANALYSIS:")

    choch_values = [counts.get("choch_against", 0) for counts in results.values()]
    momentum_values = [counts.get("momentum_fade", 0) for counts in results.values()]
    timestop_values = [counts.get("time_stop", 0) for counts in results.values()]

    choch_variance = len(set(choch_values)) > 1
    momentum_variance = len(set(momentum_values)) > 1
    timestop_variance = len(set(timestop_values)) > 1

    print(f"CHoCH variance: {'✅ YES' if choch_variance else '❌ NO'} (values: {set(choch_values)})")
    print(f"Momentum variance: {'✅ YES' if momentum_variance else '❌ NO'} (values: {set(momentum_values)})")
    print(f"TimeStop variance: {'✅ YES' if timestop_variance else '❌ NO'} (values: {set(timestop_values)})")

    # Check for any CHoCH/Momentum activity at all
    any_choch = any(counts.get("choch_against", 0) > 0 for counts in results.values())
    any_momentum = any(counts.get("momentum_fade", 0) > 0 for counts in results.values())

    print(f"\nAny CHoCH exits: {'✅ YES' if any_choch else '❌ NO'}")
    print(f"Any Momentum exits: {'✅ YES' if any_momentum else '❌ NO'}")

    # Success criteria
    if choch_variance and momentum_variance and timestop_variance:
        print(f"\n🏆 SUCCESS: All exit types show parameter variance!")
    elif timestop_variance:
        print(f"\n⚠️  PARTIAL: TimeStop working, CHoCH/Momentum need investigation")
    else:
        print(f"\n❌ PROBLEM: No parameter variance detected")


if __name__ == "__main__":
    main()
