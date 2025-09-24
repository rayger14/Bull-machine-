#!/usr/bin/env python3
"""
Acceptance Test: Exit Parameter Divergence

Tests 3 configs that only differ by one exit parameter to prove fixes work.
Expected: Different exit counts proving parameters are applied.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path


def create_base_config():
    """Base config template for all tests."""
    return {
        "run_id": "acceptance_test",
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
                "bars_confirm": 2,  # Will vary this
                "min_break_strength": 0.05,
            },
            "momentum_fade": {
                "lookback": 6,
                "drop_pct": 0.20,  # Will vary this
                "min_bars_in_pos": 4,
            },
            "time_stop": {
                "max_bars_1h": 20,  # Will vary this
                "max_bars_4h": 8,
                "max_bars_1d": 4,
            },
        },
        "logging": {"level": "INFO", "emit_fusion_debug": False, "emit_exit_debug": True},
    }


def run_test_config(test_name, config_overrides):
    """Run a single test configuration."""
    config = create_base_config()
    config["run_id"] = f"accept_{test_name}"

    # Apply config overrides
    for path, value in config_overrides.items():
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    # Create temp config file
    temp_dir = tempfile.mkdtemp(prefix=f"accept_{test_name}_")
    config_path = Path(temp_dir) / f"{test_name}.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    result_dir = Path(temp_dir) / "results"

    print(f"🧪 Running {test_name} test...")
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
            timeout=120,
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
    print("🔬 EXIT PARAMETER ACCEPTANCE TEST")
    print("=" * 50)
    print("Testing that parameter changes produce different exit counts")
    print()

    # Test configs that change only one parameter each
    tests = [
        ("baseline", {}),  # No changes - baseline
        ("choch_strict", {"exit_signals.choch_against.bars_confirm": 1}),  # More strict CHoCH
        (
            "momentum_loose",
            {"exit_signals.momentum_fade.drop_pct": 0.30},
        ),  # Less sensitive momentum
        ("time_tight", {"exit_signals.time_stop.max_bars_1h": 10}),  # Tighter time stops
    ]

    results = {}

    for test_name, overrides in tests:
        result = run_test_config(test_name, overrides)
        if result:
            results[test_name] = result

    print("\n" + "=" * 50)
    print("📊 ACCEPTANCE TEST RESULTS")
    print("=" * 50)

    if len(results) < 2:
        print("❌ Not enough successful runs to compare")
        return

    # Print comparison table
    print(f"{'Test':<15} {'CHoCH':<8} {'Momentum':<10} {'TimeStop':<10} {'None':<8}")
    print("-" * 60)

    for test_name, counts in results.items():
        print(
            f"{test_name:<15} {counts.get('choch_against', 0):<8} "
            f"{counts.get('momentum_fade', 0):<10} {counts.get('time_stop', 0):<10} "
            f"{counts.get('none', 0):<8}"
        )

    # Analyze divergence
    print(f"\n🎯 DIVERGENCE ANALYSIS:")

    if "baseline" in results:
        baseline = results["baseline"]

        for test_name, counts in results.items():
            if test_name == "baseline":
                continue

            # Check if this test differs from baseline
            differences = []
            for exit_type in ["choch_against", "momentum_fade", "time_stop"]:
                baseline_count = baseline.get(exit_type, 0)
                test_count = counts.get(exit_type, 0)
                if baseline_count != test_count:
                    diff = test_count - baseline_count
                    differences.append(f"{exit_type}: {baseline_count} → {test_count} ({diff:+d})")

            if differences:
                print(f"✅ {test_name}: DIVERGED from baseline")
                for diff in differences:
                    print(f"   {diff}")
            else:
                print(f"❌ {test_name}: IDENTICAL to baseline (parameters not applied!)")

    # Success criteria
    unique_results = len(set(str(sorted(r.items())) for r in results.values()))
    total_tests = len(results)

    print(f"\n🏆 SUMMARY:")
    print(f"   Tests run: {total_tests}")
    print(f"   Unique results: {unique_results}")

    if unique_results == total_tests:
        print(f"✅ SUCCESS: All {total_tests} configs produced different results!")
        print(f"✅ Exit parameter fixes are working correctly.")
    elif unique_results > 1:
        print(f"⚠️  PARTIAL: {unique_results}/{total_tests} configs produced unique results")
        print(f"⚠️  Some parameters may not be applied correctly")
    else:
        print(f"❌ FAILURE: All configs produced identical results")
        print(f"❌ Exit parameters are not being applied")


if __name__ == "__main__":
    main()
