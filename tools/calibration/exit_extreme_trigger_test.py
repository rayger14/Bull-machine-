#!/usr/bin/env python3
"""
Extreme Exit Parameter Test: Force CHoCH/Momentum triggers at all costs

Based on coverage report showing 0% CHoCH/Momentum triggers, this test uses
ultra-aggressive parameters to finally get these exit types to fire.

Analysis from exit_coverage_report.json:
- CHoCH: 0/12 tests (0% coverage) - need more aggressive break detection
- Momentum: 0/12 tests (0% coverage) - need lower drop thresholds
- TimeStop: 12/12 tests (100% coverage) - working correctly

Strategy: Push parameters to extreme limits to force triggers.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path

def create_base_config():
    """Base config optimized for exit signal generation."""
    return {
        "run_id": "extreme_exit_test",
        "data": {
            "sources": {
                "BTCUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv"
            },
            "timeframes": ["1H"],
            "schema": {
                "timestamp": {"name": "time", "unit": "s"},
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            }
        },
        "broker": {
            "fee_bps": 10,
            "slippage_bps": 5,
            "spread_bps": 2,
            "partial_fill": True
        },
        "portfolio": {
            "starting_cash": 100000,
            "exposure_cap_pct": 0.60,
            "max_positions": 4
        },
        "engine": {
            "lookback_bars": 50,
            "seed": 42
        },
        "strategy": {
            "version": "v1.4",
            "config": "bull_machine/configs/diagnostic_v14_step4_config.json"
        },
        "risk": {
            "base_risk_pct": 0.008,
            "max_risk_per_trade": 0.02,
            "min_stop_pct": 0.001
        },
        "exit_signals": {
            "enabled": True,
            "enabled_exits": ["choch_against", "momentum_fade", "time_stop"],
            "emit_exit_edge_logs": True,
            "choch_against": {
                "swing_lookback": 3,
                "bars_confirm": 2,
                "min_break_strength": 0.05
            },
            "momentum_fade": {
                "lookback": 6,
                "drop_pct": 0.20,
                "min_bars_in_pos": 4
            },
            "time_stop": {
                "max_bars_1h": 24,
                "max_bars_4h": 8,
                "max_bars_1d": 4
            },
            "emit_exit_debug": True
        },
        "logging": {
            "level": "INFO",
            "emit_fusion_debug": False,
            "emit_exit_debug": True
        }
    }

def run_extreme_test(test_name, config_overrides):
    """Run single extreme parameter test."""
    config = create_base_config()
    config["run_id"] = f"extreme_{test_name}"

    # Apply config overrides
    for path, value in config_overrides.items():
        keys = path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    # Create temp config file
    temp_dir = tempfile.mkdtemp(prefix=f"extreme_{test_name}_")
    config_path = Path(temp_dir) / f"{test_name}.json"

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    result_dir = Path(temp_dir) / "results"

    print(f"🚀 Running EXTREME {test_name} test...")
    try:
        result = subprocess.run([
            "python3", "-m", "bull_machine.app.main_backtest",
            "--config", str(config_path),
            "--out", str(result_dir)
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            # Read exit counts
            exit_counts_path = result_dir / "exit_counts.json"
            if exit_counts_path.exists():
                with open(exit_counts_path, 'r') as f:
                    exit_counts = json.load(f)

                print(f"✅ {test_name}: {exit_counts}")
                return exit_counts
            else:
                print(f"❌ {test_name}: No exit_counts.json found")
                return None
        else:
            print(f"❌ {test_name}: Failed with code {result.returncode}")
            print(f"Error: {result.stderr[:300]}")
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
    print("🚀 EXTREME EXIT PARAMETER TEST")
    print("=" * 60)
    print("MISSION: Force CHoCH/Momentum to trigger with ultra-aggressive params")
    print("Based on exit_coverage_report.json showing 0% CHoCH/Momentum coverage")
    print()

    # Extreme parameter tests designed to force triggers
    extreme_tests = [
        # Test 1: Baseline (for comparison)
        ("baseline", {}),

        # Test 2: Ultra-aggressive CHoCH
        ("choch_ultra_aggressive", {
            "exit_signals.choch_against.bars_confirm": 1,      # Instant confirmation
            "exit_signals.choch_against.min_break_strength": 0.005,  # 0.5% break (was 5%)
            "exit_signals.choch_against.swing_lookback": 2     # Very short lookback
        }),

        # Test 3: Hyper-sensitive Momentum
        ("momentum_hyper_sensitive", {
            "exit_signals.momentum_fade.drop_pct": 0.05,      # 5% drop (was 20%)
            "exit_signals.momentum_fade.lookback": 3,         # Shorter lookback
            "exit_signals.momentum_fade.min_bars_in_pos": 1   # Exit immediately
        }),

        # Test 4: Nuclear CHoCH (extreme of extreme)
        ("choch_nuclear", {
            "exit_signals.choch_against.bars_confirm": 1,
            "exit_signals.choch_against.min_break_strength": 0.001,  # 0.1% break
            "exit_signals.choch_against.swing_lookback": 1     # Single bar lookback
        }),

        # Test 5: Trigger-happy Momentum
        ("momentum_trigger_happy", {
            "exit_signals.momentum_fade.drop_pct": 0.02,      # 2% drop
            "exit_signals.momentum_fade.lookback": 2,         # Minimal lookback
            "exit_signals.momentum_fade.min_bars_in_pos": 1
        }),

        # Test 6: Everything extreme
        ("everything_extreme", {
            "exit_signals.choch_against.bars_confirm": 1,
            "exit_signals.choch_against.min_break_strength": 0.001,
            "exit_signals.choch_against.swing_lookback": 1,
            "exit_signals.momentum_fade.drop_pct": 0.02,
            "exit_signals.momentum_fade.lookback": 2,
            "exit_signals.momentum_fade.min_bars_in_pos": 1,
            "exit_signals.time_stop.max_bars_1h": 6            # Also make TimeStop aggressive
        }),

        # Test 7: Disable TimeStop to force other exits
        ("no_timestop_force_others", {
            "exit_signals.enabled_exits": ["choch_against", "momentum_fade"],  # Remove time_stop
            "exit_signals.choch_against.bars_confirm": 1,
            "exit_signals.choch_against.min_break_strength": 0.01,
            "exit_signals.momentum_fade.drop_pct": 0.08,
            "exit_signals.momentum_fade.min_bars_in_pos": 2
        })
    ]

    results = {}

    for test_name, overrides in extreme_tests:
        result = run_extreme_test(test_name, overrides)
        if result:
            results[test_name] = result

    print("\\n" + "=" * 60)
    print("🎯 EXTREME TEST RESULTS")
    print("=" * 60)

    if len(results) < 2:
        print("❌ Not enough successful runs to analyze")
        return

    # Print comparison table
    print(f"{'Test':<25} {'CHoCH':<8} {'Momentum':<10} {'TimeStop':<10} {'None':<8}")
    print("-" * 70)

    for test_name, counts in results.items():
        print(f"{test_name:<25} {counts.get('choch_against', 0):<8} "
              f"{counts.get('momentum_fade', 0):<10} {counts.get('time_stop', 0):<10} "
              f"{counts.get('none', 0):<8}")

    # Final analysis
    print(f"\\n🏆 EXTREME TRIGGER ANALYSIS:")

    choch_triggered = any(counts.get('choch_against', 0) > 0 for counts in results.values())
    momentum_triggered = any(counts.get('momentum_fade', 0) > 0 for counts in results.values())

    print(f"CHoCH finally triggered: {'✅ YES' if choch_triggered else '❌ NO'}")
    print(f"Momentum finally triggered: {'✅ YES' if momentum_triggered else '❌ NO'}")

    if choch_triggered or momentum_triggered:
        print(f"\\n🎉 SUCCESS: At least one exit type was forced to trigger!")

        # Identify which configs worked
        working_configs = []
        for test_name, counts in results.items():
            if counts.get('choch_against', 0) > 0 or counts.get('momentum_fade', 0) > 0:
                working_configs.append(test_name)

        print(f"Working configs: {', '.join(working_configs)}")

    else:
        print(f"\\n❌ EXTREME FAILURE: Even ultra-aggressive params couldn't trigger CHoCH/Momentum")
        print(f"This suggests:")
        print(f"  1. Exit logic bugs beyond parameter tuning")
        print(f"  2. Market data lacks the required patterns")
        print(f"  3. Strategy entry conditions prevent exit scenarios")

    # Save results for analysis
    with open("extreme_exit_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\n📄 Results saved to: extreme_exit_test_results.json")

if __name__ == "__main__":
    main()