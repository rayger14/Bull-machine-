#!/usr/bin/env python3
"""
Exit Parameter Sweep - Fixed Version

Uses the same config structure as our successful validation tests.
Tests different exit parameter combinations to optimize exit effectiveness.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from itertools import product


def create_backtest_config(bars_confirm, drop_pct, bars_max, run_id):
    """Create backtest config with modified exit parameters."""
    config = {
        "run_id": f"exit_sweep_{run_id}",
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
                "volume": "volume",
            },
        },
        "broker": {"fee_bps": 10, "slippage_bps": 5, "spread_bps": 2, "partial_fill": True},
        "portfolio": {"starting_cash": 100000, "exposure_cap_pct": 0.60, "max_positions": 4},
        "engine": {"lookback_bars": 100, "seed": 42},
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
                "bars_confirm": bars_confirm,
                "min_break_strength": 0.05,
            },
            "momentum_fade": {"lookback": 6, "drop_pct": drop_pct, "min_bars_in_pos": 4},
            "time_stop": {"max_bars_1h": bars_max, "max_bars_4h": 8, "max_bars_1d": 4},
        },
        "logging": {"level": "INFO", "emit_fusion_debug": False, "emit_exit_debug": True},
        "meta": {
            "description": f"Exit sweep: bars_confirm={bars_confirm}, drop_pct={drop_pct:.2f}, bars_max={bars_max}",
            "sweep_params": {
                "bars_confirm": bars_confirm,
                "drop_pct": drop_pct,
                "bars_max": bars_max,
            },
        },
    }
    return config


def run_single_test(bars_confirm, drop_pct, bars_max, test_num):
    """Run a single backtest with specific exit configuration."""
    run_id = f"c{bars_confirm}_d{drop_pct:.2f}_t{bars_max}"

    # Create backtest config
    backtest_config = create_backtest_config(bars_confirm, drop_pct, bars_max, run_id)

    # Write backtest config
    backtest_config_path = f"temp_exit_backtest_{run_id}.json"
    with open(backtest_config_path, "w") as f:
        json.dump(backtest_config, f, indent=2)

    print(
        f"\\n🧪 Test {test_num}: bars_confirm={bars_confirm}, drop_pct={drop_pct:.2f}, bars_max={bars_max}"
    )

    try:
        # Run backtest
        result = subprocess.run(
            [
                "python3",
                "-m",
                "bull_machine.app.main_backtest",
                "--config",
                backtest_config_path,
                "--out",
                f"exit_sweep_results_{run_id}",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode == 0:
            # Parse JSON from stdout (first line should be the JSON result)
            try:
                lines = result.stdout.strip().split("\\n")
                json_line = None
                for line in lines:
                    if line.strip().startswith('{"ok":'):
                        json_line = line.strip()
                        break

                if json_line:
                    output_data = json.loads(json_line)
                    metrics = output_data.get("metrics", {})

                    return {
                        "bars_confirm": bars_confirm,
                        "drop_pct": drop_pct,
                        "bars_max": bars_max,
                        "trades": metrics.get("trades", 0),
                        "win_rate": metrics.get("win_rate", 0.0),
                        "expectancy": metrics.get("expectancy", 0.0),
                        "max_dd": metrics.get("max_dd", 0.0),
                        "sharpe": metrics.get("sharpe", 0.0),
                        "status": "success",
                    }
                else:
                    print(f"❌ No JSON output found")
                    return {
                        "status": "no_json",
                        "bars_confirm": bars_confirm,
                        "drop_pct": drop_pct,
                        "bars_max": bars_max,
                    }
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                return {
                    "status": "parse_error",
                    "bars_confirm": bars_confirm,
                    "drop_pct": drop_pct,
                    "bars_max": bars_max,
                }
        else:
            print(f"❌ Backtest failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return {
                "status": "backtest_error",
                "bars_confirm": bars_confirm,
                "drop_pct": drop_pct,
                "bars_max": bars_max,
            }

    except subprocess.TimeoutExpired:
        print(f"❌ Test timeout")
        return {
            "status": "timeout",
            "bars_confirm": bars_confirm,
            "drop_pct": drop_pct,
            "bars_max": bars_max,
        }

    finally:
        # Cleanup temp files
        try:
            os.unlink(backtest_config_path)
        except:
            pass


def main():
    print("🔬 Exit Parameter Sweep - Fixed Version")
    print("=" * 50)

    # Define sweep ranges (smaller grid for focused testing)
    bars_confirm_values = [1, 2, 3]  # CHoCH confirmation bars
    drop_pct_values = [0.15, 0.20, 0.25]  # Momentum fade threshold
    bars_max_values = [18, 24, 30]  # Time stop (1H bars)

    results = []
    test_count = 0
    total_tests = len(bars_confirm_values) * len(drop_pct_values) * len(bars_max_values)

    print(f"Running {total_tests} configurations...")
    print(f"bars_confirm: {bars_confirm_values}")
    print(f"drop_pct: {drop_pct_values}")
    print(f"bars_max: {bars_max_values}")
    print()

    for bars_confirm in bars_confirm_values:
        for drop_pct in drop_pct_values:
            for bars_max in bars_max_values:
                test_count += 1
                result = run_single_test(bars_confirm, drop_pct, bars_max, test_count)
                results.append(result)

                # Quick status print
                if result.get("status") == "success":
                    print(
                        f"✅ {result['trades']} trades, {result['win_rate']:.1%} win rate, {result['expectancy']:.0f} expectancy"
                    )
                else:
                    print(f"❌ {result.get('status', 'unknown_error')}")

    # Analysis and output
    print("\\n" + "=" * 50)
    print("📊 EXIT PARAMETER SWEEP RESULTS")
    print("=" * 50)

    # Filter successful tests
    successful_tests = [r for r in results if r.get("status") == "success"]

    if not successful_tests:
        print("❌ No successful tests!")
        return

    # Sort by expectancy (best first)
    successful_tests.sort(key=lambda x: x["expectancy"], reverse=True)

    print(f"\\n🎯 TOP 5 CONFIGURATIONS (by expectancy):")
    print("-" * 85)
    print(
        f"{'Rank':<4} {'Confirm':<7} {'Drop%':<7} {'MaxBars':<8} {'Trades':<6} {'WinRate':<8} {'Expectancy':<10} {'MaxDD':<8}"
    )
    print("-" * 85)

    for i, result in enumerate(successful_tests[:5]):
        print(
            f"{i + 1:<4} {result['bars_confirm']:<7} {result['drop_pct']:<7.2f} {result['bars_max']:<8} "
            f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}"
        )

    # Find sweet spot (best expectancy with ≥ 3 trades and ≥ 25% win rate)
    print(f"\\n🎯 QUALITY FILTERS (≥3 trades, ≥25% win rate):")
    print("-" * 85)

    quality_tests = [r for r in successful_tests if r["trades"] >= 3 and r["win_rate"] >= 0.25]
    if quality_tests:
        for i, result in enumerate(quality_tests[:3]):
            print(
                f"{i + 1:<4} {result['bars_confirm']:<7} {result['drop_pct']:<7.2f} {result['bars_max']:<8} "
                f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}"
            )

        best_quality = quality_tests[0]
        print(f"\\n🏆 RECOMMENDED EXIT CONFIGURATION:")
        print(f"   choch_against.bars_confirm: {best_quality['bars_confirm']}")
        print(f"   momentum_fade.drop_pct: {best_quality['drop_pct']:.2f}")
        print(f"   time_stop.max_bars_1h: {best_quality['bars_max']}")
        print(
            f"   Performance: {best_quality['trades']} trades, {best_quality['win_rate']:.1%} win rate, {best_quality['expectancy']:.0f} expectancy"
        )
    else:
        print("⚠️  No configurations meet quality criteria")

    # Save detailed results
    output_file = "exit_parameter_sweep_fixed_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "sweep_config": {
                    "bars_confirm_values": bars_confirm_values,
                    "drop_pct_values": drop_pct_values,
                    "bars_max_values": bars_max_values,
                },
                "results": results,
                "successful_tests": len(successful_tests),
                "total_tests": total_tests,
            },
            f,
            indent=2,
        )

    print(f"\\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
