#!/usr/bin/env python3
"""
Exit Parameter Sweep - Corrected Version

Uses proper config wiring with temporary strategy configs and absolute paths.
Tests different exit parameter combinations to optimize exit effectiveness.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from itertools import product


def run_single_test(bars_confirm, drop_pct, bars_max, test_num):
    """Run a single backtest with specific exit parameter configuration."""
    run_id = f"c{bars_confirm}_d{drop_pct:.0f}_m{bars_max}"

    # Create temporary directory for this test
    temp_dir = tempfile.mkdtemp(prefix=f"exit_test_{test_num}_")
    temp_dir_path = Path(temp_dir)

    try:
        print(f"\\n🧪 Test {test_num}: bars_confirm={bars_confirm}, drop_pct={drop_pct:.1f}%, bars_max={bars_max}")
        print(f"📁 Temp dir: {temp_dir}")

        # Read the base strategy config
        base_strategy_path = Path("bull_machine/configs/diagnostic_v14_step4_config.json")
        with open(base_strategy_path, "r") as f:
            strategy_config = json.load(f)

        # Modify exit parameters
        strategy_config["exit_rules"]["choch_against"]["bars_confirm"] = bars_confirm
        strategy_config["exit_rules"]["momentum_fade"]["drop_pct"] = drop_pct / 100.0
        strategy_config["exit_rules"]["time_stop"]["max_bars_1h"] = bars_max

        # Write temporary strategy config with absolute path
        temp_strategy_path = temp_dir_path / f"strategy_{run_id}.json"
        with open(temp_strategy_path, "w") as f:
            json.dump(strategy_config, f, indent=2)

        print(f"📝 Strategy config: {temp_strategy_path}")

        # Create backtest config that properly references the temp strategy config
        backtest_config = {
            "run_id": f"exit_sweep_{run_id}",
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
            "engine": {"lookback_bars": 100, "seed": 42},
            "strategy": {
                "version": "v1.4",
                "config": str(temp_strategy_path.absolute()),  # ABSOLUTE PATH
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
                "momentum_fade": {
                    "lookback": 6,
                    "drop_pct": drop_pct / 100.0,
                    "min_bars_in_pos": 4,
                },
                "time_stop": {"max_bars_1h": bars_max, "max_bars_4h": 8, "max_bars_1d": 4},
            },
            "logging": {"level": "INFO", "emit_fusion_debug": False, "emit_exit_debug": True},
        }

        # Write backtest config
        temp_backtest_path = temp_dir_path / f"backtest_{run_id}.json"
        with open(temp_backtest_path, "w") as f:
            json.dump(backtest_config, f, indent=2)

        print(f"📝 Backtest config: {temp_backtest_path}")
        print(f"🔗 Strategy config path in backtest: {backtest_config['strategy']['config']}")

        # Run backtest with absolute paths
        result = subprocess.run(
            [
                "python3",
                "-m",
                "bull_machine.app.main_backtest",
                "--config",
                str(temp_backtest_path.absolute()),
                "--out",
                str(temp_dir_path / "results"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            # Parse JSON from stdout
            try:
                lines = result.stdout.strip().split("\\n")
                print(f"🔍 Captured {len(lines)} lines of output")

                json_line = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('{"ok":'):
                        json_line = line.strip()
                        print(f"✅ Found JSON at line {i}: {json_line[:50]}...")
                        break

                if json_line:
                    output_data = json.loads(json_line)
                    metrics = output_data.get("metrics", {})

                    result_data = {
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

                    print(
                        f"✅ {result_data['trades']} trades, {result_data['win_rate']:.1%} win rate, {result_data['expectancy']:.0f} expectancy"
                    )
                    return result_data
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
                print(f"Error: {result.stderr[:500]}...")
            if result.stdout:
                print(f"Last output: {result.stdout[-500:]}")
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
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
            print(f"🗑️  Cleaned up: {temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")


def main():
    print("🔬 Exit Parameter Sweep - Corrected Version")
    print("=" * 50)

    # Define exit parameter ranges
    bars_confirm_values = [1, 2, 3]  # CHoCH bars confirmation
    drop_pct_values = [15, 20, 25]  # Momentum fade drop percentage
    bars_max_values = [18, 24, 30]  # Time stop max bars

    results = []
    test_count = 0
    total_tests = len(bars_confirm_values) * len(drop_pct_values) * len(bars_max_values)

    print(f"Running {total_tests} exit parameter configurations...")
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

    # Analysis and output
    print("\\n" + "=" * 50)
    print("📊 EXIT PARAMETER SWEEP RESULTS")
    print("=" * 50)

    # Filter successful tests
    successful_tests = [r for r in results if r.get("status") == "success"]

    if not successful_tests:
        print("❌ No successful tests!")
        print("\\nFailure breakdown:")
        failure_counts = {}
        for r in results:
            status = r.get("status", "unknown")
            failure_counts[status] = failure_counts.get(status, 0) + 1
        for status, count in failure_counts.items():
            print(f"  {status}: {count}")
        return

    # Sort by expectancy (best first)
    successful_tests.sort(key=lambda x: x["expectancy"], reverse=True)

    print(f"\\n🎯 TOP 5 CONFIGURATIONS (by expectancy):")
    print("-" * 90)
    print(
        f"{'Rank':<4} {'Confirm':<7} {'Drop%':<6} {'MaxBars':<8} {'Trades':<6} {'WinRate':<8} {'Expectancy':<10} {'MaxDD':<8}"
    )
    print("-" * 90)

    for i, result in enumerate(successful_tests[:5]):
        print(
            f"{i + 1:<4} {result['bars_confirm']:<7} {result['drop_pct']:<6.0f} {result['bars_max']:<8} "
            f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}"
        )

    # Find sweet spot (best expectancy with ≥ 3 trades and ≥ 25% win rate)
    print(f"\\n🎯 QUALITY FILTERS (≥3 trades, ≥25% win rate):")
    print("-" * 90)

    quality_tests = [r for r in successful_tests if r["trades"] >= 3 and r["win_rate"] >= 0.25]
    if quality_tests:
        for i, result in enumerate(quality_tests[:3]):
            print(
                f"{i + 1:<4} {result['bars_confirm']:<7} {result['drop_pct']:<6.0f} {result['bars_max']:<8} "
                f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}"
            )

        best_quality = quality_tests[0]
        print(f"\\n🏆 RECOMMENDED EXIT CONFIGURATION:")
        print(f"   bars_confirm: {best_quality['bars_confirm']}")
        print(f"   drop_pct: {best_quality['drop_pct']:.0f}%")
        print(f"   bars_max: {best_quality['bars_max']}")
        print(
            f"   Performance: {best_quality['trades']} trades, {best_quality['win_rate']:.1%} win rate, {best_quality['expectancy']:.0f} expectancy"
        )
    else:
        print("⚠️  No configurations meet quality criteria")

    # Save detailed results
    output_file = "exit_parameter_sweep_corrected_results.json"
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
