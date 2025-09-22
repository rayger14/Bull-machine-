#!/usr/bin/env python3
"""
Quality Floor Sweep Runner - Fixed Version

Uses the same config structure as our successful validation tests.
Tests different quality floor combinations to optimize performance.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from itertools import product

def create_backtest_config(wyckoff_floor, structure_floor, liquidity_floor, run_id):
    """Create backtest config with modified quality floors."""
    config = {
        "run_id": f"floor_sweep_{run_id}",
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
            "lookback_bars": 100,
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
            }
        },
        "logging": {
            "level": "INFO",
            "emit_fusion_debug": False,
            "emit_exit_debug": True
        },
        "meta": {
            "description": f"Floor sweep: W={wyckoff_floor:.2f}, S={structure_floor:.2f}, L={liquidity_floor:.2f}",
            "sweep_params": {
                "wyckoff": wyckoff_floor,
                "structure": structure_floor,
                "liquidity": liquidity_floor
            }
        }
    }
    return config

def run_single_test(wyckoff_floor, structure_floor, liquidity_floor, test_num):
    """Run a single backtest with specific floor configuration."""
    run_id = f"w{wyckoff_floor:.2f}_s{structure_floor:.2f}_l{liquidity_floor:.2f}"

    # Create modified strategy config
    strategy_config_path = "bull_machine/configs/diagnostic_v14_step4_config.json"

    # Read base strategy config
    with open(strategy_config_path, 'r') as f:
        strategy_config = json.load(f)

    # Update quality floors
    strategy_config["quality_floors"] = {
        "wyckoff": wyckoff_floor,
        "liquidity": liquidity_floor,
        "structure": structure_floor,
        "momentum": 0.12,  # Keep fixed
        "volume": 0.18,    # Keep fixed
        "context": 0.22    # Keep fixed
    }

    # Write temporary strategy config
    temp_strategy_path = f"temp_strategy_{run_id}.json"
    with open(temp_strategy_path, 'w') as f:
        json.dump(strategy_config, f, indent=2)

    # Create backtest config
    backtest_config = create_backtest_config(wyckoff_floor, structure_floor, liquidity_floor, run_id)
    backtest_config["strategy"]["config"] = temp_strategy_path

    # Write backtest config
    backtest_config_path = f"temp_backtest_{run_id}.json"
    with open(backtest_config_path, 'w') as f:
        json.dump(backtest_config, f, indent=2)

    print(f"\\n🧪 Test {test_num}: wyckoff={wyckoff_floor:.2f}, structure={structure_floor:.2f}, liquidity={liquidity_floor:.2f}")

    try:
        # Run backtest
        result = subprocess.run([
            "python3", "-m", "bull_machine.app.main_backtest",
            "--config", backtest_config_path,
            "--out", f"floor_sweep_results_{run_id}"
        ], capture_output=True, text=True, timeout=180)

        if result.returncode == 0:
            # Parse JSON from stdout (first line should be the JSON result)
            try:
                lines = result.stdout.strip().split('\\n')
                json_line = None
                for line in lines:
                    if line.strip().startswith('{"ok":'):
                        json_line = line.strip()
                        break

                if json_line:
                    output_data = json.loads(json_line)
                    metrics = output_data.get("metrics", {})

                    return {
                        "wyckoff_floor": wyckoff_floor,
                        "structure_floor": structure_floor,
                        "liquidity_floor": liquidity_floor,
                        "trades": metrics.get("trades", 0),
                        "win_rate": metrics.get("win_rate", 0.0),
                        "expectancy": metrics.get("expectancy", 0.0),
                        "max_dd": metrics.get("max_dd", 0.0),
                        "sharpe": metrics.get("sharpe", 0.0),
                        "status": "success"
                    }
                else:
                    print(f"❌ No JSON output found")
                    return {"status": "no_json", "wyckoff_floor": wyckoff_floor, "structure_floor": structure_floor, "liquidity_floor": liquidity_floor}
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                return {"status": "parse_error", "wyckoff_floor": wyckoff_floor, "structure_floor": structure_floor, "liquidity_floor": liquidity_floor}
        else:
            print(f"❌ Backtest failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return {"status": "backtest_error", "wyckoff_floor": wyckoff_floor, "structure_floor": structure_floor, "liquidity_floor": liquidity_floor}

    except subprocess.TimeoutExpired:
        print(f"❌ Test timeout")
        return {"status": "timeout", "wyckoff_floor": wyckoff_floor, "structure_floor": structure_floor, "liquidity_floor": liquidity_floor}

    finally:
        # Cleanup temp files
        try:
            os.unlink(temp_strategy_path)
            os.unlink(backtest_config_path)
        except:
            pass

def main():
    print("🔬 Quality Floor Sweep - Fixed Version")
    print("=" * 50)

    # Define sweep ranges
    wyckoff_floors = [0.35, 0.40, 0.45]    # Current: 0.35
    structure_floors = [0.40, 0.45, 0.50]  # Current: 0.40
    liquidity_floors = [0.35, 0.40, 0.45]  # Current: 0.35

    results = []
    test_count = 0
    total_tests = len(wyckoff_floors) * len(structure_floors) * len(liquidity_floors)

    print(f"Running {total_tests} configurations...")
    print(f"wyckoff: {wyckoff_floors}")
    print(f"structure: {structure_floors}")
    print(f"liquidity: {liquidity_floors}")
    print()

    for wyckoff_floor in wyckoff_floors:
        for structure_floor in structure_floors:
            for liquidity_floor in liquidity_floors:
                test_count += 1
                result = run_single_test(wyckoff_floor, structure_floor, liquidity_floor, test_count)
                results.append(result)

                # Quick status print
                if result.get("status") == "success":
                    print(f"✅ {result['trades']} trades, {result['win_rate']:.1%} win rate, {result['expectancy']:.0f} expectancy")
                else:
                    print(f"❌ {result.get('status', 'unknown_error')}")

    # Analysis and output
    print("\\n" + "=" * 50)
    print("📊 QUALITY FLOOR SWEEP RESULTS")
    print("=" * 50)

    # Filter successful tests
    successful_tests = [r for r in results if r.get("status") == "success"]

    if not successful_tests:
        print("❌ No successful tests!")
        return

    # Sort by expectancy (best first)
    successful_tests.sort(key=lambda x: x["expectancy"], reverse=True)

    print(f"\\n🎯 TOP 5 CONFIGURATIONS (by expectancy):")
    print("-" * 80)
    print(f"{'Rank':<4} {'Wyckoff':<7} {'Structure':<9} {'Liquidity':<9} {'Trades':<6} {'WinRate':<8} {'Expectancy':<10} {'MaxDD':<8}")
    print("-" * 80)

    for i, result in enumerate(successful_tests[:5]):
        print(f"{i+1:<4} {result['wyckoff_floor']:<7.2f} {result['structure_floor']:<9.2f} {result['liquidity_floor']:<9.2f} "
              f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}")

    # Find sweet spot (best expectancy with ≥ 3 trades and ≥ 25% win rate)
    print(f"\\n🎯 QUALITY FILTERS (≥3 trades, ≥25% win rate):")
    print("-" * 80)

    quality_tests = [r for r in successful_tests if r["trades"] >= 3 and r["win_rate"] >= 0.25]
    if quality_tests:
        for i, result in enumerate(quality_tests[:3]):
            print(f"{i+1:<4} {result['wyckoff_floor']:<7.2f} {result['structure_floor']:<9.2f} {result['liquidity_floor']:<9.2f} "
                  f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}")

        best_quality = quality_tests[0]
        print(f"\\n🏆 RECOMMENDED CONFIGURATION:")
        print(f"   wyckoff_floor: {best_quality['wyckoff_floor']:.2f}")
        print(f"   structure_floor: {best_quality['structure_floor']:.2f}")
        print(f"   liquidity_floor: {best_quality['liquidity_floor']:.2f}")
        print(f"   Performance: {best_quality['trades']} trades, {best_quality['win_rate']:.1%} win rate, {best_quality['expectancy']:.0f} expectancy")
    else:
        print("⚠️  No configurations meet quality criteria")

    # Save detailed results
    output_file = "quality_floor_sweep_fixed_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "sweep_config": {
                "wyckoff_floors": wyckoff_floors,
                "structure_floors": structure_floors,
                "liquidity_floors": liquidity_floors
            },
            "results": results,
            "successful_tests": len(successful_tests),
            "total_tests": total_tests
        }, f, indent=2)

    print(f"\\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()