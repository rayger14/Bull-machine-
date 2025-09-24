#!/usr/bin/env python3
"""
Quality Floor Sweep Runner - 3x3 Grid Calibration

Implements surgical floor calibration around Step 3 starting values:
- wyckoff: 0.40, 0.42, 0.44
- structure: 0.43, 0.45, 0.47
- liquidity: 0.36, 0.38, 0.40

Keeps momentum/volume/context fixed for speed.
Records trades, win%, expectancy, max DD for each configuration.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from itertools import product

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_floor_config(wyckoff_floor, structure_floor, liquidity_floor, run_id):
    """Create diagnostic config with specific floor values."""
    config = {
        "run_id": f"floor_sweep_{run_id}",
        "fusion": {
            "enter_threshold": 0.30,
            "weights": {
                "wyckoff": 0.30,
                "liquidity": 0.25,
                "structure": 0.20,
                "momentum": 0.10,
                "volume": 0.10,
                "context": 0.05,
            },
            "quality_floors": {
                "wyckoff": wyckoff_floor,
                "liquidity": liquidity_floor,
                "structure": structure_floor,
                "momentum": 0.12,  # Fixed
                "volume": 0.18,  # Fixed
                "context": 0.22,  # Fixed
            },
            "triad": {
                "require": True,
                "members": ["wyckoff", "structure", "liquidity"],
                "min_pass": 2,
                "override": {"enabled": True, "if_sum_of_two_cores_ge": 1.10},
            },
            "layer_caps": {"volume": 0.25, "context": 0.30},
            "min_variance_guard": {"momentum": 0.08, "volume": 0.12, "context": 0.15},
            "hysteresis": {"enter": 0.30, "hold": 0.27},
            "mtf": {"gate": False},
            "eq_magnet_gate": False,
        },
        "mode": {"enter_threshold": 0.30},
        "meta": {
            "description": f"Floor sweep: W={wyckoff_floor:.2f}, S={structure_floor:.2f}, L={liquidity_floor:.2f}",
            "sweep_params": {
                "wyckoff": wyckoff_floor,
                "structure": structure_floor,
                "liquidity": liquidity_floor,
            },
        },
    }
    return config


def create_backtest_config(strategy_config_path, run_id):
    """Create backtest config pointing to strategy config."""
    config = {
        "run_id": f"floor_sweep_backtest_{run_id}",
        "data": {
            "sources": {
                "BTCUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv",
                "ETHUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv",
            },
            "timeframes": ["1H"],
        },
        "broker": {"fee_bps": 10, "slippage_bps": 5, "spread_bps": 2, "partial_fill": True},
        "portfolio": {"starting_cash": 100000, "exposure_cap_pct": 0.65, "max_positions": 6},
        "engine": {"lookback_bars": 100, "seed": 42},
        "strategy": {"version": "v1.4", "config": strategy_config_path},
        "risk": {
            "base_risk_pct": 0.0075,
            "max_risk_per_trade": 0.05,
            "tp_ladder": {
                "tp1": {"r_multiple": 1.0, "size_pct": 33, "action": "move_stop_to_breakeven"},
                "tp2": {"r_multiple": 2.0, "size_pct": 33, "action": "trail_stop"},
                "tp3": {"r_multiple": 3.0, "size_pct": 34, "action": "hold"},
            },
        },
        "meta": {"description": f"Quality floor sweep test {run_id}", "floor_sweep": True},
    }
    return config


def run_single_test(wyckoff_floor, structure_floor, liquidity_floor, test_num):
    """Run a single backtest with specific floor configuration."""
    run_id = f"w{wyckoff_floor:.2f}_s{structure_floor:.2f}_l{liquidity_floor:.2f}"

    # Create temp config files
    strategy_config = create_floor_config(wyckoff_floor, structure_floor, liquidity_floor, run_id)
    backtest_config = create_backtest_config(f"temp_strategy_{run_id}.json", run_id)

    # Write temp files
    strategy_config_path = f"temp_strategy_{run_id}.json"
    backtest_config_path = f"temp_backtest_{run_id}.json"

    with open(strategy_config_path, "w") as f:
        json.dump(strategy_config, f, indent=2)

    # Update backtest config to use correct path
    backtest_config["strategy"]["config"] = strategy_config_path
    with open(backtest_config_path, "w") as f:
        json.dump(backtest_config, f, indent=2)

    print(
        f"\\n🧪 Test {test_num}/9: wyckoff={wyckoff_floor:.2f}, structure={structure_floor:.2f}, liquidity={liquidity_floor:.2f}"
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
                f"floor_sweep_results_{run_id}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            # Parse results
            try:
                output_data = json.loads(result.stdout.strip())
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
                    "status": "success",
                }
            except json.JSONDecodeError:
                print(f"❌ Failed to parse JSON output")
                return {
                    "status": "parse_error",
                    "wyckoff_floor": wyckoff_floor,
                    "structure_floor": structure_floor,
                    "liquidity_floor": liquidity_floor,
                }
        else:
            print(f"❌ Backtest failed: {result.stderr}")
            return {
                "status": "backtest_error",
                "wyckoff_floor": wyckoff_floor,
                "structure_floor": structure_floor,
                "liquidity_floor": liquidity_floor,
            }

    except subprocess.TimeoutExpired:
        print(f"❌ Test timeout")
        return {
            "status": "timeout",
            "wyckoff_floor": wyckoff_floor,
            "structure_floor": structure_floor,
            "liquidity_floor": liquidity_floor,
        }

    finally:
        # Cleanup temp files
        try:
            os.unlink(strategy_config_path)
            os.unlink(backtest_config_path)
        except:
            pass


def main():
    print("🔬 Quality Floor Sweep - 3x3 Grid Calibration")
    print("=" * 50)

    # Define sweep ranges (around Step 3 starting values)
    wyckoff_floors = [0.40, 0.42, 0.44]
    structure_floors = [0.43, 0.45, 0.47]
    liquidity_floors = [0.36, 0.38, 0.40]

    results = []
    test_count = 0
    total_tests = len(wyckoff_floors) * len(structure_floors) * len(liquidity_floors)

    print(f"Running {total_tests} configurations...")
    print("wyckoff: [0.40, 0.42, 0.44]")
    print("structure: [0.43, 0.45, 0.47]")
    print("liquidity: [0.36, 0.38, 0.40]")
    print()

    for wyckoff_floor in wyckoff_floors:
        for structure_floor in structure_floors:
            for liquidity_floor in liquidity_floors:
                test_count += 1
                result = run_single_test(wyckoff_floor, structure_floor, liquidity_floor, test_count)
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
    print(
        f"{'Rank':<4} {'Wyckoff':<7} {'Structure':<9} {'Liquidity':<9} {'Trades':<6} {'WinRate':<8} {'Expectancy':<10} {'MaxDD':<8}"
    )
    print("-" * 80)

    for i, result in enumerate(successful_tests[:5]):
        print(
            f"{i + 1:<4} {result['wyckoff_floor']:<7.2f} {result['structure_floor']:<9.2f} {result['liquidity_floor']:<9.2f} "
            f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}"
        )

    # Find sweet spot (best expectancy with >= 25 trades and >= 30% win rate)
    print(f"\\n🎯 QUALITY FILTERS (≥25 trades, ≥30% win rate):")
    print("-" * 80)

    quality_tests = [r for r in successful_tests if r["trades"] >= 25 and r["win_rate"] >= 0.30]
    if quality_tests:
        for i, result in enumerate(quality_tests[:3]):
            print(
                f"{i + 1:<4} {result['wyckoff_floor']:<7.2f} {result['structure_floor']:<9.2f} {result['liquidity_floor']:<9.2f} "
                f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['max_dd']:<8.0f}"
            )

        best_quality = quality_tests[0]
        print(f"\\n🏆 RECOMMENDED CONFIGURATION:")
        print(f"   wyckoff_floor: {best_quality['wyckoff_floor']:.2f}")
        print(f"   structure_floor: {best_quality['structure_floor']:.2f}")
        print(f"   liquidity_floor: {best_quality['liquidity_floor']:.2f}")
        print(
            f"   Performance: {best_quality['trades']} trades, {best_quality['win_rate']:.1%} win rate, {best_quality['expectancy']:.0f} expectancy"
        )
    else:
        print("⚠️  No configurations meet quality criteria")
        print("📋 Using best expectancy regardless of filters:")
        best_overall = successful_tests[0]
        print(f"   wyckoff_floor: {best_overall['wyckoff_floor']:.2f}")
        print(f"   structure_floor: {best_overall['structure_floor']:.2f}")
        print(f"   liquidity_floor: {best_overall['liquidity_floor']:.2f}")

    # Save detailed results
    output_file = "quality_floor_sweep_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "sweep_config": {
                    "wyckoff_floors": wyckoff_floors,
                    "structure_floors": structure_floors,
                    "liquidity_floors": liquidity_floors,
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
