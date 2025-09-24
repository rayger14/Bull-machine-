#!/usr/bin/env python3
"""
Risk Management Parameter Sweep

Tests different risk management configurations to optimize position sizing,
exposure limits, and risk per trade for better risk-adjusted returns.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from itertools import product

def create_backtest_config(base_risk_pct, max_risk_per_trade, exposure_cap_pct, run_id):
    """Create backtest config with modified risk management parameters."""
    config = {
        "run_id": f"risk_sweep_{run_id}",
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
            "exposure_cap_pct": exposure_cap_pct,
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
            "base_risk_pct": base_risk_pct,
            "max_risk_per_trade": max_risk_per_trade,
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
            "description": f"Risk sweep: base_risk={base_risk_pct:.3f}, max_risk={max_risk_per_trade:.3f}, exposure_cap={exposure_cap_pct:.2f}",
            "sweep_params": {
                "base_risk_pct": base_risk_pct,
                "max_risk_per_trade": max_risk_per_trade,
                "exposure_cap_pct": exposure_cap_pct
            }
        }
    }
    return config

def run_single_test(base_risk_pct, max_risk_per_trade, exposure_cap_pct, test_num):
    """Run a single backtest with specific risk management configuration."""
    run_id = f"br{base_risk_pct:.3f}_mr{max_risk_per_trade:.3f}_ec{exposure_cap_pct:.2f}"

    # Create backtest config
    backtest_config = create_backtest_config(base_risk_pct, max_risk_per_trade, exposure_cap_pct, run_id)

    # Write backtest config
    backtest_config_path = f"temp_risk_backtest_{run_id}.json"
    with open(backtest_config_path, 'w') as f:
        json.dump(backtest_config, f, indent=2)

    print(f"\\n🧪 Test {test_num}: base_risk={base_risk_pct:.3f}, max_risk={max_risk_per_trade:.3f}, exposure_cap={exposure_cap_pct:.2f}")

    try:
        # Run backtest
        result = subprocess.run([
            "python3", "-m", "bull_machine.app.main_backtest",
            "--config", backtest_config_path,
            "--out", f"risk_sweep_results_{run_id}"
        ], capture_output=True, text=True, timeout=180)

        if result.returncode == 0:
            # Parse JSON from stdout (should be the last line with JSON)
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
                        "base_risk_pct": base_risk_pct,
                        "max_risk_per_trade": max_risk_per_trade,
                        "exposure_cap_pct": exposure_cap_pct,
                        "trades": metrics.get("trades", 0),
                        "win_rate": metrics.get("win_rate", 0.0),
                        "expectancy": metrics.get("expectancy", 0.0),
                        "max_dd": metrics.get("max_dd", 0.0),
                        "sharpe": metrics.get("sharpe", 0.0),
                        "cagr": metrics.get("cagr", 0.0),
                        "status": "success"
                    }
                else:
                    print(f"❌ No JSON output found")
                    return {"status": "no_json", "base_risk_pct": base_risk_pct, "max_risk_per_trade": max_risk_per_trade, "exposure_cap_pct": exposure_cap_pct}
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                return {"status": "parse_error", "base_risk_pct": base_risk_pct, "max_risk_per_trade": max_risk_per_trade, "exposure_cap_pct": exposure_cap_pct}
        else:
            print(f"❌ Backtest failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return {"status": "backtest_error", "base_risk_pct": base_risk_pct, "max_risk_per_trade": max_risk_per_trade, "exposure_cap_pct": exposure_cap_pct}

    except subprocess.TimeoutExpired:
        print(f"❌ Test timeout")
        return {"status": "timeout", "base_risk_pct": base_risk_pct, "max_risk_per_trade": max_risk_per_trade, "exposure_cap_pct": exposure_cap_pct}

    finally:
        # Cleanup temp files
        try:
            os.unlink(backtest_config_path)
        except:
            pass

def main():
    print("🔬 Risk Management Parameter Sweep")
    print("=" * 50)

    # Define sweep ranges (focused on key risk management parameters)
    base_risk_values = [0.006, 0.008, 0.010, 0.012]     # Base position size risk
    max_risk_values = [0.015, 0.020, 0.025, 0.030]      # Maximum risk per trade
    exposure_cap_values = [0.50, 0.60, 0.70]            # Portfolio exposure cap

    results = []
    test_count = 0
    total_tests = len(base_risk_values) * len(max_risk_values) * len(exposure_cap_values)

    print(f"Running {total_tests} configurations...")
    print(f"base_risk_pct: {base_risk_values}")
    print(f"max_risk_per_trade: {max_risk_values}")
    print(f"exposure_cap_pct: {exposure_cap_values}")
    print()

    for base_risk_pct in base_risk_values:
        for max_risk_per_trade in max_risk_values:
            for exposure_cap_pct in exposure_cap_values:
                test_count += 1
                result = run_single_test(base_risk_pct, max_risk_per_trade, exposure_cap_pct, test_count)
                results.append(result)

                # Quick status print
                if result.get("status") == "success":
                    print(f"✅ {result['trades']} trades, {result['win_rate']:.1%} win rate, {result['expectancy']:.0f} expectancy, {result['sharpe']:.2f} sharpe")
                else:
                    print(f"❌ {result.get('status', 'unknown_error')}")

    # Analysis and output
    print("\\n" + "=" * 50)
    print("📊 RISK MANAGEMENT SWEEP RESULTS")
    print("=" * 50)

    # Filter successful tests
    successful_tests = [r for r in results if r.get("status") == "success"]

    if not successful_tests:
        print("❌ No successful tests!")
        return

    # Sort by expectancy (best first)
    successful_tests.sort(key=lambda x: x["expectancy"], reverse=True)

    print(f"\\n🎯 TOP 5 CONFIGURATIONS (by expectancy):")
    print("-" * 95)
    print(f"{'Rank':<4} {'BaseRisk':<9} {'MaxRisk':<8} {'ExposureCap':<11} {'Trades':<6} {'WinRate':<8} {'Expectancy':<10} {'Sharpe':<8}")
    print("-" * 95)

    for i, result in enumerate(successful_tests[:5]):
        print(f"{i+1:<4} {result['base_risk_pct']:<9.3f} {result['max_risk_per_trade']:<8.3f} {result['exposure_cap_pct']:<11.2f} "
              f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['sharpe']:<8.2f}")

    # Sort by Sharpe ratio for risk-adjusted performance
    successful_tests.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\\n🎯 TOP 3 CONFIGURATIONS (by Sharpe ratio - risk-adjusted):")
    print("-" * 95)
    print(f"{'Rank':<4} {'BaseRisk':<9} {'MaxRisk':<8} {'ExposureCap':<11} {'Trades':<6} {'WinRate':<8} {'Expectancy':<10} {'Sharpe':<8}")
    print("-" * 95)

    for i, result in enumerate(successful_tests[:3]):
        print(f"{i+1:<4} {result['base_risk_pct']:<9.3f} {result['max_risk_per_trade']:<8.3f} {result['exposure_cap_pct']:<11.2f} "
              f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['sharpe']:<8.2f}")

    # Find configurations with good trade count and positive metrics
    quality_tests = [r for r in successful_tests if r["trades"] >= 3 and r["expectancy"] > -500]
    if quality_tests:
        # Re-sort by expectancy for quality tests
        quality_tests.sort(key=lambda x: x["expectancy"], reverse=True)

        print(f"\\n🎯 QUALITY FILTERS (≥3 trades, expectancy > -500):")
        print("-" * 95)

        for i, result in enumerate(quality_tests[:3]):
            print(f"{i+1:<4} {result['base_risk_pct']:<9.3f} {result['max_risk_per_trade']:<8.3f} {result['exposure_cap_pct']:<11.2f} "
                  f"{result['trades']:<6} {result['win_rate']:<8.1%} {result['expectancy']:<10.0f} {result['sharpe']:<8.2f}")

        best_quality = quality_tests[0]
        print(f"\\n🏆 RECOMMENDED RISK CONFIGURATION:")
        print(f"   base_risk_pct: {best_quality['base_risk_pct']:.3f}")
        print(f"   max_risk_per_trade: {best_quality['max_risk_per_trade']:.3f}")
        print(f"   exposure_cap_pct: {best_quality['exposure_cap_pct']:.2f}")
        print(f"   Performance: {best_quality['trades']} trades, {best_quality['win_rate']:.1%} win rate, {best_quality['expectancy']:.0f} expectancy")
        print(f"   Risk metrics: {best_quality['sharpe']:.2f} Sharpe, {best_quality['max_dd']:.0f} max drawdown")
    else:
        print("⚠️  No configurations meet quality criteria")

    # Save detailed results
    output_file = "risk_management_sweep_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "sweep_config": {
                "base_risk_values": base_risk_values,
                "max_risk_values": max_risk_values,
                "exposure_cap_values": exposure_cap_values
            },
            "results": results,
            "successful_tests": len(successful_tests),
            "total_tests": total_tests
        }, f, indent=2)

    print(f"\\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()