#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Complete Stage A Optimization
Full grid search for risk and exit parameters
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from run_complete_confluence_system import (
    load_multi_timeframe_data,
    run_complete_confluence_backtest
)

def run_complete_stage_a():
    """Run complete Stage A optimization with all configurations"""

    print("="*80)
    print("BULL MACHINE v1.6.2 - COMPLETE STAGE A OPTIMIZATION")
    print("Optimizing: Risk Sizing & Exit Parameters")
    print("="*80)

    # ETH data paths
    data_paths = {
        '1D': "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv",
        '4H': "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv",
        '1H': "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
    }

    # Load full dataset once
    print("\nLoading ETH data...")
    full_data = load_multi_timeframe_data("ETH", data_paths)

    # Define parameter grid
    param_grid = {
        "risk_pct": [0.015, 0.02, 0.025, 0.03],  # 1.5% to 3%
        "sl_atr": [1.4, 1.6, 1.8, 2.0],           # Stop loss ATR multiplier
        "tp_atr": [2.5, 3.0, 3.5, 4.0],           # Take profit ATR multiplier
        "trail_atr": [0.8, 1.0, 1.2]              # Trailing stop ATR multiplier
    }

    # Walk-forward validation windows
    validation_windows = [
        ("2023-01-01", "2023-06-30", "Q1-Q2 2023"),
        ("2023-07-01", "2023-12-31", "Q3-Q4 2023"),
        ("2024-01-01", "2024-06-30", "Q1-Q2 2024"),
        ("2024-07-01", "2024-12-31", "Q3-Q4 2024"),
    ]

    # Base configuration
    base_config = {
        "entry_threshold": 0.30,
        "min_active_domains": 3,
        "cooldown_days": 7
    }

    results = []
    total_configs = len(param_grid["risk_pct"]) * len(param_grid["sl_atr"]) * \
                   len(param_grid["tp_atr"]) * len(param_grid["trail_atr"])

    print(f"\nTesting {total_configs} configurations across {len(validation_windows)} windows")
    print(f"Total backtests: {total_configs * len(validation_windows)}")
    print("\nTarget metrics:")
    print("  - Win Rate ≥ 50%")
    print("  - Max Drawdown ≤ 9.2%")
    print("  - Profit Factor ≥ 1.3")
    print("  - Frequency: 1-2 trades/month")
    print("-"*80)

    config_count = 0

    # Grid search
    for risk in param_grid["risk_pct"]:
        for sl in param_grid["sl_atr"]:
            for tp in param_grid["tp_atr"]:
                for trail in param_grid["trail_atr"]:
                    config_count += 1

                    # Create test configuration
                    test_config = base_config.copy()
                    test_config["risk_pct"] = risk
                    test_config["sl_atr_multiplier"] = sl
                    test_config["tp_atr_multiplier"] = tp
                    test_config["trail_atr_multiplier"] = trail

                    config_name = f"r{risk:.3f}_sl{sl}_tp{tp}_tr{trail}"
                    print(f"\n[{config_count}/{total_configs}] Testing: {config_name}")

                    # Run walk-forward validation
                    window_results = []

                    for start, end, period_name in validation_windows:
                        # Filter data to window
                        window_data = {}
                        for tf in full_data:
                            if full_data[tf] is not None and len(full_data[tf]) > 0:
                                window_data[tf] = full_data[tf][start:end]

                        # Run backtest
                        try:
                            backtest_result = run_complete_confluence_backtest(
                                "ETH", window_data, test_config
                            )

                            if backtest_result and "metrics" in backtest_result:
                                metrics = backtest_result["metrics"]
                                window_results.append({
                                    "period": period_name,
                                    "metrics": metrics
                                })

                                print(f"    {period_name}: WR={metrics.get('win_rate', 0):.1f}% "
                                      f"DD={abs(metrics.get('max_drawdown_pct', 100)):.1f}% "
                                      f"PF={metrics.get('profit_factor', 0):.2f}")
                        except Exception as e:
                            print(f"    {period_name}: Error - {str(e)[:50]}")
                            continue

                    # Calculate aggregate metrics if we have results
                    if window_results:
                        # Calculate average metrics
                        avg_metrics = {
                            "win_rate": np.mean([w["metrics"].get("win_rate", 0)
                                               for w in window_results]),
                            "max_drawdown_pct": np.mean([abs(w["metrics"].get("max_drawdown_pct", 100))
                                                        for w in window_results]),
                            "profit_factor": np.mean([w["metrics"].get("profit_factor", 0)
                                                     for w in window_results]),
                            "sharpe_ratio": np.mean([w["metrics"].get("sharpe_ratio", 0)
                                                    for w in window_results]),
                            "trades_per_month": np.mean([w["metrics"].get("trades_per_month", 0)
                                                        for w in window_results]),
                            "total_pnl_pct": np.mean([w["metrics"].get("total_pnl_pct", 0)
                                                     for w in window_results])
                        }

                        # Calculate utility score
                        utility = calculate_utility(avg_metrics)

                        # Store result
                        result = {
                            "name": config_name,
                            "config": test_config,
                            "avg_metrics": avg_metrics,
                            "utility": utility,
                            "windows": len(window_results),
                            "window_details": window_results
                        }
                        results.append(result)

                        print(f"    Average: Utility={utility:.3f} WR={avg_metrics['win_rate']:.1f}% "
                              f"DD={avg_metrics['max_drawdown_pct']:.1f}%")

    # Sort by utility
    results.sort(key=lambda x: x["utility"], reverse=True)

    # Save detailed results
    output_dir = "reports/tuning"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/stage_a_complete_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Display top 10 results
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY UTILITY SCORE")
    print("="*80)

    for i, result in enumerate(results[:10]):
        print(f"\n#{i+1}: {result['name']}")
        print(f"  Utility Score: {result['utility']:.3f}")
        metrics = result['avg_metrics']
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Trades/Month: {metrics['trades_per_month']:.2f}")
        print(f"  Avg PnL: {metrics['total_pnl_pct']:.2f}%")

        # Show configuration
        cfg = result['config']
        print(f"  Config: Risk={cfg['risk_pct']:.1%}, SL={cfg['sl_atr_multiplier']}x, "
              f"TP={cfg['tp_atr_multiplier']}x, Trail={cfg['trail_atr_multiplier']}x")

    # Save best configuration
    if results:
        best_config = results[0]["config"]
        best_config_file = "configs/v160/assets/ETH_stage_a_optimal.json"
        os.makedirs(os.path.dirname(best_config_file), exist_ok=True)

        with open(best_config_file, "w") as f:
            json.dump(best_config, f, indent=2)

        print(f"\n✅ Best configuration saved to: {best_config_file}")
        print(f"📊 Full results saved to: {output_file}")

        # Print optimization summary
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"Configurations tested: {len(results)}/{total_configs}")
        print(f"Best utility score: {results[0]['utility']:.3f}")
        print(f"Best configuration: {results[0]['name']}")

        # Check if best config meets targets
        best_metrics = results[0]['avg_metrics']
        targets_met = []
        targets_missed = []

        if best_metrics['win_rate'] >= 50:
            targets_met.append("Win Rate ✓")
        else:
            targets_missed.append(f"Win Rate ({best_metrics['win_rate']:.1f}% < 50%)")

        if best_metrics['max_drawdown_pct'] <= 9.2:
            targets_met.append("Drawdown ✓")
        else:
            targets_missed.append(f"Drawdown ({best_metrics['max_drawdown_pct']:.1f}% > 9.2%)")

        if best_metrics['profit_factor'] >= 1.3:
            targets_met.append("Profit Factor ✓")
        else:
            targets_missed.append(f"Profit Factor ({best_metrics['profit_factor']:.2f} < 1.3)")

        if 1.0 <= best_metrics['trades_per_month'] <= 2.0:
            targets_met.append("Trade Frequency ✓")
        else:
            targets_missed.append(f"Trade Frequency ({best_metrics['trades_per_month']:.2f} outside 1-2)")

        print(f"\nTargets Met: {', '.join(targets_met) if targets_met else 'None'}")
        if targets_missed:
            print(f"Targets Missed: {', '.join(targets_missed)}")

        print(f"\nRecommendation: {'✅ Ready for Stage B' if len(targets_met) >= 3 else '⚠️ Consider expanding parameter ranges'}")

    return results

def calculate_utility(metrics):
    """
    Calculate utility score for configuration ranking
    """
    # Extract metrics
    wr = metrics.get("win_rate", 0) / 100.0
    dd = abs(metrics.get("max_drawdown_pct", 100))
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    freq = metrics.get("trades_per_month", 0)
    pnl = metrics.get("total_pnl_pct", 0)

    # Normalize positive components
    n_pnl = min(max(pnl / 10.0, -1), 1.0)  # Target 10%, cap at ±100%
    n_pf = min(pf / 2.0, 1.0)  # Target PF 2.0
    n_sharpe = min(sharpe / 3.0, 1.0)  # Target Sharpe 3.0

    # Calculate penalties
    dd_penalty = max(0, (dd - 9.2) / 9.2) ** 2 if dd > 9.2 else 0
    wr_penalty = max(0, (0.50 - wr) / 0.50) ** 2 if wr < 0.50 else 0

    # Frequency band penalty (target: 1-2 trades/month)
    if freq < 1.0:
        freq_penalty = ((1.0 - freq) / 1.0) ** 2
    elif freq > 2.0:
        freq_penalty = ((freq - 2.0) / 2.0) ** 2
    else:
        freq_penalty = 0

    # Calculate total utility
    utility = (1.00 * n_pnl +
               0.60 * n_pf +
               0.40 * n_sharpe -
               0.80 * dd_penalty -
               0.60 * freq_penalty -
               0.50 * wr_penalty)

    return utility

if __name__ == "__main__":
    results = run_complete_stage_a()
    print(f"\n🎯 Stage A optimization complete!")