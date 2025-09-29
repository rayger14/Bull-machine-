#!/usr/bin/env python3
"""
Bull Machine v1.6.2 Tuning Runner
Simplified interface to run walk-forward optimization
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Import our complete confluence system
from run_complete_confluence_system import (
    load_multi_timeframe_data,
    calculate_complete_confluence_score,
    run_complete_confluence_backtest
)

def run_stage_a_optimization():
    """
    Stage A: Optimize Risk & Exit Parameters
    Focus on ETH 1D first, then transfer to other assets
    """
    print("="*70)
    print("BULL MACHINE v1.6.2 - STAGE A OPTIMIZATION")
    print("Optimizing: Risk Sizing & Exit Parameters")
    print("="*70)

    # Load ETH data
    data_1d = "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv"
    data_4h = "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv"
    data_1h = "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"

    # Check if files exist
    for f in [data_1d, data_4h, data_1h]:
        if not os.path.exists(f):
            print(f"ERROR: Data file not found: {f}")
            return

    # Base configuration
    base_config = {
        "entry_threshold": 0.30,
        "min_active_domains": 3,
        "cooldown_days": 7,
        "risk_pct": 0.025,  # Will be optimized
        "sl_atr_multiplier": 1.8,  # Will be optimized
        "tp_atr_multiplier": 3.0,  # Will be optimized
        "trail_atr_multiplier": 1.0  # Will be optimized
    }

    # Define parameter grid for Stage A
    param_grid = {
        "risk_pct": [0.015, 0.02, 0.025, 0.03],
        "sl_atr": [1.4, 1.6, 1.8, 2.0],
        "tp_atr": [2.5, 3.0, 3.5],
        "trail_atr": [0.9, 1.1, 1.3]
    }

    # Walk-forward validation windows
    validation_windows = [
        ("2023-01-01", "2023-06-30"),
        ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
    ]

    results = []
    total_configs = len(param_grid["risk_pct"]) * len(param_grid["sl_atr"]) * len(param_grid["tp_atr"]) * len(param_grid["trail_atr"])
    config_count = 0

    print(f"\nTesting {total_configs} configurations...")
    print("Target metrics: WR≥50%, DD≤9.2%, PF≥1.3, Freq 1-2/month\n")

    # Grid search
    for risk in param_grid["risk_pct"]:
        for sl in param_grid["sl_atr"]:
            for tp in param_grid["tp_atr"]:
                for trail in param_grid["trail_atr"]:
                    config_count += 1

                    # Update config
                    test_config = base_config.copy()
                    test_config["risk_pct"] = risk
                    test_config["sl_atr_multiplier"] = sl
                    test_config["tp_atr_multiplier"] = tp
                    test_config["trail_atr_multiplier"] = trail

                    config_name = f"r{risk}_sl{sl}_tp{tp}_tr{trail}"

                    # Skip some configs for speed in demo (remove in production)
                    if config_count % 3 != 1:  # Test every 3rd config for demo
                        continue

                    print(f"[{config_count}/{total_configs}] Testing: {config_name}")

                    # Run validation windows
                    window_metrics = []
                    for start, end in validation_windows:
                        try:
                            # Load data for this window
                            data_paths = {
                                '1D': data_1d,
                                '4H': data_4h,
                                '1H': data_1h
                            }
                            window_data = load_multi_timeframe_data("ETH", data_paths)

                            # Filter to validation window
                            for tf in window_data:
                                if window_data[tf] is not None and len(window_data[tf]) > 0:
                                    window_data[tf] = window_data[tf][start:end]

                            # Run backtest
                            backtest_results = run_complete_confluence_backtest(
                                "ETH", window_data, test_config
                            )

                            if "metrics" in backtest_results:
                                metrics = backtest_results["metrics"]
                                window_metrics.append(metrics)
                        except Exception as e:
                            print(f"  Error in window {start}-{end}: {e}")
                            continue

                    if window_metrics:
                        # Calculate average metrics
                        avg_metrics = {
                            "win_rate": np.mean([m.get("win_rate", 0) for m in window_metrics]),
                            "max_drawdown_pct": np.mean([abs(m.get("max_drawdown_pct", 100)) for m in window_metrics]),
                            "profit_factor": np.mean([m.get("profit_factor", 0) for m in window_metrics]),
                            "sharpe_ratio": np.mean([m.get("sharpe_ratio", 0) for m in window_metrics]),
                            "trades_per_month": np.mean([m.get("trades_per_month", 0) for m in window_metrics]),
                            "total_pnl_pct": np.mean([m.get("total_pnl_pct", 0) for m in window_metrics])
                        }

                        # Calculate utility score
                        utility = calculate_utility_score(avg_metrics)

                        result = {
                            "name": config_name,
                            "config": test_config,
                            "metrics": avg_metrics,
                            "utility": utility,
                            "windows_tested": len(window_metrics)
                        }
                        results.append(result)

                        print(f"  Utility: {utility:.3f} | WR: {avg_metrics['win_rate']:.1f}% | "
                              f"DD: {avg_metrics['max_drawdown_pct']:.1f}% | "
                              f"PF: {avg_metrics['profit_factor']:.2f}")

    # Sort by utility
    results.sort(key=lambda x: x["utility"], reverse=True)

    # Save results
    output_dir = "reports/tuning"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/stage_a_ETH_1d_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Display top 5 results
    print("\n" + "="*70)
    print("TOP 5 CONFIGURATIONS BY UTILITY")
    print("="*70)

    for i, result in enumerate(results[:5]):
        print(f"\n#{i+1}: {result['name']}")
        print(f"  Utility Score: {result['utility']:.3f}")
        print(f"  Win Rate: {result['metrics']['win_rate']:.1f}%")
        print(f"  Max DD: {result['metrics']['max_drawdown_pct']:.1f}%")
        print(f"  Profit Factor: {result['metrics']['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
        print(f"  Trades/Month: {result['metrics']['trades_per_month']:.2f}")
        print(f"  Config: Risk={result['config']['risk_pct']}, "
              f"SL={result['config']['sl_atr_multiplier']}, "
              f"TP={result['config']['tp_atr_multiplier']}, "
              f"Trail={result['config']['trail_atr_multiplier']}")

    # Save best configuration
    if results:
        best_config_path = "configs/v160/assets/ETH_tuned_stage_a.json"
        with open(best_config_path, "w") as f:
            json.dump(results[0]["config"], f, indent=2)
        print(f"\n✅ Best configuration saved to: {best_config_path}")
        print(f"Results saved to: {output_file}")

    return results

def calculate_utility_score(metrics):
    """
    Calculate utility score for ranking configurations

    utility =
      + 1.00 * normalized_PnL
      + 0.60 * normalized_PF
      + 0.40 * normalized_Sharpe
      - 0.80 * penalty(DD > 9.2%)
      - 0.60 * penalty(freq outside 1-2)
      - 0.50 * penalty(WR < 50%)
    """
    # Extract metrics
    wr = metrics.get("win_rate", 0) / 100.0
    dd = abs(metrics.get("max_drawdown_pct", 100))
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    freq = metrics.get("trades_per_month", 0)
    pnl = metrics.get("total_pnl_pct", 0)

    # Normalize positive components
    n_pnl = min(pnl / 10.0, 1.0)  # 10% target
    n_pf = min(pf / 2.0, 1.0)  # PF 2.0 target
    n_sharpe = min(sharpe / 3.0, 1.0)  # Sharpe 3.0 target

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

def run_quick_test():
    """Quick test with current best config"""
    print("="*70)
    print("BULL MACHINE v1.6.2 - QUICK TEST")
    print("Running single backtest with current config")
    print("="*70)

    # Load data
    data_1d = "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv"
    data_4h = "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv"
    data_1h = "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"

    # Current best config
    config = {
        "entry_threshold": 0.30,
        "min_active_domains": 3,
        "cooldown_days": 7,
        "risk_pct": 0.025
    }

    # Load and prepare data
    print("\nLoading data...")
    data_paths = {
        '1D': data_1d,
        '4H': data_4h,
        '1H': data_1h
    }
    data = load_multi_timeframe_data("ETH", data_paths)

    # Filter to date range
    for tf in data:
        if data[tf] is not None and len(data[tf]) > 0:
            data[tf] = data[tf]['2023-01-01':'2024-12-31']

    # Run backtest
    print("Running backtest from 2023-01-01 to 2024-12-31...")
    results = run_complete_confluence_backtest("ETH", data, config)

    # Display results
    if results and "metrics" in results:
        metrics = results["metrics"]
        print("\nResults:")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Total PnL: {metrics.get('total_pnl_pct', 0):.2f}%")
        print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Trades/Month: {metrics.get('trades_per_month', 0):.2f}")

        # Calculate utility
        utility = calculate_utility_score(metrics)
        print(f"\n  Utility Score: {utility:.3f}")

    # Save results (handle datetime serialization)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/quick_test_{timestamp}.json"

    # Convert timestamps to strings for JSON serialization
    if results and "trades" in results:
        for trade in results["trades"]:
            for key in ["entry_time", "exit_time"]:
                if key in trade and hasattr(trade[key], 'isoformat'):
                    trade[key] = trade[key].isoformat()

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run quick test
        run_quick_test()
    else:
        # Run full Stage A optimization
        results = run_stage_a_optimization()

        if results:
            print(f"\n🎯 Optimization complete!")
            print(f"Best utility score: {results[0]['utility']:.3f}")
            print(f"Best config: {results[0]['name']}")