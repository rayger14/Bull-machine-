#!/usr/bin/env python3
"""
Bull Machine v1.6.2 Walk-Forward Parameter Tuning
Systematic optimization with out-of-sample validation
"""

import json
import itertools
import subprocess
import os
import tempfile
import statistics as stats
from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# Asset configurations
ASSETS = {
    "ETH": {
        "data_1d": "data/Chart Logs 2/COINBASE_ETHUSD_1D_0e638.csv",
        "data_4h": "data/Chart Logs 2/COINBASE_ETHUSD_240_c01aa.csv",
        "data_1h": "data/Chart Logs 2/COINBASE_ETHUSD_60_a3f43.csv",
        "targets_1d": {"wr": 0.50, "dd": 9.2, "pf": 1.3, "freq_lo": 1.0, "freq_hi": 2.0},
        "targets_4h": {"wr": 0.45, "dd": 20.0, "pf": 1.2, "freq_lo": 2.0, "freq_hi": 4.0}
    },
    "BTC": {
        "data_1d": "data/Chart Logs 2/COINBASE_BTCUSD_1D_b43fb.csv",
        "data_4h": "data/Chart Logs 2/COINBASE_BTCUSD_240_fde74.csv",
        "data_1h": "data/Chart Logs 2/COINBASE_BTCUSD_60_5f1ac.csv",
        "targets_1d": {"wr": 0.48, "dd": 10.0, "pf": 1.25, "freq_lo": 0.8, "freq_hi": 1.8},
        "targets_4h": {"wr": 0.43, "dd": 22.0, "pf": 1.15, "freq_lo": 1.8, "freq_hi": 3.5}
    },
    "SOL": {
        "data_1d": "data/Chart Logs 2/COINBASE_SOLUSD_1D_8ffaf.csv",
        "data_4h": "data/Chart Logs 2/COINBASE_SOLUSD_240_f9a53.csv",
        "data_1h": "data/Chart Logs 2/COINBASE_SOLUSD_60_07764.csv",
        "targets_1d": {"wr": 0.45, "dd": 15.0, "pf": 1.2, "freq_lo": 0.8, "freq_hi": 2.5},
        "targets_4h": {"wr": 0.42, "dd": 25.0, "pf": 1.1, "freq_lo": 2.0, "freq_hi": 5.0}
    }
}

# Walk-forward validation windows
FOLDS = [
    {"train_start": "2019-07-01", "train_end": "2021-01-01", "val_start": "2021-01-01", "val_end": "2021-06-30"},
    {"train_start": "2020-01-01", "train_end": "2021-07-01", "val_start": "2021-07-01", "val_end": "2021-12-31"},
    {"train_start": "2020-07-01", "train_end": "2022-01-01", "val_start": "2022-01-01", "val_end": "2022-06-30"},
    {"train_start": "2021-01-01", "train_end": "2022-07-01", "val_start": "2022-07-01", "val_end": "2022-12-31"},
    {"train_start": "2021-07-01", "train_end": "2023-01-01", "val_start": "2023-01-01", "val_end": "2023-06-30"},
    {"train_start": "2022-01-01", "train_end": "2023-07-01", "val_start": "2023-07-01", "val_end": "2023-12-31"},
    {"train_start": "2022-07-01", "train_end": "2024-01-01", "val_start": "2024-01-01", "val_end": "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-07-01", "val_start": "2024-07-01", "val_end": "2024-12-31"},
]

def penalty_over(x, limit, w=1.0):
    """Quadratic penalty for exceeding limit"""
    return 0.0 if x <= limit else w * ((x - limit) / max(1e-9, limit))**2

def penalty_under(x, limit, w=1.0):
    """Quadratic penalty for falling below limit"""
    return 0.0 if x >= limit else w * ((limit - x) / max(1e-9, limit))**2

def penalty_band(x, lo, hi, w=1.0):
    """Quadratic penalty for being outside band"""
    if lo <= x <= hi:
        return 0.0
    d = (lo - x) if x < lo else (x - hi)
    base = (abs(d) / max(1e-9, (hi - lo)))**2
    return w * base

def calculate_utility(metrics, targets):
    """
    Calculate utility score with penalties

    utility =
      + 1.00 * normalized_PnL
      + 0.60 * normalized_PF
      + 0.40 * normalized_Sharpe
      - 0.80 * penalty(DD > target_DD)
      - 0.60 * penalty(freq outside target band)
      - 0.50 * penalty(WR < target_WR)
    """
    wr = metrics.get("win_rate", 0) / 100.0  # Convert to 0-1
    dd = abs(metrics.get("max_drawdown_pct", 100))
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    freq = metrics.get("trades_per_month", 0)
    pnl = metrics.get("total_pnl_pct", 0)

    # Normalize components
    n_pnl = pnl / 10.0  # Scale to ~0-1 for 10% target
    n_pf = min(pf / 2.0, 1.0)  # PF 2.0+ capped
    n_sh = min(sharpe / 3.0, 1.0)  # Sharpe 3.0+ capped

    # Calculate utility
    u = 1.00 * n_pnl + 0.60 * n_pf + 0.40 * n_sh
    u -= 0.80 * penalty_over(dd, targets["dd"])
    u -= 0.60 * penalty_band(freq, targets["freq_lo"], targets["freq_hi"])
    u -= 0.50 * penalty_under(wr, targets["wr"])

    return u

def run_backtest_complete_confluence(config, asset, start, end, outdir):
    """
    Run backtest using the complete confluence system
    Returns metrics dictionary
    """
    # Create temporary config file
    config_path = os.path.join(outdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Import and run the complete confluence system
    import sys
    sys.path.append('.')

    # Create a simple runner that uses our complete confluence system
    runner_code = f"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from run_complete_confluence_system import (
    load_multi_timeframe_data,
    calculate_complete_confluence_score,
    run_complete_confluence_backtest
)

# Load config
with open("{config_path}") as f:
    config = json.load(f)

# Run backtest
results = run_complete_confluence_backtest(
    "{ASSETS[asset]['data_1d']}",
    "{ASSETS[asset]['data_4h']}",
    "{ASSETS[asset]['data_1h']}",
    "{start}",
    "{end}",
    config
)

# Save results
with open("{outdir}/results.json", "w") as f:
    json.dump(results, f, indent=2)
"""

    runner_path = os.path.join(outdir, "runner.py")
    with open(runner_path, "w") as f:
        f.write(runner_code)

    # Execute runner
    subprocess.run([
        "python", runner_path
    ], check=True, capture_output=True)

    # Load and return results
    with open(os.path.join(outdir, "results.json")) as f:
        results = json.load(f)

    return results.get("metrics", {})

def trial_walkforward(config, asset, targets, timeframe="1d"):
    """
    Run walk-forward validation across all folds
    Returns mean utility and standard deviation
    """
    fold_utilities = []
    fold_metrics = []

    for i, fold in enumerate(FOLDS):
        print(f"  Fold {i+1}/{len(FOLDS)}: {fold['val_start']} to {fold['val_end']}")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Run validation period backtest
                metrics = run_backtest_complete_confluence(
                    config, asset,
                    fold["val_start"], fold["val_end"],
                    tmpdir
                )

                # Calculate utility
                utility = calculate_utility(metrics, targets)
                fold_utilities.append(utility)
                fold_metrics.append(metrics)

                print(f"    Utility: {utility:.3f} | WR: {metrics.get('win_rate', 0):.1f}% | "
                      f"DD: {metrics.get('max_drawdown_pct', 0):.1f}% | "
                      f"PF: {metrics.get('profit_factor', 0):.2f}")

            except Exception as e:
                print(f"    Error in fold: {e}")
                fold_utilities.append(-10.0)  # Severe penalty for failed runs

    mean_utility = np.mean(fold_utilities)
    std_utility = np.std(fold_utilities)

    return mean_utility, std_utility, fold_metrics

# ========== STAGE A: Risk & Exit Parameters ==========
def generate_stage_a_configs(base_config):
    """Generate configs for Stage A: Risk & Exit optimization"""
    configs = []

    for risk_pct in [0.005, 0.01, 0.015, 0.02, 0.025]:
        for sl_atr in [1.4, 1.6, 1.8, 2.0, 2.2]:
            for tp_atr in [2.5, 3.0, 3.5, 4.0]:
                for trail_atr in [0.8, 1.0, 1.2, 1.4]:
                    cfg = deepcopy(base_config)

                    # Update risk parameters
                    cfg["risk_pct"] = risk_pct
                    cfg["sl_atr_multiplier"] = sl_atr
                    cfg["tp_atr_multiplier"] = tp_atr
                    cfg["trail_atr_multiplier"] = trail_atr

                    name = f"risk{risk_pct}_sl{sl_atr}_tp{tp_atr}_tr{trail_atr}"
                    configs.append((name, cfg))

    return configs

# ========== STAGE B: Entry Threshold & Consensus ==========
def generate_stage_b_configs(base_config):
    """Generate configs for Stage B: Entry threshold & consensus"""
    configs = []

    for entry_threshold in np.arange(0.24, 0.42, 0.02):
        for min_consensus in [2, 3]:
            for consensus_penalty in [0.00, 0.01, 0.02, 0.03]:
                cfg = deepcopy(base_config)

                cfg["entry_threshold"] = round(entry_threshold, 2)
                cfg["min_active_domains"] = min_consensus
                cfg["consensus_penalty"] = consensus_penalty

                name = f"thr{entry_threshold:.2f}_cons{min_consensus}_pen{consensus_penalty}"
                configs.append((name, cfg))

    return configs

# ========== STAGE C: Quality Floors ==========
def generate_stage_c_configs(base_config):
    """Generate configs for Stage C: Quality floor tuning"""
    configs = []

    # Define floor keys and their current defaults
    floor_keys = ["wyckoff", "liquidity", "momentum", "temporal", "fusion"]

    # Generate combinations (only modify 2-3 at a time)
    for num_changes in [2, 3]:
        for keys_to_change in itertools.combinations(floor_keys, num_changes):
            for deltas in itertools.product([-0.03, 0, 0.03], repeat=num_changes):
                cfg = deepcopy(base_config)

                name_parts = []
                for key, delta in zip(keys_to_change, deltas):
                    if delta != 0:
                        # Adjust floor (ensure stays in [0, 1])
                        current = cfg.get(f"{key}_floor", 0.1)
                        new_val = max(0, min(1, current + delta))
                        cfg[f"{key}_floor"] = round(new_val, 2)
                        name_parts.append(f"{key}{new_val:.2f}")

                if name_parts:  # Only if we made changes
                    name = "floor_" + "_".join(name_parts)
                    configs.append((name, cfg))

    return configs[:50]  # Limit to 50 configs to avoid explosion

# ========== STAGE D: Layer Weights ==========
def generate_stage_d_configs(base_config):
    """Generate configs for Stage D: Layer weight optimization"""
    configs = []

    # Sample weight combinations on simplex
    for _ in range(30):  # 30 random samples
        # Generate 3 random weights for wyckoff, liquidity, momentum
        weights = np.random.dirichlet([1, 1, 1])

        cfg = deepcopy(base_config)
        cfg["weight_wyckoff"] = round(weights[0], 3)
        cfg["weight_liquidity"] = round(weights[1], 3)
        cfg["weight_momentum"] = round(weights[2], 3)

        name = f"w_wy{weights[0]:.2f}_lq{weights[1]:.2f}_mo{weights[2]:.2f}"
        configs.append((name, cfg))

    return configs

def run_stage_optimization(stage, asset="ETH", timeframe="1d"):
    """Run a single stage of optimization"""
    print(f"\n{'='*60}")
    print(f"STAGE {stage.upper()} OPTIMIZATION - {asset} {timeframe}")
    print(f"{'='*60}")

    # Load base config
    base_config_path = f"configs/v160/assets/{asset}.json"
    if not os.path.exists(base_config_path):
        # Create default config if doesn't exist
        base_config = {
            "entry_threshold": 0.30,
            "min_active_domains": 3,
            "cooldown_days": 7,
            "risk_pct": 0.025,
            "sl_atr_multiplier": 1.8,
            "tp_atr_multiplier": 3.0,
            "trail_atr_multiplier": 1.0
        }
    else:
        with open(base_config_path) as f:
            base_config = json.load(f)

    # Get targets for this asset/timeframe
    targets = ASSETS[asset][f"targets_{timeframe}"]

    # Generate configs based on stage
    if stage == "a":
        configs = generate_stage_a_configs(base_config)
    elif stage == "b":
        configs = generate_stage_b_configs(base_config)
    elif stage == "c":
        configs = generate_stage_c_configs(base_config)
    elif stage == "d":
        configs = generate_stage_d_configs(base_config)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    print(f"Testing {len(configs)} configurations...")

    # Track results
    results = []

    for i, (name, config) in enumerate(configs[:10]):  # Limit to 10 for initial testing
        print(f"\n[{i+1}/{min(10, len(configs))}] Testing: {name}")

        mean_utility, std_utility, fold_metrics = trial_walkforward(
            config, asset, targets, timeframe
        )

        results.append({
            "name": name,
            "config": config,
            "mean_utility": mean_utility,
            "std_utility": std_utility,
            "fold_metrics": fold_metrics
        })

        print(f"  Mean Utility: {mean_utility:.3f} ± {std_utility:.3f}")

    # Sort by utility
    results.sort(key=lambda x: x["mean_utility"], reverse=True)

    # Save results
    output_dir = f"reports/tuning/stage_{stage}"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{asset}_{timeframe}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print top 3
    print(f"\n{'='*60}")
    print(f"TOP 3 CONFIGURATIONS")
    print(f"{'='*60}")

    for i, result in enumerate(results[:3]):
        print(f"\n#{i+1}: {result['name']}")
        print(f"  Utility: {result['mean_utility']:.3f} ± {result['std_utility']:.3f}")
        print(f"  Config changes: {json.dumps(result['config'], indent=2)[:200]}...")

    # Save best config
    if results:
        best_config_path = f"configs/v160/assets/{asset}_stage{stage}_best.json"
        with open(best_config_path, "w") as f:
            json.dump(results[0]["config"], f, indent=2)
        print(f"\nBest config saved to: {best_config_path}")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bull Machine v1.6.2 Walk-Forward Tuning")
    parser.add_argument("--stage", choices=["a", "b", "c", "d"], default="a",
                       help="Optimization stage: a=risk/exits, b=thresholds, c=floors, d=weights")
    parser.add_argument("--asset", choices=["ETH", "BTC", "SOL"], default="ETH",
                       help="Asset to optimize")
    parser.add_argument("--timeframe", choices=["1d", "4h"], default="1d",
                       help="Timeframe to optimize")

    args = parser.parse_args()

    # Create output directories
    os.makedirs("reports/tuning", exist_ok=True)

    # Run optimization
    results = run_stage_optimization(args.stage, args.asset, args.timeframe)

    print(f"\nOptimization complete! Results saved to reports/tuning/stage_{args.stage}/")