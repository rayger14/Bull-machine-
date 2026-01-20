#!/usr/bin/env python3
"""
S2 (Failed Rally) Archetype Optimization Script

Systematically test parameter combinations to find optimal settings for 2022 bear market.

Target: PF > 1.4, WR > 55%, Trades 80-150
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import itertools
import time

# Parameter grid for S2 optimization (reduced to ~300 combinations for speed)
# Phase 1: Coarse grid search (1-2 hours)
PARAM_GRID = {
    'fusion_threshold': [0.32, 0.36, 0.40],  # Entry gate
    'atr_stop_mult': [1.8, 2.0, 2.5],        # Stop loss distance
    'trail_atr_mult': [1.3, 1.5, 2.0],       # Trailing stop
    'max_risk_pct': [0.01, 0.015],           # Position sizing
    'cooldown_bars': [6, 8, 12],             # Prevent overtrading
    'wick_ratio_min': [1.5, 2.0, 2.5],       # Rejection strength
    'archetype_weight': [2.0, 2.5]           # Signal boost
}

# Total combinations: 3 * 3 * 3 * 2 * 3 * 3 * 2 = 972 (~8 hours @ 30s/test)
# For quick test, use: fusion=[0.36], atr=[2.0], trail=[1.5], risk=[0.015], cool=[8], wick=[2.0], weight=[2.0]

def create_s2_config(params: Dict) -> Dict:
    """Create S2 config with given parameters"""
    return {
        "version": "s2_optimization",
        "profile": "s2_test",
        "description": f"S2 optimization test - {params}",
        "adaptive_fusion": True,
        "regime_classifier": {
            "model_path": "models/regime_classifier_gmm.pkl",
            "feature_order": [
                "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                "funding", "oi", "rv_20d", "rv_60d"
            ],
            "zero_fill_missing": False,
            "regime_override": {"2022": "risk_off"}
        },
        "ml_filter": {"enabled": False},
        "fusion": {
            "entry_threshold_confidence": 0.36,
            "weights": {"wyckoff": 0.35, "liquidity": 0.30, "momentum": 0.35, "smc": 0.0}
        },
        "archetypes": {
            "use_archetypes": True,
            "max_trades_per_day": 8,
            "enable_A": False, "enable_B": False, "enable_C": False, "enable_D": False,
            "enable_E": False, "enable_F": False, "enable_G": False, "enable_H": False,
            "enable_K": False, "enable_L": False, "enable_M": False,
            "enable_S1": False, "enable_S2": True, "enable_S3": False, "enable_S4": False,
            "enable_S5": False, "enable_S6": False, "enable_S7": False, "enable_S8": False,
            "thresholds": {"min_liquidity": 0.20},
            "failed_rally": {
                "direction": "short",
                "archetype_weight": params['archetype_weight'],
                "fusion_threshold": params['fusion_threshold'],
                "final_fusion_gate": params['fusion_threshold'],
                "cooldown_bars": params['cooldown_bars'],
                "max_risk_pct": params['max_risk_pct'],
                "atr_stop_mult": params['atr_stop_mult'],
                "wick_ratio_min": params['wick_ratio_min'],
                "require_rsi_divergence": False,
                "weights": {
                    "ob_retest": 0.25,
                    "wick_rejection": 0.25,
                    "rsi_signal": 0.20,
                    "volume_fade": 0.15,
                    "tf4h_confirm": 0.15
                }
            },
            "routing": {
                "risk_on": {"weights": {"failed_rally": 0.5}, "final_gate_delta": 0.0},
                "neutral": {"weights": {"failed_rally": 1.0}, "final_gate_delta": 0.0},
                "risk_off": {"weights": {"failed_rally": 2.0}, "final_gate_delta": 0.02},
                "crisis": {"weights": {"failed_rally": 2.5}, "final_gate_delta": 0.04}
            },
            "exits": {
                "failed_rally": {
                    "enable_trail": True,
                    "trail_atr_mult": params['trail_atr_mult'],
                    "time_limit_hours": 48
                }
            }
        },
        "context": {"crisis_fuse": {"enabled": False}},
        "risk": {
            "base_risk_pct": params['max_risk_pct'],
            "max_position_size_pct": 0.15,
            "max_portfolio_risk_pct": 0.08
        }
    }

def run_backtest(config: Dict, test_id: str) -> Dict:
    """Run backtest and extract metrics"""
    # Write config to temp file
    config_path = f'/tmp/s2_opt_{test_id}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run backtest
    cmd = [
        'python3', 'bin/backtest_knowledge_v2.py',
        '--asset', 'BTC',
        '--start', '2022-01-01',
        '--end', '2022-12-31',
        '--config', config_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    # Parse output
    output = result.stdout + result.stderr

    metrics = {
        'test_id': test_id,
        'total_trades': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'total_return': 0.0
    }

    # Extract metrics from output
    for line in output.split('\n'):
        if 'Total Trades:' in line:
            metrics['total_trades'] = int(line.split(':')[1].strip())
        elif 'Win Rate:' in line:
            metrics['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Profit Factor:' in line:
            metrics['profit_factor'] = float(line.split(':')[1].strip())
        elif 'Sharpe Ratio:' in line:
            metrics['sharpe_ratio'] = float(line.split(':')[1].strip())
        elif 'Max Drawdown:' in line:
            metrics['max_drawdown'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Avg Win:' in line:
            metrics['avg_win'] = float(line.split('$')[1].strip())
        elif 'Avg Loss:' in line:
            metrics['avg_loss'] = float(line.split('$')[1].strip())
        elif 'Total Return:' in line:
            try:
                metrics['total_return'] = float(line.split(':')[1].strip().replace('%', '').replace('$', ''))
            except:
                pass

    return metrics

def generate_param_combinations() -> List[Dict]:
    """Generate all parameter combinations"""
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations

def main():
    print("=" * 80)
    print("S2 (FAILED RALLY) OPTIMIZATION")
    print("=" * 80)
    print(f"Parameter grid: {PARAM_GRID}")
    print(f"\nGenerating parameter combinations...")

    param_combos = generate_param_combinations()
    print(f"Total configurations to test: {len(param_combos)}")
    print(f"Estimated time: {len(param_combos) * 30 / 60:.1f} minutes (30s per test)\n")

    results = []

    for i, params in enumerate(param_combos, 1):
        print(f"\n[{i}/{len(param_combos)}] Testing config: {params}")

        test_id = f"test_{i:04d}"
        config = create_s2_config(params)

        try:
            start_time = time.time()
            metrics = run_backtest(config, test_id)
            elapsed = time.time() - start_time

            # Add params to metrics
            metrics.update(params)
            metrics['elapsed_sec'] = elapsed

            results.append(metrics)

            print(f"   Trades: {metrics['total_trades']}, WR: {metrics['win_rate']:.1f}%, "
                  f"PF: {metrics['profit_factor']:.2f}, Sharpe: {metrics['sharpe_ratio']:.2f}")

            # Save intermediate results every 10 tests
            if i % 10 == 0:
                df = pd.DataFrame(results)
                df.to_csv('results/optimization/s2_optimization_progress.csv', index=False)
                print(f"\n   Progress saved ({len(results)} results)")

        except Exception as e:
            print(f"   ERROR: {e}")
            continue

    # Save final results
    df = pd.DataFrame(results)
    df.to_csv('results/optimization/s2_optimization_full.csv', index=False)

    # Filter and rank
    df_filtered = df[(df['total_trades'] >= 80) & (df['total_trades'] <= 150)]
    df_ranked = df_filtered.sort_values('profit_factor', ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by Profit Factor)")
    print("=" * 80)

    if len(df_ranked) > 0:
        top10 = df_ranked.head(10)
        print(top10[['fusion_threshold', 'atr_stop_mult', 'trail_atr_mult', 'cooldown_bars',
                     'total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio']].to_string(index=False))

        # Save top configs
        top10.to_csv('results/optimization/s2_optimization_top10.csv', index=False)
    else:
        print("No configurations met the criteria (80-150 trades)")
        print("\nTop 10 by Profit Factor (no trade count filter):")
        top10_unfiltered = df.sort_values('profit_factor', ascending=False).head(10)
        print(top10_unfiltered[['fusion_threshold', 'atr_stop_mult', 'trail_atr_mult', 'cooldown_bars',
                                 'total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio']].to_string(index=False))

    print(f"\nFull results saved to: results/optimization/s2_optimization_full.csv")
    print(f"Total configurations tested: {len(results)}")

if __name__ == '__main__':
    main()
