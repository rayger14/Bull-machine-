#!/usr/bin/env python3
"""
Fast S2 Optimization using Random Search

Instead of exhaustive grid search (972 combos), test 100 random configurations
to find promising regions, then fine-tune. This reduces 8 hours to 1 hour.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import time

# Parameter ranges for random sampling
PARAM_RANGES = {
    'fusion_threshold': (0.30, 0.45),
    'atr_stop_mult': (1.5, 3.0),
    'trail_atr_mult': (1.0, 2.5),
    'max_risk_pct': (0.008, 0.025),
    'cooldown_bars': (4, 16),
    'wick_ratio_min': (1.2, 3.0),
    'archetype_weight': (1.5, 3.5)
}

def sample_params(n_samples: int) -> List[Dict]:
    """Generate random parameter samples"""
    samples = []

    for _ in range(n_samples):
        params = {
            'fusion_threshold': round(np.random.uniform(*PARAM_RANGES['fusion_threshold']), 2),
            'atr_stop_mult': round(np.random.uniform(*PARAM_RANGES['atr_stop_mult']), 1),
            'trail_atr_mult': round(np.random.uniform(*PARAM_RANGES['trail_atr_mult']), 1),
            'max_risk_pct': round(np.random.uniform(*PARAM_RANGES['max_risk_pct']), 3),
            'cooldown_bars': int(np.random.uniform(*PARAM_RANGES['cooldown_bars'])),
            'wick_ratio_min': round(np.random.uniform(*PARAM_RANGES['wick_ratio_min']), 1),
            'archetype_weight': round(np.random.uniform(*PARAM_RANGES['archetype_weight']), 1)
        }
        samples.append(params)

    return samples

def create_s2_config(params: Dict) -> Dict:
    """Create S2 config with given parameters"""
    return {
        "version": "s2_optimization",
        "profile": "s2_test",
        "description": f"S2 optimization test",
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
    config_path = f'/tmp/s2_opt_{test_id}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    cmd = [
        'python3', 'bin/backtest_knowledge_v2.py',
        '--asset', 'BTC',
        '--start', '2022-01-01',
        '--end', '2022-12-31',
        '--config', config_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
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

def main():
    n_samples = 100  # Test 100 random configurations

    print("=" * 80)
    print("S2 FAST RANDOM SEARCH OPTIMIZATION")
    print("=" * 80)
    print(f"Random samples: {n_samples}")
    print(f"Estimated time: {n_samples * 30 / 60:.1f} minutes\n")

    # Generate random samples
    param_samples = sample_params(n_samples)

    results = []

    for i, params in enumerate(param_samples, 1):
        print(f"\n[{i}/{n_samples}] Testing: fusion={params['fusion_threshold']:.2f}, "
              f"stop={params['atr_stop_mult']:.1f}, trail={params['trail_atr_mult']:.1f}, "
              f"cool={params['cooldown_bars']}, wick={params['wick_ratio_min']:.1f}")

        test_id = f"test_{i:04d}"
        config = create_s2_config(params)

        try:
            start_time = time.time()
            metrics = run_backtest(config, test_id)
            elapsed = time.time() - start_time

            metrics.update(params)
            metrics['elapsed_sec'] = elapsed

            results.append(metrics)

            print(f"   → Trades: {metrics['total_trades']}, WR: {metrics['win_rate']:.1f}%, "
                  f"PF: {metrics['profit_factor']:.2f}, Sharpe: {metrics['sharpe_ratio']:.2f}")

            # Save progress
            if i % 10 == 0:
                df = pd.DataFrame(results)
                df.to_csv('results/optimization/s2_random_search_progress.csv', index=False)

        except Exception as e:
            print(f"   ERROR: {e}")
            continue

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/optimization/s2_random_search_full.csv', index=False)

    # Analyze results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    # Filter by trade count
    df_filtered = df[(df['total_trades'] >= 80) & (df['total_trades'] <= 150)]

    if len(df_filtered) > 0:
        print(f"\nConfigs with 80-150 trades: {len(df_filtered)}")
        df_ranked = df_filtered.sort_values('profit_factor', ascending=False)
        print("\nTOP 10 (by Profit Factor, 80-150 trades):")
        top10 = df_ranked.head(10)
        print(top10[['fusion_threshold', 'atr_stop_mult', 'trail_atr_mult', 'cooldown_bars',
                     'wick_ratio_min', 'total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio']].to_string(index=False))
        top10.to_csv('results/optimization/s2_random_search_top10.csv', index=False)
    else:
        print("\nNo configs met 80-150 trade criteria. Top 10 overall:")
        df_ranked = df.sort_values('profit_factor', ascending=False)
        top10 = df_ranked.head(10)
        print(top10[['fusion_threshold', 'atr_stop_mult', 'trail_atr_mult', 'cooldown_bars',
                     'wick_ratio_min', 'total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio']].to_string(index=False))

    # Best overall
    best = df.loc[df['profit_factor'].idxmax()]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Fusion Threshold: {best['fusion_threshold']:.2f}")
    print(f"ATR Stop Mult: {best['atr_stop_mult']:.1f}")
    print(f"Trail ATR Mult: {best['trail_atr_mult']:.1f}")
    print(f"Max Risk %: {best['max_risk_pct']:.3f}")
    print(f"Cooldown Bars: {best['cooldown_bars']}")
    print(f"Wick Ratio Min: {best['wick_ratio_min']:.1f}")
    print(f"Archetype Weight: {best['archetype_weight']:.1f}")
    print(f"\nPerformance:")
    print(f"Total Trades: {best['total_trades']}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")

    # Save best config
    best_config = create_s2_config({
        'fusion_threshold': best['fusion_threshold'],
        'atr_stop_mult': best['atr_stop_mult'],
        'trail_atr_mult': best['trail_atr_mult'],
        'max_risk_pct': best['max_risk_pct'],
        'cooldown_bars': int(best['cooldown_bars']),
        'wick_ratio_min': best['wick_ratio_min'],
        'archetype_weight': best['archetype_weight']
    })

    with open('configs/s2_optimized.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest config saved to: configs/s2_optimized.json")

if __name__ == '__main__':
    main()
