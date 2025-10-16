#!/usr/bin/env python3
"""
Threshold Sensitivity Sweep for Bull Machine v1.8.6
Tests fusion_threshold values: 0.60, 0.62, 0.65, 0.68, 0.70, 0.74
With best performing domain weight combinations from baseline
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import sys

def create_config_variant(base_config: dict, fusion_threshold: float,
                         wyckoff: float, momentum: float, output_path: str):
    """Create a config file with specific threshold and weights"""
    config = base_config.copy()
    config['fusion']['entry_threshold_confidence'] = fusion_threshold
    config['fusion']['weights']['wyckoff'] = wyckoff
    config['fusion']['weights']['momentum'] = momentum

    # Recalculate remaining weights (smc + hob + temporal = remainder)
    total = wyckoff + momentum
    remaining = 1.0 - total
    # Keep SMC at 0.15 from baseline, split rest between HOB and temporal
    config['fusion']['weights']['smc'] = 0.15
    hob_temporal_split = remaining - 0.15
    config['fusion']['weights']['hob'] = hob_temporal_split / 2
    # Temporal gets remainder (if defined in config)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    return output_path

def run_backtest_with_config(config_path: str, asset: str) -> dict:
    """Run optimizer with specific config and return results"""
    output_file = f"sweep_results_{asset}_{Path(config_path).stem}.json"

    cmd = [
        'python3', 'bin/optimize_v19.py',
        '--asset', asset,
        '--mode', 'quick',  # Use quick mode for speed
        '--output', output_file
    ]

    print(f"\n{'='*70}")
    print(f"Testing: {Path(config_path).stem}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return None

    # Parse results
    try:
        with open(output_file) as f:
            results = json.load(f)

        if not results:
            return None

        # Get best config by Sharpe
        best = max(results, key=lambda x: x.get('sharpe', -999))
        return best

    except Exception as e:
        print(f"⚠️  Error parsing results: {e}")
        return None

def main():
    print("🎯 Bull Machine v1.8.6 - Threshold Sensitivity Sweep")
    print("="*70)

    # Load baseline BTC conservative config
    with open('configs/v18/BTC_conservative.json') as f:
        base_config = json.load(f)

    # Define test matrix (from baseline analysis)
    # BTC best: fusion=0.65, wyckoff=0.25, momentum=0.31
    # ETH best conservative: fusion=0.74, wyckoff=0.25, momentum=0.23
    # ETH best aggressive: fusion=0.62, wyckoff=0.20, momentum=0.23

    test_matrix = [
        # Threshold, Wyckoff, Momentum, Label
        (0.60, 0.25, 0.30, "aggressive_balanced"),
        (0.62, 0.20, 0.23, "eth_aggressive_winner"),
        (0.65, 0.25, 0.31, "btc_winner"),
        (0.68, 0.25, 0.30, "moderate_balanced"),
        (0.70, 0.25, 0.25, "conservative_balanced"),
        (0.74, 0.25, 0.23, "eth_conservative_winner"),
    ]

    results_summary = []

    for threshold, wyckoff, momentum, label in test_matrix:
        # Test on BTC
        config_path = f"configs/sweep/BTC_{label}_t{int(threshold*100)}.json"
        create_config_variant(
            base_config, threshold, wyckoff, momentum, config_path
        )

        btc_result = run_backtest_with_config(config_path, 'BTC')

        if btc_result:
            results_summary.append({
                'asset': 'BTC',
                'label': label,
                'fusion_threshold': threshold,
                'wyckoff_weight': wyckoff,
                'momentum_weight': momentum,
                'trades': btc_result.get('trades', 0),
                'win_rate': btc_result.get('win_rate', 0),
                'profit_factor': btc_result.get('pf', 0),
                'sharpe': btc_result.get('sharpe', 0),
                'total_return': btc_result.get('total_return', 0),
                'avg_r': btc_result.get('avg_r', 0)
            })

        # Test on ETH
        eth_result = run_backtest_with_config(config_path, 'ETH')

        if eth_result:
            results_summary.append({
                'asset': 'ETH',
                'label': label,
                'fusion_threshold': threshold,
                'wyckoff_weight': wyckoff,
                'momentum_weight': momentum,
                'trades': eth_result.get('trades', 0),
                'win_rate': eth_result.get('win_rate', 0),
                'profit_factor': eth_result.get('pf', 0),
                'sharpe': eth_result.get('sharpe', 0),
                'total_return': eth_result.get('total_return', 0),
                'avg_r': eth_result.get('avg_r', 0)
            })

    # Create summary report
    df = pd.DataFrame(results_summary)

    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY SWEEP RESULTS")
    print("="*70)

    # Sort by asset and Sharpe
    df_sorted = df.sort_values(['asset', 'sharpe'], ascending=[True, False])

    print("\n📊 BTC Results (sorted by Sharpe):")
    print(df_sorted[df_sorted['asset'] == 'BTC'].to_string(index=False))

    print("\n📊 ETH Results (sorted by Sharpe):")
    print(df_sorted[df_sorted['asset'] == 'ETH'].to_string(index=False))

    # Save to CSV
    output_csv = "threshold_sensitivity_sweep.csv"
    df_sorted.to_csv(output_csv, index=False)
    print(f"\n💾 Full results saved to: {output_csv}")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Best configs per asset
    btc_best = df[df['asset'] == 'BTC'].nlargest(1, 'sharpe').iloc[0]
    eth_best = df[df['asset'] == 'ETH'].nlargest(1, 'sharpe').iloc[0]

    print(f"\n🏆 Best BTC Config:")
    print(f"   Label: {btc_best['label']}")
    print(f"   Fusion Threshold: {btc_best['fusion_threshold']:.2f}")
    print(f"   Trades: {btc_best['trades']:.0f}")
    print(f"   Win Rate: {btc_best['win_rate']:.1f}%")
    print(f"   Profit Factor: {btc_best['profit_factor']:.3f}")
    print(f"   Sharpe: {btc_best['sharpe']:.3f}")
    print(f"   Return: {btc_best['total_return']:.1f}%")

    print(f"\n🏆 Best ETH Config:")
    print(f"   Label: {eth_best['label']}")
    print(f"   Fusion Threshold: {eth_best['fusion_threshold']:.2f}")
    print(f"   Trades: {eth_best['trades']:.0f}")
    print(f"   Win Rate: {eth_best['win_rate']:.1f}%")
    print(f"   Profit Factor: {eth_best['profit_factor']:.3f}")
    print(f"   Sharpe: {eth_best['sharpe']:.3f}")
    print(f"   Return: {eth_best['total_return']:.1f}%")

    # Threshold impact analysis
    print(f"\n📈 Threshold Impact (median metrics by threshold):")
    threshold_impact = df.groupby('fusion_threshold').agg({
        'trades': 'median',
        'win_rate': 'median',
        'profit_factor': 'median',
        'sharpe': 'median'
    }).round(2)
    print(threshold_impact)

    print("\n✅ Threshold sensitivity sweep complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
