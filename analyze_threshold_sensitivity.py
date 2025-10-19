#!/usr/bin/env python3
"""
Analyze threshold sensitivity from existing ML dataset
No need to rerun optimizations - data already captured
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    print("🎯 Bull Machine v1.8.6 - Threshold Sensitivity Analysis")
    print("="*70)

    # Load the ML dataset
    dataset_path = Path("data/ml/optimization_results.parquet")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    df = pd.read_parquet(dataset_path)

    print(f"\n📊 Loaded {len(df)} optimization results")
    print(f"   Assets: {df['asset'].unique().tolist()}")
    print(f"   Date range: {df['start_date'].min()} to {df['end_date'].max()}")

    # Filter for meaningful results (at least 10 trades)
    df_filtered = df[df['total_trades'] >= 10].copy()
    print(f"\n✂️  Filtered to {len(df_filtered)} configs with ≥10 trades")

    # Analyze by threshold bins
    df_filtered['threshold_bin'] = pd.cut(
        df_filtered['config_fusion_threshold'],
        bins=[0.50, 0.62, 0.66, 0.70, 0.75, 1.0],
        labels=['0.55-0.61', '0.62-0.65', '0.66-0.69', '0.70-0.74', '0.75+']
    )

    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)

    # Overall stats by threshold
    print("\n📊 Overall Performance by Threshold Range:")
    print("-"*70)

    threshold_stats = df_filtered.groupby('threshold_bin').agg({
        'pf': ['count', 'median', 'mean', lambda x: (x >= 1.0).sum()],
        'sharpe': ['median', 'mean'],
        'total_trades': ['median', 'mean'],
        'win_rate': ['median', 'mean'],
        'total_return_pct': ['median', 'mean']
    }).round(3)

    threshold_stats.columns = ['_'.join(col) for col in threshold_stats.columns]
    threshold_stats = threshold_stats.rename(columns={
        'pf_count': 'n_configs',
        'pf_median': 'median_pf',
        'pf_mean': 'mean_pf',
        'pf_<lambda_0>': 'profitable_count',
        'sharpe_median': 'median_sharpe',
        'sharpe_mean': 'mean_sharpe',
        'total_trades_median': 'median_trades',
        'total_trades_mean': 'mean_trades',
        'win_rate_median': 'median_wr',
        'win_rate_mean': 'mean_wr',
        'total_return_pct_median': 'median_return',
        'total_return_pct_mean': 'mean_return'
    })

    print(threshold_stats)

    # By asset
    print("\n" + "="*70)
    print("BTC THRESHOLD SENSITIVITY")
    print("="*70)

    btc = df_filtered[df_filtered['asset'] == 'BTC']
    if len(btc) > 0:
        btc_stats = btc.groupby('threshold_bin').agg({
            'pf': ['count', 'median', lambda x: (x >= 1.0).sum()],
            'sharpe': 'median',
            'total_trades': 'median',
            'win_rate': 'median',
            'total_return_pct': 'median'
        }).round(3)

        btc_stats.columns = ['n_configs', 'median_pf', 'profitable', 'median_sharpe',
                            'median_trades', 'median_wr', 'median_return']
        btc_stats['profitable_pct'] = (btc_stats['profitable'] / btc_stats['n_configs'] * 100).round(1)

        print(btc_stats)

        # Top 5 BTC configs per threshold range
        print("\n🏆 Top 5 BTC Configs by Threshold Range:")
        for thresh_bin in btc['threshold_bin'].dropna().unique():
            subset = btc[btc['threshold_bin'] == thresh_bin].nlargest(3, 'sharpe')
            if len(subset) > 0:
                print(f"\n  Threshold {thresh_bin}:")
                for _, row in subset.iterrows():
                    print(f"    fusion={row['config_fusion_threshold']:.2f}, "
                          f"wyckoff={row['config_wyckoff_weight']:.2f}, "
                          f"momentum={row['config_momentum_weight']:.2f} | "
                          f"PF={row['pf']:.3f}, Sharpe={row['sharpe']:.3f}, "
                          f"Trades={row['total_trades']:.0f}, WR={row['win_rate']:.1f}%")

    print("\n" + "="*70)
    print("ETH THRESHOLD SENSITIVITY")
    print("="*70)

    eth = df_filtered[df_filtered['asset'] == 'ETH']
    if len(eth) > 0:
        eth_stats = eth.groupby('threshold_bin').agg({
            'pf': ['count', 'median', lambda x: (x >= 1.0).sum()],
            'sharpe': 'median',
            'total_trades': 'median',
            'win_rate': 'median',
            'total_return_pct': 'median'
        }).round(3)

        eth_stats.columns = ['n_configs', 'median_pf', 'profitable', 'median_sharpe',
                            'median_trades', 'median_wr', 'median_return']
        eth_stats['profitable_pct'] = (eth_stats['profitable'] / eth_stats['n_configs'] * 100).round(1)

        print(eth_stats)

        # Top 5 ETH configs per threshold range
        print("\n🏆 Top 5 ETH Configs by Threshold Range:")
        for thresh_bin in eth['threshold_bin'].dropna().unique():
            subset = eth[eth['threshold_bin'] == thresh_bin].nlargest(3, 'sharpe')
            if len(subset) > 0:
                print(f"\n  Threshold {thresh_bin}:")
                for _, row in subset.iterrows():
                    print(f"    fusion={row['config_fusion_threshold']:.2f}, "
                          f"wyckoff={row['config_wyckoff_weight']:.2f}, "
                          f"momentum={row['config_momentum_weight']:.2f} | "
                          f"PF={row['pf']:.3f}, Sharpe={row['sharpe']:.3f}, "
                          f"Trades={row['total_trades']:.0f}, WR={row['win_rate']:.1f}%")

    # Correlation analysis
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    print("\n📈 Correlation of fusion_threshold with performance metrics:")
    correlations = df_filtered[['config_fusion_threshold', 'pf', 'sharpe', 'total_trades',
                                'win_rate', 'total_return_pct']].corr()['config_fusion_threshold'].drop('config_fusion_threshold')
    print(correlations.round(3))

    # Optimal threshold recommendation
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find threshold range with best median profitability
    best_thresh_all = threshold_stats.nlargest(1, 'median_pf').index[0]
    print(f"\n🎯 Best Overall Threshold Range: {best_thresh_all}")
    print(f"   Median PF: {threshold_stats.loc[best_thresh_all, 'median_pf']:.3f}")
    print(f"   Median Sharpe: {threshold_stats.loc[best_thresh_all, 'median_sharpe']:.3f}")
    print(f"   Profitable Configs: {threshold_stats.loc[best_thresh_all, 'profitable_count']:.0f} / "
          f"{threshold_stats.loc[best_thresh_all, 'n_configs']:.0f} "
          f"({threshold_stats.loc[best_thresh_all, 'profitable_count']/threshold_stats.loc[best_thresh_all, 'n_configs']*100:.1f}%)")

    if len(btc) > 0:
        best_thresh_btc = btc_stats.nlargest(1, 'median_pf').index[0]
        print(f"\n🎯 Best BTC Threshold Range: {best_thresh_btc}")
        print(f"   Median PF: {btc_stats.loc[best_thresh_btc, 'median_pf']:.3f}")
        print(f"   Median Sharpe: {btc_stats.loc[best_thresh_btc, 'median_sharpe']:.3f}")
        print(f"   Profitable: {btc_stats.loc[best_thresh_btc, 'profitable_pct']:.1f}%")

    if len(eth) > 0:
        best_thresh_eth = eth_stats.nlargest(1, 'median_pf').index[0]
        print(f"\n🎯 Best ETH Threshold Range: {best_thresh_eth}")
        print(f"   Median PF: {eth_stats.loc[best_thresh_eth, 'median_pf']:.3f}")
        print(f"   Median Sharpe: {eth_stats.loc[best_thresh_eth, 'median_sharpe']:.3f}")
        print(f"   Profitable: {eth_stats.loc[best_thresh_eth, 'profitable_pct']:.1f}%")

    # Trade frequency impact
    print("\n📊 Trade Frequency by Threshold:")
    trade_freq = threshold_stats[['n_configs', 'median_trades', 'mean_trades']].copy()
    print(trade_freq)

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Calculate correlation direction
    pf_corr = correlations['pf']
    trade_corr = correlations['total_trades']

    if pf_corr > 0.05:
        print("\n✅ Higher thresholds → Better profitability (positive correlation)")
        print("   Recommendation: Use higher thresholds (0.65-0.75) for quality over quantity")
    elif pf_corr < -0.05:
        print("\n⚠️  Higher thresholds → Worse profitability (negative correlation)")
        print("   Recommendation: Use lower thresholds (0.55-0.65) for more opportunities")
    else:
        print("\n➖ Threshold has minimal impact on profitability")
        print("   Recommendation: Focus on domain weights and other parameters")

    if trade_corr < -0.1:
        print(f"\n📉 Higher thresholds strongly reduce trade frequency (r={trade_corr:.2f})")
        print("   Trade-off: Selectivity vs. opportunity count")

    # Save detailed results
    output_file = "threshold_sensitivity_analysis.csv"
    df_filtered_export = df_filtered[[
        'asset', 'config_fusion_threshold', 'config_wyckoff_weight',
        'config_momentum_weight', 'total_trades', 'win_rate', 'pf',
        'sharpe', 'max_dd_pct', 'total_return_pct', 'threshold_bin'
    ]].sort_values(['asset', 'sharpe'], ascending=[True, False])

    df_filtered_export.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    print("\n✅ Threshold sensitivity analysis complete!")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
