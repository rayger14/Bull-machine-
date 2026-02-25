#!/usr/bin/env python3
"""Quick test of multi-position mode."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bin.baseline_wyckoff_multi_position import MultiPositionBacktester

# Run backtest
backtester = MultiPositionBacktester(
    feature_store_path='data/features_mtf/BTC_1H_LATEST.parquet',
    config_path='configs/ultra_permissive_v3.json',
    initial_capital=10000.0,
    risk_per_trade_pct=0.02,
    max_positions=3,
    min_relative_score=0.7
)

results = backtester.run(start='2023-01-01', end='2023-03-31')

# Print results
print("\n" + "=" * 80)
print("MULTI-POSITION BACKTEST RESULTS (Q1 2023)")
print("=" * 80)
print(f"Total Trades: {results['metrics']['total_trades']}")
print(f"Winners: {results['metrics']['winners']}")
print(f"Losers: {results['metrics']['losers']}")
print(f"Win Rate: {results['metrics']['win_rate']}%")
print(f"Profit Factor: {results['metrics']['profit_factor']}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']}%")
print(f"Total PnL: ${results['metrics']['total_pnl']:,.2f}")
print("\nARCHETYPE BREAKDOWN:")
print("-" * 80)

# Get unique archetypes
import pandas as pd
df_trades = pd.DataFrame(results['trades'])
if len(df_trades) > 0:
    archetype_stats = df_trades.groupby('archetype').agg({
        'pnl': ['count', 'sum', 'mean']
    })
    print(archetype_stats)

    # Count unique archetypes
    unique_archetypes = df_trades['archetype'].nunique()
    print(f"\nUnique Archetypes Firing: {unique_archetypes}")
