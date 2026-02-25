#!/usr/bin/env python3
"""Run multi-position backtest on full 2022-2023 period."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bin.baseline_wyckoff_multi_position import MultiPositionBacktester
import pandas as pd

# Run backtest
backtester = MultiPositionBacktester(
    feature_store_path='data/features_mtf/BTC_1H_LATEST.parquet',
    config_path='configs/ultra_permissive_v3.json',
    initial_capital=10000.0,
    risk_per_trade_pct=0.02,
    max_positions=3,
    min_relative_score=0.7
)

print("\nRunning multi-position backtest on 2022-2023...")
results = backtester.run(start='2022-01-01', end='2023-12-31')

# Print results
print("\n" + "=" * 80)
print("MULTI-POSITION BACKTEST RESULTS (2022-2023)")
print("=" * 80)
print(f"Initial Capital: $10,000")
print(f"Period: 2022-01-01 to 2023-12-31 (2 years)")
print(f"Max Positions: 3")
print(f"Min Relative Score: 0.7 (70% of best)")
print("\nPERFORMANCE METRICS:")
print("-" * 80)
print(f"Total Trades:  {results['metrics']['total_trades']}")
print(f"Winners:       {results['metrics']['winners']}")
print(f"Losers:        {results['metrics']['losers']}")
print(f"Win Rate:      {results['metrics']['win_rate']:.2f}%")
print(f"Profit Factor: {results['metrics']['profit_factor']:.3f}")
print(f"Sharpe Ratio:  {results['metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown:  {results['metrics']['max_drawdown']:.2f}%")
print(f"Total PnL:     ${results['metrics']['total_pnl']:,.2f}")
print(f"Gross Profit:  ${results['metrics']['gross_profit']:,.2f}")
print(f"Gross Loss:    ${results['metrics']['gross_loss']:,.2f}")
print(f"Avg Win:       ${results['metrics']['avg_win']:,.2f}")
print(f"Avg Loss:      ${results['metrics']['avg_loss']:,.2f}")

print("\nARCHETYPE BREAKDOWN:")
print("-" * 80)
df_trades = pd.DataFrame(results['trades'])
if len(df_trades) > 0:
    archetype_stats = df_trades.groupby('archetype').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)

    for archetype in archetype_stats.index:
        count = archetype_stats.loc[archetype, ('pnl', 'count')]
        total_pnl = archetype_stats.loc[archetype, ('pnl', 'sum')]
        avg_pnl = archetype_stats.loc[archetype, ('pnl', 'mean')]
        print(f"{archetype:24s}: {count:4.0f} trades, PnL: ${total_pnl:+10.2f}, Avg: ${avg_pnl:+7.2f}")

    unique_archetypes = df_trades['archetype'].nunique()
    print(f"\n{'TOTAL UNIQUE ARCHETYPES':<24s}: {unique_archetypes}/16 ({unique_archetypes/16*100:.1f}% diversity)")

print("\nCOMPARISON TO SINGLE-POSITION:")
print("-" * 80)
print("Single-Position Results (from previous session):")
print("  Total Trades: 822")
print("  Win Rate: 61.8%")
print("  Profit Factor: 1.14")
print("  Sharpe: 0.84")
print("  Total PnL: ~$1,740")
print(f"\nMulti-Position Results:")
print(f"  Total Trades: {results['metrics']['total_trades']}")
print(f"  Win Rate: {results['metrics']['win_rate']:.1f}%")
print(f"  Profit Factor: {results['metrics']['profit_factor']:.2f}")
print(f"  Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
print(f"  Total PnL: ${results['metrics']['total_pnl']:,.2f}")

# Calculate improvement
pf_improvement = ((results['metrics']['profit_factor'] / 1.14) - 1) * 100
sharpe_improvement = ((results['metrics']['sharpe_ratio'] / 0.84) - 1) * 100
print(f"\nIMPROVEMENT:")
print(f"  Profit Factor: {pf_improvement:+.1f}%")
print(f"  Sharpe Ratio: {sharpe_improvement:+.1f}%")

# Save trades
df_trades.to_csv('MULTI_POSITION_2022_2023_TRADES.csv', index=False)
print(f"\nTrades saved to: MULTI_POSITION_2022_2023_TRADES.csv")
