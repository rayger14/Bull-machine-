#!/usr/bin/env python3
"""
Combine Router v10 backtest results from separate feature stores.
Avoids NaN corruption from column mismatches.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_results(results_dir):
    """Load results.json and trades.csv from a results directory."""
    results_path = Path(results_dir) / "results.json"
    trades_path = Path(results_dir) / "trades.csv"

    with open(results_path) as f:
        results = json.load(f)

    trades = pd.read_csv(trades_path)

    return results, trades

def combine_trades(trades_list):
    """Combine trade DataFrames and sort by entry_time."""
    combined = pd.concat(trades_list, ignore_index=True)
    combined['entry_time'] = pd.to_datetime(combined['entry_time'])
    combined = combined.sort_values('entry_time').reset_index(drop=True)
    return combined

def calculate_combined_metrics(trades_df, starting_capital=10000):
    """Calculate performance metrics from combined trades."""
    total_pnl = trades_df['net_pnl'].sum()
    final_equity = starting_capital + total_pnl
    return_pct = (total_pnl / starting_capital) * 100

    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calculate equity curve for Sharpe and drawdown
    equity_curve = starting_capital + trades_df['net_pnl'].cumsum()
    returns = trades_df['net_pnl'] / starting_capital

    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

    # Max drawdown
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = abs(drawdown.min())

    return {
        'total_pnl': total_pnl,
        'final_equity': final_equity,
        'return_pct': return_pct,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }

def analyze_by_period(trades_df):
    """Break down performance by year."""
    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['year'] = trades_df['entry_time'].dt.year

    yearly_stats = {}
    for year in sorted(trades_df['year'].unique()):
        year_trades = trades_df[trades_df['year'] == year]
        yearly_stats[year] = {
            'trades': len(year_trades),
            'pnl': year_trades['net_pnl'].sum(),
            'win_rate': (year_trades['net_pnl'] > 0).sum() / len(year_trades),
            'avg_pnl': year_trades['net_pnl'].mean()
        }

    return yearly_stats

def analyze_by_archetype(trades_df):
    """Analyze performance by entry archetype."""
    archetype_stats = trades_df.groupby('entry_reason').agg({
        'net_pnl': ['count', 'sum', 'mean'],
        'entry_price': 'count'
    }).round(2)

    # Add win rate
    win_rates = trades_df.groupby('entry_reason', group_keys=False).apply(
        lambda x: (x['net_pnl'] > 0).sum() / len(x), include_groups=False
    )

    result = pd.DataFrame({
        'trades': archetype_stats[('net_pnl', 'count')],
        'total_pnl': archetype_stats[('net_pnl', 'sum')],
        'avg_pnl': archetype_stats[('net_pnl', 'mean')],
        'win_rate': win_rates
    })

    return result.sort_values('total_pnl', ascending=False)

def main():
    # Load both result sets
    print("Loading 2022-2023 results...")
    results_2022_2023, trades_2022_2023 = load_results('results/router_v10_full_2022_2023')

    print("Loading 2024 results...")
    results_2024, trades_2024 = load_results('results/router_v10_full_2022_2024')

    # Combine trades
    print("\nCombining trade logs...")
    combined_trades = combine_trades([trades_2022_2023, trades_2024])

    # Calculate combined metrics
    print("Calculating combined metrics...")
    combined_metrics = calculate_combined_metrics(combined_trades, starting_capital=10000)

    # Analyze by period
    yearly_stats = analyze_by_period(combined_trades)

    # Analyze by archetype
    archetype_stats = analyze_by_archetype(combined_trades)

    # Create combined results JSON
    # Convert year keys to strings for JSON serialization
    yearly_stats_serializable = {str(k): v for k, v in yearly_stats.items()}

    combined_results = {
        'asset': 'BTC',
        'period': '2022-01-01 to 2024-12-31',
        'performance': combined_metrics,
        'yearly_breakdown': yearly_stats_serializable,
        'archetype_performance': {k: {k2: float(v2) if hasattr(v2, 'item') else v2
                                      for k2, v2 in v.items()}
                                 for k, v in archetype_stats.to_dict('index').items()},
        'router': {
            'config_switches': results_2022_2023['router']['config_switches'] + results_2024['router']['config_switches'],
            'regime_switches': results_2022_2023['router']['regime_switches'] + results_2024['router']['regime_switches']
        },
        'component_results': {
            '2022-2023': {
                'pnl': results_2022_2023['performance']['total_pnl'],
                'trades': results_2022_2023['performance']['total_trades'],
                'win_rate': results_2022_2023['performance']['win_rate']
            },
            '2024': {
                'pnl': results_2024['performance']['total_pnl'],
                'trades': results_2024['performance']['total_trades'],
                'win_rate': results_2024['performance']['win_rate']
            }
        }
    }

    # Save combined results
    output_dir = Path('results/router_v10_full_2022_2024_combined')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(combined_results, f, indent=2)

    combined_trades.to_csv(output_dir / 'trades.csv', index=False)

    # Print summary
    print("\n" + "="*80)
    print("ROUTER V10 COMBINED RESULTS (2022-2024)")
    print("="*80)
    print(f"\nOverall Performance:")
    print(f"  Total PNL: ${combined_metrics['total_pnl']:.2f}")
    print(f"  Final Equity: ${combined_metrics['final_equity']:.2f}")
    print(f"  Return: {combined_metrics['return_pct']:.2f}%")
    print(f"  Total Trades: {combined_metrics['total_trades']}")
    print(f"  Win Rate: {combined_metrics['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {combined_metrics['profit_factor']:.2f}")
    print(f"  Sharpe Ratio: {combined_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {combined_metrics['max_drawdown']*100:.2f}%")

    print(f"\nYearly Breakdown:")
    for year, stats in yearly_stats.items():
        print(f"  {year}: {stats['trades']} trades, ${stats['pnl']:.2f} PNL ({stats['win_rate']*100:.1f}% WR)")

    print(f"\nArchetype Performance (Top 5):")
    for archetype, stats in archetype_stats.head(5).iterrows():
        print(f"  {archetype}:")
        print(f"    Trades: {int(stats['trades'])}, PNL: ${stats['total_pnl']:.2f}, Avg: ${stats['avg_pnl']:.2f}, WR: {stats['win_rate']*100:.1f}%")

    print(f"\nComponent Results:")
    print(f"  2022-2023: {combined_results['component_results']['2022-2023']['trades']} trades, ${combined_results['component_results']['2022-2023']['pnl']:.2f} PNL")
    print(f"  2024: {combined_results['component_results']['2024']['trades']} trades, ${combined_results['component_results']['2024']['pnl']:.2f} PNL")

    print(f"\nResults saved to: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
