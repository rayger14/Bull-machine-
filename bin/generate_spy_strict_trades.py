#!/usr/bin/env python3
"""
Generate Trades CSV from SPY Strict Optimizer Best Config

Exports detailed trade log from rank #9 (highest PNL: $1,208.27)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from dataclasses import asdict
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest


def main():
    # Load SPY full year 2024 data
    df = pd.read_parquet('data/features_mtf/SPY_1H_2024-01-01_to_2025-10-17.parquet')

    # Filter to actual 2024 data
    df = df[(df.index >= '2024-01-01') & (df.index <= '2024-12-31')].copy()

    print(f"Loaded {len(df)} bars of SPY 2024 data")

    # Load best strict config results
    with open('reports/optuna_results/SPY_knowledge_v3_strict_best_configs.json') as f:
        results = json.load(f)

    # Get rank #9 (highest PNL)
    best_config = results['top_10_configs'][8]  # 0-indexed
    params_dict = best_config['params']

    print(f"\nUsing Rank #9 Config:")
    print(f"  Expected PNL: ${best_config['metrics']['total_pnl']:,.2f}")
    print(f"  Expected Trades: {best_config['metrics']['total_trades']}")
    print(f"  Win Rate: {best_config['metrics']['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {best_config['metrics']['profit_factor']:.2f}")
    print(f"  Max Drawdown: {best_config['metrics']['max_drawdown']*100:.2f}%")
    print()

    # Build params
    params = KnowledgeParams(
        wyckoff_weight=params_dict['wyckoff_weight'],
        liquidity_weight=params_dict['liquidity_weight'],
        momentum_weight=params_dict['momentum_weight'],
        macro_weight=params_dict['macro_weight'],
        pti_weight=params_dict['pti_weight'],
        tier1_threshold=params_dict['tier1_threshold'],
        tier2_threshold=params_dict['tier2_threshold'],
        tier3_threshold=params_dict['tier3_threshold'],
        require_m1m2_confirmation=params_dict['require_m1m2_confirmation'],
        require_macro_alignment=params_dict['require_macro_alignment'],
        atr_stop_mult=params_dict['atr_stop_mult'],
        trailing_atr_mult=params_dict['trailing_atr_mult'],
        max_hold_bars=params_dict['max_hold_bars'],
        max_risk_pct=params_dict['max_risk_pct'],
        volatility_scaling=params_dict['volatility_scaling'],
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    # Run backtest
    print("Running backtest...")
    backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
    results = backtest.run()

    print(f"\nBacktest Results:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Total PNL: ${results['total_pnl']:,.2f}")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"  Final Equity: ${results['final_equity']:,.2f}")

    # Convert trades to DataFrame
    trades_data = []
    running_equity = 10000.0

    for i, t in enumerate(results['trades'], 1):
        # Calculate PNL for this trade
        direction_mult = 1 if t.direction > 0 else -1
        pnl = direction_mult * (t.exit_price - t.entry_price) / t.entry_price * t.position_size

        running_equity += pnl

        # Calculate trade duration
        duration_hours = (t.exit_time - t.entry_time).total_seconds() / 3600

        trades_data.append({
            'trade_num': i,
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'position_size': t.position_size,
            'direction': 'long' if t.direction > 0 else 'short',
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'duration_hours': duration_hours,
            'pnl': pnl,
            'pnl_pct': (pnl / t.position_size) * 100,
            'running_equity': running_equity,
            'drawdown_from_peak': max(0, (max([10000.0] + [td['running_equity'] for td in trades_data]) - running_equity) / max([10000.0] + [td['running_equity'] for td in trades_data]) * 100)
        })

    # Create DataFrame and save
    trades_df = pd.DataFrame(trades_data)

    output_path = Path('reports/optuna_results/SPY_strict_rank9_trades_detailed.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"Trades CSV saved to: {output_path}")
    print(f"Total rows: {len(trades_df)}")
    print(f"{'='*80}")

    # Show summary by exit reason
    print("\nExit Reason Breakdown:")
    exit_summary = trades_df.groupby('exit_reason').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    print(exit_summary)

    # Show win/loss breakdown
    print("\nWin/Loss Breakdown:")
    trades_df['outcome'] = trades_df['pnl'].apply(lambda x: 'win' if x > 0 else 'loss')
    outcome_summary = trades_df.groupby('outcome').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    print(outcome_summary)

    # Show monthly breakdown
    print("\nMonthly Breakdown:")
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
    monthly = trades_df.groupby('month').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    print(monthly)

    # Show top 5 winners and losers
    print("\nTop 5 Winning Trades:")
    top_winners = trades_df.nlargest(5, 'pnl')[['trade_num', 'entry_time', 'exit_time', 'direction', 'pnl', 'pnl_pct', 'exit_reason']]
    print(top_winners.to_string(index=False))

    print("\nTop 5 Losing Trades:")
    top_losers = trades_df.nsmallest(5, 'pnl')[['trade_num', 'entry_time', 'exit_time', 'direction', 'pnl', 'pnl_pct', 'exit_reason']]
    print(top_losers.to_string(index=False))

    # Show duration statistics
    print("\nTrade Duration Statistics:")
    print(f"  Mean: {trades_df['duration_hours'].mean():.1f} hours")
    print(f"  Median: {trades_df['duration_hours'].median():.1f} hours")
    print(f"  Min: {trades_df['duration_hours'].min():.1f} hours")
    print(f"  Max: {trades_df['duration_hours'].max():.1f} hours")
    print(f"  Std: {trades_df['duration_hours'].std():.1f} hours")


if __name__ == '__main__':
    main()
