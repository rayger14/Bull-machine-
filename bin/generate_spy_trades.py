#!/usr/bin/env python3
"""
Generate Trades CSV from SPY v3.0 Backtest

Creates trades CSV for SPY full year 2024 analysis
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
    df = pd.read_parquet('data/features_mtf/SPY_1H_2024-01-01_to_2024-12-31.parquet')

    # Load best v3.0 config
    with open('reports/optuna_results/SPY_knowledge_v3_best_configs.json') as f:
        best_config = json.load(f)['top_10_configs'][0]['params']

    # Build params
    params = KnowledgeParams(
        wyckoff_weight=best_config['wyckoff_weight'],
        liquidity_weight=best_config['liquidity_weight'],
        momentum_weight=best_config['momentum_weight'],
        macro_weight=best_config['macro_weight'],
        pti_weight=best_config['pti_weight'],
        tier1_threshold=best_config['tier1_threshold'],
        tier2_threshold=best_config['tier2_threshold'],
        tier3_threshold=best_config['tier3_threshold'],
        require_m1m2_confirmation=best_config['require_m1m2_confirmation'],
        require_macro_alignment=best_config['require_macro_alignment'],
        atr_stop_mult=best_config['atr_stop_mult'],
        trailing_atr_mult=best_config['trailing_atr_mult'],
        max_hold_bars=best_config['max_hold_bars'],
        max_risk_pct=best_config['max_risk_pct'],
        volatility_scaling=best_config['volatility_scaling'],
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    # Run backtest
    print("Running SPY v3.0 backtest (full year 2024)...")
    backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
    results = backtest.run()

    print(f"Total Trades: {results['total_trades']}")
    print(f"Total PNL: ${results['total_pnl']:,.2f}")

    # Convert trades to DataFrame
    trades_data = []
    for t in results['trades']:
        # Convert dataclass to dict
        trade_dict = asdict(t)

        # Calculate PNL for this trade
        direction_mult = 1 if t.direction > 0 else -1
        pnl = direction_mult * (t.exit_price - t.entry_price) / t.entry_price * t.position_size

        trades_data.append({
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'position_size': t.position_size,
            'direction': 'long' if t.direction > 0 else 'short',
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'pnl': pnl
        })

    # Create DataFrame and save
    trades_df = pd.DataFrame(trades_data)

    output_path = Path('reports/optuna_results/SPY_v3_2024_trades.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)

    print(f"\nTrades CSV saved to: {output_path}")
    print(f"Rows: {len(trades_df)}")

    # Show summary by exit reason
    if len(trades_df) > 0:
        print("\nExit Reason Breakdown:")
        print(trades_df.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2))
    else:
        print("\nNo trades generated!")


if __name__ == '__main__':
    main()
