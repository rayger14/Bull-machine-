#!/usr/bin/env python3
"""
Test All Top-10 SPY Configs from v3 Optimizer

Systematically tests each config to find which (if any) are profitable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest


def test_config(df, config_params, rank):
    """Test a single configuration"""
    params = KnowledgeParams(
        wyckoff_weight=config_params['wyckoff_weight'],
        liquidity_weight=config_params['liquidity_weight'],
        momentum_weight=config_params['momentum_weight'],
        macro_weight=config_params['macro_weight'],
        pti_weight=config_params['pti_weight'],
        tier1_threshold=config_params['tier1_threshold'],
        tier2_threshold=config_params['tier2_threshold'],
        tier3_threshold=config_params['tier3_threshold'],
        require_m1m2_confirmation=config_params['require_m1m2_confirmation'],
        require_macro_alignment=config_params['require_macro_alignment'],
        atr_stop_mult=config_params['atr_stop_mult'],
        trailing_atr_mult=config_params['trailing_atr_mult'],
        max_hold_bars=config_params['max_hold_bars'],
        max_risk_pct=config_params['max_risk_pct'],
        volatility_scaling=config_params['volatility_scaling'],
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
    results = backtest.run()

    return {
        'rank': rank,
        'total_pnl': results['total_pnl'],
        'total_trades': results['total_trades'],
        'win_rate': results['win_rate'],
        'profit_factor': results['profit_factor'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'final_equity': results['final_equity']
    }


def main():
    print("=" * 80)
    print("SPY v3.0 - Testing All Top-10 Optimizer Configs")
    print("=" * 80)

    # Load SPY full year 2024 data
    df = pd.read_parquet('data/features_mtf/SPY_1H_2024-01-01_to_2024-12-31.parquet')
    print(f"Loaded {len(df)} bars of SPY data\n")

    # Load all top-10 configs
    with open('reports/optuna_results/SPY_knowledge_v3_best_configs.json') as f:
        optimizer_results = json.load(f)

    configs = optimizer_results['top_10_configs']

    results = []
    for config in configs:
        rank = config['rank']
        print(f"Testing Rank #{rank}...", end=" ", flush=True)

        result = test_config(df, config['params'], rank)
        results.append(result)

        status = "✅ PROFIT" if result['total_pnl'] > 0 else "❌ LOSS"
        print(f"{status} - PNL: ${result['total_pnl']:.2f}, Trades: {result['total_trades']}")

    # Sort by PNL
    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # Display results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (sorted by PNL)")
    print("=" * 80)

    header = f"{'Rank':<6} {'PNL':>12} {'Trades':>8} {'Win%':>8} {'PF':>8} {'Sharpe':>10} {'MaxDD':>8}"
    print(header)
    print("-" * 80)

    profitable_count = 0
    for r in results:
        if r['total_pnl'] > 0:
            profitable_count += 1
            status_icon = "✅"
        else:
            status_icon = "❌"

        line = f"{status_icon} #{r['rank']:<3} ${r['total_pnl']:>10,.2f} {r['total_trades']:>8} "
        line += f"{r['win_rate']*100:>7.1f}% {r['profit_factor']:>8.2f} "
        line += f"{r['sharpe_ratio']:>10.2f} {r['max_drawdown']*100:>7.2f}%"
        print(line)

    print("=" * 80)
    print(f"\nProfitable configs: {profitable_count}/10")

    if profitable_count > 0:
        best = results[0]
        print(f"\nBEST CONFIG: Rank #{best['rank']}")
        print(f"  PNL: ${best['total_pnl']:,.2f}")
        print(f"  Trades: {best['total_trades']}")
        print(f"  Win Rate: {best['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
    else:
        print("\n⚠️  NO PROFITABLE CONFIGS FOUND!")
        print("Recommendation: Re-run optimizer with:")
        print("  - Minimum 20 trades requirement")
        print("  - Lower thresholds for more trade frequency")
        print("  - Equity-specific parameter ranges")

    print("=" * 80)


if __name__ == '__main__':
    main()
