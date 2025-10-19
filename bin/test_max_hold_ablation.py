#!/usr/bin/env python3
"""
Max Hold Ablation Study - Test impact of different time caps on performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest

# Load full year BTC data
df = pd.read_parquet('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet')

# Load best config
with open('reports/optuna_results/BTC_knowledge_v3_best_configs.json') as f:
    base_config = json.load(f)['top_10_configs'][0]['params']

# Test different max_hold settings
max_hold_tests = [
    (72, '3 days'),
    (120, '5 days'),
    (168, '7 days'),
    (196, '8.2 days - BASELINE'),
    (240, '10 days'),
    (336, '14 days'),
    (999999, 'NO CAP')
]

print('=' * 100)
print('BTC v3.0 - MAX_HOLD ABLATION STUDY')
print('=' * 100)
print('Testing impact of different max_hold time caps on full year 2024 performance')
print()

results = []
for max_hold_bars, label in max_hold_tests:
    params = KnowledgeParams(
        wyckoff_weight=base_config['wyckoff_weight'],
        liquidity_weight=base_config['liquidity_weight'],
        momentum_weight=base_config['momentum_weight'],
        macro_weight=base_config['macro_weight'],
        pti_weight=base_config['pti_weight'],
        tier1_threshold=base_config['tier1_threshold'],
        tier2_threshold=base_config['tier2_threshold'],
        tier3_threshold=base_config['tier3_threshold'],
        require_m1m2_confirmation=base_config['require_m1m2_confirmation'],
        require_macro_alignment=base_config['require_macro_alignment'],
        atr_stop_mult=base_config['atr_stop_mult'],
        trailing_atr_mult=base_config['trailing_atr_mult'],
        max_hold_bars=max_hold_bars,
        max_risk_pct=base_config['max_risk_pct'],
        volatility_scaling=base_config['volatility_scaling'],
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
    result = backtest.run()

    results.append({
        'max_hold_bars': max_hold_bars,
        'label': label,
        'total_pnl': result['total_pnl'],
        'total_trades': result['total_trades'],
        'win_rate': result['win_rate'],
        'profit_factor': result['profit_factor'],
        'sharpe_ratio': result['sharpe_ratio'],
        'max_drawdown': result['max_drawdown']
    })

header = f"{'Setting':<25} {'PNL':>12} {'Trades':>8} {'Win%':>8} {'PF':>8} {'Sharpe':>8} {'MaxDD':>8}"
print(header)
print('-' * 100)

baseline_pnl = None
for r in results:
    if '8.2 days - BASELINE' in r['label']:
        baseline_pnl = r['total_pnl']

for r in results:
    delta = ''
    if baseline_pnl and r['label'] != '8.2 days - BASELINE':
        pct_change = (r['total_pnl'] - baseline_pnl) / baseline_pnl * 100
        delta = f' ({pct_change:+.1f}%)'

    pnl_str = f"${r['total_pnl']:,.2f}{delta}"
    line = f"{r['label']:<25} {pnl_str:>23} {r['total_trades']:>8} {r['win_rate']*100:>7.1f}% {r['profit_factor']:>8.2f} {r['sharpe_ratio']:>8.2f} {r['max_drawdown']*100:>7.2f}%"
    print(line)

print('=' * 100)
print('KEY FINDINGS')
print('=' * 100)

best = max(results, key=lambda x: x['total_pnl'])
print(f"Best PNL: {best['label']} with ${best['total_pnl']:,.2f}")
print(f"Baseline (196 bars): ${baseline_pnl:,.2f}")
print(f"Difference: ${best['total_pnl'] - baseline_pnl:,.2f} ({(best['total_pnl'] - baseline_pnl)/baseline_pnl*100:+.1f}%)")
print('=' * 100)
