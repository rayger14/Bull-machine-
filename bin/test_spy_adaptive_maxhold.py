#!/usr/bin/env python3
"""
Test SPY Adaptive Max-Hold Implementation

Compares:
1. Baseline: Fixed max_hold=24h (adaptive_max_hold=False)
2. Adaptive: Smart extension 24h → 48h/72h (adaptive_max_hold=True)

Uses the best SPY strict optimizer config (rank #9).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest


def main():
    print("=" * 80)
    print("SPY ADAPTIVE MAX-HOLD TEST")
    print("=" * 80)

    # Load SPY 2024 data (use the properly rebuilt file with Wyckoff/M1/M2 features)
    df = pd.read_parquet('data/features_mtf/SPY_1H_2024-01-01_to_2024-12-31.parquet')

    print(f"Loaded {len(df)} bars of SPY 2024 data\n")

    # Load best strict config (rank #9)
    with open('reports/optuna_results/SPY_knowledge_v3_strict_best_configs.json') as f:
        results = json.load(f)

    best_config = results['top_10_configs'][8]  # Rank #9 (0-indexed)
    params_dict = best_config['params']

    print(f"Using Rank #9 Config:")
    print(f"  Expected PNL (baseline): ${best_config['metrics']['total_pnl']:,.2f}")
    print(f"  Expected Trades: {best_config['metrics']['total_trades']}")
    print(f"  Win Rate: {best_config['metrics']['win_rate']*100:.1f}%")
    print(f"  Max Drawdown: {best_config['metrics']['max_drawdown']*100:.2f}%")
    print()

    # ===== TEST 1: BASELINE (Fixed max_hold) =====
    print("=" * 80)
    print("TEST 1: BASELINE (Fixed max_hold=24h)")
    print("=" * 80)

    params_baseline = KnowledgeParams(
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
        breakeven_after_tp1=True,
        adaptive_max_hold=False  # BASELINE: Fixed max-hold
    )

    backtest_baseline = KnowledgeAwareBacktest(df, params_baseline, starting_capital=10000.0)
    results_baseline = backtest_baseline.run()

    print(f"\nBaseline Results:")
    print(f"  Total Trades: {results_baseline['total_trades']}")
    print(f"  Total PNL: ${results_baseline['total_pnl']:,.2f}")
    print(f"  Win Rate: {results_baseline['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {results_baseline['profit_factor']:.2f}")
    print(f"  Max Drawdown: {results_baseline['max_drawdown']*100:.2f}%")
    print(f"  Final Equity: ${results_baseline['final_equity']:,.2f}")

    # Count exit reasons
    max_hold_count_baseline = sum(1 for t in results_baseline['trades'] if 'max_hold' in str(t.exit_reason))
    print(f"  Max-hold exits: {max_hold_count_baseline}")

    # ===== TEST 2: ADAPTIVE MAX-HOLD =====
    print("\n" + "=" * 80)
    print("TEST 2: ADAPTIVE MAX-HOLD (24h → 48h/72h based on regime)")
    print("=" * 80)

    params_adaptive = KnowledgeParams(
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
        breakeven_after_tp1=True,
        adaptive_max_hold=True  # ADAPTIVE: Market-aware extension
    )

    backtest_adaptive = KnowledgeAwareBacktest(df, params_adaptive, starting_capital=10000.0)
    results_adaptive = backtest_adaptive.run()

    print(f"\nAdaptive Results:")
    print(f"  Total Trades: {results_adaptive['total_trades']}")
    print(f"  Total PNL: ${results_adaptive['total_pnl']:,.2f}")
    print(f"  Win Rate: {results_adaptive['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {results_adaptive['profit_factor']:.2f}")
    print(f"  Max Drawdown: {results_adaptive['max_drawdown']*100:.2f}%")
    print(f"  Final Equity: ${results_adaptive['final_equity']:,.2f}")

    # Count exit reasons
    max_hold_count_adaptive = sum(1 for t in results_adaptive['trades'] if 'max_hold' in str(t.exit_reason))
    print(f"  Max-hold exits: {max_hold_count_adaptive}")

    # ===== COMPARISON =====
    print("\n" + "=" * 80)
    print("COMPARISON: Adaptive vs Baseline")
    print("=" * 80)

    delta_pnl = results_adaptive['total_pnl'] - results_baseline['total_pnl']
    delta_pct = (delta_pnl / results_baseline['total_pnl'] * 100) if results_baseline['total_pnl'] > 0 else 0
    delta_trades = results_adaptive['total_trades'] - results_baseline['total_trades']
    delta_winrate = (results_adaptive['win_rate'] - results_baseline['win_rate']) * 100
    delta_dd = (results_adaptive['max_drawdown'] - results_baseline['max_drawdown']) * 100

    print(f"\nPNL Improvement:")
    print(f"  Baseline PNL: ${results_baseline['total_pnl']:,.2f}")
    print(f"  Adaptive PNL: ${results_adaptive['total_pnl']:,.2f}")
    print(f"  Delta: ${delta_pnl:+,.2f} ({delta_pct:+.1f}%)")

    print(f"\nTrade Count:")
    print(f"  Baseline: {results_baseline['total_trades']}")
    print(f"  Adaptive: {results_adaptive['total_trades']}")
    print(f"  Delta: {delta_trades:+d}")

    print(f"\nWin Rate:")
    print(f"  Baseline: {results_baseline['win_rate']*100:.1f}%")
    print(f"  Adaptive: {results_adaptive['win_rate']*100:.1f}%")
    print(f"  Delta: {delta_winrate:+.1f}%")

    print(f"\nMax Drawdown:")
    print(f"  Baseline: {results_baseline['max_drawdown']*100:.2f}%")
    print(f"  Adaptive: {results_adaptive['max_drawdown']*100:.2f}%")
    print(f"  Delta: {delta_dd:+.2f}%")

    print(f"\nMax-hold Exits:")
    print(f"  Baseline: {max_hold_count_baseline}")
    print(f"  Adaptive: {max_hold_count_adaptive}")
    print(f"  Reduction: {max_hold_count_baseline - max_hold_count_adaptive} fewer forced exits")

    # ===== VERDICT =====
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if delta_pnl > 100:
        print(f"✅ ADAPTIVE LOGIC WORKS! +${delta_pnl:,.2f} improvement ({delta_pct:+.1f}%)")
        print(f"   Recommendation: Enable adaptive_max_hold=True for SPY")
    elif delta_pnl > 0:
        print(f"✅ MODEST IMPROVEMENT: +${delta_pnl:,.2f} ({delta_pct:+.1f}%)")
        print(f"   Recommendation: Consider enabling adaptive_max_hold=True")
    elif delta_pnl > -50:
        print(f"⚠️  MINIMAL IMPACT: ${delta_pnl:+,.2f} ({delta_pct:+.1f}%)")
        print(f"   Recommendation: Optional, no strong benefit/harm")
    else:
        print(f"❌ NEGATIVE IMPACT: ${delta_pnl:+,.2f} ({delta_pct:+.1f}%)")
        print(f"   Recommendation: Keep adaptive_max_hold=False (baseline better)")

    print("\n" + "=" * 80)
    print(f"Expected Improvement (from what-if analysis): +$1,315 to +$1,772")
    print(f"Actual Improvement: ${delta_pnl:+,.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
