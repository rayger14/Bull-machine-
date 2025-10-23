#!/usr/bin/env python3
"""
Phase 1: Structure Invalidation Exit - Backtest Validation

Runs backtest on Phase 1 enhanced feature stores and compares to baseline.

Usage:
    python3 bin/test_phase1.py --asset BTC
    python3 bin/test_phase1.py --asset ETH
    python3 bin/test_phase1.py --asset SPY
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest

# Baseline metrics (from REPLAY_PARITY_RESOLUTION.md)
BASELINES = {
    'BTC': {
        'total_trades': 31,
        'total_pnl': 5715.29,
        'win_rate': 0.548,
        'profit_factor': 2.39,
        'max_drawdown': 0.0,  # Need to compute
    },
    'ETH': {
        'total_trades': 320,
        'total_pnl': 4701.69,
        'win_rate': None,  # Need to compute
        'profit_factor': None,  # Need to compute
        'max_drawdown': None,  # Need to compute
    },
    'SPY': {
        'total_trades': 31,
        'total_pnl': 809.38,
        'win_rate': None,  # Need to compute
        'profit_factor': None,  # Need to compute
        'max_drawdown': None,  # Need to compute
    }
}

def load_config(asset: str) -> dict:
    """Load frozen config for asset."""
    if asset == 'SPY':
        config_path = Path('configs/v3_replay_2024/SPY_2024_equity_tuned.json')
    else:
        config_path = Path(f'configs/v3_replay_2024/{asset}_2024_best.json')

    with open(config_path, 'r') as f:
        return json.load(f)

def config_to_params(config: dict) -> KnowledgeParams:
    """Convert config dict to KnowledgeParams."""
    return KnowledgeParams(
        wyckoff_weight=config['wyckoff_weight'],
        liquidity_weight=config['liquidity_weight'],
        momentum_weight=config['momentum_weight'],
        macro_weight=config['macro_weight'],
        pti_weight=config['pti_weight'],
        tier1_threshold=config['tier1_threshold'],
        tier2_threshold=config['tier2_threshold'],
        tier3_threshold=config['tier3_threshold'],
        require_m1m2_confirmation=config['require_m1m2_confirmation'],
        require_macro_alignment=config['require_macro_alignment'],
        atr_stop_mult=config['atr_stop_mult'],
        trailing_atr_mult=config['trailing_atr_mult'],
        max_hold_bars=config['max_hold_bars'],
        max_risk_pct=config['max_risk_pct'],
        volatility_scaling=config['volatility_scaling'],
        # Defaults from KnowledgeParams
        use_smart_exits=True,
        breakeven_after_tp1=True,
        partial_exit_1=0.33,
        partial_exit_2=0.33,
    )

def analyze_exit_reasons(trades: list) -> dict:
    """Analyze exit reason distribution."""
    from collections import Counter
    exit_counts = Counter([t.exit_reason for t in trades])
    total = len(trades)

    return {
        reason: {
            'count': count,
            'pct': count / total * 100 if total > 0 else 0
        }
        for reason, count in exit_counts.items()
    }

def main():
    parser = argparse.ArgumentParser(description='Phase 1 backtest validation')
    parser.add_argument('--asset', type=str, required=True, choices=['BTC', 'ETH', 'SPY'],
                        help='Asset to backtest')
    args = parser.parse_args()

    asset = args.asset

    print(f"\n{'='*70}")
    print(f"Phase 1: Structure Invalidation Exit - {asset} Backtest")
    print(f"{'='*70}\n")

    # Load Phase 1 feature store
    feature_path = Path(f'data/features_mtf/{asset}_1H_2024-01-01_to_2024-12-31.parquet')
    print(f"📂 Loading feature store: {feature_path}")

    if not feature_path.exists():
        print(f"❌ Feature store not found: {feature_path}")
        print(f"   Run: python3 bin/build_mtf_feature_store.py --asset {asset} --start 2024-01-01 --end 2024-12-31")
        return 1

    df = pd.read_parquet(feature_path)
    print(f"   Loaded {len(df)} bars")

    # Verify Phase 1 columns exist
    required_cols = ['tf1h_ob_low', 'tf1h_ob_high', 'tf1h_bb_low', 'tf1h_bb_high',
                     'tf1h_fvg_low', 'tf1h_fvg_high', 'tf1h_fvg_present',
                     'tf1h_bos_bearish', 'tf1h_bos_bullish']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing Phase 1 columns: {missing_cols}")
        print(f"   Feature store needs to be rebuilt with Phase 1 SMC columns")
        return 1

    print(f"✅ Phase 1 SMC columns present")

    # Load config
    print(f"\n⚙️  Loading config...")
    config = load_config(asset)
    params = config_to_params(config)
    print(f"   Loaded {asset} frozen config")

    # Run backtest
    print(f"\n🚀 Running Phase 1 backtest...")
    backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
    results = backtest.run()

    # Analyze results
    print(f"\n{'='*70}")
    print(f"PHASE 1 RESULTS")
    print(f"{'='*70}\n")

    print(f"Total Trades:    {results['total_trades']}")
    print(f"Total PNL:       ${results['total_pnl']:,.2f}")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Profit Factor:   {results['profit_factor']:.2f}")
    print(f"Max Drawdown:    {results['max_drawdown']*100:.2f}%")
    print(f"Final Equity:    ${results['final_equity']:,.2f}")
    print(f"Total Return:    {(results['final_equity']/10000 - 1)*100:+.2f}%")

    # Exit reason breakdown
    print(f"\n{'='*70}")
    print(f"EXIT REASON BREAKDOWN")
    print(f"{'='*70}\n")

    exit_analysis = analyze_exit_reasons(results['trades'])
    for reason, stats in sorted(exit_analysis.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"{reason:25s} {stats['count']:3d} ({stats['pct']:5.1f}%)")

    # Compare to baseline
    baseline = BASELINES.get(asset, {})
    if baseline:
        print(f"\n{'='*70}")
        print(f"COMPARISON TO BASELINE")
        print(f"{'='*70}\n")

        print(f"{'Metric':<20s} {'Baseline':>15s} {'Phase 1':>15s} {'Change':>15s}")
        print(f"{'-'*70}")

        # Total PNL
        baseline_pnl = baseline.get('total_pnl')
        if baseline_pnl:
            pnl_change = (results['total_pnl'] - baseline_pnl) / baseline_pnl * 100
            print(f"{'Total PNL':<20s} ${baseline_pnl:>14,.2f} ${results['total_pnl']:>14,.2f} {pnl_change:>+14.1f}%")

        # Trade count
        baseline_trades = baseline.get('total_trades')
        if baseline_trades:
            trade_change = (results['total_trades'] - baseline_trades) / baseline_trades * 100
            print(f"{'Trade Count':<20s} {baseline_trades:>15d} {results['total_trades']:>15d} {trade_change:>+14.1f}%")

        # Win rate
        baseline_wr = baseline.get('win_rate')
        if baseline_wr:
            wr_change = (results['win_rate'] - baseline_wr) / baseline_wr * 100
            print(f"{'Win Rate':<20s} {baseline_wr*100:>14.1f}% {results['win_rate']*100:>14.1f}% {wr_change:>+14.1f}%")

        # Drawdown
        baseline_dd = baseline.get('max_drawdown')
        if baseline_dd is not None:
            dd_reduction = baseline_dd - results['max_drawdown']
            print(f"{'Max Drawdown':<20s} {baseline_dd*100:>14.2f}% {results['max_drawdown']*100:>14.2f}% {dd_reduction*100:>+14.2f}%")

    # Phase 1 acceptance gates
    print(f"\n{'='*70}")
    print(f"PHASE 1 ACCEPTANCE GATES")
    print(f"{'='*70}\n")

    print("Target Metrics:")
    print("  - Drawdown reduction: 5-10%")
    print("  - PNL change: maintain or improve (±5% acceptable)")
    print("  - Structure invalidation exits: >0 (new exit reason firing)")
    print()

    # Check structure invalidation exits
    structure_exits = [t for t in results['trades'] if t.exit_reason == 'structure_invalidated']
    structure_exit_pct = len(structure_exits) / results['total_trades'] * 100 if results['total_trades'] > 0 else 0

    print(f"Structure Invalidation Exits: {len(structure_exits)} ({structure_exit_pct:.1f}%)")

    # Compute acceptance
    baseline_pnl = baseline.get('total_pnl', results['total_pnl'])
    pnl_change_pct = (results['total_pnl'] - baseline_pnl) / baseline_pnl * 100 if baseline_pnl > 0 else 0

    baseline_dd = baseline.get('max_drawdown', 0.0)
    dd_reduction_pct = (baseline_dd - results['max_drawdown']) * 100

    gates_passed = []
    gates_failed = []

    if len(structure_exits) > 0:
        gates_passed.append("✅ Structure exits firing")
    else:
        gates_failed.append("❌ No structure exits (logic may not be triggering)")

    if pnl_change_pct >= -5:
        gates_passed.append(f"✅ PNL maintained ({pnl_change_pct:+.1f}%)")
    else:
        gates_failed.append(f"❌ PNL declined significantly ({pnl_change_pct:+.1f}%)")

    if dd_reduction_pct >= 5:
        gates_passed.append(f"✅ Drawdown reduced ({dd_reduction_pct:+.1f}%)")
    elif dd_reduction_pct >= 0:
        gates_passed.append(f"⚠️  Drawdown slightly reduced ({dd_reduction_pct:+.1f}%)")
    else:
        gates_failed.append(f"❌ Drawdown increased ({dd_reduction_pct:+.1f}%)")

    print()
    for gate in gates_passed:
        print(gate)
    for gate in gates_failed:
        print(gate)

    print()
    if len(gates_failed) == 0:
        print("🎉 PHASE 1 ACCEPTANCE GATES PASSED")
        print()
        print("Next steps:")
        print("  1. Run Phase 1 backtest on other assets (BTC, ETH, SPY)")
        print("  2. Commit Phase 1 changes with message: 'feat(exits): Phase 1 structure invalidation exit'")
        print("  3. Proceed to Phase 2: Pattern-Triggered Exits")
        return 0
    else:
        print("⚠️  PHASE 1 NEEDS TUNING")
        print()
        print("Tuning options:")
        print("  1. Adjust RSI momentum threshold (currently 40/60)")
        print("  2. Require 2/3 structure breaks instead of any 1")
        print("  3. Add minimum hold time (don't exit via structure in first 3 bars)")
        print("  4. Review structure invalidation logic for false positives")
        return 1

if __name__ == '__main__':
    sys.exit(main())
