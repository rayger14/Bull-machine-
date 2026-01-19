#!/usr/bin/env python3
"""
Domain Engine Wiring Verification Test

Compares Core vs Full variants for S1, S4, S5 to verify domain engines are working.

Expected Results (AFTER backfill):
- S1 Full > S1 Core (domain engines boost performance)
- S4 Full > S4 Core (domain engines boost performance)
- S5 Full > S5 Core (domain engines improve precision)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

# Import backtest engine
from engine.backtest import Backtester
from engine.execution import Executor
from engine.config import Config

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_single_backtest(config_path: str, start_date: str = "2022-01-01", end_date: str = "2022-12-31") -> Dict:
    """Run a single backtest and return metrics."""

    print(f"Running: {Path(config_path).name}")

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    config = Config(config_dict)

    # Load feature store
    feature_store_path = Path("data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")
    df = pd.read_parquet(feature_store_path)

    # Filter to date range
    df = df[start_date:end_date]

    print(f"  Data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    # Run backtest
    executor = Executor(config)
    backtester = Backtester(executor, config)

    portfolio, trades = backtester.run(df)

    # Calculate metrics
    metrics = {
        'config': Path(config_path).stem,
        'total_trades': len(trades),
        'final_equity': portfolio['equity'].iloc[-1] if len(portfolio) > 0 else 10000,
        'total_return_pct': ((portfolio['equity'].iloc[-1] / 10000) - 1) * 100 if len(portfolio) > 0 else 0,
        'trades': trades
    }

    # Calculate PF (profit factor)
    if len(trades) > 0:
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]

        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 1

        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
        metrics['win_rate'] = len(wins) / len(trades) if len(trades) > 0 else 0
        metrics['avg_win'] = total_wins / len(wins) if wins else 0
        metrics['avg_loss'] = total_losses / len(losses) if losses else 0
    else:
        metrics['profit_factor'] = 0
        metrics['win_rate'] = 0
        metrics['avg_win'] = 0
        metrics['avg_loss'] = 0

    print(f"  ✅ Trades: {metrics['total_trades']}, PF: {metrics['profit_factor']:.2f}, WR: {metrics['win_rate']:.1%}")

    return metrics


def compare_variants(archetype: str, core_config: str, full_config: str, test_year: str = "2022") -> None:
    """Compare Core vs Full variant for an archetype."""

    print(f"\n{'=' * 80}")
    print(f"{archetype.upper()} COMPARISON - {test_year}")
    print(f"{'=' * 80}")

    start_date = f"{test_year}-01-01"
    end_date = f"{test_year}-12-31"

    # Run Core variant
    print(f"\n{archetype} CORE (Wyckoff only):")
    core_metrics = run_single_backtest(core_config, start_date, end_date)

    # Run Full variant
    print(f"\n{archetype} FULL (All domain engines):")
    full_metrics = run_single_backtest(full_config, start_date, end_date)

    # Compare
    print(f"\n{'─' * 80}")
    print(f"RESULTS:")
    print(f"{'─' * 80}")

    print(f"\n{'Metric':<30} {'Core':>15} {'Full':>15} {'Δ':>15}")
    print(f"{'-' * 80}")

    print(f"{'Trades':<30} {core_metrics['total_trades']:>15} {full_metrics['total_trades']:>15} {full_metrics['total_trades'] - core_metrics['total_trades']:>15}")

    pf_delta = full_metrics['profit_factor'] - core_metrics['profit_factor']
    pf_pct = (pf_delta / core_metrics['profit_factor'] * 100) if core_metrics['profit_factor'] > 0 else 0
    print(f"{'Profit Factor':<30} {core_metrics['profit_factor']:>15.2f} {full_metrics['profit_factor']:>15.2f} {pf_pct:>14.1f}%")

    wr_delta = full_metrics['win_rate'] - core_metrics['win_rate']
    print(f"{'Win Rate':<30} {core_metrics['win_rate']:>14.1%} {full_metrics['win_rate']:>14.1%} {wr_delta:>14.1%}")

    ret_delta = full_metrics['total_return_pct'] - core_metrics['total_return_pct']
    print(f"{'Total Return %':<30} {core_metrics['total_return_pct']:>14.1f}% {full_metrics['total_return_pct']:>14.1f}% {ret_delta:>14.1f}%")

    # Verdict
    print(f"\n{'─' * 80}")
    if full_metrics['profit_factor'] > core_metrics['profit_factor']:
        print(f"✅ PASS: Full variant outperforms Core (PF improvement: +{pf_pct:.1f}%)")
        print(f"   Domain engines are WORKING as intended")
    elif abs(pf_delta) < 0.1:
        print(f"⚠️  NEUTRAL: Full and Core performance similar (PF delta: {pf_delta:.2f})")
        print(f"   Domain engines have minimal impact (features may be weak)")
    else:
        print(f"❌ FAIL: Core outperforms Full (PF degradation: {pf_pct:.1f}%)")
        print(f"   Domain engines are DEGRADING performance")

    return {
        'archetype': archetype,
        'core': core_metrics,
        'full': full_metrics,
        'pf_improvement': pf_pct,
        'verdict': 'PASS' if full_metrics['profit_factor'] > core_metrics['profit_factor'] else 'NEUTRAL' if abs(pf_delta) < 0.1 else 'FAIL'
    }


def main():
    print("=" * 80)
    print("DOMAIN ENGINE WIRING VERIFICATION TEST")
    print("=" * 80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Feature Store: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")
    print(f"Test Period: 2022 (Bear Market)")
    print("=" * 80)

    results = []

    # S1 Comparison
    s1_result = compare_variants(
        archetype="S1",
        core_config="configs/variants/s1_core.json",
        full_config="configs/variants/s1_full.json",
        test_year="2022"
    )
    results.append(s1_result)

    # S4 Comparison
    s4_result = compare_variants(
        archetype="S4",
        core_config="configs/variants/s4_core.json",
        full_config="configs/variants/s4_full.json",
        test_year="2022"
    )
    results.append(s4_result)

    # S5 Comparison
    s5_result = compare_variants(
        archetype="S5",
        core_config="configs/variants/s5_core.json",
        full_config="configs/variants/s5_full.json",
        test_year="2022"
    )
    results.append(s5_result)

    # Summary
    print(f"\n\n{'=' * 80}")
    print("DOMAIN ENGINE VERIFICATION SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Archetype':<15} {'Core PF':>12} {'Full PF':>12} {'Improvement':>15} {'Status':>15}")
    print(f"{'-' * 80}")

    for r in results:
        status_icon = '✅' if r['verdict'] == 'PASS' else '⚠️' if r['verdict'] == 'NEUTRAL' else '❌'
        print(f"{r['archetype']:<15} {r['core']['profit_factor']:>12.2f} {r['full']['profit_factor']:>12.2f} {r['pf_improvement']:>14.1f}% {status_icon + ' ' + r['verdict']:>15}")

    # Overall verdict
    passed = sum(1 for r in results if r['verdict'] == 'PASS')

    print(f"\n{'=' * 80}")
    if passed == len(results):
        print("✅ OVERALL: ALL TESTS PASSED - Domain wiring is working correctly!")
    elif passed > 0:
        print(f"⚠️  OVERALL: PARTIAL SUCCESS - {passed}/{len(results)} tests passed")
    else:
        print("❌ OVERALL: ALL TESTS FAILED - Domain wiring not working as expected")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
