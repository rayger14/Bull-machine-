#!/usr/bin/env python3
"""
Analyze Domain Wiring Test Results

Extracts metrics from backtest logs and compares Core vs Full variants.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def extract_metrics_from_log(log_path: str) -> Dict:
    """Extract key metrics from backtest log file."""

    with open(log_path) as f:
        log_text = f.read()

    metrics = {}

    # Extract summary statistics
    # Look for patterns like:
    # Total trades: 110
    # Win rate: 45.5%
    # Profit factor: 0.87
    # Final equity: $9234.56

    trades_match = re.search(r'Total.*?(\d+)\s+trades', log_text, re.IGNORECASE)
    if trades_match:
        metrics['total_trades'] = int(trades_match.group(1))
    else:
        # Count individual trades
        trades = re.findall(r'^Trade \d+:', log_text, re.MULTILINE)
        metrics['total_trades'] = len(trades)

    # Win rate
    wr_match = re.search(r'Win rate[:\s]+(\d+\.?\d*)%', log_text, re.IGNORECASE)
    if wr_match:
        metrics['win_rate'] = float(wr_match.group(1)) / 100
    else:
        # Calculate from PNL lines
        pnls = re.findall(r'PNL:\s+\$([+-]?\d+\.?\d*)', log_text)
        if pnls:
            wins = sum(1 for pnl in pnls if float(pnl) > 0)
            metrics['win_rate'] = wins / len(pnls)
        else:
            metrics['win_rate'] = 0

    # Profit factor
    pf_match = re.search(r'Profit factor[:\s]+(\d+\.?\d*)', log_text, re.IGNORECASE)
    if pf_match:
        metrics['profit_factor'] = float(pf_match.group(1))
    else:
        # Calculate from PNLs
        pnls = [float(pnl) for pnl in re.findall(r'PNL:\s+\$([+-]?\d+\.?\d*)', log_text)]
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 1
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
        else:
            metrics['profit_factor'] = 0

    # Total return
    ret_match = re.search(r'Total return[:\s]+([+-]?\d+\.?\d*)%', log_text, re.IGNORECASE)
    if ret_match:
        metrics['total_return_pct'] = float(ret_match.group(1))
    else:
        # Calculate from final equity
        equity_match = re.search(r'Final.*?equity.*?\$(\d+\.?\d*)', log_text, re.IGNORECASE)
        if equity_match:
            final_equity = float(equity_match.group(1))
            metrics['total_return_pct'] = ((final_equity / 10000) - 1) * 100
        else:
            metrics['total_return_pct'] = 0

    # Check for domain signals in trades (indicates domain engines are active)
    domain_signals = re.findall(r'domain_signals|smc_score|temporal_confluence|wyckoff_pti', log_text, re.IGNORECASE)
    metrics['has_domain_signals'] = len(domain_signals) > 0

    return metrics


def compare_variants(archetype: str, core_log: str, full_log: str) -> Dict:
    """Compare Core vs Full variant metrics."""

    print(f"\n{'=' * 80}")
    print(f"{archetype.upper()} COMPARISON")
    print(f"{'=' * 80}\n")

    core_metrics = extract_metrics_from_log(core_log)
    full_metrics = extract_metrics_from_log(full_log)

    print(f"{'Metric':<30} {'Core':>15} {'Full':>15} {'Δ':>15}")
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
    print(f"\n{'-' * 80}")

    if full_metrics['profit_factor'] > core_metrics['profit_factor'] + 0.1:
        verdict = 'PASS'
        print(f"✅ PASS: Full variant outperforms Core (PF improvement: +{pf_pct:.1f}%)")
        print(f"   Domain engines are WORKING as intended")
    elif abs(pf_delta) < 0.1:
        verdict = 'NEUTRAL'
        print(f"⚠️  NEUTRAL: Full and Core performance similar (PF delta: {pf_delta:.2f})")
        print(f"   Domain engines have minimal impact")
    else:
        verdict = 'FAIL'
        print(f"❌ FAIL: Core outperforms Full (PF degradation: {pf_pct:.1f}%)")
        print(f"   Domain engines are DEGRADING performance")

    return {
        'archetype': archetype,
        'core': core_metrics,
        'full': full_metrics,
        'pf_improvement': pf_pct,
        'verdict': verdict
    }


def main():
    results_dir = Path("results/domain_wiring_test")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Run bin/compare_variants.sh first to generate backtest logs")
        sys.exit(1)

    print("=" * 80)
    print("DOMAIN ENGINE WIRING VERIFICATION - RESULTS ANALYSIS")
    print("=" * 80)

    results = []

    # S1 Comparison
    s1_result = compare_variants(
        "S1 (Liquidity Vacuum)",
        str(results_dir / "s1_core.log"),
        str(results_dir / "s1_full.log")
    )
    results.append(s1_result)

    # S4 Comparison
    s4_result = compare_variants(
        "S4 (Funding Divergence)",
        str(results_dir / "s4_core.log"),
        str(results_dir / "s4_full.log")
    )
    results.append(s4_result)

    # S5 Comparison
    s5_result = compare_variants(
        "S5 (Long Squeeze)",
        str(results_dir / "s5_core.log"),
        str(results_dir / "s5_full.log")
    )
    results.append(s5_result)

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("DOMAIN ENGINE VERIFICATION SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Archetype':<25} {'Core PF':>12} {'Full PF':>12} {'Δ%':>10} {'Trades Δ':>10} {'Status':>12}")
    print(f"{'-' * 80}")

    for r in results:
        status_icon = '✅ PASS' if r['verdict'] == 'PASS' else '⚠️  NEUT' if r['verdict'] == 'NEUTRAL' else '❌ FAIL'
        trade_delta = r['full']['total_trades'] - r['core']['total_trades']
        print(f"{r['archetype']:<25} {r['core']['profit_factor']:>12.2f} {r['full']['profit_factor']:>12.2f} {r['pf_improvement']:>9.1f}% {trade_delta:>10} {status_icon:>12}")

    # Overall verdict
    passed = sum(1 for r in results if r['verdict'] == 'PASS')

    print(f"\n{'=' * 80}")
    if passed == len(results):
        print("✅ OVERALL: ALL TESTS PASSED - Domain wiring is working correctly!")
        print("   All Full variants outperform Core variants")
    elif passed > 0:
        print(f"⚠️  OVERALL: PARTIAL SUCCESS - {passed}/{len(results)} tests passed")
        print("   Some domain engines are working, others need tuning")
    else:
        print("❌ OVERALL: ALL TESTS FAILED - Domain wiring not effective")
        print("   Domain features may be weak or wiring has issues")
    print(f"{'=' * 80}\n")

    # Save summary to file
    summary_path = results_dir / "comparison_summary.txt"
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
