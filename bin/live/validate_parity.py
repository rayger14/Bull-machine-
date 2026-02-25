#!/usr/bin/env python3
"""
Signal Parity Validator for Bull Machine

Compares signals from the backtester vs shadow runner to verify
that the live pipeline produces identical trading decisions.

Usage:
    # Run both engines on Q1 2024 and compare:
    python3 bin/live/validate_parity.py --period 2024-01-01:2024-03-31

    # Compare existing trade logs:
    python3 bin/live/validate_parity.py --compare-logs \
        --backtest-log results/v11_standalone/trade_log.csv \
        --shadow-log results/live_signals/signals.csv
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def load_backtest_trades(path: Path) -> pd.DataFrame:
    """Load backtester trade log CSV."""
    df = pd.read_csv(path, parse_dates=['timestamp', 'exit_timestamp'])
    # Keep only entry events (each trade row is one exit event, but we can
    # reconstruct entries by grouping on timestamp+archetype)
    return df


def load_shadow_signals(path: Path) -> pd.DataFrame:
    """Load shadow runner signal log CSV."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df


def extract_entries(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique entries from backtest trade log."""
    entries = backtest_df.groupby(['timestamp', 'archetype', 'direction']).first().reset_index()
    entries = entries[['timestamp', 'archetype', 'direction', 'entry_price',
                       'entry_regime', 'fusion_score']].copy()
    entries = entries.rename(columns={'entry_regime': 'regime'})
    entries = entries.drop_duplicates(subset=['timestamp', 'archetype', 'direction'])
    entries = entries.sort_values('timestamp').reset_index(drop=True)
    return entries


def extract_shadow_entries(shadow_df: pd.DataFrame) -> pd.DataFrame:
    """Extract entries from shadow runner signal log."""
    entries = shadow_df[shadow_df['action'] == 'ENTRY'].copy()
    entries = entries[['timestamp', 'archetype', 'direction', 'entry_price',
                       'regime', 'fusion_score']].copy()
    entries = entries.drop_duplicates(subset=['timestamp', 'archetype', 'direction'])
    entries = entries.sort_values('timestamp').reset_index(drop=True)
    return entries


def compare_entries(bt_entries: pd.DataFrame, sh_entries: pd.DataFrame) -> dict:
    """
    Compare backtester entries vs shadow runner entries.

    Returns dict with match statistics.
    """
    results = {
        'backtest_entries': len(bt_entries),
        'shadow_entries': len(sh_entries),
        'matched': 0,
        'backtest_only': 0,
        'shadow_only': 0,
        'fusion_diffs': [],
        'price_diffs': [],
        'mismatches': [],
    }

    # Create keys for matching: (timestamp, archetype, direction)
    bt_keys = set()
    bt_lookup = {}
    for _, row in bt_entries.iterrows():
        key = (str(row['timestamp']), row['archetype'], row['direction'])
        bt_keys.add(key)
        bt_lookup[key] = row

    sh_keys = set()
    sh_lookup = {}
    for _, row in sh_entries.iterrows():
        key = (str(row['timestamp']), row['archetype'], row['direction'])
        sh_keys.add(key)
        sh_lookup[key] = row

    # Matched entries
    matched_keys = bt_keys & sh_keys
    results['matched'] = len(matched_keys)
    results['backtest_only'] = len(bt_keys - sh_keys)
    results['shadow_only'] = len(sh_keys - bt_keys)

    # Compare fusion scores and prices for matched entries
    for key in matched_keys:
        bt_row = bt_lookup[key]
        sh_row = sh_lookup[key]

        fusion_diff = abs(bt_row['fusion_score'] - sh_row['fusion_score'])
        results['fusion_diffs'].append(fusion_diff)

        # Price comparison (shadow stores raw signal price, backtester stores fill price)
        # So prices will differ by slippage amount — that's expected
        price_diff_pct = abs(bt_row['entry_price'] - sh_row['entry_price']) / bt_row['entry_price'] * 100
        results['price_diffs'].append(price_diff_pct)

    # Log mismatches (entries in one but not the other)
    for key in (bt_keys - sh_keys):
        results['mismatches'].append({
            'source': 'backtest_only',
            'timestamp': key[0],
            'archetype': key[1],
            'direction': key[2],
        })
    for key in (sh_keys - bt_keys):
        results['mismatches'].append({
            'source': 'shadow_only',
            'timestamp': key[0],
            'archetype': key[1],
            'direction': key[2],
        })

    return results


def compare_exit_sequences(backtest_df: pd.DataFrame, shadow_df: pd.DataFrame) -> dict:
    """Compare exit reasons and sequences between the two logs."""
    bt_exits = backtest_df[['timestamp', 'archetype', 'exit_reason', 'exit_timestamp']].copy()
    bt_exits = bt_exits.sort_values('exit_timestamp')

    sh_exits = shadow_df[shadow_df['action'] == 'EXIT'][['timestamp', 'archetype', 'reason']].copy()
    sh_exits = sh_exits.rename(columns={'reason': 'exit_reason', 'timestamp': 'exit_timestamp'})
    sh_exits = sh_exits.sort_values('exit_timestamp')

    # Count exit reasons
    bt_reason_counts = bt_exits['exit_reason'].value_counts().to_dict()
    sh_reason_counts = sh_exits['exit_reason'].value_counts().to_dict()

    return {
        'backtest_exits': len(bt_exits),
        'shadow_exits': len(sh_exits),
        'backtest_reason_counts': bt_reason_counts,
        'shadow_reason_counts': sh_reason_counts,
    }


def print_report(entry_results: dict, exit_results: dict):
    """Print formatted parity report."""
    print("\n" + "=" * 72)
    print("SIGNAL PARITY VALIDATION REPORT")
    print("=" * 72)

    # Entry matching
    total = max(entry_results['backtest_entries'], entry_results['shadow_entries'])
    match_rate = entry_results['matched'] / total * 100 if total > 0 else 0

    print(f"\n{'ENTRY MATCHING':^72}")
    print("-" * 72)
    print(f"  Backtest entries:  {entry_results['backtest_entries']}")
    print(f"  Shadow entries:    {entry_results['shadow_entries']}")
    print(f"  Matched:           {entry_results['matched']}")
    print(f"  Backtest-only:     {entry_results['backtest_only']}")
    print(f"  Shadow-only:       {entry_results['shadow_only']}")
    print(f"  Match Rate:        {match_rate:.1f}%")

    passed = match_rate >= 95.0
    status = "PASS" if passed else "FAIL"
    print(f"\n  >>> ENTRY PARITY: {status} (target: >= 95%) <<<")

    # Fusion score comparison
    if entry_results['fusion_diffs']:
        diffs = np.array(entry_results['fusion_diffs'])
        print(f"\n{'FUSION SCORE COMPARISON (matched entries)':^72}")
        print("-" * 72)
        print(f"  Mean diff:    {np.mean(diffs):.6f}")
        print(f"  Max diff:     {np.max(diffs):.6f}")
        print(f"  Exact match:  {np.sum(diffs < 0.001)}/{len(diffs)} ({np.mean(diffs < 0.001)*100:.1f}%)")

    # Price comparison
    if entry_results['price_diffs']:
        pdiffs = np.array(entry_results['price_diffs'])
        print(f"\n{'PRICE COMPARISON (matched entries)':^72}")
        print("-" * 72)
        print(f"  Mean diff:    {np.mean(pdiffs):.4f}%")
        print(f"  Max diff:     {np.max(pdiffs):.4f}%")
        print(f"  Note: Differences expected from slippage modeling")

    # Exit comparison
    print(f"\n{'EXIT COMPARISON':^72}")
    print("-" * 72)
    print(f"  Backtest exits:  {exit_results['backtest_exits']}")
    print(f"  Shadow exits:    {exit_results['shadow_exits']}")

    print(f"\n  Backtest exit reasons:")
    for reason, count in sorted(exit_results['backtest_reason_counts'].items()):
        print(f"    {reason:30s}: {count}")

    print(f"\n  Shadow exit reasons:")
    for reason, count in sorted(exit_results['shadow_reason_counts'].items()):
        print(f"    {reason:30s}: {count}")

    # Mismatches detail
    if entry_results['mismatches']:
        print(f"\n{'MISMATCHED ENTRIES (first 10)':^72}")
        print("-" * 72)
        for m in entry_results['mismatches'][:10]:
            print(f"  [{m['source']:15s}] {m['timestamp']} {m['archetype']} {m['direction']}")

    print("\n" + "=" * 72)
    print(f"OVERALL VERDICT: {'PASS - Signal parity confirmed' if passed else 'FAIL - Signals diverge'}")
    print("=" * 72)

    return passed


def run_both_engines(start_date: str, end_date: str, commission: float, slippage: float):
    """Run both backtester and shadow runner on the same period."""
    print(f"Running both engines on {start_date} to {end_date}...")

    # Run backtester
    print("\n[1/2] Running backtester...")
    bt_cmd = (
        f"python3 {PROJECT_ROOT}/bin/backtest_v11_standalone.py "
        f"--start-date {start_date} --end-date {end_date} "
        f"--commission-rate {commission} --slippage-bps {slippage}"
    )
    result = subprocess.run(bt_cmd, shell=True, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"Backtester failed:\n{result.stderr[-500:]}")
        return False

    # Run shadow runner
    print("[2/2] Running shadow runner replay...")
    sh_cmd = (
        f"python3 {PROJECT_ROOT}/bin/live/v11_shadow_runner.py --replay "
        f"--start-date {start_date} --end-date {end_date} "
        f"--commission-rate {commission} --slippage-bps {slippage}"
    )
    result = subprocess.run(sh_cmd, shell=True, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"Shadow runner failed:\n{result.stderr[-500:]}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Validate signal parity between backtester and shadow runner')
    parser.add_argument('--period', type=str, default='2024-01-01:2024-03-31',
                        help='Date range as START:END (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--compare-logs', action='store_true',
                        help='Compare existing logs without re-running engines')
    parser.add_argument('--backtest-log', type=str,
                        default='results/v11_standalone/trade_log.csv',
                        help='Path to backtester trade log')
    parser.add_argument('--shadow-log', type=str,
                        default='results/live_signals/signals.csv',
                        help='Path to shadow runner signal log')
    parser.add_argument('--commission-rate', type=float, default=0.0004)
    parser.add_argument('--slippage-bps', type=float, default=5.0)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse period
    parts = args.period.split(':')
    start_date = parts[0]
    end_date = parts[1] if len(parts) > 1 else None

    # Run both engines if not just comparing
    if not args.compare_logs:
        success = run_both_engines(start_date, end_date,
                                   args.commission_rate, args.slippage_bps)
        if not success:
            print("Failed to run engines. Use --compare-logs with existing outputs.")
            sys.exit(1)

    # Load logs
    bt_path = PROJECT_ROOT / args.backtest_log
    sh_path = PROJECT_ROOT / args.shadow_log

    if not bt_path.exists():
        print(f"Backtest log not found: {bt_path}")
        sys.exit(1)
    if not sh_path.exists():
        print(f"Shadow log not found: {sh_path}")
        sys.exit(1)

    bt_df = load_backtest_trades(bt_path)
    sh_df = load_shadow_signals(sh_path)

    # Extract and compare entries
    bt_entries = extract_entries(bt_df)
    sh_entries = extract_shadow_entries(sh_df)

    entry_results = compare_entries(bt_entries, sh_entries)
    exit_results = compare_exit_sequences(bt_df, sh_df)

    # Print report
    passed = print_report(entry_results, exit_results)

    # Save report
    report_path = PROJECT_ROOT / 'results' / 'live_signals' / 'parity_report.json'
    report = {
        'period': args.period,
        'entry_match_rate': entry_results['matched'] / max(1, max(entry_results['backtest_entries'], entry_results['shadow_entries'])) * 100,
        'backtest_entries': entry_results['backtest_entries'],
        'shadow_entries': entry_results['shadow_entries'],
        'matched': entry_results['matched'],
        'passed': passed,
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
