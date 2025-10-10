#!/usr/bin/env python3
"""
Fast 7-Day Entry Parity Test - Bull Machine v1.8.4

Quick validation that batch mode produces same entry signals as full mode.
Target: ≤3 minutes runtime for fast iteration during development.

Usage:
    python3 tests/parity/test_entry_parity_week.py [--asset ETH] [--date 2025-07-01]
"""

import subprocess
import json
import pandas as pd
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_trades(path: str, event_filter: str = None) -> list:
    """Load trades from JSONL, optionally filtering by event type."""
    trades = []
    try:
        with open(path, 'r') as f:
            for line in f:
                trade = json.loads(line)
                if event_filter is None or trade.get('event') == event_filter:
                    trades.append(trade)
    except FileNotFoundError:
        print(f"⚠️  Warning: {path} not found")
        return []
    return trades


def test_week_parity(asset: str = 'ETH', start_date: str = '2025-07-01',
                     config: str = 'configs/v18/ETH_comprehensive.json'):
    """
    Test entry parity on a single week (7 days).

    Args:
        asset: Asset symbol (BTC, ETH, SOL)
        start_date: Start date (YYYY-MM-DD)
        config: Config file path

    Returns:
        True if parity ≥95%, False otherwise
    """
    # Calculate end date (7 days later)
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = start + timedelta(days=7)
    end_date = end.strftime('%Y-%m-%d')

    print("=" * 70)
    print("🧪 Fast 7-Day Entry Parity Test")
    print("=" * 70)
    print(f"Asset:  {asset}")
    print(f"Period: {start_date} → {end_date} (7 days)")
    print(f"Config: {config}")
    print(f"Target: ≤3 minutes, ≥95% parity")
    print("=" * 70)

    start_time = datetime.now()

    # 1. Run batch screener
    print("\n🔍 Step 1/3: Batch screener...")
    try:
        subprocess.run([
            'python3', 'bin/research/batch_screener.py',
            '--asset', asset,
            '--start', start_date,
            '--end', end_date,
            '--config', config,
            '--output', 'results/test_week_candidates.jsonl'
        ], check=True, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("❌ Batch screener timeout (>60s)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch screener failed: {e}")
        return False

    # Count candidates
    try:
        with open('results/test_week_candidates.jsonl', 'r') as f:
            candidate_count = sum(1 for _ in f)
        print(f"✅ Generated {candidate_count} candidates")
    except FileNotFoundError:
        print("❌ Candidates file not created")
        return False

    # 2. Run full replay
    print("\n📊 Step 2/3: Full replay...")
    try:
        subprocess.run([
            'python3', 'bin/live/hybrid_runner.py',
            '--asset', asset,
            '--start', start_date,
            '--end', end_date,
            '--config', config
        ], check=True, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        print("❌ Full replay timeout (>90s)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Full replay failed: {e}")
        return False

    # Rename trade log
    subprocess.run(['mv', 'results/trade_log.jsonl', 'results/trade_log_full_week.jsonl'], check=False)
    full_trades = load_trades('results/trade_log_full_week.jsonl', event_filter='open')
    print(f"✅ Full replay: {len(full_trades)} entry signals")

    # 3. Run batch replay
    print("\n🎯 Step 3/3: Batch replay...")
    try:
        subprocess.run([
            'python3', 'bin/live/hybrid_runner.py',
            '--asset', asset,
            '--start', start_date,
            '--end', end_date,
            '--config', config,
            '--candidates', 'results/test_week_candidates.jsonl'
        ], check=True, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        print("❌ Batch replay timeout (>90s)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch replay failed: {e}")
        return False

    batch_trades = load_trades('results/trade_log.jsonl', event_filter='open')
    print(f"✅ Batch replay: {len(batch_trades)} entry signals")

    # 4. Compare with ±1 bar tolerance
    print("\n🔬 Comparing entry parity...")

    # Build batch lookup
    batch_lookup = {}
    for t in batch_trades:
        ts = pd.to_datetime(t.get('ts'))
        batch_lookup[ts] = t

    matched = 0
    missed_in_batch = []

    for ft in full_trades:
        ft_ts = pd.to_datetime(ft.get('ts'))
        ft_side = ft.get('side')
        ft_price = ft.get('entry', 0)

        # Try exact match, then ±1 bar
        found = False
        for offset_hours in [0, -1, 1]:
            check_ts = ft_ts + pd.Timedelta(hours=offset_hours)
            if check_ts in batch_lookup:
                bt = batch_lookup[check_ts]
                bt_side = bt.get('side')
                bt_price = bt.get('entry', 0)

                if bt_side == ft_side and abs(bt_price - ft_price) < 0.01:
                    matched += 1
                    found = True
                    break

        if not found:
            missed_in_batch.append(ft)

    # Calculate parity
    parity_pct = (matched / len(full_trades) * 100) if len(full_trades) > 0 else 0

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ Matched:        {matched} / {len(full_trades)} ({parity_pct:.1f}%)")
    print(f"⚠️  Missed:         {len(missed_in_batch)}")
    print(f"⏱️  Total time:     {elapsed:.1f}s")

    # Pass/fail
    print("\n" + "=" * 70)
    if parity_pct >= 95 and elapsed <= 180:
        print("✅ FAST PARITY TEST PASSED")
        print(f"   Entry parity:   {parity_pct:.1f}% ≥ 95%")
        print(f"   Runtime:        {elapsed:.1f}s ≤ 180s")
        print("=" * 70)
        return True
    else:
        print("❌ FAST PARITY TEST FAILED")
        if parity_pct < 95:
            print(f"   Entry parity:   {parity_pct:.1f}% < 95%")
        if elapsed > 180:
            print(f"   Runtime:        {elapsed:.1f}s > 180s")

        # Show missed entries
        if missed_in_batch and len(missed_in_batch) <= 10:
            print(f"\n❌ Missed entries in batch mode:")
            for i, t in enumerate(missed_in_batch):
                print(f"   {i+1}. {t.get('ts')} {t.get('side')} @ {t.get('entry')}")

        print("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(description='Fast 7-day entry parity test')
    parser.add_argument('--asset', default='ETH', help='Asset symbol (default: ETH)')
    parser.add_argument('--date', default='2025-07-01', help='Start date (default: 2025-07-01)')
    parser.add_argument('--config', default='configs/v18/ETH_comprehensive.json',
                       help='Config file (default: configs/v18/ETH_comprehensive.json)')

    args = parser.parse_args()

    success = test_week_parity(args.asset, args.date, args.config)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
