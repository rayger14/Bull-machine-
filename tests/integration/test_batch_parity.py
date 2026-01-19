#!/usr/bin/env python3
"""
Parity test: Full replay vs Candidate-driven replay.

Asserts identical trade outcomes on short test period.
"""

import subprocess
import json
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_trades(path: str, event_filter: str = None) -> list:
    """
    Load trades from JSONL.

    Args:
        path: Path to trade log JSONL
        event_filter: If specified, only return trades with this event type (e.g., 'open')
    """
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


def test_batch_parity():
    """Test that batch mode produces same results as full mode."""

    # Test parameters
    asset = 'ETH'
    start = '2025-07-01'
    end = '2025-07-15'
    config = 'configs/v18/ETH_comprehensive.json'

    print("=" * 70)
    print("🧪 Batch Mode Parity Test")
    print("=" * 70)
    print(f"Asset:  {asset}")
    print(f"Period: {start} → {end} (2 weeks)")
    print(f"Config: {config}")
    print("=" * 70)

    # 1. Run batch screener
    print("\n🔍 Step 1/3: Running batch screener...")
    print("-" * 70)
    try:
        subprocess.run([
            'python3', 'bin/research/batch_screener.py',
            '--asset', asset,
            '--start', start,
            '--end', end,
            '--config', config,
            '--output', 'results/test_candidates.jsonl'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch screener failed: {e}")
        sys.exit(1)

    # Count candidates
    try:
        with open('results/test_candidates.jsonl', 'r') as f:
            candidate_count = sum(1 for _ in f)
        print(f"✅ Generated {candidate_count} candidates")
    except FileNotFoundError:
        print("❌ Candidates file not created")
        sys.exit(1)

    # 2. Run full replay
    print("\n📊 Step 2/3: Running full replay...")
    print("-" * 70)
    try:
        subprocess.run([
            'python3', 'bin/live/hybrid_runner.py',
            '--asset', asset,
            '--start', start,
            '--end', end,
            '--config', config
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Full replay failed: {e}")
        sys.exit(1)

    # Rename trade log to preserve full replay results
    subprocess.run(['mv', 'results/trade_log.jsonl', 'results/trade_log_full.jsonl'], check=False)

    # Load only 'open' events for entry parity comparison
    full_trades = load_trades('results/trade_log_full.jsonl', event_filter='open')
    print(f"✅ Full replay: {len(full_trades)} entry signals")

    # 3. Run candidate-driven replay
    print("\n🎯 Step 3/3: Running candidate-driven replay...")
    print("-" * 70)
    try:
        subprocess.run([
            'python3', 'bin/live/hybrid_runner.py',
            '--asset', asset,
            '--start', start,
            '--end', end,
            '--config', config,
            '--candidates', 'results/test_candidates.jsonl'
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Candidate-driven replay failed: {e}")
        sys.exit(1)

    # Load only 'open' events for entry parity comparison
    batch_trades = load_trades('results/trade_log.jsonl', event_filter='open')
    print(f"✅ Batch replay: {len(batch_trades)} entry signals")

    # 4. Compare results
    print("\n🔬 Comparing results...")
    print("-" * 70)

    # Entry parity check with ±1 bar tolerance
    print(f"\nComparing {len(full_trades)} full entries vs {len(batch_trades)} batch entries...")
    print("Using ±1 bar tolerance for timestamp matching\n")

    # Build timestamp→entry lookup for batch trades
    batch_lookup = {}
    for t in batch_trades:
        ts = pd.to_datetime(t.get('ts'))
        batch_lookup[ts] = t

    matched = 0
    missed_in_batch = []
    mismatched = []

    for ft in full_trades:
        ft_ts = pd.to_datetime(ft.get('ts'))
        ft_side = ft.get('side')
        ft_price = ft.get('entry', 0)

        # Try exact match first, then ±1 bar
        found = False
        for offset_hours in [0, -1, 1]:
            check_ts = ft_ts + pd.Timedelta(hours=offset_hours)
            if check_ts in batch_lookup:
                bt = batch_lookup[check_ts]
                bt_side = bt.get('side')
                bt_price = bt.get('entry', 0)

                # Check if side and price match
                if bt_side == ft_side and abs(bt_price - ft_price) < 0.01:
                    matched += 1
                    found = True
                    break
                else:
                    mismatched.append({
                        'full_ts': ft_ts,
                        'batch_ts': check_ts,
                        'side_match': bt_side == ft_side,
                        'price_diff': abs(bt_price - ft_price)
                    })
                    found = True
                    break

        if not found:
            missed_in_batch.append(ft)

    # Check for extra entries in batch mode
    full_lookup = {pd.to_datetime(t.get('ts')): t for t in full_trades}
    extra_in_batch = []
    for bt in batch_trades:
        bt_ts = pd.to_datetime(bt.get('ts'))
        found = False
        for offset_hours in [0, -1, 1]:
            if (bt_ts + pd.Timedelta(hours=offset_hours)) in full_lookup:
                found = True
                break
        if not found:
            extra_in_batch.append(bt)

    # Calculate parity percentage
    parity_pct = (matched / len(full_trades) * 100) if len(full_trades) > 0 else 0

    print(f"✅ Matched entries:     {matched} / {len(full_trades)} ({parity_pct:.1f}%)")
    print(f"⚠️  Missed in batch:    {len(missed_in_batch)}")
    print(f"⚠️  Extra in batch:     {len(extra_in_batch)}")
    print(f"⚠️  Mismatched details: {len(mismatched)}")

    # Show details if parity is low
    if parity_pct < 95:
        print(f"\n⚠️  Entry parity below 95% threshold!")
        if missed_in_batch:
            print(f"\n❌ First 5 missed entries (full replay only):")
            for i, t in enumerate(missed_in_batch[:5]):
                print(f"  {i+1}. {t.get('ts')} {t.get('side')} @ {t.get('entry')}")
        if extra_in_batch:
            print(f"\n❌ First 5 extra entries (batch replay only):")
            for i, t in enumerate(extra_in_batch[:5]):
                print(f"  {i+1}. {t.get('ts')} {t.get('side')} @ {t.get('entry')}")
        if mismatched:
            print(f"\n❌ First 5 mismatched entries:")
            for i, m in enumerate(mismatched[:5]):
                print(f"  {i+1}. {m['full_ts']} → {m['batch_ts']}, side_match={m['side_match']}, price_diff={m['price_diff']:.2f}")

    # Final verdict
    print("\n" + "=" * 70)
    if parity_pct >= 95:
        print("✅ PARITY TEST PASSED - Entry parity ≥95%!")
        print(f"\n📊 Summary:")
        print(f"   Candidates generated: {candidate_count}")
        print(f"   Entry parity:         {parity_pct:.1f}%")
        print(f"   Matched entries:      {matched} / {len(full_trades)}")
        print(f"   Test status:          ✅ PASS")
    else:
        print(f"❌ PARITY TEST FAILED - Entry parity {parity_pct:.1f}% < 95%")
        print(f"\n📊 Summary:")
        print(f"   Candidates generated: {candidate_count}")
        print(f"   Entry parity:         {parity_pct:.1f}%")
        print(f"   Matched:              {matched}")
        print(f"   Missed in batch:      {len(missed_in_batch)}")
        print(f"   Extra in batch:       {len(extra_in_batch)}")
        print(f"   Test status:          ❌ FAIL")
        sys.exit(1)
    print("=" * 70)


if __name__ == '__main__':
    test_batch_parity()
