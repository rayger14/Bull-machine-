#!/usr/bin/env python3
"""Analyze S1 V2 validation backtest results."""

import re
from datetime import datetime
from collections import defaultdict

# Major capitulation events to check
MAJOR_EVENTS = {
    'LUNA May-12': ('2022-05-12', '2022-05-13'),
    'June 18 Bottom': ('2022-06-18', '2022-06-20'),
    'FTX Collapse': ('2022-11-09', '2022-11-10'),
    'Japan Carry Unwind': ('2024-08-05', '2024-08-06'),
}

def parse_entry(line):
    """Parse ENTRY line."""
    # ENTRY archetype_breakdown: 2022-01-21 12:00:00+00:00 @ $38184.23, size=$2956.71, fusion=0.690
    match = re.search(r'ENTRY archetype_(\w+): ([\d-]+ [\d:]+)\+00:00 @ \$([\d.]+), size=\$([\d.]+), fusion=([\d.]+)', line)
    if match:
        return {
            'archetype': match.group(1),
            'timestamp': match.group(2),
            'price': float(match.group(3)),
            'size': float(match.group(4)),
            'fusion': float(match.group(5)),
        }
    return None

def parse_exit(line):
    """Parse EXIT line."""
    # EXIT archetype_breakdown @ $38184.23, size=$2956.71, fusion=0.690, PNL=$123.45
    match = re.search(r'EXIT archetype_(\w+).*PNL=\$([-\d.]+)', line)
    if match:
        return {
            'archetype': match.group(1),
            'pnl': float(match.group(2)),
        }
    return None

def analyze_log(log_path):
    """Analyze backtest log file."""
    entries = []
    exits = []

    with open(log_path, 'r') as f:
        for line in f:
            if 'ENTRY archetype_' in line:
                entry = parse_entry(line)
                if entry:
                    entries.append(entry)
            elif 'EXIT archetype_' in line:
                exit_trade = parse_exit(line)
                if exit_trade:
                    exits.append(exit_trade)

    # Calculate metrics
    total_trades = len(entries)
    total_pnl = sum(e['pnl'] for e in exits)
    winning_trades = [e for e in exits if e['pnl'] > 0]
    losing_trades = [e for e in exits if e['pnl'] <= 0]

    win_rate = (len(winning_trades) / len(exits) * 100) if exits else 0
    avg_win = sum(e['pnl'] for e in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(e['pnl'] for e in losing_trades) / len(losing_trades) if losing_trades else 0

    # Check major events
    events_caught = {}
    for event_name, (start_date, end_date) in MAJOR_EVENTS.items():
        entries_in_window = [e for e in entries if start_date <= e['timestamp'][:10] <= end_date]
        events_caught[event_name] = len(entries_in_window) > 0
        if entries_in_window:
            events_caught[f"{event_name}_count"] = len(entries_in_window)
            events_caught[f"{event_name}_first"] = entries_in_window[0]['timestamp']

    # Period analysis (2022, 2023, 2024)
    trades_by_year = defaultdict(int)
    for entry in entries:
        year = entry['timestamp'][:4]
        trades_by_year[year] += 1

    return {
        'total_trades': total_trades,
        'trades_per_year': total_trades / 3.0,  # 2022-2024 = 3 years
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'events_caught': events_caught,
        'trades_by_year': dict(trades_by_year),
    }

if __name__ == '__main__':
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else 'results/liquidity_vacuum_calibration/s1_v2_quick_fix_validation_FINAL.log'

    results = analyze_log(log_path)

    print("="*80)
    print("S1 V2 QUICK FIX VALIDATION RESULTS")
    print("="*80)
    print(f"\nTOTAL TRADES: {results['total_trades']}")
    print(f"TRADES/YEAR: {results['trades_per_year']:.1f}")
    print(f"  2022: {results['trades_by_year'].get('2022', 0)}")
    print(f"  2023: {results['trades_by_year'].get('2023', 0)}")
    print(f"  2024: {results['trades_by_year'].get('2024', 0)}")

    print(f"\nPERFORMANCE:")
    print(f"  Total PNL: ${results['total_pnl']:.2f}")
    print(f"  Win Rate: {results['win_rate']:.1f}%")
    print(f"  Avg Win: ${results['avg_win']:.2f}")
    print(f"  Avg Loss: ${results['avg_loss']:.2f}")

    print(f"\nMAJOR EVENTS DETECTED:")
    for event_name, (start, end) in MAJOR_EVENTS.items():
        caught = results['events_caught'].get(event_name, False)
        status = "✓ CAUGHT" if caught else "✗ MISSED"
        print(f"  {event_name:25s} {status}", end="")
        if caught:
            count = results['events_caught'].get(f"{event_name}_count", 0)
            first = results['events_caught'].get(f"{event_name}_first", "")
            print(f"  ({count} trades, first @ {first})")
        else:
            print()

    events_caught_count = sum(1 for v in [results['events_caught'].get(name) for name in MAJOR_EVENTS.keys()] if v)
    print(f"\n  TOTAL: {events_caught_count}/{len(MAJOR_EVENTS)} major events ({events_caught_count/len(MAJOR_EVENTS)*100:.0f}%)")

    print("\n" + "="*80)
    print("VERDICT:")
    print("="*80)

    target_trades = (50, 80)
    if target_trades[0] <= results['trades_per_year'] <= target_trades[1]:
        print(f"✓ Trade frequency: {results['trades_per_year']:.1f}/year IN TARGET RANGE ({target_trades[0]}-{target_trades[1]})")
    else:
        print(f"✗ Trade frequency: {results['trades_per_year']:.1f}/year OUTSIDE TARGET ({target_trades[0]}-{target_trades[1]})")

    if events_caught_count >= 3:
        print(f"✓ Event detection: {events_caught_count}/4 ACCEPTABLE (≥3 required)")
    else:
        print(f"✗ Event detection: {events_caught_count}/4 INSUFFICIENT (<3)")

    if results['win_rate'] >= 50:
        print(f"✓ Win rate: {results['win_rate']:.1f}% GOOD (≥50% target)")
    else:
        print(f"~ Win rate: {results['win_rate']:.1f}% NEEDS IMPROVEMENT (<50%)")

    print("\n" + "="*80)
