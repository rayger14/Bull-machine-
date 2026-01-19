#!/usr/bin/env python3
"""Quick analysis of S4 trades from backtest log."""

import re

log_file = "results/s4_baseline_backtest.log"

# Parse trades
s4_trades = []
baseline_trades = []

with open(log_file, 'r') as f:
    content = f.read()

# Find all S4 trades
s4_pattern = r'Trade \d+: archetype_funding_divergence.*?PNL: \$(-?\d+\.\d+) \((-?\d+\.\d+)%\)'
for match in re.finditer(s4_pattern, content, re.DOTALL):
    pnl_dollars = float(match.group(1))
    pnl_pct = float(match.group(2))
    s4_trades.append({'pnl_dollars': pnl_dollars, 'pnl_pct': pnl_pct})

# Find all baseline trades
baseline_pattern = r'Trade \d+: tier1_market.*?PNL: \$(-?\d+\.\d+) \((-?\d+\.\d+)%\)'
for match in re.finditer(baseline_pattern, content, re.DOTALL):
    pnl_dollars = float(match.group(1))
    pnl_pct = float(match.group(2))
    baseline_trades.append({'pnl_dollars': pnl_dollars, 'pnl_pct': pnl_pct})

# Calculate S4 metrics
s4_count = len(s4_trades)
s4_wins = sum(1 for t in s4_trades if t['pnl_pct'] > 0)
s4_losses = sum(1 for t in s4_trades if t['pnl_pct'] < 0)
s4_wr = (s4_wins / s4_count * 100) if s4_count > 0 else 0

s4_total_profit = sum(t['pnl_dollars'] for t in s4_trades if t['pnl_dollars'] > 0)
s4_total_loss = abs(sum(t['pnl_dollars'] for t in s4_trades if t['pnl_dollars'] < 0))
s4_pf = (s4_total_profit / s4_total_loss) if s4_total_loss > 0 else 0

s4_net_pnl = sum(t['pnl_dollars'] for t in s4_trades)

print("=" * 70)
print("S4 (FUNDING DIVERGENCE) PERFORMANCE - 2022")
print("=" * 70)
print(f"Trade Count:     {s4_count} trades/year")
print(f"Win Rate:        {s4_wr:.1f}% ({s4_wins}W / {s4_losses}L)")
print(f"Profit Factor:   {s4_pf:.2f}")
print(f"Gross Profit:    ${s4_total_profit:.2f}")
print(f"Gross Loss:      ${s4_total_loss:.2f}")
print(f"Net PNL:         ${s4_net_pnl:.2f}")
print()
print(f"TARGET: 6-10 trades/year, PF > 2.0")
print(f"STATUS: {'✓ PASS' if 6 <= s4_count <= 30 and s4_pf >= 1.5 else '✗ FAIL'}")
print()
print("BASELINE TRADES (SHOULD BE 0):")
print(f"Count: {len(baseline_trades)} (PROBLEM: baseline trades leaking through)")
print("=" * 70)

# Show top S4 winners
print("\nTop 5 S4 Winners:")
s4_sorted = sorted(s4_trades, key=lambda x: x['pnl_pct'], reverse=True)
for i, trade in enumerate(s4_sorted[:5], 1):
    print(f"  {i}. PNL: ${trade['pnl_dollars']:.2f} ({trade['pnl_pct']:.2f}%)")

print("\nTop 5 S4 Losers:")
for i, trade in enumerate(reversed(s4_sorted[-5:]), 1):
    print(f"  {i}. PNL: ${trade['pnl_dollars']:.2f} ({trade['pnl_pct']:.2f}%)")
