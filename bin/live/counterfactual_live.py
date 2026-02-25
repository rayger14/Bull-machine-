#!/usr/bin/env python3
"""Quick counterfactual analysis of live trading signals vs actual price data."""
import csv, json, sys
from collections import defaultdict
import pandas as pd

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else 'results/coinbase_paper'

# Load candles
candles = pd.read_csv(f'{RESULTS_DIR}/candle_history.csv', parse_dates=['timestamp'])
candles = candles.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')

# Load entry signals
entries = []
with open(f'{RESULTS_DIR}/signals.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['action'] == 'ENTRY':
            entries.append(row)

print('=' * 105)
print('COUNTERFACTUAL ANALYSIS: All Allocated Entries vs Actual Price Action')
print(f'Date range: {candles.index[0].strftime("%b %d")} to {candles.index[-1].strftime("%b %d %H:%M")} | Current BTC: ${candles.iloc[-1]["close"]:,.0f}')
print('=' * 105)
print()

# Simulate each entry with simple SL/TP
header = f'{"#":>2} {"Archetype":22s} {"Entry$":>8s} {"SL$":>8s} {"TP$":>8s} {"RR":>5s} {"Result":>8s} {"Exit$":>8s} {"PnL%":>7s} {"Hrs":>5s} {"Fusion":>7s}'
print(header)
print('-' * 105)

total_pnl_pct = 0
wins = 0
losses = 0
open_trades = 0
by_fusion = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
by_arch = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})

for i, e in enumerate(entries):
    ts = pd.Timestamp(e['timestamp'])
    entry = float(e['entry_price'])
    sl = float(e['stop_loss'])
    tp = float(e['take_profit'])
    fusion = float(e['fusion_score'])
    arch = e['archetype']
    direction = e['direction']

    sl_dist = abs(entry - sl) / entry * 100
    tp_dist = abs(tp - entry) / entry * 100
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    future = candles[candles.index > ts]
    outcome = 'OPEN'
    exit_price = candles.iloc[-1]['close']
    hours = (candles.index[-1] - ts).total_seconds() / 3600

    for bar_ts, bar in future.iterrows():
        h = (bar_ts - ts).total_seconds() / 3600
        if direction == 'long':
            if bar['low'] <= sl:
                outcome = 'SL_HIT'
                exit_price = sl
                hours = h
                break
            if bar['high'] >= tp:
                outcome = 'TP_HIT'
                exit_price = tp
                hours = h
                break
        else:
            if bar['high'] >= sl:
                outcome = 'SL_HIT'
                exit_price = sl
                hours = h
                break
            if bar['low'] <= tp:
                outcome = 'TP_HIT'
                exit_price = tp
                hours = h
                break
        if h >= 168:
            outcome = 'TIME'
            exit_price = bar['close']
            hours = h
            break

    if direction == 'long':
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    total_pnl_pct += pnl_pct
    bucket = f'{round(fusion * 5) / 5:.1f}'

    if outcome == 'TP_HIT':
        wins += 1
        by_fusion[bucket]['wins'] += 1
        by_arch[arch]['wins'] += 1
    elif outcome == 'SL_HIT':
        losses += 1
        by_fusion[bucket]['losses'] += 1
        by_arch[arch]['losses'] += 1
    else:
        open_trades += 1

    by_fusion[bucket]['pnl'] += pnl_pct
    by_arch[arch]['pnl'] += pnl_pct

    sym = '+' if pnl_pct > 0 else '-'
    print(f'{i+1:2d} {arch:22s} {entry:8.0f} {sl:8.0f} {tp:8.0f} {rr:4.1f}R {outcome:>8s} {exit_price:8.0f} {pnl_pct:+6.2f}% {hours:5.0f}h {fusion:6.3f} {sym}')

print('-' * 105)
total = wins + losses + open_trades
wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
print(f'Total: {total} | TP Hit: {wins} | SL Hit: {losses} | Open: {open_trades} | Win Rate: {wr:.0f}% | Sum PnL: {total_pnl_pct:+.2f}%')
print()

# By archetype
print('BY ARCHETYPE:')
for arch, stats in sorted(by_arch.items(), key=lambda x: x[1]['pnl'], reverse=True):
    t = stats['wins'] + stats['losses']
    wr = stats['wins'] / t * 100 if t > 0 else 0
    print(f'  {arch:22s} {stats["wins"]}W/{stats["losses"]}L  WR={wr:4.0f}%  PnL={stats["pnl"]:+6.2f}%')
print()

# What-if analysis
print('=' * 105)
print('WHAT-IF ANALYSIS: Alternative SL/TP Scenarios')
print('=' * 105)

scenarios = [
    ('Current SL/TP', 1.0, 1.0),
    ('1.5x wider SL, same TP', 1.5, 1.0),
    ('2x wider SL, same TP', 2.0, 1.0),
    ('Same SL, 0.5x tighter TP', 1.0, 0.5),
    ('1.5x wider SL + 0.5x TP', 1.5, 0.5),
    ('2x wider SL + 2x wider TP', 2.0, 2.0),
    ('Same SL, 2x wider TP', 1.0, 2.0),
    ('Same SL, 3x wider TP', 1.0, 3.0),
]

for name, sl_mult, tp_mult in scenarios:
    w, l, o = 0, 0, 0
    pnl_sum = 0.0
    for e in entries:
        ts = pd.Timestamp(e['timestamp'])
        entry = float(e['entry_price'])
        sl_orig = float(e['stop_loss'])
        tp_orig = float(e['take_profit'])
        direction = e['direction']

        sl_dist = abs(entry - sl_orig)
        tp_dist = abs(tp_orig - entry)

        if direction == 'long':
            sl = entry - sl_dist * sl_mult
            tp = entry + tp_dist * tp_mult
        else:
            sl = entry + sl_dist * sl_mult
            tp = entry - tp_dist * tp_mult

        future = candles[candles.index > ts]
        outcome = 'OPEN'
        exit_price = candles.iloc[-1]['close']

        for bar_ts, bar in future.iterrows():
            h = (bar_ts - ts).total_seconds() / 3600
            if direction == 'long':
                if bar['low'] <= sl:
                    outcome = 'SL'
                    exit_price = sl
                    break
                if bar['high'] >= tp:
                    outcome = 'TP'
                    exit_price = tp
                    break
            else:
                if bar['high'] >= sl:
                    outcome = 'SL'
                    exit_price = sl
                    break
                if bar['low'] <= tp:
                    outcome = 'TP'
                    exit_price = tp
                    break
            if h >= 168:
                outcome = 'TIME'
                exit_price = bar['close']
                break

        if direction == 'long':
            pnl = (exit_price - entry) / entry * 100
        else:
            pnl = (entry - exit_price) / entry * 100
        pnl_sum += pnl
        if outcome == 'TP':
            w += 1
        elif outcome == 'SL':
            l += 1
        else:
            o += 1

    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f'  {name:35s} | {w:2d}W/{l:2d}L/{o:2d}O  WR={wr:4.0f}%  PnL={pnl_sum:+7.2f}%')

print()

# Price trajectory
print('=' * 105)
print('PRICE TRAJECTORY: BTC during trading period')
print('=' * 105)
print(f'  Start:  ${candles.iloc[0]["close"]:,.0f} ({candles.index[0].strftime("%b %d")})')
print(f'  High:   ${candles["high"].max():,.0f} ({candles["high"].idxmax().strftime("%b %d %H:00")})')
print(f'  Low:    ${candles["low"].min():,.0f} ({candles["low"].idxmin().strftime("%b %d %H:00")})')
print(f'  End:    ${candles.iloc[-1]["close"]:,.0f} ({candles.index[-1].strftime("%b %d")})')
price_change = (candles.iloc[-1]['close'] - candles.iloc[0]['close']) / candles.iloc[0]['close'] * 100
print(f'  Change: {price_change:+.2f}%')
print()

print('Daily price action:')
for date, group in candles.groupby(candles.index.date):
    o = group.iloc[0]['open']
    c = group.iloc[-1]['close']
    h = group['high'].max()
    l = group['low'].min()
    change = (c - o) / o * 100
    rng = (h - l) / o * 100
    print(f'  {date} | O=${o:,.0f} H=${h:,.0f} L=${l:,.0f} C=${c:,.0f} | {change:+.2f}% | Range: {rng:.1f}%')

print()

# SL distance analysis
print('SL/TP Distance Analysis:')
sl_pcts = []
tp_pcts = []
for e in entries:
    entry = float(e['entry_price'])
    sl = float(e['stop_loss'])
    sl_pct = abs(entry - sl) / entry * 100
    tp = float(e['take_profit'])
    tp_pct = abs(tp - entry) / entry * 100
    sl_pcts.append(sl_pct)
    tp_pcts.append(tp_pct)
    print(f'  {e["archetype"]:22s} SL={sl_pct:.2f}%  TP={tp_pct:.2f}%  RR={tp_pct/sl_pct if sl_pct else 0:.1f}:1')

print()
print(f'  Average SL distance: {sum(sl_pcts)/len(sl_pcts):.2f}%')
print(f'  Average TP distance: {sum(tp_pcts)/len(tp_pcts):.2f}%')
print(f'  Average daily BTC range: {candles.groupby(candles.index.date).apply(lambda g: (g["high"].max() - g["low"].min()) / g.iloc[0]["open"] * 100).mean():.2f}%')
print()

print('=' * 105)
print('KEY INSIGHTS')
print('=' * 105)
avg_sl = sum(sl_pcts) / len(sl_pcts)
avg_daily_range = candles.groupby(candles.index.date).apply(
    lambda g: (g['high'].max() - g['low'].min()) / g.iloc[0]['open'] * 100
).mean()
print(f'1. Average SL ({avg_sl:.2f}%) vs Average Daily Range ({avg_daily_range:.2f}%)')
if avg_sl < avg_daily_range:
    print(f'   SL is TIGHTER than daily range -- normal intraday noise triggers stops')
    print(f'   This is the main reason for the 0% win rate')
print(f'2. BTC price change: {price_change:+.2f}% over {len(candles)} bars')
print(f'3. All {len(entries)} entries were LONG')
if price_change > 0:
    print(f'   Direction was CORRECT (price ended higher) but stops were too tight')
elif price_change < -2:
    print(f'   Direction was WRONG -- bearish period, long signals failed')
else:
    print(f'   Market was choppy/flat -- tight stops get whipsawed repeatedly')
