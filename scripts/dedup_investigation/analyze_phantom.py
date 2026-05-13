"""
Phase 1.4 — phantom outcomes for dedup losers.

For each (winner, loser) pair in dedup_events.csv, compute what would have
happened to the LOSER if it had been taken instead of the winner.

Approach:
1. Group losers by archetype.
2. For each loser signal, look up the next bar's price and compute outcome
   based on a synthetic stop+TP rule:
      - Long: TP at 1.5R above entry (where R = |entry - SL|), stop at SL.
      - Short: symmetric.
3. Walk forward up to N=240 bars (10 days) or until SL/TP hit.
4. Report phantom PF = sum(wins) / |sum(losses)| per archetype.

Caveats:
- This ignores cooling periods, position-size constraints, and the fact that
  taking the loser would have OTHER downstream knock-on effects.
- 1.5R TP is a simplifying choice; the real exit logic is far richer.
- Result is an upper bound on "what is at stake" when dedup blocks an archetype,
  not a true counterfactual PnL.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PARQUET = ROOT / 'data' / 'features_mtf' / 'BTC_1H_LATEST.parquet'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events-csv', required=True,
                    help='Path to dedup_events.csv from Phase 1 diagnostic')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--parquet', default=str(DEFAULT_PARQUET))
    ap.add_argument('--tp-r-multiple', type=float, default=1.5,
                    help='Take-profit at this multiple of entry-to-SL distance')
    ap.add_argument('--max-bars', type=int, default=240,
                    help='Time-stop horizon in bars')
    return ap.parse_args()


def simulate_outcome(df, ts, entry, sl, direction, tp_r, max_bars):
    """Walk forward from ts and return (pnl_pct, exit_reason, bars_held)."""
    if entry <= 0 or sl <= 0:
        return None
    R = abs(entry - sl)
    if R == 0:
        return None
    if direction == 'long':
        tp = entry + tp_r * R
    elif direction == 'short':
        tp = entry - tp_r * R
    else:
        return None

    # Find next bar.
    try:
        loc = df.index.get_indexer([pd.Timestamp(ts)], method='bfill')[0]
    except Exception:
        return None
    if loc < 0 or loc + 1 >= len(df):
        return None
    # Skip the signal bar — assume entry on next bar's open.
    start = loc + 1
    end = min(start + max_bars, len(df))
    sub = df.iloc[start:end]
    if len(sub) == 0:
        return None

    fill_price = float(sub['open'].iloc[0]) if 'open' in sub.columns else entry

    for i in range(len(sub)):
        bar = sub.iloc[i]
        high = float(bar['high']) if 'high' in sub.columns else float(bar.get('close', entry))
        low = float(bar['low']) if 'low' in sub.columns else float(bar.get('close', entry))
        if direction == 'long':
            if low <= sl:
                pnl_pct = (sl - fill_price) / fill_price
                return pnl_pct, 'stop', i + 1
            if high >= tp:
                pnl_pct = (tp - fill_price) / fill_price
                return pnl_pct, 'tp', i + 1
        else:
            if high >= sl:
                pnl_pct = (fill_price - sl) / fill_price
                return pnl_pct, 'stop', i + 1
            if low <= tp:
                pnl_pct = (fill_price - tp) / fill_price
                return pnl_pct, 'tp', i + 1

    # Time exit at last bar close.
    last_close = float(sub['close'].iloc[-1]) if 'close' in sub.columns else fill_price
    if direction == 'long':
        pnl_pct = (last_close - fill_price) / fill_price
    else:
        pnl_pct = (fill_price - last_close) / fill_price
    return pnl_pct, 'time', len(sub)


def main():
    args = parse_args()
    events_csv = Path(args.events_csv)
    df = pd.read_parquet(args.parquet)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    by_arch = defaultdict(list)
    n_skipped = 0

    with events_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row['timestamp'].split('+')[0]  # strip tz suffix
            loser_id = row['loser_id']
            entry = float(row['loser_entry'])
            sl = float(row['loser_sl'])
            direction = row['loser_dir']
            res = simulate_outcome(
                df, ts, entry, sl, direction,
                args.tp_r_multiple, args.max_bars,
            )
            if res is None:
                n_skipped += 1
                continue
            pnl_pct, reason, bars = res
            by_arch[loser_id].append({
                'pnl_pct': pnl_pct,
                'reason': reason,
                'bars': bars,
            })

    out = {}
    for arch, trades in by_arch.items():
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        gross_win = sum(t['pnl_pct'] for t in wins)
        gross_loss = abs(sum(t['pnl_pct'] for t in losses))
        pf = (gross_win / gross_loss) if gross_loss > 1e-9 else float('inf')
        out[arch] = {
            'n_phantom_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / max(len(trades), 1),
            'gross_win_pct': round(gross_win * 100, 3),
            'gross_loss_pct': round(gross_loss * 100, 3),
            'net_pct': round((gross_win - gross_loss) * 100, 3),
            'phantom_pf': round(pf, 3) if pf != float('inf') else 'inf',
            'mean_pnl_pct_bp': round((gross_win - gross_loss) / max(len(trades), 1) * 10000, 1),
            'reasons': {
                'stop': sum(1 for t in trades if t['reason'] == 'stop'),
                'tp':   sum(1 for t in trades if t['reason'] == 'tp'),
                'time': sum(1 for t in trades if t['reason'] == 'time'),
            },
        }

    output = {
        'meta': {
            'events_csv': str(events_csv),
            'tp_r_multiple': args.tp_r_multiple,
            'max_bars': args.max_bars,
            'n_skipped': n_skipped,
        },
        'by_archetype': out,
        # Aggregate across all losers.
        'aggregate': {
            'n_phantom_trades': sum(v['n_phantom_trades'] for v in out.values()),
            'sum_net_pct': sum(v['net_pct'] for v in out.values()),
        },
    }
    Path(args.out_json).write_text(json.dumps(output, indent=2))

    # Pretty-print to stdout.
    print(f'\n=== Phantom outcomes (TP={args.tp_r_multiple}R, max_bars={args.max_bars}) ===')
    print(f'{"archetype":<22} {"n":>4} {"win%":>5} {"net%":>7} {"PF":>6} {"bp/trd":>7}')
    rows = sorted(out.items(), key=lambda x: -x[1]['n_phantom_trades'])
    for arch, v in rows:
        print(f'{arch:<22} {v["n_phantom_trades"]:>4} '
              f'{v["win_rate"]*100:>5.1f} {v["net_pct"]:>7.2f} '
              f'{str(v["phantom_pf"]):>6} {v["mean_pnl_pct_bp"]:>7.1f}')
    print(f'\nSkipped {n_skipped} events (missing bars or invalid entry/SL).')


if __name__ == '__main__':
    main()
