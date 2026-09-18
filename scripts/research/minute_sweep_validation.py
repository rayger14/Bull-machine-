#!/usr/bin/env python3
"""Research-only, causal equal-low sweep replay. No optimizer or live connectivity.

Candidates use only their own confirmed history through reclaim. Shared cooldown
is applied AFTER sorting by observable reclaim time, never by pivot discovery
order. This intentionally changes the legacy noncausal event population.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_bars(bars):
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError('Timezone-aware DatetimeIndex required')
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError('Minute timestamps must be unique and sorted')
    if bars.index.hasnans or np.any(bars.index.asi8 % pd.Timedelta(minutes=1).value):
        raise ValueError('Minute timestamps must align exactly to the UTC minute grid')
    if len(bars) > 1 and not np.all(np.diff(bars.index.asi8) == pd.Timedelta(minutes=1).value):
        raise ValueError('Missing or non-minute bars; do not silently fill gaps')
    try:
        prices = bars[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Numeric OHLC required') from exc
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Finite positive OHLC required')
    if len(prices) and ((prices[:, 2] > prices[:, [0, 3]].min(axis=1)).any()
                        or (prices[:, 1] < prices[:, [0, 3]].max(axis=1)).any()):
        raise ValueError('Invalid OHLC envelope')


def detect_events(bars):
    validate_bars(bars)
    low, close = bars.low.to_numpy(), bars.close.to_numpy()
    # A pivot at i becomes observable at i+15, not at i.
    pivots = np.flatnonzero((bars.low.rolling(31, center=True).min() == bars.low).to_numpy())
    candidates = []
    for k in range(1, len(pivots)):
        i = int(pivots[k]); level = float(low[i])
        j = bisect.bisect_left(pivots, i-1440, 0, k)
        prior = pivots[j:k]
        matching = (abs(low[prior]-level)/level <= .001) & (prior <= i-30)
        if not matching.any():
            continue
        # Include the last available bar: appending data must not retroactively
        # make an already-observable final-bar sweep eligible.
        breaks = np.flatnonzero(low[i+15:min(len(bars), i+1440)] < level*.9998)
        if not len(breaks):
            continue
        sweep = i+15+int(breaks[0])
        reclaims = np.flatnonzero(close[sweep:min(sweep+31, len(bars))] > level)
        if not len(reclaims):
            continue
        reclaim = sweep+int(reclaims[0])
        candidates.append(dict(pivot_idx=i, confirmed_idx=i+15, sweep_idx=sweep,
                               reclaim_idx=reclaim, level=level,
                               sweep_low=float(low[sweep:reclaim+1].min()),
                               touches=int(matching.sum()+1)))
    accepted, last_sweep = [], -10**9
    for event in sorted(candidates, key=lambda e: (e['reclaim_idx'], e['pivot_idx'])):
        if event['sweep_idx']-last_sweep < 60:
            continue
        accepted.append(event)
        last_sweep = event['sweep_idx']
    return accepted


def simulate_events(bars, events, *, entry_mode='next_open', notional=50000.,
                    stop_buffer=.0015, hold_minutes=240, cost_bps=12.,
                    decision_delay_seconds=0):
    """One-position, fixed-notional diagnostic; no funded margin model.

    Lockout is fixed from entry, not released by an early stop (legacy policy).
    Half the fixed-notional round-trip fee is charged at each side. Next-open
    includes the entry bar and exits at the deadline OPEN, before another entry
    can use that open. Close-mode exits at deadline close. Stop gaps fill at
    min(stop, bar open). Nonzero decision delay is next-open only: availability
    is reclaim candle close, readiness adds the requested seconds, and entry
    samples the first minute open at or after readiness. Events and sweep-derived
    stops stay frozen; hold/lockout begin at the actual delayed entry.
    """
    validate_bars(bars)
    if entry_mode not in ('close', 'next_open'):
        raise ValueError('Unknown entry mode')
    if (isinstance(decision_delay_seconds, (bool, np.bool_))
            or not isinstance(decision_delay_seconds, (int, np.integer))
            or decision_delay_seconds < 0):
        raise ValueError('Decision delay must be finite nonnegative integer seconds')
    decision_delay_seconds = int(decision_delay_seconds)
    if decision_delay_seconds and entry_mode != 'next_open':
        raise ValueError('Nonzero decision delay requires next_open entry mode')
    try:
        delay = pd.Timedelta(seconds=decision_delay_seconds)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Decision delay exceeds the representable timestamp range') from exc
    sampled_delay_seconds = ((decision_delay_seconds+59)//60)*60
    timing = dict(requested_delay_seconds=decision_delay_seconds,
                  sampled_delay_seconds=sampled_delay_seconds,
                  rounding_delay_seconds=sampled_delay_seconds-decision_delay_seconds,
                  policy='ceil(reclaim candle close + requested delay, UTC minute) for next_open')
    if not np.isfinite([notional, stop_buffer, cost_bps]).all() or notional <= 0 or not 0 <= stop_buffer < 1 or cost_bps < 0:
        raise ValueError('Invalid sizing, stop or cost')
    if not isinstance(hold_minutes, int) or hold_minutes <= 0:
        raise ValueError('Positive integer hold required')
    reclaim_indices = [e['reclaim_idx'] for e in events]
    if any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(bars) for i in reclaim_indices):
        raise ValueError('Event index outside data')
    if any(a >= b for a, b in zip(reclaim_indices, reclaim_indices[1:])):
        raise ValueError('Events must have unique sorted reclaim indices')
    for e in events:
        if not np.isfinite(e['sweep_low']) or e['sweep_low'] <= 0:
            raise ValueError('Invalid sweep low')
        if not 0 <= e['pivot_idx'] <= e['sweep_idx'] <= e['reclaim_idx']:
            raise ValueError('Event chronology invalid')
    op, low, close = [bars[k].to_numpy() for k in ('open', 'low', 'close')]
    marks = np.full(len(bars), np.nan)
    trades, opened, ledger = [], [], []
    busy, realized, skipped_busy, skipped_invalid, unfilled = -1, 0., 0, 0, 0
    fee = notional*cost_bps/20000.
    for e in events:
        reclaim_time = bars.index[e['reclaim_idx']].tz_convert('UTC')
        pivot_time = bars.index[e['pivot_idx']].tz_convert('UTC')
        signal_available = reclaim_time + pd.Timedelta(minutes=1)
        order_ready = signal_available + delay
        eligible_open = order_ready.ceil('min') if entry_mode == 'next_open' else None
        # Exact grid and continuity were validated above. No nearest-index
        # lookup or silent missing-bar fill is allowed.
        i = (e['reclaim_idx'] + 1 + sampled_delay_seconds//60
             if entry_mode == 'next_open' else e['reclaim_idx'])
        stop = e['sweep_low']*(1-stop_buffer)
        event_id = f'reclaim:{reclaim_time.isoformat()}|pivot:{pivot_time.isoformat()}'
        item = dict(event_id=event_id, reclaim_idx=int(e['reclaim_idx']),
                    pivot_idx=int(e['pivot_idx']), stop_price=float(stop),
                    signal_available_at=str(signal_available), order_ready_at=str(order_ready),
                    eligible_open_at=str(eligible_open) if eligible_open is not None else None,
                    eligible_entry_idx=int(i), **timing)
        ledger.append(item)
        if i >= len(bars):
            unfilled += 1
            item['status'] = 'unfilled'
            continue
        if i < busy:
            skipped_busy += 1
            item['status'] = 'skipped_busy'
            continue
        entry = float(op[i] if entry_mode == 'next_open' else close[i])
        item['eligible_entry_price'] = entry
        if stop >= entry:
            skipped_invalid += 1
            item['status'] = 'invalid_stop'
            continue
        deadline = i+hold_minutes
        end = min(deadline-(entry_mode == 'next_open'), len(bars)-1)
        first = i if entry_mode == 'next_open' else i+1
        hits = np.flatnonzero(low[first:end+1] <= stop)
        stopped = bool(len(hits))
        closed = stopped or deadline < len(bars)
        exit_idx = first+int(hits[0]) if stopped else deadline if closed else end
        qty = notional/entry
        # Mark only information available while the position is held. On a stop
        # bar the close after execution does not belong to this position.
        marks[i] = realized-fee
        mark_end = exit_idx if closed else exit_idx+1
        marks[first:mark_end] = realized + (close[first:mark_end]-entry)*qty-fee
        record = dict(entry_idx=int(i), entry_time=str(bars.index[i]),
                      entry_phase='open' if entry_mode == 'next_open' else 'close',
                      entry_price=entry, stop_price=float(stop), notional=notional,
                      initial_risk=float((entry-stop)*qty), event=e,
                      event_id=event_id, signal_available_at=str(signal_available),
                      order_ready_at=str(order_ready),
                      eligible_open_at=item['eligible_open_at'], **timing)
        item.update(entry_idx=int(i), entry_time=str(bars.index[i]),
                    entry_phase=record['entry_phase'], deadline_idx=int(deadline))
        if closed:
            fill = min(stop, float(op[exit_idx])) if stopped else float(
                op[exit_idx] if entry_mode == 'next_open' else close[exit_idx])
            pnl = (fill-entry)*qty-2*fee
            record.update(exit_idx=int(exit_idx), exit_time=str(bars.index[exit_idx]),
                          exit_price=fill, reason='stop' if stopped else 'time',
                          exit_phase='stop' if stopped else 'open' if entry_mode == 'next_open' else 'close',
                          pnl=float(pnl), fees=2*fee)
            trades.append(record)
            item.update(status='completed', exit_idx=int(exit_idx),
                        exit_time=str(bars.index[exit_idx]), reason=record['reason'])
            realized += pnl
            marks[exit_idx] = realized
        else:
            record.update(marked_pnl=float((close[end]-entry)*qty-fee),
                          mark_time=str(bars.index[end]), fees_paid=fee)
            opened.append(record)
            item.update(status='open_censored', mark_time=str(bars.index[end]))
        busy = i+hold_minutes
    marked = pd.Series(marks).ffill().fillna(0.).to_numpy()
    peaks = np.maximum.accumulate(np.r_[0., marked])[1:]
    drawdown = float((marked-peaks).min()) if len(marked) else 0.
    positive = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    negative = -sum(t['pnl'] for t in trades if t['pnl'] < 0)
    yearly = {}
    for t in trades:
        yr = t['exit_time'][:4]
        row = yearly.setdefault(yr, {'n': 0, 'pnl': 0.})
        row['n'] += 1; row['pnl'] += t['pnl']
    return dict(trades=trades, open_positions=opened, event_ledger=ledger,
                decision_timing=timing, unfilled_at_end=unfilled,
                skipped_busy=skipped_busy, skipped_invalid_stop=skipped_invalid,
                summary=dict(completed=len(trades), realized_pnl=float(realized),
                             profit_factor=positive/negative if negative else None,
                             win_rate=float(np.mean([t['pnl'] > 0 for t in trades])) if trades else None,
                             average_initial_risk=float(np.mean([t['initial_risk'] for t in trades])) if trades else None,
                             minute_close_mtm_drawdown=drawdown,
                             ending_marked_pnl=float(marked[-1]) if len(marked) else 0.,
                             by_exit_year=yearly))


def prefix_checks(bars, events, cutoffs):
    results = []
    for cutoff in cutoffs:
        end = int(bars.index.searchsorted(pd.Timestamp(cutoff, tz='UTC')))
        prefix = detect_events(bars.iloc[:end])
        full = [e for e in events if e['reclaim_idx'] < end]
        results.append(dict(cutoff=str(cutoff), prefix_n=len(prefix), full_n=len(full),
                            passed=prefix == full))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bars', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--entry-mode', choices=['close', 'next_open'], default='next_open')
    p.add_argument('--decision-delay-seconds', type=int, default=0,
                   help='Nonnegative integer processing delay; nonzero requires next_open')
    p.add_argument('--check-prefixes', action='store_true')
    args = p.parse_args()
    bars = pd.read_parquet(args.bars)
    events = detect_events(bars)
    result = simulate_events(bars, events, entry_mode=args.entry_mode,
                             decision_delay_seconds=args.decision_delay_seconds)
    result['events'] = events
    cutoffs = [str(t.date()) for t in pd.date_range(bars.index[0], bars.index[-1], freq='MS')]
    result['prefix_checks'] = prefix_checks(bars, events, cutoffs) if args.check_prefixes else []
    result['provenance'] = dict(input_path=str(args.bars.resolve()),
        sha256=hashlib.sha256(args.bars.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        start=str(bars.index[0]), end=str(bars.index[-1]), bars=len(bars))
    result['parameters'] = dict(entry_mode=args.entry_mode,
        decision_delay_seconds=args.decision_delay_seconds, notional=50000, stop_buffer=.0015,
        hold_minutes=240, cost_bps=12, pivot_side=15, cluster_tolerance=.001,
        prior_window_minutes=1440, prior_touch_min_age=30, sweep_depth=.0002,
        reclaim_max_minutes=30, sweep_spacing_minutes=60, tie_break='oldest_pivot',
        lockout='fixed_from_entry', starting_equity=None)
    result['limitations'] = ['Historical diagnostic, not independent OOS or WFO.',
        'Fixed notional, no funded wallet/margin/liquidation model or compounding.',
        '12bps fixed-notional round-trip assumption; funding and market impact unmodeled.',
        'Eligible minute open is a reference fill, not a claim of obtainable execution or native runner parity.',
        'Requested decision delay is rounded up to the next sampled minute open; 90 seconds samples 120 seconds after reclaim close.',
        'Events and stops are frozen across delay scenarios; changed fill prices and entry lockout can change the traded population.',
        'Minute-close MTM drawdown omits intraminute excursion.',
        'Chronological event population differs from the noncausal legacy selector.']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'minute_replay.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(events=len(events), summary=result['summary'],
                          prefix_checks=result['prefix_checks']), indent=2))
    if any(not x['passed'] for x in result['prefix_checks']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
