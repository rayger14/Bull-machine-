"""Bounded exposed-history diagnostics, not fitted rules or trading authority."""
import math

import numpy as np
import pandas as pd
import talib

from scripts.research.conditional_entry import _clock, score_conditional


LABELS = {
    'parent_4h_lifecycle': ('intact', 'broken_up', 'broken_down', 'broken_multiple', 'absent', 'unknown'),
    'parent_1d_lifecycle': ('intact', 'broken_up', 'broken_down', 'broken_multiple', 'absent', 'unknown'),
    'parent_4h_location': ('below', 'inside', 'above', 'absent', 'unknown'),
    'parent_1d_location': ('below', 'inside', 'above', 'absent', 'unknown'),
    'mapped_overhead': ('below_2r', 'at_least_2r', 'no_reference', 'unknown'),
    'last_two_5m': ('both', 'other', 'unknown'),
}


def hourly_features(hourly, start, end):
    """Decision-indexed facts; ATR seeds are supplied independently per month.

    Matching covariates deliberately exclude the setup candle. No outcomes or
    rolling/full-sample ranks are consumed. Caller verifies minute aggregation.
    """
    idx = hourly.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or not len(idx)
            or idx[0] != idx[0].floor('h')
            or not idx.equals(pd.date_range(idx[0], idx[-1], freq='h'))):
        raise ValueError('complete unique ordered hourly grid required')
    prices = hourly[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (prices[:, 2] > prices[:, [0, 3]].min(axis=1)).any()
            or (prices[:, 1] < prices[:, [0, 3]].max(axis=1)).any()):
        raise ValueError('valid hourly prices required')
    atr = pd.Series(talib.ATR(prices[:, 1], prices[:, 2], prices[:, 3], timeperiod=14), index=idx)
    close = hourly['close']
    result = pd.DataFrame(dict(close=close, atr=atr, previous_atr_pct=atr.shift(1)/close.shift(1),
                               prior_return_24h=close.shift(1)/close.shift(25)-1,
                               upside=close > hourly['high'].shift(1)))
    result.index = idx.tz_convert('UTC') + pd.Timedelta('1h')
    start, end = _clock(start), _clock(end)
    if start >= end:
        raise ValueError('nonempty feature interval required')
    return result.loc[(result.index >= start) & (result.index < end)]


def match_controls(cases, features, native_decisions):
    """Fixed past-only, no-replacement match, with every missing case retained."""
    ids = [c['candidate_id'] for c in cases]
    clocks = [_clock(c['decision_time']) for c in cases]
    if len(ids) != len(set(ids)) or len(clocks) != len(set(clocks)):
        raise ValueError('unique case identities and clocks required')
    idx = features.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.tz is None
            or not idx.is_unique or not idx.is_monotonic_increasing):
        raise ValueError('unique ordered aware features required')
    frame = features.copy()
    frame.index = idx.tz_convert('UTC')
    if any(t != t.floor('h') for t in clocks) or not (frame.index == frame.index.floor('h')).all():
        raise ValueError('exact hourly decisions required')
    excluded = set(map(_clock, native_decisions)) | set(clocks)
    finite = np.isfinite(frame[['previous_atr_pct', 'prior_return_24h']]).all(axis=1)
    valid = finite & (frame['previous_atr_pct'] > 0)
    pool = frame.loc[valid & (frame['upside'] == True) & ~frame.index.isin(excluded)]
    used, result = set(), []
    for case in sorted(cases, key=lambda c: _clock(c['decision_time'])):
        d = _clock(case['decision_time'])
        row = dict(candidate_id=case['candidate_id'], decision_time=d.isoformat(),
                   status='unmatched', control_id=None, control_time=None,
                   reason='no_control_within_fixed_rules')
        if d not in frame.index or not valid.loc[d]:
            result.append(dict(row, reason='case_covariates_unavailable'))
            continue
        own = frame.loc[d]
        eligible = pool.loc[(pool.index >= d-pd.Timedelta(days=90))
                            & (pool.index <= d-pd.Timedelta(days=1))
                            & (pool.index.hour == d.hour) & ~pool.index.isin(used)
                            & (np.sign(pool['prior_return_24h']) == np.sign(own['prior_return_24h']))].copy()
        eligible['distance'] = abs(np.log(eligible['previous_atr_pct']/own['previous_atr_pct']))
        eligible = eligible.loc[eligible['distance'] <= math.log(1.5)]
        if len(eligible):
            at = min(eligible.index, key=lambda t: (eligible.loc[t, 'distance'], -t.value))
            selected = eligible.loc[at]
            used.add(at)
            row.update(status='matched', reason=None, control_id='control:'+at.isoformat(),
                       control_time=at.isoformat(), lag_days=float((d-at)/pd.Timedelta(days=1)),
                       volatility_ratio=float(selected['previous_atr_pct']/own['previous_atr_pct']),
                       lc_prior_return_24h=float(own['prior_return_24h']),
                       control_prior_return_24h=float(selected['prior_return_24h']))
        result.append(row)
    return result


def context_labels(facts, *, close, stop):
    """Categorize caller-validated context facts; no outcome or permission rule."""
    result, overhead = {}, []
    geometry_known = math.isfinite(close) and math.isfinite(stop) and 0 < stop < close
    room_known = geometry_known
    for key in ('parent_4h', 'parent_1d'):
        parent = facts[key]
        known = parent['evidence_status'] == 'known'
        lifecycle = parent['lifecycle'] if known else 'unknown'
        result[key+'_lifecycle'] = lifecycle
        location = 'unknown'
        room_known = room_known and known
        if known and lifecycle == 'absent':
            location = 'absent'
        elif known and geometry_known:
            bound = parent['bound']
            low, high = bound['range_low'], bound['range_high']
            location = 'below' if close < low else 'above' if close > high else 'inside'
            overhead.extend((v-close)/(close-stop) for v in (low, high) if v > close)
        result[key+'_location'] = location
    result['mapped_overhead'] = ('unknown' if not room_known else 'no_reference' if not overhead
                                 else 'below_2r' if min(overhead) < 2 else 'at_least_2r')
    sequence = facts['last_two_5m']
    result['last_two_5m'] = ('unknown' if sequence['status'] != 'known' else 'both'
                            if sequence['higher_low'] and sequence['higher_close'] else 'other')
    return result


def event_result(plan, bars):
    """Independent fixed-plan event outcome; nonentries zero, unavailable null."""
    raw = score_conditional(bars, as_of=plan['exit_deadline'], **plan)
    out = raw['outcome']
    valid = out['status'] == 'valid'
    nonentry = out['status'] in ('cancelled', 'expired', 'rejected')
    return dict(resolved=valid or nonentry, filled=raw['resolution']['status'] == 'entry_ready',
                net_pnl=out['net_pnl'], net_r=(out['net_pnl']/(out['initial_risk']+out['fees'])
                if valid else 0. if nonentry else None), raw=raw)


def summarize_events(rows):
    known = [r for r in rows if r['resolved']]
    complete = len(known) == len(rows)
    dollars = [r['net_pnl'] for r in known]
    losses = -math.fsum(v for v in dollars if v < 0)
    total = math.fsum(dollars)
    total_r = math.fsum(r['net_r'] for r in known)
    top = sorted((v for v in dollars if v > 0), reverse=True)[:3]
    return dict(candidate_count=len(rows), resolved_count=len(known), unresolved=len(rows)-len(known),
                filled_count=sum(r['filled'] for r in known),
                nonentries=sum(not r['filled'] for r in known),
                wins=sum(v > 0 for v in dollars), losses=sum(v < 0 for v in dollars),
                known_net_subtotal=total, net_pnl=total if complete else None,
                mean_net_r=total_r/len(rows) if complete and rows else None,
                profit_factor=math.fsum(v for v in dollars if v > 0)/losses
                if complete and losses else None,
                net_without_top_three=total-math.fsum(top) if complete else None)


def paired_summary(pairs, events, first_month, last_month):
    """Descriptive calendar-month bootstrap of paired per-candidate R deltas."""
    matched = [p for p in pairs if p['status'] == 'matched']
    ids = [p['candidate_id'] for p in pairs]
    control_ids = [p['control_id'] for p in matched]
    if len(set(ids)) != len(ids) or len(set(control_ids)) != len(control_ids):
        raise ValueError('unique pairs and controls required')
    months = [str(m) for m in pd.period_range(first_month, last_month, freq='M')]
    monthly_sums, monthly_counts = np.zeros(len(months)), np.zeros(len(months), dtype=int)
    deltas = []
    for pair in matched:
        a, b = events[pair['candidate_id']], events[pair['control_id']]
        month = _clock(pair['decision_time']).strftime('%Y-%m')
        if month not in months:
            raise ValueError('pair outside calendar')
        if a['resolved'] and b['resolved']:
            delta = a['net_r']-b['net_r']
            deltas.append(delta)
            i = months.index(month)
            monthly_sums[i] += delta
            monthly_counts[i] += 1
    complete = len(deltas) == len(matched)
    interval, empty = None, None
    if complete and len(deltas) and np.count_nonzero(monthly_counts) >= 2:
        rng = np.random.default_rng(20260930)
        sample = rng.integers(0, len(months), size=(5000, len(months)))
        counts = monthly_counts[sample].sum(axis=1)
        sums = monthly_sums[sample].sum(axis=1)
        populated = counts > 0
        empty = int((~populated).sum())
        interval = np.quantile(sums[populated]/counts[populated], [.025, .975]).tolist()
    return dict(matched_count=len(matched), unmatched_count=len(pairs)-len(matched),
                resolved_pairs=len(deltas), calendar_months=len(months),
                mean_delta_net_r=math.fsum(deltas)/len(deltas) if complete and deltas else None,
                bootstrap_95_delta=interval, bootstrap_zero_count_draws=empty,
                bootstrap_samples=5000, bootstrap_seed=20260930,
                selection_adjusted=False, pristine_holdout=False,
                lc=summarize_events([events[p['candidate_id']] for p in matched]),
                control=summarize_events([events[p['control_id']] for p in matched]))
