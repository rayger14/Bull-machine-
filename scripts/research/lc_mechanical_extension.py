"""Fixed-rule retrospective extension; no model calls, tuning or live side effects.

Caller verifies and freezes census/input hashes before passing outcome bars.
Groups are contributions from continuous books, not reset/subtype portfolios.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
from numbers import Real

import pandas as pd

from scripts.research.conditional_entry import _clock
from scripts.research.conditional_occupancy import replay_sleeve
from scripts.research.lc_context_facts import _hourly
from scripts.research.lc_single_accounting import _admission_aware_mtm


def _positive(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def prepare_cases(rows, bars):
    """Keep every source case; freeze plans from exactly two completed hours."""
    if not rows:
        raise ValueError('nonempty census required')
    if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None
            or not bars.index.is_unique or not bars.index.is_monotonic_increasing):
        raise ValueError('unique ordered timezone-aware bars required')
    ids = [r['candidate_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate candidate identity')
    clocks = [_clock(r['decision_time']) for r in rows]
    if len(set(clocks)) != len(clocks):
        raise ValueError('duplicate LC decision time')
    cases = []
    for raw in sorted(rows, key=lambda r: _clock(r['decision_time'])):
        d = _clock(raw['decision_time'])
        if d != d.floor('h'):
            raise ValueError('hourly decision required')
        case = dict(candidate_id=raw['candidate_id'], decision_time=d.isoformat(),
                    subtype='unresolved', plans=None, unavailable_reason=None)
        try:
            if raw['native_diagnostic']['native_signal']['direction'] != 'long':
                raise ValueError('not_native_long')
            prefix = bars.loc[(bars.index >= d-pd.Timedelta(hours=2)) & (bars.index < d)]
            if not prefix.index.equals(pd.date_range(d-pd.Timedelta(hours=2), periods=120, freq='min')):
                raise ValueError('incomplete_predecision_minutes')
            if (not all(_positive(v) for v in prefix[['open','high','low','close']].to_numpy().flat)
                    or (prefix['low'] > prefix[['open','close']].min(axis=1)).any()
                    or (prefix['high'] < prefix[['open','close']].max(axis=1)).any()):
                raise ValueError('invalid_predecision_prices')
            hourly = []
            for frame, features in [(prefix.iloc[:60], raw['previous_features']),
                                    (prefix.iloc[60:], raw['features'])]:
                candle = dict(open=float(frame.iloc[0]['open']), high=float(frame['high'].max()),
                              low=float(frame['low'].min()), close=float(frame.iloc[-1]['close']))
                if any(not _positive(features[k]) or not math.isclose(features[k], v, rel_tol=0, abs_tol=1e-8)
                       for k, v in candle.items()):
                    raise ValueError('source_hourly_mismatch')
                hourly.append(candle)
            atr = raw['features']['atr_14']
            if not _positive(atr):
                raise ValueError('invalid_source_atr')
            stop = hourly[1]['close'] - 2.7*float(atr)
            if not _positive(stop):
                raise ValueError('invalid_stop')
            context = _hourly(hourly[1], hourly[0])
            if context['close_relation'] == 'above_prior_high':
                case['subtype'] = 'upside_expansion_candidate'
            elif context['close_relation'] == 'below_prior_low' or context['reclaimed_prior_low']:
                case['subtype'] = 'downside_rebound_candidate'
            base = dict(decision_time=d.isoformat(), stop=stop,
                        entry_expiry=(d+pd.Timedelta(minutes=15)).isoformat(),
                        exit_deadline=(d+pd.Timedelta(days=1)).isoformat(),
                        processing_seconds=90, routing_seconds=0, notional=50000., cost_bps=12)
            case['plans'] = dict(immediate=dict(base, action='enter', level=None),
                mechanical_wait=dict(base, action='wait_close_above', level=float(prefix.iloc[-5:]['high'].max())))
        except (KeyError, TypeError, ValueError) as exc:
            case['unavailable_reason'] = str(exc)
        cases.append(case)
    return cases


def _groups(ledger, labels):
    grouped = defaultdict(list)
    for row in ledger:
        grouped[labels[row['candidate_id']]].append(row)
    result = {}
    for label, rows in sorted(grouped.items()):
        closed = [r['position']['net_pnl'] for r in rows
                  if r['position'] and r['position']['status'] == 'closed']
        complete = all(r['status'] in ('admitted','skipped_busy','rejected','expired','cancelled')
                       and (not r['position'] or r['position']['status']=='closed') for r in rows)
        subtotal = math.fsum(closed)
        result[label] = dict(candidate_count=len(rows), closed_count=len(closed),
            wins=sum(v > 0 for v in closed), losses=sum(v < 0 for v in closed),
            statuses=dict(Counter(r['status'] for r in rows)), known_net_subtotal=subtotal,
            policy_net_pnl=subtotal if complete else None,
            contribution_only=True)
    return result


def score_extension(cases, bars):
    """Replay the frozen cohort continuously, preserving timing/occupancy gaps."""
    if not cases or len({c['candidate_id'] for c in cases}) != len(cases):
        raise ValueError('nonempty unique cases required')
    as_of = max(_clock(c['decision_time']) for c in cases)+pd.Timedelta(days=1)
    groups = {
        'by_year': {c['candidate_id']: str(_clock(c['decision_time']).year) for c in cases},
        'by_quarter': {c['candidate_id']: f"{_clock(c['decision_time']).year}Q{_clock(c['decision_time']).quarter}" for c in cases},
        'by_subtype': {c['candidate_id']: c['subtype'] for c in cases},
    }
    scenarios = {}
    for cost, delay in ((12,90),(24,90),(12,300),(24,300)):
        arms = {}
        for arm in ('immediate','mechanical_wait'):
            candidates = []
            for case in cases:
                plan = deepcopy(case['plans'][arm]) if case['plans'] else None
                if plan is not None:
                    plan.update(cost_bps=cost, processing_seconds=delay)
                candidates.append(dict(candidate_id=case['candidate_id'], track='hourly',
                    decision_time=case['decision_time'], plan=plan,
                    unavailable_reason=case['unavailable_reason']))
            book = replay_sleeve(bars, candidates, track='hourly', as_of=as_of.isoformat())
            book.update({key: _groups(book['ledger'], labels) for key, labels in groups.items()})
            book['mtm'] = (_admission_aware_mtm(bars,book) if (cost,delay)==(12,90)
                           else dict(status='not_calculated',max_drawdown_dollars=None))
            arms[arm] = book
        scenarios[f'{cost}bps_{delay}s'] = arms
    return dict(version='lc_mechanical_extension_v1', case_count=len(cases), scenarios=scenarios,
                execution_authorized=False, pristine_holdout=False, optimization_performed=False)
