"""As-of price evidence. Integrity failures raise; unavailable evidence abstains."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from scripts.research.lc_context_contract import MINUTE, clock, positive, protocol, seal, sealed_case

OHLCV = ['open', 'high', 'low', 'close', 'volume']


def validate_index(frame):
    idx = frame.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or not idx.is_unique
            or not idx.is_monotonic_increasing or not (idx == idx.floor('min')).all()):
        raise ValueError('unique ordered aware minute index required')


def candle(frame, start, end):
    """Aggregate only a complete half-open constituent window; no forward fill."""
    start, end = clock(start), clock(end)
    cut = frame.loc[start:end-MINUTE, OHLCV]
    if not cut.index.equals(pd.date_range(start, end-MINUTE, freq='min')):
        return None
    values = cut.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
            or (values[:, 2] > values[:, [0, 3]].min(axis=1)).any()
            or (values[:, 1] < values[:, [0, 3]].max(axis=1)).any()):
        raise ValueError('invalid minute source geometry')
    return {'open': float(values[0, 0]), 'high': float(values[:, 1].max()),
            'low': float(values[:, 2].min()), 'close': float(values[-1, 3]),
            'volume': math.fsum(values[:, 4])}


def select_parent(ledger, setup_open, instrument, stream, timeframe):
    """Freeze newest past version, including broken historical references."""
    s = clock(setup_open)
    blank = {'status': 'unknown', 'bound': None, 'state': 'unknown', 'events': [],
             'legacy_state': 'unknown', 'reason': 'missing_parent_ledger'}
    if ledger is None:
        return blank
    manifest = ledger['manifest']
    if (manifest['instrument'] != instrument or manifest['data_stream_id'] != stream
            or manifest['parameters']['anchor_timeframe'] != timeframe
            or manifest['parameters']['pivot_n'] != 3):
        raise ValueError('parent constructor/source binding mismatch')
    coverage = ledger['coverage']
    if (clock(coverage['first_open']) > s-pd.Timedelta('30d')
            or clock(coverage['last_processed_close']) < s):
        return dict(blank, reason='incomplete_parent_coverage')
    past = [v for v in ledger['versions'] if clock(v['available_at']) < s]
    if len({v['id'] for v in past}) != len(past):
        raise ValueError('duplicate parent version identity')
    recent = [v for v in past if s-pd.Timedelta('30d') <= clock(v['formation_hour']) < s]
    if not recent:
        return dict(blank, status='absent', state='absent', reason='no_parent_reference')
    bound = max(recent, key=lambda v: (clock(v['available_at']), v['id']))
    if (not positive(bound['range_low']) or not positive(bound['range_high'])
            or bound['range_low'] >= bound['range_high'] or not bound['lineage_id']
            or clock(bound['available_at']) != clock(bound['formation_hour']) + pd.Timedelta('1h')):
        raise ValueError('invalid parent version geometry/clock')
    pivots = {p['id']: p for p in ledger['pivots']}
    for side in ('low', 'high'):
        pivot = pivots.get(bound[side+'_pivot_id'])
        if (pivot is None or pivot['side'] != side or pivot['data_stream_id'] != stream
                or clock(pivot['available_at']) > clock(bound['available_at'])):
            raise ValueError('unavailable or foreign parent pivot')
    # Legacy lifecycle is annotation only, and never selects the parent.
    transitions = [e for e in ledger['transitions'] if clock(e['available_at']) <= s
                   and bound['id'] in (e.get('evaluated_version_id'), e.get('pre_version_id'),
                                       e.get('post_version_id'))]
    last = max(transitions, key=lambda e: clock(e['available_at']), default={})
    return dict(blank, status='known', bound=deepcopy(bound), reason=None,
                legacy_state=last.get('post_state', 'unobserved'),
                constructor_id=manifest['contract_id'], timeframe=timeframe)


def acceptance(bound, minutes, as_of, timeframe, stream):
    """Completed originating-timeframe closes after level availability only."""
    validate_index(minutes)
    at, delta = clock(as_of), pd.Timedelta(timeframe.lower())
    opened = clock(bound['available_at']).ceil(timeframe.lower())
    lower, upper = bound['range_low'], bound['range_high']
    events, missing = [], False
    while opened + delta <= at:
        closed = opened + delta
        observed = candle(minutes, opened, closed)
        if observed is None:
            state = 'unknown'
            missing = True
        else:
            price = observed['close']
            state = ('accepted_above' if price > upper else 'accepted_below' if price < lower
                     else 'inside' if lower < price < upper else 'boundary')
        event = {'kind': 'originating_timeframe_close', 'parent_version_id': bound['id'],
                 'lineage_id': bound['lineage_id'], 'data_stream_id': stream,
                 'observation_start': opened.isoformat(), 'observation_end': closed.isoformat(),
                 'available_at': closed.isoformat(), 'state': state, 'candle': observed,
                 'excursion_below': observed['low'] < lower if observed else None,
                 'excursion_above': observed['high'] > upper if observed else None}
        events.append(dict(event, id=seal(event)))
        opened = closed
    return {'state': 'unknown' if missing else events[-1]['state'] if events else 'not_established',
            'events': events}


def prepare_evidence(raw, minutes, parents, provenance):
    validate_index(minutes)
    t = clock(raw['decision_time'])
    if t != t.floor('h') or clock(raw['feature_available_at']) > t:
        raise ValueError('invalid native availability/hour clock')
    if raw['native_diagnostic']['native_signal']['direction'] != 'long':
        raise ValueError('native long LC source required')
    if provenance.get('reconstruction_verified') is not True:
        raise ValueError('unverified source provenance')
    s = t-pd.Timedelta('1h')
    prior, current = candle(minutes, s-pd.Timedelta('1h'), s), candle(minutes, s, t)
    case = {'schema': 'lc-context-case-v1', 'candidate_id': raw['candidate_id'],
            'decision_time': t.isoformat(), 'setup_open': s.isoformat(),
            'instrument': provenance['instrument'], 'data_stream_id': provenance['data_stream_id'],
            'source_status': 'known', 'source_reason': None, 'subtype': 'unresolved',
            'current': current, 'prior': prior, 'risk_status': 'unknown', 'stop': None,
            'h5_status': 'unknown', 'h5': None, 'policy_seal': seal(protocol()),
            'provenance': deepcopy(provenance), 'execution_authorized': False}
    if prior is None or current is None:
        case.update(source_status='unknown', source_reason='incomplete_predecision_minutes')
    else:
        for observed, field in ((prior, 'previous_features'), (current, 'features')):
            for key, value in observed.items():
                if (not isinstance(raw[field].get(key), (int, float))
                        or not math.isfinite(raw[field][key])
                        or not math.isclose(raw[field][key], value, abs_tol=1e-8, rel_tol=0)):
                    raise ValueError('native candle reconstruction mismatch: '+field+'.'+key)
        if current['close'] > prior['high']:
            case['subtype'] = 'upside_expansion_candidate'
        elif current['close'] < prior['low'] or current['low'] < prior['low'] < current['close']:
            case['subtype'] = 'downside_rebound_candidate'
        atr = raw['features'].get('atr_14')
        if positive(atr) and positive(current['close']-2.7*atr):
            case.update(risk_status='known', stop=current['close']-2.7*atr)
    five = candle(minutes, t-pd.Timedelta('5min'), t)
    if five is not None:
        case.update(h5_status='known', h5=five['high'])
    for tf, name in [('4H', 'parent_4h'), ('1D', 'parent_1d')]:
        parent = select_parent(parents.get(tf+'_N3'), s, case['instrument'], case['data_stream_id'], tf)
        if parent['status'] == 'known':
            parent.update(acceptance(parent['bound'], minutes, t, tf, case['data_stream_id']))
            if parent['state'] == 'unknown':
                parent.update(status='unknown', reason='incomplete_acceptance_constituents')
        case[name] = parent
    # Native scores are retained as unqualified snapshots, never directional votes.
    fields = ('fusion_total', 'macro_regime', 'funding_rate', 'oi_change_4h', 'taker_imbalance')
    case['optional'] = {k: {'value': raw['features'].get(k), 'status': 'unqualified_native_snapshot',
                            'available_at': t.isoformat(), 'permission_role': 'none'}
                        for k in fields if isinstance(raw['features'].get(k), (str, int, float))
                        and (not isinstance(raw['features'][k], (int, float))
                             or math.isfinite(raw['features'][k]))}
    case['destinations'] = [{'parent_version_id': case[name]['bound']['id'],
                             'timeframe': case[name]['timeframe'], 'price': case[name]['bound'][level],
                             'lifecycle': case[name]['state'], 'role': 'reference_not_permission'}
                            for name in ('parent_4h', 'parent_1d') if case[name]['bound'] is not None
                            for level in ('range_low', 'range_high')]
    return sealed_case(case)
