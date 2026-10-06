"""Causal support-reaction hypothesis. Evidence roles, not a fusion score.

No orders, fitted parameters or automatic Wyckoff phase classification.
"""
from copy import deepcopy
from decimal import Decimal
import math
from statistics import median

import pandas as pd

from scripts.research.thesis_contract import clock, positive, protocol, seal, signed, verify, validate_event, verify_packet

MINUTE = pd.Timedelta('1min')


def enough_room(price, stop, ceiling):
    """Compare serialized price values exactly at the inclusive2R boundary."""
    p, s, h = (Decimal(str(x)) for x in (price, stop, ceiling))
    return p > s and h-p >= 2*(p-s)


def policy():
    return dict(schema='support-reaction-v1', recovery_hours=48, support_hours=72,
                child_minutes=60, pivot_side_bars=2, volume_hours=20, room_r=2.,
                demand_close=2/3, adverse_close=1/3, supply_close=.5,
                evidence_rule='one_supportive_neither_adverse_both_known',
                pre_support_invalidation='completed_hour_low_at_or_below_original_stop',
                phase='unclassified', numeric_rules='project_hypotheses',
                execution_authorized=False, pristine_holdout=False)


def origin_record(packet):
    """Explicit projection: old future management tails never enter the assessor."""
    verify_packet(packet)
    keys = ('id', 'stream_id', 'parent', 'origin', 'atr4h', 'original_stop',
            'deadline', 'daily_context', 'source_status', 'execution_authorized')
    return signed(dict(schema='support-origin-v1', **{k: deepcopy(packet[k]) for k in keys}))


def _validate_origin(raw):
    verify(raw)
    o, p = raw['origin'], raw['parent']
    validate_event(o, raw['stream_id'])
    c = o['payload']
    expected_id = 'episode:'+seal({'origin': o['id'], 'parent': p['id'], 'policy': seal(protocol())})
    if (raw.get('schema') != 'support-origin-v1' or raw.get('execution_authorized') is not False
            or raw.get('id') != expected_id
            or o['timeframe'] != '4h' or not p.get('id') or not p.get('lineage_id')
            or clock(p['available_at']) >= clock(o['start'])
            or not all(positive(p[k]) for k in ('range_low', 'range_high'))
            or not c['low'] < p['range_low'] < c['close'] < p['range_high']
            or clock(raw['deadline']) != clock(o['available_at'])+pd.Timedelta('7d')):
        raise ValueError('invalid origin/parent/authority')
    if raw['atr4h'] is not None:
        if (not positive(raw['atr4h']) or not positive(raw['original_stop'])
                or raw['original_stop'] != c['low']-.1*raw['atr4h']):
            raise ValueError('invalid original risk')
    elif raw['original_stop'] is not None:
        raise ValueError('stop without ATR')


def verify_record(record):
    verify(record)
    _validate_origin(record['origin'])
    if record.get('policy_seal') != seal(policy()) or record.get('execution_authorized') is not False:
        raise ValueError('foreign support policy/authority')
    if record['phase']['state'] != 'unclassified':
        raise ValueError('no qualified phase constructor')
    for decision in record['decisions'].values():
        if clock(decision['at']) > clock(record['observed_through']):
            raise ValueError('future decision')
        for cid in decision['citations']:
            if cid not in record['catalog'] or clock(record['catalog'][cid]['available_at']) > clock(decision['at']):
                raise ValueError('foreign/future evidence citation')


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def candle(frame, start, timeframe, stream):
    """Price and volume identities are separate so absent volume cannot alter B."""
    start = clock(start); end = start+pd.Timedelta(timeframe)
    rows = frame.loc[(frame.index >= start) & (frame.index < end)]
    expected = pd.date_range(start, end, inclusive='left', freq='min')
    good = rows.index.equals(expected)
    if good:
        a = rows[['open', 'high', 'low', 'close']].to_numpy(float)
        good = all(positive(float(v)) for v in a.flat)
        good = good and bool((a[:, 2] <= a[:, [0, 3]].min(axis=1)).all()
                            and (a[:, 1] >= a[:, [0, 3]].max(axis=1)).all())
    payload = None if not good else dict(open=float(rows.iloc[0]['open']), high=float(rows['high'].max()),
        low=float(rows['low'].min()), close=float(rows.iloc[-1]['close']))
    core = dict(kind='candle', timeframe=timeframe, start=start.isoformat(), end=end.isoformat(),
                available_at=end.isoformat(), status='known' if good else 'unknown',
                payload=payload, stream_id=stream)
    price_id = 'price:'+seal(core)
    values = rows['volume'].tolist() if 'volume' in rows else []
    vol = (math.fsum(float(v) for v in values) if good and len(values) == len(expected)
           and all(_finite(v) and v >= 0 for v in values) else None)
    if payload is not None:
        payload = dict(payload, volume=vol)
    return dict(core, payload=payload, id=price_id,
                volume_id='volume:'+seal(dict(price_id=price_id, value=vol)))


def evidence(current, prior, role):
    if role not in ('demand', 'supply'):
        raise ValueError('unknown evidence role')
    expected = [clock(current['start'])-pd.Timedelta(hours=i) for i in range(20, 0, -1)]
    ids = [r.get('volume_id', r['id']) for r in prior]
    out = dict(role=role, state='unknown', candle_id=current['id'],
               volume_id=current.get('volume_id', current['id']), baseline_ids=ids,
               relative_volume=None, relative_spread=None, close_location=None,
               available_at=current['available_at'], reason='incomplete_or_invalid_baseline')
    if len(prior) != 20 or [clock(r['start']) for r in prior] != expected:
        return out
    if any(r['status'] != 'known' for r in [*prior, current]):
        return out
    cs = [r['payload'] for r in [*prior, current]]
    if any(not _finite(c.get('volume')) or c['volume'] < 0 for c in cs):
        return out
    volumes = [c['volume'] for c in cs[:-1]]
    spreads = [c['high']-c['low'] for c in cs[:-1]]
    c = cs[-1]; width = c['high']-c['low']
    if median(volumes) <= 0 or median(spreads) <= 0 or width <= 0:
        return out
    rv, rs, cl = c['volume']/median(volumes), width/median(spreads), (c['close']-c['low'])/width
    supportive = (c['close'] > c['open'] and rv >= 1 and rs >= 1 and cl >= 2/3) if role == 'demand' else (rv < 1 and rs < 1 and cl >= .5)
    adverse = (rv > 1 and cl <= 1/3) if role == 'demand' else (c['close'] < c['open'] and rv > 1 and rs > 1 and cl <= 1/3)
    state = 'supportive' if supportive else 'adverse' if adverse else 'neutral'
    return dict(out, state=state, relative_volume=rv, relative_spread=rs,
                close_location=cl, reason=role+'_'+state)


def evidence_action(demand, supply):
    states = [demand['state'], supply['state']]
    if any(s not in ('supportive', 'neutral', 'adverse', 'unknown') for s in states):
        raise ValueError('invalid evidence state')
    return ('unknown' if 'unknown' in states else 'reject' if 'adverse' in states
            else 'allow' if 'supportive' in states else 'reject')


def assess(origin, minutes, *, as_of):
    _validate_origin(origin)
    raw = deepcopy(origin); t0 = clock(raw['origin']['available_at']); cutoff = clock(as_of)
    if cutoff < t0:
        raise ValueError('cutoff precedes origin')
    begin = clock(raw['origin']['start'])-pd.Timedelta('20h')
    frame = minutes.loc[(minutes.index >= begin) & (minutes.index < cutoff)].copy()
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing
            or not (frame.index == frame.index.floor('min')).all()):
        raise ValueError('unique ordered aware minute source required')
    stream, parent = raw['stream_id'], raw['parent']
    lo, hi = parent['range_low'], parent['range_high']
    spring = raw['origin']['payload']; k = spring['high']; stop = raw['original_stop']
    reconstructed = candle(frame, raw['origin']['start'], '4h', stream)
    if reconstructed['status'] != 'known' or any(reconstructed['payload'][f] != spring[f] for f in ('open', 'high', 'low', 'close')):
        raise ValueError('origin price reconstruction mismatch')
    r = dict(schema='support-assessment-v1', policy_seal=seal(policy()), origin=raw,
             observed_through=cutoff.isoformat(), execution_authorized=False,
             phase=dict(state='unclassified', reason='no_qualified_phase_sequence'),
             structure=dict(range_kind='pivot_range', parent_id=parent['id'], parent_breakouts=[],
                            recovery=None, support=None, child_high=None, child_low=None, trigger=None),
             catalog={}, demand=None, supply=None, parent_invalid_at=None,
             price_unknown_at=None, child_invalid_at=None, setup_invalid_at=None, decisions={})
    catalog = r['catalog']
    def remember(e):
        catalog[e['id']] = deepcopy(e)
        if 'volume_id' in e:
            catalog[e['volume_id']] = dict(id=e['volume_id'], kind='volume', available_at=e['available_at'],
                                           price_id=e['id'], value=(e['payload'] or {}).get('volume'))
        return e
    remember(dict(raw['origin']))
    remember(dict(parent, available_at=clock(parent['available_at']).isoformat(), kind='parent'))
    for at in pd.date_range(t0-pd.Timedelta('20h'), t0-pd.Timedelta('1h'), freq='h'):
        remember(candle(frame, at, '1h', stream))
    citations = [parent['id'], raw['origin']['id']]
    def decision(arm, status, reason, at, extra=None):
        d = dict(arm=arm, status=status, reason=reason, at=clock(at).isoformat(),
                 origin_id=raw['id'], parent_id=parent['id'], citations=list(citations),
                 execution_authorized=False, **(extra or {}))
        d['id'] = 'support-decision:'+seal(d)
        return d
    def finish(status, reason, at, extra=None):
        r['decisions'] = {a: decision(a, status, reason, at, extra) for a in ('B', 'C')}
        return signed(r)
    if raw['source_status'] != 'known' or stop is None:
        r['price_unknown_at'] = t0.isoformat()
        return finish('unknown', 'unknown_initial_risk', t0)
    hourly = {}
    end = min(cutoff, t0+pd.Timedelta('73h'))
    for at in pd.date_range(begin, end.floor('h')-pd.Timedelta('1h'), freq='h'):
        hourly[at] = candle(frame, at, '1h', stream)
    recovery = support = None
    stage = 'recovery'; boundary = t0+pd.Timedelta('48h')
    for at in pd.date_range(t0+pd.Timedelta('1h'), min(cutoff, t0+pd.Timedelta('72h')), freq='h'):
        if at > boundary:
            return finish('expired', stage+'_expired', boundary)
        h = remember(hourly[at-pd.Timedelta('1h')])
        if at == at.floor('4h'):
            four = remember(candle(frame, at-pd.Timedelta('4h'), '4h', stream))
            if four['status'] != 'known':
                citations.append(four['id'])
                r['price_unknown_at'] = at.isoformat()
                return finish('unknown', 'missing_parent_price', at)
            if four['payload']['close'] < lo:
                r['parent_invalid_at'] = at.isoformat(); citations.append(four['id'])
                return finish('rejected', 'parent_invalidated', at)
            if four['payload']['close'] > hi:
                r['structure']['parent_breakouts'].append(dict(evidence_id=four['id'], available_at=four['available_at'],
                                                               state='close_above_range_not_acceptance'))
        if h['status'] != 'known':
            citations.append(h['id'])
            r['price_unknown_at'] = at.isoformat()
            return finish('unknown', 'missing_hourly_price', at)
        c = h['payload']
        if c['low'] <= stop:
            citations.append(h['id']); r['setup_invalid_at'] = at.isoformat()
            return finish('rejected', 'original_spring_stop_failed', at)
        if stage == 'recovery' and c['close'] > k:
            recovery = h; r['structure']['recovery'] = h['id']; citations.append(h['id'])
            prior = [remember(hourly[clock(h['start'])-pd.Timedelta(hours=i)]) for i in range(20, 0, -1)]
            r['demand'] = evidence(h, prior, 'demand')
            stage = 'support'; boundary = t0+pd.Timedelta('72h')
        elif stage == 'support' and spring['low'] < c['low'] <= k < c['close']:
            support = h; r['structure']['support'] = h['id']; citations.append(h['id'])
            prior = [remember(hourly[clock(h['start'])-pd.Timedelta(hours=i)]) for i in range(20, 0, -1)]
            r['supply'] = evidence(h, prior, 'supply')
            break
    if support is None:
        return finish('expired' if cutoff >= boundary else 'pending', stage+'_expired' if cutoff >= boundary else 'await_'+stage,
                      boundary if cutoff >= boundary else cutoff)
    s = clock(support['available_at']); expiry = min(s+pd.Timedelta('60min'), clock(raw['deadline']))
    child = []; high = low = None
    for at in pd.date_range(s, min(cutoff, expiry)-MINUTE, freq='min'):
        e = remember(candle(frame, at, '1min', stream)); available = at+MINUTE
        if available == available.floor('4h'):
            four = remember(candle(frame, available-pd.Timedelta('4h'), '4h', stream))
            if four['status'] != 'known':
                citations.append(four['id'])
                r['price_unknown_at'] = available.isoformat()
                return finish('unknown', 'missing_parent_price', available)
            if four['payload']['close'] < lo:
                citations.append(four['id']); r['parent_invalid_at'] = available.isoformat()
                return finish('rejected', 'parent_invalidated', available)
        if e['status'] != 'known':
            citations.append(e['id'])
            r['price_unknown_at'] = available.isoformat()
            return finish('unknown', 'missing_child_price', available)
        c = e['payload']
        if c['low'] <= support['payload']['low']:
            citations.append(e['id']); r['child_invalid_at'] = available.isoformat()
            return finish('rejected', 'child_support_failed', available)
        if low and at >= clock(low['available_at']) and c['close'] > high['price'] and available < expiry:
            r['structure']['trigger'] = e['id']; citations.append(e['id'])
            price = c['close']; room = (hi-price)/(price-stop) if price > stop else None
            extra = dict(trigger_id=e['id'], decision_price=price, original_stop=stop, room_r=room,
                         expires_at=expiry.isoformat(), support_low=support['payload']['low'])
            if not lo < price < hi or not enough_room(price, stop, hi):
                return finish('rejected', 'insufficient_parent_room', available, extra)
            b = decision('B', 'intent', 'structure_confirmed', available, extra)
            action = evidence_action(r['demand'], r['supply'])
            for v in (r['demand'], r['supply']):
                citations.extend([v['volume_id'], *v['baseline_ids']])
            citations[:] = list(dict.fromkeys(citations))
            c_decision = decision('C', {'allow': 'intent', 'reject': 'rejected', 'unknown': 'unknown'}[action],
                                 {'allow': 'contextual_evidence_supported', 'reject': 'volume_evidence_challenge',
                                  'unknown': 'unknown_volume_evidence'}[action], available, extra)
            r['decisions'] = {'B': b, 'C': c_decision}
            return signed(r)
        child.append(e)
        if len(child) < 5:
            continue
        window = child[-5:]; center = window[2]; cp = center['payload']
        kind = ('high' if high is None and all(cp['high'] > x['payload']['high'] for i, x in enumerate(window) if i != 2)
                else 'low' if high is not None and low is None and clock(center['start']) > clock(high['center_start'])
                and cp['low'] > support['payload']['low'] and all(cp['low'] < x['payload']['low'] for i, x in enumerate(window) if i != 2) else None)
        if kind:
            pivot = dict(kind='pivot_'+kind, price=cp[kind], center_start=center['start'],
                         available_at=available.isoformat(), input_ids=[x['id'] for x in window])
            pivot['id'] = 'child:'+seal(pivot); remember(pivot); citations.append(pivot['id'])
            r['structure']['child_'+kind] = pivot
            if kind == 'high': high = pivot
            else: low = pivot
    return finish('expired' if cutoff >= expiry else 'pending', 'child_expired' if cutoff >= expiry else 'await_child',
                  expiry if cutoff >= expiry else cutoff)
