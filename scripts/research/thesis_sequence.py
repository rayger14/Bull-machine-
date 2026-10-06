"""Causal ordered episode compilation. Later events never rewrite past decisions."""
from copy import deepcopy

import pandas as pd

from scripts.research.thesis_contract import clock, positive, protocol, seal, signed, validate_event


def compile_episode(base, events):
    p = deepcopy(base)
    policy = protocol()
    origin, parent = p['origin'], p['parent']
    validate_event(origin, p['stream_id'])
    if origin['kind'] != 'candle' or origin['timeframe'] != '4h':
        raise ValueError('origin must be a four-hour candle')
    t0 = clock(origin['available_at'])
    if (p.get('execution_authorized') is not False or not parent['lineage_id']
            or clock(parent['available_at']) >= clock(origin['start'])
            or not positive(parent['range_low']) or not positive(parent['range_high'])
            or parent['range_low'] >= parent['range_high'] or origin['status'] != 'known'):
        raise ValueError('invalid initial episode parent/authority')
    lo, hi = parent['range_low'], parent['range_high']
    c0 = origin['payload']
    if not (c0['low'] < lo < c0['close'] < hi):
        raise ValueError('not a spring/reclaim origin')
    atr = p['atr4h']
    if atr is not None and not positive(atr):
        raise ValueError('invalid ATR')
    stop = None if atr is None else c0['low']-policy['stop_buffer_atr']*atr
    if stop is not None and not positive(stop):
        raise ValueError('invalid initial stop')
    eid = 'episode:'+seal({'origin': origin['id'], 'parent': parent['id'], 'policy': seal(policy)})
    ordered = sorted(deepcopy(events), key=lambda e: (clock(e['available_at']),
                     {'4h': 0, '1h': 1, '1min': 2}.get(e['timeframe'], 3), e['id']))
    seen, observations = set(), set()
    for e in ordered:
        validate_event(e, p['stream_id'])
        if e['id'] in seen:
            raise ValueError('duplicate event')
        seen.add(e['id'])
        key = (e['kind'], e['timeframe'], e['start'], e['end'])
        if key in observations:
            raise ValueError('duplicate observation')
        observations.add(key)
        if clock(e['available_at']) <= t0:
            raise ValueError('episode event must follow origin')
    deadline = t0+pd.Timedelta(days=policy['deadline_days'])
    p.update(id=eid, schema='thesis-episode-v1', policy_seal=seal(policy), original_stop=stop,
             deadline=deadline.isoformat(), events=ordered, milestones=[], fib=None,
             reviews=[], unknown_at=None, terminal_at=None, sequence_status='await_test',
             entry_closed_at=None, entry_close_reason=None)
    p['source_status'] = 'known' if stop is not None and base['source_status'] == 'known' else 'unknown'

    def intent(entry, at, expires, citations):
        row = {'episode_id': eid, 'entry': entry, 'decision_time': at.isoformat(),
               'expires_at': expires.isoformat(), 'original_stop': stop, 'input_ids': citations}
        return dict(row, id='intent:'+seal(row))

    p['entry_intents'] = {'simple': intent('simple', t0, t0+pd.Timedelta('15min'),
                                         [origin['id'], parent['id']]) if stop else None, 'thesis': None}
    if p['source_status'] != 'known':
        p['entry_intents']['simple'] = None
        p['unknown_at'] = t0.isoformat()
    last_at, support, stage = t0, None, 'test'
    hours = {'test': 24, 'strength': 48, 'last_support': 72}
    known = {(e['timeframe'], clock(e['start'])) for e in ordered if e['kind'] == 'candle' and e['status'] == 'known'}

    def expire_complete_window(at, include_boundary):
        nonlocal stage
        if stage not in hours and stage != 'trigger':
            return
        tf = '1min' if stage == 'trigger' else '1h'
        limit = last_at+pd.Timedelta('15min') if stage == 'trigger' else t0+pd.Timedelta(hours=hours[stage])
        if at < limit or (at == limit and not include_boundary and stage != 'trigger'):
            return
        delta = pd.Timedelta(tf)
        # Minute confirmation at the expiry itself is ineligible.
        last_start = limit-delta*(2 if stage == 'trigger' else 1)
        required = pd.date_range(last_at, last_start, freq=tf)
        if len(required) and all((tf, s) in known for s in required):
            p['entry_closed_at'], p['entry_close_reason'] = limit.isoformat(), 'sequence_expired'
            stage, p['sequence_status'] = 'expired', 'expired'

    for e in ordered:
        t = clock(e['available_at'])
        if t > deadline:
            break
        expire_complete_window(t, False)
        if e['status'] == 'unknown' and (e['timeframe'] in ('1h', '4h') or stage == 'trigger'):
            p['unknown_at'] = p['unknown_at'] or t.isoformat()
        if e['kind'] == 'candle' and e['timeframe'] == '4h' and e['status'] == 'known':
            if e['payload']['close'] < lo:
                p['terminal_at'] = p['terminal_at'] or t.isoformat()
        if p['unknown_at'] or p['terminal_at'] or stage in ('done', 'failed', 'expired'):
            continue
        if e['kind'] != 'candle' or e['status'] != 'known' or clock(e['start']) < last_at:
            continue
        candle = e['payload']
        ok = False
        if stage in hours:
            if t > t0+pd.Timedelta(hours=hours[stage]):
                stage, p['sequence_status'] = 'expired', 'expired'
                continue
            if e['timeframe'] != '1h':
                continue
            if stage == 'test':
                ok = c0['low'] < candle['low'] <= lo and candle['close'] > lo
            elif stage == 'strength':
                ok = candle['close'] > c0['high']
            elif candle['low'] <= c0['high']:
                if candle['close'] <= c0['high']:
                    p['entry_closed_at'], p['entry_close_reason'] = t.isoformat(), 'failed_touch'
                    stage, p['sequence_status'] = 'failed', 'failed_touch'
                    continue
                ok = True
        elif stage == 'trigger':
            if t >= last_at+pd.Timedelta('15min'):
                stage, p['sequence_status'] = 'expired', 'expired'
                continue
            ok = e['timeframe'] == '1min' and candle['close'] > support['payload']['high']
        if not ok:
            expire_complete_window(t, True)
            continue
        milestone = {'kind': stage, 'available_at': t.isoformat(), 'evidence_id': e['id'],
                     'episode_id': eid}
        milestone['id'] = 'milestone:'+seal(milestone)
        p['milestones'].append(milestone)
        if stage == 'strength':
            a, b = c0['low'], candle['high']
            p['fib'] = {'a': a, 'b': b, 'anchor_ids': [origin['id'], e['id']],
                        'available_at': t.isoformat(), 'definition_status': 'project_hypothesis',
                        'levels': {str(r): a+r*(b-a) for r in policy['fib_ratios']}}
            delta = t-t0
            for r in policy['fib_time_ratios']:
                p['reviews'].append({'at': (t+delta*r).ceil('h').isoformat(),
                                     'families': ['fib_time'], 'input_ids': [milestone['id']]})
        elif stage == 'last_support':
            support = e
        elif stage == 'trigger':
            p['entry_intents']['thesis'] = intent('thesis', t, last_at+pd.Timedelta('15min'),
                                                [m['id'] for m in p['milestones']])
        stage = {'test': 'strength', 'strength': 'last_support', 'last_support': 'trigger', 'trigger': 'done'}[stage]
        last_at, p['sequence_status'] = t, stage
    if p['terminal_at']:
        p['sequence_status'] = 'invalidated'
    elif p['unknown_at']:
        p['sequence_status'] = 'unknown'
    for hours in policy['gann_hours']:
        p['reviews'].append({'at': (t0+pd.Timedelta(hours=hours)).isoformat(),
                             'families': ['gann'], 'input_ids': [origin['id']]})
    merged = {}
    for r in p['reviews']:
        if clock(r['at']) > deadline:
            continue
        old = merged.setdefault(r['at'], {'at': r['at'], 'families': [], 'input_ids': []})
        old['families'].extend(r['families'])
        old['input_ids'].extend(r['input_ids'])
    p['reviews'] = [dict(r, id='review:'+seal(r)) for _, r in sorted(merged.items())]
    return signed(p)
