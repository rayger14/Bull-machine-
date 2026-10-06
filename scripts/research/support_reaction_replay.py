"""Isolated fixed-policy comparisons through an explicitly tagged legacy runtime.

Synthetic/engineering use does not grant clearance for natural-history economics.
"""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from scripts.research.event_walkforward import split_events
from scripts.research.support_reaction import assess, candle, enough_room, origin_record, policy, verify_record
from scripts.research.thesis_contract import clock, positive, protocol, seal, signed, verify_packet
from scripts.research.thesis_execution import _unknown, replay_book

BLOCKS = ('2024-01-01T00:00Z', '2024-09-01T00:00Z', '2025-05-01T00:00Z',
          '2026-01-01T00:00Z', '2026-08-24T00:00Z')


def folds(origins):
    events = [dict(id=o['id'], decision_time=o['origin']['available_at'],
                   label_end=(clock(o['origin']['available_at'])+pd.Timedelta('7d')).isoformat()) for o in origins]
    return [split_events(events, test_start=a, test_end=b) for a, b in zip(BLOCKS[1:-1], BLOCKS[2:])]


def _guard(record, frame, execution, cutoff):
    d = record['decisions']['B']; raw = record['origin']
    at, expiry = clock(d['at']), clock(d['expires_at'])
    ready = (at+pd.Timedelta(seconds=execution['delay_seconds'])).ceil('min')
    lo, hi, stop = raw['parent']['range_low'], raw['parent']['range_high'], raw['original_stop']
    def result(status, reason, t, observations):
        return signed(dict(status=status, reason=reason, at=t.isoformat(),
                           ready_at=ready.isoformat(), evidence=observations,
                           episode_id=raw['id'], decision_id=d['id']))
    for t in pd.date_range(at, min(ready, cutoff, expiry-pd.Timedelta('1min')), freq='min'):
        obs = []
        if t == t.floor('4h'):
            parent = candle(frame, t-pd.Timedelta('4h'), '4h', raw['stream_id']); obs.append(parent)
            if parent['status'] != 'known': return result('unknown', 'unknown_parent_price', t, obs)
            if parent['payload']['close'] < lo: return result('cancelled', 'parent_invalidated', t, obs)
        opening = float(frame.at[t, 'open']) if t in frame.index else None
        if not positive(opening): opening = None
        obs.append(dict(kind='opening', available_at=t.isoformat(), price=opening))
        if opening is None or not positive(opening):
            return result('unknown', 'unknown_admission_price', t, obs)
        if t > at:
            prev = candle(frame, t-pd.Timedelta('1min'), '1min', raw['stream_id']); obs.append(prev)
            if prev['status'] != 'known': return result('unknown', 'unknown_admission_price', t, obs)
            if prev['payload']['low'] <= stop or opening <= stop:
                return result('cancelled', 'pending_stop_touch', t, obs)
            if prev['payload']['low'] <= d['support_low']:
                return result('cancelled', 'admission_support_failed', t, obs)
        if opening <= stop: return result('cancelled', 'pending_stop_touch', t, obs)
        if opening <= d['support_low']: return result('cancelled', 'admission_support_failed', t, obs)
        if t == ready:
            if not lo < opening < hi or not enough_room(opening, stop, hi):
                return result('cancelled', 'admission_insufficient_room', t, obs)
            return result('allowed', 'admission_valid', t, obs)
    return result('expired' if cutoff >= expiry else 'pending',
                  'entry_expired' if cutoff >= expiry else 'await_admission', min(cutoff, expiry), [])


def _transport(record, packet, arm, guard):
    """A legacy *execution envelope*, NOT an unchanged legacy strategy packet."""
    d = record['decisions'][arm]
    p = deepcopy(packet)
    p.update(events=[], milestones=[], reviews=[], fib=None, unknown_at=None,
             terminal_at=None, entry_closed_at=None, entry_close_reason=None,
             entry_intents={'simple': None, 'thesis': None},
             adapter=dict(schema='support-runtime-transport-v1', strategy_policy_seal=seal(policy()),
                          source_packet_seal=packet['seal'], assessment_seal=record['seal'], arm=arm,
                          runtime_policy_seal=seal(protocol()), unchanged_legacy_strategy=False))
    if d['status'] in ('intent', 'unknown'):
        intent = dict(episode_id=p['id'], entry='thesis', decision_time=d['at'],
                      expires_at=d.get('expires_at', p['deadline']), original_stop=p['original_stop'],
                      input_ids=d['citations'], decision_id=d['id'],
                      directive='entry' if d['status'] == 'intent' else 'unknown_occupancy')
        p['entry_intents']['thesis'] = dict(intent, id='support-runtime-intent:'+seal(intent))
        if d['status'] == 'unknown': p['unknown_at'] = d['at']
    elif d['status'] != 'pending':
        p['entry_closed_at'], p['entry_close_reason'] = d['at'], d['reason']
    if guard and d['status'] == 'intent':
        if guard['status'] == 'unknown': p['unknown_at'] = guard['at']
        if guard['status'] == 'cancelled': p['terminal_at'] = guard['at']
    return signed(p)


def _execute(packets, frame, execution, capacity, cutoff, final):
    """Preserve a known opening when its later completed OHLC is unknown.

    The frozen reducer accepts finite OHLC only. Neutral transport candles are
    never consumed by exposed positions: inject its own unknown transition at
    the completion clock, via a signed checkpoint, BEFORE resuming that minute.
    This adapter does not invent the missing extrema or rewrite earlier fills.
    """
    values = frame[['open', 'high', 'low', 'close']].to_numpy(float)
    valid_open = np.isfinite(values[:, 0]) & (values[:, 0] > 0)
    complete = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    complete &= (values[:, 2] <= values[:, [0, 3]].min(axis=1)) & (values[:, 1] >= values[:, [0, 3]].max(axis=1))
    transport = frame.loc[valid_open].copy()
    bad = frame.index[valid_open & ~complete]
    for t in bad:
        transport.loc[t, ['high', 'low', 'close']] = transport.at[t, 'open']
    start = min(clock(p['origin']['available_at']) for p in packets)
    checkpoint, gaps = None, []
    for t in bad:
        if t < start or t+pd.Timedelta('1min') > cutoff:
            continue
        book = replay_book(packets, transport, 'thesis', 'fixed', execution=execution,
                           capacity=capacity, until=t, checkpoint=checkpoint)
        checkpoint = deepcopy(book['checkpoint']); state = checkpoint['state']
        available = t+pd.Timedelta('1min'); affected = []
        for eid, row in state['rows'].items():
            if row['status'] in ('open', 'pending'):
                reason = 'unknown_preentry_structure' if row['status'] == 'pending' else 'missing_execution_minute'
                _unknown(state, row, state['positions'].get(eid), available, reason)
                affected.append(eid)
        gaps.append(dict(start=t.isoformat(), available_at=available.isoformat(), opening=float(transport.at[t, 'open']),
                         status='unknown_completed_ohlc', affected_episode_ids=affected))
        checkpoint = signed(checkpoint)
    book = replay_book(packets, transport, 'thesis', 'fixed', execution=execution, capacity=capacity,
                       until=None if final else cutoff, checkpoint=checkpoint)
    return book, gaps


def replay(records, old_packets, minutes, *, arm, execution=None, capacity=False, as_of=None):
    if arm not in ('A', 'B', 'C'):
        raise ValueError('invalid support arm')
    execution = dict(protocol()['primary'] if execution is None else execution)
    if (set(execution) != {'delay_seconds', 'fee_bps', 'funding_bps'}
            or not positive(execution['delay_seconds'])
            or any(isinstance(execution[k], bool) or not isinstance(execution[k], (int, float))
                   or not math.isfinite(execution[k]) or execution[k] < 0 for k in ('fee_bps', 'funding_bps'))):
        raise ValueError('invalid execution assumptions')
    if not records or len(records) != len(old_packets):
        raise ValueError('one assessment per raw episode required')
    by_id = {p['id']: p for p in old_packets}
    if len(by_id) != len(old_packets) or {r['origin']['id'] for r in records} != set(by_id):
        raise ValueError('raw population identity mismatch')
    cutoff = clock(as_of) if as_of is not None else min(minutes.index[-1]+pd.Timedelta('1min'),
                       max(clock(p['deadline']) for p in old_packets)+pd.Timedelta('1min'))
    frame = minutes.loc[minutes.index <= cutoff].copy()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('ordered aware unique source required')
    if cutoff in frame.index:
        # At the current clock only the opening exists. Its H/L/C is future data.
        frame.loc[cutoff, ['high', 'low', 'close']] = frame.at[cutoff, 'open']
    packets, guards, decisions = [], {}, {}
    for record in records:
        verify_record(record)
        p = by_id[record['origin']['id']]; verify_packet(p)
        if origin_record(p) != record['origin']:
            raise ValueError('source origin binding mismatch')
        if arm == 'A':
            packets.append(p)
            continue
        d = record['decisions'][arm]; decisions[p['id']] = d
        check_at = min(clock(record['observed_through']), cutoff)
        if check_at >= clock(p['origin']['available_at']):
            rebuilt = assess(record['origin'], minutes, as_of=check_at)['decisions'][arm]
            if (clock(d['at']) <= check_at or rebuilt['status'] != 'pending') and rebuilt != d:
                raise ValueError('assessment/source prefix mismatch')
        guard = _guard(record, frame, execution, cutoff) if d['status'] == 'intent' and clock(d['at']) <= cutoff else None
        if guard: guards[p['id']] = guard
        packets.append(_transport(record, p, arm, guard))
    runtime, gaps = _execute(packets, frame, execution, capacity, cutoff, as_of is None)
    rows = deepcopy(runtime['rows'])
    for row in rows:
        d, guard = decisions.get(row['episode_id']), guards.get(row['episode_id'])
        if d and clock(d['at']) <= cutoff and row['status'] == 'watch' and d['status'] in ('rejected', 'expired'):
            row.update(status='no_entry', reason=d['reason'], net=0.)
        if guard and clock(guard['at']) <= cutoff:
            if row['reason'] == 'preentry_invalidation' and guard['status'] == 'cancelled':
                row['reason'] = guard['reason']
            if row['reason'] == 'unknown_preentry_structure' and guard['status'] == 'unknown':
                row['reason'] = guard['reason']
    return signed(dict(schema='support-book-v1', arm=arm, policy_seal=seal(policy()),
                       runtime_policy_seal=seal(protocol()), runtime=runtime, admission=guards,
                       rows=rows, capacity=capacity, source_ids=sorted(by_id), execution_evidence_gaps=gaps,
                       execution_authorized=False, economic_clearance=False))


def _totals(book, origins):
    rows = book['rows']; complete = all(r['net'] is not None for r in rows)
    filled = [r['episode_id'] for r in rows if r['status'] == 'closed']
    months = {clock(origins[i]['origin']['available_at']).strftime('%Y-%m') for i in filled}
    subtotal = math.fsum(r['net'] for r in rows if r['net'] is not None)
    return dict(raw_episodes=len(rows), closed_fills=len(filled), origin_months=len(months),
                unknown_or_unfinished=sum(r['net'] is None for r in rows), known_subtotal=subtotal,
                complete_net=subtotal if complete else None, research_floor_met=len(filled) >= 50 and len(months) >= 12,
                edge_demonstrated=False)


def _pairs(books):
    pairs = {}
    for a, b in ('AB', 'BC'):
        match = {r['episode_id']: r for r in books[b]['rows']}
        counts = dict(complete_pairs=0, unknown_pairs=0, winners_preserved=0, winners_missed=0,
                      losers_avoided=0, losers_still_taken=0, net_change=0.)
        for x in books[a]['rows']:
            y = match[x['episode_id']]
            if x['net'] is None or y['net'] is None:
                counts['unknown_pairs'] += 1; continue
            counts['complete_pairs'] += 1; counts['net_change'] += y['net']-x['net']
            if x['net'] > 0: counts['winners_preserved' if y['net'] > 0 else 'winners_missed'] += 1
            if x['net'] < 0: counts['losers_avoided' if y['status'] in ('no_entry', 'busy') else 'losers_still_taken'] += 1
        pairs[a+'_to_'+b] = counts
    return pairs


def compare(records, old_packets, minutes, *, execution=None, as_of=None):
    books = {a: replay(records, old_packets, minutes, arm=a, execution=execution, as_of=as_of) for a in 'ABC'}
    occupied = {a: replay(records, old_packets, minutes, arm=a, execution=execution, as_of=as_of, capacity=True) for a in 'ABC'}
    origins = {r['origin']['id']: r['origin'] for r in records}
    splits = folds(list(origins.values())); reports = []
    for split in splits:
        selected = set(split['test_ids'])
        subset = {a: {'rows': [r for r in b['rows'] if r['episode_id'] in selected]} for a, b in books.items()}
        rsub = [r for r in records if r['origin']['id'] in selected]
        psub = [p for p in old_packets if p['id'] in selected]
        occupied_subset = ({a: replay(rsub, psub, minutes, arm=a, execution=execution, as_of=as_of, capacity=True) for a in 'ABC'}
                           if selected else {a: {'rows': []} for a in 'ABC'})
        reports.append(dict(split=split, test_raw_episodes=len(selected),
                            arms={a: _totals(b, origins) for a, b in subset.items()}, pairs=_pairs(subset),
                            occupied_arms={a: _totals(b, origins) for a, b in occupied_subset.items()},
                            occupied_books=occupied_subset, occupied_boundary='reset_to_empty_at_test_window',
                            fitted=False, pristine_holdout=False))
    values = list(origins.values())
    overlaps = sum(clock(a['origin']['available_at']) <= clock(b['origin']['available_at']) < clock(a['deadline'])
                   or clock(b['origin']['available_at']) <= clock(a['origin']['available_at']) < clock(b['deadline'])
                   for i, a in enumerate(values) for b in values[i+1:])
    return signed(dict(schema='support-comparison-v1', raw_episodes=len(records),
                       arms={a: _totals(b, origins) for a, b in books.items()}, pairs=_pairs(books),
                       attribution_books=books, occupied_books=occupied,
                       folds=splits, chronological_reports=reports, dependence=dict(overlapping_origin_pairs=overlaps,
                       parent_lineages=len({o['parent']['lineage_id'] for o in values})),
                       capacity_free_is_portfolio=False, fitted=False, pristine_holdout=False,
                       edge_demonstrated=False, execution_authorized=False))
