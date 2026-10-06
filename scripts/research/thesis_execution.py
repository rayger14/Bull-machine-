"""One deterministic minute-clock reducer for fixed/adaptive offline books.

OHLC market orders are all-or-none. Reductions model partial positions, not
exchange partial-fill liquidity. No production broker or trading API is imported.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np
import pandas as pd

from scripts.research.thesis_contract import MINUTE, clock, positive, protocol, seal, signed, verify, verify_packet


def _validate_minutes(frame):
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or not index.is_unique
            or not index.is_monotonic_increasing or not (index == index.floor('min')).all() or frame.empty):
        raise ValueError('ordered unique aware minute index required')
    values = frame[['open', 'high', 'low', 'close']].to_numpy(float)
    if (not np.isfinite(values).all() or (values <= 0).any()
            or (values[:, 2] > values[:, [0, 3]].min(axis=1)).any()
            or (values[:, 1] < values[:, [0, 3]].max(axis=1)).any()):
        raise ValueError('invalid minute geometry')


def _prefix(frame, cursor):
    # Only consumed full bars plus current opening: never hash its future H/L/C.
    t = clock(cursor)
    cut = frame.loc[frame.index < t, ['open', 'high', 'low', 'close']]
    digest = hashlib.sha256(pd.util.hash_pandas_object(cut, index=True).values.tobytes())
    digest.update(seal({'at': t.isoformat(), 'open': float(frame.at[t, 'open']) if t in frame.index else None}).encode())
    return digest.hexdigest()


def _action(pos, kind, at, **values):
    pos['actions'].append({'kind': kind, 'at': at.isoformat(), **values})


def _unknown(state, row, pos, at, reason):
    row.update(status='unknown', reason=reason, net=None)
    if pos is not None:
        pos.update(status='unknown', net=None, unknown_at=at.isoformat())
    if state['capacity']:
        state['unknown_occupancy'] = True


def _close(pos, row, at, price, quantity, reason, fee, order_id=None, destinations=()):
    quantity = min(quantity, pos['remaining_qty'])
    cost = quantity*price*fee
    gross = quantity*(price-pos['entry_price'])
    pos['remaining_qty'] = max(0., pos['remaining_qty']-quantity)
    pos['fees'] += cost
    pos['gross'] += gross
    for d in destinations:
        pos['destinations'][d] = 'executed'
    flat = pos['remaining_qty'] <= pos['original_qty']*1e-12
    pos['cashflows'].append({'kind': 'exit' if flat else 'partial', 'at': at.isoformat(),
                            'price': price, 'quantity': quantity, 'fee': cost, 'funding': 0.,
                            'gross_pnl': gross, 'remaining_qty': pos['remaining_qty'],
                            'reason': reason, 'order_id': order_id})
    if flat:
        pos['remaining_qty'] = 0.
        pos.update(status='closed', exit_time=at.isoformat(), net=pos['gross']-pos['fees']-pos['funding'])
        row.update(status='closed', reason=reason, net=pos['net'])
        for order in pos['orders']:
            if order['status'] == 'pending' and order['id'] != order_id:
                order['status'] = 'cancelled'
                _action(pos, 'order_cancelled', at, order_id=order['id'], reason='flat')
                for d in order.get('destinations', []):
                    pos['destinations'][d] = 'cancelled'


def _schedule(pos, kind, at, execution, evidence_ids, **values):
    order = {'kind': kind, 'decision_time': at.isoformat(),
             'ready_at': (at+pd.Timedelta(seconds=execution['delay_seconds'])).ceil('min').isoformat(),
             'evidence_ids': evidence_ids, **values}
    order['id'] = 'order:'+seal({'position_id': pos['id'], **order})
    order['status'] = 'pending'
    if any(o['id'] == order['id'] for o in pos['orders']):
        return
    pos['orders'].append(order)
    _action(pos, kind+'_scheduled', at, order_id=order['id'], ready_at=order['ready_at'], **values)


def _manage(pos, packet, at, events, reviews, execution, disabled):
    for e in events:
        if clock(e['available_at']) <= clock(pos['entry_time']) or e['status'] != 'known':
            continue
        if e['kind'] == 'candle' and e['timeframe'] == '1h':
            price = e['payload']['close']
            pos['last_hour_close'] = price
            destinations = []
            if price >= packet['parent']['range_high'] and pos['destinations']['range'] == 'untriggered':
                destinations.append('range')
            fib = packet['fib']
            if ('fib_price' not in disabled and fib and clock(fib['available_at']) <= at
                    and price >= fib['levels']['1.618'] and pos['destinations']['fib'] == 'untriggered'):
                destinations.append('fib')
            if destinations:
                for d in destinations:
                    pos['destinations'][d] = 'scheduled'
                _schedule(pos, 'reduce', at, execution, [e['id']],
                          fraction=.25*len(destinations), destinations=destinations, reason='destination')
        if e['kind'] == 'pivot_low' and clock(e['payload']['center_start']) >= clock(packet['origin']['available_at']):
            proposed = e['payload']['price']-.1*packet['atr4h']
            ceiling = max([pos['effective_stop']]+[o['stop'] for o in pos['orders']
                          if o['kind'] == 'replace_stop' and o['status'] == 'pending'])
            if positive(proposed) and ceiling < proposed < e['payload']['decision_close']:
                _schedule(pos, 'replace_stop', at, execution, [e['id']], stop=proposed)
    for review in reviews:
        if at <= clock(pos['entry_time']) or not (set(review['families'])-set(disabled)):
            continue
        progress = any(clock(pos['review_boundary']) < clock(m['available_at']) <= at for m in packet['milestones'])
        leave = not progress and pos['last_hour_close'] <= pos['entry_price']
        _action(pos, 'clock_review', at, review_id=review['id'], families=review['families'],
                action='exit' if leave else 'hold', progress=progress)
        pos['review_boundary'] = at.isoformat()
        if leave and not any(o['kind'] == 'close' and o['status'] == 'pending' for o in pos['orders']):
            _schedule(pos, 'close', at, execution, [review['id']], reason='no_progress')


def replay_book(packets, minutes, entry, management, execution=None, checkpoint=None,
                until=None, capacity=True, disabled=()):
    """Replay one isolated book; explicit ``until`` returns resumable open state.

    Resume supplies the same packets and consumed minute prefix, optionally with
    additional future minutes. ``capacity=False`` is overlapping attribution only.
    """
    if entry not in ('simple', 'thesis') or management not in ('fixed', 'adaptive') or not isinstance(capacity, bool):
        raise ValueError('invalid arm')
    if set(disabled)-{'fib_price', 'fib_time', 'gann'} or (disabled and management != 'adaptive'):
        raise ValueError('invalid ablation')
    _validate_minutes(minutes)
    policy = protocol()
    execution = dict(policy['primary'] if execution is None else execution)
    if set(execution) != {'delay_seconds', 'fee_bps', 'funding_bps'}:
        raise ValueError('execution contract fields required')
    if (not positive(execution['delay_seconds']) or any(isinstance(execution[k], bool)
            or not isinstance(execution[k], (int, float)) or not math.isfinite(execution[k])
            or execution[k] < 0 for k in ('fee_bps', 'funding_bps'))):
        raise ValueError('invalid execution settings')
    for p in packets:
        verify_packet(p)
    by_id = {p['id']: p for p in packets}
    if len(by_id) != len(packets):
        raise ValueError('duplicate episode')
    binding = seal({'packets': sorted((p['id'], p['seal']) for p in packets),
                    'policy': seal(policy), 'entry': entry, 'management': management,
                    'execution': execution, 'capacity': capacity, 'disabled': sorted(disabled)})
    if checkpoint:
        verify(checkpoint)
        if checkpoint['binding'] != binding:
            raise ValueError('checkpoint binding mismatch')
        if checkpoint['prefix_hash'] != _prefix(minutes, checkpoint['state']['cursor']):
            raise ValueError('checkpoint prefix mismatch')
        state = deepcopy(checkpoint['state'])
        start = clock(state['cursor'])+MINUTE
    else:
        state = {'cursor': None, 'capacity': capacity, 'unknown_occupancy': False,
                 'rows': {p['id']: {'episode_id': p['id'], 'status': 'watch', 'reason': None, 'net': None} for p in packets},
                 'positions': {}, 'entry_tape': [], 'peak': 0., 'drawdown': 0., 'last_mark': 0.}
        start = min((clock(p['origin']['available_at']) for p in packets), default=minutes.index[0])
    end = clock(until) if until is not None else minutes.index[-1]+MINUTE
    if end < start-MINUTE or end > minutes.index[-1]+MINUTE:
        raise ValueError('replay clock outside minute coverage')
    prices = {t: tuple(float(x) for x in row) for t, row in zip(minutes.index, minutes[['open', 'high', 'low', 'close']].to_numpy())}
    schedule = {}
    for p in packets:
        for e in p['events']:
            schedule.setdefault(clock(e['available_at']), {}).setdefault(p['id'], [[], []])[0].append(e)
        for r in p['reviews']:
            schedule.setdefault(clock(r['at']), {}).setdefault(p['id'], [[], []])[1].append(r)
    fee = execution['fee_bps']/10000.
    ordered_packets = sorted(packets, key=lambda p: (clock((p['entry_intents'][entry] or {}).get(
                             'decision_time', p['origin']['available_at'])), p['id']))
    for t in pd.date_range(start, end, freq='min'):
        prev, now = prices.get(t-MINUTE), prices.get(t)
        # Existing positions: the old effective orders govern the prior bar and gap.
        for eid, pos in state['positions'].items():
            if pos['status'] != 'open':
                continue
            p, row = by_id[eid], state['rows'][eid]
            if prev is None:
                _unknown(state, row, pos, t, 'missing_execution_minute')
                continue
            pos['exposure_minutes'] += 1
            if t.hour in (0, 8, 16) and t.minute == 0 and t > clock(pos['entry_time']):
                charge = pos['remaining_qty']*pos['entry_price']*execution['funding_bps']/10000.
                pos['funding'] += charge
                pos['cashflows'].append({'kind': 'funding', 'at': t.isoformat(), 'price': pos['entry_price'],
                                        'quantity': pos['remaining_qty'], 'fee': 0., 'funding': charge,
                                        'gross_pnl': 0., 'remaining_qty': pos['remaining_qty'], 'reason': 'adverse_stress', 'order_id': None})
            if prev[2] <= pos['effective_stop']:
                _close(pos, row, t, pos['effective_stop'], pos['remaining_qty'], 'stop', fee)
                continue
            if management == 'fixed' and prev[1] >= pos['target']:
                _close(pos, row, t, pos['target'], pos['remaining_qty'], 'target', fee)
                continue
            if now is None:
                _unknown(state, row, pos, t, 'missing_execution_open')
                continue
            if now[0] <= pos['effective_stop']:
                _close(pos, row, t, now[0], pos['remaining_qty'], 'stop_gap', fee)
                continue
            if management == 'fixed' and now[0] >= pos['target']:
                _close(pos, row, t, pos['target'], pos['remaining_qty'], 'target_gap', fee)
                continue
            if management == 'adaptive':
                if p['unknown_at'] and clock(p['unknown_at']) <= t:
                    _unknown(state, row, pos, t, 'unknown_structure')
                    continue
                if p['terminal_at'] and clock(p['terminal_at']) == t:
                    _schedule(pos, 'close', t, execution, [e['id'] for e in p['events'] if e['available_at'] == t.isoformat()], reason='invalidation')
                events, reviews = schedule.get(t, {}).get(eid, [[], []])
                _manage(pos, p, t, events, reviews, execution, disabled)
            due = [o for o in pos['orders'] if o['status'] == 'pending' and clock(o['ready_at']) <= t]
            full = sorted([o for o in due if o['kind'] == 'close'], key=lambda o: (o['reason'] != 'invalidation', o['id']))
            if full and full[0]['reason'] == 'invalidation':
                o = full[0]
                o['status'] = 'executed'
                _close(pos, row, t, now[0], pos['remaining_qty'], o['reason'], fee, o['id'])
            elif t >= clock(p['deadline']):
                _close(pos, row, t, now[0], pos['remaining_qty'], 'deadline', fee)
            elif full:
                o = full[0]
                o['status'] = 'executed'
                _close(pos, row, t, now[0], pos['remaining_qty'], o['reason'], fee, o['id'])
            if pos['status'] != 'open':
                continue
            for o in due:
                if o['kind'] != 'reduce' or o['status'] != 'pending':
                    continue
                o['status'] = 'executed'
                _close(pos, row, t, now[0], pos['original_qty']*o['fraction'], o['reason'], fee, o['id'], o['destinations'])
            for o in due:
                if pos['status'] != 'open' or o['kind'] != 'replace_stop' or o['status'] != 'pending':
                    continue
                o['status'] = 'executed'
                pos['effective_stop'] = max(pos['effective_stop'], o['stop'])
                _action(pos, 'stop_effective', t, order_id=o['id'], stop=pos['effective_stop'])
            if pos['status'] == 'open' and now[0] <= pos['effective_stop']:
                _close(pos, row, t, now[0], pos['remaining_qty'], 'new_stop_marketable', fee)
        # Watchers do not reserve; pending entries do. Cancellation is before fill.
        for p in ordered_packets:
            eid, intent = p['id'], p['entry_intents'][entry]
            row = state['rows'][eid]
            if row['status'] not in ('watch', 'pending'):
                continue
            t0 = clock(p['origin']['available_at'])
            if t < t0:
                continue
            if p['source_status'] != 'known':
                _unknown(state, row, None, t, 'unknown_initial_risk')
                continue
            if intent is None:
                continue
            decision = clock(intent['decision_time'])
            if t < decision:
                continue
            if row['status'] == 'watch':
                if state['unknown_occupancy'] and capacity:
                    _unknown(state, row, None, t, 'unknown_prior_occupancy')
                    continue
                if capacity and any(r['status'] in ('pending', 'open') for r in state['rows'].values()):
                    row.update(status='busy', reason='occupied_at_decision', net=0.)
                    continue
                row.update(status='pending', ready_at=(decision+pd.Timedelta(seconds=execution['delay_seconds'])).ceil('min').isoformat())
            if p['unknown_at'] and clock(p['unknown_at']) <= t:
                _unknown(state, row, None, t, 'unknown_preentry_structure')
                continue
            if p['terminal_at'] and clock(p['terminal_at']) <= t:
                row.update(status='no_entry', reason='preentry_invalidation', net=0.)
                continue
            if t >= clock(intent['expires_at']):
                row.update(status='no_entry', reason='expired', net=0.)
                continue
            if now is None or (t > decision and prev is None):
                _unknown(state, row, None, t, 'missing_pending_minute')
                continue
            if now[0] <= p['original_stop'] or (t > decision and prev[2] <= p['original_stop']):
                row.update(status='no_entry', reason='pending_stop_touch', net=0.)
                continue
            if t < clock(row['ready_at']):
                continue
            price, stop = now[0], p['original_stop']
            qty = min(policy['risk_budget']/((price-stop)+fee*(price+stop)), policy['notional_cap']/price)
            initial_fee = qty*price*fee
            pos = {'id': eid, 'status': 'open', 'entry_time': t.isoformat(), 'entry_price': price,
                   'original_qty': qty, 'remaining_qty': qty, 'original_stop': stop, 'effective_stop': stop,
                   'target': price+policy['target_r']*(price-stop), 'fees': initial_fee, 'funding': 0.,
                   'gross': 0., 'net': None, 'orders': [], 'actions': [], 'exposure_minutes': 0,
                   'destinations': {'range': 'untriggered', 'fib': 'untriggered'},
                   'last_hour_close': price, 'review_boundary': t.isoformat(),
                   'cashflows': [{'kind': 'entry', 'at': t.isoformat(), 'price': price, 'quantity': qty,
                                 'fee': initial_fee, 'funding': 0., 'gross_pnl': 0., 'remaining_qty': qty,
                                 'reason': entry, 'order_id': intent['id']}]}
            state['positions'][eid] = pos
            state['entry_tape'].append({'episode_id': eid, 'at': t.isoformat(), 'price': price, 'quantity': qty, 'stop': stop})
            row.update(status='open', reason=None)
        if now is not None and not any(r['status'] == 'unknown' for r in state['rows'].values()):
            equity = math.fsum(p['gross']-p['fees']-p['funding']+p['remaining_qty']*(now[0]-p['entry_price'])
                              for p in state['positions'].values())
            state['last_mark'] = equity
            state['peak'] = max(state['peak'], equity)
            state['drawdown'] = max(state['drawdown'], state['peak']-equity)
        state['cursor'] = t.isoformat()
    # A checkpoint is the unfinalized state; no assumed end-of-data liquidation.
    checkpoint_out = signed({'binding': binding, 'prefix_hash': _prefix(minutes, state['cursor']), 'state': deepcopy(state)})
    if until is None:
        for eid, row in state['rows'].items():
            if row['status'] in ('pending', 'open'):
                _unknown(state, row, state['positions'].get(eid), end, 'incomplete_outcome_tail')
            elif row['status'] == 'watch':
                p = by_id[eid]
                closures = [clock(x) for x in (p.get('entry_closed_at'), p['terminal_at']) if x]
                closed_at = min(closures, default=None)
                if closed_at and closed_at <= end and (not p['unknown_at'] or closed_at < clock(p['unknown_at'])):
                    row.update(status='no_entry', reason=p.get('entry_close_reason') or 'invalidation', net=0.)
                elif p['unknown_at'] and clock(p['unknown_at']) <= end:
                    row.update(status='unknown', reason='unknown_sequence', net=None)
                else:
                    row.update(status='unknown', reason='unfinished_entry_path', net=None)
    result = {'schema': 'thesis-book-v1', 'entry': entry, 'management': management,
              'capacity': capacity, 'disabled': list(disabled), 'execution': execution,
              'rows': list(state['rows'].values()), 'positions': state['positions'],
              'entry_tape': state['entry_tape'], 'unknown_occupancy': state['unknown_occupancy'],
              'drawdown': state['drawdown'], 'mark_basis': 'minute_open_net_incurred_costs',
              'checkpoint': checkpoint_out, 'execution_authorized': False}
    return signed(result)
