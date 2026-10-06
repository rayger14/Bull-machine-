"""Offline long brackets with fixed intended risk; never a live execution adapter.

OHLC barrier hits are booked at the bar's availability/close, conservatively
stop-first. Settlement at that same modeled exit clock is charged before exit.
This is an explicit minute-data convention, not sub-minute execution evidence.
"""
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from scripts.research.r3_census import validated_minutes
from scripts.research.study_contract import finite_number, utc_minute

MINUTE = pd.Timedelta('1min')


class _MinuteLookup:
    """Compact array lookup: do not inflate full history into millions of dicts."""
    def __init__(self, frame):
        self.times = frame.index.asi8
        self.values = frame.to_numpy(dtype=float)
        self.contiguous = bool(len(frame) < 2 or np.all(np.diff(self.times) == MINUTE.value))

    def position(self, at):
        if not len(self.times):
            return None
        if self.contiguous:
            index = int((at.value - self.times[0]) // MINUTE.value)
        else:
            index = int(np.searchsorted(self.times, at.value))
        return index if 0 <= index < len(self.times) and self.times[index] == at.value else None

    def get(self, at):
        index = self.position(at)
        return dict(zip(('open', 'high', 'low', 'close', 'volume'), self.values[index])) if index is not None else None

    def opening(self, at):
        index = self.position(at)
        return float(self.values[index, 0]) if index is not None else None


def position_terms(entry, stop, cost_bps):
    entry, stop = finite_number(entry, True), finite_number(stop, True)
    cost = finite_number(cost_bps) / 10000.
    if entry <= stop or cost < 0:
        raise ValueError('long requires entry > stop > 0 and nonnegative costs')
    quantity = min(100. / ((entry - stop) + cost * entry), 50000. / entry)
    fee = cost * quantity * entry / 2.
    return {'entry_price': entry, 'stop': stop, 'target': entry + 2. * (entry - stop),
            'quantity': quantity, 'initial_risk': quantity * ((entry - stop) + cost * entry),
            'entry_fee': fee, 'exit_fee': fee}


def replay_book(minutes, opportunities, signals, *, as_of, cost_bps, delay_seconds,
                funding_mode, parent_down=(), occupied=True):
    """Replay one family/arm. Unresolved records stay null, never silently zero.

    Non-emitting opportunities need source_status='complete' and an as-of
    source_available_at receipt. Raw immutable opportunities alone cannot prove
    that a sequence finished. A required data gap invalidates occupied-book path
    thereafter; later busy/vacant decisions are not guessed.
    """
    asof = utc_minute(as_of)
    delay = finite_number(delay_seconds)
    cost = finite_number(cost_bps)
    if delay < 0 or cost < 0 or not isinstance(occupied, bool):
        raise ValueError('invalid execution policy')
    if funding_mode not in ('adverse_stress', 'zero_diagnostic'):
        raise ValueError('actual funding requires a separately qualified settlement contract')
    if not isinstance(minutes.index, pd.DatetimeIndex) or minutes.index.tz is None:
        raise ValueError('aware minute source required')
    # At asof, the current opening price may be known; its future extrema are not.
    closed = validated_minutes(minutes.loc[minutes.index < asof])
    records = _MinuteLookup(closed)
    asof_open = None
    if asof in minutes.index:
        opening = minutes.loc[asof, 'open']
        asof_open = finite_number(opening, True)
    ops, rows = {}, {}
    bindings, families = set(), set()
    for op in opportunities:
        oid = op['id']
        if oid in ops:
            raise ValueError('duplicate opportunity ID')
        if op['family'] not in ('R1', 'R3'):
            raise ValueError('unsupported family')
        origin = utc_minute(op['origin_time'])
        if origin > asof:
            continue
        ops[oid] = deepcopy(op)
        bindings.add((op.get('instrument'), op.get('data_stream_id')))
        families.add(op['family'])
        source_at = op.get('source_available_at')
        complete = (op.get('source_status') == 'complete' and source_at is not None
                    and utc_minute(source_at) <= asof)
        rows[oid] = {'opportunity_id': oid, 'status': 'not_entered' if complete else 'unknown',
                     'reason': 'no_signal' if complete else 'unqualified_source',
                     'net_pnl': 0. if complete else None, 'position': None}
    if len(bindings) > 1 or any(not all(b) for b in bindings):
        raise ValueError('foreign or missing source binding')
    if len(families) > 1:
        raise ValueError('one isolated family per book required')
    queue, uncertain, seen, arms = defaultdict(list), defaultdict(list), set(), set()
    for oid, row in rows.items():
        if row['reason'] == 'unqualified_source':
            uncertain[utc_minute(ops[oid]['origin_time'])].append(oid)
    for raw in signals:
        decision = utc_minute(raw['decision_time'])
        if decision > asof:
            continue
        oid = raw['opportunity_id']
        if oid not in ops or raw['family'] != ops[oid]['family']:
            raise ValueError('foreign signal/opportunity family')
        if oid in seen:
            raise ValueError('duplicate signal per raw opportunity')
        seen.add(oid)
        arms.add(raw.get('arm'))
        stop = finite_number(raw['stop'], True)
        expiry, deadline = utc_minute(raw['entry_expiry']), utc_minute(raw['exit_deadline'])
        if decision < utc_minute(ops[oid]['origin_time']) or expiry < decision or deadline <= decision:
            raise ValueError('invalid signal clock sequence')
        signal = dict(deepcopy(raw), decision=decision, expiry=expiry, deadline=deadline, stop=stop)
        signal['ready'] = max(decision + MINUTE, (decision + pd.Timedelta(seconds=delay)).ceil('min'))
        queue[decision].append(signal)
    if len(arms) > 1:
        raise ValueError('one independent arm per book required')
    downs = defaultdict(list)
    for event in parent_down:
        if event.get('source_break_direction') != 'down':
            continue
        at = utc_minute(event['available_at'])
        if at <= asof:
            if not event.get('pre_lineage_id'):
                raise ValueError('parent down requires pre_lineage_id')
            downs[event['pre_lineage_id']].append(at)
    for times in downs.values():
        times.sort()
    active, marks, events, blockers = {}, [], [], []
    path_unknown = False

    def event(oid, at, kind, **values):
        events.append({'opportunity_id': oid, 'available_at': at.isoformat(), 'kind': kind, **values})

    def nonentry(oid, at, reason):
        rows[oid].update(status='not_entered', reason=reason, net_pnl=0.)
        active.pop(oid, None)
        event(oid, at, 'cancel', reason=reason)

    def unknown(oid, at, reason):
        nonlocal path_unknown
        rows[oid].update(status='unknown', reason=reason, net_pnl=None)
        active.pop(oid, None)
        blockers.append({'opportunity_id': oid, 'available_at': at.isoformat(), 'reason': reason})
        if occupied:
            path_unknown = True
        event(oid, at, 'unknown', reason=reason)

    def close_position(oid, at, price, reason):
        p = rows[oid]['position']
        net = p['quantity'] * (price - p['entry_price']) - p['entry_fee'] - p['exit_fee'] - p['funding']
        p.update(exit_time=at.isoformat(), exit_price=float(price), net_r=net / p['initial_risk'])
        rows[oid].update(status='closed', reason=reason, net_pnl=net)
        active.pop(oid, None)
        event(oid, at, 'exit', reason=reason, cashflow=p['quantity'] * (price - p['entry_price']) - p['exit_fee'])

    def parent_invalid(signal, at):
        origin = utc_minute(ops[signal['opportunity_id']]['origin_time'])
        return any(origin <= t <= at for t in downs.get(signal.get('parent_lineage_id'), ()))

    start_times = list(queue) + list(uncertain)
    for at in pd.date_range(min(start_times) if start_times else asof, asof, freq='min'):
        previous_open = at - MINUTE
        previous = records.get(previous_open)
        opening = asof_open if at == asof else records.opening(at)
        # Funding precedes same-clock modeled exits. New entries at at do not pay.
        for oid in list(active):
            p = rows[oid]['position']
            if (p is not None and funding_mode == 'adverse_stress' and at.minute == 0
                    and at.hour in (0, 8, 16) and utc_minute(p['entry_time']) < at):
                charge = .0008 * p['quantity'] * p['entry_price']
                p['funding'] += charge
                event(oid, at, 'funding', cashflow=-charge)

        # Consume completed previous-bar evidence before any new order/fill.
        for oid, state in list(active.items()):
            p = rows[oid]['position']
            if p is None:
                if previous_open >= state['decision']:
                    if previous is None:
                        unknown(oid, at, 'missing_required_pending_bar')
                    elif previous['low'] <= state['stop']:
                        nonentry(oid, at, 'pending_stop_touch')
                if oid in active and parent_invalid(state, at):
                    nonentry(oid, at, 'parent_down')
            elif utc_minute(p['entry_time']) <= previous_open:
                if previous is None:
                    unknown(oid, at, 'missing_required_position_bar')
                elif previous['low'] <= p['stop']:
                    close_position(oid, at, p['stop'], 'stop')
                elif previous['high'] >= p['target']:
                    close_position(oid, at, p['target'], 'target')
                else:
                    marks.append({'opportunity_id': oid, 'available_at': at.isoformat(),
                                  'liquidation_value': p['quantity'] * (previous['close'] - p['entry_price'])
                                  - p['entry_fee'] - p['exit_fee'] - p['funding']})

        # Opening gaps and mandatory deadline exits release capacity before orders.
        for oid, state in list(active.items()):
            p = rows[oid]['position']
            if p is None:
                if at > state['expiry']:
                    nonentry(oid, at, 'entry_expired')
                elif opening is not None and opening <= state['stop']:
                    nonentry(oid, at, 'entry_gap_at_or_below_stop')
                elif at >= state['ready']:
                    if opening is None:
                        if at < asof:
                            unknown(oid, at, 'missing_required_entry_open')
                    else:
                        p = position_terms(opening, state['stop'], cost)
                        p.update(entry_time=at.isoformat(), exit_time=None, exit_price=None,
                                 funding=0., net_r=None)
                        rows[oid].update(status='open', reason='open', net_pnl=None, position=p)
                        event(oid, at, 'entry', cashflow=-p['entry_fee'])
            elif opening is None:
                if at >= state['deadline'] and at < asof:
                    unknown(oid, at, 'missing_required_deadline_open')
            elif opening <= p['stop']:
                close_position(oid, at, opening, 'stop_gap')
            elif at >= state['deadline']:
                close_position(oid, at, opening, 'deadline')
            elif opening >= p['target']:
                close_position(oid, at, p['target'], 'target_gap')

        for oid in uncertain.get(at, ()):
            unknown(oid, at, 'unqualified_source')
        for signal in sorted(queue.get(at, ()), key=lambda s: s['opportunity_id']):
            oid = signal['opportunity_id']
            if rows[oid]['reason'] == 'unqualified_source':
                continue
            elif path_unknown and occupied:
                unknown(oid, at, 'occupied_book_path_unknown')
            elif occupied and active:
                rows[oid].update(status='not_entered', reason='busy', net_pnl=0.)
                event(oid, at, 'busy')
            elif parent_invalid(signal, at):
                nonentry(oid, at, 'parent_down')
            elif opening is not None and opening <= signal['stop']:
                nonentry(oid, at, 'entry_gap_at_or_below_stop')
            else:
                active[oid] = signal
                rows[oid].update(status='pending', reason='pending', net_pnl=None)
                event(oid, at, 'pending')

    for oid in active:
        rows[oid].update(status='right_censored', reason='source_cutoff', net_pnl=None)
    unresolved = sum(row['net_pnl'] is None for row in rows.values())
    return {'schema': 'study-book-v1', 'as_of': asof.isoformat(), 'rows': list(rows.values()),
            'marks': marks, 'events': events, 'blockers': blockers, 'unresolved_opportunities': unresolved,
            'portfolio': occupied, 'scope': 'isolated_occupied_book' if occupied else 'nonportfolio_fixed_event_diagnostic',
            'cost_bps': cost, 'delay_seconds': delay, 'funding_mode': funding_mode,
            'barrier_exit_clock': 'completed_bar_close_stop_first', 'execution_authorized': False}
