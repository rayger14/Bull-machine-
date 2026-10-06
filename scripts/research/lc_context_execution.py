"""Offline LC books with causal reservations and separately modeled entry clocks.

Case-window simulation is equivalent to continuous occupancy: future releases are
used only to ask whether they have happened by a later candidate's original T.
No outcome ranks, candidate retries, shared subtype capital or live execution.
"""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from scripts.research.lc_context_contract import ARMS, MINUTE, SUBTYPES, clock, positive, protocol, seal, verify_case
from scripts.research.lc_context_controller import classify, entry_location_ok
from scripts.research.lc_context_evidence import validate_index
from scripts.research.study_execution import position_terms


def _plan(case, arm):
    if arm == 'context':
        return classify(case)
    reason = ('unknown_common_source' if case['source_status'] != 'known' else
              'unknown_risk' if case['risk_status'] != 'known' else
              'unknown_h5' if arm == 'unconditional_wait' and case['h5_status'] != 'known' else None)
    return {'action': 'none' if reason else 'immediate' if arm == 'immediate' else 'wait',
            'state': 'insufficient_evidence' if reason else 'eligible' if arm == 'immediate' else 'awaiting_confirmation',
            'scenario': arm, 'reason': reason or 'control_entry',
            'trigger_level': case['h5'] if arm == 'unconditional_wait' else None,
            'execution_authorized': False}


def _window(case, data, plan, *, arm, cost_bps, delay_seconds, funding_mode):
    t = clock(case['decision_time'])
    expiry, deadline = t+pd.Timedelta('15min'), t+pd.Timedelta('24h')
    delay = pd.Timedelta(seconds=delay_seconds)
    first_observation_open = (t+delay).ceil('min')
    ready = first_observation_open if plan['action'] == 'immediate' else None
    stop = case['stop']
    validate_index(data)
    # Values after deadline are never consulted. Do not validate future extrema
    # at the entry/deadline opening, before those observations are available.
    frame = data.loc[(data.index >= t) & (data.index <= deadline),
                     ['open', 'high', 'low', 'close', 'volume']]
    locations = {at: i for i, at in enumerate(frame.index)}
    values = frame.to_numpy(dtype=float)
    row = {'candidate_id': case['candidate_id'], 'decision_time': case['decision_time'],
           'subtype': case['subtype'], 'scenario': plan['scenario'],
           'status': 'pending', 'reason': 'pending', 'position': None, 'net_pnl': None,
           'terminal_at': None, 'decision': deepcopy(plan), 'execution_authorized': False}
    events, marks = [], []

    def event(at, kind, **kwargs):
        events.append({'candidate_id': case['candidate_id'], 'available_at': at.isoformat(),
                       'kind': kind, 'execution_authorized': False, **kwargs})

    def end(at, status, reason, price=None):
        row.update(status=status, reason=reason, terminal_at=at.isoformat())
        p = row['position']
        if status == 'closed':
            net = p['quantity']*(price-p['entry_price'])-p['entry_fee']-p['exit_fee']-p['funding']
            p.update(exit_time=at.isoformat(), exit_price=float(price), net_r=net/p['initial_risk'])
            row['net_pnl'] = net
            mark(at, price, 'exit')
        elif status == 'not_entered':
            row['net_pnl'] = 0.
        event(at, status, reason=reason, price=price)
        return row, events, marks

    def mark(at, price, kind):
        p = row['position']
        marks.append({'candidate_id': case['candidate_id'], 'available_at': at.isoformat(),
                      'kind': kind, 'liquidation_value': p['quantity']*(price-p['entry_price'])
                      -p['entry_fee']-p['exit_fee']-p['funding']})

    def completed(at):
        i = locations.get(at)
        if i is None:
            return None
        bar = values[i]
        if (not np.isfinite(bar).all() or (bar[:4] <= 0).any() or bar[4] < 0
                or bar[2] > min(bar[0], bar[3]) or bar[1] < max(bar[0], bar[3])):
            raise ValueError('invalid execution minute source')
        return bar

    event(t, 'admitted', state=plan['state'])
    for at in pd.date_range(t, deadline, freq='min'):
        p = row['position']
        if (p is not None and funding_mode == 'adverse_stress' and at.minute == 0
                and at.hour in (0, 8, 16) and clock(p['entry_time']) < at):
            charge = .0008*p['quantity']*p['entry_price']
            p['funding'] += charge
            event(at, 'funding', cashflow=-charge)
        previous_open = at-MINUTE
        if previous_open >= t:
            previous = completed(previous_open)
            if previous is None:
                return end(at, 'unknown', 'missing_required_position_bar' if p else 'missing_required_pending_bar')
            if p is None:
                if previous[2] <= stop:
                    return end(at, 'not_entered', 'pending_stop_touch')
                if (ready is None and previous_open >= first_observation_open
                        and previous[3] > plan['trigger_level']):
                    ready = max(first_observation_open, (at+delay).ceil('min'))
                    event(at, 'confirmation', observation_start=previous_open.isoformat(),
                          level=plan['trigger_level'], close=float(previous[3]), ready_at=ready.isoformat())
            elif previous_open >= clock(p['entry_time']):
                if previous[2] <= p['stop']:
                    return end(at, 'closed', 'stop', p['stop'])
                if previous[1] >= p['target']:
                    return end(at, 'closed', 'target', p['target'])
                mark(at, previous[3], 'completed_close')
        if p is None and at >= expiry:
            return end(at, 'not_entered', 'entry_expired')
        i = locations.get(at)
        opening = None if i is None else float(values[i, 0])
        if opening is None:
            return end(at, 'unknown', 'missing_required_deadline_open' if at == deadline else 'missing_required_open')
        if not positive(opening):
            raise ValueError('invalid execution opening price')
        if p is None:
            if opening <= stop:
                return end(at, 'not_entered', 'entry_gap_at_or_below_stop')
            if ready is not None and at >= ready:
                # No new 4H candle completes during an hourly T+15m entry window.
                # classify bound the as-of state; never replace its version here.
                if arm == 'context' and not entry_location_ok(plan, opening):
                    return end(at, 'not_entered', 'entry_location_changed')
                p = position_terms(opening, stop, cost_bps)
                p.update(entry_time=at.isoformat(), exit_time=None, exit_price=None, funding=0., net_r=None)
                row.update(position=p, status='open', reason='open')
                event(at, 'entry', price=opening, quantity=p['quantity'], cashflow=-p['entry_fee'])
                mark(at, opening, 'entry')
        else:
            if opening <= p['stop']:
                return end(at, 'closed', 'stop_gap', opening)
            if at >= deadline:
                return end(at, 'closed', 'deadline', opening)
            if opening >= p['target']:
                return end(at, 'closed', 'target_gap', p['target'])
            mark(at, opening, 'opening')
    raise AssertionError('finite execution deadline did not terminate')


def replay_book(cases, windows, *, arm, subtype, cost_bps, delay_seconds, funding_mode, checkpoint=None):
    policy = protocol()
    if (arm not in ARMS or subtype not in SUBTYPES or cost_bps not in policy['cost_bps']
            or delay_seconds not in policy['delay_seconds'] or funding_mode not in policy['funding_modes']):
        raise ValueError('undeclared LC book policy')
    binding = {'arm': arm, 'subtype': subtype, 'cost_bps': cost_bps,
               'delay_seconds': delay_seconds, 'funding_mode': funding_mode, 'policy_seal': seal(policy)}
    release, uncertain, last = None, None, None
    if checkpoint is not None:
        value = {k: v for k, v in checkpoint.items() if k != 'seal'}
        if checkpoint.get('seal') != seal(value) or value['binding'] != binding:
            raise ValueError('foreign or invalid checkpoint')
        release = clock(value['release_time']) if value['release_time'] else None
        uncertain = clock(value['unknown_since']) if value['unknown_since'] else None
        last = clock(value['last_candidate_time']) if value['last_candidate_time'] else None
    ids, times, sources = set(), set(), set()
    for case in cases:
        verify_case(case)
        t = clock(case['decision_time'])
        if t != t.floor('h') or case['candidate_id'] in ids or t in times:
            raise ValueError('unique hourly candidates required')
        ids.add(case['candidate_id']); times.add(t)
        sources.add((case['instrument'], case['data_stream_id']))
    if len(sources) > 1 or any(not all(s) for s in sources):
        raise ValueError('mixed or missing data stream')
    rows, events, marks = [], [], []
    for case in sorted(cases, key=lambda c: (clock(c['decision_time']), c['candidate_id'])):
        if case['subtype'] != subtype:
            continue
        t = clock(case['decision_time'])
        if last is not None and t <= last:
            raise ValueError('checkpoint requires later original candidate times')
        last = t
        plan = _plan(case, arm)
        row = {'candidate_id': case['candidate_id'], 'decision_time': case['decision_time'],
               'subtype': subtype, 'scenario': plan['scenario'], 'status': 'not_entered',
               'reason': plan['reason'], 'net_pnl': 0., 'position': None,
               'terminal_at': t.isoformat(), 'decision': deepcopy(plan), 'execution_authorized': False}
        if plan['action'] == 'none':
            pass
        elif uncertain is not None and t >= uncertain:
            row.update(status='unknown', reason='occupied_book_path_unknown', net_pnl=None)
        elif release is not None and t < release:
            row['reason'] = 'busy'
        else:
            row, case_events, case_marks = _window(case, windows[case['candidate_id']], plan,
                arm=arm, cost_bps=cost_bps, delay_seconds=delay_seconds, funding_mode=funding_mode)
            events.extend(case_events); marks.extend(case_marks)
            release = clock(row['terminal_at'])
            if row['net_pnl'] is None:
                uncertain = release
        rows.append(row)
    saved = {'binding': binding, 'release_time': release.isoformat() if release else None,
             'unknown_since': uncertain.isoformat() if uncertain else None,
             'last_candidate_time': last.isoformat() if last else None}
    return {'schema': 'lc-context-book-v1', **binding, 'rows': rows, 'events': events,
            'marks': marks, 'unresolved': sum(r['net_pnl'] is None for r in rows),
            'checkpoint': dict(saved, seal=seal(saved)), 'execution_authorized': False}
