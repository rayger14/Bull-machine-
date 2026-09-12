"""Causal, unfunded single-position sleeves for immutable conditional plans.

Actual-ready ordering and actual-exit release, not legacy deadline lockout.
Source completeness, role provenance and source/plan locks remain caller duties.
"""
from copy import deepcopy
import math
from numbers import Real

import pandas as pd

from scripts.research.conditional_entry import MINUTE, _clock, resolve_entry
from scripts.research.conditional_assessment import gate_choice


PLAN_KEYS = {'decision_time','action','stop','level','entry_expiry','exit_deadline',
             'processing_seconds','routing_seconds','notional','cost_bps'}
CANDIDATE_KEYS = {'candidate_id','track','decision_time','plan','unavailable_reason'}


def _number(value, *, positive=True):
    try:
        return (not isinstance(value,bool) and isinstance(value,Real)
                and math.isfinite(value) and (value > 0 if positive else value >= 0))
    except OverflowError:
        return False


def _prices(bars, at, names):
    try:
        if (not isinstance(bars,pd.DataFrame) or not isinstance(bars.index,pd.DatetimeIndex)
                or bars.index.tz is None or not bars.columns.is_unique):
            return None
        values = [bars.at[at,k] for k in names]
        return [float(v) for v in values] if all(_number(v) for v in values) else None
    except (KeyError,TypeError,ValueError):
        return None


def _open_position(resolution, plan):
    entry = resolution['entry_price']; stop = float(plan['stop'])
    quantity = plan['notional']/entry
    risk = (entry-stop)*quantity
    target = entry+2*(entry-stop)
    fee = plan['notional']*plan['cost_bps']/10000
    if not all(math.isfinite(v) for v in (quantity,risk,target,fee)) or risk <= 0:
        raise ValueError('unrepresentable position economics')
    return dict(status='open',entry_time=resolution['entry_time'],entry_price=entry,
                stop_price=stop,target_price=target,quantity=quantity,initial_risk=risk,
                fees=fee,exit_deadline=plan['exit_deadline'],net_pnl=None,
                release_time=None,execution_certified=False)


def _advance(bars, position, as_of):
    """Read no future extrema; stop at the first known exit or missing observation."""
    if position['status'] != 'open': return position
    entry,deadline = _clock(position['entry_time']),_clock(position['exit_deadline'])
    stop,target = position['stop_price'],position['target_price']

    def unknown(at,reason):
        return dict(position,status='unknown',unknown_from=at.isoformat(),reason=reason)

    def close(at, price, reason, phase, ambiguous=False):
        release = at if phase == 'open' else at+MINUTE
        gross = (price-position['entry_price'])*position['quantity']
        net = gross-position['fees']
        if not all(math.isfinite(v) for v in (gross,net)):
            return unknown(release,'unrepresentable_exit_economics')
        return dict(position,status='closed',exit_time=at.isoformat(),exit_price=price,
                    exit_reason=reason,exit_phase=phase,ambiguous_bar=ambiguous,
                    release_time=release.isoformat(),exit_observed_at=release.isoformat(),
                    gross_pnl=gross,net_pnl=net)

    at = entry
    while at <= as_of:
        opened = _prices(bars,at,('open',))
        if opened is None: return unknown(at,'missing_or_invalid_open')
        op = opened[0]
        if at == deadline: return close(at,op,'deadline','open')
        if op <= stop: return close(at,op,'stop','open')
        if op >= target: return close(at,target,'target','open')
        if at+MINUTE > as_of:
            return dict(position,observed_through=as_of.isoformat())
        row = _prices(bars,at,('open','high','low','close'))
        if row is None: return unknown(at+MINUTE,'missing_or_invalid_completed_bar')
        op,hi,lo,cl = row
        if lo > min(op,cl) or hi < max(op,cl):
            return unknown(at+MINUTE,'invalid_ohlc_envelope')
        if lo <= stop: return close(at,stop,'stop','intrabar',hi >= target)
        if hi >= target: return close(at,target,'target','intrabar')
        at += MINUTE
    return dict(position,observed_through=as_of.isoformat())


def replay_sleeve(bars, candidates, *, track, as_of):
    """Retain every supplied candidate; do not certify population completeness.

    Missing pending/position observations end the trustworthy admission prefix.
    Unavailable assessments are fail-closed nonorders but invalidate full-policy
    accounting. Pending intents reserve no capacity; busy entries never retry.
    """
    cutoff = _clock(as_of)
    if track not in ('hourly','minute') or not isinstance(candidates,list):
        raise ValueError('single supported track and candidate list required')
    copied = deepcopy(candidates); seen = set(); economics = None
    for c in copied:
        if not isinstance(c,dict) or set(c) != CANDIDATE_KEYS:
            raise ValueError('exact candidate fields required')
        identity = c['candidate_id']
        if type(identity) is not str or not identity.strip() or identity in seen or c['track'] != track:
            raise ValueError('unique IDs and unmixed track required')
        seen.add(identity); decision = _clock(c['decision_time']); plan = c['plan']
        if plan is None:
            if type(c['unavailable_reason']) is not str or not c['unavailable_reason'].strip():
                raise ValueError('unavailable plan needs explicit reason')
            continue
        if (not isinstance(plan,dict) or set(plan) != PLAN_KEYS or c['unavailable_reason'] is not None
                or _clock(plan['decision_time']) != decision
                or not _number(plan['notional']) or not _number(plan['cost_bps'],positive=False)):
            raise ValueError('invalid plan schema, clock or economics')
        pair = (plan['notional'],plan['cost_bps'])
        if economics is not None and pair != economics:
            raise ValueError('one notional/cost convention per sleeve required')
        economics = pair
    copied.sort(key=lambda c:(_clock(c['decision_time']),c['candidate_id']))
    rows,events = [],[]
    for c in copied:
        row = dict(candidate_id=c['candidate_id'],decision_time=_clock(c['decision_time']).isoformat(),
                   status='plan_unavailable',unavailable_reason=c['unavailable_reason'],
                   resolution=None,position=None,blocked_by=None)
        rows.append(row)
        if c['plan'] is None: continue
        params = {k:v for k,v in c['plan'].items() if k not in ('notional','cost_bps')}
        resolution = resolve_entry(bars,as_of=cutoff,**params)
        row.update(status=resolution['status'],resolution=resolution)
        if resolution['status'] in ('entry_ready','data_unavailable'):
            at = _clock(resolution['resolved_at'])
            priority = 0 if resolution['status'] == 'data_unavailable' else 1
            events.append((at,priority,_clock(c['decision_time']),c['candidate_id'],len(rows)-1))
    active = None; uncertain = None; admissions = []

    def advance(at):
        nonlocal active,uncertain
        if active is None: return
        position = _advance(bars,rows[active]['position'],at)
        rows[active]['position'] = position
        if position['status'] == 'unknown':
            uncertain = _clock(position['unknown_from'])
        elif position['status'] == 'closed':
            active = None

    for at,priority,decision,identity,index in sorted(events):
        row = rows[index]
        if uncertain is not None:
            if priority == 1: row['status'] = 'admission_indeterminate'
            continue
        advance(at)
        if uncertain is not None:
            if priority == 1: row['status'] = 'admission_indeterminate'
            continue
        if priority == 0:
            uncertain = at
            continue
        if active is not None:
            row.update(status='skipped_busy',blocked_by=rows[active]['candidate_id'])
            continue
        try:
            position = _open_position(row['resolution'],copied[index]['plan'])
        except (ValueError,OverflowError):
            row.update(status='invalid_economics')
            continue
        row.update(status='admitted',position=position)
        admissions.append(identity); active = index
    if uncertain is None: advance(cutoff)
    closed = [r['position'] for r in rows if r['position'] and r['position']['status'] == 'closed']
    admitted = [r['position'] for r in rows if r['position']]
    complete = (uncertain is None
                and all(r['status'] in ('admitted','skipped_busy','rejected','cancelled','expired') for r in rows)
                and all(p['status'] == 'closed' for p in admitted))
    try:
        subtotal = math.fsum(p['net_pnl'] for p in closed)
        mean_risk = math.fsum(p['initial_risk'] for p in admitted)/len(admitted) if admitted else None
    except (OverflowError,ValueError):
        subtotal,mean_risk,complete = None,None,False
    return dict(track=track,as_of=cutoff.isoformat(),ledger=rows,admission_order=admissions,
                uncertain_from=uncertain.isoformat() if uncertain is not None else None,
                execution_authorized=False,source_population_verified=False,
                summary=dict(candidate_count=len(rows),admitted_count=len(admissions),closed_count=len(closed),
                    skipped_busy_count=sum(r['status']=='skipped_busy' for r in rows),
                    admission_indeterminate_count=sum(r['status']=='admission_indeterminate' for r in rows),
                    supplied_population_resolved=complete,closed_trade_net_pnl=subtotal,
                    policy_net_pnl=subtotal if complete else None,average_initial_risk=mean_risk,
                    starting_equity=None,funded=False,execution_certified=False))


def replay_reviewed_sleeve(bars, requests, *, track, as_of):
    """Compile each bound choice through the real gate before occupancy replay.

    Hand-authored reviews can satisfy this contract: caller must establish actual
    independent roles, transport and prior locks before interpreting agent results.
    """
    if not isinstance(requests,list): raise ValueError('request list required')
    candidates,gates = [],{}
    for request in requests:
        if not isinstance(request,dict) or set(request) != {'candidate_id','packet','menu','choice','review','settings'}:
            raise ValueError('exact reviewed request fields required')
        packet = request['packet']; identity = request['candidate_id']
        if not isinstance(packet,dict) or packet.get('track') != track:
            raise ValueError('packet track mismatch')
        gate = gate_choice(packet,request['menu'],request['choice'],request['review'],request['settings'])
        candidates.append(dict(candidate_id=identity,track=track,decision_time=packet['decision_time'],
            plan=gate['research_plan'],unavailable_reason=gate['status'] if gate['research_plan'] is None else None))
        gates[identity] = gate
    result = replay_sleeve(bars,candidates,track=track,as_of=as_of)
    result['review_gates'] = gates
    return result
