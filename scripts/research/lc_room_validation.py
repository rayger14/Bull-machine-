"""One frozen known-room gate; independent unfunded books, never live authority."""
from collections import Counter
from copy import deepcopy

import pandas as pd

from scripts.research.conditional_entry import _clock
from scripts.research.conditional_occupancy import replay_sleeve
from scripts.research.lc_single_accounting import _admission_aware_mtm
from scripts.research.lc_upside_scorecard import book_metrics


ROOM_LABELS = {'at_least_2r', 'below_2r', 'no_reference', 'unknown'}
SCENARIOS = ((12, 90), (24, 90), (12, 300), (24, 300))


def compile_room_plans(cases, contexts):
    """Join already validated predecision labels exactly; unknown means abstain.

    Missing rows are integrity failures, unlike an explicit unknown label. An
    unavailable original price plan remains unavailable in both economic books.
    """
    ids = [c['candidate_id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('unique source identities required')
    selected = [c for c in cases if c['subtype'] == 'upside_expansion_candidate']
    ctx = {c['candidate_id']: c for c in contexts}
    if len(ctx) != len(contexts) or set(ctx) != {c['candidate_id'] for c in selected}:
        raise ValueError('exact unique context join required')
    result = dict(baseline=[], known_room=[], decisions=[])
    for case in selected:
        cid = case['candidate_id']; context = ctx[cid]
        if _clock(context['decision_time']) != _clock(case['decision_time']):
            raise ValueError('context decision clock differs')
        label = context['labels']['mapped_overhead']
        if label not in ROOM_LABELS:
            raise ValueError('unknown room label vocabulary')
        allowed = label == 'at_least_2r'
        result['decisions'].append(dict(candidate_id=cid, decision_time=case['decision_time'],
            room_label=label, permission='enter' if allowed else 'abstain',
            reason=None if allowed else label))
        for arm in ('baseline', 'known_room'):
            plan = deepcopy(case['plans']['immediate']) if case['plans'] else None
            if plan is not None and arm == 'known_room' and not allowed:
                plan.update(action='reject', level=None)
            result[arm].append(dict(candidate_id=cid, track='hourly',
                decision_time=case['decision_time'], plan=plan,
                unavailable_reason=case['unavailable_reason']))
    return result


def _attribution(baseline, gated, decisions):
    labels = {r['candidate_id']: r['room_label'] for r in decisions}
    base = {r['candidate_id']: r for r in baseline['ledger']}
    gate = {r['candidate_id']: r for r in gated['ledger']}
    if set(base) != set(gate) or set(base) != set(labels):
        raise ValueError('economic attribution identities differ')
    pairs = []
    nonentries = {'rejected', 'skipped_busy', 'expired', 'cancelled'}
    for cid, b in base.items():
        g = gate[cid]
        bp, gp = b['position'], g['position']
        bn = bp['net_pnl'] if bp and bp['status'] == 'closed' else None
        gn = gp['net_pnl'] if gp and gp['status'] == 'closed' else None
        category = 'not_comparable'
        if bn is not None and gn is not None:
            category = 'winner_preserved' if bn > 0 else 'loser_retained' if bn < 0 else 'flat_retained'
        elif bn is not None and g['status'] in nonentries:
            category = 'missed_winner' if bn > 0 else 'avoided_loser' if bn < 0 else 'flat_nonentry'
        elif gn is not None and b['status'] in nonentries:
            category = ('new_winner_after_capacity_change' if gn > 0 else
                        'new_loser_after_capacity_change' if gn < 0 else 'new_flat_after_capacity_change')
        elif b['status'] in nonentries and g['status'] in nonentries:
            category = 'both_nonentry'
        pairs.append(dict(candidate_id=cid, room_label=labels[cid], category=category,
                          baseline_net=bn, gated_net=gn,
                          baseline_status=b['status'], gated_status=g['status']))
    return dict(counts=dict(Counter(p['category'] for p in pairs)), cases=pairs)


def score_room(cases, contexts, bars, *, first_month, last_month):
    """Replay fixed decisions against minute prices; no threshold optimization."""
    compiled = compile_room_plans(cases, contexts)
    as_of = (max(_clock(c['decision_time']) for c in cases)+pd.Timedelta('1d')
             if cases else pd.Timestamp(last_month+'-01T00:00Z')+pd.offsets.MonthBegin(1))
    scenarios = {}
    for cost, delay in SCENARIOS:
        books, metrics = {}, {}
        for arm in ('baseline', 'known_room'):
            candidates = deepcopy(compiled[arm])
            for row in candidates:
                if row['plan'] is not None:
                    row['plan'].update(cost_bps=cost, processing_seconds=delay)
            book = replay_sleeve(bars, candidates, track='hourly', as_of=as_of.isoformat())
            curve = _admission_aware_mtm(bars, book)
            book['mtm'] = {k: v for k, v in curve.items() if k != 'points'}
            book['mtm']['point_count'] = (len(curve['points'])
                                           if curve.get('points') is not None else None)
            books[arm] = book
            m = book_metrics(book, first_month=first_month, last_month=last_month)
            m['mean_net_r_per_supplied_candidate'] = (
                m['total_net_r']/m['candidate_count'] if m['complete'] and m['candidate_count'] else None)
            m['net_dollars_per_supplied_candidate'] = (
                m['net_pnl']/m['candidate_count'] if m['complete'] and m['candidate_count'] else None)
            metrics[arm] = m
        a, b = metrics['baseline'], metrics['known_room']
        scenarios[f'{cost}bps_{delay}s'] = dict(books=books, metrics=metrics,
            attribution=_attribution(books['baseline'], books['known_room'], compiled['decisions']),
            net_pnl_delta=b['net_pnl']-a['net_pnl'] if a['complete'] and b['complete'] else None)
    return dict(version='lc_known_room_v1', source_case_count=len(cases),
        source_subtypes=dict(Counter(c['subtype'] for c in cases)),
        selected_count=len(compiled['decisions']), decisions=compiled['decisions'],
        room_counts=dict(Counter(c['room_label'] for c in compiled['decisions'])), scenarios=scenarios,
        execution_authorized=False, pristine_holdout=False, profitability_certified=False,
        optimization_performed=False, model_calls=0)
