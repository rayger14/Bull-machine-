"""Score a terminal-locked single-assessor cohort in four isolated LC books."""
from collections import Counter
from copy import deepcopy
import math

import pandas as pd

from scripts.research.conditional_occupancy import replay_sleeve
from scripts.research.lc_campaign_accounting import mtm_curve
from scripts.research.lc_judgment_runner import _load


def _admission_aware_mtm(bars, book):
    """Add fee-event observations without changing the pinned legacy calculator.

    Each book has at most one active position, so at a new admission all earlier
    positions have exited. Same-time realized exits precede the new fee debit.
    """
    curve = mtm_curve(bars, book)
    if curve['status'] != 'available':
        return curve
    positions = [r['position'] for r in book['ledger'] if r['position']]
    points = [dict(p, phase='mark') for p in curve['points']]
    for position in positions:
        at = pd.Timestamp(position['entry_time'])
        gross = math.fsum(p['gross_pnl'] for p in positions
                         if p['status'] == 'closed' and pd.Timestamp(p['exit_observed_at']) <= at)
        prior_fees = math.fsum(p['fees'] for p in positions if pd.Timestamp(p['entry_time']) < at)
        points.extend([
            dict(observed_at=at.isoformat(), dollars=gross-prior_fees, phase='before_admission'),
            dict(observed_at=at.isoformat(), dollars=gross-prior_fees-position['fees'], phase='admission'),
        ])
    order = {'before_admission': 0, 'admission': 1, 'mark': 2}
    points.sort(key=lambda p: (pd.Timestamp(p['observed_at']), order[p['phase']]))
    peak = 0.; drawdown = 0.
    for point in points:
        peak = max(peak, point['dollars'])
        drawdown = max(drawdown, peak-point['dollars'])
    return dict(curve, points=points, max_drawdown_dollars=drawdown)


def _plan(source, name):
    selected = source['plan_menu']['plans'][name]
    return dict(deepcopy(selected['parameters']), notional=selected['notional'],
                cost_bps=selected['cost_bps'])


def _attribution(baseline, agent):
    counts = Counter()
    pairs = []
    for a, c in zip(baseline['ledger'], agent['ledger']):
        ap, cp = a['position'], c['position']
        a_pnl = ap['net_pnl'] if ap and ap['status'] == 'closed' else None
        c_pnl = cp['net_pnl'] if cp and cp['status'] == 'closed' else None
        category = 'not_comparable'
        if a_pnl is not None:
            if c['status'] in ('rejected', 'expired', 'cancelled', 'skipped_busy'):
                category = 'missed_winner' if a_pnl > 0 else 'avoided_loser' if a_pnl < 0 else 'flat_nonentry'
            elif c_pnl is not None:
                category = ('winner_preserved' if c_pnl > 0 else 'winner_not_preserved') if a_pnl > 0 else 'baseline_loser_entered'
        counts[category] += 1
        pairs.append(dict(case_id=a['candidate_id'], baseline_net=a_pnl,
                          agent_net=c_pnl, agent_status=c['status'], category=category))
    return dict(counts=dict(counts), cases=pairs)


def score_single_campaign(controller, bars):
    """Caller loads outcomes only after assert_reveal_allowed; no IO for prices here.

    Independent unfunded single-position books, not live engine management.
    Invalid/unknown assessments make full agent-policy PnL null, not zero.
    """
    locked = controller.assert_reveal_allowed()
    manifest = controller.verify()
    cases = manifest['cases']; terminals = locked['terminals']
    sources = {c['case_id']: _load(c['source_path']) for c in cases}
    as_of = max(_plan(s, 'enter')['exit_deadline'] for s in sources.values())
    scenarios = [(f'fixed_{cost}bps_{delay}s', cost, delay, 'fixed')
                 for cost, delay in manifest['policy']['scenarios']]
    scenarios += [(f'{mode}_{cost}bps', cost, None, mode)
                  for mode in ('measured_agent', 'measured_equal')
                  for cost in manifest['policy']['measured_delay_cost_bps']]
    reports = {}
    for name, cost, delay, mode in scenarios:
        arms = {}
        for arm, menu in [('immediate', 'enter'), ('mechanical_wait', 'wait_5m_high'),
                          ('agent', None), ('reject_all', 'reject')]:
            candidates = []
            for case in cases:
                cid = case['case_id']; terminal = terminals[cid]
                plan = (_plan(sources[cid], menu) if menu else
                        deepcopy(terminal['grade']['research_plan']))
                reason = terminal['grade']['status'] if plan is None else None
                processing = delay
                if mode != 'fixed':
                    processing = (90 if mode == 'measured_agent' and arm != 'agent'
                                  else terminal['processing_seconds'])
                if processing is None:
                    plan = None; reason = 'measured_timing_unavailable'
                if plan is not None:
                    plan.update(cost_bps=cost, processing_seconds=processing)
                candidates.append(dict(candidate_id=cid, track='hourly',
                                       decision_time=case['decision_time'], plan=plan,
                                       unavailable_reason=reason))
            book = replay_sleeve(bars, candidates, track='hourly', as_of=as_of)
            book['mtm'] = _admission_aware_mtm(bars, book)
            arms[arm] = book
        reports[name] = dict(arms=arms,
            attribution_vs_immediate=_attribution(arms['immediate'], arms['agent']),
            attribution_vs_mechanical=_attribution(arms['mechanical_wait'], arms['agent']))
    usable = sum(t['grade']['research_plan'] is not None for t in terminals.values())
    return dict(version='lc_single_accounting_v1', manifest_sha256=manifest['sha256'],
                terminal_lock_sha256=locked['sha256'], scenarios=reports,
                reliability=dict(case_count=len(cases), usable_count=usable,
                                 unavailable_count=len(cases)-usable,
                                 statuses=dict(Counter(t['grade']['status'] for t in terminals.values()))),
                review_status='unreviewed', execution_authorized=False,
                profitability_certified=False, chronological_validation=False,
                limitations=['filtered retrospective cohort, not pristine holdout',
                             'no funding, market impact or account capital model',
                             'schema/citation validity is not semantic verification'])
