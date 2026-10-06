"""Exposed-history upside LC scorecard; no new rules, models or live effects.

Caller verifies source/archive/code hashes and publishes an immutable result.
Risk-normalized statistics are not a funded portfolio or selection-adjusted test.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from scripts.research.lc_mechanical_extension import score_extension


def upside_cases(cases):
    """Select by the frozen predecision label, keeping unavailable selected rows."""
    ids = [c['candidate_id'] for c in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('nonempty unique source identities required')
    return deepcopy([c for c in cases if c['subtype'] == 'upside_expansion_candidate'])


def book_metrics(book, *, first_month, last_month):
    """Summarize one independent book, with descriptive monthly resampling.

    R denominator is initial stop-distance risk plus round-trip modeled costs.
    Complete book totals are null if any supplied row is unresolved. Month blocks
    include empty months. Inter-month dependence and strategy selection are not
    captured by this descriptive interval; it is never a promotion certificate.
    """
    months = [str(p) for p in pd.period_range(first_month, last_month, freq='M')]
    if not months:
        raise ValueError('nonempty calendar required')
    month_index = {m: i for i, m in enumerate(months)}
    monthly_r = np.zeros(len(months))
    monthly_n = np.zeros(len(months), dtype=int)
    rows = book['ledger']
    complete = bool(book['summary']['supplied_population_resolved'])
    dollars, returns = [], []
    for row in rows:
        at = pd.Timestamp(row['decision_time'])
        if pd.isna(at) or at.tzinfo is None:
            raise ValueError('aware calendar decision required')
        month = at.tz_convert('UTC').strftime('%Y-%m')
        if month not in month_index:
            raise ValueError('decision outside declared calendar')
        position = row['position']
        resolved = row['status'] in ('admitted', 'skipped_busy', 'rejected', 'expired', 'cancelled')
        if row['status'] == 'admitted' and not position:
            resolved = False
        if not resolved or (position and position['status'] != 'closed'):
            complete = False
        if not position or position['status'] != 'closed':
            continue
        vals = [position[k] for k in ('net_pnl', 'initial_risk', 'fees')]
        if any(isinstance(v, bool) for v in vals):
            raise ValueError('invalid position economics')
        net, risk, fees = map(float, vals)
        if not all(math.isfinite(v) for v in (net, risk, fees)) or risk <= 0 or fees < 0:
            raise ValueError('invalid position economics')
        budget = risk + fees
        r = net / budget
        if not math.isfinite(budget) or not math.isfinite(r):
            raise ValueError('unrepresentable position economics')
        dollars.append(net)
        returns.append(r)
        monthly_r[month_index[month]] += r
        monthly_n[month_index[month]] += 1
    count = len(dollars)
    subtotal = math.fsum(dollars)
    total_r = math.fsum(returns)

    def profit_factor(values):
        gains = math.fsum(v for v in values if v > 0)
        losses = -math.fsum(v for v in values if v < 0)
        return gains / losses if losses else None

    interval, empty_draws = None, None
    if complete and count and np.count_nonzero(monthly_n) >= 2:
        rng = np.random.default_rng(20260930)
        sampled = rng.integers(0, len(months), size=(5000, len(months)))
        counts = monthly_n[sampled].sum(axis=1)
        sums = monthly_r[sampled].sum(axis=1)
        populated = counts > 0
        empty_draws = int((~populated).sum())
        interval = np.quantile(sums[populated] / counts[populated], [.025, .975]).tolist()
    top = sorted((v for v in dollars if v > 0), reverse=True)[:3]
    return dict(
        candidate_count=len(rows), closed_count=count,
        statuses=dict(Counter(row['status'] for row in rows)),
        complete=complete, known_net_subtotal=subtotal,
        net_pnl=subtotal if complete else None,
        wins=sum(v > 0 for v in dollars), losses=sum(v < 0 for v in dollars),
        breakevens=sum(v == 0 for v in dollars),
        win_rate=sum(v > 0 for v in dollars) / count if complete and count else None,
        profit_factor_dollars=profit_factor(dollars) if complete else None,
        profit_factor_r=profit_factor(returns) if complete else None,
        mean_net_r=total_r / count if complete and count else None,
        total_net_r=total_r if complete else None,
        net_without_top_three_winners=subtotal - math.fsum(top) if complete else None,
        bootstrap_months=len(months), bootstrap_populated_months=int(np.count_nonzero(monthly_n)),
        bootstrap_samples=5000, bootstrap_seed=20260930,
        bootstrap_zero_fill_draws=empty_draws, bootstrap_95_mean_net_r=interval,
        bootstrap_selection_adjusted=False,
        monthly=[dict(month=m, closed_count=int(monthly_n[i]), known_net_r=float(monthly_r[i]))
                 for i, m in enumerate(months)],
    )


def score_upside(cases, bars):
    """Replay only upside plans with fresh occupancy; preserve old comparator rules."""
    selected = upside_cases(cases)
    if not selected:
        raise ValueError('no source upside candidates')
    clocks = [pd.Timestamp(c['decision_time']).tz_convert('UTC') for c in cases]
    first, last = min(clocks).strftime('%Y-%m'), max(clocks).strftime('%Y-%m')
    replay = score_extension(selected, bars)
    metrics = {scenario: {arm: book_metrics(book, first_month=first, last_month=last)
                          for arm, book in arms.items()}
               for scenario, arms in replay['scenarios'].items()}
    return dict(version='lc_upside_scorecard_v1', source_case_count=len(cases),
                source_subtypes=dict(Counter(c['subtype'] for c in cases)),
                selected_case_count=len(selected), selected_ids=[c['candidate_id'] for c in selected],
                replay=replay, metrics=metrics, pristine_holdout=False,
                optimization_performed=False, execution_authorized=False,
                live_parity_certified=False, profitability_certified=False)
