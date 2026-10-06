"""Frozen paired-opportunity estimand, month bootstrap and conservative decisions."""
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd

from scripts.research.study_contract import finite_number, protocol, utc_minute

MONTHS = tuple(str(p) for p in pd.period_range('2024-01', '2026-08', freq='M'))
BLOCKS = ('2024 H1', '2024 H2', '2025 H1', '2025 H2', '2026 Jan-Aug')


def _rows(book):
    rows = book['rows'] if isinstance(book, dict) else book
    indexed = {r['opportunity_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate book opportunity')
    return indexed


def _opportunities(opportunities):
    indexed = {o['id']: o for o in opportunities}
    if len(indexed) != len(opportunities):
        raise ValueError('duplicate raw opportunity')
    if len({(o['instrument'], o['data_stream_id'], o['family']) for o in opportunities}) > 1:
        raise ValueError('foreign family/source denominator')
    return indexed


def paired_months(opportunities, baseline, repair, months):
    if len(set(months)) != len(months):
        raise ValueError('duplicate calendar month')
    ops, base, fixed = _opportunities(opportunities), _rows(baseline), _rows(repair)
    if set(ops) != set(base) or set(ops) != set(fixed):
        raise ValueError('common denominator mismatch')
    output = {m: {'month': m, 'opportunity_count': 0, 'baseline_net': 0., 'repair_net': 0.,
                  'baseline_fills': 0, 'repair_fills': 0, 'unresolved_count': 0} for m in months}
    for oid, op in ops.items():
        month = utc_minute(op['origin_time']).strftime('%Y-%m')
        if month not in output:
            raise ValueError('opportunity outside locked calendar')
        row = output[month]
        row['opportunity_count'] += 1
        unknown = False
        for name, arm in [('baseline', base), ('repair', fixed)]:
            value = arm[oid]['net_pnl']
            if value is None:
                row[name + '_net'] = None
                unknown = True
            elif row[name + '_net'] is not None:
                row[name + '_net'] += finite_number(value)
            if arm[oid].get('position') is not None and arm[oid]['status'] == 'closed':
                row[name + '_fills'] += 1
        row['unresolved_count'] += int(unknown)
    return list(output.values())


def paired_interval(month_rows, *, draws=5000, seed=20260930):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1 or not month_rows:
        raise ValueError('positive draws and nonempty calendar required')
    result = {'draws': draws, 'seed': seed, 'quantiles': protocol()['interval_quantiles'],
              'status': 'insufficient_evidence', 'estimate': None, 'lower': None, 'upper': None,
              'undefined_draws': None, 'undefined_fraction': None, 'undefined_limit_failed': None}
    if any(r['unresolved_count'] or r['baseline_net'] is None or r['repair_net'] is None for r in month_rows):
        return dict(result, status='blocked', reason='incomplete_paired_outcomes')
    counts = np.array([finite_number(r['opportunity_count']) for r in month_rows])
    delta = np.array([finite_number(r['repair_net']) - finite_number(r['baseline_net']) for r in month_rows])
    if (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError('nonnegative integer opportunity counts required')
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(month_rows), size=(draws, len(month_rows)))
    denominator = 100. * counts[indexes].sum(axis=1)
    defined = denominator > 0
    values = delta[indexes].sum(axis=1)[defined] / denominator[defined]
    undefined = int((~defined).sum())
    result.update(estimate=float(delta.sum() / (100. * counts.sum())) if counts.sum() else None,
                  undefined_draws=undefined, undefined_fraction=undefined / draws,
                  undefined_limit_failed=undefined / draws > .01, defined_draws=int(defined.sum()))
    if len(values):
        result['lower'], result['upper'] = map(float, np.quantile(values, result['quantiles']))
    result['status'] = 'insufficient_evidence' if result['undefined_limit_failed'] else 'computed'
    return result


def campaign_decision(summary):
    """Ordered table; retain secondary failures without choosing a nicer scenario."""
    failures = []
    if summary.get('required_data_ok') is not True or summary.get('blockers'):
        failures.append(('blocked', 'required_contract_failed'))
    names = ('repair_fills', 'origin_months_with_repair_fills', 'undefined_fraction', 'primary_repair_net',
             'positive_blocks', 'repair_net_without_top3')
    try:
        values = {name: finite_number(summary[name]) for name in names}
        scenarios = [finite_number(v) for v in summary['scenario_repair_net']]
        if len(scenarios) != 4:
            raise ValueError('four frozen scenarios required')
        for name in ('repair_fills', 'origin_months_with_repair_fills', 'positive_blocks'):
            if values[name] < 0 or values[name] != int(values[name]):
                raise ValueError('integer count required')
        if not 0 <= values['undefined_fraction'] <= 1:
            raise ValueError('invalid undefined fraction')
        empty_undefined = (values['repair_fills'] == 0 and values['origin_months_with_repair_fills'] == 0
                           and values['undefined_fraction'] == 1
                           and summary.get('incremental_estimate') is None
                           and summary.get('incremental_lower_bound') is None)
        for name in ('incremental_estimate', 'incremental_lower_bound'):
            values[name] = None if empty_undefined else finite_number(summary[name])
    except (ValueError, TypeError, KeyError):
        failures.insert(0, ('blocked', 'invalid_or_incomplete_summary'))
    else:
        checks = [
            (values['repair_fills'] < 50, 'insufficient_evidence', 'repair_fill_floor'),
            (values['origin_months_with_repair_fills'] < 12, 'insufficient_evidence', 'origin_month_floor'),
            (values['undefined_fraction'] > .01, 'insufficient_evidence', 'undefined_bootstrap_limit'),
            (values['primary_repair_net'] <= 0, 'park', 'nonpositive_repair_net'),
            (values['incremental_estimate'] is not None and values['incremental_estimate'] <= 0, 'park', 'nonpositive_increment'),
            (any(v <= 0 for v in scenarios), 'park', 'stress_scenario_failure'),
            (values['positive_blocks'] < 3, 'park', 'period_fragility'),
            (values['repair_net_without_top3'] <= 0, 'park', 'top_three_concentration'),
            (values['incremental_lower_bound'] is not None and values['incremental_lower_bound'] <= 0, 'insufficient_evidence', 'nonpositive_adjusted_lower_bound'),
        ]
        failures.extend((decision, reason) for condition, decision, reason in checks if condition)
    decision, reason = failures[0] if failures else ('eligible_for_forward_proposal', 'all_frozen_checks_passed')
    return {'decision': decision, 'reason': reason, 'secondary_failures': [r for _, r in failures[1:]],
            'execution_authorized': False, 'actual_funding_and_receipt_qualification_required_before_forward_launch': True}


def _block(at):
    at = utc_minute(at)
    if at.year in (2024, 2025):
        return f'{at.year} H{1 if at.month <= 6 else 2}'
    if at.year == 2026 and at.month <= 8:
        return '2026 Jan-Aug'
    raise ValueError('origin outside calendar reporting blocks')


def liquidation_path(book):
    """Dollar liquidation-value path; not compounded account/equity returns."""
    updates = defaultdict(lambda: {'entry': [], 'exit': [], 'marks': [], 'funding': []})
    for row in book['rows']:
        p = row.get('position')
        if p is None:
            continue
        oid = row['opportunity_id']
        updates[utc_minute(p['entry_time'])]['entry'].append((oid, -p['entry_fee'] - p['exit_fee']))
        if p.get('exit_time') is not None and row['net_pnl'] is not None:
            updates[utc_minute(p['exit_time'])]['exit'].append((oid, row['net_pnl']))
    for mark in book.get('marks', []):
        updates[utc_minute(mark['available_at'])]['marks'].append((mark['opportunity_id'], mark['liquidation_value']))
    for event in book.get('events', []):
        if event['kind'] == 'funding':
            updates[utc_minute(event['available_at'])]['funding'].append((event['opportunity_id'], event['cashflow']))
    active, realized, peak, worst, path = {}, 0., 0., 0., []
    for at, update in sorted(updates.items()):
        for oid, charge in update['funding']:
            if oid in active:
                active[oid] += charge
        active.update(update['marks'])
        for oid, value in update['exit']:
            active.pop(oid, None)
            realized += value
        active.update(update['entry'])
        value = realized + sum(active.values())
        peak = max(peak, value)
        worst = max(worst, peak - value)
        path.append({'available_at': at.isoformat(), 'net_liquidation_value': value})
    return {'path': path, 'maximum_drawdown_dollars': worst}


def summarize_book(book, opportunities, minutes=None):
    rows, ops = _rows(book), _opportunities(opportunities)
    if set(rows) != set(ops):
        raise ValueError('summary common denominator mismatch')
    unresolved = sum(r['net_pnl'] is None for r in rows.values())
    closed = [r for r in rows.values() if r['status'] == 'closed']
    pnls = [finite_number(r['net_pnl']) for r in closed]
    wins = sorted([p for p in pnls if p > 0], reverse=True)
    losses = [p for p in pnls if p < 0]
    net = None if unresolved else sum(finite_number(r['net_pnl']) for r in rows.values())
    positions = [r['position'] for r in closed]
    contributions = {b: 0. for b in BLOCKS}
    for oid, row in rows.items():
        block = _block(ops[oid]['origin_time'])
        if row['net_pnl'] is None:
            contributions[block] = None
        elif contributions[block] is not None:
            contributions[block] += row['net_pnl']
    excursions = []
    if minutes is not None:
        if not minutes.index.is_monotonic_increasing or not minutes.index.is_unique:
            raise ValueError('sorted unique minute index required for excursions')
        for row in closed:
            p = row['position']
            entry, exit_at = utc_minute(p['entry_time']), utc_minute(p['exit_time'])
            window = minutes.loc[entry:exit_at - pd.Timedelta('1min')]
            highs = [p['entry_price'], p['exit_price']] + window.high.tolist()
            lows = [p['entry_price'], p['exit_price']] + window.low.tolist()
            excursions.append({'opportunity_id': row['opportunity_id'],
                               'mfe_r_upper_bound': max(0., max(highs) - p['entry_price']) * p['quantity'] / p['initial_risk'],
                               'mae_r_upper_bound': max(0., p['entry_price'] - min(lows)) * p['quantity'] / p['initial_risk']})
    path = liquidation_path(book)
    return {'raw_opportunities': len(ops), 'completed_fills': len(closed), 'unresolved': unresolved,
            'net_pnl': net, 'realized_net_pnl': sum(pnls), 'net_expectancy_per_fill': sum(pnls) / len(pnls) if pnls and not unresolved else None,
            'net_per_100_risk_per_opportunity': net / (100. * len(ops)) if net is not None and ops else None,
            'origin_months_with_fills': len({utc_minute(ops[r['opportunity_id']]['origin_time']).strftime('%Y-%m') for r in closed}),
            'win_rate': len(wins) / len(pnls) if pnls else None,
            'profit_factor': sum(wins) / -sum(losses) if losses else None,
            'worst_net_loss': min(pnls + [0.]), 'net_without_top_three_winners': net - sum(wins[:3]) if net is not None else None,
            'top_three_winner_net': sum(wins[:3]), 'fees': sum(p['entry_fee'] + p['exit_fee'] for p in positions),
            'funding': sum(p['funding'] for p in positions),
            'exposure_hours': sum((utc_minute(p['exit_time']) - utc_minute(p['entry_time'])).total_seconds() / 3600 for p in positions),
            'initial_risk_sum': sum(p['initial_risk'] for p in positions), 'reason_counts': dict(Counter(r['reason'] for r in rows.values())),
            'calendar_contributions': contributions, 'positive_blocks': sum(v is not None and v > 0 for v in contributions.values()),
            'maximum_drawdown_dollars': path['maximum_drawdown_dollars'] if not unresolved else None,
            'liquidation_path': path['path'], 'excursions': excursions,
            'excursion_basis': 'minute-bar extrema through modeled exit; within-exit-bar order unresolved, upper bounds only',
            'execution_authorized': False}


def render_report(results):
    decision = results['decision']
    rows = [f"{results['family']} frozen comparison: {decision['decision']}",
            f"Reason: {decision['reason']}. Not live-ready.",
            'History is exposed development evidence, not a pristine holdout or walk-forward optimization.',
            'Independent books; no compounded account-return claim. August 2026 is a partial setup month.', '']
    for arm in ('baseline', 'repair'):
        s = results[arm]
        rows.append(f"{arm}: {s['raw_opportunities']} raw opportunities, {s['completed_fills']} completed fills, "
                    f"net {s['net_pnl']}, unresolved {s['unresolved']}.")
    rows.extend(['', 'Paired monthly interval: ' + json.dumps(results['interval'], sort_keys=True),
                 'Secondary failures: ' + ', '.join(decision['secondary_failures']),
                 'Funding stress is a scenario, not observed funding or a guarantee on real costs.'])
    return '\n'.join(rows) + '\n'
