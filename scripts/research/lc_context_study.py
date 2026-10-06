"""Fixed LC comparison and descriptive uncertainty; never optimization or orders."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.research.lc_context_contract import ARMS, SUBTYPES, clock, protocol, seal, verify_case
from scripts.research.lc_context_execution import replay_book
from scripts.research.lc_context_source import (
    MONTHS, ROOT, implementation_files, new_output, sha, verify_files,
)
from scripts.research.study_source import _Budget, _deadline, load_minutes


def indexed(book, cases):
    rows = {r['candidate_id']: r for r in book['rows']}
    ids = [c['candidate_id'] for c in cases]
    if len(rows) != len(book['rows']) or len(ids) != len(set(ids)) or set(rows) != set(ids):
        raise ValueError('common raw candidate denominator mismatch')
    for c in cases:
        if clock(c['decision_time']) != clock(rows[c['candidate_id']]['decision_time']):
            raise ValueError('candidate clock mismatch')
        value = rows[c['candidate_id']]['net_pnl']
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError('invalid economic value')
    return rows


def summarize(book, cases):
    rows = indexed(book, cases)
    complete = all(r['net_pnl'] is not None for r in rows.values())
    known = [r['net_pnl'] for r in rows.values() if r['net_pnl'] is not None]
    closed = [r for r in rows.values() if r['status'] == 'closed']
    total = math.fsum(known)
    total_r = math.fsum(r['position']['net_r'] for r in closed)
    calendar = {m: {'month': m, 'candidate_count': 0, 'filled_count': 0, 'net_pnl': 0.} for m in MONTHS}
    exposure = 0.
    for c in cases:
        r = rows[c['candidate_id']]
        month = clock(c['decision_time']).strftime('%Y-%m')
        if month not in calendar:
            raise ValueError('candidate outside declared calendar')
        item = calendar[month]
        item['candidate_count'] += 1
        item['filled_count'] += int(r['position'] is not None)
        if r['net_pnl'] is None:
            item['net_pnl'] = None
        elif item['net_pnl'] is not None:
            item['net_pnl'] += r['net_pnl']
        if r['status'] == 'closed':
            exposure += (clock(r['position']['exit_time'])-clock(r['position']['entry_time'])).total_seconds()
    settled, peak, drawdown = 0., 0., 0.
    seen_exits = set()
    for mark in sorted(book.get('marks', []), key=lambda m: clock(m['available_at'])):
        equity = settled+mark['liquidation_value']
        peak = max(peak, equity)
        drawdown = max(drawdown, peak-equity)
        if mark['kind'] == 'exit':
            if mark['candidate_id'] in seen_exits:
                raise ValueError('duplicate MTM exit')
            seen_exits.add(mark['candidate_id'])
            settled += rows[mark['candidate_id']]['net_pnl']
    if complete and (seen_exits != {r['candidate_id'] for r in closed}
                     or not math.isclose(settled, total, abs_tol=1e-7, rel_tol=0)):
        raise ValueError('MTM path does not reconcile with closed positions')
    losses = -math.fsum(v for v in known if v < 0)
    calendar_seconds = (clock(protocol()['end_exclusive'])-clock(protocol()['start'])).total_seconds()
    n = len(cases)
    return {'candidate_count': n, 'filled_count': sum(r['position'] is not None for r in rows.values()),
            'closed_count': len(closed), 'unresolved': sum(r['net_pnl'] is None for r in rows.values()),
            'wins': sum(v > 0 for v in known), 'losses': sum(v < 0 for v in known),
            'known_net_subtotal': total, 'net_pnl': total if complete else None,
            'net_dollars_per_candidate': total/n if complete and n else None,
            'net_r_per_candidate': total_r/n if complete and n else None,
            'mean_filled_net_r': total_r/len(closed) if complete and closed else None,
            'profit_factor': math.fsum(v for v in known if v > 0)/losses if complete and losses else None,
            'max_drawdown_dollars': drawdown if complete else None,
            'exposure_seconds': exposure if complete else None,
            'time_in_market_fraction': exposure/calendar_seconds if complete else None,
            'fills_per_calendar_month': len(closed)/32 if complete else None,
            'filled_months': sum(c['filled_count'] > 0 for c in calendar.values()),
            'calendar': list(calendar.values()), 'reasons': dict(Counter(r['reason'] for r in rows.values())),
            'net_without_top_three': total-math.fsum(sorted((v for v in known if v > 0), reverse=True)[:3])
                                      if complete else None,
            'funded_account': False, 'execution_authorized': False}


def paired(cases, baseline, candidate):
    a, b = indexed(baseline, cases), indexed(candidate, cases)
    monthly = {m: {'month': m, 'candidate_count': 0, 'delta_net': 0.} for m in MONTHS}
    outcome = Counter()
    complete = True
    for c in cases:
        month = clock(c['decision_time']).strftime('%Y-%m')
        if month not in monthly:
            raise ValueError('candidate outside declared calendar')
        row = monthly[month]; row['candidate_count'] += 1
        first, second = a[c['candidate_id']]['net_pnl'], b[c['candidate_id']]['net_pnl']
        if first is None or second is None:
            complete = False; row['delta_net'] = None
            continue
        if row['delta_net'] is not None:
            row['delta_net'] += second-first
        outcome['winners_preserved'] += first > 0 and second > 0
        outcome['winners_missed'] += first > 0 and second <= 0
        outcome['losers_avoided'] += first < 0 and second >= 0
        outcome['losers_retained'] += first < 0 and second < 0
        outcome['losses_introduced_from_nonloss'] += first >= 0 and second < 0
        outcome['wins_introduced_from_nonwin'] += first <= 0 and second > 0
    result = {'months': list(monthly.values()), 'estimate_dollars_per_candidate': None,
              'lower': None, 'upper': None, 'undefined_draws': None,
              'bootstrap_draws': 5000, 'bootstrap_seed': 20261002,
              'complete': complete, 'selection_adjusted': False, 'pristine_holdout': False,
              **dict(outcome)}
    if complete:
        counts = np.array([m['candidate_count'] for m in monthly.values()])
        deltas = np.array([m['delta_net'] for m in monthly.values()])
        idx = np.random.default_rng(20261002).integers(0, 32, size=(5000, 32))
        denominator = counts[idx].sum(axis=1)
        valid = denominator > 0
        samples = deltas[idx].sum(axis=1)[valid]/denominator[valid]
        result['undefined_draws'] = int((~valid).sum())
        if len(cases):
            result['estimate_dollars_per_candidate'] = float(deltas.sum()/len(cases))
        if len(samples):
            result['lower'], result['upper'] = map(float, np.quantile(samples, [.025, .975]))
    return result


def verdict(*, primary_net, estimate, lower, fills, filled_months, stress_nets, complete):
    failures = []
    if not complete or primary_net is None:
        failures.append('incomplete_primary_economics')
    elif primary_net < 0:
        return {'verdict': 'unsupported', 'reasons': ['negative_primary_net'], 'execution_authorized': False}
    if primary_net is None or primary_net <= 0:
        failures.append('nonpositive_primary_net')
    if estimate is None or estimate <= 0:
        failures.append('nonpositive_or_unknown_improvement')
    if lower is None or lower <= 0:
        failures.append('nonpositive_or_unknown_lower_interval')
    if fills < 50:
        failures.append('fewer_than_50_fills')
    if filled_months < 12:
        failures.append('fewer_than_12_filled_months')
    if len(stress_nets) != 4 or any(v is None or v < 0 for v in stress_nets):
        failures.append('negative_or_unknown_stress_net')
    return {'verdict': 'inconclusive' if failures else 'worth_forward_test',
            'reasons': failures or ['all_declared_forward_screen_checks_pass'], 'execution_authorized': False}


def validate_review(review, source):
    if (review.get('decision') != 'GO' or review.get('stage') != 'lc_context_economics'
            or not review.get('software_review') or not review.get('quant_review')
            or review.get('execution_authorized') is not False
            or review.get('source_receipt_sha256') != sha(Path(source)/'receipt.json')):
        raise ValueError('matching independent economic-launch review required')
    verify_files(review['files'])


def _save_book(budget, path, value):
    data = gzip.compress(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
    budget.check(len(data))
    with path.open('xb') as handle:
        handle.write(data)


def score(source, output, review_path):
    source = Path(source).resolve()
    review_path = Path(review_path).resolve()
    review = json.loads(review_path.read_text())
    validate_review(review, source)
    current = implementation_files()
    if any(review['files'].get(p) != value for p, value in current.items()):
        raise ValueError('review does not bind every current study consumer')
    receipt = json.loads((source/'receipt.json').read_text())
    if (receipt.get('stage') != 'lc_context_source' or receipt.get('completed') is not True
            or receipt.get('economic_outcomes_computed') is not False
            or receipt.get('policy_seal') != seal(protocol())):
        raise ValueError('qualified source receipt required')
    verify_files(receipt['files'])
    verify_files({str(source/name): value for name, value in receipt['artifacts'].items()})
    cases = json.loads((source/'cases.json').read_text())
    for case in cases:
        verify_case(case)
    if len(cases) != receipt['case_count']:
        raise ValueError('sealed source count differs')
    out, budget, began = new_output(output), _Budget(600, 512*1024*1024), time.monotonic()
    try:
        with _deadline(600):
            budget.save(out/'launch.json', {'review': review, 'review_sha256': sha(review_path),
                'policy': protocol(), 'execution_authorized': False})
            windows = {}
            for case in cases:
                t = clock(case['decision_time'])
                windows[case['candidate_id']] = load_minutes(t, t+pd.Timedelta('24h1min'))
            summaries, compact_books, artifacts = {}, {}, {}
            for cost in (12, 24):
                for delay in (90, 300):
                    for funding in ('adverse_stress', 'zero_diagnostic'):
                        for subtype in SUBTYPES:
                            supplied = [c for c in cases if c['subtype'] == subtype]
                            for arm in ARMS:
                                name = f'{subtype}__{arm}__{cost}bps_{delay}s__{funding}'
                                book = replay_book(cases, windows, arm=arm, subtype=subtype,
                                    cost_bps=cost, delay_seconds=delay, funding_mode=funding)
                                summaries[name] = summarize(book, supplied)
                                # Paired comparisons only consume rows; do not retain every mark in RAM.
                                compact_books[name] = {'rows': book['rows']}
                                path = out/(name+'.json.gz')
                                _save_book(budget, path, book)
                                artifacts[path.name] = sha(path)
                        print(f'Completed {cost}bps/{delay}s/{funding} isolated books.', flush=True)
            comparisons, decisions = {}, {}
            for subtype in SUBTYPES:
                supplied = [c for c in cases if c['subtype'] == subtype]
                for cost in (12, 24):
                    for delay in (90, 300):
                        for funding in ('adverse_stress', 'zero_diagnostic'):
                            suffix = f'{cost}bps_{delay}s__{funding}'
                            base = compact_books[f'{subtype}__immediate__{suffix}']
                            for arm in ('unconditional_wait', 'context'):
                                name = f'{subtype}__{arm}__{suffix}'
                                comparisons[name] = paired(supplied, base, compact_books[name])
                name = f'{subtype}__context__12bps_90s__adverse_stress'
                primary, pair = summaries[name], comparisons[name]
                decisions[subtype] = verdict(primary_net=primary['net_pnl'],
                    estimate=pair['estimate_dollars_per_candidate'], lower=pair['lower'] if pair['undefined_draws'] == 0 else None,
                    fills=primary['filled_count'], filled_months=primary['filled_months'],
                    complete=primary['unresolved'] == 0,
                    stress_nets=[summaries[f'{subtype}__context__{c}bps_{d}s__adverse_stress']['net_pnl']
                                 for c, d in [(12, 90), (12, 300), (24, 90), (24, 300)]])
            report = {'schema': 'lc-context-comparison-v1', 'summaries': summaries,
                'paired': comparisons, 'verdicts': decisions, 'case_count': len(cases),
                'unresolved_subtype_count': sum(c['subtype'] not in SUBTYPES for c in cases),
                'execution_authorized': False, 'pristine_holdout': False, 'optimization_performed': False,
                'primary_scenario': '12bps_90s__adverse_stress'}
            budget.save(out/'comparison.json', report)
            artifacts.update({p.name: sha(p) for p in out.iterdir() if p.is_file()})
            verify_files(receipt['files']); validate_review(review, source)
            result = {'stage': 'lc_context_economics', 'completed': True, 'book_count': len(summaries),
                'unknown_outcomes': sum(v['unresolved'] for v in summaries.values()), 'artifacts': artifacts,
                'review_sha256': sha(review_path), 'source_receipt_sha256': sha(source/'receipt.json'),
                'elapsed_seconds': time.monotonic()-began, 'artifact_bytes_before_receipt': budget.bytes,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'execution_authorized': False}
            budget.save(out/'receipt.json', result)
            return {k: v for k, v in result.items() if k != 'artifacts'}
    except BaseException as exc:
        _Budget(60, 65536).save(out/'failure.json', {'stage': 'lc_context_economics', 'completed': False,
            'error': type(exc).__name__+': '+str(exc), 'execution_authorized': False})
        raise
