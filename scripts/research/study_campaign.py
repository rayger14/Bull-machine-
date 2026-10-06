"""Reviewed R3-only continuation; R1 stays explicitly blocked, no native rerun.

Separate entry point preserves the exact pilot-bound source/CLI files. A local
review receipt records controller/quant authority, not a cryptographic credential.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import platform
import resource
import time

import pandas as pd

from scripts.research.r3_census import build_r3_census, compile_r3_signals, merge_censuses
from scripts.research.replay_clock import digest, json_safe
from scripts.research.study_contract import finite_number, protocol, utc_minute
from scripts.research.study_execution import replay_book
from scripts.research.study_parents import build_study_parents
from scripts.research.study_report import MONTHS, campaign_decision, paired_interval, paired_months, render_report, summarize_book
from scripts.research import study_source as source

SCENARIOS = [[12, 5], [12, 65], [24, 5], [24, 65]]


def read_json(path):
    return json.loads(Path(path).read_text())


def launch_file_hashes():
    manifest = source.preflight()
    if not manifest['source_ready']:
        raise ValueError('source preflight no longer ready')
    result = dict(manifest['files'])
    paths = [Path(__file__), source.ROOT/'docs/knowledge/archetype_repair_scorecard_2026_09_30.md',
             source.ROOT/'docs/superpowers/plans/2026-10-01-archetype-study-replay.md']
    paths.extend(source.ROOT/'scripts/research'/name for name in ('study_execution.py', 'study_parents.py', 'study_report.py'))
    result.update({str(p): source.sha(p) for p in paths})
    return result


def load_review(path, stage):
    review = read_json(path)
    if (review.get('decision') != 'GO' or review.get('scope') != 'R3-only'
            or not review.get('reviewer') or stage not in review.get('approved_stages', ())):
        raise ValueError('explicit quant/controller review required for this stage')
    if (review.get('funding_mode') != 'adverse_stress' or review.get('scenarios') != SCENARIOS
            or json.dumps(review.get('protocol'), sort_keys=True, allow_nan=False) != json.dumps(protocol(), sort_keys=True)):
        raise ValueError('review changed the frozen economic protocol')
    trial = review.get('trial_ledger', {})
    if trial.get('active') != ['R1', 'R3'] or trial.get('parked') != ['R2'] or trial.get('historic_searches_incomplete') is not True:
        raise ValueError('complete trial/selection-history lock required')
    expected = launch_file_hashes()
    if review.get('reviewed_files') != expected:
        raise ValueError('review hash set differs from current source/code')
    for name, expected_hash in expected.items():
        if source.sha(name) != expected_hash:
            raise ValueError('review hash no longer matches file: ' + name)
    limit = review['limits'][stage]
    if not 0 < finite_number(limit['seconds']) <= 1800 or not 1100 <= finite_number(limit['bytes']) <= 4294967296:
        raise ValueError('invalid reviewed resource limit')
    return review


def claim_stage(review_path, stage, output_dir):
    marker = Path(review_path).with_suffix('.' + stage + '.started.json')
    with marker.open('x') as handle:
        json.dump({'stage': stage, 'output_dir': str(Path(output_dir).resolve()),
                   'review_sha256': source.sha(review_path), 'automatic_retry_authorized': False}, handle, sort_keys=True)


def verify_artifacts(directory):
    directory = Path(directory)
    receipt = read_json(directory/'receipt.json')
    if receipt.get('completed') is not True:
        raise ValueError('incomplete stage receipt')
    for name, expected in receipt['artifacts'].items():
        if Path(name).name != name or source.sha(directory/name) != expected:
            raise ValueError('artifact hash mismatch: ' + name)
    return receipt


def verify_parent_prefix(full, prefix, cutoff):
    cutoff = utc_minute(cutoff)
    for key in ('pivots', 'versions', 'transitions'):
        visible = [r for r in full[key] if utc_minute(r['available_at']) <= cutoff]
        if digest(visible) != digest(prefix[key]):
            raise ValueError('parent prefix changed: ' + key)


def book_inputs(census, *, start, end):
    start, end = utc_minute(start), utc_minute(end)
    raw = [o for o in census['opportunities'] if start <= utc_minute(o['origin_time']) < end]
    events = defaultdict(list)
    for event in census['events']:
        events[event['opportunity_id']].append(event)
    arms = {}
    for arm in ('baseline', 'repair'):
        qualifying_kind = 'breakout' if arm == 'baseline' else 'trigger'
        views = []
        for op in raw:
            view = {key: op[key] for key in ('id', 'family', 'origin_time', 'instrument', 'data_stream_id')}
            relevant = events[op['id']]
            emitted = next((e for e in relevant if e['kind'] == qualifying_kind), None)
            terminal = next((e for e in relevant if e['kind'] in ('cancelled', 'expired', 'unknown')), None)
            source_event = emitted or terminal
            view.update(source_status=('complete' if source_event['kind'] != 'unknown' else 'unknown') if source_event else 'censored',
                        source_available_at=source_event['available_at'] if source_event else census['coverage']['last_available_at'])
            views.append(view)
        ids = {o['id'] for o in raw}
        arms[arm] = {'opportunities': views, 'signals': [s for s in compile_r3_signals(census, arm) if s['opportunity_id'] in ids]}
    return raw, arms


def _parents(hourly, pilot_manifest, pilot_parents):
    import talib
    bars = hourly.copy()
    bars['atr_14'] = talib.ATR(bars.high.to_numpy(), bars.low.to_numpy(), bars.close.to_numpy(), timeperiod=14)
    bars['atr_available_at'] = bars.index + pd.Timedelta('1h')
    return {anchor: build_study_parents(bars, instrument='BTC', data_stream_id=pilot_manifest['data_stream_id'],
                                        anchor_timeframe=anchor.split('_')[0],
                                        atr_contract=pilot_parents[anchor]['manifest']['atr_contract'],
                                        source_paths=pilot_manifest['parent_paths'], expected_hashes=pilot_manifest['parent_hashes'])
            for anchor in ('4H_N3', '1D_N3')}


def _new_output(path):
    path = Path(path).resolve()
    root = source.OUTPUT_ROOT.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('new study subdirectory required')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def _complete(out, budget, **fields):
    files = {p.name: source.sha(p) for p in out.iterdir() if p.is_file()}
    receipt = {'completed': True, 'artifacts': files, 'execution_authorized': False,
               'elapsed_seconds': time.monotonic() - budget.began, 'artifact_bytes_before_receipt': budget.bytes,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024), **fields}
    budget.save(out/'receipt.json', receipt)
    return receipt


def _failure(out, exc):
    with (out/'failure.json').open('x') as handle:
        json.dump({'completed': False, 'error_type': type(exc).__name__, 'error': str(exc)[:500],
                   'automatic_retry_authorized': False}, handle)


def qualify_continuous(pilot_dir, output_dir):
    """Bounded same-January R3 parity/resource probe; no repeated native replay."""
    process_began = time.monotonic()
    code_hashes = launch_file_hashes()
    pilot = Path(pilot_dir)
    receipt = verify_artifacts(pilot)
    manifest = read_json(pilot/'manifest.json')
    source.verify_files(manifest)
    if receipt['cohort_blockers']['R3']:
        raise ValueError('R3 pilot source remains blocked')
    out = _new_output(output_dir)
    preflight_seconds = time.monotonic() - process_began
    budget = source._Budget(300, 2147483648)
    try:
        with source._deadline(300):
            policy = protocol()
            minute = source.load_minutes(policy['seed'], policy['pilot_end'])
            hourly = source.hourly_from_minutes(minute, policy['seed'], policy['pilot_end'])
            frozen_parents = read_json(pilot/'parent_ledgers.json')
            started = time.monotonic()
            parents = _parents(hourly, manifest, frozen_parents)
            parent_seconds = time.monotonic() - started
            for key in parents:
                normalized = deepcopy(parents[key])
                normalized['manifest'].pop('study_adapter')
                if digest(normalized) != digest(frozen_parents[key]):
                    raise ValueError('real pilot continuous-parent semantic parity failed')
            started = time.monotonic()
            census = build_r3_census(minute, parents['4H_N3'], instrument='BTC', data_stream_id=manifest['data_stream_id'],
                                      emit_from=policy['start'], end_exclusive=policy['pilot_end'])
            census_seconds = time.monotonic() - started
            expected = read_json(pilot/'r3_census.json')
            if digest(census) != digest(expected):
                raise ValueError('real pilot continuous census parity failed')
            if code_hashes != launch_file_hashes():
                raise ValueError('source/code changed during qualification')
            metrics = {'parent_seconds': parent_seconds, 'r3_census_seconds': census_seconds,
                       'preflight_seconds': preflight_seconds,
                       'total_elapsed_seconds': time.monotonic() - process_began,
                       'input_minutes': len(minute), 'input_hours': len(hourly),
                       'parent_parity': True, 'r3_census_parity': True,
                       'full_source_minutes': int((pd.Timestamp(policy['source_end_exclusive']) - pd.Timestamp(policy['seed'])) / pd.Timedelta('1min')),
                       'native_hourly_replayed': False, 'economic_outcomes_computed': False,
                       'code_hashes': code_hashes}
            budget.save(out/'qualification.json', metrics)
            return _complete(out, budget, stage='continuous_r3_qualification',
                             pilot_receipt_sha256=source.sha(pilot/'receipt.json'), economic_outcomes_computed=False)
    except BaseException as exc:
        _failure(out, exc)
        raise


def run_census(pilot_dir, qualification_dir, output_dir, review_path):
    review = load_review(review_path, 'census')
    pilot, qualification = Path(pilot_dir), Path(qualification_dir)
    pilot_receipt, qual_receipt = verify_artifacts(pilot), verify_artifacts(qualification)
    if (source.sha(pilot/'receipt.json') != review['pilot_receipt_sha256']
            or source.sha(qualification/'receipt.json') != review['qualification_receipt_sha256']
            or pilot_receipt['cohort_blockers']['R3'] or qual_receipt['stage'] != 'continuous_r3_qualification'):
        raise ValueError('reviewed source qualification mismatch')
    manifest = read_json(pilot/'manifest.json')
    source.verify_files(manifest)
    claim_stage(review_path, 'census', output_dir)
    out = _new_output(output_dir)
    limit = review['limits']['census']
    budget = source._Budget(limit['seconds'], limit['bytes'])
    try:
        with source._deadline(limit['seconds']):
            budget.save(out/'launch.json', {'review_sha256': source.sha(review_path), 'review': review})
            policy = protocol()
            minute = source.load_minutes(policy['seed'], policy['source_end_exclusive'])
            hourly = source.hourly_from_minutes(minute, policy['seed'], policy['source_end_exclusive'])
            old_parents = read_json(pilot/'parent_ledgers.json')
            parents = _parents(hourly, manifest, old_parents)
            for key in parents:
                verify_parent_prefix(parents[key], old_parents[key], policy['pilot_end'])
            budget.save(out/'parent_ledgers.json', parents)
            print('CENSUS_PROGRESS qualified continuous parents', len(hourly), 'hours', flush=True)
            prefix = read_json(pilot/'r3_census.json')
            suffix = build_r3_census(minute.loc[minute.index >= pd.Timestamp(policy['pilot_end'])], parents['4H_N3'],
                                      instrument='BTC', data_stream_id=manifest['data_stream_id'], emit_from=policy['start'],
                                      end_exclusive=policy['source_end_exclusive'], checkpoint=prefix['checkpoint'])
            census = merge_censuses(prefix, suffix)
            budget.save(out/'r3_census.json', census)
            raw, arms = book_inputs(census, start=policy['start'], end=policy['end_exclusive'])
            blockers = [b['reason'] for b in census['blockers']]
            for name, inputs in arms.items():
                if any(o['source_status'] != 'complete' for o in inputs['opportunities']):
                    blockers.append('incomplete_' + name + '_source_windows')
            source.verify_files(manifest)
            if review['reviewed_files'] != launch_file_hashes():
                raise ValueError('reviewed code/source changed during census')
            return _complete(out, budget, stage='full_r3_source_census', raw_opportunities=len(raw),
                             signal_counts={name: len(inputs['signals']) for name, inputs in arms.items()},
                             r3_blockers=sorted(set(blockers)), r1_blockers=pilot_receipt['cohort_blockers']['R1'],
                             economic_outcomes_computed=False, source_end_exclusive=policy['source_end_exclusive'])
    except BaseException as exc:
        _failure(out, exc)
        raise


def run_score(census_dir, output_dir, review_path):
    review = load_review(review_path, 'score')
    directory = Path(census_dir)
    claim_stage(review_path, 'score', output_dir)
    out = _new_output(output_dir)
    limit = review['limits']['score']
    budget = source._Budget(limit['seconds'], limit['bytes'])
    try:
        with source._deadline(limit['seconds']):
            receipt = verify_artifacts(directory)
            launch = read_json(directory/'launch.json')
            if (receipt['stage'] != 'full_r3_source_census' or receipt['r3_blockers']
                    or launch['review_sha256'] != source.sha(review_path)):
                raise ValueError('qualified complete reviewed census required before economic reveal')
            census = read_json(directory/'r3_census.json')
            policy = protocol()
            raw, arms = book_inputs(census, start=policy['start'], end=policy['end_exclusive'])
            if any(o['source_status'] != 'complete' for inputs in arms.values() for o in inputs['opportunities']):
                raise ValueError('incomplete source dispositions')
            parents = read_json(directory/'parent_ledgers.json')['4H_N3']
            minute = source.load_minutes(policy['start'], policy['source_end_exclusive'])
            source.hourly_from_minutes(minute, policy['start'], policy['source_end_exclusive'])
            budget.save(out/'launch.json', {'review_sha256': source.sha(review_path), 'census_receipt_sha256': source.sha(directory/'receipt.json')})
            summaries, primary_books, primary_summary, blockers = {}, {}, {}, set()
            def collect_blockers(book, summary, key, arm):
                blockers.update(key + ':' + arm + ':' + b['reason'] for b in book['blockers'])
                if summary['unresolved']:
                    blockers.add(key + ':' + arm + ':incomplete_outcomes')
            for occupied in (False, True):
                scope = 'occupied' if occupied else 'fixed_event'
                for cost, delay in SCENARIOS:
                    key = f'{scope}_{cost}bps_{delay}s'
                    results = {}
                    for arm, inputs in arms.items():
                        budget.check()
                        print('SCORE_PROGRESS', key, arm, flush=True)
                        book = replay_book(minute, inputs['opportunities'], inputs['signals'], as_of=policy['source_end_exclusive'],
                                            cost_bps=cost, delay_seconds=delay, funding_mode=review['funding_mode'],
                                            parent_down=parents['transitions'], occupied=occupied)
                        budget.save(out/(key + '_' + arm + '.json'), book)
                        summary = summarize_book(book, raw, minute)
                        collect_blockers(book, summary, key, arm)
                        budget.save(out/(key + '_' + arm + '_diagnostics.json'), summary)
                        results[arm] = book
                        summaries.setdefault(key, {})[arm] = {k: v for k, v in summary.items() if k not in ('liquidation_path', 'excursions')}
                        if occupied and cost == 12 and delay == 5:
                            primary_books[arm], primary_summary[arm] = book, summary
                    months = paired_months(raw, results['baseline'], results['repair'], MONTHS)
                    budget.save(out/(key + '_months.json'), months)
                    if occupied and cost == 12 and delay == 5:
                        primary_months = months
            # Never choose zero funding as primary after seeing the stress results.
            for arm, inputs in arms.items():
                print('SCORE_PROGRESS zero_funding_diagnostic', arm, flush=True)
                book = replay_book(minute, inputs['opportunities'], inputs['signals'], as_of=policy['source_end_exclusive'],
                                    cost_bps=12, delay_seconds=5, funding_mode='zero_diagnostic',
                                    parent_down=parents['transitions'], occupied=True)
                budget.save(out/('zero_funding_diagnostic_' + arm + '.json'), book)
                diagnostic = summarize_book(book, raw)
                collect_blockers(book, diagnostic, 'zero_funding_diagnostic', arm)
                summaries.setdefault('zero_funding_diagnostic', {})[arm] = {k: v for k, v in diagnostic.items() if k not in ('liquidation_path', 'excursions')}
            interval = paired_interval(primary_months)
            repair = primary_summary['repair']
            scenario_nets = [summaries[f'occupied_{cost}bps_{delay}s']['repair']['net_pnl'] for cost, delay in SCENARIOS]
            if any(r['unresolved_count'] for r in primary_months):
                blockers.add('incomplete_paired_outcomes')
            blockers = sorted(blockers)
            decision = campaign_decision({'required_data_ok': not blockers, 'blockers': blockers,
                'repair_fills': repair['completed_fills'], 'origin_months_with_repair_fills': repair['origin_months_with_fills'],
                'undefined_fraction': interval['undefined_fraction'], 'primary_repair_net': repair['net_pnl'],
                'incremental_estimate': interval['estimate'], 'scenario_repair_net': scenario_nets,
                'positive_blocks': repair['positive_blocks'], 'repair_net_without_top3': repair['net_without_top_three_winners'],
                'incremental_lower_bound': interval['lower']})
            control = {r['opportunity_id']: r for r in primary_books['baseline']['rows']}
            skipped = [control[r['opportunity_id']] for r in primary_books['repair']['rows']
                       if r['position'] is None and r['net_pnl'] == 0 and control[r['opportunity_id']]['net_pnl'] is not None]
            result = {'family': 'R3', 'decision': decision, 'baseline': summaries['occupied_12bps_5s']['baseline'],
                      'repair': summaries['occupied_12bps_5s']['repair'], 'interval': interval, 'scenarios': summaries,
                      'primary_months': primary_months, 'blockers': blockers,
                      'r1_status': 'blocked', 'r1_blockers': receipt['r1_blockers'],
                      'skipped_baseline_winners': sum(r['net_pnl'] > 0 for r in skipped),
                      'avoided_baseline_losers': sum(r['net_pnl'] < 0 for r in skipped),
                      'funding_mode_locked': review['funding_mode'], 'execution_authorized': False}
            if review['reviewed_files'] != launch_file_hashes():
                raise ValueError('reviewed source/code changed during economic replay')
            budget.save(out/'comparison.json', result)
            text = render_report(result)
            budget.check(len(text.encode()))
            with (out/'comparison.txt').open('x') as handle:
                handle.write(text)
            return _complete(out, budget, stage='frozen_r3_economic_comparison', decision=decision,
                             r1_status='blocked', economic_outcomes_computed=True)
    except BaseException as exc:
        _failure(out, exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    qualification = commands.add_parser('qualify')
    qualification.add_argument('--pilot-dir', required=True)
    qualification.add_argument('--output-dir', required=True)
    census = commands.add_parser('census')
    for name in ('pilot-dir', 'qualification-dir', 'output-dir', 'review'):
        census.add_argument('--' + name, required=True)
    score = commands.add_parser('score')
    for name in ('census-dir', 'output-dir', 'review'):
        score.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    if args.stage == 'qualify':
        result = qualify_continuous(args.pilot_dir, args.output_dir)
    elif args.stage == 'census':
        result = run_census(args.pilot_dir, args.qualification_dir, args.output_dir, args.review)
    else:
        result = run_score(args.census_dir, args.output_dir, args.review)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
