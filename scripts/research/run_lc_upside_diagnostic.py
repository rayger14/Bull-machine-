"""Reproducible two-stage local LC study. Never invokes models or live services.

Run from repository root: python3 -m scripts.research.run_lc_upside_diagnostic
prepare (source facts only), then score (requires immutable prepared bindings).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import talib

from scripts.research.lc_campaign import _sha_file
from scripts.research.lc_campaign_source import _hourly_from_minutes
from scripts.research.lc_context_facts import describe_lc_context
from scripts.research.lc_judgment_runner import _digest, _load, _save_equal, _verify_digest
from scripts.research.lc_master_assessment import build_lc_packet
from scripts.research.lc_mechanical_extension import prepare_cases
from scripts.research.lc_upside_diagnostic import (
    LABELS, context_labels, event_result, hourly_features, match_controls,
    paired_summary, summarize_events,
)
from scripts.research.lc_upside_scorecard import upside_cases
from scripts.research.replay_clock import json_safe


BASE = Path('results/lc_mechanical_extension_2026_09_22/run_v1')
BASELINE = Path('results/lc_upside_baseline_2026_09_30/run_v1/result.json')
ARCHIVE = Path('data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet')
OUTPUT = Path('results/lc_upside_diagnostic_2026_09_30/run_v2')
PROTOCOL = Path('docs/knowledge/lc_upside_diagnostic_protocol_2026_09_30.md')


def verify_bindings(files):
    for path, expected in files.items():
        if _sha_file(path) != expected:
            raise ValueError('changed input binding: '+str(path))


def _minutes(start, end):
    return pd.read_parquet(ARCHIVE, columns=['open', 'high', 'low', 'close', 'vol'],
                           filters=[('ts', '>=', pd.Timestamp(start).to_pydatetime()),
                                    ('ts', '<', pd.Timestamp(end).to_pydatetime())]).rename(
                                        columns={'vol': 'volume'})


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    original = _verify_digest(BASE/'input_lock.json')
    files = dict(original['files'])
    for path in [BASE/'input_lock.json', BASE/'cases.json', BASELINE, PROTOCOL,
                 Path('tests/research/test_lc_upside_diagnostic.py'),
                 *Path('scripts/research').glob('*.py')]:
        key = str(path.resolve())
        actual = _sha_file(path)
        if key in files and files[key] != actual:
            raise ValueError('original binding changed: '+key)
        files[key] = actual
    verify_bindings(files)
    if _sha_file(BASELINE) != 'a15da5b8f7e096164e387e41a30af87ecd63de1751fae4a9acbae1ec3161faab':
        raise ValueError('baseline result differs')
    sources = [(path, json.loads(Path(path).read_bytes())) for path in original['files']
               if path.endswith('/source.json')]
    sources.sort(key=lambda row: pd.Timestamp(row[1]['start']))
    start = min(pd.Timestamp(s['seed']) for _, s in sources)
    end = max(pd.Timestamp(s['end_exclusive']) for _, s in sources)
    minute = _minutes(start, end)
    feature_frames, raw_cases, by_id = [], [], {}
    parity = 0
    for path, source in sources:
        seed, finish = pd.Timestamp(source['seed']), pd.Timestamp(source['end_exclusive'])
        if source['runtime']['talib'] != 'python-talib={};library={}'.format(
                talib.__version__, talib.__ta_version__.decode()):
            raise ValueError('ATR runtime differs from source')
        monthly = minute.loc[(minute.index >= seed) & (minute.index < finish)]
        hourly = _hourly_from_minutes(monthly, seed, finish, max_input_hours=2000)
        feature = hourly_features(hourly, source['start'], source['end_exclusive'])
        feature_frames.append(feature)
        for raw in source['candidates']:
            d = pd.Timestamp(raw['decision_time'])
            recomputed = feature.loc[d]
            for k in ('open', 'high', 'low', 'close'):
                if not math.isclose(float(hourly.loc[d-pd.Timedelta('1h'), k]),
                                    raw['features'][k], rel_tol=0, abs_tol=1e-8):
                    raise ValueError('source candle mismatch: '+raw['candidate_id'])
            if not math.isclose(recomputed['atr'], raw['features']['atr_14'], rel_tol=0, abs_tol=1e-8):
                raise ValueError('source ATR mismatch: '+raw['candidate_id'])
            expected_previous = raw['previous_features']['atr_14']/raw['previous_features']['close']
            if not math.isclose(recomputed['previous_atr_pct'], expected_previous, rel_tol=0, abs_tol=1e-12):
                raise ValueError('source previous ATR mismatch: '+raw['candidate_id'])
            if raw['candidate_id'] in by_id:
                raise ValueError('duplicate source candidate')
            raw_cases.append(raw)
            by_id[raw['candidate_id']] = (raw, source, path)
            parity += 1
    features = pd.concat(feature_frames).sort_index()
    if not features.index.is_unique:
        raise ValueError('overlapping source census months')
    cases = _load(BASE/'cases.json')
    if prepare_cases(raw_cases, minute) != cases:
        raise ValueError('frozen cases do not reconstruct')
    selected = upside_cases(cases)
    matches = match_controls(selected, features, [r['decision_time'] for r in raw_cases])
    plans = {c['candidate_id']: c['plans']['immediate'] for c in selected}
    for pair in matches:
        if pair['status'] != 'matched':
            continue
        d = pd.Timestamp(pair['control_time'])
        row = features.loc[d]
        plans[pair['control_id']] = dict(decision_time=d.isoformat(), action='enter', level=None,
            stop=float(row['close']-2.7*row['atr']), entry_expiry=(d+pd.Timedelta('15min')).isoformat(),
            exit_deadline=(d+pd.Timedelta('1d')).isoformat(), processing_seconds=90,
            routing_seconds=0, notional=50000., cost_bps=12)
    contexts = []
    for i, case in enumerate(selected):
        raw, source, path = by_id[case['candidate_id']]
        provenance = {k: source[k] for k in ('data_stream_id', 'source_sha256', 'runtime',
                       'atr_contract', 'source_manifest', 'code_manifest', 'config_manifest',
                       'missing_input_limits', 'replay_blockers')}
        provenance.update(instrument='BTC', reconstruction_verified=True,
                          source_artifact_sha256=_sha_file(path))
        d = pd.Timestamp(case['decision_time'])
        # Keep the packet boundary physically free of postdecision bars.
        prefix = minute.loc[(minute.index >= pd.Timestamp(source['seed'])) & (minute.index < d)]
        packet = build_lc_packet(raw, prefix, source['parent_ledgers'], provenance, case['candidate_id'])
        facts = describe_lc_context(packet)
        labels = context_labels(facts, close=raw['features']['close'], stop=case['plans']['immediate']['stop'])
        contexts.append(dict(candidate_id=case['candidate_id'], decision_time=case['decision_time'],
                             facts=facts, labels=labels, source_path=path))
        if (i+1) % 10 == 0:
            print('source contexts', i+1, '/', len(selected), flush=True)
    outputs = dict(features=json_safe(features.rename_axis('decision_time').reset_index().to_dict('records')),
                   contexts=contexts, matches=matches, plans=plans)
    verify_bindings(files)
    for name, value in outputs.items():
        path = _save_equal(output/(name+'.json'), value)
        files[str(path)] = _sha_file(path)
    preflight = dict(version='lc_upside_diagnostic_v1', files=files,
                     source_candidate_parity=parity, source_months=len(sources),
                     hourly_rows=len(features), selected_count=len(selected),
                     matched_count=sum(p['status'] == 'matched' for p in matches),
                     runtime=dict(pandas=pd.__version__, numpy=np.__version__, talib=talib.__version__),
                     first_month='2024-01', last_month='2026-07',
                     pristine_holdout=False, execution_authorized=False, market_calls=0)
    preflight['sha256'] = _digest(preflight)
    _save_equal(output/'preflight.json', preflight)
    print(json.dumps({k: v for k, v in preflight.items() if k != 'files'}, sort_keys=True), flush=True)
    return preflight


def score(output=OUTPUT):
    output = Path(output).resolve()
    preflight = _verify_digest(output/'preflight.json')
    verify_bindings(preflight['files'])
    consumed = [output/(n+'.json') for n in ('plans', 'matches', 'contexts')]
    consumed += [ARCHIVE, BASELINE, *Path('scripts/research').glob('*.py')]
    for path in consumed:
        key = str(path.resolve())
        if key not in preflight['files'] or _sha_file(path) != preflight['files'][key]:
            raise ValueError('unbound consumed input: '+key)
    plans, matches, contexts = (_load(output/(n+'.json')) for n in ('plans', 'matches', 'contexts'))
    events, complete_windows = {}, 0
    for cid, plan in plans.items():
        d, end = pd.Timestamp(plan['decision_time']), pd.Timestamp(plan['exit_deadline'])
        bars = _minutes(d, end+pd.Timedelta('1min'))
        complete_windows += int(bars.index.equals(pd.date_range(d, end, freq='min')))
        events[cid] = event_result(plan, bars)
    # Economic parity against the isolated, already exposed immediate baseline.
    baseline = _load(BASELINE)['replay']['scenarios']['12bps_90s']['immediate']['ledger']
    for row in baseline:
        event, position = events[row['candidate_id']], row['position']
        if row['status'] != 'admitted' or not position or position['status'] != 'closed':
            raise ValueError('baseline requires explicit admission parity review')
        if not event['resolved'] or not math.isclose(event['net_pnl'], position['net_pnl'], abs_tol=1e-7, rel_tol=0):
            raise ValueError('LC outcome parity mismatch')
    years = ('2024', '2025', '2026')
    groups = {}
    for label, categories in LABELS.items():
        groups[label] = {}
        for category in categories:
            members = [c for c in contexts if c['labels'][label] == category]
            groups[label][category] = dict(
                overall=summarize_events([events[c['candidate_id']] for c in members]),
                annual={year: summarize_events([events[c['candidate_id']] for c in members
                         if c['decision_time'].startswith(year)]) for year in years})
    paired = paired_summary(matches, events, preflight['first_month'], preflight['last_month'])
    annual_pairs = {year: paired_summary([p for p in matches if p['decision_time'].startswith(year)],
                    events, preflight['first_month'], preflight['last_month']) for year in years}
    times = sorted(pd.Timestamp(plan['decision_time']) for plan in plans.values())
    overlaps = sum(b-a < pd.Timedelta('1d') for i, a in enumerate(times) for b in times[i+1:])
    result = dict(version='lc_upside_diagnostic_v1', preflight_sha256=preflight['sha256'],
                  lc_all=summarize_events([events[c['candidate_id']] for c in contexts]),
                  groups=groups, paired=paired, annual_pairs=annual_pairs,
                  events=events, complete_minute_windows=complete_windows,
                  event_count=len(plans), overlapping_event_window_pairs=int(overlaps),
                  baseline_parity_count=len(baseline), pristine_holdout=False,
                  execution_authorized=False, profitability_certified=False)
    verify_bindings(preflight['files'])
    # Detect a modified preflight during scoring as well as modified bound leaves.
    if _verify_digest(output/'preflight.json') != preflight:
        raise ValueError('preflight changed during scoring')
    _save_equal(output/'result.json', result)
    print(json.dumps(dict(lc=result['lc_all'], paired=paired, events=len(plans),
                          complete_windows=complete_windows, overlap_pairs=int(overlaps)), sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'score'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else score)(args.output)
