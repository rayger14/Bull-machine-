"""Offline known-room study: source/plan freeze, then independent minute replay.

python3 -m scripts.research.lc_august_receipt
python3 -m scripts.research.run_lc_room_validation prepare
python3 -m scripts.research.run_lc_room_validation score
No model calls, network, live changes or tuning. Old artifacts stay untouched.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import pandas as pd
import talib

from scripts.research.lc_august_source import STUDY, SEED, START, END
from scripts.research.lc_august_receipt import OUTPUT
from scripts.research.lc_campaign_source import (
    ARCHIVE, ROOT, _hourly_from_minutes, _merge_hashes, _sha, _verify_hashes,
)
from scripts.research.lc_context_facts import describe_lc_context
from scripts.research.lc_judgment_runner import _digest, _load, _save_equal, _verify_digest
from scripts.research.lc_master_assessment import build_lc_packet
from scripts.research.lc_mechanical_extension import prepare_cases
from scripts.research.lc_room_validation import compile_room_plans, score_room
from scripts.research.lc_upside_diagnostic import context_labels, event_result
from scripts.research.replay_clock import json_safe


DIAGNOSTIC = ROOT/'results/lc_upside_diagnostic_2026_09_30/run_v2'
BASE = ROOT/'results/lc_mechanical_extension_2026_09_22/run_v1'
BASELINE = ROOT/'results/lc_upside_baseline_2026_09_30/run_v1/result.json'
COHORTS = {'discovery': ('2024-01', '2026-07'), 'august_replication': ('2026-08', '2026-08')}


def require_bound(files, paths):
    for path in paths:
        path = Path(path).resolve()
        if str(path) not in files or _sha(path) != files[str(path)]:
            raise ValueError('unbound or changed consumed input: ' + str(path))


def _minutes(start, end):
    return pd.read_parquet(ARCHIVE, columns=['open', 'high', 'low', 'close', 'vol'],
        filters=[('ts', '>=', pd.Timestamp(start).to_pydatetime()),
                 ('ts', '<', pd.Timestamp(end).to_pydatetime())]).rename(columns={'vol':'volume'})


def _active_code():
    return sorted((ROOT/'scripts/research').glob('*.py')) + [
        ROOT/'tests/research/test_lc_august_source.py',
        ROOT/'tests/research/test_lc_august_receipt.py',
        ROOT/'tests/research/test_lc_room_validation.py']


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    rule = _verify_digest(STUDY/'rule_lock.json')
    exposure = _verify_digest(STUDY/'exposure.json')
    old = _verify_digest(DIAGNOSTIC/'preflight.json')
    receipt = _verify_digest(output/'source_receipt.json')
    files = {}
    for name, mapping in [('rule',rule['files']), ('exposure',exposure['evidence']),
                          ('discovery',old['files']), ('source',receipt['files'])]:
        _merge_hashes(files, mapping, name)
    require_bound(files, [DIAGNOSTIC/'contexts.json', BASE/'cases.json', BASELINE,
                          output/'august_source.json', output/'source_launch.json', ARCHIVE])
    for path in [STUDY/'rule_lock.json', STUDY/'exposure.json', DIAGNOSTIC/'preflight.json',
                 output/'source_receipt.json', *_active_code()]:
        _merge_hashes(files, {str(path): _sha(path)}, 'current consumed input')
    _verify_hashes(files)
    old_cases, old_contexts = _load(BASE/'cases.json'), _load(DIAGNOSTIC/'contexts.json')
    raw_old = {}
    for path in old['files']:
        if path.endswith('/source.json'):
            for raw in json.loads(Path(path).read_bytes())['candidates']:
                if raw['candidate_id'] in raw_old:
                    raise ValueError('duplicate discovery source identity')
                raw_old[raw['candidate_id']] = raw
    if set(raw_old) != {c['candidate_id'] for c in old_cases}:
        raise ValueError('discovery source population differs')
    old_by_id = {c['candidate_id']: c for c in old_cases}
    for context in old_contexts:
        cid = context['candidate_id']
        labels = context_labels(context['facts'], close=raw_old[cid]['features']['close'],
                                stop=old_by_id[cid]['plans']['immediate']['stop'])
        if labels != context['labels']:
            raise ValueError('discovery context labels differ')
    source = _load(output/'august_source.json')
    if source['runtime']['talib'] != 'python-talib={};library={}'.format(
            talib.__version__, talib.__ta_version__.decode()):
        raise ValueError('August ATR runtime differs')
    minute = _minutes(SEED, END)
    hourly = _hourly_from_minutes(minute, SEED, END, max_input_hours=2000)
    atr = pd.Series(talib.ATR(hourly.high.to_numpy(), hourly.low.to_numpy(),
                             hourly.close.to_numpy(), timeperiod=14), index=hourly.index)
    raw_cases = source['candidates']
    for raw in raw_cases:
        d = pd.Timestamp(raw['decision_time'])
        for offset, field in [(1, 'features'), (2, 'previous_features')]:
            at = d-pd.Timedelta(hours=offset)
            for k in ('open','high','low','close','volume'):
                if not math.isclose(float(hourly.loc[at,k]), raw[field][k], abs_tol=1e-8, rel_tol=0):
                    raise ValueError('August raw candle mismatch')
            if not math.isclose(float(atr.loc[at]), raw[field]['atr_14'], abs_tol=1e-8, rel_tol=0):
                raise ValueError('August raw ATR mismatch')
    new_cases = prepare_cases(raw_cases, minute) if raw_cases else []
    new_contexts, packets = [], []
    by_id = {r['candidate_id']: r for r in raw_cases}
    for case in new_cases:
        if case['subtype'] != 'upside_expansion_candidate':
            continue
        raw = by_id[case['candidate_id']]
        provenance = {k: source[k] for k in ('data_stream_id','source_sha256','runtime',
            'atr_contract','source_manifest','code_manifest','config_manifest',
            'missing_input_limits','replay_blockers')}
        provenance.update(instrument='BTC', reconstruction_verified=True,
                          source_artifact_sha256=_sha(output/'august_source.json'))
        # Physically restrict the structural packet to bars known at the decision.
        prefix = minute.loc[minute.index < pd.Timestamp(case['decision_time'])]
        packet = build_lc_packet(raw, prefix, source['parent_ledgers'], provenance, case['candidate_id'])
        facts = describe_lc_context(packet)
        labels = context_labels(facts, close=raw['features']['close'],
                                stop=case['plans']['immediate']['stop'])
        new_contexts.append(dict(candidate_id=case['candidate_id'], decision_time=case['decision_time'],
                                 labels=labels, facts=facts))
        packets.append(packet)
    payloads = {}
    for name, cases, contexts in [('discovery',old_cases,old_contexts),
                                   ('august_replication',new_cases,new_contexts)]:
        payloads[name+'_cases'] = cases
        payloads[name+'_contexts'] = contexts
        payloads[name+'_plans'] = compile_room_plans(cases, contexts)
    payloads['august_packets'] = packets
    _verify_hashes(files)
    for name, value in payloads.items():
        path = _save_equal(output/(name+'.json'), json_safe(value))
        files[str(path)] = _sha(path)
    lock = dict(version='lc_room_input_lock_v1', files=files,
                rule_lock_sha256=rule['sha256'], exposure_sha256=exposure['sha256'],
                cohorts={name: dict(first_month=dates[0], last_month=dates[1],
                    source_count=len(payloads[name+'_cases']),
                    subtype_counts=dict(Counter(c['subtype'] for c in payloads[name+'_cases'])),
                    room_counts=dict(Counter(c['labels']['mapped_overhead']
                                     for c in payloads[name+'_contexts'])))
                    for name, dates in COHORTS.items()},
                august_source_parity=len(raw_cases), pristine_holdout=False,
                execution_authorized=False, outcome_scoring_started=False, model_calls=0)
    lock['sha256'] = _digest(lock)
    _save_equal(output/'input_lock.json', lock)
    print(json.dumps({k:v for k,v in lock.items() if k != 'files'}, sort_keys=True), flush=True)
    return lock


def score(output=OUTPUT):
    output = Path(output).resolve()
    lock = _verify_digest(output/'input_lock.json')
    _verify_hashes(lock['files'])
    consumed = [output/(name+'_'+part+'.json') for name in COHORTS
                for part in ('cases','contexts','plans')]
    require_bound(lock['files'], consumed+[ARCHIVE, BASELINE, *_active_code()])
    baseline = _load(BASELINE)
    results, verification = {}, {}
    for name, dates in COHORTS.items():
        cases, contexts, plans = (_load(output/(name+'_'+part+'.json'))
                                  for part in ('cases','contexts','plans'))
        if compile_room_plans(cases, contexts) != plans:
            raise ValueError('frozen plans do not reconstruct')
        first = min((pd.Timestamp(c['decision_time']) for c in cases), default=START)
        last = max((pd.Timestamp(c['decision_time']) for c in cases), default=START)+pd.Timedelta('1d')
        bars = _minutes(first, last+pd.Timedelta('1min'))
        result = score_room(cases, contexts, bars, first_month=dates[0], last_month=dates[1])
        checked, complete_windows = 0, 0
        for candidate in plans['baseline']:
            d = pd.Timestamp(candidate['decision_time']); end = d+pd.Timedelta('1d')
            complete_windows += int(bars.loc[(bars.index>=d)&(bars.index<=end)].index.equals(
                pd.date_range(d,end,freq='min')))
        for scenario, value in result['scenarios'].items():
            cost, delay = scenario.replace('bps','').replace('s','').split('_')
            if name == 'discovery':
                expected = baseline['replay']['scenarios'][scenario]['immediate']['ledger']
                if value['books']['baseline']['ledger'] != expected:
                    raise ValueError('discovery baseline ledger differs: '+scenario)
            for arm, book in value['books'].items():
                for row, candidate in zip(book['ledger'], plans[arm]):
                    if row['candidate_id'] != candidate['candidate_id']:
                        raise ValueError('direct verification identity differs')
                    if candidate['plan'] is None or row['status'] in ('skipped_busy','admission_indeterminate'):
                        continue
                    plan = dict(candidate['plan'], cost_bps=int(cost), processing_seconds=int(delay))
                    d, end = pd.Timestamp(plan['decision_time']), pd.Timestamp(plan['exit_deadline'])
                    event = event_result(plan, bars.loc[(bars.index>=d)&(bars.index<=end)])
                    pos = row['position']
                    if pos and pos['status'] == 'closed':
                        if not event['resolved'] or not math.isclose(event['net_pnl'], pos['net_pnl'],
                                                                   abs_tol=1e-7,rel_tol=0):
                            raise ValueError('direct minute outcome differs')
                        for field in ('entry_price','exit_price','exit_reason'):
                            if event['raw']['outcome'][field] != pos[field]:
                                raise ValueError('direct bracket field differs: '+field)
                    elif row['status'] in ('rejected','expired','cancelled'):
                        if not event['resolved'] or event['net_pnl'] != 0:
                            raise ValueError('direct nonentry differs')
                    elif event['resolved']:
                        raise ValueError('unresolved book cannot match resolved direct outcome')
                    checked += 1
        results[name] = result
        verification[name] = dict(direct_comparisons=checked, selected=len(plans['baseline']),
            complete_24h_windows=complete_windows,
            discovery_baseline_parity_all_four_scenarios=(name=='discovery'))
        print(name, 'scored', len(plans['baseline']), 'upside candidates', flush=True)
    _verify_hashes(lock['files'])
    require_bound(lock['files'], consumed+[ARCHIVE, BASELINE, *_active_code()])
    published = dict(version='lc_room_result_v1', input_lock_sha256=lock['sha256'],
                     cohorts=results, pristine_holdout=False, model_calls=0,
                     execution_authorized=False, profitability_certified=False)
    result_path = _save_equal(output/'result.json', json_safe(published))
    verification.update(result_sha256=_sha(result_path), input_lock_sha256=lock['sha256'],
                        file_bindings_verified=len(lock['files']))
    _save_equal(output/'verification.json', json_safe(verification))
    print(json.dumps(verification, sort_keys=True), flush=True)
    return published


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','score'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else score)(args.output)
