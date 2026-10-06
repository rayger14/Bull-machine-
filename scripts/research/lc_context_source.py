"""Outcome-free LC census qualification using existing local, pinned artifacts."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import talib

from scripts.research import causal_parent_ledger as native
from scripts.research.lc_context_contract import clock, protocol, seal
from scripts.research.lc_context_controller import classify
from scripts.research.lc_context_evidence import OHLCV, prepare_evidence
from scripts.research.lc_mechanical_extension import prepare_cases
from scripts.research.replay_clock import digest
from scripts.research.study_source import sha, hourly_from_minutes, load_minutes, _Budget, _deadline

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT/'results/lc_context_study_2026_10_02'
ARCHIVE = ROOT/'data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet'
ARCHIVE_SHA = '5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035'
STREAM = 'btc_1m_2021_2026_saved_5b8a4533f70b8ccd'
BASE = ROOT/'results/lc_mechanical_extension_2026_09_22/run_v1'
AUGUST = ROOT/'results/lc_room_validation_2026_09_30/run_v2'
PARENTS = ROOT/'results/archetype_study_2026_10_01/census_v1'
MONTHS = [str(m) for m in pd.period_range('2024-01', '2026-08', freq='M')]
PINNED = {
    BASE/'input_lock.json': '4edf468e4181897d2e1595138858458cd59c0eab03d3355137e8eb668f6de4ec',
    AUGUST/'source_receipt.json': '451c33fd7f718da4fe3eb1556da0f0a7652bc3c287d759fc6770ddcda0b636ac',
    PARENTS/'receipt.json': 'ce46283a7180b9f70d69e8c6ed871cdb1325e960f674fccc7137d92180e5d0e5',
}


def checked_read(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError('bound source changed: '+str(path))
    return json.loads(Path(path).read_text())


def verify_files(files):
    for path, expected in files.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError('bound source changed: '+str(path))


def read_receipt(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != seal({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('receipt digest mismatch')
    return value


def monthly_paths(files):
    return {p: value for p, value in files.items() if p.endswith('/source.json')}


def new_output(path, *, root=OUTPUT_ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('new study subdirectory required')
    if path.exists():
        raise FileExistsError('output exists; no overwrite or automatic retry')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def implementation_files():
    paths = list(ROOT.joinpath('scripts/research').glob('lc_context_*.py'))
    paths += list(ROOT.joinpath('tests/research').glob('*lc_context_*.py'))
    paths += [ROOT/'scripts/research'/name for name in (
        'run_lc_context_study.py', 'study_source.py', 'study_parents.py', 'study_execution.py',
        'study_contract.py', 'r3_census.py', 'causal_parent_ledger.py', 'replay_clock.py',
        'virtual_book_replay.py', 'lc_mechanical_extension.py')]
    paths += [ROOT/'docs/superpowers/specs/2026-10-02-mechanical-lc-context-design.md',
              ROOT/'docs/superpowers/plans/2026-10-02-mechanical-lc-context.md']
    return {str(p): sha(p) for p in sorted(set(paths)) if p.is_file()}


def reconcile_sources(sources, expected_ids, *, stream=STREAM, months=MONTHS):
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError('duplicate expected candidate ID')
    if sorted(s['month'] for s in sources) != sorted(months):
        raise ValueError('source calendar incomplete or duplicated')
    rows, ids, times = [], set(), set()
    for source in sources:
        month = pd.Period(source['month'], 'M')
        start = month.start_time.tz_localize('UTC')
        end = (month+1).start_time.tz_localize('UTC')
        if (clock(source['start']) != start or clock(source['end_exclusive']) != end
                or clock(source['seed']) != start-pd.Timedelta('30d')
                or source['data_stream_id'] != stream
                or source['candidate_count'] != len(source['candidates'])):
            raise ValueError('source calendar/identity/count mismatch')
        for raw in source['candidates']:
            t, cid = clock(raw['decision_time']), raw['candidate_id']
            if (cid != 'hourly-lc:'+t.isoformat() or cid in ids or t in times
                    or not start <= t < end or raw['track'] != 'hourly'):
                raise ValueError('duplicate, foreign or outside-calendar candidate')
            ids.add(cid); times.add(t); rows.append(raw)
    if ids != set(expected_ids):
        raise ValueError('candidate denominator differs from saved census')
    return sorted(rows, key=lambda r: clock(r['decision_time']))


def qualify_month(source, hourly):
    seed, end = clock(source['seed']), clock(source['end_exclusive'])
    hours = hourly.loc[seed:end-pd.Timedelta('1h'), OHLCV]
    if not hours.index.equals(pd.date_range(seed, end, freq='h', inclusive='left')):
        raise ValueError('monthly hourly input incomplete')
    input_hash = digest([{'timestamp': str(t), **r} for t, r in zip(hours.index, hours.to_dict('records'))])
    if input_hash != source['hourly_input_hash']:
        raise ValueError('monthly hourly input hash differs')
    atr = pd.Series(talib.ATR(hours.high.to_numpy(), hours.low.to_numpy(), hours.close.to_numpy(),
                             timeperiod=14), index=hours.index)
    for raw in source['candidates']:
        t = clock(raw['decision_time'])
        for delta, field in [(1, 'features'), (2, 'previous_features')]:
            at = t-pd.Timedelta(hours=delta)
            for key in OHLCV:
                if not math.isclose(raw[field][key], hours.loc[at, key], abs_tol=1e-8, rel_tol=0):
                    raise ValueError('native hourly reconstruction differs')
            value = raw[field].get('atr_14')
            if value is not None and isinstance(value, (float, int)) and math.isfinite(value):
                if not math.isclose(value, atr.loc[at], abs_tol=1e-8, rel_tol=0):
                    raise ValueError('native monthly ATR reconstruction differs')
    return {'month': source['month'], 'candidate_count': len(source['candidates']),
            'hourly_input_hash': input_hash, 'hourly_rows': len(hours)}


def verify_anchors(ledger, minute, timeframe):
    delta = pd.Timedelta(timeframe.lower())
    buckets = ledger['anchor_buckets']
    if buckets['incomplete'] or buckets['developing']:
        raise ValueError('anchor coverage not fully complete')
    grouped = minute.resample(timeframe.lower(), closed='left', label='left')
    frame = grouped.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    if (grouped.size() != int(delta/pd.Timedelta('1min'))).any() or len(frame) != len(buckets['completed']):
        raise ValueError('anchor constituent coverage mismatch')
    for (at, row), bucket in zip(frame.iterrows(), buckets['completed']):
        if (clock(bucket['open_time']) != at or clock(bucket['close_time']) != at+delta
                or bucket['complete'] is not True or bucket['constituents'] != int(delta/pd.Timedelta('1h'))
                or [clock(t) for t in bucket['source_open_times']] != list(pd.date_range(at, at+delta, freq='h', inclusive='left'))):
            raise ValueError('anchor clock/source identity mismatch')
        if any(float(row[k]) != bucket[k] for k in OHLCV[:4]):
            raise ValueError('anchor price mismatch')
        tolerance = 8*math.ulp(max(abs(float(row['volume'])), abs(bucket['volume'])))
        if abs(float(row['volume'])-bucket['volume']) > tolerance:
            raise ValueError('anchor volume mismatch')
    return len(frame)


def qualify_parents(parents, minute, hourly):
    bars = hourly.copy()
    bars['atr_14'] = talib.ATR(bars.high.to_numpy(), bars.low.to_numpy(), bars.close.to_numpy(), timeperiod=14)
    bars['atr_available_at'] = bars.index+pd.Timedelta('1h')
    result = {}
    for name, ledger in parents.items():
        m = ledger['manifest']
        if native._input_hash(bars, 'BTC', STREAM, m['atr_contract']) != m['input_hash']:
            raise ValueError('continuous parent hourly/ATR input hash mismatch')
        if m['atr_contract']['version'] != f'python-talib={talib.__version__};library={talib.__ta_version__.decode()}':
            raise ValueError('parent ATR runtime differs')
        count = verify_anchors(ledger, minute, name.split('_')[0])
        pivots = {p['id']: p for p in ledger['pivots']}
        versions = {v['id']: v for v in ledger['versions']}
        if len(pivots) != len(ledger['pivots']) or len(versions) != len(ledger['versions']):
            raise ValueError('duplicate constructor identities')
        for v in versions.values():
            rebuilt = native._new_version(contract_id=m['contract_id'], data_stream_id=STREAM,
                lineage_id=v['lineage_id'], predecessor=v['predecessor_version_id'], reason=v['creation_reason'],
                source_hour=clock(v['formation_hour']), range_low=v['range_low'], range_high=v['range_high'],
                low_pivot_id=v['low_pivot_id'], high_pivot_id=v['high_pivot_id'])
            if rebuilt['id'] != v['id'] or clock(rebuilt['available_at']) != clock(v['available_at']):
                raise ValueError('parent version identity mismatch')
            for side in ('low', 'high'):
                p = pivots[v[side+'_pivot_id']]
                if p['side'] != side or clock(p['available_at']) > clock(v['formation_hour']):
                    raise ValueError('parent uses future pivot')
            predecessor = v['predecessor_version_id']
            if predecessor and (predecessor not in versions or clock(versions[predecessor]['available_at']) >= clock(v['available_at'])):
                raise ValueError('invalid parent predecessor')
        result[name] = {'hourly_input_hash': m['input_hash'], 'anchor_count': count,
                        'version_count': len(versions), 'live_certified': False}
    return result


def load_inputs():
    files = {str(p): expected for p, expected in PINNED.items()}
    verify_files(files)
    base, august = read_receipt(BASE/'input_lock.json'), read_receipt(AUGUST/'source_receipt.json')
    if august.get('verified_source') is not True:
        raise ValueError('unverified August receipt')
    mappings = monthly_paths(base['files'])
    august_path = AUGUST/'august_source.json'
    mappings[str(august_path)] = august['files'][str(august_path)]
    sources = [checked_read(p, expected) for p, expected in mappings.items()]
    files.update(mappings)
    original = checked_read(BASE/'cases.json', base['files'][str(BASE/'cases.json')])
    files[str(BASE/'cases.json')] = base['files'][str(BASE/'cases.json')]
    august_cases_path = AUGUST/'august_replication_cases.json'
    # This projection includes all four raw cases; never use the two upside packets.
    august_cases = json.loads(august_cases_path.read_text())
    if sha(august_cases_path) != '4b3173cdd188faa7a74e10109f352d953cf550fb1b21145e4ad10ae16e56aae1':
        raise ValueError('August saved case projection changed')
    files[str(august_cases_path)] = sha(august_cases_path)
    expected_cases = original+august_cases
    raw = reconcile_sources(sources, [c['candidate_id'] for c in expected_cases])
    if len(original) != base['case_count'] or len(mappings)-1 != base['source_month_count']:
        raise ValueError('original census receipt count mismatch')
    receipt = json.loads((PARENTS/'receipt.json').read_text())
    parent_path = PARENTS/'parent_ledgers.json'
    parents = checked_read(parent_path, receipt['artifacts']['parent_ledgers.json'])
    files[str(parent_path)] = receipt['artifacts']['parent_ledgers.json']
    if set(parents) != {'4H_N3', '1D_N3'}:
        raise ValueError('required parent timeframes missing')
    files[str(ARCHIVE)] = ARCHIVE_SHA
    for ledger in parents.values():
        m = ledger['manifest']
        if m['instrument'] != 'BTC' or m['data_stream_id'] != STREAM:
            raise ValueError('parent source binding mismatch')
        for key, path in m['source_paths'].items():
            files[path] = m['source_hashes'][key]
        for path, key in [('causal_parent_ledger.py', 'adapter'), ('replay_clock.py', 'replay_clock'),
                          ('virtual_book_replay.py', 'side_effect_guard')]:
            files[str(ROOT/'scripts/research'/path)] = m['helper_hashes'][key]
        files[str(ROOT/'scripts/research/study_parents.py')] = m['study_adapter']['sha256']
    verify_files(files)
    for path, value in implementation_files().items():
        if path in files and files[path] != value:
            raise ValueError('current helper differs from saved construction')
        files[path] = value
    return files, sources, parents, expected_cases, raw


def verify_cut(case, ledgers):
    """Independent brute-force selection/latest anchor witness, no policy actions."""
    t, s = clock(case['decision_time']), clock(case['setup_open'])
    for tf, name in [('4H_N3', 'parent_4h'), ('1D_N3', 'parent_1d')]:
        candidates = []
        for version in ledgers[tf]['versions']:
            if clock(version['available_at']) < s and s-pd.Timedelta('30d') <= clock(version['formation_hour']) < s:
                candidates.append(version)
        expected = sorted(candidates, key=lambda v: (clock(v['available_at']), v['id']))[-1] if candidates else None
        actual = case[name]
        if (actual['bound'] or {}).get('id') != (expected or {}).get('id'):
            raise ValueError('independent parent selection differs')
        if expected is None:
            continue
        anchors = [b for b in ledgers[tf]['anchor_buckets']['completed']
                   if clock(b['open_time']) >= clock(expected['available_at']) and clock(b['close_time']) <= t]
        state = 'not_established'
        for b in anchors:
            c, low, high = b['close'], expected['range_low'], expected['range_high']
            state = 'accepted_above' if c > high else 'accepted_below' if c < low else 'inside' if low < c < high else 'boundary'
        if actual['state'] != state or len(actual['events']) != len(anchors):
            raise ValueError('independent originating acceptance differs')


def prepare(output):
    began = time.monotonic()
    out = new_output(output)
    budget = _Budget(600, 512*1024*1024)
    try:
        with _deadline(600):
            files, sources, parents, expected, raw = load_inputs()
            budget.save(out/'launch.json', {'files': files, 'policy': protocol(),
                'policy_seal': seal(protocol()), 'candidate_ids': [r['candidate_id'] for r in raw],
                'execution_authorized': False, 'economic_outcomes_computed': False})
            print('Pinned census and policy; reconstructing minute/hour/parent evidence.', flush=True)
            minute = load_minutes(protocol()['parent_seed'], protocol()['end_exclusive'])
            hourly = hourly_from_minutes(minute, protocol()['parent_seed'], protocol()['end_exclusive'])
            parent_checks = qualify_parents(parents, minute, hourly)
            month_checks, cases = [], []
            for monthly in sorted(sources, key=lambda s: s['month']):
                budget.check()
                month_checks.append(qualify_month(monthly, hourly))
                source_path = next(p for p in files if p.endswith('/source.json') and
                                   '/'+monthly['month'] in p) if monthly['month'] != '2026-08' else str(AUGUST/'august_source.json')
                provenance = {'instrument': 'BTC', 'data_stream_id': STREAM,
                    'source_artifact_sha256': files[source_path], 'source_artifact': source_path,
                    'parent_artifact_sha256': files[str(PARENTS/'parent_ledgers.json')],
                    'reconstruction_verified': True, 'candidate_atr_contract': monthly['atr_contract'],
                    'parent_atr_contract': parents['4H_N3']['manifest']['atr_contract'],
                    'limitations': monthly['missing_input_limits'], 'native_replay_blockers': monthly['replay_blockers']}
                for candidate in monthly['candidates']:
                    t = clock(candidate['decision_time'])
                    prefix = minute.loc[t-pd.Timedelta('32d'):t-pd.Timedelta('1min')]
                    case = prepare_evidence(candidate, prefix, parents, provenance)
                    verify_cut(case, parents)
                    cases.append(case)
            cases.sort(key=lambda c: clock(c['decision_time']))
            legacy = prepare_cases(raw, minute)
            if legacy != sorted(expected, key=lambda c: clock(c['decision_time'])):
                raise ValueError('legacy candidate subtype/stop/H5 projection parity differs')
            if [c['candidate_id'] for c in cases] != [r['candidate_id'] for r in raw]:
                raise ValueError('source case population differs')
            decisions = [classify(c) for c in cases]
            coverage = {'case_count': len(cases), 'subtypes': dict(Counter(c['subtype'] for c in cases)),
                'scenarios': dict(Counter(d['scenario'] for d in decisions)),
                'reasons': dict(Counter(d['reason'] for d in decisions)),
                'source_statuses': dict(Counter(c['source_status'] for c in cases)),
                'risk_statuses': dict(Counter(c['risk_status'] for c in cases)),
                'months': month_checks, 'parent_qualification': parent_checks,
                'independent_parent_cuts_verified': len(cases), 'legacy_projection_verified': len(legacy),
                'execution_authorized': False, 'economic_outcomes_computed': False,
                'source_limits': sorted({x for s in sources for x in s['replay_blockers']})}
            verify_files(files)
            for name, value in [('cases.json', cases), ('decisions.json', decisions), ('coverage.json', coverage)]:
                budget.save(out/name, value)
            artifacts = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
            receipt = {'stage': 'lc_context_source', 'completed': True, 'files': files,
                'artifacts': artifacts, 'case_count': len(cases), 'policy_seal': seal(protocol()),
                'elapsed_seconds': time.monotonic()-began, 'artifact_bytes_before_receipt': budget.bytes,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'economic_outcomes_computed': False, 'execution_authorized': False}
            budget.save(out/'receipt.json', receipt)
            return {k: v for k, v in receipt.items() if k not in ('files', 'artifacts')}
    except BaseException as exc:
        failure = {'stage': 'lc_context_source', 'completed': False, 'error': type(exc).__name__+': '+str(exc),
                   'execution_authorized': False, 'elapsed_seconds': time.monotonic()-began}
        _Budget(60, 65536).save(out/'failure.json', failure)
        raise
