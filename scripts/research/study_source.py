"""Hash-bound, single-process R1/R3 source preparation. No outcome scoring."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import resource
import signal as os_signal
import socket
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.research.engine_signal_replay import EXPECTED_ARCHETYPES
from scripts.research.r3_census import build_r3_census, merge_censuses, validated_minutes
from scripts.research.replay_clock import digest, json_safe
from scripts.research.study_cases import project_case, validate_case
from scripts.research.study_contract import finite_number, protocol, utc_minute

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT/'data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet'
REFERENCE = ROOT/'results/research_validation_2026_09_10/h3_parent_permission/june_minute_frozen_reference.json'
PRIVATE_PREPARER = ROOT/'results/agent_layered_entry_2026_09_11/prepare_sources.py'
SPEC = ROOT/'docs/superpowers/specs/2026-09-30-archetype-repair-discovery-design.md'
OUTPUT_ROOT = ROOT/'results/archetype_study_2026_10_01'
REFERENCE_SHA = '9f9587253250b56d547d2a5af17a2d02bc1a8035e86d5f962eae0b7e0f01ff20'
PRIVATE_SHA = 'ba0fb9d1c9a43769b9113cdb743a98d04d54db88a356d2bcbf848fcfda35e5c7'
STREAM = 'btc_1m_2021_2026_saved_5b8a4533f70b8ccd'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def runtime_manifest():
    import talib
    library = talib.__ta_version__
    if isinstance(library, bytes):
        library = library.decode()
    return {'python': platform.python_version(),
            'talib': 'python-talib=' + talib.__version__ + ';library=' + library,
            'packages': {p: metadata.version(p) for p in ('numpy', 'pandas', 'pyarrow', 'pytest')}}


def preflight():
    policy, files, blockers = protocol(), {}, []
    result = {'schema': 'study-preflight-v1', 'protocol': policy, 'files': files,
              'blockers': blockers, 'cohort_blockers': {'R1': [], 'R3': []},
              'parent_paths': {}, 'parent_hashes': {}, 'data_stream_id': STREAM,
              'execution_authorized': False}
    required = {ARCHIVE: policy['source_sha256'], REFERENCE: REFERENCE_SHA,
                PRIVATE_PREPARER: PRIVATE_SHA, SPEC: policy['canonical_spec_sha256']}
    for path, expected in required.items():
        if not path.is_file():
            blockers.append('missing_dependency:' + str(path))
        elif sha(path) != expected:
            blockers.append('source_hash_mismatch:' + str(path))
        else:
            files[str(path)] = expected
    try:
        runtime = result['runtime'] = runtime_manifest()
        if str(REFERENCE) in files:
            reference = json.loads(REFERENCE.read_text())
            manifest = reference['ledgers'][0]['manifest']
            result.update(parent_paths=manifest['source_paths'], parent_hashes=manifest['source_hashes'],
                          atr_contract=reference['atr_contract'])
            if manifest['data_stream_id'] != STREAM:
                blockers.append('reference_stream_mismatch')
            if runtime['talib'] != reference['atr_contract']['version']:
                blockers.append('parent_atr_runtime_mismatch')
            for key, value in result['parent_paths'].items():
                path = Path(value)
                if not path.is_file() or sha(path) != result['parent_hashes'].get(key):
                    blockers.append('missing_or_changed_parent_helper:' + key)
                else:
                    files[str(path)] = result['parent_hashes'][key]
        config_path = ROOT/'configs/champion_paper.json'
        config = json.loads(config_path.read_text())
        model = ROOT/(config.get('regime_classifier', {}).get('model_path') or 'models/logistic_regime_v4_no_funding_stratified.pkl')
        calibrator = ROOT/'models/confidence_calibrator_v1.pkl'
        for label, path in [('regime_model', model), ('confidence_calibrator', calibrator)]:
            if not path.is_file():
                result['cohort_blockers']['R1'].append('missing_' + label)
            else:
                files[str(path)] = sha(path)
        directory = ROOT/config.get('archetype_config_dir', 'configs/archetypes')
        code = {Path(__file__), config_path, *directory.glob('*.yaml'), *directory.glob('*.yml'),
                *ROOT.joinpath('engine').rglob('*.py'), *ROOT.joinpath('bin/live').rglob('*.py')}
        code.update(ROOT/'scripts/research'/name for name in (
            'study_contract.py', 'r3_census.py', 'study_cases.py', 'study_hourly.py',
            'run_archetype_study.py', 'minute_sweep_validation.py',
            'causal_parent_ledger.py', 'engine_signal_replay.py', 'live_feature_replay.py',
            'replay_clock.py', 'virtual_book_replay.py'))
        files.update({str(p): sha(p) for p in sorted(code)})
    except (ImportError, OSError, ValueError, KeyError) as exc:
        blockers.append('source_preflight_error:' + type(exc).__name__ + ':' + str(exc))
    result['source_ready'] = not blockers
    result['availability_basis'] = 'historical_bar_close_assumption'
    return result


def verify_files(manifest):
    for name, expected in manifest['files'].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('bound source changed: ' + name)


def load_minutes(seed, end):
    return pd.read_parquet(ARCHIVE, columns=['open', 'high', 'low', 'close', 'vol'],
                           filters=[('ts', '>=', pd.Timestamp(seed)), ('ts', '<', pd.Timestamp(end))]).rename(columns={'vol': 'volume'})


def hourly_from_minutes(minute, seed, end):
    seed, end = utc_minute(seed), utc_minute(end)
    if seed != seed.floor('h') or end != end.floor('h') or end <= seed:
        raise ValueError('complete UTC hour boundaries required')
    data = validated_minutes(minute)
    expected = pd.date_range(seed, end, freq='min', inclusive='left')
    if not data.index.equals(expected):
        raise ValueError('minute coverage incomplete, gapped or outside fixed window')
    hourly = data.resample('h', closed='left', label='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    if len(hourly) != int((end - seed) / pd.Timedelta('1h')):
        raise ValueError('partial hourly coverage')
    return hourly


def feature_provenance(features, hours_seen):
    """Only explicitly inspected OHLC-derived fields receive observed status."""
    output = {key: {'status': 'unqualified_native_output'} for key in features}
    for key in ('open', 'high', 'low', 'close', 'volume', 'timestamp', 'wick_lower_ratio'):
        if key in features:
            output[key] = {'status': 'observed', 'basis': 'completed_hour_ohlcv'}
    for key, warmup in [('atr_14', 15), ('volume_zscore', 21)]:
        if key in features:
            try:
                finite_number(features[key], positive=key == 'atr_14')
                if hours_seen >= warmup:
                    output[key] = {'status': 'observed', 'basis': 'native_completed_hour_formula'}
            except ValueError:
                pass
    try:
        ema = finite_number(features.get('ema_50'), True)
        close = finite_number(features.get('close'), True)
        direction = finite_number(features.get('price_above_ema_50'))
        if hours_seen >= 50 and direction == float(close > ema):
            output['price_above_ema_50'] = {'status': 'observed', 'basis': 'native_ema50_confirmed_direction'}
    except ValueError:
        pass
    return output


def iter_hourly_source(hourly, manifest):
    """Advance one native feature engine and all17/independent arms continuously."""
    from results.agent_layered_entry_2026_09_11.prepare_sources import LogEvidence
    from scripts.research.engine_signal_replay import SignalEngine
    from scripts.research.live_feature_replay import LiveFeatureProcessor
    from scripts.research.study_hourly import HourlyStudyObserver
    from scripts.research.virtual_book_replay import side_effect_guard
    logs = LogEvidence()
    captured = io.StringIO()
    with redirect_stdout(captured), patch.object(socket, 'has_ipv6', False), side_effect_guard(logs):
        features = LiveFeatureProcessor('1h')
        observer = HourlyStudyObserver(SignalEngine(), instrument='BTC', data_stream_id=manifest['data_stream_id'])
    for count, (opened, bar) in enumerate(hourly.iterrows(), 1):
        candle = dict(bar.to_dict(), timestamp=opened, close_time=opened + pd.Timedelta('1h'))
        with redirect_stdout(captured), patch.object(socket, 'has_ipv6', False), side_effect_guard(logs):
            output = features.update(candle, {})
            provenance = feature_provenance(output['features'], count)
            diagnostic = observer.update(output['features'], candle['close_time'], provenance)
        row = {'source_hour': opened.isoformat(), 'decision_time': candle['close_time'].isoformat(),
               'features': output['features'], 'provenance': provenance, 'context': output['context'],
               'diagnostic': diagnostic, 'feature_blockers': sorted(features.blockers),
               'source_errors': list(logs), 'suppressed_warning_count': logs.warning_count}
        if count == 1:
            row['source_manifests'] = {'features': features.manifest, 'signals': observer.signal_engine.manifest}
        if count % 120 == 0 or count == len(hourly):
            row['state_checkpoint'] = {'hours': count, 'method': 'rebuild_identical_full_prefix',
                                       'state_digest': digest({'features': features.snapshot(), 'observer': observer.snapshot()})}
        logs.clear()
        captured.seek(0)
        captured.truncate()
        yield row


def make_parents(hourly, manifest):
    import talib
    from scripts.research.causal_parent_ledger import build_parent_ledger
    bars = hourly.copy()
    bars['atr_14'] = talib.ATR(bars.high.to_numpy(), bars.low.to_numpy(), bars.close.to_numpy(), timeperiod=14)
    bars['atr_available_at'] = bars.index + pd.Timedelta('1h')
    if not bars.atr_14.iloc[:14].isna().all() or not bars.atr_14.iloc[14:].notna().all():
        raise ValueError('unexpected causal ATR warmup')
    contract = dict(manifest['atr_contract'], source='archetype-study-v1 continuous same-minute-stream seed ' + manifest['protocol']['seed'])
    return {anchor + '_N3': build_parent_ledger(bars, instrument='BTC', data_stream_id=manifest['data_stream_id'],
                                                anchor_timeframe=anchor, pivot_n=3, atr_contract=contract,
                                                source_paths=manifest['parent_paths'], expected_hashes=manifest['parent_hashes'])
            for anchor in ('4H', '1D')}


class _Budget:
    def __init__(self, seconds, byte_limit):
        self.began = time.monotonic()
        self.seconds, self.byte_limit, self.bytes = seconds, byte_limit, 0

    def check(self, increment=0):
        if time.monotonic() - self.began > self.seconds:
            raise RuntimeError('source wall-time budget exceeded')
        if self.bytes + increment + 1024 > self.byte_limit:
            raise RuntimeError('source byte budget exceeded')
        self.bytes += increment

    def line(self, handle, value):
        text = json.dumps(json_safe(value), sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n'
        self.check(len(text.encode('utf-8')))
        handle.write(text)
        handle.flush()

    def save(self, path, value):
        with path.open('x') as handle:
            self.line(handle, value)


@contextmanager
def _deadline(seconds):
    def expired(signum, frame):
        raise RuntimeError('source wall-time budget exceeded')
    previous = os_signal.signal(os_signal.SIGALRM, expired)
    old_timer = os_signal.setitimer(os_signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        os_signal.setitimer(os_signal.ITIMER_REAL, *old_timer)
        os_signal.signal(os_signal.SIGALRM, previous)


def build_pilot(output_dir, *, max_seconds=1800, max_bytes=2147483648):
    seconds, byte_limit = finite_number(max_seconds, True), finite_number(max_bytes, True)
    if seconds > 1800 or byte_limit > 2147483648 or byte_limit < 1100:
        raise ValueError('pilot budget exceeds reviewed limits or cannot hold failure receipt')
    out = Path(output_dir).resolve()
    if out == OUTPUT_ROOT.resolve() or not out.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('output destination must be a new study subdirectory')
    if out.exists():
        raise FileExistsError('source output already exists; never overwrite or auto-relaunch')
    manifest = preflight()
    if not manifest['source_ready']:
        raise ValueError('source preflight blocked: ' + '; '.join(manifest['blockers']))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    budget = _Budget(seconds, int(byte_limit))
    count, r1_count, first_case = 0, 0, None
    cohort_blockers = deepcopy(manifest['cohort_blockers'])
    try:
        with _deadline(seconds):
            budget.save(out/'manifest.json', manifest)
            policy = manifest['protocol']
            seed, start, end = map(utc_minute, (policy['seed'], policy['start'], policy['pilot_end']))
            minute = load_minutes(seed, end)
            hourly = hourly_from_minutes(minute, seed, end)
            if len(hourly) > 2048:
                raise ValueError('pilot exceeds guarded parent constructor cap')
            with (out/'hourly_raw.jsonl').open('x') as raw:
                for row in iter_hourly_source(hourly, manifest):
                    budget.line(raw, row)  # durable before schema validation/projection
                    count += 1
                    diagnostic = row['diagnostic']
                    if set(diagnostic['native']['archetypes']) != EXPECTED_ARCHETYPES:
                        raise ValueError('source did not preserve exact all17 diagnostics')
                    if start <= utc_minute(row['decision_time']) < end:
                        if diagnostic['opportunity'] is not None:
                            r1_count += 1
                        cohort_blockers['R1'].extend(diagnostic['blockers'])
                        for problem in diagnostic['native'].get('blockers', []):
                            if any(k in problem for k in ('model', 'calibrator', 'structural_error')):
                                cohort_blockers['R1'].append(problem)
                    if count % 120 == 0:
                        print('SOURCE_PROGRESS', count, '/', len(hourly), 'hours', round(time.monotonic() - budget.began, 1), 'seconds', flush=True)
            if count != len(hourly):
                raise ValueError('source omitted hourly rows')
            parents = make_parents(hourly, manifest)
            budget.save(out/'parent_ledgers.json', parents)
            census = build_r3_census(minute, parents['4H_N3'], instrument='BTC',
                                      data_stream_id=manifest['data_stream_id'], emit_from=start, end_exclusive=end)
            budget.save(out/'r3_census.json', census)
            cohort_blockers['R3'].extend(b['reason'] for b in census['blockers'])
            january_ops = [o for o in census['opportunities'] if start <= utc_minute(o['origin_time']) < end]
            with (out/'r3_cases.jsonl').open('x') as cases:
                for op in january_ops:
                    events = [e for e in census['events'] if e['opportunity_id'] == op['id']]
                    at = events[-1]['available_at'] if events else op['origin_time']
                    case = project_case(census, op['id'], at)
                    budget.line(cases, case)
                    if validate_case(case):
                        raise ValueError('invalid as-of case')
                    if case['source_citations']['status'] != 'resolved':
                        cohort_blockers['R3'].append('incomplete_source_citations')
                    if first_case is None:
                        first_case = case
            restart = {'performed': False}
            cutoff = start + pd.Timedelta('15d')
            if seed < cutoff < end:
                prefix = build_r3_census(minute.loc[minute.index < cutoff], parents['4H_N3'], instrument='BTC',
                                          data_stream_id=manifest['data_stream_id'], emit_from=start, end_exclusive=cutoff)
                suffix = build_r3_census(minute.loc[minute.index >= cutoff], parents['4H_N3'], instrument='BTC',
                                          data_stream_id=manifest['data_stream_id'], emit_from=start, end_exclusive=end,
                                          checkpoint=prefix['checkpoint'])
                joined = merge_censuses(prefix, suffix)
                match = all(digest(joined[k]) == digest(census[k]) for k in ('opportunities', 'events', 'attempts', 'checkpoint'))
                if not match:
                    raise ValueError('R3 real-source restart mismatch')
                restart = {'performed': True, 'cutoff': cutoff.isoformat(), 'matched': match}
            verify_files(manifest)
            artifacts = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
            receipt = {'schema': 'study-source-receipt-v1', 'stage': 'source_only_pilot', 'completed': True,
                       'execution_authorized': False, 'hourly_rows': count, 'archetype_count': 17,
                       'minute_rows': len(minute), 'r1_raw_opportunities': r1_count,
                       'r3_raw_opportunities': len(january_ops), 'r3_source_restart': restart,
                       'first_r3_case_id': first_case['opportunity']['id'] if first_case else None,
                       'cohort_blockers': {k: sorted(set(v)) for k, v in cohort_blockers.items()},
                       'elapsed_seconds': time.monotonic() - budget.began, 'artifact_bytes_before_receipt': budget.bytes,
                       'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
                       'artifacts': artifacts, 'economic_outcomes_computed': False,
                       'next_gate': 'continuous-source/resource/integrated quant review; not automatic economics'}
            budget.save(out/'receipt.json', receipt)
            return receipt
    except BaseException as exc:
        failure = {'completed': False, 'stage': 'source_only_pilot', 'hourly_rows_preserved': count,
                   'error_type': type(exc).__name__, 'error': str(exc)[:500],
                   'elapsed_seconds': time.monotonic() - budget.began, 'execution_authorized': False}
        with (out/'failure.json').open('x') as handle:
            handle.write(json.dumps(failure, sort_keys=True) + '\n')
        raise
