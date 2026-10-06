"""Full August native LC census; separate from immutable older month allowlists.

Reuses previously saved parents only after exact hourly-input and code parity.
No outcome selection, market roles, network access, orders or live configuration.
"""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from scripts.research.lc_campaign_source import (
    ARCHIVE, ROOT, SOURCE_SHA256, EXPECTED_ARCHETYPES, _hourly_from_minutes,
    _merge_hashes, _nested_manifest_files, _production_context, _sha,
    _validate_candidates, _verify_hashes,
)
from scripts.research.lc_judgment_runner import _digest, _load, _save_equal, _verify_digest
from scripts.research.lc_source_population import collect_native_lc
from scripts.research.parent_context_policy import _parent_config
from scripts.research.replay_clock import json_safe


STUDY = ROOT / 'results/lc_room_validation_2026_09_30'
OUTPUT = STUDY / 'run_v1'
PRIOR = ROOT / 'results/teaching_market_transfer_2026_09_11/2026-08_sources.json'
START = pd.Timestamp('2026-08-01T00:00Z')
END = pd.Timestamp('2026-09-01T00:00Z')
SEED = START - pd.Timedelta('30d')


def _prior_month(prior):
    months = prior.get('months', [])
    if len(months) != 1 or months[0].get('month') != '2026-08':
        raise ValueError('exact August source required')
    month = months[0]
    for key, expected in [('seed', SEED), ('start', START), ('end_exclusive', END)]:
        if pd.Timestamp(month.get(key)) != expected:
            raise ValueError('prior August bounds differ: ' + key)
    if (prior.get('source_sha256') != SOURCE_SHA256 or month.get('seed_days') != 30
            or month.get('hourly_rows') != 1464 or month.get('minute_rows') != 87840):
        raise ValueError('prior archive identity or coverage differs')
    return month


def assemble_august(prior, replay, hourly_input_hash, provenance):
    """Pure assembly: never use the prior study's two-case selected roster."""
    month = _prior_month(prior)
    if hourly_input_hash != month.get('hourly_input_hash'):
        raise ValueError('prior August hourly input hash differs')
    order = replay.get('source_manifest', {}).get('signals', {}).get('archetype_order', [])
    if len(order) != 17 or set(order) != EXPECTED_ARCHETYPES:
        raise ValueError('all 17 native archetypes required')
    rows = replay.get('rows', [])
    clocks = pd.DatetimeIndex([r['decision_time'] for r in rows])
    if not clocks.equals(pd.date_range(SEED + pd.Timedelta('1h'), END, freq='h')):
        raise ValueError('complete consecutive hourly replay required')
    parents = {}
    for ledger in month.get('ledgers', []):
        cfg = _parent_config(ledger)
        name = cfg['anchor_timeframe'] + '_N' + str(cfg['pivot_n'])
        if name in parents:
            raise ValueError('duplicate parent configuration')
        parents[name] = deepcopy(ledger)
    if set(parents) != {'4H_N3', '1D_N3'}:
        raise ValueError('both fixed parent configurations required')
    candidates = collect_native_lc(rows, START.isoformat(), END.isoformat())
    _validate_candidates(candidates, START.to_pydatetime(), END.to_pydatetime())
    source = {k: deepcopy(month[k]) for k in (
        'month', 'seed', 'start', 'end_exclusive', 'seed_days', 'hourly_rows',
        'minute_rows', 'hourly_input_hash', 'atr_contract')}
    source.update(deepcopy(provenance))
    source.update(schema='lc-august-native-source-v1', certified=False, pristine_holdout=False,
        scope='All native long LC before winner/H2 filtering; no outcomes or trading roles.',
        source_sha256=SOURCE_SHA256, data_stream_id=prior['data_stream_id'],
        candidates=candidates, candidate_count=len(candidates), parent_ledgers=parents,
        replay_contract_id=replay['contract_id'], replay_blockers=deepcopy(replay['blockers']),
        missing_input_limits=[
            'Historical receipt times unavailable; bar-close availability assumed.',
            'Macro and derivatives observations absent; native defaults/fallbacks remain in replay_blockers.',
            'Independent 30-day cold start; saved parent ledgers reused after exact input/code parity.',
            'No live book/threshold/allocation or profitability certification.',
            'Ignored private preparer remains a local reproduction dependency.',
        ])
    return source


def prepare_august(output=OUTPUT):
    """Verify, lock, run the unchanged guarded native replay once, publish census."""
    output = Path(output).resolve()
    rule = _verify_digest(STUDY / 'rule_lock.json')
    exposure = _verify_digest(STUDY / 'exposure.json')
    guarded = dict(rule['files'])
    _merge_hashes(guarded, exposure['evidence'], 'exposure')
    for path in [STUDY/'rule_lock.json', STUDY/'exposure.json', PRIOR, Path(__file__),
                 ROOT/'tests/research/test_lc_august_source.py']:
        _merge_hashes(guarded, {str(path): _sha(path)}, 'adapter input')
    _verify_hashes(guarded)
    prior = json.loads(PRIOR.read_bytes())
    month = _prior_month(prior)
    _merge_hashes(guarded, prior['code_hashes'], 'prior August code')
    _merge_hashes(guarded, dict(zip(prior['source_paths'].values(),
                                  [prior['source_hashes'][k] for k in prior['source_paths']])),
                  'prior parent sources')
    _nested_manifest_files(month['replay_source_manifest'], guarded, 'prior August replay')
    _verify_hashes(guarded)
    if (output/'august_source.json').exists():
        lock = _verify_digest(output/'source_receipt.json')
        _verify_hashes(lock['files'])
        consumed = [output/'august_source.json', output/'source_launch.json', Path(__file__)]
        if any(str(p.resolve()) not in lock['files'] for p in consumed):
            raise ValueError('unbound consumed source input')
        return _load(output/'august_source.json')
    services = _production_context()
    for label in ('source', 'code', 'config'):
        _merge_hashes(guarded, services[label+'_hashes'], label)
    if services['stream'] != prior['data_stream_id']:
        raise ValueError('prior August stream differs')
    if services['runtime']['talib'] != month['atr_contract']['version']:
        raise ValueError('prior August ATR runtime differs')
    minute = services['load_minutes'](SEED, END)
    services['validate_bars'](minute, '1min')
    hourly = _hourly_from_minutes(minute, SEED, END, max_input_hours=services['max_input_hours'])
    hourly_hash = services['digest']([
        {'timestamp': str(t), **r} for t, r in zip(hourly.index, hourly.to_dict('records'))])
    if hourly_hash != month['hourly_input_hash']:
        raise ValueError('prior August hourly input hash differs before launch')
    _verify_hashes(guarded)
    launch = dict(version='lc_august_source_launch_v1', files=guarded,
                  hourly_input_hash=hourly_hash, pristine_holdout=False, market_calls=0)
    launch['sha256'] = _digest(launch)
    _save_equal(output/'source_launch.json', launch)
    replay, errors, warnings = services['run_replay'](hourly, '2026-08')
    _nested_manifest_files(replay['source_manifest'], guarded, 'current August replay')
    provenance = dict(source_path=str(ARCHIVE.resolve()), runtime=services['runtime'],
        source_manifest=dict(files={**services['source_hashes'], str(PRIOR): _sha(PRIOR)},
                             replay=replay['source_manifest']),
        code_manifest=dict(files={**services['code_hashes'], str(Path(__file__).resolve()): _sha(__file__)}),
        config_manifest=dict(files=services['config_hashes'], effective_signal_config=
                             replay['source_manifest']['signals']['effective_config']))
    source = assemble_august(prior, replay, hourly_hash, provenance)
    source.update(replay_error_evidence=errors, suppressed_warning_count=warnings)
    by_id = {c['candidate_id']: c for c in source['candidates']}
    for old in month['hourly_selected']:
        new = by_id.get(old['candidate_id'])
        if new is None or any(new[k] != old[k] for k in new):
            raise ValueError('prior selected candidate does not reproduce exactly')
    _verify_hashes(guarded)
    path = _save_equal(output/'august_source.json', json_safe(source))
    receipt_files = {**guarded, str(path): _sha(path),
                     str(output/'source_launch.json'): _sha(output/'source_launch.json')}
    receipt = dict(version='lc_august_source_receipt_v1', files=receipt_files,
                   candidate_count=len(by_id), prior_selected_parity=len(month['hourly_selected']),
                   pristine_holdout=False, market_calls=0)
    receipt['sha256'] = _digest(receipt)
    _save_equal(output/'source_receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}), flush=True)
    return source


if __name__ == '__main__':
    prepare_august()
