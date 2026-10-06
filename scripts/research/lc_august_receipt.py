"""Versioned recovery for August source capture, preserving the failed v1 run.

Old selected clocks used spaces, the canonical census uses ISO T. Compare actual
aware clocks, not their formatting. Persist a bound, explicitly unverified source
before selected-example parity; never expose it as verified until checks pass.
"""
import json
from pathlib import Path

import pandas as pd

from scripts.research import lc_august_source as source_api
from scripts.research.lc_campaign_source import (
    ARCHIVE, ROOT, _hourly_from_minutes, _merge_hashes, _nested_manifest_files,
    _production_context, _sha, _verify_hashes,
)
from scripts.research.lc_judgment_runner import _digest, _load, _save_equal, _verify_digest
from scripts.research.replay_clock import json_safe


OUTPUT = source_api.STUDY/'run_v2'


def verify_selected(source, month):
    by_id = {c['candidate_id']: c for c in source['candidates']}
    for old in month['hourly_selected']:
        new = by_id.get(old['candidate_id'])
        if new is None:
            raise ValueError('selected parity missing candidate: '+old['candidate_id'])
        for key, value in json_safe(new).items():
            expected = old.get(key)
            if key in ('decision_time', 'setup_open'):
                a, b = pd.Timestamp(value), pd.Timestamp(expected)
                equal = a.tzinfo is not None and b.tzinfo is not None and a == b
            else:
                equal = value == expected
            if not equal:
                raise ValueError('selected parity differs: '+old['candidate_id']+' field='+key)
    return len(month['hourly_selected'])


def _save_receipt(path, files, **metadata):
    receipt = dict(version='lc_august_source_receipt_v2', files=files, **metadata)
    receipt['sha256'] = _digest(receipt)
    _save_equal(path, receipt)
    return receipt


def prepare_august(output=OUTPUT):
    output = Path(output).resolve()
    prior_path = source_api.PRIOR
    rule = _verify_digest(source_api.STUDY/'rule_lock.json')
    exposure = _verify_digest(source_api.STUDY/'exposure.json')
    files = dict(rule['files'])
    _merge_hashes(files, exposure['evidence'], 'exposure')
    for path in [source_api.STUDY/'rule_lock.json', source_api.STUDY/'exposure.json',
                 prior_path, Path(source_api.__file__), Path(__file__),
                 ROOT/'tests/research/test_lc_august_source.py',
                 ROOT/'tests/research/test_lc_august_receipt.py']:
        _merge_hashes(files, {str(path): _sha(path)}, 'adapter input')
    prior = json.loads(prior_path.read_bytes()); month = source_api._prior_month(prior)
    _merge_hashes(files, prior['code_hashes'], 'prior August code')
    _merge_hashes(files, {v: prior['source_hashes'][k] for k,v in prior['source_paths'].items()},
                  'prior parent source')
    _nested_manifest_files(month['replay_source_manifest'], files, 'prior replay')
    _verify_hashes(files)
    if (output/'raw_capture_receipt.json').exists():
        captured = _verify_digest(output/'raw_capture_receipt.json')
        _verify_hashes(captured['files'])
        raw_path = output/'unverified_source.json'
        if str(raw_path) not in captured['files']:
            raise ValueError('unbound unverified source capture')
        _merge_hashes(files, captured['files'], 'saved raw capture')
        source = _load(raw_path)
    else:
        services = _production_context()
        for label in ('source','code','config'):
            _merge_hashes(files, services[label+'_hashes'], label)
        if (services['stream'] != prior['data_stream_id'] or
                services['runtime']['talib'] != month['atr_contract']['version']):
            raise ValueError('prior August stream or ATR runtime differs')
        seed, end = source_api.SEED, source_api.END
        minute = services['load_minutes'](seed, end)
        services['validate_bars'](minute, '1min')
        hourly = _hourly_from_minutes(minute, seed, end, max_input_hours=services['max_input_hours'])
        hourly_hash = services['digest']([
            {'timestamp':str(t), **r} for t,r in zip(hourly.index,hourly.to_dict('records'))])
        if hourly_hash != month['hourly_input_hash']:
            raise ValueError('prior August hourly hash differs before launch')
        _verify_hashes(files)
        _save_receipt(output/'source_launch.json', dict(files), hourly_input_hash=hourly_hash,
                      pristine_holdout=False, market_calls=0, verified_source=False)
        replay, errors, warnings = services['run_replay'](hourly, '2026-08')
        _nested_manifest_files(replay['source_manifest'], files, 'current replay')
        provenance = dict(source_path=str(ARCHIVE.resolve()), runtime=services['runtime'],
            source_manifest=dict(files={**services['source_hashes'],str(prior_path):_sha(prior_path)},
                                 replay=replay['source_manifest']),
            code_manifest=dict(files={**services['code_hashes'],
                str(Path(source_api.__file__).resolve()):_sha(source_api.__file__),
                str(Path(__file__).resolve()):_sha(__file__)}),
            config_manifest=dict(files=services['config_hashes'],effective_signal_config=
                replay['source_manifest']['signals']['effective_config']))
        source = json_safe(source_api.assemble_august(prior,replay,hourly_hash,provenance))
        source.update(replay_error_evidence=errors,suppressed_warning_count=warnings)
        _verify_hashes(files)
        raw_path = _save_equal(output/'unverified_source.json',source)
        files[str(raw_path)] = _sha(raw_path)
        files[str(output/'source_launch.json')] = _sha(output/'source_launch.json')
        _save_receipt(output/'raw_capture_receipt.json',dict(files),verified_source=False,
                      pristine_holdout=False,market_calls=0)
    parity = verify_selected(source,month)
    _verify_hashes(files)
    path = _save_equal(output/'august_source.json',source)
    files[str(path)] = _sha(path)
    files[str(output/'raw_capture_receipt.json')] = _sha(output/'raw_capture_receipt.json')
    receipt = _save_receipt(output/'source_receipt.json',files,verified_source=True,
        candidate_count=source['candidate_count'],prior_selected_parity=parity,
        pristine_holdout=False,market_calls=0)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}),flush=True)
    return source


if __name__ == '__main__':
    prepare_august()
