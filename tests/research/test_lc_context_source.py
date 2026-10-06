import importlib
import json

import pandas as pd
import pytest

from scripts.research.lc_context_contract import seal
from scripts.research.replay_clock import digest
from tests.research.lc_context_fixtures import T, clone, source, minutes


def api():
    return importlib.import_module('scripts.research.lc_context_source')


def monthly():
    raw = source()[0]
    raw.update(candidate_id='hourly-lc:'+T.isoformat(), track='hourly')
    return {'month': '2024-01', 'start': '2024-01-01T00:00Z',
            'end_exclusive': '2024-02-01T00:00Z', 'seed': '2023-12-02T00:00Z',
            'data_stream_id': 'fixture', 'candidate_count': 1, 'candidates': [raw]}


def test_exact_population_and_calendar_no_outcome_selection():
    value = monthly(); cid = value['candidates'][0]['candidate_id']
    rows = api().reconcile_sources([value], [cid], stream='fixture', months=['2024-01'])
    assert [r['candidate_id'] for r in rows] == [cid]
    for sources, ids in [([], [cid]), ([value, value], [cid]), ([value], []), ([value], [cid, cid])]:
        with pytest.raises(ValueError):
            api().reconcile_sources(sources, ids, stream='fixture', months=['2024-01'])


@pytest.mark.parametrize('field,value', [('data_stream_id', 'foreign'), ('candidate_count', 0),
    ('end_exclusive', '2024-01-31T00:00Z'), ('seed', '2023-12-01T00:00Z')])
def test_inconsistent_source_identity_or_endpoint_aborts(field, value):
    month = monthly(); cid = month['candidates'][0]['candidate_id']; month[field] = value
    with pytest.raises(ValueError):
        api().reconcile_sources([month], [cid], stream='fixture', months=['2024-01'])


def test_hash_verification_and_immutable_output(tmp_path):
    path = tmp_path/'input.json'
    path.write_text('{"source":true}')
    expected = api().sha(path)
    assert api().checked_read(path, expected) == {'source': True}
    path.write_text('{"source":false}')
    with pytest.raises(ValueError, match='changed'):
        api().checked_read(path, expected)
    with pytest.raises(FileExistsError):
        api().new_output(tmp_path, root=tmp_path.parent)
    with pytest.raises(ValueError, match='subdirectory'):
        api().new_output(tmp_path.parent, root=tmp_path.parent)


def test_old_receipt_digest_checked_not_only_json_shape(tmp_path):
    path = tmp_path/'receipt.json'
    body = {'files': {}, 'verified_source': True}
    path.write_text(json.dumps(dict(body, sha256=seal(body))))
    assert api().read_receipt(path)['verified_source'] is True
    path.write_text(json.dumps(dict(body, sha256='wrong')))
    with pytest.raises(ValueError, match='digest'):
        api().read_receipt(path)


def test_source_lock_whitelist_never_includes_known_outcome_results():
    mapping = {'/old/source/2024-01/source.json': 'source', '/old/result.json': 'outcome',
               '/old/assessment.json': 'assessor', '/old/cases.json': 'projection'}
    assert api().monthly_paths(mapping) == {'/old/source/2024-01/source.json': 'source'}


def hourly_source():
    idx = pd.date_range('2023-12-02T00:00Z', '2024-02-01T00:00Z', freq='h', inclusive='left')
    hours = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 60.}, index=idx)
    hours.loc[T-pd.Timedelta('1h'), ['high', 'close']] = [103., 102.]
    value = monthly()
    value['candidates'][0]['features']['atr_14'] = 30./14.
    value['hourly_input_hash'] = digest([{'timestamp': str(t), **r} for t, r in zip(hours.index, hours.to_dict('records'))])
    return value, hours


def test_native_ohlcv_and_monthly_atr_reconstruct_not_just_seal():
    value, hours = hourly_source()
    result = api().qualify_month(value, hours)
    assert result['candidate_count'] == 1
    value['candidates'][0]['features']['atr_14'] += 1.
    with pytest.raises(ValueError, match='ATR'):
        api().qualify_month(value, hours)
    value, hours = hourly_source()
    hours.iloc[0, 0] += .1
    with pytest.raises(ValueError, match='hourly input'):
        api().qualify_month(value, hours)


def test_changed_locked_file_blocks_reuse(tmp_path):
    path = tmp_path/'implementation.py'; path.write_text('v1')
    files = {str(path): api().sha(path)}
    api().verify_files(files)
    path.write_text('v2')
    with pytest.raises(ValueError, match='changed'):
        api().verify_files(files)


def test_anchor_source_reconstructed_and_partial_buckets_rejected():
    data = minutes(start='2024-01-01T00:00Z', periods=240)
    bucket = {'open_time': '2024-01-01T00:00Z', 'close_time': '2024-01-01T04:00Z',
              'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 240.,
              'complete': True, 'constituents': 4,
              'source_open_times': [f'2024-01-01T0{h}:00Z' for h in range(4)]}
    ledger = {'anchor_buckets': {'completed': [bucket], 'incomplete': [], 'developing': []}}
    assert api().verify_anchors(ledger, data, '4H') == 1
    bucket['close'] = 101.
    with pytest.raises(ValueError, match='anchor'):
        api().verify_anchors(ledger, data, '4H')
    bucket['close'] = 100.
    with pytest.raises(ValueError, match='anchor'):
        api().verify_anchors(ledger, data.iloc[1:], '4H')
