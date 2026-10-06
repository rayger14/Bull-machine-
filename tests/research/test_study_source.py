from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.research import study_source as source
from tests.research.study_fixtures import minutes, parent_ledger


def test_preflight_binds_real_archive_helpers_and_reports_r1_asset_limits():
    result = source.preflight()
    assert result['source_ready'] is True
    assert result['protocol']['execution_authorized'] is False
    assert len(result['parent_hashes']) == 2
    assert result['files'][str(source.ARCHIVE)] == result['protocol']['source_sha256']
    assert result['cohort_blockers']['R1']  # local missing assets are not assumed observed
    assert result['cohort_blockers']['R3'] == []
    assert str(source.ROOT/'scripts/research/run_archetype_study.py') in result['files']
    assert str(source.ROOT/'scripts/research/minute_sweep_validation.py') in result['files']


@pytest.mark.parametrize('target', ['ARCHIVE', 'REFERENCE', 'PRIVATE_PREPARER'])
def test_missing_required_local_dependency_blocks_source(monkeypatch, tmp_path, target):
    monkeypatch.setattr(source, target, tmp_path/'absent')
    result = source.preflight()
    assert not result['source_ready']
    assert result['blockers']


def test_wrong_archive_hash_and_runtime_fail_closed(monkeypatch, tmp_path):
    wrong = tmp_path/'wrong.parquet'
    wrong.write_bytes(b'not the frozen archive')
    monkeypatch.setattr(source, 'ARCHIVE', wrong)
    assert any('hash' in b for b in source.preflight()['blockers'])
    monkeypatch.setattr(source, 'runtime_manifest', lambda: {'talib': 'wrong'})
    assert any('runtime' in b for b in source.preflight()['blockers'])


def test_complete_utc_aggregates_and_gap_or_partial_bucket_rejection():
    data = minutes()
    hourly = source.hourly_from_minutes(data, data.index[0], data.index[-1] + pd.Timedelta('1min'))
    assert len(hourly) == 2
    assert hourly.iloc[0]['volume'] == 60
    assert hourly.iloc[0]['high'] == 102
    with pytest.raises(ValueError, match='coverage'):
        source.hourly_from_minutes(data.drop(data.index[40]), data.index[0], data.index[-1] + pd.Timedelta('1min'))
    with pytest.raises(ValueError, match='hour'):
        source.hourly_from_minutes(data.iloc[1:], data.index[1], data.index[-1] + pd.Timedelta('1min'))


def test_provenance_does_not_certify_warmup_defaults_or_unknown_macro():
    features = {'open': 100., 'ema_50': 100., 'close': 101., 'price_above_ema_50': 1,
                'atr_14': 2., 'funding_Z': 0., 'volume_zscore': 0.}
    early = source.feature_provenance(features, 10)
    late = source.feature_provenance(features, 720)
    assert early['price_above_ema_50']['status'] != 'observed'
    assert early['atr_14']['status'] != 'observed'
    assert late['price_above_ema_50']['status'] == 'observed'
    assert late['atr_14']['status'] == 'observed'
    assert late['funding_Z']['status'] != 'observed'
    assert 'funding_Z' in late


def test_actual_guarded_hourly_producer_preserves_all17_without_network():
    data = minutes(180)
    hourly = source.hourly_from_minutes(data, data.index[0], data.index[-1] + pd.Timedelta('1min'))
    rows = list(source.iter_hourly_source(hourly, {'data_stream_id': 'fixture'}))
    assert len(rows) == 3
    assert all(set(r['diagnostic']['native']['archetypes']) == source.EXPECTED_ARCHETYPES for r in rows)
    assert rows[-1]['state_checkpoint']['method'] == 'rebuild_identical_full_prefix'
    assert rows[-1]['diagnostic']['native']['bar_index'] == 3


def tiny_source(monkeypatch, tmp_path, fail=False):
    data = minutes()
    policy = source.protocol()
    policy.update(seed=data.index[0].isoformat(), start=data.index[0].isoformat(),
                  pilot_end=(data.index[-1] + pd.Timedelta('1min')).isoformat())
    preflight = {'schema': 'study-preflight-v1', 'source_ready': True, 'blockers': [],
                 'protocol': policy, 'files': {}, 'cohort_blockers': {'R1': ['fixture_unqualified'], 'R3': []},
                 'parent_hashes': {}, 'parent_paths': {}, 'data_stream_id': 'fixture', 'runtime': {}}
    monkeypatch.setattr(source, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(source, 'preflight', lambda: deepcopy(preflight))
    monkeypatch.setattr(source, 'load_minutes', lambda seed, end: data)
    monkeypatch.setattr(source, 'make_parents', lambda hourly, manifest: {'4H_N3': parent_ledger(), '1D_N3': parent_ledger()})

    def hourly_rows(hourly, manifest):
        for i, (opened, row) in enumerate(hourly.iterrows()):
            if i == 1 and fail:
                raise RuntimeError('injected source failure')
            yield {'source_hour': opened.isoformat(), 'decision_time': (opened + pd.Timedelta('1h')).isoformat(),
                   'features': row.to_dict(), 'provenance': {}, 'diagnostic': {'native': {
                       'archetypes': {name: {} for name in source.EXPECTED_ARCHETYPES}},
                       'opportunity': None, 'arms': {}, 'blockers': []}, 'feature_blockers': [], 'source_errors': []}
    monkeypatch.setattr(source, 'iter_hourly_source', hourly_rows)
    return tmp_path/'pilot'


def test_pilot_preserves_raw_capture_before_source_failure(monkeypatch, tmp_path):
    out = tiny_source(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match='injected source failure'):
        source.build_pilot(out)
    lines = (out/'hourly_raw.jsonl').read_text().splitlines()
    assert len(lines) == 1
    assert json.loads((out/'failure.json').read_text())['completed'] is False
    assert not (out/'receipt.json').exists()


def test_pilot_writes_scope_receipt_cases_and_rejects_duplicate_launch(monkeypatch, tmp_path):
    out = tiny_source(monkeypatch, tmp_path)
    result = source.build_pilot(out)
    assert result['stage'] == 'source_only_pilot'
    assert result['execution_authorized'] is False
    assert result['hourly_rows'] == 2
    assert result['archetype_count'] == 17
    assert result['peak_rss_bytes'] > 0
    assert result['cohort_blockers']['R1'] == ['fixture_unqualified']
    assert (out/'r3_census.json').is_file()
    assert (out/'r3_cases.jsonl').is_file()
    with pytest.raises(FileExistsError):
        source.build_pilot(out)


def test_budget_and_destination_guards_preserve_failure_receipt(monkeypatch, tmp_path):
    out = tiny_source(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='destination'):
        source.build_pilot(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        source.build_pilot(out, max_seconds=1801)
    with pytest.raises(RuntimeError, match='byte budget'):
        source.build_pilot(out, max_bytes=1200)
    assert (out/'failure.json').exists()
