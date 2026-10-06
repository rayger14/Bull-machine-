from copy import deepcopy
import json

import pytest

from scripts.research import study_campaign as campaign
from scripts.research.study_source import sha
from tests.research.study_fixtures import nested_minutes
from tests.research.test_r3_census import census


def test_arm_specific_source_dispositions_keep_same_raw_denominator():
    data = nested_minutes()
    result = census(data)
    raw, arms = campaign.book_inputs(result, start='2024-01-01T00:00Z', end='2024-01-01T02:00Z')
    assert len(raw) == len(result['opportunities'])
    assert {o['id'] for o in arms['baseline']['opportunities']} == {o['id'] for o in arms['repair']['opportunities']}
    assert arms['baseline']['signals'][0]['stop'] == 100
    assert arms['repair']['signals'][0]['stop'] == 101.7
    assert arms['repair']['opportunities'][0]['source_status'] == 'complete'
    assert arms['repair']['opportunities'][-1]['source_status'] == 'censored'


def test_failed_retest_is_known_zero_only_for_repair_not_comparator():
    data = nested_minutes()
    data.iloc[39, data.columns.get_loc('close')] = 102
    raw, arms = campaign.book_inputs(census(data), start='2024-01-01T00:00Z', end='2024-01-01T02:00Z')
    first = raw[0]['id']
    assert first in {s['opportunity_id'] for s in arms['baseline']['signals']}
    assert first not in {s['opportunity_id'] for s in arms['repair']['signals']}
    assert arms['repair']['opportunities'][0]['source_status'] == 'complete'
    assert arms['repair']['opportunities'][0]['source_available_at'][11:16] == '00:40'


def test_source_unknown_is_not_turned_into_a_nonentry():
    data = nested_minutes()
    raw, arms = campaign.book_inputs(census(data.drop(data.index[38])), start='2024-01-01T00:00Z', end='2024-01-01T02:00Z')
    assert arms['baseline']['opportunities'][0]['source_status'] == 'complete'
    assert arms['repair']['opportunities'][0]['source_status'] == 'unknown'


def test_prefix_verification_detects_changed_history():
    prefix = {'pivots': [], 'versions': [], 'transitions': [{'available_at': '2024-01-01T00:00Z', 'value': 1}]}
    full = deepcopy(prefix)
    full['transitions'].append({'available_at': '2024-01-02T00:00Z', 'value': 2})
    campaign.verify_parent_prefix(full, prefix, '2024-01-01T00:00Z')
    full['transitions'][0]['value'] = 3
    with pytest.raises(ValueError, match='prefix'):
        campaign.verify_parent_prefix(full, prefix, '2024-01-01T00:00Z')


def review_fixture(monkeypatch, tmp_path):
    target = tmp_path/'code.py'
    target.write_text('qualified source fixture\n')
    expected = {str(target): sha(target)}
    monkeypatch.setattr(campaign, 'launch_file_hashes', lambda: expected)
    review = {'decision': 'GO', 'scope': 'R3-only', 'approved_stages': ['census', 'score'],
              'reviewer': 'quant_agent', 'reviewed_files': expected,
              'funding_mode': 'adverse_stress', 'scenarios': [[12, 5], [12, 65], [24, 5], [24, 65]],
              'protocol': campaign.protocol(), 'pilot_receipt_sha256': 'pilot', 'qualification_receipt_sha256': 'qualification',
              'limits': {'census': {'seconds': 1800, 'bytes': 2147483648},
                         'score': {'seconds': 1800, 'bytes': 4294967296}},
              'trial_ledger': {'active': ['R1', 'R3'], 'parked': ['R2'], 'historic_searches_incomplete': True}}
    path = tmp_path/'review.json'
    path.write_text(json.dumps(review))
    return path, review, target


def test_explicit_review_and_current_hashes_required(monkeypatch, tmp_path):
    path, review, target = review_fixture(monkeypatch, tmp_path)
    assert campaign.load_review(path, 'census')['scope'] == 'R3-only'
    target.write_text('changed\n')
    with pytest.raises(ValueError, match='hash'):
        campaign.load_review(path, 'census')


@pytest.mark.parametrize('change', [
    {'decision': 'planned'}, {'scope': 'all17'}, {'funding_mode': 'zero_diagnostic'},
    {'scenarios': [[12, 5]]}, {'reviewed_files': {}}, {'trial_ledger': {}},
])
def test_incomplete_or_changed_review_does_not_unlock(monkeypatch, tmp_path, change):
    path, review, _ = review_fixture(monkeypatch, tmp_path)
    review.update(change)
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        campaign.load_review(path, 'score')


def test_consumed_stage_cannot_be_launched_twice(tmp_path):
    path = tmp_path/'review.json'
    path.write_text('{}')
    campaign.claim_stage(path, 'census', tmp_path/'out')
    with pytest.raises(FileExistsError):
        campaign.claim_stage(path, 'census', tmp_path/'out2')


@pytest.mark.parametrize('fault', [None, 'source_failure', 'stress_baseline_unknown', 'fixed_event_blocker', 'zero_diagnostic_unknown'])
def test_synthetic_source_to_economic_receipt_integration(monkeypatch, tmp_path, fault):
    from tests.research.study_fixtures import minutes, parent_ledger
    data = minutes(300, price=104.)
    data.iloc[:41] = nested_minutes().iloc[:41].to_numpy()
    result = census(data)
    policy = dict(campaign.protocol(), end_exclusive='2024-01-01T00:31:00Z',
                  source_end_exclusive='2024-01-01T05:00:00Z')
    review_path, review, _ = review_fixture(monkeypatch, tmp_path)
    review['reviewed_files'] = campaign.launch_file_hashes()
    directory = tmp_path/'source'
    directory.mkdir()
    for name, value in {'r3_census.json': result, 'parent_ledgers.json': {'4H_N3': parent_ledger()},
                        'launch.json': {'review_sha256': sha(review_path)}}.items():
        (directory/name).write_text(json.dumps(value))
    artifacts = {p.name: sha(p) for p in directory.iterdir()}
    (directory/'receipt.json').write_text(json.dumps({'completed': True, 'artifacts': artifacts,
        'stage': 'full_r3_source_census', 'r3_blockers': [], 'r1_blockers': ['missing_model']}))
    monkeypatch.setattr(campaign, 'load_review', lambda path, stage: review)
    monkeypatch.setattr(campaign, 'protocol', lambda: policy)
    monkeypatch.setattr(campaign.source, 'load_minutes', lambda start, end: data)
    monkeypatch.setattr(campaign.source, 'OUTPUT_ROOT', tmp_path)
    output = tmp_path/'economics'
    if fault == 'source_failure':
        def failed_load(*args):
            raise ValueError('synthetic source loading failure')
        monkeypatch.setattr(campaign.source, 'load_minutes', failed_load)
        with pytest.raises(ValueError, match='synthetic source loading failure'):
            campaign.run_score(directory, output, review_path)
        assert campaign.read_json(output/'failure.json')['completed'] is False
        assert not (output/'receipt.json').exists()
        return
    real_replay = campaign.replay_book
    def altered_replay(data, opportunities, signals, **kwargs):
        result = real_replay(data, opportunities, signals, **kwargs)
        baseline = bool(signals and signals[0]['arm'] == 'baseline')
        if baseline and ((fault == 'stress_baseline_unknown' and kwargs['occupied'] and kwargs['cost_bps'] == 24 and kwargs['delay_seconds'] == 65)
                         or (fault == 'zero_diagnostic_unknown' and kwargs['funding_mode'] == 'zero_diagnostic')):
            result['rows'][0].update(status='unknown', net_pnl=None, position=None)
        if baseline and fault == 'fixed_event_blocker' and not kwargs['occupied']:
            result['blockers'].append({'reason': 'synthetic_required_contract_failure'})
        return result
    monkeypatch.setattr(campaign, 'replay_book', altered_replay)
    receipt = campaign.run_score(directory, output, review_path)
    assert campaign.verify_artifacts(output) == receipt
    assert receipt['economic_outcomes_computed'] is True
    assert receipt['decision']['decision'] == ('blocked' if fault else 'insufficient_evidence')
    report = campaign.read_json(output/'comparison.json')
    assert report['baseline']['raw_opportunities'] == report['repair']['raw_opportunities'] == 1
    assert report['baseline']['unresolved'] == report['repair']['unresolved'] == 0
    assert report['r1_status'] == 'blocked'
    assert len(report['scenarios']) == 9
    assert len(report['primary_months']) == 32
    assert report['execution_authorized'] is False
    if fault:
        assert report['blockers']


def test_qualification_binds_code_before_rebuild_and_rejects_midrun_change(monkeypatch, tmp_path):
    from tests.research.study_fixtures import minutes
    pilot = tmp_path/'pilot'
    pilot.mkdir()
    parents = {key: {'manifest': {}} for key in ('4H_N3', '1D_N3')}
    for name, value in {'manifest.json': {'data_stream_id': 'fixture'},
                        'parent_ledgers.json': parents, 'r3_census.json': {},
                        'receipt.json': {'completed': True, 'artifacts': {}, 'cohort_blockers': {'R3': []}}}.items():
        (pilot/name).write_text(json.dumps(value))
    state = {'rebuilt': False}
    monkeypatch.setattr(campaign, 'launch_file_hashes', lambda: {'code': 'changed' if state['rebuilt'] else 'original'})
    def rebuild(*args):
        state['rebuilt'] = True
        return {key: {'manifest': {'study_adapter': {}}} for key in parents}
    monkeypatch.setattr(campaign, '_parents', rebuild)
    monkeypatch.setattr(campaign, 'build_r3_census', lambda *args, **kwargs: {})
    monkeypatch.setattr(campaign.source, 'verify_files', lambda manifest: None)
    monkeypatch.setattr(campaign.source, 'load_minutes', lambda *args: minutes())
    monkeypatch.setattr(campaign.source, 'hourly_from_minutes', lambda *args: minutes())
    monkeypatch.setattr(campaign.source, 'OUTPUT_ROOT', tmp_path)
    with pytest.raises(ValueError, match='changed during qualification'):
        campaign.qualify_continuous(pilot, tmp_path/'qualification')
