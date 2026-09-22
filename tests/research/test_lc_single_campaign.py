"""No retry, no invented review, no partial-batch outcome unlock."""
import importlib
import json

import pytest

from scripts.research.lc_judgment_runner import _save_equal, _sha, _digest
from tests.research.test_lc_context_assessment import context_request
from tests.research.test_lc_published_assessment import raw_answer


def api():
    name = 'scripts.research.lc_single_campaign'
    assert importlib.util.find_spec(name), 'single campaign missing'
    return importlib.import_module(name)


@pytest.fixture
def prepared(tmp_path):
    source = context_request()
    source_path = _save_equal(tmp_path / 'source.json', source)
    body = {'files': {str(source_path): _sha(source_path)}, 'roster': ['LC1']}
    lock = _save_equal(tmp_path / 'input_lock.json', dict(body, sha256=_digest(body)))
    directory = tmp_path / 'single'
    api().prepare_campaign(directory, [source_path], excluded_ids=set(), input_lock=lock)
    return directory, source


def delivery(reservation):
    return [dict(role='specialist', case_id=reservation['case_id'],
                 packet_sha256=reservation['envelope']['packet_sha256'],
                 chunk_index=c['index'], exec_result={'exit_code': 0, 'output': c['text']})
            for c in reservation['envelope']['chunks']]


def metadata():
    return dict(agent_id='fixture-agent', requested_model='gpt-6-astra',
                actual_model=None, actual_runtime_verified=True)


def test_no_reservation_until_explicit_launch_and_no_retry(prepared):
    path, _ = prepared
    with api().SingleCampaign(path) as run:
        with pytest.raises(ValueError, match='authoriz'): run.reserve('LC1')
        run.authorize('user approved bounded experiment')
        run.reserve('LC1')
        with pytest.raises(ValueError, match='attempt'): run.reserve('LC1')
        with pytest.raises(ValueError, match='unknown'): run.reserve('../outside')
        with pytest.raises(ValueError, match='terminal'): run.lock_terminals()


def test_successful_capture_is_unreviewed_bound_and_reopenable(prepared):
    path, source = prepared; ticks = [100]
    with api().SingleCampaign(path, monotonic_ns=lambda: ticks[0]) as run:
        run.authorize('approved test'); reserved = run.reserve('LC1')
        raw = raw_answer(source, reserved['wrapper']['request']).encode()
        ticks[0] += 90_000_000_001
        terminal = run.capture('LC1', raw, metadata(), delivery(reserved))
        assert terminal['grade']['review_status'] == 'unreviewed'
        assert terminal['grade']['status'] == 'schema_valid_unreviewed'
        assert terminal['processing_seconds'] == 91
        assert terminal['grade']['research_plan']['action'] == 'enter'
        locked = run.lock_terminals()
    with api().SingleCampaign(path) as reopened:
        assert reopened.assert_reveal_allowed() == locked
        with pytest.raises(ValueError): reopened.capture('LC1', raw, metadata(), delivery(reserved))


@pytest.mark.parametrize('failure', ['transport', 'malformed', 'timeout'])
def test_bad_captures_lock_as_null_not_rejection(prepared, failure):
    path, source = prepared; ticks = [100]
    with api().SingleCampaign(path, monotonic_ns=lambda: ticks[0]) as run:
        run.authorize('approved test'); reserved = run.reserve('LC1')
        captures = delivery(reserved)
        raw = raw_answer(source, reserved['wrapper']['request']).encode()
        if failure == 'transport': captures[0]['exec_result']['output'] = 'not original'
        if failure == 'malformed': raw = b'{'
        if failure == 'timeout': ticks[0] += 601_000_000_000
        terminal = run.capture('LC1', raw, metadata(), captures)
        assert terminal['grade']['research_plan'] is None
        assert terminal['grade']['status'] == {
            'transport': 'invalid_transport', 'malformed': 'invalid_assessment',
            'timeout': 'timeout'}[failure]
        assert run.lock_terminals()['terminals']['LC1'] == terminal


def test_restarted_uncertain_attempt_cannot_be_retried_or_given_valid_timing(prepared):
    path, source = prepared
    with api().SingleCampaign(path) as run:
        run.authorize('approved test'); reserved = run.reserve('LC1')
    with api().SingleCampaign(path) as run:
        with pytest.raises(ValueError, match='attempt'): run.reserve('LC1')
        raw = raw_answer(source, reserved['wrapper']['request']).encode()
        terminal = run.capture('LC1', raw, metadata(), delivery(reserved))
        assert terminal['grade']['status'] == 'interrupted_runtime'
        assert terminal['processing_seconds'] is None
        assert terminal['grade']['research_plan'] is None


def test_explicit_failure_and_tamper_detection(prepared):
    path, _ = prepared
    with api().SingleCampaign(path) as run:
        run.authorize('approved test'); run.reserve('LC1')
        terminal = run.fail('LC1', 'external_dispatch_failed')
        assert terminal['grade']['research_plan'] is None
        run.lock_terminals()
    target = path / 'cases' / '000' / 'terminal.json'
    changed = json.loads(target.read_text()); changed['reason'] = 'altered'
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        with api().SingleCampaign(path) as run: run.assert_reveal_allowed()


def test_second_writer_and_input_drift_are_blocked(prepared):
    path, _ = prepared
    with api().SingleCampaign(path):
        with pytest.raises(ValueError, match='owner'): api().SingleCampaign(path)
    source_path = path.parent / 'source.json'
    source_path.write_text('{}')
    with pytest.raises(ValueError): api().SingleCampaign(path)


def test_resealed_terminal_cannot_invent_a_plan_or_latency(prepared):
    path, source = prepared
    with api().SingleCampaign(path) as run:
        run.authorize('test'); reserved = run.reserve('LC1')
        run.capture('LC1', b'{', metadata(), delivery(reserved))
        target = path / 'cases' / '000' / 'terminal.json'
        changed = json.loads(target.read_text())
        changed['processing_seconds'] = 0
        changed['grade']['research_plan'] = {'action': 'reject'}
        changed.pop('sha256'); changed['sha256'] = _digest(changed)
        from scripts.research.assessment_evidence_guard import _canonical
        target.write_text(_canonical(changed))
        with pytest.raises(ValueError, match='terminal'):
            run.lock_terminals()


def test_transitive_source_lock_drift_blocks_reopen(tmp_path):
    source_path = _save_equal(tmp_path / 'source.json', context_request())
    dependency = _save_equal(tmp_path / 'upstream.json', {'value': 1})
    body = {'files': {str(p): _sha(p) for p in (source_path, dependency)}, 'roster': ['LC1']}
    lock = _save_equal(tmp_path / 'input_lock.json', dict(body, sha256=_digest(body)))
    api().prepare_campaign(tmp_path / 'single', [source_path], excluded_ids=set(), input_lock=lock)
    dependency.write_text('{}')
    with pytest.raises(ValueError): api().SingleCampaign(tmp_path / 'single')


def test_directory_sync_failure_prevents_dispatch_return(prepared, monkeypatch):
    import os
    import stat
    path, _ = prepared
    with api().SingleCampaign(path) as run:
        run.authorize('test')
        original = os.fsync
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError('directory sync failed')
            return original(fd)
        monkeypatch.setattr(os, 'fsync', fail_directory)
        with pytest.raises(OSError, match='directory sync failed'): run.reserve('LC1')
        monkeypatch.setattr(os, 'fsync', original)
        with pytest.raises(ValueError, match='attempt'): run.reserve('LC1')


def test_real_cohort_preparer_rejects_incomplete_original_roster(prepared):
    path, _ = prepared
    with pytest.raises(ValueError):
        api().prepare_remaining_campaign(path.parent / 'real', path.parent)
