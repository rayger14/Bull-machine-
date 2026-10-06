"""Real disk/lease/hash/timing boundaries, with no paid or external dispatch."""
import importlib
import json
from copy import deepcopy

import pytest

from scripts.research.lc_judgment_runner import _load, _save_equal, _sha, _digest
from tests.research.lc_practice_fixtures import saved_inputs
from tests.research.lc_structure_fixtures import structure_answer


def api():
    name = 'scripts.research.lc_practice_runtime'
    assert importlib.util.find_spec(name), 'practice runtime missing'
    return importlib.import_module(name)


def setup(tmp_path, count=2):
    paths, lock, archive = saved_inputs(tmp_path, count)
    root = tmp_path / 'run'
    manifest = api().prepare_run(root, paths[::-1], input_lock=lock, archive=archive, limit=count)
    return root, manifest


def response(reservation, decision='reject'):
    request = json.loads(bytes.fromhex(reservation['request_hex']))
    answer = structure_answer(request['packet'], request['policy'], decision)
    return json.dumps(answer).encode(), bytes.fromhex(reservation['request_hex'])


def metadata(agent='agent-a'):
    return dict(agent_id=agent, requested_model='gpt-6-astra', requested_effort='high',
                observed_model=None, observed_snapshot=None, delivery_complete=True)


def test_preparation_orders_cases_freezes_requests_and_does_not_reveal(tmp_path):
    root, m = setup(tmp_path)
    assert [c['case_id'] for c in m['cases']] == ['LC0', 'LC1']
    assert all(c['source_status'] == 'verified' for c in m['cases'])
    with api().PracticeRun(root) as run:
        assert run.status()['pending'] == 2
        with pytest.raises(ValueError, match='authorized'): run.reserve('LC0')
        with pytest.raises(ValueError, match='terminal'): run.assert_reveal_allowed()
    assert not (root / 'case_results.json').exists()


def test_exact_bytes_and_measured_time_survive_lock_and_reopen(tmp_path):
    root, m = setup(tmp_path, 1); now = [1000000000]
    with api().PracticeRun(root, monotonic_ns=lambda: now[0]) as run:
        run.authorize('Explicit synthetic test authorization', m['sha256'])
        res = run.reserve('LC0'); raw, delivered = response(res)
        run.attach('LC0', 'agent-a')
        now[0] += 125500000000
        t = run.capture('LC0', raw, metadata(), delivered)
        assert t['status'] == 'valid_reject'
        assert t['elapsed_seconds'] == 125.5
        assert t['net_pnl'] is None  # Outcome book, not recorder, owns zero exposure.
        locked = run.lock_terminals()
        assert bytes.fromhex(_load(root / 'cases/000/capture.json')['raw_hex']) == raw
    with api().PracticeRun(root) as run:
        assert run.assert_reveal_allowed()['sha256'] == locked['sha256']


def test_single_owner_single_inflight_and_no_retry(tmp_path):
    root, m = setup(tmp_path)
    with api().PracticeRun(root) as run:
        with pytest.raises(ValueError, match='owner'): api().PracticeRun(root)
        run.authorize('synthetic', m['sha256']); res = run.reserve('LC0')
        with pytest.raises(ValueError, match='in.flight'): run.reserve('LC1')
        with pytest.raises(ValueError, match='attempt'): run.reserve('LC0')
        run.fail('LC0', 'test dispatch failed')
        run.reserve('LC1'); run.fail('LC1', 'test dispatch failed')
        assert run.lock_terminals()['outcome_reveal_authorized'] is True


@pytest.mark.parametrize('bad', ['bytes', 'identity', 'model', 'effort', 'truncated', 'json', 'timeout'])
def test_bad_capture_remains_null_and_cannot_be_repaired(tmp_path, bad):
    root, m = setup(tmp_path, 1); now = [0]
    with api().PracticeRun(root, monotonic_ns=lambda: now[0]) as run:
        run.authorize('synthetic', m['sha256']); res = run.reserve('LC0')
        run.attach('LC0', 'agent-a'); raw, delivered = response(res); meta = metadata()
        if bad == 'bytes': delivered = delivered[:-1]
        if bad == 'identity': meta['agent_id'] = 'wrong'
        if bad == 'model': meta['observed_model'] = 'another-model'
        if bad == 'effort': meta['requested_effort'] = 'low'
        if bad == 'truncated': meta['delivery_complete'] = False
        if bad == 'json': raw = b'not json'
        now[0] = (601 if bad == 'timeout' else 1) * 1000000000
        terminal = run.capture('LC0', raw, meta, delivered)
        assert terminal['status'] in ('invalid_transport', 'invalid', 'timeout')
        assert terminal['net_pnl'] is None
        with pytest.raises(ValueError): run.capture('LC0', raw, meta, delivered)


def test_restart_marks_inflight_interrupted_and_unstarted_remain_available(tmp_path):
    root, m = setup(tmp_path)
    with api().PracticeRun(root) as run:
        run.authorize('synthetic', m['sha256']); run.reserve('LC0')
    with api().PracticeRun(root) as run:
        assert run.status()['terminals']['LC0'] == 'interrupted'
        run.mark_not_run('LC1', 'bounded test ends here')
        locked = run.lock_terminals()
        assert locked['terminals']['LC0']['elapsed_seconds'] is None
        assert locked['terminals']['LC1']['status'] == 'not_run'


def test_mismatched_source_keeps_roster_row_unavailable_without_call(tmp_path):
    paths, lock, archive = saved_inputs(tmp_path, 1)
    import pandas as pd
    bars = pd.read_parquet(archive); bars.loc['2026-01-01T03:59:00Z', 'close'] = 106.
    bars.to_parquet(archive)
    body = dict(roster=['LC0'], files={str(p.resolve()): _sha(p) for p in paths+[archive]})
    new_lock = _save_equal(tmp_path/'changed_lock.json', dict(body, sha256=_digest(body)))
    root = tmp_path/'run'
    m = api().prepare_run(root, paths, input_lock=new_lock, archive=archive, limit=1)
    assert m['cases'][0]['source_status'] == 'unavailable'
    with api().PracticeRun(root) as run:
        run.authorize('synthetic', m['sha256'])
        with pytest.raises(ValueError): run.reserve('LC0')
        assert run.lock_terminals()['terminals']['LC0']['status'] == 'source_unavailable'


def test_changed_frozen_bytes_and_wrong_manifest_authorization_are_rejected(tmp_path):
    root, m = setup(tmp_path, 1)
    with api().PracticeRun(root) as run:
        with pytest.raises(ValueError): run.authorize('synthetic', '0'*64)
    request = root/'cases/000/request.json'
    request.write_bytes(request.read_bytes()+b' ')
    with pytest.raises(ValueError, match='changed'): api().PracticeRun(root)
