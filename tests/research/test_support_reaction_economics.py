from copy import deepcopy
import importlib
import importlib.util
import json
import signal

import pandas as pd
import pytest

from scripts.research.support_reaction import assess, origin_record
from scripts.research.support_reaction_replay import compare, replay
from scripts.research.thesis_contract import event, protocol, signed
from scripts.research.thesis_sequence import compile_episode
from tests.research.support_reaction_fixtures import fixture


def api():
    assert importlib.util.find_spec('scripts.research.support_reaction_economics'), 'bounded economics missing'
    return importlib.import_module('scripts.research.support_reaction_economics')


def sample(at='2024-01-02T04:00Z', *, missing_volume=False, ceiling=150.):
    raw, bars = fixture(missing_volume=missing_volume, ceiling=ceiling)
    delta = pd.Timestamp(at)-pd.Timestamp(raw['origin']['available_at'])
    shifted = lambda e: event(e['kind'], e['timeframe'], pd.Timestamp(e['start'])+delta,
                             pd.Timestamp(e['end'])+delta, e['payload'], stream_id=e['stream_id'])
    base = deepcopy(raw); base['origin'] = shifted(raw['origin'])
    base['parent']['available_at'] = (pd.Timestamp(base['parent']['available_at'])+delta).isoformat()
    base['parent']['id'] = str(at)
    raw = compile_episode(base, [shifted(e) for e in raw['events']])
    bars.index += delta
    bars.loc[pd.Timestamp(at)+pd.Timedelta('2h12min'), 'high'] = 140.
    r = assess(origin_record(raw), bars, as_of=raw['deadline'])
    return r, raw, bars


def together(*cases):
    bars = pd.concat([c[2] for c in cases])
    return [c[0] for c in cases], [c[1] for c in cases], bars.loc[~bars.index.duplicated(keep='last')].sort_index()


@pytest.mark.parametrize('scenario', ['primary', 'stress'])
def test_sparse_books_preserve_continuous_rows_entries_cashflows_and_admission(scenario):
    rs, ps, bars = together(sample(), sample('2024-01-10T04:00Z'))
    settings = protocol()[scenario]
    for arm in 'ABC':
        for capacity in (False, True):
            old = replay(rs, ps, bars, arm=arm, execution=settings, capacity=capacity)
            new = api().partitioned_book(rs, ps, bars, arm=arm, execution=settings, capacity=capacity)
            assert {r['episode_id']: r for r in new['rows']} == {r['episode_id']: r for r in old['rows']}
            assert new['entry_tape'] == old['runtime']['entry_tape']
            assert new['positions'] == old['runtime']['positions']
            assert new['admission'] == old['admission']
            assert new['global_drawdown'] is None
    assert len(api().components(rs)) == 2


def test_equal_deadline_boundary_components_merge_without_using_actual_exits():
    a, b = sample(), sample('2024-01-09T04:00Z')
    groups = api().components([b[0], a[0]])
    assert len(groups) == 1
    assert groups[0]['ids'] == [a[1]['id'], b[1]['id']]
    assert groups[0]['end'] == '2024-01-16T04:00:00+00:00'


def test_pending_reservation_is_not_removed_by_later_admission_failure():
    r, raw, bars = sample()
    base = deepcopy(raw); base['parent']['id'] = 'second-parent'
    other = compile_episode(base, raw['events'])
    r2 = assess(origin_record(other), bars, as_of=other['deadline'])
    bars.loc['2024-01-02T06:11Z', ['open', 'high', 'low', 'close']] = [125.,126.,124.,125.]
    old = replay([r,r2], [raw,other], bars, arm='B', capacity=True)
    new = api().partitioned_book([r,r2], [raw,other], bars, arm='B', capacity=True)
    assert new['rows'] == old['rows']
    assert sorted(r['status'] for r in new['rows']) == ['busy', 'no_entry']


def test_sticky_unknown_occupancy_must_not_reset_between_components():
    rs, ps, bars = together(sample(missing_volume=True), sample('2024-01-10T04:00Z'))
    free = api().partitioned_book(rs, ps, bars, arm='C')
    assert free['rows'][0]['net'] is None
    assert free['rows'][1]['net'] > 0
    with pytest.raises(api().UnknownOccupancy, match='sticky unknown'):
        api().partitioned_book(rs, ps, bars, arm='C', capacity=True)


def test_missing_execution_tail_cannot_be_pruned_as_no_entry():
    r, raw, bars = sample()
    bars.loc['2024-01-02T06:11Z', 'close'] = float('nan')
    new = api().partitioned_book([r], [raw], bars, arm='B')
    assert new['rows'][0]['net'] is None
    assert new['entry_tape'][0]['at'] == '2024-01-02T06:11:00+00:00'
    with pytest.raises(api().UnknownOccupancy):
        api().partitioned_book([r], [raw], bars, arm='B', capacity=True)


def test_sparse_replay_preserves_full_deadline_holding_and_funding_cashflows():
    raw,bars=fixture()  # deliberately no early target: hold for the full horizon
    r=assess(origin_record(raw),bars,as_of=raw['deadline'])
    old=replay([r],[raw],bars,arm='B',capacity=True)
    new=api().partitioned_book([r],[raw],bars,arm='B',capacity=True)
    pos=new['positions'][raw['id']]
    assert pos==old['runtime']['positions'][raw['id']]
    assert pos['exposure_minutes']==9949
    assert pos['cashflows'][-1]['at']=='2024-01-09T04:00:00+00:00'
    assert pos['cashflows'][-1]['reason']=='deadline'
    assert len([c for c in pos['cashflows'] if c['kind']=='funding'])==21


def test_sparse_chronological_metrics_use_purged_test_ids_and_full_denominator():
    rs, ps, bars = together(sample('2025-04-23T04:00Z'), sample('2025-04-28T04:00Z'))
    old = compare(rs, ps, bars)
    new = api().partitioned_compare(rs, ps, bars)
    for key in ('arms', 'pairs', 'folds', 'dependence', 'raw_episodes'):
        assert new[key] == old[key]
    assert new['raw_episodes'] == 2
    for x,y in zip(new['chronological_reports'], old['chronological_reports']):
        for key in ('split', 'test_raw_episodes', 'arms', 'pairs', 'occupied_arms'):
            assert x[key] == y[key]
    assert new['chronological_reports'][0]['arms']['B']['closed_fills'] == 1


def test_bounded_stage_does_not_run_computation_without_clearance_or_on_changed_binding(tmp_path):
    data = tmp_path/'input'; data.write_text('one')
    files = {str(data): api().sha(data)}
    called = []
    review = signed(dict(status='blocked', files=files))
    with pytest.raises(ValueError): api().bounded_economics(tmp_path/'blocked', review, files, lambda out: called.append(True))
    assert not called
    review = signed(dict(status='exploratory_economics_cleared', files=files))
    data.write_text('two')
    with pytest.raises(ValueError): api().bounded_economics(tmp_path/'changed', review, files, lambda out: called.append(True))
    assert not called
    assert (tmp_path/'changed'/'failure.json').exists()


def test_bounded_stage_is_exclusive_verifies_after_and_leaves_failure_not_success(tmp_path):
    data = tmp_path/'input'; data.write_text('one'); files={str(data):api().sha(data)}
    review = signed(dict(status='exploratory_economics_cleared', files=files))
    def compute(out):
        out.write('result.json', signed(dict(example='synthetic')))
        data.write_text('two')
    with pytest.raises(ValueError): api().bounded_economics(tmp_path/'run', review, files, compute)
    assert (tmp_path/'run'/'failure.json').exists()
    assert not (tmp_path/'run'/'receipt.json').exists()
    with pytest.raises(FileExistsError): api().bounded_economics(tmp_path/'run', review, files, compute)


def test_runtime_alarm_writes_failure_without_success_and_restores_signal_handler(tmp_path):
    data=tmp_path/'input'; data.write_text('one'); files={str(data):api().sha(data)}
    review=signed(dict(status='exploratory_economics_cleared',files=files))
    handler=signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        api().bounded_economics(tmp_path/'run',review,files,lambda out:signal.raise_signal(signal.SIGALRM))
    failure=json.loads((tmp_path/'run'/'failure.json').read_text())
    assert failure['error_type']=='TimeoutError'
    assert signal.getsignal(signal.SIGALRM)==handler
    assert not (tmp_path/'run'/'receipt.json').exists()


def test_missing_review_preflight_is_within_failure_receipt_boundary(tmp_path):
    with pytest.raises(FileNotFoundError):
        api().run(tmp_path/'run',tmp_path/'missing_review.json')
    assert (tmp_path/'run'/'failure.json').exists()
    assert not (tmp_path/'run'/'receipt.json').exists()
