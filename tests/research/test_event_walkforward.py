import importlib

import pytest


def api():
    name = 'scripts.research.event_walkforward'
    assert importlib.util.find_spec(name), 'event-aware walk-forward splitter is missing'
    return importlib.import_module(name).split_events


EVENTS = [
    dict(id='old', decision_time='2026-01-01T00:00Z', label_end='2026-01-02T00:00Z'),
    dict(id='overlap', decision_time='2026-01-31T23:00Z', label_end='2026-02-01T00:00Z'),
    dict(id='test', decision_time='2026-02-01T00:00Z', label_end='2026-02-01T04:00Z'),
    dict(id='tail', decision_time='2026-02-28T23:00Z', label_end='2026-03-01T00:00Z'),
    dict(id='later', decision_time='2026-03-01T00:00Z', label_end='2026-03-02T00:00Z'),
]


def test_calendar_split_purges_boundary_and_quarantines_test_tail():
    r = api()(EVENTS, test_start='2026-02-01T00:00Z', test_end='2026-03-01T00:00Z')
    assert r['train_ids'] == ['old']
    assert r['test_ids'] == ['test']
    assert r['excluded'] == {'overlap': 'training_label_overlap_or_gap',
                             'tail': 'test_label_outside_window', 'later': 'outside_window'}


def test_expanding_fold_can_only_learn_preceding_labels():
    r = api()(EVENTS, test_start='2026-03-01T00:00Z', test_end='2026-04-01T00:00Z')
    assert r['train_ids'] == ['old', 'overlap', 'test']
    assert r['test_ids'] == ['later']
    assert r['excluded']['tail'] == 'training_label_overlap_or_gap'


def test_explicit_gap_uses_time_not_candidate_count():
    r = api()(EVENTS, test_start='2026-02-01T00:00Z', test_end='2026-03-01T00:00Z', gap_minutes=43200)
    assert r['train_ids'] == []


def test_identical_decision_times_remain_on_same_side_and_input_order_is_stable():
    e = [dict(id='b', decision_time='2026-02-01T00:00Z', label_end='2026-02-01T01:00Z'),
         dict(id='a', decision_time='2026-02-01T00:00Z', label_end='2026-02-01T01:00Z')]
    assert api()(e, test_start='2026-02-01T00:00Z', test_end='2026-03-01T00:00Z')['test_ids'] == ['a', 'b']


@pytest.mark.parametrize('events', [EVENTS+EVENTS[:1], [dict(id='x', decision_time='2026-02-01', label_end='2026-02-02')],
    [dict(id='x', decision_time='2026-02-02T00:00Z', label_end='2026-02-01T00:00Z')]])
def test_bad_event_intervals_rejected(events):
    with pytest.raises(ValueError):
        api()(events, test_start='2026-02-01T00:00Z', test_end='2026-03-01T00:00Z')


@pytest.mark.parametrize('gap', [-1, True, 1.5])
def test_invalid_gap_rejected(gap):
    with pytest.raises(ValueError):
        api()(EVENTS, test_start='2026-02-01T00:00Z', test_end='2026-03-01T00:00Z', gap_minutes=gap)
