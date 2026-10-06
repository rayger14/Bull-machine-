from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.thesis_contract import event, seal, signed
from scripts.research.thesis_execution import replay_book
from scripts.research.thesis_sequence import compile_episode
from tests.research.thesis_fixtures import base_episode, candle_event, minutes, packet, sequence_events


def replay(p=None, bars=None, **kwargs):
    return replay_book([packet([]) if p is None else p], minutes() if bars is None else bars,
                       kwargs.pop('entry', 'simple'), kwargs.pop('management', 'fixed'), **kwargs)


def position(book):
    return next(iter(book['positions'].values()))


def target_path():
    bars = minutes(periods=10)
    bars.loc['2024-01-01T04:05Z', 'high'] = 119.
    return bars


def partial_packet():
    return packet([candle_event('2024-01-01T04:00Z', low=101., high=122., close=121.)])


def test_literal_risk_actual_fees_target_and_delay():
    b = replay(bars=target_path())
    p = position(b)
    q = 100./7.1206
    assert p['entry_time'] == '2024-01-01T04:02:00+00:00'
    assert p['original_qty'] == pytest.approx(q)
    assert p['exit_time'] == '2024-01-01T04:06:00+00:00'
    assert p['cashflows'][-1]['price'] == 118.
    assert p['net'] == pytest.approx(q*14-q*(104+118)*.0006)
    assert b['rows'][0]['status'] == 'closed'


def test_same_minute_stop_wins_over_target_and_gap_is_worse():
    bars = target_path()
    bars.loc['2024-01-01T04:05Z', 'low'] = 96.
    p = position(replay(bars=bars))
    assert p['cashflows'][-1]['price'] == 97.
    assert p['net'] == pytest.approx(-100.)
    bars = minutes(periods=10)
    bars.loc['2024-01-01T04:05Z', ['open', 'high', 'low', 'close']] = [95., 96., 94., 95.]
    assert position(replay(bars=bars))['cashflows'][-1]['price'] == 95.


def test_pending_stop_touch_cancels_before_entry():
    bars = minutes(periods=10)
    bars.loc['2024-01-01T04:01Z', 'low'] = 96.
    b = replay(bars=bars)
    assert not b['positions']
    assert b['rows'][0]['reason'] == 'pending_stop_touch'


def test_partial_is_original_quarter_once_and_funding_on_remaining():
    bars = minutes(periods=250)
    bars.loc['2024-01-01T05:00Z':, ['open', 'close']] = 121.
    bars.loc['2024-01-01T05:00Z':, ['high', 'low']] = [122., 120.]
    p = position(replay(partial_packet(), bars, management='adaptive', until='2024-01-01T08:00Z'))
    q = 100/7.1206
    reductions = [f for f in p['cashflows'] if f['kind'] == 'partial']
    assert len(reductions) == 1
    assert reductions[0]['at'] == '2024-01-01T05:02:00+00:00'
    assert reductions[0]['quantity'] == pytest.approx(q*.25)
    assert p['remaining_qty'] == pytest.approx(q*.75)
    assert p['funding'] == pytest.approx(q*.75*104*.0008)
    assert p['fees'] == pytest.approx(q*104*.0006+q*.25*121*.0006)
    assert p['destinations']['range'] == 'executed'


def test_old_stop_gap_beats_due_reduction_and_cancels_it():
    bars = minutes(periods=70)
    bars.loc['2024-01-01T05:02Z', ['open', 'high', 'low', 'close']] = [96., 96., 95., 96.]
    p = position(replay(partial_packet(), bars, management='adaptive'))
    assert p['cashflows'][-1]['price'] == 96.
    assert not any(f['kind'] == 'partial' for f in p['cashflows'])
    assert p['destinations']['range'] == 'cancelled'


def pivot_packet():
    pivot = event('pivot_low', '1h', '2024-01-01T04:00Z', '2024-01-01T07:00Z',
                  {'price': 104., 'decision_close': 110., 'center_start': '2024-01-01T04:00:00+00:00'},
                  stream_id='fixture', input_ids=['a', 'b', 'c', 'd', 'e'])
    return packet([pivot])


def test_trail_cannot_protect_its_observation_or_latency_bar():
    bars = minutes(periods=200, price=110.)
    bars.loc[:'2024-01-01T06:59Z', ['open', 'high', 'low', 'close']] = [104., 105., 103.5, 104.]
    bars.loc['2024-01-01T07:01Z', 'low'] = 102.
    b = replay(pivot_packet(), bars, management='adaptive', until='2024-01-01T07:03Z')
    p = position(b)
    assert p['status'] == 'open'
    assert p['effective_stop'] == 103.
    assert [a['at'] for a in p['actions'] if a['kind'] == 'stop_effective'] == ['2024-01-01T07:02:00+00:00']
    bars.loc['2024-01-01T07:02Z', 'low'] = 102.
    assert position(replay(pivot_packet(), bars, management='adaptive'))['exit_time'] == '2024-01-01T07:03:00+00:00'


def test_funding_precedes_prior_minute_stop_at_settlement():
    bars = minutes(periods=245)
    bars.loc['2024-01-01T07:59Z', 'low'] = 96.
    p = position(replay(bars=bars))
    assert p['cashflows'][-2]['kind'] == 'funding'
    assert p['cashflows'][-1]['at'] == '2024-01-01T08:00:00+00:00'
    assert p['funding'] == pytest.approx((100/7.1206)*104*.0008)


def test_unknown_execution_path_stays_unknown_not_liquidated_at_end():
    bars = minutes(periods=10).drop(pd.Timestamp('2024-01-01T04:04Z'))
    b = replay(bars=bars)
    assert b['rows'][0]['status'] == 'unknown'
    assert position(b)['net'] is None
    assert b['unknown_occupancy'] is True
    assert not any(f['kind'] == 'exit' for f in position(b)['cashflows'])


def test_structural_unknown_does_not_poison_fixed_after_fill():
    p = packet([candle_event('2024-01-01T04:00Z', status='unknown')])
    bars = minutes(periods=70)
    bars.loc['2024-01-01T05:05Z', 'high'] = 119.
    assert replay(p, bars)['rows'][0]['status'] == 'closed'
    assert replay(p, bars, management='adaptive')['rows'][0]['status'] == 'unknown'


def test_restart_with_pending_partial_is_exact_and_foreign_binding_rejected():
    p, bars = partial_packet(), minutes(periods=75)
    bars.loc['2024-01-01T05:05Z', 'low'] = 96.
    full = replay(p, bars, management='adaptive')
    prefix = replay(p, bars, management='adaptive', until='2024-01-01T05:01Z')
    resumed = replay(p, bars, management='adaptive', checkpoint=prefix['checkpoint'])
    assert resumed == full
    with pytest.raises(ValueError, match='binding'):
        replay(p, bars, checkpoint=prefix['checkpoint'])
    altered = bars.copy()
    altered.loc['2024-01-01T04:10Z', 'close'] += .01
    with pytest.raises(ValueError, match='prefix'):
        replay(p, altered, management='adaptive', checkpoint=prefix['checkpoint'])


def test_capacity_free_fixed_adaptive_entry_tapes_identical():
    p, bars = partial_packet(), target_path()
    a = replay(p, bars, capacity=False)
    b = replay(p, bars, capacity=False, management='adaptive')
    assert a['entry_tape'] == b['entry_tape']


def test_fib_and_gann_reviews_exit_after_delay_not_before_entry():
    p = packet()
    bars = minutes(periods=370)
    b = replay(p, bars, management='adaptive', until='2024-01-01T09:03Z')
    pos = position(b)
    # 08:00 has a new trigger since entry; 10:00 next review would see no progress.
    reviews = [a for a in pos['actions'] if a['kind'] == 'clock_review']
    assert reviews[0]['at'] == '2024-01-01T08:00:00+00:00'
    assert reviews[0]['action'] == 'hold'
    p = signed(dict(p, reviews=[{'id': 'fixture-review', 'at': '2024-01-01T05:00:00+00:00',
                                'families': ['gann'], 'input_ids': [p['origin']['id']]}], milestones=[]))
    pos = position(replay(p, bars, management='adaptive', until='2024-01-01T05:03Z'))
    assert pos['exit_time'] == '2024-01-01T05:02:00+00:00'
    assert pos['cashflows'][-1]['reason'] == 'no_progress'


def test_occupied_books_are_independent_but_shadow_entries_clone_exactly():
    base = base_episode()
    base['parent'].update(id='parent2', lineage_id='line2', available_at='2024-01-01T07:00:00+00:00')
    base['origin'] = candle_event('2024-01-01T08:00Z', '4h', low=98., high=108., close=102.)
    second = compile_episode(base, [])
    ps, bars = [packet([]), second], minutes(periods=490)
    bars.loc['2024-01-01T04:05Z', 'high'] = 119.
    bars.loc['2024-01-01T12:05Z', 'low'] = 96.
    fixed = replay_book(ps, bars, 'simple', 'fixed')
    adaptive = replay_book(ps, bars, 'simple', 'adaptive')
    assert len(fixed['entry_tape']) == 2
    assert len(adaptive['entry_tape']) == 1
    assert adaptive['rows'][1]['status'] == 'busy'
    fixed_shadow = replay_book(ps, bars, 'simple', 'fixed', capacity=False)
    adaptive_shadow = replay_book(ps, bars, 'simple', 'adaptive', capacity=False)
    assert fixed_shadow['entry_tape'] == adaptive_shadow['entry_tape']


def test_deadline_is_exact_with_no_observation_latency():
    p = signed(dict(packet([]), deadline='2024-01-01T04:05:00+00:00'))
    pos = position(replay(p, minutes(periods=8)))
    assert pos['exit_time'] == '2024-01-01T04:05:00+00:00'
    assert pos['cashflows'][-1]['reason'] == 'deadline'


def test_both_destinations_reduce_half_original_and_each_only_once():
    p = packet(sequence_events()+[candle_event('2024-01-01T08:00Z', low=118., high=124., close=123., opened=120.),
                                  candle_event('2024-01-01T09:00Z', low=118., high=124., close=123., opened=120.)])
    bars = minutes(periods=380)
    pos = position(replay(p, bars, management='adaptive', until='2024-01-01T10:03Z'))
    reductions = [f for f in pos['cashflows'] if f['kind'] == 'partial']
    assert len(reductions) == 1
    assert reductions[0]['quantity'] == pytest.approx(pos['original_qty']*.5)
    assert pos['destinations'] == {'range': 'executed', 'fib': 'executed'}


def test_known_failed_touch_stays_known_no_entry_despite_later_structure_gap():
    bad = candle_event('2024-01-01T06:00Z', low=106., high=110., close=107., opened=109.)
    p = packet(sequence_events()[:2]+[bad, candle_event('2024-01-01T08:00Z', status='unknown')])
    b = replay(p, minutes(periods=400), entry='thesis')
    assert b['rows'][0]['status'] == 'no_entry'
    assert b['rows'][0]['net'] == 0.


@pytest.mark.parametrize('p', [packet([]), packet()])
def test_unfinished_watcher_is_unknown_not_known_zero(p):
    b = replay(p, minutes(periods=10), entry='thesis')
    assert b['rows'][0]['status'] == 'unknown'
    assert b['rows'][0]['net'] is None


def test_future_known_closure_does_not_finalize_truncated_watch():
    bad = candle_event('2024-01-01T06:00Z', low=106., high=110., close=107., opened=109.)
    p = packet(sequence_events()[:2]+[bad])
    assert replay(p, minutes(periods=10), entry='thesis')['rows'][0]['status'] == 'unknown'
