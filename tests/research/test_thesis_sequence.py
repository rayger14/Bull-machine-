from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.thesis_contract import clock, event, protocol, seal, verify_packet
from scripts.research.thesis_sequence import compile_episode
from tests.research.thesis_fixtures import base_episode, candle_event, packet, sequence_events


def test_complete_sequence_literal_clocks_and_immutable_fib_anchors():
    p = packet()
    assert p['original_stop'] == 97.
    assert p['deadline'] == '2024-01-08T04:00:00+00:00'
    assert [m['kind'] for m in p['milestones']] == ['test', 'strength', 'last_support', 'trigger']
    assert p['entry_intents']['simple']['decision_time'] == '2024-01-01T04:00:00+00:00'
    assert p['entry_intents']['thesis']['decision_time'] == '2024-01-01T07:01:00+00:00'
    assert p['entry_intents']['thesis']['expires_at'] == '2024-01-01T07:15:00+00:00'
    assert p['fib']['a'] == 98. and p['fib']['b'] == 111.
    assert p['fib']['levels']['1.618'] == pytest.approx(119.034)
    assert p['fib']['available_at'] == '2024-01-01T06:00:00+00:00'
    assert p['reviews'][0]['at'] == '2024-01-01T08:00:00+00:00'
    assert 'gann' in p['reviews'][-1]['families']
    verify_packet(p)


def test_append_does_not_reanchor_or_change_past_entry_ids():
    p = packet()
    later = candle_event('2024-01-01T08:00Z', low=105., high=130., close=125., opened=106.)
    extended = packet(sequence_events()+[later])
    assert extended['id'] == p['id']
    assert extended['fib'] == p['fib']
    assert extended['entry_intents'] == p['entry_intents']
    assert extended['milestones'] == p['milestones']


def test_strength_before_test_cannot_be_reused_and_same_bar_cannot_advance_twice():
    test = candle_event('2024-01-01T04:00Z', low=99., high=112., close=111., opened=103.)
    p = packet([test])
    assert [m['kind'] for m in p['milestones']] == ['test']
    assert p['entry_intents']['thesis'] is None


def test_first_failed_last_support_touch_cancels_even_if_later_touch_is_good():
    rows = sequence_events()
    bad = candle_event('2024-01-01T06:00Z', low=106., high=110., close=107., opened=109.)
    p = packet(rows[:2]+[bad]+[candle_event('2024-01-01T07:00Z', low=107., high=111., close=110., opened=109.)])
    assert p['sequence_status'] == 'failed_touch'
    assert p['entry_intents']['thesis'] is None
    assert p['terminal_at'] is None


def test_late_test_does_not_extend_sequence_deadlines():
    late = candle_event('2024-01-02T04:00Z', low=99., high=105., close=102., opened=102.)
    assert packet([late])['milestones'] == []


def test_minute_trigger_is_exclusive_at_last_support_expiry():
    rows = sequence_events()[:3]
    rows += [candle_event('2024-01-01T07:14Z', '1min', low=109., high=112., close=111., opened=109.)]
    assert packet(rows)['entry_intents']['thesis'] is None


def test_invalidation_cancels_later_sequence_but_preserves_prior_entry():
    rows = sequence_events()
    rows += [candle_event('2024-01-01T04:00Z', '4h', low=97., high=110., close=99.)]
    p = packet(rows)
    assert p['terminal_at'] == '2024-01-01T08:00:00+00:00'
    assert p['entry_intents']['thesis']['decision_time'] == '2024-01-01T07:01:00+00:00'


def test_unknown_has_first_availability_not_retroactive_suppression():
    p = packet(sequence_events()+[candle_event('2024-01-01T08:00Z', status='unknown')])
    assert p['unknown_at'] == '2024-01-01T09:00:00+00:00'
    assert p['entry_intents']['thesis'] is not None
    assert p['source_status'] == 'known'


@pytest.mark.parametrize('mutation', ['parent', 'lineage', 'authority', 'atr'])
def test_bad_initial_contract_fails_closed(mutation):
    base = base_episode()
    if mutation == 'parent':
        base['parent']['available_at'] = base['origin']['start']
    elif mutation == 'lineage':
        base['parent']['lineage_id'] = ''
    elif mutation == 'authority':
        base['execution_authorized'] = True
    else:
        base['atr4h'] = float('nan')
    with pytest.raises(ValueError):
        compile_episode(base, [])


def test_missing_initial_atr_keeps_raw_episode_unknown_not_zero_trade():
    base = base_episode()
    base['atr4h'] = None
    p = compile_episode(base, [])
    assert p['source_status'] == 'unknown'
    assert p['entry_intents'] == {'simple': None, 'thesis': None}


def test_event_clock_and_source_validation():
    p = sequence_events()
    p[0]['available_at'] = '2024-01-01T04:00:00+00:00'
    with pytest.raises(ValueError, match='event'):
        packet(p)
    with pytest.raises(ValueError):
        clock('2024-01-01')
    with pytest.raises(ValueError):
        event('candle', '1h', '2024-01-01T00:00Z', '2024-01-01T01:00Z', {'close': float('nan')})


def test_seal_tamper_and_duplicate_events_are_rejected():
    p = packet()
    altered = deepcopy(p)
    altered['original_stop'] = 90.
    with pytest.raises(ValueError):
        verify_packet(altered)
    rows = sequence_events()
    with pytest.raises(ValueError, match='duplicate'):
        packet(rows+rows[:1])
    assert protocol()['execution_authorized'] is False
    assert seal({'a': 1}) == seal({'a': 1})


def test_equal_clock_invalidation_beats_last_support():
    rows = sequence_events()[:3]
    rows[2] = candle_event('2024-01-01T07:00Z', '1h', low=107., high=110., close=109., opened=109.)
    rows += [candle_event('2024-01-01T08:00Z', '1min', low=109., high=112., close=111., opened=109.),
             candle_event('2024-01-01T04:00Z', '4h', low=97., high=112., close=99.)]
    assert packet(rows)['entry_intents']['thesis'] is None


@pytest.mark.parametrize('timeframe,start', [('4H', '2024-01-01T04:00Z'), ('1h', '2024-01-01T04:59Z')])
def test_noncanonical_or_off_grid_event_rejected(timeframe, start):
    with pytest.raises(ValueError, match='timeframe|grid'):
        packet([candle_event(start, timeframe)])


def test_spring_origin_must_be_four_hour_candle():
    base = base_episode()
    base['origin'] = candle_event('2024-01-01T00:00Z', '1h', low=98., high=108., close=102.)
    with pytest.raises(ValueError, match='origin'):
        compile_episode(base, [])


def test_conflicting_candles_for_same_observation_rejected():
    rows = sequence_events()
    rows += [candle_event('2024-01-01T04:00Z', low=99., high=106., close=103.)]
    with pytest.raises(ValueError, match='duplicate'):
        packet(rows)


def test_missing_trigger_minute_cannot_be_skipped_for_later_better_confirmation():
    missing = candle_event('2024-01-01T07:00Z', '1min', status='unknown')
    late = candle_event('2024-01-01T07:01Z', '1min', low=109., high=112., close=111., opened=109.)
    p = packet(sequence_events()[:3]+[missing, late])
    assert p['entry_intents']['thesis'] is None
    assert p['unknown_at'] == '2024-01-01T07:01:00+00:00'


def test_complete_failed_test_window_expires_before_later_unknown():
    rows = [candle_event(t) for t in pd.date_range('2024-01-01T04:00Z', periods=24, freq='h')]
    assert packet(rows)['entry_closed_at'] == '2024-01-02T04:00:00+00:00'
    p = packet(rows+[candle_event('2024-01-02T04:00Z', status='unknown')])
    assert p['entry_closed_at'] == '2024-01-02T04:00:00+00:00'
    assert p['unknown_at'] == '2024-01-02T05:00:00+00:00'
