from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.study_execution import position_terms, replay_book
from tests.research.study_fixtures import minutes


def bars(count=10, start='2024-01-01T00:00Z'):
    data = minutes(count, start, 100.)
    data['high'], data['low'] = 100.2, 99.8
    return data


def opportunity(oid='a', decision='2024-01-01T00:00Z', family='R3', source_status='complete'):
    return {'id': oid, 'origin_time': pd.Timestamp(decision).isoformat(), 'family': family,
            'instrument': 'BTC', 'data_stream_id': 'fixture', 'source_status': source_status,
            'source_available_at': pd.Timestamp(decision).isoformat()}


def signal(oid='a', decision='2024-01-01T00:00Z', family='R3', stop=99.):
    at = pd.Timestamp(decision)
    return {'opportunity_id': oid, 'family': family, 'arm': 'baseline', 'decision_time': at.isoformat(),
            'stop': stop, 'entry_expiry': (at + pd.Timedelta(minutes=5 if family == 'R3' else 15)).isoformat(),
            'exit_deadline': (at + pd.Timedelta(hours=4 if family == 'R3' else 24)).isoformat(),
            'parent_lineage_id': 'parent' if family == 'R3' else None}


def replay(data=None, ops=None, signals=None, **kwargs):
    data = bars() if data is None else data
    options = {'as_of': data.index[-1] + pd.Timedelta('1min'), 'cost_bps': 12,
               'delay_seconds': 5, 'funding_mode': 'zero_diagnostic'}
    options.update(kwargs)
    return replay_book(data, [opportunity()] if ops is None else ops,
                       [signal()] if signals is None else signals, **options)


def test_cost_inclusive_risk_and_notional_cap():
    p = position_terms(100., 99., 12)
    assert p['quantity'] == pytest.approx(100 / 1.12)
    assert p['initial_risk'] == pytest.approx(100)
    assert p['target'] == 102.
    assert p['entry_fee'] == p['exit_fee'] == pytest.approx(p['quantity'] * 100 * .0006)
    capped = position_terms(100., 99.999, 12)
    assert capped['quantity'] * 100 == pytest.approx(50000)
    assert capped['initial_risk'] < 100


@pytest.mark.parametrize('entry,stop,cost', [(True, 99, 12), (100, 100, 12), (100, 0, 12),
                                           (float('nan'), 99, 12), (100, 99, -1)])
def test_invalid_geometry_or_economics_rejected(entry, stop, cost):
    with pytest.raises(ValueError):
        position_terms(entry, stop, cost)


def test_entry_bar_ambiguous_bracket_is_stop_first_and_net_r_includes_costs():
    data = bars()
    data.iloc[1, data.columns.get_indexer(['high', 'low'])] = [103., 98.5]
    result = replay(data)
    row = result['rows'][0]
    assert row['status'] == 'closed'
    assert row['position']['entry_time'] == '2024-01-01T00:01:00+00:00'
    assert row['position']['exit_time'] == '2024-01-01T00:02:00+00:00'
    assert row['reason'] == 'stop'
    assert row['net_pnl'] == pytest.approx(-100)
    assert row['position']['net_r'] == pytest.approx(-1)


@pytest.mark.parametrize('delay,entry_minute', [(0, '00:01'), (5, '00:01'), (65, '00:02'), (300, '00:05')])
def test_first_eligible_open_delay_and_expiry_equality(delay, entry_minute):
    row = replay(delay_seconds=delay)['rows'][0]
    assert row['position']['entry_time'][11:16] == entry_minute
    assert row['status'] == 'right_censored'
    assert row['net_pnl'] is None


def test_delay_past_expiry_is_known_nonentry():
    row = replay(delay_seconds=301)['rows'][0]
    assert row['status'] == 'not_entered'
    assert row['reason'] == 'entry_expired'
    assert row['net_pnl'] == 0


def test_pending_stop_touch_and_same_fill_parent_down_cancel_before_fill():
    data = bars()
    data.iloc[0, data.columns.get_loc('low')] = 99.
    row = replay(data)['rows'][0]
    assert row['reason'] == 'pending_stop_touch'
    assert row['position'] is None
    down = [{'available_at': '2024-01-01T00:01Z', 'pre_lineage_id': 'parent',
             'post_lineage_id': None, 'source_break_direction': 'down'}]
    row = replay(parent_down=down)['rows'][0]
    assert row['reason'] == 'parent_down'
    assert row['net_pnl'] == 0


def test_entry_open_at_stop_cancels_not_stop_loss():
    data = bars()
    data.iloc[1, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [99, 99.2, 98.8, 99]
    row = replay(data)['rows'][0]
    assert row['reason'] == 'entry_gap_at_or_below_stop'
    assert row['position'] is None


def test_parent_down_does_not_liquidate_open_trade():
    data = bars()
    data.iloc[3, data.columns.get_loc('high')] = 102.1
    down = [{'available_at': '2024-01-01T00:02Z', 'pre_lineage_id': 'parent',
             'source_break_direction': 'down'}]
    row = replay(data, parent_down=down)['rows'][0]
    assert row['reason'] == 'target'
    assert row['net_pnl'] > 0


def test_stop_gap_uses_adverse_open_and_loss_can_exceed_initial_risk():
    data = bars()
    data.iloc[2, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [98, 98.2, 97.8, 98]
    row = replay(data)['rows'][0]
    assert row['reason'] == 'stop_gap'
    assert row['position']['exit_price'] == 98
    assert row['net_pnl'] < -row['position']['initial_risk']


def test_deadline_checks_open_gap_first_and_never_uses_later_extremes():
    data = bars(241)
    data.iloc[240, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [100, 200, 50, 100]
    row = replay(data)['rows'][0]
    assert row['reason'] == 'deadline'
    assert row['position']['exit_price'] == 100
    data.iloc[240, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [98, 200, 50, 100]
    row = replay(data)['rows'][0]
    assert row['reason'] == 'stop_gap'
    assert row['position']['exit_price'] == 98


def test_pending_and_open_occupancy_and_exit_before_new_order():
    data = bars()
    data.iloc[1, data.columns.get_loc('high')] = 102.1
    ops = [opportunity('a'), opportunity('b', '2024-01-01T00:01Z'), opportunity('c', '2024-01-01T00:02Z')]
    sigs = [signal('a'), signal('b', '2024-01-01T00:01Z'), signal('c', '2024-01-01T00:02Z')]
    result = replay(data, ops, sigs)
    assert result['rows'][1]['reason'] == 'busy'
    assert result['rows'][1]['net_pnl'] == 0
    assert result['rows'][2]['position']['entry_time'][11:16] == '00:03'
    pending = replay(ops=[opportunity('b'), opportunity('a')], signals=[signal('b'), signal('a')], delay_seconds=65)
    assert next(r for r in pending['rows'] if r['opportunity_id'] == 'b')['reason'] == 'busy'
    diagnostic = replay(data, ops, sigs, occupied=False)
    assert diagnostic['portfolio'] is False
    assert diagnostic['rows'][1]['position'] is not None


def test_unknown_data_is_not_zero_or_a_search_for_later_fill():
    data = bars().drop(bars().index[1])
    result = replay(data)
    assert result['rows'][0]['status'] == 'unknown'
    assert result['rows'][0]['net_pnl'] is None
    assert result['blockers']
    assert result['rows'][0]['position'] is None


def test_pending_open_gap_releases_capacity_before_ready_and_new_orders():
    data = bars()
    data.iloc[1, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [98., 100.2, 97.8, 100.]
    ops = [opportunity('a'), opportunity('b', '2024-01-01T00:01Z')]
    sigs = [signal('a'), signal('b', '2024-01-01T00:01Z', stop=97.)]
    result = replay(data, ops, sigs, delay_seconds=65)
    assert result['rows'][0]['reason'] == 'entry_gap_at_or_below_stop'
    assert result['rows'][1]['position']['entry_time'][11:16] == '00:03'


def test_new_pending_order_checks_known_same_clock_open_stop_touch():
    data = bars()
    data.iloc[0, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [98., 100.2, 97.8, 100.]
    result = replay(data, delay_seconds=65)
    cancellation = next(e for e in result['events'] if e['kind'] == 'cancel')
    assert cancellation['available_at'][11:16] == '00:00'
    assert result['rows'][0]['position'] is None


def test_unknown_source_without_signal_contaminates_later_occupied_admissions():
    ops = [opportunity('a', source_status='unknown'), opportunity('b', '2024-01-01T00:01Z')]
    sigs = [signal('b', '2024-01-01T00:01Z')]
    result = replay(ops=ops, signals=sigs)
    assert result['rows'][1]['status'] == 'unknown'
    assert result['rows'][1]['net_pnl'] is None
    assert result['blockers']
    diagnostic = replay(ops=ops, signals=sigs, occupied=False)
    assert diagnostic['rows'][1]['position'] is not None


def test_known_no_signal_zero_but_unqualified_source_unknown():
    assert replay(signals=[])['rows'][0]['net_pnl'] == 0
    row = replay(ops=[opportunity(source_status='unknown')], signals=[])['rows'][0]
    assert row['net_pnl'] is None
    assert row['status'] == 'unknown'


def test_settlement_at_entry_excluded_and_at_exit_included():
    data = bars(8, '2024-01-01T07:57Z')
    data.loc[pd.Timestamp('2024-01-01T08:00Z'), ['open', 'high', 'low', 'close']] = [102., 102.2, 101.8, 102.]
    early = replay(data, [opportunity(decision='2024-01-01T07:58Z')],
                   [signal(decision='2024-01-01T07:58Z')], funding_mode='adverse_stress')['rows'][0]
    p = early['position']
    assert p['funding'] == pytest.approx(.0008 * p['quantity'] * p['entry_price'])
    assert p['exit_time'][11:16] == '08:00'
    late = replay(bars(8, '2024-01-01T07:57Z'), [opportunity(decision='2024-01-01T07:59Z')],
                  [signal(decision='2024-01-01T07:59Z')], funding_mode='adverse_stress')['rows'][0]
    assert late['position']['entry_time'][11:16] == '08:00'
    assert late['position']['funding'] == 0


def test_multiple_settlements_do_not_change_risk_or_target():
    data = bars(970, '2023-12-31T23:57Z')
    at = pd.Timestamp('2024-01-01T16:00Z')
    data.loc[at, ['open', 'high', 'low', 'close']] = [102., 102.2, 101.8, 102.]
    op, sig = opportunity(decision='2023-12-31T23:58Z', family='R1'), signal(decision='2023-12-31T23:58Z', family='R1')
    row = replay(data, [op], [sig], funding_mode='adverse_stress')['rows'][0]
    p = row['position']
    assert p['funding'] == pytest.approx(3 * .0008 * p['quantity'] * 100)
    assert p['initial_risk'] == pytest.approx(100)
    assert p['target'] == 102
    assert row['net_pnl'] == pytest.approx(p['quantity'] * 2 - p['entry_fee'] - p['exit_fee'] - p['funding'])


def test_asof_excludes_future_extremes_and_preserves_marks():
    data = bars()
    prefix = replay(data.iloc[:3])
    full_asof = replay(data, as_of='2024-01-01T00:03Z')
    assert prefix['rows'] == full_asof['rows']
    assert prefix['marks'] == full_asof['marks']
    altered = data.copy()
    altered.iloc[3:, altered.columns.get_loc('high')] = 1000
    assert replay(altered, as_of='2024-01-01T00:03Z')['rows'] == full_asof['rows']


def test_unqualified_actual_funding_foreign_and_duplicate_opportunities_rejected():
    with pytest.raises(ValueError, match='funding'):
        replay(funding_mode='actual')
    with pytest.raises(ValueError, match='duplicate'):
        replay(ops=[opportunity(), opportunity()])
    with pytest.raises(ValueError, match='foreign'):
        replay(signals=[signal('absent')])
    with pytest.raises(ValueError, match='family'):
        replay(ops=[opportunity(family='short')])
