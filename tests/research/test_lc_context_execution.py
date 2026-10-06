import importlib

import pandas as pd
import pytest

from scripts.research.lc_context_contract import SUBTYPES, sealed_case
from tests.research.lc_context_fixtures import T, clone, minutes
from tests.research.test_lc_context_controller import changed, contextual


def api():
    return importlib.import_module('scripts.research.lc_context_execution')


def bars(price=104., start=T, periods=1441):
    return minutes(start=start, periods=periods, price=price)


def run(case=None, data=None, arm='immediate', delay=90, funding='zero_diagnostic', **kwargs):
    case = contextual() if case is None else case
    data = bars() if data is None else data
    return api().replay_book([case], {case['candidate_id']: data}, arm=arm,
        subtype=case['subtype'], cost_bps=12, delay_seconds=delay, funding_mode=funding, **kwargs)


@pytest.mark.parametrize('arm,delay,minute', [('immediate', 90, 2), ('immediate', 300, 5),
    ('unconditional_wait', 90, 5), ('unconditional_wait', 300, 11), ('context', 90, 5)])
def test_first_eligible_execution_open(arm, delay, minute):
    row = run(arm=arm, delay=delay)['rows'][0]
    assert row['status'] == 'closed'
    assert row['position']['entry_time'] == (T+pd.Timedelta(minutes=minute)).isoformat()
    assert row['reason'] == 'deadline'
    assert row['position']['exit_time'] == (T+pd.Timedelta('24h')).isoformat()


def test_wait_requires_close_not_high_and_ignores_preavailability_confirmation():
    data = bars(100.)
    data['high'] = 105.
    data.iloc[:2, data.columns.get_loc('close')] = 104.
    row = run(data=data, arm='unconditional_wait')['rows'][0]
    assert row['status'] == 'not_entered'
    assert row['reason'] == 'entry_expired'
    assert row['net_pnl'] == 0.


def test_fill_at_expiry_is_not_permitted():
    data = bars(100.)
    data.iloc[12, data.columns.get_indexer(['high', 'close'])] = [105., 104.]
    row = run(data=data, arm='unconditional_wait')['rows'][0]
    assert row['reason'] == 'entry_expired'
    assert row['position'] is None


@pytest.mark.parametrize('minute', [0, 2, 4])
def test_stop_touch_before_trigger_or_fill_cancels_irreversibly(minute):
    data = bars()
    data.iloc[minute, data.columns.get_loc('low')] = 96.6
    row = run(data=data, arm='context')['rows'][0]
    assert row['reason'] == 'pending_stop_touch'
    assert row['position'] is None


def test_fill_location_change_cancels_only_context_not_controls():
    data = bars()
    data.iloc[5] = [112., 113., 111., 112., 1.]
    assert run(data=data, arm='context')['rows'][0]['reason'] == 'entry_location_changed'
    assert run(data=data, arm='unconditional_wait')['rows'][0]['position']['entry_price'] == 112.


def test_controls_do_not_require_parent_or_h5_for_immediate():
    case = contextual()
    case = changed(case, parent_4h=dict(case['parent_4h'], status='unknown', state='unknown', bound=None),
                   h5=None, h5_status='unknown')
    assert run(case=case)['rows'][0]['status'] == 'closed'
    assert run(case=case, arm='context')['rows'][0]['reason'] == 'unknown_parent_context'
    assert run(case=case, arm='unconditional_wait')['rows'][0]['reason'] == 'unknown_h5'


def test_ambiguous_entry_bar_is_stop_first_with_cost_inclusive_r():
    data = bars(100.)
    data.iloc[2, data.columns.get_indexer(['high', 'low'])] = [120., 90.]
    row = run(data=data)['rows'][0]
    assert row['reason'] == 'stop'
    p = row['position']
    assert p['entry_time'] == (T+pd.Timedelta('2min')).isoformat()
    assert p['exit_time'] == (T+pd.Timedelta('3min')).isoformat()
    assert p['quantity'] == pytest.approx(100/3.52)
    assert p['target'] == pytest.approx(106.8)
    assert row['net_pnl'] == pytest.approx(-100.)
    assert p['net_r'] == pytest.approx(-1.)


def test_stop_gap_and_target_gap_use_declared_conservative_prices():
    data = bars(100.)
    data.iloc[3] = [95., 96., 94., 95., 1.]
    row = run(data=data)['rows'][0]
    assert row['reason'] == 'stop_gap'
    assert row['position']['exit_price'] == 95.
    assert row['net_pnl'] < -100.
    data.iloc[3] = [120., 121., 119., 120., 1.]
    row = run(data=data)['rows'][0]
    assert row['reason'] == 'target_gap'
    assert row['position']['exit_price'] == pytest.approx(106.8)


def test_deadline_open_ignores_future_extrema():
    data = bars(100.)
    data.iloc[-1] = [100., 999., 1., 200., 1.]
    row = run(data=data)['rows'][0]
    assert row['reason'] == 'deadline'
    assert row['position']['exit_price'] == 100.


def test_funding_counts_settlements_through_modeled_exit_and_fee_cap():
    case = changed(contextual(), stop=99.999)
    data = bars(100.)
    data['high'], data['low'] = 100.0001, 100.
    row = run(case=case, data=data, funding='adverse_stress')['rows'][0]
    p = row['position']
    assert p['quantity'] == 500.
    assert p['funding'] == pytest.approx(120.)  #16:00,00:00,08:00
    assert p['entry_fee'] == p['exit_fee'] == 30.
    assert row['net_pnl'] == pytest.approx(-180.)
    # A completed barrier at16:00 pays that coincident settlement first.
    case = contextual()
    data = bars(100.)
    data.loc[T+pd.Timedelta('239min'), 'low'] = 90.
    row = run(case=case, data=data, funding='adverse_stress')['rows'][0]
    assert row['position']['exit_time'] == '2024-01-03T16:00:00+00:00'
    assert row['position']['funding'] == pytest.approx((100/3.52)*100*.0008)


def next_case(case, hours, cid):
    time = T+pd.Timedelta(hours=hours)
    return changed(case, candidate_id=cid, decision_time=time.isoformat(),
                   setup_open=(time-pd.Timedelta('1h')).isoformat())


def test_busy_skip_does_not_replay_later_and_restart_matches():
    first = contextual()
    second = next_case(first, 1, 'later')
    third = next_case(first, 24, 'after_release')
    cases = [first, second, third]
    windows = {c['candidate_id']: bars(100., start=c['decision_time']) for c in cases}
    opts = dict(arm='immediate', subtype=SUBTYPES[0], cost_bps=12, delay_seconds=90,
                funding_mode='zero_diagnostic')
    whole = api().replay_book(cases, windows, **opts)
    assert whole['rows'][1]['reason'] == 'busy'
    assert whole['rows'][1]['position'] is None
    assert whole['rows'][2]['status'] == 'closed'
    part1 = api().replay_book(cases[:1], windows, **opts)
    part2 = api().replay_book(cases[1:], windows, checkpoint=part1['checkpoint'], **opts)
    assert part1['rows']+part2['rows'] == whole['rows']
    assert part1['marks']+part2['marks'] == whole['marks']


def test_execution_gap_poisoning_and_known_abstention_not_unknown():
    first = contextual()
    second = next_case(first, 1, 'later')
    windows = {first['candidate_id']: bars().drop(T+pd.Timedelta('3min')),
               second['candidate_id']: bars(start=second['decision_time'])}
    book = api().replay_book([first, second], windows, arm='immediate', subtype=SUBTYPES[0],
               cost_bps=12, delay_seconds=90, funding_mode='zero_diagnostic')
    assert book['rows'][0]['net_pnl'] is None
    assert book['rows'][1]['reason'] == 'occupied_book_path_unknown'
    assert book['rows'][1]['net_pnl'] is None
    unavailable = changed(first, risk_status='unknown', stop=None)
    row = run(case=unavailable)['rows'][0]
    assert row['net_pnl'] == 0.
    assert row['position'] is None


def test_later_gap_does_not_retroactively_hide_known_busy_skip():
    first = contextual()
    second = next_case(first, 1, 'before_gap')
    third = next_case(first, 3, 'after_gap')
    windows = {c['candidate_id']: bars(100., start=c['decision_time']) for c in [first, second, third]}
    windows[first['candidate_id']] = windows[first['candidate_id']].drop(T+pd.Timedelta('2h'))
    book = api().replay_book([first, second, third], windows, arm='immediate', subtype=SUBTYPES[0],
               cost_bps=12, delay_seconds=90, funding_mode='zero_diagnostic')
    assert book['rows'][1]['reason'] == 'busy'
    assert book['rows'][1]['net_pnl'] == 0.
    assert book['rows'][2]['reason'] == 'occupied_book_path_unknown'


def test_watchers_do_not_reserve_and_books_are_independent():
    first = contextual(close=112.)
    second = next_case(contextual(), 1, 'later')
    windows = {c['candidate_id']: bars(start=c['decision_time']) for c in [first, second]}
    opts = dict(subtype=SUBTYPES[0], cost_bps=12, delay_seconds=90, funding_mode='zero_diagnostic')
    book = api().replay_book([first, second], windows, arm='context', **opts)
    assert book['rows'][0]['reason'] == 'no_new_4h_close_before_expiry'
    assert book['rows'][1]['status'] == 'closed'
    control = api().replay_book([first, second], windows, arm='immediate', **opts)
    assert control['rows'][1]['reason'] == 'busy'


def test_future_append_cannot_change_finished_row_or_marks():
    data = bars(100.)
    original = run(data=data)
    future = bars(1000., start=T+pd.Timedelta('24h1min'), periods=100)
    assert run(data=pd.concat([data, future])) == original


def test_missing_deadline_open_is_unknown_not_synthetic_exit():
    row = run(data=bars(100., periods=1440))['rows'][0]
    assert row['status'] == 'unknown'
    assert row['net_pnl'] is None


def test_subtype_isolation_rejects_unknown_and_cross_bound_checkpoints():
    case = contextual()
    with pytest.raises(ValueError):
        api().replay_book([case], {}, arm='immediate', subtype='unresolved', cost_bps=12,
                         delay_seconds=90, funding_mode='zero_diagnostic')
    checkpoint = run()['checkpoint']
    with pytest.raises(ValueError, match='checkpoint'):
        run(arm='context', checkpoint=checkpoint)
