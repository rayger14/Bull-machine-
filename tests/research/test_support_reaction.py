from copy import deepcopy
import importlib
import importlib.util

import pandas as pd
import pytest

from scripts.research.thesis_contract import event, signed
from tests.research.support_reaction_fixtures import T0, fixture


def api():
    assert importlib.util.find_spec('scripts.research.support_reaction'), 'support-reaction implementation missing'
    return importlib.import_module('scripts.research.support_reaction')


def assess(bars=None, raw=None, at='2024-01-02T06:10Z'):
    original, source = fixture()
    return api().assess(api().origin_record(original if raw is None else raw),
                        source if bars is None else bars, as_of=at)


def test_local_recovery_is_not_breakout_or_accumulation_and_child_is_causal():
    r = assess()
    assert r['phase']['state'] == 'unclassified'
    assert r['structure']['range_kind'] == 'pivot_range'
    assert r['structure']['parent_breakouts'] == []
    assert r['decisions']['B']['status'] == r['decisions']['C']['status'] == 'intent'
    assert r['decisions']['B']['at'] == '2024-01-02T06:09:00+00:00'
    assert r['decisions']['B']['expires_at'] == '2024-01-02T07:00:00+00:00'
    assert r['structure']['child_high']['available_at'] == '2024-01-02T06:05:00+00:00'
    assert r['structure']['child_low']['available_at'] == '2024-01-02T06:08:00+00:00'
    assert r['demand']['state'] == r['supply']['state'] == 'supportive'
    assert r['demand']['relative_volume'] == 2.
    assert r['supply']['relative_volume'] == .5
    assert len(r['demand']['baseline_ids']) == 20
    api().verify_record(r)


def test_future_prices_volume_and_prefix_do_not_change_past_decision():
    raw, bars = fixture()
    a = assess(bars, raw)
    bars.loc['2024-01-02T06:10Z':, ['high', 'close', 'volume']] = [1000., 999., float('nan')]
    b = assess(bars, raw)
    c = assess(bars.loc[:'2024-01-02T06:09Z'], raw)
    assert a == b == c
    for d in a['decisions'].values():
        assert all(pd.Timestamp(a['catalog'][i]['available_at']) <= pd.Timestamp(d['at'])
                   for i in d['citations'] if i in a['catalog'])


def test_missing_volume_blocks_only_c_without_changing_b_or_prices():
    raw, bars = fixture(missing_volume=True)
    r = assess(bars, raw)
    assert r['decisions']['B'] == assess()['decisions']['B']
    assert r['decisions']['C']['status'] == 'unknown'
    assert r['decisions']['C']['reason'] == 'unknown_volume_evidence'


def test_missing_minute_blocks_price_path_instead_of_skipping_gap():
    raw, bars = fixture()
    r = assess(bars.drop(pd.Timestamp('2024-01-02T06:03Z')), raw)
    assert r['decisions']['B']['status'] == 'unknown'
    assert r['decisions']['B']['at'] == '2024-01-02T06:04:00+00:00'
    assert any(r['catalog'][cid].get('status') == 'unknown' for cid in r['decisions']['B']['citations'])


def test_right_side_confirmation_and_same_bar_break_are_not_backdated():
    assert assess(at='2024-01-02T06:04Z')['structure']['child_high'] is None
    assert assess(at='2024-01-02T06:07Z')['structure']['child_low'] is None
    raw, bars = fixture()
    bars.loc['2024-01-02T06:07Z', ['high', 'close']] = [111., 110.5]
    bars.loc['2024-01-02T06:08Z', ['high', 'close']] = [111., 109.8]
    r = assess(bars, raw, at='2024-01-02T06:09Z')
    assert r['decisions']['B']['status'] == 'pending'


def test_equal_pivots_cannot_supply_the_locked_child_high():
    raw, bars = fixture()
    bars.loc['2024-01-02T06:03Z', 'high'] = 110.
    r = assess(bars, raw)
    assert r['structure']['child_high'] is None
    assert r['decisions']['B']['status'] == 'pending'


def test_support_failure_cancels_child_not_parent_range():
    raw, bars = fixture()
    bars.loc['2024-01-02T06:03Z', 'low'] = 107.
    r = assess(bars, raw)
    assert r['decisions']['B']['reason'] == 'child_support_failed'
    assert r['parent_invalid_at'] is None


def test_room_rejection_is_known_and_not_a_breakout_entry():
    raw, bars = fixture(ceiling=130.)
    r = assess(bars, raw)
    assert r['decisions']['B']['status'] == 'rejected'
    assert r['decisions']['B']['reason'] == 'insufficient_parent_room'
    assert r['decisions']['B']['room_r'] == pytest.approx(19.8/13.2)


def candle(at, values):
    t = pd.Timestamp(at)
    return event('candle', '1h', t, t+pd.Timedelta('1h'), values, stream_id='fixture')


def baseline():
    return [candle(t, dict(open=100., high=108., low=100., close=104., volume=100.))
            for t in pd.date_range('2024-01-01T00:00Z', periods=20, freq='h')]


@pytest.mark.parametrize('role,values,want', [
    ('demand', dict(open=101., high=110., low=100., close=109., volume=200.), 'supportive'),
    ('demand', dict(open=101., high=110., low=100., close=102., volume=200.), 'adverse'),
    ('demand', dict(open=101., high=110., low=100., close=106., volume=100.), 'neutral'),
    ('supply', dict(open=104., high=106., low=102., close=105., volume=50.), 'supportive'),
    ('supply', dict(open=109., high=112., low=100., close=102., volume=200.), 'adverse'),
    ('supply', dict(open=104., high=108., low=100., close=105., volume=100.), 'neutral'),
])
def test_literal_contextual_evidence(role, values, want):
    out = api().evidence(candle('2024-01-01T20:00Z', values), baseline(), role)
    assert out['state'] == want


@pytest.mark.parametrize('a,b,want', [('supportive', 'neutral', 'allow'),
    ('neutral', 'supportive', 'allow'), ('neutral', 'neutral', 'reject'),
    ('supportive', 'adverse', 'reject'), ('unknown', 'supportive', 'unknown')])
def test_distinct_evidence_roles_not_unanimous_positive_votes(a, b, want):
    assert api().evidence_action({'state': a}, {'state': b}) == want


def test_gap_in_twenty_hour_baseline_cannot_be_replaced_with_older_hour():
    rows = baseline()
    rows[5] = candle('2023-12-31T23:00Z', dict(open=100., high=108., low=100., close=104., volume=100.))
    c = candle('2024-01-01T20:00Z', dict(open=101., high=110., low=100., close=109., volume=200.))
    assert api().evidence(c, rows, 'demand')['state'] == 'unknown'


@pytest.mark.parametrize('change', ['future_parent', 'bad_stop', 'authority', 'stream', 'parent_identity', 'origin_identity'])
def test_corrupted_origin_never_becomes_a_good_trade(change):
    raw, bars = fixture()
    origin = api().origin_record(raw)
    if change == 'future_parent': origin['parent']['available_at'] = origin['origin']['start']
    if change == 'bad_stop': origin['original_stop'] = 1000.
    if change == 'authority': origin['execution_authorized'] = True
    if change == 'stream': origin['stream_id'] = 'foreign'
    if change == 'parent_identity': origin['parent']['id'] = 'foreign'
    if change == 'origin_identity': origin['id'] = 'episode:foreign'
    with pytest.raises(ValueError):
        api().assess(signed(origin), bars, as_of='2024-01-02T06:10Z')


def test_record_tamper_and_future_annotation_are_not_accepted():
    r = assess(); altered = deepcopy(r)
    altered['phase']['state'] = 'accumulation'
    with pytest.raises(ValueError): api().verify_record(altered)


def test_earlier_failed_touch_does_not_cancel_later_qualifying_support():
    raw, bars = fixture()
    child = bars.loc['2024-01-02T06:00Z':'2024-01-02T06:08Z'].copy()
    bars.loc['2024-01-02T05:00Z':'2024-01-02T05:59Z', ['open', 'high', 'low', 'close']] = [110., 111., 106., 107.]
    bars.loc['2024-01-02T06:00Z':'2024-01-02T06:59Z', ['open', 'high', 'low', 'close']] = [111., 111.5, 107., 109.5]
    child.index += pd.Timedelta('1h')
    bars.loc[child.index] = child
    r = assess(bars, raw, at='2024-01-02T07:10Z')
    assert r['decisions']['B']['status'] == 'intent'
    assert r['decisions']['B']['at'] == '2024-01-02T07:09:00+00:00'
    assert r['catalog'][r['structure']['support']]['available_at'] == '2024-01-02T07:00:00+00:00'


def test_recovery_and_support_hour_boundaries_are_inclusive():
    raw, bars = fixture()
    bars.loc[T0:, ['open', 'high', 'low', 'close']] = [104., 108., 100., 104.]
    origin = api().origin_record(raw)
    before = api().assess(origin, bars, as_of=T0+pd.Timedelta('47h'))
    expired = api().assess(origin, bars, as_of=T0+pd.Timedelta('48h'))
    assert before['decisions']['B']['status'] == 'pending'
    assert expired['decisions']['B']['reason'] == 'recovery_expired'
    bars.loc[T0+pd.Timedelta('47h'):T0+pd.Timedelta('48h')-pd.Timedelta('1min'),
             ['open', 'high', 'low', 'close']] = [107., 112., 106., 111.]
    bars.loc[T0+pd.Timedelta('48h'):, ['open', 'high', 'low', 'close']] = [110., 112., 109., 110.]
    r = api().assess(origin, bars, as_of=T0+pd.Timedelta('72h'))
    assert r['structure']['recovery'] is not None
    assert r['decisions']['B']['reason'] == 'support_expired'
    bars.loc[T0+pd.Timedelta('71h'):T0+pd.Timedelta('72h')-pd.Timedelta('1min'),
             ['open', 'high', 'low', 'close']] = [110., 112., 107., 110.]
    r = api().assess(origin, bars, as_of=T0+pd.Timedelta('72h'))
    assert r['structure']['support'] is not None
    assert r['decisions']['B']['reason'] == 'await_child'


@pytest.mark.parametrize('minute,want', [('06:58', 'intent'), ('06:59', 'expired')])
def test_minute_trigger_expiry_is_exclusive(minute, want):
    raw, bars = fixture()
    bars.loc['2024-01-02T06:08Z':, ['open', 'high', 'low', 'close']] = [109.5, 110., 109., 109.5]
    bars.loc['2024-01-02T'+minute+'Z', ['high', 'close']] = [111., 111.]
    r = assess(bars, raw, at='2024-01-02T07:00Z')
    assert r['decisions']['B']['status'] == want


def test_parent_invalidation_precedes_child_failure_at_same_clock():
    raw, bars = fixture()
    bars.loc['2024-01-02T05:00Z':'2024-01-02T05:59Z', ['open', 'high', 'low', 'close']] = [110., 112., 109., 110.]
    bars.loc['2024-01-02T06:00Z':'2024-01-02T06:59Z', ['open', 'high', 'low', 'close']] = [110., 112., 107., 110.]
    bars.loc['2024-01-02T07:59Z', ['low', 'close']] = [98., 99.]
    r = assess(bars, raw, at='2024-01-02T08:00Z')
    assert r['decisions']['B']['reason'] == 'parent_invalidated'
    assert r['parent_invalid_at'] == '2024-01-02T08:00:00+00:00'


def test_parent_breakout_is_observation_not_qualified_accumulation():
    raw, bars = fixture()
    bars.loc[T0:'2024-01-02T08:59Z', ['open', 'high', 'low', 'close']] = [151., 156., 150., 155.]
    r = assess(bars, raw, at='2024-01-02T09:00Z')
    assert r['structure']['parent_breakouts'][0]['state'] == 'close_above_range_not_acceptance'
    assert r['phase']['state'] == 'unclassified'


def test_original_spring_failure_cancels_this_setup_not_the_parent_range():
    raw, bars = fixture()
    bars.loc['2024-01-02T04:15Z', 'low'] = 96.
    r = assess(bars, raw)
    assert r['decisions']['B']['reason'] == 'original_spring_stop_failed'
    assert r['decisions']['B']['at'] == '2024-01-02T05:00:00+00:00'
    assert r['parent_invalid_at'] is None
    assert r['setup_invalid_at'] == '2024-01-02T05:00:00+00:00'
    # The source policy inspects complete hourly bars until support, not a
    # fictional active stop order before it has proposed an entry.
    assert assess(bars, raw, at='2024-01-02T04:59Z')['decisions']['B']['status'] == 'pending'


def test_first_child_high_is_locked_not_replaced_by_easier_later_pivot():
    raw, bars = fixture()
    bars.loc['2024-01-02T06:08Z':, ['open', 'high', 'low', 'close']] = [109.2, 109.3, 109., 109.2]
    bars.loc['2024-01-02T06:09Z', 'high'] = 109.8
    bars.loc['2024-01-02T06:12Z', ['high', 'close']] = [109.95, 109.9]
    r = assess(bars, raw, at='2024-01-02T06:13Z')
    assert r['structure']['child_high']['price'] == 110.
    assert r['decisions']['B']['status'] == 'pending'


def test_exact_two_r_room_is_inclusive_despite_binary_float_roundoff():
    raw, bars = fixture(ceiling=136.6)
    r = assess(bars, raw)
    assert r['decisions']['B']['status'] == 'intent'
    assert r['decisions']['B']['room_r'] == pytest.approx(2.)
