import importlib

import pandas as pd
import pytest


def api():
    name = 'scripts.research.conditional_entry'
    assert importlib.util.find_spec(name), 'conditional entry resolver is missing'
    return importlib.import_module(name)


def bars():
    return pd.DataFrame(
        {'open': [100., 101., 102., 103., 104., 105.],
         'high': [102., 103., 104., 105., 106., 107.],
         'low': [99., 100., 101., 102., 103., 104.],
         'close': [101., 102., 103., 104., 105., 106.]},
        index=pd.date_range('2026-01-01', periods=6, freq='min', tz='UTC'))


def plan(**changes):
    out = dict(decision_time='2026-01-01T00:00Z', as_of='2026-01-01T00:05Z',
               action='wait_close_above', stop=98., level=101.,
               entry_expiry='2026-01-01T00:04Z', exit_deadline='2026-01-01T00:05Z')
    out.update(changes)
    return out


def test_first_strict_close_enters_next_open_not_confirmation_close():
    r = api().resolve_entry(bars(), **plan())
    assert (r['status'], r['entry_time'], r['entry_price']) == (
        'entry_ready', '2026-01-01T00:02:00+00:00', 102.)


def test_processing_delay_does_not_use_prearm_confirmation():
    r = api().resolve_entry(bars(), **plan(processing_seconds=90))
    assert r['entry_time'] == '2026-01-01T00:03:00+00:00'


@pytest.mark.parametrize('row', [0, 1, 2])
def test_pending_stop_cancels_before_confirmation_or_during_routing(row):
    b = bars(); b.iloc[row, b.columns.get_loc('low')] = 97.
    r = api().resolve_entry(b, **plan(routing_seconds=60))
    assert r['status'] == 'cancelled'
    assert r['resolved_at'] == b.index[row+1].isoformat()


def test_adverse_entry_open_cancels_without_reading_future_low():
    b = bars(); b.loc[b.index[2], ['open', 'high', 'low', 'close']] = [97., float('nan'), 1., 50.]
    assert api().resolve_entry(b, **plan())['status'] == 'cancelled'
    b.loc[b.index[2], 'open'] = 102.
    assert api().resolve_entry(b, **plan())['status'] == 'entry_ready'


@pytest.mark.parametrize('changes', [dict(entry_expiry='2026-01-01T00:02Z'), dict(routing_seconds=120)])
def test_entry_at_expiry_never_fills(changes):
    assert api().resolve_entry(bars(), **plan(**changes))['status'] == 'expired'


def test_asof_and_missing_data_are_not_early_expiry():
    assert api().resolve_entry(bars().iloc[:1], **plan(as_of='2026-01-01T00:01Z'))['status'] == 'pending'
    assert api().resolve_entry(bars().iloc[:1], **plan())['status'] == 'data_unavailable'


def test_resolved_entry_unchanged_by_future_corruption_and_input_not_mutated():
    b = bars(); original = b.copy(deep=True)
    first = api().resolve_entry(b.iloc[:3], **plan(as_of='2026-01-01T00:02Z'))
    b.iloc[3:] = float('nan')
    assert api().resolve_entry(b, **plan()) == first
    pd.testing.assert_frame_equal(b.iloc[:3], original.iloc[:3])


def test_immediate_entry_has_delay_and_no_current_candle_lookahead():
    b = bars(); b.iloc[2, b.columns.get_loc('low')] = 1.
    r = api().resolve_entry(b, **plan(action='enter', level=None, processing_seconds=90))
    assert (r['entry_time'], r['entry_price']) == ('2026-01-01T00:02:00+00:00', 102.)


def test_reject_does_not_read_market_data():
    assert api().resolve_entry(pd.DataFrame(), **plan(action='reject', level=None))['status'] == 'rejected'


def test_invalid_time_plan_distinct_from_rejection():
    assert api().resolve_entry(bars(), **plan(entry_expiry='2026-01-01T00:00Z'))['status'] == 'invalid_plan'


@pytest.mark.parametrize('changes', [dict(stop=True), dict(level=float('nan')), dict(processing_seconds=-1),
                                     dict(routing_seconds=True), dict(decision_time='2026-01-01'),
                                     dict(action='buy'), dict(as_of='2026-01-01T00:00:01Z')])
def test_invalid_inputs_raise(changes):
    with pytest.raises(ValueError):
        api().resolve_entry(bars(), **plan(**changes))


def test_malformed_consumed_bar_is_unavailable():
    b = bars(); b.iloc[0, b.columns.get_loc('high')] = 1.
    assert api().resolve_entry(b, **plan())['status'] == 'data_unavailable'


def test_wait_preserves_original_deadline_in_scoring():
    r = api().score_conditional(bars(), **plan())
    assert r['outcome']['entry_price'] == 102.
    assert r['outcome']['exit_time'] == '2026-01-01T00:05:00+00:00'
    assert r['outcome']['net_pnl'] == pytest.approx(3.*50000./102.-60.)


def test_immediate_scoring_matches_existing_reference():
    from scripts.research.entry_case_outcome import score_case
    r = api().score_conditional(bars(), **plan(action='enter', level=None))
    assert r['outcome'] == score_case(bars(), decision_time='2026-01-01T00:00Z',
                                    step_minutes=1, horizon_bars=5, stop=98.)


def test_missing_outcome_tail_is_not_zero_pnl():
    r = api().score_conditional(bars().iloc[:3], **plan())
    assert r['resolution']['status'] == 'entry_ready'
    assert r['outcome']['status'] == 'data_unavailable'
    assert r['outcome']['net_pnl'] is None


def test_future_duplicate_cannot_change_resolved_entry():
    b = bars()
    expected = api().resolve_entry(b, **plan(action='enter', level=None, as_of='2026-01-01T00:00Z'))
    duplicate = pd.concat([b, b.iloc[5:6]])
    assert api().resolve_entry(duplicate, **plan(action='enter', level=None, as_of='2026-01-01T00:00Z')) == expected


def test_consumed_duplicate_is_unavailable():
    b = pd.concat([bars(), bars().iloc[:1]])
    assert api().resolve_entry(b, **plan())['status'] == 'data_unavailable'


def test_scorer_ignores_duplicates_beyond_outcome_horizon():
    b = bars()
    extra = b.iloc[-1:].copy(); extra.index = extra.index+pd.Timedelta('10min')
    expected = api().score_conditional(b, **plan())
    assert api().score_conditional(pd.concat([b, extra, extra]), **plan()) == expected


def test_timezone_normalization_preserves_resolution():
    p = plan(decision_time='2025-12-31T19:00-05:00', as_of='2025-12-31T19:05-05:00')
    b = bars(); b.index = b.index.tz_convert('America/New_York')
    assert api().resolve_entry(b, **p) == api().resolve_entry(bars(), **plan())


def test_prearm_stop_cancels_even_when_confirmation_not_yet_allowed():
    b = bars(); b.iloc[0, b.columns.get_loc('low')] = 97.
    assert api().resolve_entry(b, **plan(processing_seconds=90))['status'] == 'cancelled'


def test_expiry_equals_deadline_disallows_that_open():
    r = api().resolve_entry(bars(), **plan(level=104., entry_expiry='2026-01-01T00:05Z'))
    assert r['status'] == 'expired'
    assert r['resolved_at'] == '2026-01-01T00:05:00+00:00'
