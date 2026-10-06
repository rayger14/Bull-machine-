"""Causality, matched selection and null/cost accounting on hand-built data."""
from copy import deepcopy
import importlib

import pandas as pd
import pytest


def api():
    name = 'scripts.research.lc_upside_diagnostic'
    assert importlib.util.find_spec(name), 'upside diagnostic API missing'
    return importlib.import_module(name)


def hourly():
    close = pd.Series(range(100, 180), dtype=float).to_numpy()
    return pd.DataFrame(dict(open=close-.2, high=close+.4, low=close-1.1,
                             close=close, volume=1.),
                        index=pd.date_range('2024-01-01T00:00Z', periods=80, freq='h'))


def test_features_use_completed_candle_and_pre_setup_volatility_and_trend():
    row = api().hourly_features(hourly(), '2024-01-02T00:00Z', '2024-01-03T00:00Z').loc[
        '2024-01-02T12:00Z']
    assert row['close'] == 135.
    assert row['atr'] == pytest.approx(1.5)
    assert row['previous_atr_pct'] == pytest.approx(1.5/134.)
    assert row['prior_return_24h'] == pytest.approx(134./110.-1)
    assert bool(row['upside']) is True


def test_future_prices_cannot_change_prior_matching_features():
    frame = hourly()
    original = api().hourly_features(frame, '2024-01-02T00:00Z', '2024-01-03T00:00Z')
    frame.loc['2024-01-02T12:00Z':, ['open', 'high', 'low', 'close']] *= 10
    changed = api().hourly_features(frame, '2024-01-02T00:00Z', '2024-01-03T00:00Z')
    pd.testing.assert_frame_equal(original.loc[:'2024-01-02T12:00Z'],
                                  changed.loc[:'2024-01-02T12:00Z'])


def test_setup_volatility_does_not_leak_into_matching_covariate():
    frame = hourly()
    frame.loc['2024-01-02T11:00Z', 'high'] = 1000.
    row = api().hourly_features(frame, '2024-01-02T00:00Z', '2024-01-03T00:00Z').loc[
        '2024-01-02T12:00Z']
    assert row['previous_atr_pct'] == pytest.approx(1.5/134.)
    assert row['atr'] > 50


def test_incomplete_hour_grid_is_not_silently_accepted():
    with pytest.raises(ValueError, match='complete'):
        api().hourly_features(hourly().drop(hourly().index[20]),
                              '2024-01-02T00:00Z', '2024-01-03T00:00Z')


def candidates():
    return [dict(candidate_id='lc1', decision_time='2024-03-10T12:00:00+00:00'),
            dict(candidate_id='lc2', decision_time='2024-03-12T12:00:00+00:00')]


def matching_frame():
    return pd.DataFrame(dict(close=100., atr=1., previous_atr_pct=.01,
                             prior_return_24h=.1, upside=True),
                        index=pd.to_datetime(['2024-03-07T12:00Z', '2024-03-08T12:00Z',
                                              '2024-03-09T12:00Z', '2024-03-10T12:00Z',
                                              '2024-03-12T12:00Z']))


def test_match_is_past_only_native_excluded_and_without_replacement():
    cases = candidates()
    native = [c['decision_time'] for c in cases] + ['2024-03-09T12:00Z']
    result = api().match_controls(cases, matching_frame(), native)
    assert [r['control_time'] for r in result] == ['2024-03-08T12:00:00+00:00',
                                                '2024-03-07T12:00:00+00:00']
    assert all(r['status'] == 'matched' for r in result)


@pytest.mark.parametrize('column,value', [('upside', False), ('prior_return_24h', -.1),
                                         ('previous_atr_pct', .02),
                                         ('previous_atr_pct', float('nan'))])
def test_ineligible_covariates_leave_an_unmatched_row(column, value):
    frame = matching_frame().loc[['2024-03-09T12:00Z', '2024-03-10T12:00Z']].copy()
    frame.loc[frame.index[0], column] = value
    rows = api().match_controls(candidates()[:1], frame, [candidates()[0]['decision_time']])
    assert len(rows) == 1
    assert rows[0]['status'] == 'unmatched'
    assert rows[0]['control_time'] is None


@pytest.mark.parametrize('control', ['2024-03-10T11:00Z', '2024-03-11T12:00Z',
                                     '2023-12-01T12:00Z'])
def test_wrong_hour_future_or_too_old_control_cannot_match(control):
    frame = matching_frame().iloc[:2].copy()
    frame.index = pd.to_datetime([control, '2024-03-10T12:00Z'])
    frame = frame.sort_index()
    assert api().match_controls(candidates()[:1], frame,
                               [candidates()[0]['decision_time']])[0]['status'] == 'unmatched'


def test_missing_case_covariates_stay_visible_and_duplicate_cases_fail():
    assert api().match_controls(candidates(), matching_frame().iloc[:3], [])[0][
        'reason'] == 'case_covariates_unavailable'
    with pytest.raises(ValueError, match='unique'):
        api().match_controls(candidates()[:1]*2, matching_frame(), [])


def facts():
    parent = dict(evidence_status='known', lifecycle='intact',
                  bound=dict(range_low=90., range_high=110.))
    return dict(parent_4h=deepcopy(parent), parent_1d=deepcopy(parent),
                last_two_5m=dict(status='known', higher_low=True, higher_close=True))


def test_room_is_nearest_mapped_boundary_not_unobstructed_space():
    f = facts()
    f['parent_4h']['lifecycle'] = 'broken_up'
    f['parent_1d']['bound'] = dict(range_low=105., range_high=130.)
    result = api().context_labels(f, close=100., stop=95.)
    assert result['parent_4h_lifecycle'] == 'broken_up'
    assert result['parent_4h_location'] == 'inside'
    assert result['parent_1d_location'] == 'below'
    assert result['mapped_overhead'] == 'below_2r'
    assert result['last_two_5m'] == 'both'


def test_exact_two_r_is_in_high_room_bucket_and_range_boundaries_are_inside():
    assert api().context_labels(facts(), close=100., stop=95.)['mapped_overhead'] == 'at_least_2r'
    assert api().context_labels(facts(), close=110., stop=105.)['parent_4h_location'] == 'inside'
    assert api().context_labels(facts(), close=111., stop=105.)['mapped_overhead'] == 'no_reference'


def test_absent_unknown_and_no_overhead_are_distinct():
    f = facts()
    f['parent_4h'].update(lifecycle='absent', bound=None)
    f['parent_1d'].update(lifecycle='absent', bound=None)
    labels = api().context_labels(f, close=100., stop=95.)
    assert labels['parent_4h_location'] == 'absent'
    assert labels['mapped_overhead'] == 'no_reference'
    f['parent_4h']['evidence_status'] = 'unknown'
    assert api().context_labels(f, close=100., stop=95.)['mapped_overhead'] == 'unknown'


def plan():
    return dict(decision_time='2024-01-01T00:00:00+00:00', action='enter', level=None,
                stop=95., entry_expiry='2024-01-01T00:15:00+00:00',
                exit_deadline='2024-01-02T00:00:00+00:00', processing_seconds=90,
                routing_seconds=0, notional=50000., cost_bps=12)


def bars():
    frame = pd.DataFrame(dict(open=100., high=101., low=99., close=100.),
                         index=pd.date_range('2024-01-01T00:00Z', periods=1441, freq='min'))
    frame.iloc[3, frame.columns.get_loc('high')] = 110.
    return frame


def test_event_economics_include_fees_and_original_delayed_entry():
    event = api().event_result(plan(), bars())
    assert event['filled'] is True
    assert event['resolved'] is True
    assert event['net_pnl'] == 4940.
    assert event['net_r'] == 1.9296875
    assert event['raw']['resolution']['entry_time'] == '2024-01-01T00:02:00+00:00'


def test_nonentry_zero_is_distinct_from_missing_outcome():
    frame = bars()
    frame.iloc[0, frame.columns.get_loc('low')] = 94.
    cancelled = api().event_result(plan(), frame)
    assert cancelled['net_r'] == 0.
    assert cancelled['filled'] is False
    assert cancelled['resolved'] is True
    unknown = api().event_result(plan(), bars().drop(bars().index[100]))
    assert unknown['net_r'] is None
    assert unknown['resolved'] is False


def test_summary_does_not_hide_unknowns_or_count_nonentries_as_winners():
    rows = [dict(resolved=True, filled=True, net_pnl=200., net_r=2.),
            dict(resolved=True, filled=False, net_pnl=0., net_r=0.),
            dict(resolved=False, filled=True, net_pnl=None, net_r=None)]
    result = api().summarize_events(rows)
    assert result['candidate_count'] == 3
    assert result['unresolved'] == 1
    assert result['known_net_subtotal'] == 200.
    assert result['mean_net_r'] is None
    assert result['wins'] == 1
    assert result['nonentries'] == 1


def test_paired_r_contrast_keeps_unmatched_and_uses_month_clusters():
    pairs = [dict(candidate_id='a', decision_time='2024-01-10T00:00:00+00:00',
                  control_id='x', status='matched'),
             dict(candidate_id='b', decision_time='2024-03-10T00:00:00+00:00',
                  control_id='y', status='matched'),
             dict(candidate_id='c', decision_time='2024-03-15T00:00:00+00:00',
                  control_id=None, status='unmatched')]
    def e(r):
        return dict(resolved=True, filled=True, net_pnl=100*r, net_r=r)
    events = {'a': e(2.), 'x': e(1.), 'b': e(0.), 'y': e(-1.), 'c': e(100.)}
    result = api().paired_summary(pairs, events, '2024-01', '2024-03')
    assert result['matched_count'] == 2
    assert result['unmatched_count'] == 1
    assert result['mean_delta_net_r'] == 1.
    assert result['bootstrap_95_delta'] == [1., 1.]
    assert result['calendar_months'] == 3
    events['y'] = dict(resolved=False, filled=True, net_pnl=None, net_r=None)
    assert api().paired_summary(pairs, events, '2024-01', '2024-03')[
        'mean_delta_net_r'] is None


def runner():
    name = 'scripts.research.run_lc_upside_diagnostic'
    assert importlib.util.find_spec(name), 'two-stage runner missing'
    return importlib.import_module(name)


def test_modified_and_newly_appeared_missing_bindings_abort(tmp_path):
    from scripts.research.lc_campaign import _sha_file
    source = tmp_path/'source.json'
    source.write_bytes(b'original')
    absent = tmp_path/'unavailable.json'
    bindings = {str(source): _sha_file(source), str(absent): None}
    runner().verify_bindings(bindings)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='binding'):
        runner().verify_bindings(bindings)
    source.write_bytes(b'original')
    absent.write_bytes(b'appeared')
    with pytest.raises(ValueError, match='binding'):
        runner().verify_bindings(bindings)


def test_score_refuses_modified_code_before_reading_prices(tmp_path):
    from scripts.research.lc_campaign import _sha_file
    from scripts.research.lc_judgment_runner import _digest, _save_equal
    source = tmp_path/'source.py'
    source.write_bytes(b'original')
    preflight = dict(files={str(source): _sha_file(source)})
    preflight['sha256'] = _digest(preflight)
    _save_equal(tmp_path/'preflight.json', preflight)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='binding'):
        runner().score(tmp_path)


@pytest.mark.parametrize('copy_run', [True, False])
def test_consumed_paths_must_be_bound_not_just_same_named_originals(tmp_path, monkeypatch, copy_run):
    from scripts.research.lc_campaign import _sha_file
    from scripts.research.lc_judgment_runner import _digest, _save_equal
    mod = runner()
    original, copied = tmp_path/'original', tmp_path/'copied'
    files = {}
    for name, content in [('plans', {'control': plan()}), ('matches', []), ('contexts', [])]:
        a = _save_equal(original/(name+'.json'), content)
        _save_equal(copied/(name+'.json'), content)
        files[str(a)] = _sha_file(a)
    archive, baseline = tmp_path/'prices.parquet', tmp_path/'baseline.json'
    archive.write_bytes(b'not to be read as prices')
    baseline.write_bytes(b'{}')
    monkeypatch.setattr(mod, 'ARCHIVE', archive)
    monkeypatch.setattr(mod, 'BASELINE', baseline)
    files[str(baseline)] = _sha_file(baseline)
    if copy_run:
        files[str(archive)] = _sha_file(archive)
    output = copied if copy_run else original
    preflight = dict(files=files)
    preflight['sha256'] = _digest(preflight)
    _save_equal(output/'preflight.json', preflight)
    def forbidden_price_read(*args):
        raise AssertionError('unbound inputs reached future price reading')
    monkeypatch.setattr(mod, '_minutes', forbidden_price_read)
    with pytest.raises(ValueError, match='unbound consumed input'):
        mod.score(output)
