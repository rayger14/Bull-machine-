import importlib

import pandas as pd
import pytest

from tests.research.lc_context_fixtures import T, clone, ledger, minutes, source


@pytest.fixture
def api():
    return importlib.import_module('scripts.research.lc_context_evidence')


def test_reconstructed_geometry_and_bound_source_are_sealed(api):
    case = api.prepare_evidence(*source())
    assert case['subtype'] == 'upside_expansion_candidate'
    assert case['source_status'] == case['risk_status'] == 'known'
    assert case['stop'] == pytest.approx(96.6)
    assert case['h5'] == 103.
    assert case['parent_4h']['state'] == 'inside'
    assert case['parent_4h']['legacy_state'] == 'broken_up'
    assert case['parent_4h']['bound']['id'] == 'range-old'
    assert case['execution_authorized'] is False
    assert len(case['seal']) == 64


def test_missing_atr_does_not_erase_known_subtype(api):
    raw, bars, parents, provenance = source()
    del raw['features']['atr_14']
    case = api.prepare_evidence(raw, bars, parents, provenance)
    assert case['subtype'] == 'upside_expansion_candidate'
    assert case['risk_status'] == 'unknown'
    assert case['stop'] is None


def test_upside_precedence_over_simultaneous_prior_low_sweep(api):
    raw, bars, parents, provenance = source()
    raw['features']['low'] = 98.
    bars.loc[T-pd.Timedelta('30min'), 'low'] = 98.
    assert api.prepare_evidence(raw, bars, parents, provenance)['subtype'] == 'upside_expansion_candidate'


@pytest.mark.parametrize('close,low,want', [(98., 97., 'downside_rebound_candidate'),
                                          (100., 98., 'downside_rebound_candidate'),
                                          (100., 99., 'unresolved')])
def test_rebound_subtype_requires_strict_geometry(api, close, low, want):
    raw, bars, parents, provenance = source()
    raw['features'].update(close=close, low=low)
    bars.loc[T-pd.Timedelta('1h'):, ['close', 'low']] = [close, low]
    assert api.prepare_evidence(raw, bars, parents, provenance)['subtype'] == want


def test_hourly_break_cannot_impersonate_originating_timeframe_acceptance(api):
    bars = minutes(start='2024-01-02T00:00Z', periods=12*60)
    bars.loc['2024-01-02T09:00Z':, ['close', 'high']] = [112., 113.]
    bound = dict(ledger()['versions'][0], available_at='2024-01-02T00:00Z')
    before = api.acceptance(bound, bars, '2024-01-02T11:00Z', '4H', 'fixture')
    after = api.acceptance(bound, bars, '2024-01-02T12:00Z', '4H', 'fixture')
    assert before['state'] == 'inside'
    assert after['state'] == 'accepted_above'
    assert after['events'][-1]['available_at'] == '2024-01-02T12:00:00+00:00'


@pytest.mark.parametrize('close,want', [(110., 'boundary'), (90., 'boundary'),
                                      (89., 'accepted_below'), (111., 'accepted_above')])
def test_acceptance_strict_bounds_and_retained_events(api, close, want):
    bars = minutes(start='2024-01-02T00:00Z', periods=8*60)
    bars.iloc[-1] = [close, max(101, close), min(99, close), close, 1.]
    bound = dict(ledger()['versions'][0], available_at='2024-01-02T00:00Z')
    result = api.acceptance(bound, bars, '2024-01-02T08:00Z', '4H', 'fixture')
    assert result['state'] == want
    assert len(result['events']) == 2
    assert result['events'][0]['state'] == 'inside'
    assert all(e['parent_version_id'] == 'range-old' for e in result['events'])


def test_preavailability_candle_not_promoted_and_gap_not_filled(api):
    bars = minutes(start='2024-01-02T00:00Z', periods=8*60)
    bound = dict(ledger()['versions'][0], available_at='2024-01-02T01:00Z')
    assert api.acceptance(bound, bars, '2024-01-02T04:00Z', '4H', 'fixture')['state'] == 'not_established'
    missing = bars.drop(pd.Timestamp('2024-01-02T05:00Z'))
    assert api.acceptance(bound, missing, '2024-01-02T08:00Z', '4H', 'fixture')['state'] == 'unknown'


def test_parent_selection_keeps_broken_latest_and_strict_pre_setup_clock(api):
    value = ledger()
    newer = dict(value['versions'][0], id='latest-broken-version',
                 formation_hour='2024-01-03T07:00Z', available_at='2024-01-03T08:00Z')
    future = dict(newer, id='future', formation_hour='2024-01-03T10:00Z', available_at='2024-01-03T11:00Z')
    value['versions'] += [newer, future]
    assert api.select_parent(value, T-pd.Timedelta('1h'), 'BTC', 'fixture', '4H')['bound']['id'] == 'latest-broken-version'


def test_parent_absence_is_distinct_from_missing_coverage(api):
    value = ledger()
    value['versions'] = []
    assert api.select_parent(value, T-pd.Timedelta('1h'), 'BTC', 'fixture', '4H')['status'] == 'absent'
    value['coverage']['last_processed_close'] = '2024-01-02T00:00Z'
    assert api.select_parent(value, T-pd.Timedelta('1h'), 'BTC', 'fixture', '4H')['status'] == 'unknown'


@pytest.mark.parametrize('corruption', ['stream', 'future_pivot', 'bounds', 'duplicate'])
def test_malformed_bound_context_is_integrity_error_not_abstention(api, corruption):
    value = ledger()
    if corruption == 'stream': value['manifest']['data_stream_id'] = 'foreign'
    if corruption == 'future_pivot': value['pivots'][0]['available_at'] = T.isoformat()
    if corruption == 'bounds': value['versions'][0]['range_low'] = 120.
    if corruption == 'duplicate': value['versions'].append(clone(value['versions'][0]))
    with pytest.raises(ValueError):
        api.select_parent(value, T-pd.Timedelta('1h'), 'BTC', 'fixture', '4H')


def test_future_append_and_order_cannot_change_case(api):
    raw, bars, parents, provenance = source()
    first = api.prepare_evidence(raw, bars, parents, provenance)
    future = minutes(start=T, periods=120, price=200.)
    parents['4H_N3']['versions'].append(dict(parents['4H_N3']['versions'][0],
        id='future', formation_hour=T.isoformat(), available_at=(T+pd.Timedelta('1h')).isoformat()))
    parents['4H_N3']['versions'].reverse()
    assert api.prepare_evidence(raw, pd.concat([bars, future]), parents, provenance) == first


def test_hourly_mismatch_or_future_features_aborts(api):
    raw, bars, parents, provenance = source()
    raw['features']['close'] += 1.
    with pytest.raises(ValueError, match='reconstruction'):
        api.prepare_evidence(raw, bars, parents, provenance)
    raw, bars, parents, provenance = source()
    raw['feature_available_at'] = (T+pd.Timedelta('1min')).isoformat()
    with pytest.raises(ValueError, match='availability'):
        api.prepare_evidence(raw, bars, parents, provenance)
