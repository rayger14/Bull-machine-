from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import pytest

from engine.archetypes import archetype_instance as native
from scripts.research.engine_signal_replay import EXPECTED_ARCHETYPES, SignalEngine
from scripts.research.replay_clock import digest
from scripts.research.study_hourly import HourlyStudyObserver, directional_permission
from tests.research.test_engine_signal_replay import wick_fixture


def observer_for(engine=None):
    return HourlyStudyObserver(engine or SignalEngine(), instrument='BTC', data_stream_id='fixture')


def features(hour=0, above=1.):
    return dict(wick_fixture(), timestamp=pd.Timestamp('2024-01-01T00:00Z') + pd.Timedelta(hours=hour),
                price_above_ema_50=above, tf4h_fusion_score=.2)


def observed(values):
    return {k: {'status': 'observed'} for k in values}


def update(observer, values, provenance=None):
    return observer.update(values, values['timestamp'] + pd.Timedelta('1h'),
                           observed(values) if provenance is None else provenance)


@pytest.mark.parametrize('value,status,expected', [
    (1., 'observed', True), (1.01, 'observed', True), (0., 'observed', False),
    (.99, 'observed', False), (1., 'defaulted', False), (True, 'observed', False),
    (None, 'missing', False), (float('nan'), 'observed', False), (float('inf'), 'observed', False),
])
def test_permission_requires_observed_finite_numeric_alignment(value, status, expected):
    assert directional_permission(value, status) is expected


def test_observer_preserves_full_native_outputs_and_state_and_calls_identity_once():
    control, wrapped = SignalEngine(), SignalEngine()
    observer = observer_for(wrapped)
    original = wrapped.engine.structural_checker.logic._check_H
    with patch.object(wrapped.engine.structural_checker.logic, '_check_H', wraps=original) as check:
        for hour in range(3):
            row = features(hour)
            expected = control.update(row, row['timestamp'] + pd.Timedelta('1h'))
            result = update(observer, row)
            assert digest(result['native']) == digest(expected)
            assert set(result['gate_receipts']) == EXPECTED_ARCHETYPES
            assert result['opportunity'] is not None  # even on native cooldown
            assert all(r['detect_calls'] == 1 for r in result['native']['archetypes'].values())
        assert check.call_count == 3
    assert digest(control.snapshot()) == digest(wrapped.snapshot())


def test_below_ema_rejects_before_cooldown_and_arms_are_independent():
    observer = observer_for()
    below = update(observer, features(0, above=0.))
    assert below['opportunity']
    assert below['arms']['baseline']['signal']
    assert below['arms']['repair']['reason'] == 'directional_permission_rejected'
    assert below['arms']['repair']['last_signal_bar_after'] is None
    above = update(observer, features(1, above=1.))
    assert above['arms']['baseline']['reason'] == 'cooldown'
    assert above['arms']['repair']['signal']
    # Emission arms cooldown regardless of subsequent book occupancy.
    assert above['arms']['repair']['last_signal_bar_after'] == 2
    again = update(observer, features(2))
    assert again['arms']['repair']['reason'] == 'cooldown'


def test_above_ema_preserves_signal_geometry_and_exposes_every_gate():
    result = update(observer_for(), features())
    baseline, repair = result['arms']['baseline'], result['arms']['repair']
    assert baseline['native_signal'] == repair['native_signal']
    assert baseline['signal']['stop'] == 100 - 4.6 * 2
    gates = result['gate_receipts']['trap_within_trend']
    assert len(gates) == 4
    assert gates[0]['native_status'] == 'passed'
    assert gates[0]['derived_inputs']['open']['value'] == 99.
    assert gates[-1]['native_status'] == 'skipped_nan'
    assert result['blockers'] == []


def test_defaulted_direction_is_not_permission():
    row = features()
    provenance = observed(row)
    provenance['price_above_ema_50'] = {'status': 'defaulted'}
    result = update(observer_for(), row, provenance)
    assert result['arms']['baseline']['signal']
    assert not result['arms']['repair']['signal']


def test_normal_identity_rejection_is_not_a_data_error_or_cooldown():
    observer = observer_for()
    row = features()
    row['adx'] = row['adx_14'] = 5.
    result = update(observer, row)
    assert result['identity'] == {'passed': False, 'reason': 'structural_H_failed'}
    assert result['blockers'] == []
    assert result['opportunity'] is None
    assert result['arms']['baseline']['last_signal_bar_after'] is None


def test_structural_exception_is_error_not_raw_passing_identity():
    observer = observer_for()
    with patch.object(observer.signal_engine.engine.structural_checker.logic, '_check_H', side_effect=ValueError('fixture')):
        result = update(observer, features())
    assert result['native']['archetypes']['trap_within_trend']['structural']['passed'] is True
    assert result['opportunity'] is None
    assert 'R1:structural_error' in result['blockers']
    assert result['arms']['baseline']['signal'] is None


def test_derived_exception_is_recorded_once_and_blocks_even_permissive_native_gate():
    observer = observer_for()
    # Native hard TWT blocks; other modes may skip/penalize. Never hide source error.
    with patch.dict(native.DERIVED_FEATURES, {'wick_anomaly': lambda f: 1 / 0}):
        result = update(observer, features())
    receipt = result['gate_receipts']['trap_within_trend'][0]
    assert receipt['error']['type'] == 'ZeroDivisionError'
    assert 'R1:derived_gate_error' in result['blockers']
    assert receipt['native_status'] == 'failed_compute'


def test_atr_fallback_is_explicit_and_not_qualified_as_observed():
    row = features()
    del row['atr_14']
    result = update(observer_for(), row)
    assert result['atr']['source'] == 'close_times_0.02'
    assert result['atr']['status'] == 'defaulted'
    assert 'R1:unqualified_atr' in result['blockers']
    assert result['arms']['baseline']['signal']  # diagnostic native behavior preserved


def test_patches_restore_on_native_error():
    engine = SignalEngine()
    observer = observer_for(engine)
    original_detect = engine.engine.archetypes['trap_within_trend'].detect
    original_derived = native.DERIVED_FEATURES.copy()
    row = features()
    with patch.object(engine.engine, 'get_signals', side_effect=RuntimeError('fixture')):
        with pytest.raises(RuntimeError):
            update(observer, row)
    assert engine.engine.archetypes['trap_within_trend'].detect == original_detect
    assert native.DERIVED_FEATURES == original_derived


def test_hourly_source_stream_binding_is_preserved_and_foreign_rows_rejected():
    observer = observer_for()
    result = update(observer, features())
    assert result['opportunity']['data_stream_id'] == 'fixture'
    assert observer.snapshot()['data_stream_id'] == 'fixture'
    row = dict(features(1), data_stream_id='foreign')
    with pytest.raises(ValueError, match='stream'):
        update(observer, row)
