"""Wyckoff contribution is directional evidence, never an availability substitute."""
import pytest

from engine.archetypes.archetype_instance import ArchetypeInstance


def score(features, direction='long'):
    instance = object.__new__(ArchetypeInstance)
    instance.direction = direction
    return instance._get_wyckoff_score(features)


@pytest.mark.parametrize('direction,opposite', [('long', 'bearish'), ('short', 'bullish')])
@pytest.mark.parametrize('prefix', ['', 'tf4h_', 'tf1d_'])
def test_opposite_only_schema_never_falls_back_to_generic(direction, opposite, prefix):
    features = {prefix + 'wyckoff_' + opposite + '_score': .8,
                'wyckoff_event_confidence': .8, 'tf4h_wyckoff_phase_score': 1.,
                'tf1d_wyckoff_m1_signal': 1, 'tf1d_wyckoff_m2_signal': 1}
    assert score(features, direction) == 0.


@pytest.mark.parametrize('value', [0., None, float('nan'), float('inf'), 'broken'])
@pytest.mark.parametrize('key', ['wyckoff_bullish_score', 'wyckoff_bearish_event_confidence'])
def test_present_but_empty_directional_schema_blocks_both_legacy_fallbacks(key, value):
    assert score({key: value, 'wyckoff_event_confidence': .9,
                  'tf1d_wyckoff_m1_signal': 1}) == 0.


@pytest.mark.parametrize('status', ['unavailable', 'proxy', 'error', 'stale', None])
@pytest.mark.parametrize('direction,side', [('long', 'bullish'), ('short', 'bearish')])
def test_unavailable_source_overrides_stale_positive_values(status, direction, side):
    assert score({'wyckoff_evidence_status': status, 'wyckoff_' + side + '_score': .9,
                  'wyckoff_' + side + '_event_confidence': 1.,
                  'wyckoff_event_confidence': 1.}, direction) == 0.


def test_ema_proxy_is_not_wyckoff_even_without_directional_columns():
    for direction in ('long', 'short'):
        assert score({'tf4h_wyckoff_evidence_source': 'ema_proxy',
                      'tf4h_wyckoff_phase_score': 1.}, direction) == 0.


def test_invalid_source_disables_legacy_binary_escape():
    assert score({'tf1d_wyckoff_evidence_status': 'unavailable',
                  'tf1d_wyckoff_m1_signal': 1}) == 0.


def test_available_hourly_is_not_suppressed_by_unavailable_higher_timeframes():
    assert score({'wyckoff_evidence_status': 'available', 'wyckoff_bullish_score': .4,
                  'tf4h_wyckoff_evidence_status': 'unavailable', 'tf4h_wyckoff_bullish_score': 1.,
                  'tf1d_wyckoff_evidence_status': 'unavailable', 'tf1d_wyckoff_bullish_score': 1.}) == .4


@pytest.mark.parametrize('direction,side', [('long', 'bullish'), ('short', 'bearish')])
def test_valid_same_direction_weighting_is_unchanged(direction, side):
    features = {f'wyckoff_{side}_score': .4, f'tf4h_wyckoff_{side}_score': .8,
                f'tf1d_wyckoff_{side}_score': .2}
    assert score(features, direction) == pytest.approx(.624)
    assert score({f'wyckoff_{side}_event_confidence': .7}, direction) == .7


@pytest.mark.parametrize('features,direction,expected', [
    ({'wyckoff_event_confidence': .8}, 'long', .8),
    ({'tf4h_wyckoff_phase_score': .6}, 'short', .6),
    ({'tf1d_wyckoff_m1_signal': 1}, 'long', .6),
    ({'tf1d_wyckoff_m2_signal': 1}, 'short', .6),
    ({'wyckoff_event_confidence': float('inf')}, 'long', 0.),
    ({'tf4h_wyckoff_phase_score': -1.}, 'short', 0.),
    ({'tf1d_wyckoff_m1_signal': float('nan')}, 'long', 0.),
])
def test_only_valid_legacy_only_payloads_keep_compatibility(features, direction, expected):
    assert score(features, direction) == expected
