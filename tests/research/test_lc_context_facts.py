"""Hand-calculated context facts; no private archive or economic outcomes."""
from copy import deepcopy
import importlib

import pytest

from scripts.research.conditional_assessment import digest
from tests.research.test_lc_master_assessment import packet


def describe(p):
    module = 'scripts.research.lc_context_facts'
    assert importlib.util.find_spec(module), 'context fact extractor missing'
    return importlib.import_module(module).describe_lc_context(p)


def reseal(p):
    p.pop('seal', None)
    p['seal'] = digest(p)
    return p


def price_case(low, high, close):
    p = packet()
    for row in (p['current']['features'], p['current']['source_candle']):
        row.update(low=low, high=high, close=close)
    p['plan']['indicative_close'] = close
    return reseal(p)


@pytest.mark.parametrize('low,high,close,relation,low_sweep,high_sweep,reclaim,rejection', [
    (95., 112., 98., 'below_prior_low', True, True, False, True),
    (101., 115., 114., 'above_prior_high', False, True, False, False),
    (101., 112., 108., 'inside_or_boundary', False, True, False, True),
    (99., 110., 105., 'inside_or_boundary', True, False, True, False),
    (100., 110., 100., 'inside_or_boundary', False, False, False, False),
    (100., 110., 110., 'inside_or_boundary', False, False, False, False),
    (99., 110., 100., 'inside_or_boundary', True, False, False, False),
    (100., 111., 110., 'inside_or_boundary', False, True, False, False),
])
def test_geometry_is_not_rsi_and_sweep_flags_are_independent(
        low, high, close, relation, low_sweep, high_sweep, reclaim, rejection):
    p = price_case(low, high, close)
    # A high RSI does not turn a downside close into an upside expansion.
    p['current']['features']['rsi_14'] = 80.
    result = describe(reseal(p))['hourly']
    assert result['status'] == 'known'
    assert result['close_relation'] == relation
    assert result['swept_prior_low'] is low_sweep
    assert result['swept_prior_high'] is high_sweep
    assert result['reclaimed_prior_low'] is reclaim
    assert result['rejected_prior_high'] is rejection


def test_raw_final_selection_and_old_reference_failure_do_not_erase_facts():
    p = price_case(95., 110., 98.)
    before = deepcopy(p)
    result = describe(p)
    assert result['native_long'] is True  # native selected is False in fixture
    assert result['hourly']['close_relation'] == 'below_prior_low'
    assert result['parent_4h']['lifecycle'] == 'intact'
    assert result['execution_authorized'] is False
    assert 'plan_id' not in result
    assert p == before


@pytest.mark.parametrize('change', ['unvalidated', 'future', 'mismatch', 'boolean', 'provenance'])
def test_untrusted_hourly_operands_stay_unknown(change):
    p = packet()
    if change == 'unvalidated': p['current']['validated'] = False
    if change == 'future': p['previous']['available_at'] = '2026-01-01T04:00:00Z'
    if change == 'mismatch': p['current']['features']['close'] = 1.
    if change == 'boolean': p['current']['source_candle']['close'] = True
    if change == 'provenance': p['provenance']['reconstruction_verified'] = False
    result = describe(reseal(p))
    assert result['hourly']['status'] == 'unknown'
    assert result['hourly']['close_relation'] is None
    assert result['parent_4h']['ceiling_distance_r'] is None


@pytest.mark.parametrize('direction,want', [('up', 'broken_up'), ('down', 'broken_down')])
def test_known_parent_break_is_a_directional_fact_not_missing_data(direction, want):
    p = packet(); parent = p['evidence']['parent_4h']
    parent['status'] = 'fail'; parent['lineage_broken'] = True
    parent['reasons'] = ['bound_lineage_broken']
    parent['updates'][-1]['source_break_direction'] = direction
    result = describe(reseal(p))['parent_4h']
    assert result['evidence_status'] == 'known'
    assert result['pre_setup'] == 'present'
    assert result['lifecycle'] == want
    assert result['child_nested'] is True
    assert result['ceiling_distance_r'] == pytest.approx(15./5.4)


def test_absence_is_distinct_from_unknown_and_boundary_formation_is_not_prior():
    p = packet(); parent = p['evidence']['parent_4h']
    parent.update(status='fail', bound=None, pivots=[], reasons=['absent_parent'],
                  lineage_broken=False)
    result = describe(reseal(p))['parent_4h']
    assert result['evidence_status'] == 'known'
    assert result['pre_setup'] == 'absent'
    assert result['lifecycle'] == 'absent'
    assert result['child_nested'] is None
    assert result['ceiling_distance_r'] is None
    assert result['updates']  # boundary updates are retained, not backdated
    assert result['decision_state'] == 'active'
    assert result['lifecycle_scope'] == 'strict_before_bound_only'
    parent['status'] = 'unknown'; parent['reasons'] = ['insufficient_history']
    result = describe(reseal(p))['parent_4h']
    assert result['evidence_status'] == 'unknown'
    assert result['pre_setup'] == 'unknown'


def test_mismatched_parent_pivot_stream_is_unknown():
    p = packet()
    p['evidence']['parent_4h']['pivots'][0]['data_stream_id'] = 'other-stream'
    assert describe(reseal(p))['parent_4h']['evidence_status'] == 'unknown'


def test_future_raw_tail_cannot_change_facts():
    from tests.research.test_lc_master_assessment import inputs, api
    raw, bars, parents, provenance = inputs()
    first = describe(api().build_lc_packet(raw, bars, parents, provenance, 'LC1'))
    bars.loc['2026-01-01 04:00':, :] = 999.
    second = describe(api().build_lc_packet(raw, bars, parents, provenance, 'LC1'))
    assert second == first


@pytest.mark.parametrize('change', ['bound_at_setup', 'future_update', 'false_break', 'unknown', 'wrong_anchor'])
def test_invalid_parent_metadata_cannot_become_known_structure(change):
    p = packet(); parent = p['evidence']['parent_4h']
    if change == 'bound_at_setup': parent['bound']['available_at'] = p['setup_open']
    if change == 'future_update': parent['updates'][-1]['available_at'] = '2026-01-01T05:00:00Z'
    if change == 'false_break': parent['lineage_broken'] = True
    if change == 'unknown': parent['status'] = 'unknown'
    if change == 'wrong_anchor': parent['anchor'] = '1D'
    result = describe(reseal(p))['parent_4h']
    assert result['evidence_status'] == 'unknown'
    assert result['ceiling_distance_r'] is None


def test_signed_distance_is_not_clipped_or_a_room_permission():
    result = describe(price_case(101., 125., 123.))['parent_4h']
    assert result['ceiling_distance_r'] == pytest.approx(-3./23.4)
    assert result['position_in_bound'] == pytest.approx(1.1)


def test_predecision_sequence_reports_higher_low_without_claiming_later_confirmation():
    p = packet()
    p['evidence']['5m'][-2][1:5] = [104., 108., 99., 101.]
    p['evidence']['5m'][-1][1:5] = [101., 109., 100., 107.]
    result = describe(reseal(p))['last_two_5m']
    assert result['status'] == 'known'
    assert result['higher_low'] is True
    assert result['higher_close'] is True
    assert result['last_body'] == 'up'
    assert result['last_close_location'] == pytest.approx(7./9.)
    assert result['postdecision_confirmation_evaluated'] is False


@pytest.mark.parametrize('change', ['future', 'stale', 'duplicate', 'gap', 'bad_ohlc', 'boolean', 'missing'])
def test_invalid_minute_sequence_is_unknown(change):
    p = packet(); rows = p['evidence']['5m']
    if change == 'future': rows[-1][0] = p['decision_time']
    if change == 'stale': rows.pop()
    if change == 'duplicate': rows[-2][0] = rows[-1][0]
    if change == 'gap': rows[-2][0] = '2026-01-01T03:45:00Z'
    if change == 'bad_ohlc': rows[-1][3] = 200.
    if change == 'boolean': rows[-1][4] = True
    if change == 'missing': p['evidence']['5m'] = []
    result = describe(reseal(p))['last_two_5m']
    assert result['status'] == 'unknown'
    assert result['higher_low'] is None


def test_changed_seal_is_refused_not_credited_as_rejection():
    p = packet(); p['current']['features']['close'] = 999.
    with pytest.raises(ValueError, match='seal'):
        describe(p)
