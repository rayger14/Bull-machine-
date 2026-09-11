import copy
import json
from decimal import Decimal

import pandas as pd
import pytest

from scripts.research.minute_parent_comparison import compare_minute_parent_arms
from scripts.research.minute_sweep_validation import simulate_events


def fixture():
    bars = pd.DataFrame(
        {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0},
        index=pd.date_range('2026-01-01', periods=720, freq='min', tz='UTC'),
    )
    events = [
        dict(pivot_idx=1, confirmed_idx=16, sweep_idx=18, reclaim_idx=20,
             level=98.0, sweep_low=95.0, touches=2),
        dict(pivot_idx=80, confirmed_idx=95, sweep_idx=98, reclaim_idx=100,
             level=98.0, sweep_low=95.0, touches=2),
    ]
    ids = [
        'reclaim:' + bars.index[event['reclaim_idx']].isoformat()
        + '|pivot:' + bars.index[event['pivot_idx']].isoformat()
        for event in events
    ]
    permissions = {
        name: {key: True for key in ids}
        for name in ('4H:3', '4H:5', '1D:3', '1D:5')
    }
    return bars, events, ids, permissions


def test_permission_rejection_happens_before_lockout_and_exposes_later_candidate():
    """Catches an adapter that runs the lockout before applying permission."""
    bars, events, ids, permissions = fixture()
    permissions['4H:3'][ids[0]] = False

    result = compare_minute_parent_arms(
        bars, events, permissions,
        window_start='2026-01-01T00:00:00Z', window_end='2026-01-01T12:00:00Z',
    )

    base = result['arms']['baseline']
    arm = result['arms']['4H:3']
    assert [row['status'] for row in base['event_ledger']] == ['completed', 'skipped_busy']
    assert [row['status'] for row in arm['event_ledger']] == ['permission_rejected', 'completed']
    assert arm['comparison']['arm_only_entered_ids'] == [ids[1]]
    assert arm['comparison']['baseline_only_entered_ids'] == [ids[0]]


def test_all_permitted_arms_preserve_the_whole_unmodified_simulator_result():
    """Catches an adapter that alters fills, management, or simulator records."""
    bars, events, _, permissions = fixture()
    result = compare_minute_parent_arms(
        bars, events, permissions,
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    direct = simulate_events(
        bars, events, entry_mode='next_open', notional=50000., stop_buffer=.0015,
        hold_minutes=240, cost_bps=12., decision_delay_seconds=0,
    )
    assert result['arms']['baseline']['simulation'] == direct
    for name in permissions:
        assert result['arms'][name]['simulation'] == direct
        assert result['arms'][name]['comparison']['status_transitions'] == [
            {'baseline_status': 'completed', 'arm_status': 'completed', 'count': 1},
            {'baseline_status': 'skipped_busy', 'arm_status': 'skipped_busy', 'count': 1},
        ]


def test_unknown_and_rejected_permissions_remain_in_complete_ledger_without_entries():
    """Catches treating unknown/false as truthy or dropping denied event identities."""
    bars, events, ids, permissions = fixture()
    permissions['4H:3'] = {ids[0]: None, ids[1]: False}
    result = compare_minute_parent_arms(
        bars, events, permissions,
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    arm = result['arms']['4H:3']
    assert [row['event_id'] for row in arm['event_ledger']] == ids
    assert [row['status'] for row in arm['event_ledger']] == [
        'permission_unknown', 'permission_rejected',
    ]
    assert arm['simulation']['trades'] == []
    assert arm['comparison']['status_transitions'] == [
        {'baseline_status': 'completed', 'arm_status': 'permission_unknown', 'count': 1},
        {'baseline_status': 'skipped_busy', 'arm_status': 'permission_rejected', 'count': 1},
    ]


def test_empty_event_population_with_complete_empty_permissions_is_valid():
    """Catches an adapter that mistakes a valid empty frozen population for a partial map."""
    bars, _, _, _ = fixture()
    result = compare_minute_parent_arms(
        bars, [], {name: {} for name in ('4H:3', '4H:5', '1D:3', '1D:5')},
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    assert result['coverage']['event_count'] == 0
    assert all(arm['event_ledger'] == [] for arm in result['arms'].values())


def test_completed_fee_gross_and_initial_risk_ratio_are_derived_from_actual_trade():
    """Catches confusing one-side $30 fee with the specified $60 completed fee."""
    bars, events, _, permissions = fixture()
    result = compare_minute_parent_arms(
        bars, [events[0]],
        {name: {next(iter(values)): True} for name, values in permissions.items()},
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    summary = result['arms']['baseline']['summary']
    risk = (100.0 - 95.0 * .9985) * 500.0
    assert summary['completed_fees'] == 60.0
    assert summary['completed_gross_pnl'] == 0.0
    assert summary['mean_net_pnl_over_initial_risk'] == pytest.approx(-60.0 / risk)
    assert summary['median_net_pnl_over_initial_risk'] == pytest.approx(-60.0 / risk)


@pytest.mark.parametrize('stop_idx, stop_open, expected_fill', [
    (21, 100.0, 95.0 * .9985),
    (22, 90.0, 90.0),
])
def test_same_entry_bar_and_gap_stops_are_left_to_frozen_simulator(stop_idx, stop_open, expected_fill):
    """Catches a wrapper that changes entry-bar stop or gap-fill behavior."""
    bars, events, ids, permissions = fixture()
    bars = bars.copy()
    bars.loc[bars.index[stop_idx], ['open', 'high', 'low', 'close']] = [stop_open, 101.0, 89.0, 100.0]
    maps = {name: {ids[0]: True} for name in permissions}
    result = compare_minute_parent_arms(
        bars, [events[0]], maps,
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    trade = result['arms']['baseline']['simulation']['trades'][0]
    assert trade['reason'] == 'stop'
    assert trade['exit_idx'] == stop_idx
    assert trade['exit_price'] == pytest.approx(expected_fill)


def test_early_stop_does_not_release_the_fixed_entry_lockout():
    """Catches a wrapper that reruns candidates after a stopped position closes early."""
    bars, events, ids, permissions = fixture()
    bars = bars.copy()
    bars.loc[bars.index[21], ['open', 'high', 'low', 'close']] = [100.0, 101.0, 90.0, 100.0]
    result = compare_minute_parent_arms(
        bars, events, permissions,
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    assert [row['status'] for row in result['arms']['baseline']['event_ledger']] == [
        'completed', 'skipped_busy',
    ]


def test_censored_and_unfilled_tail_statuses_are_preserved_without_invented_closure():
    """Catches tail handling that converts censored or unfilled candidates into trades."""
    bars, events, ids, permissions = fixture()
    bars = bars.iloc[:101].copy()
    maps = {name: {event_id: True for event_id in ids} for name in permissions}
    result = compare_minute_parent_arms(
        bars, events, maps,
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    simulation = result['arms']['baseline']['simulation']
    assert [row['status'] for row in result['arms']['baseline']['event_ledger']] == [
        'open_censored', 'unfilled',
    ]
    assert len(simulation['open_positions']) == 1
    assert simulation['trades'] == []


def test_rejected_out_of_window_event_is_not_silently_admitted_or_ignored():
    """Catches validating only permitted events instead of the complete source population."""
    bars, events, ids, permissions = fixture()
    permissions['4H:3'][ids[1]] = False
    with pytest.raises(ValueError, match='window'):
        compare_minute_parent_arms(
            bars, events, permissions,
            window_start=bars.index[0], window_end=bars.index[50],
        )


@pytest.mark.parametrize('mutation', [
    lambda maps, ids: maps.pop('4H:3'),
    lambda maps, ids: maps.__setitem__('bad', maps['4H:3'].copy()),
    lambda maps, ids: maps['4H:3'].pop(ids[0]),
    lambda maps, ids: maps['4H:3'].__setitem__('reclaim:extra|pivot:extra', True),
])
def test_incomplete_or_extra_permission_identities_fail_closed(mutation):
    """Catches partial/guessed permission joins."""
    bars, events, ids, permissions = fixture()
    mutation(permissions, ids)
    with pytest.raises(ValueError):
        compare_minute_parent_arms(
            bars, events, permissions,
            window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
        )


@pytest.mark.parametrize('value', [1, 0, 'true', 'False'])
def test_non_boolean_permission_values_fail_instead_of_using_truthiness(value):
    """Catches numeric/string truthiness in frozen annotations."""
    bars, events, ids, permissions = fixture()
    permissions['4H:3'][ids[0]] = value
    with pytest.raises(ValueError, match='strict'):
        compare_minute_parent_arms(
            bars, events, permissions,
            window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
        )


def test_malformed_index_clock_and_containers_fail_before_economic_calls():
    """Catches adapter validation delegated too late to a partial simulator call."""
    bars, events, ids, permissions = fixture()
    naive = bars.copy()
    naive.index = naive.index.tz_localize(None)
    for bad_bars, bad_events, bad_permissions in [
        (naive, events, permissions),
        (bars, tuple(events), permissions),
        (bars, events, []),
    ]:
        with pytest.raises(ValueError):
            compare_minute_parent_arms(
                bad_bars, bad_events, bad_permissions,
                window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
            )
    broken = copy.deepcopy(events)
    broken[0]['confirmed_idx'] = 0
    with pytest.raises(ValueError):
        compare_minute_parent_arms(
            bars, broken, permissions,
            window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
        )


def test_inputs_are_not_mutated_and_output_is_repeatable_and_strict_json_safe():
    """Catches source mutation, unstable joins, or JSON-incompatible output."""
    bars, events, _, permissions = fixture()
    original_bars, original_events, original_permissions = bars.copy(deep=True), copy.deepcopy(events), copy.deepcopy(permissions)
    kwargs = dict(window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1))
    first = compare_minute_parent_arms(bars, events, permissions, **kwargs)
    second = compare_minute_parent_arms(bars.copy(deep=True), copy.deepcopy(events), copy.deepcopy(permissions), **kwargs)
    assert bars.equals(original_bars)
    assert events == original_events
    assert permissions == original_permissions
    assert first == second
    assert json.dumps(first, allow_nan=False)


@pytest.mark.parametrize('field, value', [('level', 10 ** 10000), ('sweep_low', float('nan'))])
def test_nonfinite_or_unrepresentable_event_prices_raise_value_error(field, value):
    """Catches a public validation boundary leaking conversion/JSON exceptions."""
    bars, events, _, permissions = fixture()
    events = copy.deepcopy(events)
    events[0][field] = value
    with pytest.raises(ValueError):
        compare_minute_parent_arms(
            bars, events, permissions,
            window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
        )


def test_nonfinite_extra_raw_event_content_fails_strict_json_instead_of_being_silently_changed():
    """Catches a wrapper that returns nonfinite raw event content or clips it silently."""
    bars, events, _, permissions = fixture()
    events = copy.deepcopy(events)
    events[0]['frozen_annotation'] = float('nan')
    with pytest.raises(ValueError, match='JSON'):
        compare_minute_parent_arms(
            bars, events, permissions,
            window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
        )


def test_unrepresentable_bar_price_raises_value_error_at_public_boundary():
    """Catches an oversized OHLC conversion leaking an implementation exception."""
    bars, events, _, permissions = fixture()
    bars = bars.astype(object)
    bars.loc[bars.index[0], 'open'] = 10 ** 10000
    with pytest.raises(ValueError):
        compare_minute_parent_arms(
            bars, events, permissions,
            window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
        )


@pytest.mark.parametrize('window_start, window_end', [
    ('2026-01-01T00:00:30Z', '2026-01-01T12:00:00Z'),
    ('2026-01-01T00:00:00Z', '2026-01-01T12:00:00.001Z'),
])
def test_subminute_window_bounds_fail_before_replay(window_start, window_end):
    """Catches accepting a window that cannot align to the minute simulator grid."""
    bars, events, _, permissions = fixture()
    with pytest.raises(ValueError, match='minute'):
        compare_minute_parent_arms(
            bars, events, permissions, window_start=window_start, window_end=window_end,
        )


def test_timezone_equivalent_aligned_window_is_normalized_to_utc():
    """Catches treating non-UTC but aligned permission windows as invalid."""
    bars, events, _, permissions = fixture()
    result = compare_minute_parent_arms(
        bars, events, permissions,
        window_start='2025-12-31T16:00:00-08:00',
        window_end='2026-01-01T04:00:00-08:00',
    )
    assert result['coverage']['window_start'] == '2026-01-01T00:00:00+00:00'
    assert result['coverage']['window_end'] == '2026-01-01T12:00:00+00:00'


@pytest.mark.parametrize('field, value', [
    ('level', '98'),
    ('sweep_low', Decimal('95')),
    ('level', True),
])
def test_non_json_event_price_types_fail_before_the_unchanged_simulator(field, value):
    """Catches coercing a raw frozen price that later breaks simulator validation."""
    bars, events, _, permissions = fixture()
    events = copy.deepcopy(events)
    events[0][field] = value
    with pytest.raises(ValueError, match='price'):
        compare_minute_parent_arms(
            bars, events, permissions,
            window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
        )


def test_standard_json_numeric_event_prices_remain_valid_without_normalization():
    """Catches overrestricting integer frozen values while adding the raw-type contract."""
    bars, events, _, permissions = fixture()
    events = copy.deepcopy(events)
    events[0]['level'], events[0]['sweep_low'] = 98, 95
    result = compare_minute_parent_arms(
        bars, events, permissions,
        window_start=bars.index[0], window_end=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    assert result['arms']['baseline']['simulation']['trades'][0]['event']['sweep_low'] == 95
