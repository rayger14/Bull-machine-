"""Pure comparison wrapper for frozen minute parent-context sleeves.

This adapter deliberately delegates every fill and management decision to the
frozen minute simulator.  It only validates precomputed permissions, filters
candidates before each independent replay, and joins their statuses back to the
unchanged source population.
"""
import copy
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from scripts.research.minute_sweep_validation import simulate_events, validate_bars


VARIANTS = ('4H:3', '4H:5', '1D:3', '1D:5')
ENTRY_STATUSES = frozenset(('completed', 'open_censored'))
FIXED_PARAMETERS = {
    'entry_mode': 'next_open',
    'notional': 50000.0,
    'stop_buffer': 0.0015,
    'hold_minutes': 240,
    'cost_bps': 12.0,
    'decision_delay_seconds': 0,
    'lockout': 'fixed_from_actual_entry_until_deadline',
    'starting_state': 'flat',
    'starting_equity': None,
}


def _is_index(value):
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)))


def _utc_timestamp(value, label):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid {label}') from exc
    if timestamp.tz is None or pd.isna(timestamp):
        raise ValueError(f'{label} must be timezone-aware')
    try:
        timestamp = timestamp.tz_convert('UTC')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid {label}') from exc
    try:
        subminute = timestamp.value % pd.Timedelta(minutes=1).value
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid {label}') from exc
    if subminute:
        raise ValueError(f'{label} must align exactly to the UTC minute grid')
    return timestamp


def _event_id(index, event):
    reclaim = index[event['reclaim_idx']].tz_convert('UTC').isoformat()
    pivot = index[event['pivot_idx']].tz_convert('UTC').isoformat()
    return f'reclaim:{reclaim}|pivot:{pivot}'


def _validate_events(bars, events, start, end):
    if not isinstance(events, list):
        raise ValueError('Events must be a list')
    copied = copy.deepcopy(events)
    required = ('pivot_idx', 'confirmed_idx', 'sweep_idx', 'reclaim_idx',
                'level', 'sweep_low', 'touches')
    ids, previous_reclaim = [], -1
    for event in copied:
        if not isinstance(event, dict) or any(key not in event for key in required):
            raise ValueError('Malformed event')
        indices = [event[key] for key in required[:4]]
        if (any(not _is_index(value) or not 0 <= value < len(bars) for value in indices)
                or not event['pivot_idx'] <= event['confirmed_idx'] <= event['sweep_idx'] <= event['reclaim_idx']):
            raise ValueError('Invalid event indices or chronology')
        if event['reclaim_idx'] <= previous_reclaim:
            raise ValueError('Events must have unique increasing reclaim indices')
        previous_reclaim = event['reclaim_idx']
        if (not isinstance(event['touches'], (int, np.integer))
                or isinstance(event['touches'], (bool, np.bool_))
                or event['touches'] < 1):
            raise ValueError('Invalid event touches')
        prices = (event['level'], event['sweep_low'])
        if any(type(value) not in (int, float) for value in prices):
            raise ValueError('Invalid event price type')
        try:
            numeric = np.asarray(prices, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Invalid event prices') from exc
        if not np.isfinite(numeric).all() or (numeric <= 0).any():
            raise ValueError('Invalid event prices')
        reclaim_time = bars.index[event['reclaim_idx']].tz_convert('UTC')
        if not start <= reclaim_time < end:
            raise ValueError('Every event reclaim must lie inside the declared window')
        ids.append(_event_id(bars.index, event))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate derived event identity')
    return copied, ids


def _validate_permissions(permissions, event_ids):
    if not isinstance(permissions, dict) or set(permissions) != set(VARIANTS):
        raise ValueError('Permissions must contain exactly the four required variants')
    expected = set(event_ids)
    copied = {}
    for variant in VARIANTS:
        mapping = permissions[variant]
        if not isinstance(mapping, dict) or set(mapping) != expected:
            raise ValueError('Every permission map must contain every event identity exactly once')
        for event_id, value in mapping.items():
            if value is not True and value is not False and value is not None:
                raise ValueError('Permission values must be strict True, False, or None')
        copied[variant] = copy.deepcopy(mapping)
    return copied


def _run(events, bars):
    return simulate_events(
        bars.copy(deep=True), copy.deepcopy(events), entry_mode='next_open',
        notional=50000.0, stop_buffer=.0015, hold_minutes=240, cost_bps=12.0,
        decision_delay_seconds=0,
    )


def _joined_ledger(event_ids, events, simulation, permissions=None):
    simulated = {row['event_id']: copy.deepcopy(row) for row in simulation['event_ledger']}
    if len(simulated) != len(simulation['event_ledger']):
        raise ValueError('Simulator returned duplicate event identity')
    rows = []
    for event_id, event in zip(event_ids, events):
        if permissions is not None and permissions[event_id] is not True:
            rows.append({
                'event_id': event_id,
                'reclaim_idx': int(event['reclaim_idx']),
                'pivot_idx': int(event['pivot_idx']),
                'status': ('permission_rejected' if permissions[event_id] is False
                           else 'permission_unknown'),
            })
        else:
            try:
                rows.append(simulated[event_id])
            except KeyError as exc:
                raise ValueError('Simulator ledger did not cover selected event') from exc
    return rows


def _summary(simulation):
    summary = copy.deepcopy(simulation['summary'])
    trades = simulation['trades']
    fees = float(sum(trade['fees'] for trade in trades))
    ratios = [float(trade['pnl'] / trade['initial_risk']) for trade in trades]
    summary.update(
        completed_fees=fees,
        completed_gross_pnl=float(sum(trade['pnl'] + trade['fees'] for trade in trades)),
        mean_net_pnl_over_initial_risk=float(np.mean(ratios)) if ratios else None,
        median_net_pnl_over_initial_risk=float(np.median(ratios)) if ratios else None,
    )
    return summary


def _comparison(baseline_ledger, arm_ledger):
    baseline = {row['event_id']: row['status'] for row in baseline_ledger}
    arm = {row['event_id']: row['status'] for row in arm_ledger}
    baseline_entered = [event_id for event_id, status in baseline.items() if status in ENTRY_STATUSES]
    arm_entered = [event_id for event_id, status in arm.items() if status in ENTRY_STATUSES]
    arm_entered_set = set(arm_entered)
    baseline_entered_set = set(baseline_entered)
    transitions = OrderedDict()
    for event_id in baseline:
        pair = (baseline[event_id], arm[event_id])
        transitions[pair] = transitions.get(pair, 0) + 1
    return {
        'shared_entered_ids': [event_id for event_id in baseline_entered if event_id in arm_entered_set],
        'baseline_only_entered_ids': [event_id for event_id in baseline_entered if event_id not in arm_entered_set],
        'arm_only_entered_ids': [event_id for event_id in arm_entered if event_id not in baseline_entered_set],
        'status_transitions': [
            {'baseline_status': pair[0], 'arm_status': pair[1], 'count': count}
            for pair, count in transitions.items()
        ],
    }


def compare_minute_parent_arms(bars, events, permissions, *, window_start, window_end):
    """Compare baseline and four precomputed parent-permission sleeves.

    Inputs are validated completely before any simulator call.  Each permission
    arm replays only its allowed candidates, so denial cannot create a lockout.
    """
    try:
        validate_bars(bars)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid minute bars') from exc
    start = _utc_timestamp(window_start, 'window_start')
    end = _utc_timestamp(window_end, 'window_end')
    if not start < end:
        raise ValueError('window_start must precede window_end')
    bars_copy = bars.copy(deep=True)
    copied_events, event_ids = _validate_events(bars_copy, events, start, end)
    copied_permissions = _validate_permissions(permissions, event_ids)

    baseline_simulation = _run(copied_events, bars_copy)
    baseline_ledger = _joined_ledger(event_ids, copied_events, baseline_simulation)
    arms = OrderedDict()
    arms['baseline'] = {
        'simulation': baseline_simulation,
        'event_ledger': baseline_ledger,
        'summary': _summary(baseline_simulation),
    }
    for variant in VARIANTS:
        allowed = [event for event, event_id in zip(copied_events, event_ids)
                   if copied_permissions[variant][event_id] is True]
        simulation = _run(allowed, bars_copy)
        ledger = _joined_ledger(event_ids, copied_events, simulation, copied_permissions[variant])
        arms[variant] = {
            'simulation': simulation,
            'event_ledger': ledger,
            'summary': _summary(simulation),
        }
    for arm in arms.values():
        arm['comparison'] = _comparison(baseline_ledger, arm['event_ledger'])
    result = {
        'parameters': copy.deepcopy(FIXED_PARAMETERS),
        'coverage': {
            'window_start': start.isoformat(),
            'window_end': end.isoformat(),
            'event_count': len(copied_events),
        },
        'arms': arms,
        'limitations': [
            'Unfunded reference sleeves: starting equity, funding, margin, and compounding are unspecified.',
            'Conditional replay of precomputed permissions; this adapter does not verify parent evidence.',
            'All sleeves start flat at the declared window and preserve the upstream selector population.',
            'Missing tails remain open_censored or unfilled; no closure is manufactured.',
            'Fixed-notional fee, spread, impact, and receipt latency assumptions remain unmodeled beyond the frozen simulator.',
        ],
        'certified': False,
    }
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Comparison result is not strict JSON representable') from exc
    return result
