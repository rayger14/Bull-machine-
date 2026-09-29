"""Pure hypothetical pre-entry eligibility. No exit replay, orders or PnL.

Execution inputs are supplied fixtures, not authenticated venue fills. Quantity
is a modeled upper bound, not an order size or guarantee against gap losses.
"""
from decimal import Decimal, ROUND_FLOOR, InvalidOperation

import pandas as pd

from scripts.research.lc_structure_packet import clock, number, prices
from scripts.research.lc_structure_proposal import validate_structure_proposal, obstacle_ids


VERSION = 'lc_structure_preentry_v1'
EXECUTION_KEYS = {'response_available_at', 'proposed_fill_at', 'fill_open', 'completed_minutes'}
MINUTE_KEYS = {'open_time', 'open', 'high', 'low', 'close', 'volume'}


def _result(status, reason, geometry=None):
    return dict(status=status, reason=reason, geometry=geometry, execution_authorized=False)


def _execution(execution, decision):
    if not isinstance(execution, dict) or set(execution) != EXECUTION_KEYS:
        raise ValueError('execution shape')
    response = clock(execution['response_available_at'])
    fill = clock(execution['proposed_fill_at'])
    if (response < decision or response > fill or fill != fill.floor('min')
            or not number(execution['fill_open'])):
        raise ValueError('execution clocks or fill')
    rows = execution['completed_minutes']
    if not isinstance(rows, list):
        raise ValueError('minute list required')
    previous = None
    coverage = True
    for row in rows:
        if not isinstance(row, dict) or set(row) != MINUTE_KEYS or not prices(row):
            raise ValueError('minute shape/prices')
        opened = clock(row['open_time'])
        if (opened != opened.floor('min') or opened < decision
                or opened + pd.Timedelta(minutes=1) > fill
                or (previous is not None and opened <= previous)):
            raise ValueError('minute time/order')
        expected = decision if previous is None else previous + pd.Timedelta(minutes=1)
        coverage = coverage and opened == expected
        previous = opened
    end = decision if previous is None else previous + pd.Timedelta(minutes=1)
    return response, fill, rows, coverage and end == fill


def _rounded(value, tick):
    return (Decimal(str(value)) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def _check(packet, proposal, policy, execution):
    decision = clock(packet['decision_time'])
    response, fill_at, rows, covered = _execution(execution, decision)
    if not covered:
        return _result('not_ready', 'coverage_gap')
    arm = (max(response, decision + pd.Timedelta(seconds=policy['processing_seconds']))
           + pd.Timedelta(seconds=policy['routing_seconds'])).ceil('min')
    expiry = decision + pd.Timedelta(minutes=policy['entry_expiry_minutes'])
    plan = proposal['plan']; levels = packet['levels']
    tick = Decimal(str(policy['tick_size']))
    stop = _rounded(levels[plan['stop_level_id']]['price'], tick)
    destination = _rounded(levels[plan['destination_level_id']]['price'], tick)
    cap = _rounded(policy['max_entry_price'], tick)
    invalidation = Decimal(str(levels[plan['invalidation_level_id']]['price']))
    fill = Decimal(str(execution['fill_open']))
    if min(stop, destination, cap) <= 0:
        return _result('invalid', 'rounded_geometry')
    boundary = max(stop, invalidation)
    if any(Decimal(str(row['low'])) <= boundary for row in rows) or fill <= boundary:
        return _result('cancelled', 'preentry_invalidation')
    if fill_at >= expiry:
        return _result('cancelled', 'expired')
    if fill_at < arm:
        return _result('not_ready', 'before_arm')
    if plan['trigger']['kind'] == 'immediate':
        first_fill = arm
    else:
        trigger = Decimal(str(levels[plan['trigger']['level_id']]['price']))
        triggers = [clock(row['open_time']) + pd.Timedelta(minutes=1) for row in rows
                    if clock(row['open_time']) >= arm and Decimal(str(row['close'])) > trigger]
        if not triggers:
            return _result('not_ready', 'trigger_not_met')
        first_fill = triggers[0]
    if fill_at != first_fill:
        return _result('not_ready', 'missed_entry')
    if fill >= destination:
        return _result('cancelled', 'destination_reached')
    if fill > cap:
        return _result('cancelled', 'entry_cap')
    risk, reward = fill - stop, destination - fill
    cost = fill * Decimal(str(policy['roundtrip_cost_bps'])) / Decimal(10000)
    rr = (reward - cost) / (risk + cost)
    if rr < Decimal(str(policy['minimum_net_rr'])):
        return _result('cancelled', 'insufficient_room')
    quantity = min(Decimal(str(policy['risk_budget_usd'])) / (risk + cost),
                   Decimal(str(policy['max_notional_usd'])) / fill,
                   Decimal(str(policy['equity_usd'])) * Decimal(str(policy['max_leverage'])) / fill)
    values = dict(fill_price=fill, stop=stop, destination=destination, risk_per_unit=risk,
                  reward_per_unit=reward, cost_per_unit=cost, net_rr=rr, quantity_upper_bound=quantity)
    geometry = {k: float(v) for k,v in values.items()}
    if not all(number(v, zero=k in ('cost_per_unit', 'net_rr')) for k,v in geometry.items()):
        return _result('invalid', 'nonfinite_geometry')
    geometry['intervening_level_ids'] = obstacle_ids(packet, float(fill), float(destination))
    geometry['arm_at'] = arm.isoformat()
    geometry['fill_at'] = fill_at.isoformat()
    geometry['source_authentication'] = 'supplied hypothetical execution; not authenticated fills'
    return _result('eligible_hypothetical', 'checks_passed', geometry)


def check_structure_preentry(source_request: dict, packet: dict,
                            raw_response: str, policy, execution: dict) -> dict:
    """Revalidate original source and answer; never consume an asserted grade."""
    checked = validate_structure_proposal(source_request, packet, raw_response, policy)
    if checked['status'] == 'invalid':
        return _result('invalid', 'proposal_invalid')
    if checked['status'] != 'valid_proposal':
        return _result('not_ready', 'no_entry_proposal')
    if policy is None:
        return _result('not_ready', 'missing_policy')
    try:
        return _check(packet, checked['proposal'], policy, execution)
    except (KeyError, ValueError, TypeError, OverflowError, InvalidOperation, ZeroDivisionError):
        return _result('invalid', 'execution_input')
