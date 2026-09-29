"""Offline LC bracket outcomes; never import into an outcome-hidden assessor.

Revalidates the frozen proposal/pre-entry evidence, then reads only the future
prefix needed to resolve its structural stop/target or fill-relative deadline.
This is a single-case theoretical calculation, not a portfolio or live executor.
No model calls, persistence, default economic policy or strategy selection.
"""
from decimal import Decimal
from hashlib import sha256
import math

import pandas as pd

from scripts.research.conditional_assessment import digest
from scripts.research.lc_structure_packet import clock, number, prices
from scripts.research.lc_structure_preentry import check_structure_preentry
from scripts.research.lc_structure_proposal import validate_structure_proposal


VERSION = 'lc_structure_outcome_v1'
MINUTE = pd.Timedelta(minutes=1)
FUTURE_KEYS = {'instrument', 'data_stream_id', 'minutes'}
SEMANTICS = dict(version=VERSION, horizon='fill_relative_minutes',
    bracket='frozen_rounded_stop_and_destination', same_bar='stop_first_flagged',
    stop_gap='open', target_gap='target_no_improvement', deadline='open_only',
    sizing='fractional_preentry_quantity_upper_bound',
    costs='entry_notional_times_frozen_roundtrip_bps_once',
    invalidation='must_equal_effective_protective_stop',
    coverage='contiguous_until_resolution', nonentries='not_scored_null')
SEMANTICS_SHA256 = digest(SEMANTICS)
LIMITATIONS = [
    'Caller must freeze proposal/policy before revealing prices; hashes do not prove chronology.',
    'Supplied historical prices/stream labels are not independently authenticated venue receipts.',
    'Fractional modeled quantity, not lot-rounded executable size; gaps can exceed risk budget.',
    'Costs are the frozen flat assumption, not measured funding, impact or fill quality.',
    'No capital competition, scale-outs, trailing stops, re-anchoring or separate thesis exit.',
    'This scorer validates neither interpretation quality nor profitability of an agent policy.',
]


class _Unavailable(ValueError):
    pass


def _row(rows, i, expected, observed):
    if i >= len(rows):
        raise _Unavailable('missing_minute')
    row = rows[i]
    try:
        if not isinstance(row, dict):
            raise ValueError('minute object required')
        at = clock(row['open_time'])
        if at != at.floor('min') or not number(row['open']):
            raise ValueError('minute clock/open')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise _Unavailable('invalid_minute') from exc
    if at != expected:
        raise _Unavailable('coverage_gap' if at > expected else 'minute_order')
    # Only consumed fields enter the outcome binding. No future HLC at an open exit.
    observed.append(dict(open_time=at.isoformat(), open=row['open']))
    return row


def _exit(rows, geometry, deadline, observed):
    at = clock(geometry['fill_at'])
    stop, target = geometry['stop'], geometry['destination']
    i = 0
    while True:
        row = _row(rows, i, at, observed)
        op = row['open']
        if i == 0 and op != geometry['fill_price']:
            raise _Unavailable('entry_open_mismatch')
        # OPEN events have known precedence over any extremes later in that minute.
        if op <= stop:
            return op, 'stop', at, True, False, op < stop
        if op >= target:
            return target, 'target', at, True, False, False
        if at == deadline:
            return op, 'deadline', at, True, False, False
        if not prices(row):
            raise _Unavailable('invalid_minute')
        observed[-1].update({k: row[k] for k in ('high', 'low', 'close', 'volume')})
        if row['low'] <= stop:
            return stop, 'stop', at, False, row['high'] >= target, False
        if row['high'] >= target:
            return target, 'target', at, False, False, False
        at += MINUTE
        i += 1


def _economics(geometry, policy, deadline, exit_event):
    price, reason, at, open_exit, ambiguous, gap = exit_event
    entry, stop, target, qty, cost = (Decimal(str(geometry[k])) for k in (
        'fill_price', 'stop', 'destination', 'quantity_upper_bound', 'cost_per_unit'))
    gross = (Decimal(str(price)) - entry) * qty
    costs = cost * qty
    modeled_risk = (entry - stop + cost) * qty
    values = dict(quantity=qty, entry_price=entry, stop_price=stop, target_price=target,
        exit_price=Decimal(str(price)), entry_notional=entry * qty, gross_pnl=gross,
        modeled_costs=costs, net_pnl=gross - costs, modeled_loss_at_stop=modeled_risk,
        net_r=(gross - costs) / modeled_risk)
    out = {k: float(v) for k, v in values.items()}
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError('unrepresentable outcome arithmetic')
    out.update(entry_time=geometry['fill_at'], deadline=deadline.isoformat(),
        horizon_minutes=policy['horizon_minutes'], exit_reason=reason,
        exit_time=at.isoformat() if open_exit else None,
        exit_bar_open=at.isoformat(),
        exit_observed_at=(at if open_exit else at + MINUTE).isoformat(),
        exit_timing='open' if open_exit else 'intrabar_unknown',
        ambiguous_bar=ambiguous, stop_gap=gap)
    return out


def score_structure_outcome(source_request: dict, packet: dict, raw_response: str,
                            policy, execution: dict, future: dict) -> dict:
    """Score one eligible hypothetical LC long; all nonentries retain null PnL.

    ``future`` is {instrument, data_stream_id, minutes}. Rows start at the
    pre-entry checker's fill open and are consecutive one-minute OHLCV records.
    At a resolved OPEN (including the deadline) only open_time/open are read.
    Missing data before resolution is unknown; data after resolution is ignored.
    Source reconstruction faults raise ValueError (controller errors).

    consumed_future_sha256 binds only the validated observation fields. On a
    data failure it excludes rejected HLCV/rows, so it is NOT a fingerprint of
    all inspected bad input. A campaign must separately retain its raw archive
    and failure evidence; distinct corrupt sources can share this prefix hash.

    Holding limit is fill + policy.horizon_minutes, unlike the old conditional
    scorer's decision-relative deadline. Both comparators must use this same
    version for a structural-menu study. No zero-PnL rejection credit or book
    accounting is inferred here; future protocol owns those distinct states.
    """
    checked = validate_structure_proposal(source_request, packet, raw_response, policy)
    preentry = check_structure_preentry(source_request, packet, raw_response, policy, execution)
    proposal = checked['proposal']
    bindings = dict(packet_sha256=packet['seal'], policy_sha256=checked['policy_sha256'],
        raw_response_sha256=(sha256(raw_response.encode('utf-8', errors='surrogatepass')).hexdigest()
                             if isinstance(raw_response, str) else None),
        execution_sha256=None, consumed_future_sha256=None, semantics_sha256=SEMANTICS_SHA256)
    result = dict(version=VERSION, case_id=packet['case_id'],
        thesis=proposal['thesis'] if proposal else None,
        decision=proposal['decision'] if proposal else None,
        status='not_scored', reason=preentry['reason'], preentry=preentry,
        outcome={'net_pnl': None}, bindings=bindings, limitations=list(LIMITATIONS),
        execution_authorized=False)
    if preentry['status'] != 'eligible_hypothetical':
        return result
    bindings['execution_sha256'] = digest(execution)
    geometry = preentry['geometry']
    invalidation = packet['levels'][proposal['plan']['invalidation_level_id']]['price']
    if Decimal(str(invalidation)) != Decimal(str(geometry['stop'])):
        result['reason'] = 'separate_postentry_invalidation'
        return result
    try:
        deadline = clock(geometry['fill_at']) + pd.Timedelta(minutes=policy['horizon_minutes'])
    except (ValueError, OverflowError):
        result['reason'] = 'invalid_horizon'
        return result
    if not isinstance(future, dict) or set(future) != FUTURE_KEYS or not isinstance(future['minutes'], list):
        result.update(status='data_unavailable', reason='future_shape')
        return result
    if any(future[k] != packet[k] for k in ('instrument', 'data_stream_id')):
        result.update(status='data_unavailable', reason='stream_mismatch')
        return result
    observed = []
    try:
        event = _exit(future['minutes'], geometry, deadline, observed)
        outcome = _economics(geometry, policy, deadline, event)
    except _Unavailable as exc:
        result.update(status='data_unavailable', reason=str(exc))
    except (ValueError, OverflowError, ArithmeticError):
        result.update(status='not_scored', reason='unrepresentable_outcome')
    else:
        result.update(status='scored', reason='resolved', outcome=outcome)
    bindings['consumed_future_sha256'] = digest(dict(
        instrument=future['instrument'], data_stream_id=future['data_stream_id'], minutes=observed))
    return result
