"""Frozen long-entry policies for research; not an agent or portfolio executor.

The resolver reads only information observable by `as_of` and stops consuming
prices when the policy resolves. The separate scorer intentionally reads future
outcomes. Callers must freeze evidence/plan provenance before either is used.
"""
import math
from numbers import Real

import pandas as pd

from scripts.research.entry_case_outcome import score_case


MINUTE = pd.Timedelta(minutes=1)


def _clock(value):
    try:
        at = pd.Timestamp(value)
        if pd.isna(at) or at.tzinfo is None or at.value % MINUTE.value:
            raise ValueError('aware exact-minute clock required')
        return at.tz_convert('UTC')
    except (TypeError, OverflowError) as exc:
        raise ValueError('invalid clock') from exc


def _positive(value):
    return (not isinstance(value, bool) and isinstance(value, Real)
            and math.isfinite(value) and value > 0)


def resolve_entry(bars, *, decision_time, as_of, action, stop, entry_expiry,
                  exit_deadline, level=None, processing_seconds=0, routing_seconds=0):
    """Resolve immutable enter/wait_close_above/reject at an observable cutoff.

    Delays are nonnegative integer assumptions, not measured runtime. Waiting
    confirms only on fully closed post-arm candles. Stop cancellation monitors
    completed candles from decision, including pre-arm and routing time.
    A ready open is read without its subsequent high/low/close.
    """
    decision, cutoff, expiry, deadline = map(_clock, (
        decision_time, as_of, entry_expiry, exit_deadline))
    if action not in ('enter', 'wait_close_above', 'reject'):
        raise ValueError('invalid action')
    if not _positive(stop) or (action == 'wait_close_above' and not _positive(level)):
        raise ValueError('finite positive stop and wait level required')
    if action != 'wait_close_above' and level is not None:
        raise ValueError('level only applies to waiting')
    if any(type(v) is not int or v < 0 for v in (processing_seconds, routing_seconds)):
        raise ValueError('nonnegative integer delays required')
    try:
        arm = (decision + pd.Timedelta(seconds=processing_seconds)).ceil('min')
        route = pd.Timedelta(seconds=routing_seconds)
        fill = (arm + route).ceil('min') if action == 'enter' else None
    except (ValueError, OverflowError) as exc:
        raise ValueError('unrepresentable delay') from exc
    confirmation = None

    def result(status, at, reason=None, price=None):
        return dict(status=status, resolved_at=at.isoformat(), reason=reason,
                    arm_at=arm.isoformat(), confirmation_at=(confirmation.isoformat()
                    if confirmation is not None else None),
                    entry_time=at.isoformat() if status == 'entry_ready' else None,
                    entry_price=price, exit_deadline=deadline.isoformat(),
                    execution_certified=False)

    if expiry <= decision or deadline <= decision or expiry > deadline:
        return result('invalid_plan', decision, 'invalid_expiry_or_deadline')
    if cutoff < decision:
        return result('pending', cutoff, 'decision_not_yet_available')
    if action == 'reject':
        return result('rejected', decision)
    if (not isinstance(bars, pd.DataFrame) or not isinstance(bars.index, pd.DatetimeIndex)
            or bars.index.tz is None
            or not bars.columns.is_unique):
        return result('data_unavailable', decision, 'invalid_source_index_or_columns')

    def prices(at, names):
        try:
            values = [bars.at[at, name] for name in names]
            if not all(_positive(v) for v in values):
                return None
            return [float(v) for v in values]
        except (KeyError, TypeError, ValueError):
            return None

    at = decision
    while at <= cutoff:
        if at >= expiry:
            return result('expired', expiry, 'entry_expiry')
        if fill is not None and at == fill:
            op = prices(at, ('open',))
            if op is None:
                return result('data_unavailable', at, 'missing_or_invalid_entry_open')
            if op[0] <= stop:
                return result('cancelled', at, 'entry_open_at_or_below_stop')
            return result('entry_ready', at, price=op[0])
        available = at + MINUTE
        if available > cutoff:
            return result('pending', cutoff, 'awaiting_observation')
        row = prices(at, ('open', 'high', 'low', 'close'))
        if row is None:
            return result('data_unavailable', available, 'missing_or_invalid_completed_bar')
        op, hi, lo, cl = row
        if lo > min(op, cl) or hi < max(op, cl):
            return result('data_unavailable', available, 'invalid_ohlc_envelope')
        if lo <= stop:
            return result('cancelled', available, 'pending_stop_touched')
        if (action == 'wait_close_above' and fill is None and at >= arm
                and available < expiry and cl > level):
            confirmation = available
            try:
                fill = (available + route).ceil('min')
            except (ValueError, OverflowError) as exc:
                raise ValueError('unrepresentable routing time') from exc
        at = available
    return result('pending', cutoff, 'awaiting_observation')


def score_conditional(bars, *, notional=50000., cost_bps=12., **plan):
    """Offline future scorer. NOT suitable for the outcome-hidden agent packet.

    Resolved nonentries have zero exposure; pending/invalid/missing remain null.
    Outcomes use the original deadline, not a new holding period after waiting.
    """
    if (not _positive(notional) or isinstance(cost_bps, bool)
            or not isinstance(cost_bps, Real) or not math.isfinite(cost_bps) or cost_bps < 0):
        raise ValueError('invalid economics')
    resolution = resolve_entry(bars, **plan)
    status = resolution['status']
    if status != 'entry_ready':
        pnl = 0. if status in ('rejected', 'cancelled', 'expired') else None
        return dict(resolution=resolution, outcome=dict(status=status, net_pnl=pnl,
                    starting_equity=None, execution_certified=False))
    entry, deadline = map(_clock, (resolution['entry_time'], resolution['exit_deadline']))
    try:
        # The old scorer validates its whole input index. Limit that input to
        # its authorized outcome window so later metadata cannot change it.
        window = bars.loc[(bars.index >= entry) & (bars.index <= deadline)]
        outcome = score_case(window, decision_time=entry, step_minutes=1,
                             horizon_bars=int((deadline-entry)/MINUTE), stop=plan['stop'],
                             notional=notional, cost_bps=cost_bps)
    except ValueError as exc:
        outcome = dict(status='data_unavailable', net_pnl=None, reason=str(exc),
                       starting_equity=None, execution_certified=False)
    return dict(resolution=resolution, outcome=outcome)
