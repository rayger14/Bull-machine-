"""Independent long-only case diagnostics; no book, live fills or agent authority."""
import math
from numbers import Real

import numpy as np
import pandas as pd


def score_case(bars, *, decision_time, step_minutes, horizon_bars, stop,
               notional=50000., cost_bps=12.):
    """Benchmark 2R bracket; full-horizon excursions continue AFTER bracket exit.

    Known OPEN gaps precede intrabar extremes. Otherwise a bar touching both
    barriers is ambiguous and conservatively stop-first. Deadline OPEN only.
    Starting equity, funding, impact and inference latency are unspecified.
    """
    for value in (step_minutes, horizon_bars):
        if type(value) is not int or value <= 0:
            raise ValueError('positive integer grid/horizon required')
    for value in (stop, notional, cost_bps):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError('finite numeric parameters required')
    if stop <= 0 or notional <= 0 or cost_bps < 0:
        raise ValueError('invalid economics')
    at=pd.Timestamp(decision_time)
    if at.tzinfo is None or not isinstance(bars.index,pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError('timezone-aware clocks required')
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing or bars.index.hasnans:
        raise ValueError('invalid bar index')
    at=at.tz_convert('UTC')
    step=pd.Timedelta(minutes=step_minutes)
    if at.value % step.value:
        raise ValueError('decision must align to grid')
    wanted=pd.date_range(at,periods=horizon_bars+1,freq=step)
    if not wanted.isin(bars.index).all():
        raise ValueError('missing bars or full deadline tail')
    window=bars.loc[(bars.index>=wanted[0]) & (bars.index<=wanted[-1])]
    if not window.index.equals(wanted):
        raise ValueError('unexpected bars inside grid')
    try:
        # Never consume the deadline's high/low/close, even for validation.
        prices=window.iloc[:-1][['open','high','low','close']].to_numpy(dtype=float)
        deadline=float(window.iloc[-1]['open'])
    except (TypeError,ValueError,KeyError) as exc:
        raise ValueError('numeric OHLC and deadline open required') from exc
    if (not np.isfinite(prices).all() or (prices<=0).any() or not math.isfinite(deadline) or deadline<=0
            or (prices[:,2]>prices[:,[0,3]].min(axis=1)).any()
            or (prices[:,1]<prices[:,[0,3]].max(axis=1)).any()):
        raise ValueError('invalid OHLC')
    entry=float(prices[0,0])
    out=dict(status='valid',entry_time=at.isoformat(),entry_price=entry,stop_price=float(stop),
             horizon_return=(deadline-entry)/entry,
             mfe=max(0.,float(prices[:,1].max()-entry))/entry,
             mae=min(0.,float(prices[:,2].min()-entry))/entry,
             excursion_scope='full horizon including after hypothetical exit',
             starting_equity=None,execution_certified=False)
    if stop>=entry:
        return dict(out,status='invalid_plan',net_pnl=None,gross_pnl=None,fees=None,
                    initial_risk=None,target_price=None)
    target=entry+2*(entry-stop)
    qty=notional/entry
    initial_risk=(entry-stop)*qty
    if not all(math.isfinite(v) for v in (target,qty,initial_risk)):
        raise ValueError('unrepresentable risk arithmetic')
    exit_price,reason,ambiguous,exit_i=deadline,'deadline',False,horizon_bars
    for i,(op,hi,lo,cl) in enumerate(prices):
        if op<=stop:
            exit_price,reason,exit_i=float(op),'stop',i
            break
        if op>=target:
            exit_price,reason,exit_i=float(target),'target',i
            break
        if lo<=stop:
            exit_price,reason,ambiguous,exit_i=float(stop),'stop',bool(hi>=target),i
            break
        if hi>=target:
            exit_price,reason,exit_i=float(target),'target',i
            break
    gross=(exit_price-entry)*qty
    fees=notional*cost_bps/10000
    if not all(math.isfinite(v) for v in (gross,fees,gross-fees)):
        raise ValueError('unrepresentable PnL arithmetic')
    return dict(out,target_price=target,quantity=qty,initial_risk=initial_risk,
                exit_price=exit_price,exit_reason=reason,exit_time=wanted[exit_i].isoformat(),
                ambiguous_bar=ambiguous,gross_pnl=gross,fees=fees,net_pnl=gross-fees)
