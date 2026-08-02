"""Stables-rotation flag — shared batch/live computation (parity by construction).

Validated 2026-08-02 (wick_trap refusal rule: train PF 1.36->1.56 DD -15.1->-11.6;
holdout 1.26->1.51 DD -7.9->-6.1). rot_rising = stables dominance (real
USDT+USDC circulating / [stables + majors mcap approx]) rising over 3 days.
Supply models are slow deterministic approximations; the 3-day DELTA is what
matters and is dominated by real prices + real stables circulation.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def btc_supply(ts: pd.Timestamp) -> float:
    eras = [(pd.Timestamp('2016-07-09'), 15.75e6, 12.5 * 144),
            (pd.Timestamp('2020-05-11'), 18.375e6, 6.25 * 144),
            (pd.Timestamp('2024-04-19'), 19.6875e6, 3.125 * 144)]
    for start, base, rate in reversed(eras):
        if ts >= start:
            return base + (ts - start).days * rate
    return 15.75e6


def _supplies(idx: pd.DatetimeIndex):
    days = pd.Series(idx, index=idx)
    eth = np.clip(96e6 + (days - pd.Timestamp('2018-01-01')).dt.days * (24.5e6 / 1710), 96e6, 120.5e6)
    bnb = np.clip(200e6 - (days - pd.Timestamp('2019-01-01')).dt.days * (55e6 / 2700), 145e6, 200e6)
    xrp = np.clip(39e9 + (days - pd.Timestamp('2018-01-01')).dt.days * (18e9 / 3100), 39e9, 57e9)
    return days, eth, bnb, xrp


def stables_dominance(btc_close_d: pd.Series, eth_close_d: pd.Series,
                      bnb_close_d: pd.Series, xrp_close_d: pd.Series,
                      stables_circ_d: pd.Series) -> pd.Series:
    """Daily stables-dominance series from daily closes + real stables circ."""
    idx = btc_close_d.index
    days, eth_s, bnb_s, xrp_s = _supplies(idx)
    mcap = (btc_close_d * days.map(btc_supply)
            + eth_close_d.reindex(idx) * eth_s
            + bnb_close_d.reindex(idx) * bnb_s
            + xrp_close_d.reindex(idx) * xrp_s)
    stables = stables_circ_d.reindex(idx).ffill()
    return stables / (stables + mcap)


def rotation_rising(dom: pd.Series, days: int = 3) -> pd.Series:
    """The validated flag: dominance rising over the trailing `days`."""
    return (dom.diff(days) > 0).astype(float)
