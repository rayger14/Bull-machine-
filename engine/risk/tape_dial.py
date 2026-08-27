"""Tape-state sizing dial + flat-notional sizing — the validated sizing package.

Evidence (docs/knowledge/sizing_package_verdict_2026_08_24.md, June autopsy
2026-08-25): with entries/exits frozen (3,512 identical positions, 2020-2024),
flat-notional + this dial = +14.2% PnL, MaxDD −16.5%→−13.9%, 5/5 years
improved (2022 bear included), wick_trap +19%. The dial alone absorbed ~47%
of June-2026's live damage in the stand-down autopsy.

STRICT CAUSALITY: the multiplier for trading day D is computed from COMPLETED
daily closes through D-1 only. Live can never know today's close; this module
enforces the same information set everywhere so backtest == live.

The dial is a SIZING multiplier, never a filter: 0.75x still trades (junk-book
data collection intact; Standing Orders honored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pre-registered constants (round numbers, never tuned — see verdict doc)
IMPULSE_R3 = 0.10        # +10% over 3 days = ignition
IMPULSE_WINDOW_D = 4     # ignition within the last 3 completed days (rolling 4)
BASE_RANGE_D = 12        # tight-base lookback
BASE_RANGE_PCT = 0.08    # <=8% close-to-close range = base
MULT_IMPULSE = 1.25
MULT_BASE = 1.00
MULT_DEAD = 0.75
FLAT_NOTIONAL_DIVISOR = 0.025   # constant stop divisor ~= median stop width


def compute_tape_dial(daily_closes: pd.Series) -> pd.Series:
    """Multiplier per trading day from COMPLETED daily closes.

    Input: daily close series (the close indexed at day D is D's completed
    close). Output: multiplier indexed at day D computed from closes through
    D-1 (shift(1) applied INSIDE — callers pass raw closes and simply look up
    their trading day; no caller-side shifting, no way to hold it wrong).
    """
    px = daily_closes.dropna()
    if len(px) == 0:
        return pd.Series(dtype=float)
    r3 = px.pct_change(3)
    recent_impulse = r3.rolling(IMPULSE_WINDOW_D).max() >= IMPULSE_R3
    hi = px.rolling(BASE_RANGE_D).max()
    lo = px.rolling(BASE_RANGE_D).min()
    in_base = (hi / lo - 1) <= BASE_RANGE_PCT
    mult = pd.Series(
        np.select([recent_impulse, in_base], [MULT_IMPULSE, MULT_BASE],
                  default=MULT_DEAD),
        index=px.index, name='tape_dial')
    return mult.shift(1).fillna(1.0)   # strict t-1 causality


def dial_for_day(daily_closes: pd.Series, day: pd.Timestamp) -> float:
    """Convenience: the multiplier for one trading day (1.0 if unknown)."""
    m = compute_tape_dial(daily_closes)
    if len(m) == 0:
        return 1.0
    day = pd.Timestamp(day)
    if day.tzinfo is not None and m.index.tz is None:
        day = day.tz_localize(None)
    elif day.tzinfo is None and m.index.tz is not None:
        day = day.tz_localize(m.index.tz)
    v = m.asof(day.normalize())
    return float(v) if v == v else 1.0
