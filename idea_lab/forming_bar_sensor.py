"""
FORMING-BAR SENSOR  (wyckoff_audit add.66 — Deliverable 2; STUDY ONLY; nothing ships)
=====================================================================================
WI trades the FORMING weekly candle (mid-week rejection reads); our door's entry path
reads only COMPLETED bars. This module builds the cheap, causal week-to-date (WTD)
aggregate sensor and the ONE pre-registered forming-bar door variant so we can measure
whether reading forming-week evidence shifts entry timing / outcomes.

WHERE THE COMPLETED-BAR DEPENDENCY ACTUALLY LIVES (inventory, verified in add.66):
  The trend-continuation door's ENTRY machinery (eye break-confirmation + retest-hold +
  proximity) runs on the DAILY eye (compute_eye_features on the daily frame): every read
  is a COMPLETED DAILY bar, already intra-week granular. The only WEEKLY-completed-bar
  object is the struct_range's N=5 weekly pivots, and those feed ONLY (a) the TP anchor
  struct_range_high and (b) the rarely-binding R0 `struct_range_state != broken_down`
  veto — NOT entry timing. So the one place a "forming-week" read could move an ENTRY is
  the retest-HOLD check T4 (close>=break_level), which today requires a completed daily
  close back above the reclaimed level.

THE PRE-REGISTERED FORMING-BAR VARIANT (single; fixed before measuring; NO grid):
  Door = the FROZEN TrendContinuationDoor (M2-broad). Keep T1,T2,T3,T5, the stop, the
  struct exits, dedup, R0 — all identical. Replace ONLY the retest-hold T4:
     status-quo T4 :  close[i] >= break_level
     forming    T4':  close[i] >= break_level
                      OR ( wtd_high[i] >= break_level                     # week reclaimed
                           AND close[i] >= break_level - RTZ_ATR*ATR )    # still in-zone
  wtd_high[i] = running max of daily highs within the CURRENT ISO week, bars <= i only
  (causal by construction). Interpretation: accept the retest-hold when the FORMING week
  has already tagged/reclaimed the break_level, even if today's daily close sits marginally
  (<= RTZ_ATR*ATR) below it — the "mid-week rejection that holds" WI reads live.
  This can only ADD fires or move a fire EARLIER (T4' is strictly weaker than T4); it can
  never remove one. That asymmetry is the point: does forming-week evidence pull entries
  earlier, and does that help or hurt?

CAUSALITY: wtd_high[i] uses only daily bars in [week_start .. i]. No future data. A
3-point truncation check (below) confirms wtd_high computed on truncated history matches
the full-history values up to the truncation point (no repaint).
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unified_archetype_v2 import (
    UnifiedArchetypeV2, DEDUP_K, RTZ_ATR, M2_SOS_WIN, LPS_LOOKBACK,
)
from engine.features.eye_state import MODEL_FORMING, CONFIRMED_BREAK, IN_RANGE, MANIPULATION
from trend_continuation_door import _SingleDoor


def week_to_date_high(index: pd.DatetimeIndex, highs: np.ndarray) -> np.ndarray:
    """Causal running max of daily highs within each ISO calendar week.
    wtd_high[i] = max(high[j] for j in current-week bars with j <= i)."""
    idx = pd.DatetimeIndex(index)
    iso = idx.isocalendar()
    # unique week id = year*100 + week (monotone within a year; resets across years but
    # the running-max resets on any change of week id, which is what we want)
    wid = (iso["year"].to_numpy(int) * 100 + iso["week"].to_numpy(int))
    n = len(highs)
    out = np.empty(n, dtype=float)
    run = -np.inf
    prev = None
    for i in range(n):
        if wid[i] != prev:
            run = highs[i]
            prev = wid[i]
        else:
            if highs[i] > run:
                run = highs[i]
        out[i] = run
    return out


class FormingRetestDoor(_SingleDoor):
    """FROZEN M2-broad door with ONLY the retest-hold T4 relaxed to accept forming-week
    evidence (wtd_high has reclaimed break_level and close is still within RTZ of it)."""

    def __init__(self, df, sr, bj, eye, variant="struct", conviction=False):
        super().__init__(df, sr, bj, eye, variant=variant, conviction=conviction)
        self._wtd_high = week_to_date_high(df.index, df["high"].to_numpy(float))

    def _door(self, i):
        u = self._u
        atr = u.atr[i]
        if not np.isfinite(atr) or atr <= 0:
            return None
        if not (u.eye_dir[i] == "bull"):
            return None
        if u.eye_state[i] not in (MODEL_FORMING, IN_RANGE, MANIPULATION):
            return None
        lo = max(0, i - M2_SOS_WIN)
        break_level = np.nan
        for c in range(i - 1, lo - 1, -1):
            if u.eye_state[c] == CONFIRMED_BREAK and u.eye_dir[c] == "bull":
                break_level = u.range_upper[c]
                break
        if not np.isfinite(break_level):
            return None
        # ---- T4' : forming-week retest-hold ----
        held = (u.closes[i] >= break_level)
        if not held:
            wtd = self._wtd_high[i]
            if (wtd >= break_level) and (u.closes[i] >= break_level - RTZ_ATR * atr):
                held = True
        if not held:
            return None
        # ---- T5 proximity (unchanged) ----
        if not (u.lows[i] <= break_level + RTZ_ATR * atr):
            return None
        lps_lo = max(0, i - LPS_LOOKBACK + 1)
        created_low = np.nanmin(u.lows[lps_lo:i + 1])
        return {"pathway": "M2", "created_low": created_low, "sweep_bar": None,
                "aligned_forming": bool(u.eye_state[i] == MODEL_FORMING),
                "break_level": float(break_level)}


def truncation_check_wtd(index, highs, points=(0.5, 0.7, 0.9)):
    """3-point no-repaint check: wtd_high on truncated history must equal full-history
    wtd_high up to the truncation point."""
    full = week_to_date_high(index, highs)
    n = len(highs); mism = 0; checked = 0
    for frac in points:
        k = int(n * frac)
        trunc = week_to_date_high(index[:k], highs[:k])
        d = np.abs(trunc - full[:k])
        checked += k
        mism += int((d > 1e-9).sum())
    return checked, mism
