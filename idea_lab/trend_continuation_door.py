"""
THE TREND-CONTINUATION DOOR  (price-only breakout-retest; STANDALONE; STUDY ONLY)
=================================================================================
wyckoff_audit add.48 -- formalization of the M2-broad door (add.45/47) into a
clean, self-contained, PRE-REGISTERED standalone archetype, run cross-asset with
identical params. NOTHING SHIPS. No production code/config/live/deploy is touched.

WHAT THIS DOOR IS  (grounded; the behavioral OPPOSITE of the dead spring)
------------------------------------------------------------------------
NOT the strict Wyckoff M2 (which needs a wyckoff-store phase label = 'C_accum'
and fired n=0 cross-asset). This is the PRICE-ONLY generalization = the classic
breakout-retest, "buy the pullback after the break":

    1. A bull structural up-break is CONFIRMED (2 consecutive daily closes above
       the live rolling range ceiling).  break_level = that ceiling.
    2. Price pulls back toward the break_level (old ceiling = new floor).
    3. Enter LONG when the pullback HOLDS: close back >= break_level, low dipped
       into the retest zone (<= break_level + RTZ_ATR*ATR), and price is no longer
       extending (eye_state is a non-extension pullback state, not TRENDING).

It BUYS STRENGTH after a break -- the opposite of the M1 dead-spring dip-buyer
(which buys capitulation below EMA200). It SELF-REGIME-FILTERS: no up-break => no
trade, so it structurally cannot fire deep in a bear.

PRICE-ONLY GUARANTEE (the add.48 cleanup vs the add.45 M2-broad):
  The eye state machine (engine/features/eye_state.py) is computed here from OHLCV
  ONLY on EVERY asset -- the wyckoff_phase_dir / wyckoff_sos columns are DROPPED
  before eye computation. Consequence: MODEL_FORMING (needs phase=='C_accum') and
  the m2_bias lift (needs C_accum + SOS) are INERT on all assets, INCLUDING BTC.
  So the door's eye_state filter {MODEL_FORMING, IN_RANGE, MANIPULATION} reduces to
  {IN_RANGE, MANIPULATION} on every asset, and eye_dir=='bull' is driven ONLY by a
  price CONFIRMED up-break. The BTC door is therefore identical in character to the
  SPX/NDX/GOLD door -- a genuine cross-asset apples-to-apples test. (This is a
  DELIBERATE difference from the add.45 BTC M2-broad flicker, which still had the
  wyckoff store live; the add.48 door is stricter/cleaner. Divergence is expected
  and flagged.)

VOLUME: the door path (eye range boundaries = BODY-close extremes; break; retest;
stop; struct TP) uses NO volume. Verified. (Yahoo index volume is unreliable ->
this matters: the door is volume-independent, so index-volume noise cannot touch it.)

-----------------------------------------------------------------------------
PRE-REGISTERED SPEC  (fixed BEFORE any outcome is measured; NO per-asset tuning,
                      NO grids, NO iterating against results. All magnitudes are
                      the BTC-tuned bar-unit inherits from add.45/unified v2.)
-----------------------------------------------------------------------------
REGIME PERMISSION (R0, inherited faithfully from unified v2 _regime_ok):
    stables_rot_rising==0 (always true off-crypto)  AND
    (close > ema_200  OR  struct_range_state != broken_down).
  (A second, weaker regime protection on top of the structural break requirement.)

TRIGGER + ENTRY (the breakout-retest; = M2-broad, reused verbatim from
                 UnifiedArchetypeV2._m2 with m2_mode='broad'):
    T1  eye_dir[i] == 'bull'   (standing bias set bull only by a confirmed up-break)
    T2  eye_state[i] in {MODEL_FORMING(inert), IN_RANGE, MANIPULATION}  (non-extension:
        NOT CONFIRMED_BREAK/TRENDING -> we enter the PULLBACK, never the extension)
    T3  a bull CONFIRMED_BREAK occurred within trailing M2_SOS_WIN bars ->
        break_level = range_upper at that break bar (else the setup does not exist)
    T4  close[i] >= break_level          (the up-break was NOT given back = HOLD)
    T5  low[i] <= break_level + RTZ_ATR*ATR   (pulled back INTO the retest zone)
    ENTRY at close[i].

STOP:  created_low = min(low) over trailing LPS_LOOKBACK bars (the pullback leg);
       stop = created_low - STOP_BUF_ATR*ATR14(entry).

MANAGE (two exit variants; identical entries+stops):
    'struct' (HEADLINE) = TP1 40% at struct_range_high (else swing_high_50, else
        entry+1R; must clear >= MIN_TP1_R) -> stop to BREAKEVEN -> runner 60% to the
        MEASURED MOVE struct_range_high + (high-low), floored at entry+2R.
    'naive' = plain 1R/2R/3R equal thirds.
    max_hold = MAX_HOLD bars.  HEADLINE = struct / flat (rmult=1.0, conviction OFF).

SIZE (conviction, reported but NOT the headline): base 1%; CONVICTION_MULT if
      (fib_time_confluence>0) OR (a Bojan-low zone active). The struct/flat headline
      has rmult=1.0 so it is immune to fib/bojan sensors.

DEDUP: no re-entry within DEDUP_K bars of the last entry.

VALIDITY WINDOW: T3's M2_SOS_WIN is the SOS lookback = the retest must form within
  M2_SOS_WIN bars of the up-break, else the setup expires (window expiry = skip).

FLAGGED PARAMS (OURS; identical across ALL assets; single pre-registered values):
  M2_SOS_WIN=360   LPS_LOOKBACK=48   RTZ_ATR=0.5   STOP_BUF_ATR=0.25
  DEDUP_K=3   CONVICTION_MULT=1.5   MAX_HOLD=168   HTF_N=5 (weekly struct range)
  eye: N_RANGE_1D=40, ACCEPT_CONSEC=2 (=break_confirm_bars), TREND_CONSEC=5.
  DAILY RESCALE CAVEAT (flagged): on daily exec these bar counts scale up
  (MAX_HOLD 168d ~ 8mo; M2_SOS_WIN 360d ~ 17mo). Identical across assets; the
  point is CROSS-ASSET CONSISTENCY of an unchanged model, not per-asset optimality.

BASELINE CONTRAST: the DEAD-SPRING door (M1 RTZ-spring, UnifiedArchetypeV2._m1)
  run standalone on the same assets/regimes, to confirm the trend-continuation door
  is genuinely the OPPOSITE character (above-EMA200, bear stand-down) and not a
  relabel of the falling-knife.

CAUSALITY: every read at bar i uses only arrays at indices <= i (eye lags 1 bar via
  the broadcast). Entry is this bar's close. No future data anywhere.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_range import build_structural_range
from bojan_detector import build_bojan
from engine.features.eye_state import compute_eye_features
from unified_archetype_v2 import (
    UnifiedArchetypeV2, DEDUP_K, BOJAN_W, HTF_N,
)
from xasset_spx_port import (
    build_base_sensors, reanchor_frame_weekly,
)

OHLCV = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Sensors: daily exec, weekly N=5 struct range, PRICE-ONLY eye (wyckoff dropped)
# ---------------------------------------------------------------------------
def build_daily_sensors(df_daily: pd.DataFrame):
    """Attach causal OHLCV sensors and build (df, sr, bj, eye) for a DAILY frame.
    eye is computed from OHLCV ONLY (price-only, identical across assets)."""
    df = build_base_sensors(df_daily)                     # ema200, atr14, swing, fib, stables=0
    reanch = reanchor_frame_weekly(df, HTF_N)             # daily -> weekly N=5 reanchor
    sr = build_structural_range(reanch)
    bj = build_bojan(reanch, sr, BOJAN_W)
    eye = compute_eye_features(df[OHLCV].copy())          # PRICE-ONLY (no wyckoff cols)
    return df, sr, bj, eye


def resample_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """1H OHLCV -> 1D calendar OHLCV (for BTC, to match the daily assets)."""
    o = df_1h["open"].resample("1D").first()
    h = df_1h["high"].resample("1D").max()
    l = df_1h["low"].resample("1D").min()
    c = df_1h["close"].resample("1D").last()
    v = df_1h["volume"].resample("1D").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return out.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Standalone single-door strategies (reuse the vetted v2 building blocks;
# NO R-CONFLICT confound -> clean per-door attribution)
# ---------------------------------------------------------------------------
class _SingleDoor:
    """Wrap UnifiedArchetypeV2 but fire exactly ONE door. Reuses _regime_ok,
    _m1/_m2, _plan verbatim so the logic is identical to the vetted v2."""

    def __init__(self, df, sr, bj, eye, variant="struct", conviction=False):
        self._u = UnifiedArchetypeV2(df, sr, bj, eye, variant=variant,
                                     conviction=conviction, m2_mode="broad")
        self.entries_log = self._u.entries_log

    def _door(self, i):
        raise NotImplementedError

    def __call__(self, df, i):
        u = self._u
        if i - u._last_entry_i < DEDUP_K:
            return None
        if not u._regime_ok(i):
            return None
        sig = self._door(i)
        if sig is None:
            return None
        plan = u._plan(i, sig)
        if plan is None:
            return None
        u._last_entry_i = i
        return plan


class TrendContinuationDoor(_SingleDoor):
    """The breakout-retest door ONLY (M2-broad, price-only)."""
    def _door(self, i):
        return self._u._m2(i)


class DeadSpringDoor(_SingleDoor):
    """The M1 RTZ-spring dip-buyer ONLY (the falling-knife baseline for contrast)."""
    def _door(self, i):
        return self._u._m1(i)
