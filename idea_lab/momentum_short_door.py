"""
THE MOMENTUM-SHORT DOOR  (breakdown-retest; STANDALONE; STUDY ONLY)
==================================================================
wyckoff_audit add.68 -- EXACT GEOMETRIC MIRROR of the validated add.48 LONG
trend-continuation door (trend_continuation_door.py / UnifiedArchetypeV2._m2,
m2_mode='broad'), inverted term-by-term. NO NEW PARAMETERS -- every magnitude is
imported from unified_archetype_v2.py, none redefined. NOTHING SHIPS.

Faithful inversion of the pre-registered spec (idea_lab/momentum_short_PREREGISTRATION.txt,
committed c94e3b0 BEFORE any outcome was measured). LONG term -> SHORT term:

  R0 REGIME (permission; mirror of long "not-markdown"):
     long : stables_rot_rising==0 AND (close> ema200 OR state != BROKEN_DOWN)
     SHORT: stables_rot_rising==0 AND (close< ema200 OR state != BROKEN_UP)

  TRIGGER + ENTRY (breakdown-retest = mirror of breakout-retest):
     T1  eye_dir[i]=='bear'
     T2  eye_state[i] in {MODEL_FORMING(inert), IN_RANGE, MANIPULATION}   (non-extension)
     T3  a bear CONFIRMED_BREAK within trailing M2_SOS_WIN bars
         -> break_level = range_LOWER_1d at that break bar (mirror of range_upper)
     T4  close[i] <= break_level          (the down-break was NOT given back = HELD below)
     T5  high[i] >= break_level - RTZ_ATR*ATR   (pulled back UP into the retest zone)
     ENTRY SHORT at close[i].

  STOP: created_high = max(high) over trailing LPS_LOOKBACK bars (the pullback leg);
        stop = created_high + STOP_BUF_ATR*ATR14(entry).   (mirror of created_low - buf)

  MANAGE 'struct' (HEADLINE): TP1 40% at struct_range_LOW (else swing_low_50, else
        entry-1R; must clear <= entry - MIN_TP1_R*R) -> stop to BREAKEVEN -> runner 60%
        to the DOWNSIDE measured move struct_range_low - (range_high-range_low), floored
        (capped further-down) at entry-2R via tt = min(entry-2R, measured).
  MANAGE 'naive': plain 1R/2R/3R equal thirds to the downside.
  max_hold = MAX_HOLD.  HEADLINE = struct/flat (rmult=1.0, conviction OFF).

  DEDUP: no re-entry within DEDUP_K bars of the last entry.

CAUSALITY: every read at bar i uses only arrays at indices <= i (eye lags 1 bar via the
  broadcast; entry is this bar's close). No future data anywhere. Params frozen at the
  long door's values (imported, none redefined).
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester import EntryPlan
from structural_range import BROKEN_UP
from engine.features.eye_state import (
    MODEL_FORMING, CONFIRMED_BREAK, IN_RANGE, MANIPULATION,
)
from unified_archetype_v2 import (
    UnifiedArchetypeV2, DEDUP_K, M2_SOS_WIN, LPS_LOOKBACK, RTZ_ATR,
    STOP_BUF_ATR, MAX_HOLD, MIN_TP1_R,
)


class MomentumShortDoor(UnifiedArchetypeV2):
    """The breakdown-retest SHORT door ONLY -- exact mirror of the broad M2 long door.
    Reuses the vetted v2 __init__ arrays, _conviction; overrides the regime gate, the
    door, and the plan builder with their term-by-term SHORT inversions."""

    def __init__(self, df, sr, bj, eye, variant="struct", conviction=False):
        # reuse the vetted long init (m2_mode broad = the mirrored source); adds the
        # short-side sensors the long init did not read.
        super().__init__(df, sr, bj, eye, variant=variant, conviction=conviction,
                         m2_mode="broad")
        self.range_lower = eye["range_lower_1d"].to_numpy(float)   # mirror of range_upper
        self.swl50 = df["swing_low_50"].to_numpy(float)            # mirror of swing_high_50
        self.armed_log = []   # T5-diagnostic: appended by t5_scan (NOT the trade loop)

    # ---------- SHORT regime gate (mirror of _regime_ok) ----------
    def _regime_ok(self, i):
        not_markup = (self.closes[i] < self.ema200[i]) or (self.state[i] != BROKEN_UP)
        return bool(not_markup and self.stables[i] == 0)

    # ---------- SHORT door (mirror of _m2 broad) ----------
    def _door_short(self, i, want_stage=False):
        """Returns the short signal dict, or None. If want_stage, returns
        (sig_or_None, stage) where stage is the highest T-gate reached -- used only by
        the causal T5 diagnostic to separate 'armed (T1-T4)' from 'fired (T5)'."""
        atr = self.atr[i]
        if not np.isfinite(atr) or atr <= 0:
            return (None, 0) if want_stage else None
        # T1: standing bias bear (set only by a confirmed DOWN-break)
        if not (self.eye_dir[i] == "bear"):
            return (None, 0) if want_stage else None
        # T2: non-extension state (pullback, NOT the CONFIRMED_BREAK/TRENDING extension)
        if self.eye_state[i] not in (MODEL_FORMING, IN_RANGE, MANIPULATION):
            return (None, 0) if want_stage else None
        # T3: a bear CONFIRMED_BREAK within the window (exclude entry bar); break_level
        #     = range_lower at that break bar (mirror of range_upper)
        lo = max(0, i - M2_SOS_WIN)
        break_level = np.nan
        for c in range(i - 1, lo - 1, -1):
            if self.eye_state[c] == CONFIRMED_BREAK and self.eye_dir[c] == "bear":
                break_level = self.range_lower[c]
                break
        if not np.isfinite(break_level):
            return (None, 2) if want_stage else None
        # T4: the down-break was NOT given back (HELD below)
        if not (self.closes[i] <= break_level):
            return (None, 3) if want_stage else None
        # T5: pulled back UP into the retest zone (mirror of low <= break+RTZ)
        t5 = self.highs[i] >= break_level - RTZ_ATR * atr
        if not t5:
            return (None, 4) if want_stage else None   # ARMED (T1-T4) but retest absent
        lps_hi = max(0, i - LPS_LOOKBACK + 1)
        created_high = np.nanmax(self.highs[lps_hi:i + 1])
        sig = {"pathway": "S2", "created_high": created_high,
               "break_level": float(break_level)}
        return (sig, 5) if want_stage else sig

    # ---------- SHORT plan builder (mirror of _plan) ----------
    def _plan_short(self, i, sig):
        atr = self.atr[i]
        entry_raw = self.closes[i]
        created_high = sig["created_high"]
        if not np.isfinite(created_high):
            return None
        stop = created_high + STOP_BUF_ATR * atr
        R = stop - entry_raw                     # short R = stop above entry
        if R <= 0:
            return None
        time_present, bojan_present, rmult = self._conviction(i)
        rlow = self.rlow[i]
        rhigh = self.rhigh[i]
        swl = self.swl50[i]

        self.entries_log.append({
            "entry_time": self.index[i], "pathway": sig["pathway"],
            "aligned_forming": bool(self.eye_state[i] == MODEL_FORMING),
            "below_ema200": bool(entry_raw < self.ema200[i]),
            "time_present": bool(time_present), "bojan_present": bool(bojan_present),
            "risk_mult": rmult, "R_price": float(R),
            "break_level": sig["break_level"],
        })

        if self.variant == "naive":
            targets = [(entry_raw - 1 * R, 1 / 3.0),
                       (entry_raw - 2 * R, 1 / 3.0),
                       (entry_raw - 3 * R, 1 / 3.0)]
            return EntryPlan(direction="short", stop=stop, targets=targets,
                             move_stop_to_after_first_tp=None, runner_target=None,
                             max_hold_bars=MAX_HOLD, risk_mult=rmult,
                             meta={"variant": "naive", "pathway": sig["pathway"],
                                   "_rmult": rmult})
        # struct geometry (mirror): TP1 at struct_range_LOW / swing_low_50 / entry-1R
        tp1 = None
        for cand in (rlow, swl):
            if np.isfinite(cand) and cand <= entry_raw - MIN_TP1_R * R:
                tp1 = cand
                break
        if tp1 is None:
            tp1 = entry_raw - 1 * R
        if np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow:
            measured = rlow - (rhigh - rlow)     # downside measured move
        else:
            measured = np.inf
        tt = min(entry_raw - 2 * R, measured)    # floored (capped further-down) at entry-2R
        return EntryPlan(direction="short", stop=stop, targets=[(tp1, 0.40)],
                         move_stop_to_after_first_tp=entry_raw, runner_target=tt,
                         max_hold_bars=MAX_HOLD, risk_mult=rmult,
                         meta={"variant": "struct", "pathway": sig["pathway"],
                               "tp1": float(tp1), "tt": float(tt), "_rmult": rmult})

    # ---------- callable entry_fn ----------
    def __call__(self, df, i):
        if i - self._last_entry_i < DEDUP_K:
            return None
        if not self._regime_ok(i):
            return None
        sig = self._door_short(i)
        if sig is None:
            return None
        plan = self._plan_short(i, sig)
        if plan is None:
            return None
        self._last_entry_i = i
        return plan


def t5_scan(door: MomentumShortDoor):
    """CAUSAL T5 fire-rate diagnostic. Independent single pass over EVERY bar (not the
    position loop), so it is not distorted by dedup / one-position-at-a-time holds.
    A bar is ARMED if regime_ok AND T1-T4 hold (a real prior down-break, held below).
    It FIRES if T5 (the up-retest into the zone) also holds. Returns a per-bar list
    [{time, armed, fired}] over armed bars only -- the honest markdown-resilience test:
    do down-breaks even GET a retest to short into?"""
    out = []
    for i in range(door.n):
        if not door._regime_ok(i):
            continue
        _, stage = door._door_short(i, want_stage=True)
        if stage >= 4:                          # T1-T4 satisfied = ARMED
            out.append({"time": door.index[i], "armed": True, "fired": stage == 5})
    return out
