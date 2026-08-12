"""
CAMPAIGN-v2 STRATEGY  (STUDY ONLY -- WI add.57 topology on the validated door)
==============================================================================
Wraps the VALIDATED trend-continuation door (UnifiedArchetypeV2._m2, m2_mode='broad';
add.48/54) -- entries and stops are UNCHANGED -- and adds WI's campaign management
(add.57 verbatim), pre-registered here BEFORE any outcome is measured.

INTERPRETATION REGISTER (every choice that required interpretation is flagged OURS):
  CAMPAIGN start   : the first door _m2 fire while flat with no live campaign.
  CAMPAIGN death   : (D1) eye_dir loses 'bull'  OR  (D2) a daily BODY CLOSE below the
                     campaign floor. [verbatim: "body-close breakdown / eye_dir loses bull"]
  campaign FLOOR   : running-highest CONFIRMED LPS low since campaign start, where an
                     "LPS" = a confirmed daily N=SWING_N(10) fractal swing low (the pivots
                     already in the sensor stack). [OURS: LPS proxy = the stack's fractal low;
                     WI's LPS is discretionary.] Monotone-up.
  ENTRIES 2-3      : subsequent door _m2 fires INSIDE the live campaign, only when FLAT
                     (re-entry after banking), at NEWLY CREATED HIGHER structure ==
                     break_level strictly ABOVE the campaign's last entry break_level.
                     [verbatim: "new/higher range forms or a new break's retest holds"].
                     Cap E=3 entries/campaign (verbatim "up to 3, sometimes more"; we cap 3).
  MANAGEMENT       : TP1 40% at structure (tp1 = struct_range_high / swing_high_50 / entry+1R,
                     clears >=0.5R) -> AFTER TP1 the stop trails under each newly-created LPS
                     (floor - 0.25*ATR, structure-EVENT trail, monotone-up; NOT the add.56
                     pivot-ATR trail) -> TP2 bank 50% at the NEXT structure target (measured
                     move = rhigh+(rhigh-rlow), floored to exceed both tp1 and entry+2R) leaving
                     a ~10% RUNNER -> runner rides to campaign death or the trailing stop.
                     [OURS: tp2 floored to exceed tp1; runner = 0.10 remainder.]
  NO fixed max_hold: campaign structure IS the time limit; sanity cap 400 daily bars (OURS).
  SIZING           : headline rmult=1.0 (struct/flat, boost-immune) -- identical to the v1
                     headline. Gann conviction sizing is a SEPARATE test (gann layer), not here.
  PYRAMIDING       : while-positioned pyramiding is a FLAGGED secondary WI variant. It needs a
                     multi-position engine (this engine is one-position/re-enter-when-flat) so
                     it is NOT implemented here -- reported as out-of-scope, not cheap.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from unified_archetype_v2 import (
    UnifiedArchetypeV2, DEDUP_K, STOP_BUF_ATR, MIN_TP1_R, MAX_HOLD,
)
from xasset_spx_port import fractal_swings, SWING_N
from campaign_backtester import CampaignEntryPlan, SANITY_CAP_BARS

TP1_FRAC = 0.40
TP2_FRAC = 0.50           # leaves a 0.10 runner (bank down to ~1/10, add.57)
E_MAX = 3                 # up to 3 sized entries per campaign (add.57)


class CampaignV2Door:
    """Campaign strategy over the trend-continuation door (m2_mode='broad', struct).
    Entries/stops are the door's; management is add.57. One position at a time
    (re-enter only when FLAT). Exposes observe()/entry()/campaign_floor/campaign_alive
    for run_campaign_backtest.

    An optional gann_exit callable can veto the runner (G-EXIT): gann_exit(i) -> bool
    means 'this bar is a Gann DANGER window'; combined with structure weakness (close
    below the campaign floor) the engine's death-exit fires. We fold G-EXIT into the
    campaign_alive flag when enabled (so no engine change is needed)."""

    def __init__(self, df, sr, bj, eye, gann_danger=None, enable_gexit=False):
        self._u = UnifiedArchetypeV2(df, sr, bj, eye, variant="struct",
                                     conviction=False, m2_mode="broad")
        self.closes = self._u.closes
        self.lows = self._u.lows
        self.atr = self._u.atr
        self.rhigh = self._u.rhigh
        self.rlow = self._u.rlow
        self.swh = self._u.swh50
        self.eye_dir = self._u.eye_dir
        self.index = df.index
        self.n = len(df)
        # confirmed daily fractal swing lows (LPS events), value placed at confirm bar, else ffill
        _, sl = fractal_swings(df, SWING_N)
        self.swing_low = sl                       # ffilled highest-recent confirmed low value
        # detect the confirm bars where a NEW confirmed low value appears
        self._new_low_bar = np.zeros(self.n, dtype=bool)
        prev = np.nan
        for k in range(self.n):
            v = sl[k]
            if np.isfinite(v) and (not np.isfinite(prev) or v != prev):
                self._new_low_bar[k] = True
            prev = v

        # campaign state
        self.campaign_active = False
        self.campaign_alive = False
        self.campaign_floor = np.nan
        self._camp_start_i = -1
        self._camp_last_break = -np.inf
        self._camp_entries = 0
        self._camp_id = 0
        self._last_entry_i = -10_000
        self._pending_entry_ord = 0
        self.entries_log = []

        self._gann_danger = gann_danger           # bool[n] or None
        self._enable_gexit = bool(enable_gexit)

    # ---- causal per-bar state update (birth handled in entry(); here: death + floor) ----
    def observe(self, i):
        if not self.campaign_active:
            return
        # rising floor: a NEW confirmed LPS low that is higher than the current floor
        if self._new_low_bar[i]:
            v = self.swing_low[i]
            if np.isfinite(v) and (not np.isfinite(self.campaign_floor) or v > self.campaign_floor):
                self.campaign_floor = float(v)
        # death tests (causal)
        d1 = (self.eye_dir[i] != "bull")                                   # lost bull
        d2 = (np.isfinite(self.campaign_floor) and self.closes[i] < self.campaign_floor)  # body break
        dead = d1 or d2
        # G-EXIT (optional): Gann danger window AND structure weakness (close below floor)
        if self._enable_gexit and self._gann_danger is not None and self._gann_danger[i]:
            if np.isfinite(self.campaign_floor) and self.closes[i] < self.campaign_floor:
                dead = True
        if dead:
            self.campaign_active = False
            self.campaign_alive = False
            self.campaign_floor = np.nan

    # ---- entry (called only when FLAT) ----
    def entry(self, df, i):
        u = self._u
        if i - self._last_entry_i < DEDUP_K:
            return None
        if not u._regime_ok(i):
            return None
        sig = u._m2(i)
        if sig is None:
            return None
        break_level = sig["break_level"]

        if self.campaign_active and self.campaign_alive:
            # entries 2-3: require NEWLY CREATED HIGHER structure + cap
            if self._camp_entries >= E_MAX:
                return None
            if not (np.isfinite(break_level) and break_level > self._camp_last_break):
                return None
            entry_ord = self._camp_entries + 1
        else:
            # start a new campaign (entry 1)
            self._camp_id += 1
            self.campaign_active = True
            self.campaign_alive = True
            self._camp_start_i = i
            self._camp_entries = 0
            self._camp_last_break = -np.inf
            self.campaign_floor = float(sig["created_low"]) if np.isfinite(sig["created_low"]) else np.nan
            entry_ord = 1

        plan = self._build_plan(i, sig, entry_ord)
        if plan is None:
            return None
        self._camp_entries += 1
        self._last_entry_i = i
        if np.isfinite(break_level):
            self._camp_last_break = max(self._camp_last_break, break_level)
        # floor initialised to entry LPS if not yet set
        if not np.isfinite(self.campaign_floor):
            self.campaign_floor = float(sig["created_low"])
        return plan

    def _build_plan(self, i, sig, entry_ord):
        atr = self.atr[i]
        entry_raw = self.closes[i]
        created_low = sig["created_low"]
        if not (np.isfinite(created_low) and np.isfinite(atr) and atr > 0):
            return None
        stop = created_low - STOP_BUF_ATR * atr
        R = entry_raw - stop
        if R <= 0:
            return None
        rhigh, rlow, swh = self.rhigh[i], self.rlow[i], self.swh[i]
        # TP1 = structure high (clears >=0.5R)
        tp1 = None
        for cand in (rhigh, swh):
            if np.isfinite(cand) and cand >= entry_raw + MIN_TP1_R * R:
                tp1 = cand
                break
        if tp1 is None:
            tp1 = entry_raw + 1 * R
        # TP2 = next structure target (measured move), floored to exceed tp1 and entry+2R
        measured = rhigh + (rhigh - rlow) if (np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow) else -np.inf
        tp2 = max(measured, entry_raw + 2 * R, tp1 + 0.5 * R)
        self.entries_log.append({
            "entry_time": self.index[i], "campaign_id": self._camp_id,
            "entry_ord": entry_ord, "break_level": float(break_level_of(sig)),
            "below_ema200": bool(entry_raw < self._u.ema200[i]),
            "R_price": float(R), "tp1": float(tp1), "tp2": float(tp2),
        })
        return CampaignEntryPlan(
            direction="long", stop0=stop,
            targets=[(tp1, TP1_FRAC), (tp2, TP2_FRAC)],
            atr_entry=float(atr), trail_buf_atr=STOP_BUF_ATR,
            sanity_cap_bars=SANITY_CAP_BARS, risk_mult=1.0,
            meta={"campaign_id": self._camp_id, "entry_ord": entry_ord,
                  "tp1": float(tp1), "tp2": float(tp2)})


def break_level_of(sig):
    bl = sig.get("break_level", np.nan)
    return bl if np.isfinite(bl) else np.nan
