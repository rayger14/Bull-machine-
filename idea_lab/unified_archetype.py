"""
THE UNIFIED ARCHETYPE  (LONG-ONLY, pre-registered, conjunctive; STUDY ONLY)

The campaign synthesis (wyckoff_audit addenda 27-42). Prior unified attempts
(PO3 x5, campaign v1/v2, integrated eye add.27) ALL ran on BROKEN sensors
(rolling springs mislocated 5-8 ATR, 0.4-day ranges, dead-center Bojan). The
sensor-rebuild arc (add.31-41) FIXED them: 1D N=5 structural range holds ~13.7d
(add.37), struct springs fire AT the drawn low by construction (add.34), Bojan
fires 97.4% at the range low (add.41), and the deployed TIME layer is net-positive
(add.38). This is the FIRST FAIR TEST of the unified "right place + right time"
entry as a single clean archetype that GENERATES ITS OWN TRADES — NOT a boost/
filter on the champion book (that shape was add.39-41, all rejected as double-dips).

Steps 39-41 REUSED champion entries and boosted/filtered them; add.42 found the
book's OOS edge sits in signals the fixed sensors DE-prioritize. This build AVOIDS
that by REPLACING: it produces an independent trade list from the fixed sensors.

-----------------------------------------------------------------------------
PRE-REGISTERED SPEC (fixed before validation; ALL params flagged OURS, add.35)
-----------------------------------------------------------------------------
Conjunctive — must ALL pass to fire (weakest-link, the add.29 correct SHAPE):

1. REGIME GATE (permission): NOT in a confirmed markdown, using only raw-input-tier
   features present both historically and live (add.32):
     (close > ema_200  OR  struct_range_state != broken_down)  AND  stables_rot_rising == 0
2. STRUCTURAL CONTEXT (where): price inside an ACTIVE 1D N=5 structural range
     struct_range_state == active  AND  struct_range_pos <= 0.4   (discount half)
3. TRIGGER (the corrected spring / entry event): a struct_sweep_low this bar OR
   within trailing K=3 bars (a wick that pierced the DRAWN range low and closed
   back inside). Enter on the reclaim close (= this bar's close).
4. TIME (when): CONVICTION TIER (pre-registered as tier, not hard gate, to avoid
   starving n): fib_time_confluence > 0 at entry -> conviction.
5. SIZE (conviction, conjunctive-gated NOT additive fusion): base risk 1%;
   1.5x if (fib_time_confluence > 0)  OR  (a Bojan-low zone is active at entry).
6. STOP (WI model-invalidation): under the CREATED structural low (min low over the
   sweep cluster .. entry bar) minus 0.25 * ATR14(entry). NOT the deep range low.
7. MANAGE: TP1 40% at structural resistance (struct_range_high, else nearest
   swing_high_50, else entry+1R; require >= entry+0.5R). Move stop to BREAKEVEN on
   TP1 fill. Runner 60% to the structural target = measured move
   struct_range_high + (struct_range_high - struct_range_low), floored at entry+2R.
   max_hold = 168h.

EXIT VARIANTS (identical entries + stops; isolate whether structural geometry helps):
  'struct' : the R7 geometry above (TP1 at range high -> BE -> runner to measured move)
  'naive'  : plain 1R/2R/3R equal-thirds ladder, fixed stop, max_hold 168h

FLAGGED HYPOTHESES (ours; none are WI numbers, add.35): K=3, discount<=0.4, 1D N=5,
MIN_WIDTH_ATR=1.5, stop buffer 0.25 ATR, TP geometry, conviction 1.5x, Bojan W=0.5.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from backtester import EntryPlan
from htf_pivots import reanchor_frame
from structural_range import build_structural_range, ACTIVE, BROKEN_DOWN
from bojan_detector import build_bojan

# ---- FLAGGED PARAMS (OURS, add.35) ----
HTF_FREQ = "1D"
HTF_N = 5
K_TRIGGER = 3            # sweep-recency window (bars, inclusive)
DISCOUNT_MAX = 0.4       # discount-half threshold on struct_range_pos
STOP_BUF_ATR = 0.25      # stop buffer below the created structural low
CONVICTION_MULT = 1.5    # conviction sizing multiplier
BOJAN_W = 0.5            # loosest Bojan wick threshold (most fires)
MAX_HOLD = 168
MIN_TP1_R = 0.5          # TP1 must clear >= this many R above entry to use a level


def build_sensors(df: pd.DataFrame):
    """Compute the fixed sensors once: 1D N=5 structural range + Bojan-low activity."""
    reanch = reanchor_frame(df, HTF_FREQ, HTF_N)
    sr = build_structural_range(reanch)
    bj = build_bojan(reanch, sr, BOJAN_W)
    return sr, bj


class UnifiedArchetype:
    """Callable entry_fn. variant in {'struct','naive'}; conviction toggles R5 sizing."""

    def __init__(self, df: pd.DataFrame, sr: pd.DataFrame, bj: pd.DataFrame,
                 variant: str = "struct", conviction: bool = False):
        self.variant = variant
        self.conviction = conviction
        self.highs = df["high"].to_numpy(float)
        self.lows = df["low"].to_numpy(float)
        self.closes = df["close"].to_numpy(float)
        self.atr = df["atr_14"].to_numpy(float)
        self.ema200 = df["ema_200"].to_numpy(float)
        self.stables = np.nan_to_num(df["stables_rot_rising"].to_numpy(float))
        self.fib = np.nan_to_num(df["fib_time_confluence"].to_numpy(float))
        self.state = sr["struct_range_state"].to_numpy(dtype=object)
        self.pos = sr["struct_range_pos"].to_numpy(float)
        self.rhigh = sr["struct_range_high"].to_numpy(float)
        self.rlow = sr["struct_range_low"].to_numpy(float)
        self.sweep_lo = sr["struct_sweep_low"].to_numpy(np.int8)
        self.swh50 = df["swing_high_50"].to_numpy(float)   # 1H swings for TP fallback
        self.bojan_active = bj["bojan_low_active"].to_numpy(np.int8)
        self.index = df.index
        self.n = len(df)
        self._last_entry_i = -10_000    # episode dedup: no re-entry within K of last
        self.entries_log = []

    def _regime_ok(self, i):
        not_markdown = (self.closes[i] > self.ema200[i]) or (self.state[i] != BROKEN_DOWN)
        return bool(not_markdown and self.stables[i] == 0)

    def _context_ok(self, i):
        return bool(self.state[i] == ACTIVE
                    and not np.isnan(self.pos[i]) and self.pos[i] <= DISCOUNT_MAX)

    def _recent_sweep_bar(self, i):
        """Return the most-recent sweep-bar index within [i-K+1, i], else None."""
        lo = max(0, i - K_TRIGGER + 1)
        for c in range(i, lo - 1, -1):
            if self.sweep_lo[c] == 1:
                return c
        return None

    def __call__(self, df: pd.DataFrame, i: int):
        # one candidate per sweep cluster: skip if we just entered within K bars
        if i - self._last_entry_i < K_TRIGGER:
            return None
        if not self._regime_ok(i) or not self._context_ok(i):
            return None
        sweep_c = self._recent_sweep_bar(i)
        if sweep_c is None:
            return None

        atr = self.atr[i]
        entry_raw = self.closes[i]
        # created structural low = min low over the sweep cluster .. entry bar (pessimistic)
        created_low = np.nanmin(self.lows[sweep_c:i + 1])
        if not np.isfinite(atr) or not np.isfinite(created_low):
            return None
        stop = created_low - STOP_BUF_ATR * atr
        R = entry_raw - stop
        if R <= 0:
            return None

        self._last_entry_i = i

        # conviction (R4/R5)
        time_present = self.fib[i] > 0
        bojan_present = self.bojan_active[i] == 1
        rmult = CONVICTION_MULT if (self.conviction and (time_present or bojan_present)) else 1.0

        rhigh = self.rhigh[i]
        rlow = self.rlow[i]
        swh = self.swh50[i]

        self.entries_log.append({
            "entry_time": self.index[i], "sweep_time": self.index[sweep_c],
            "struct_range_pos": float(self.pos[i]),
            "time_present": bool(time_present), "bojan_present": bool(bojan_present),
            "risk_mult": rmult, "R_price": float(R),
        })

        if self.variant == "naive":
            targets = [(entry_raw + 1 * R, 1 / 3.0),
                       (entry_raw + 2 * R, 1 / 3.0),
                       (entry_raw + 3 * R, 1 / 3.0)]
            return EntryPlan(direction="long", stop=stop, targets=targets,
                             move_stop_to_after_first_tp=None, runner_target=None,
                             max_hold_bars=MAX_HOLD, risk_mult=rmult,
                             meta={"variant": "naive"})

        # ---- structural (R7) geometry ----
        # TP1 = struct_range_high, else nearest swing_high_50, else entry+1R; must clear 0.5R
        tp1 = None
        for cand in (rhigh, swh):
            if np.isfinite(cand) and cand >= entry_raw + MIN_TP1_R * R:
                tp1 = cand
                break
        if tp1 is None:
            tp1 = entry_raw + 1 * R
        # runner target = measured move off the drawn range, floored at entry+2R
        if np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow:
            measured = rhigh + (rhigh - rlow)
        else:
            measured = -np.inf
        tt = max(entry_raw + 2 * R, measured)
        move_stop = entry_raw   # breakeven
        return EntryPlan(direction="long", stop=stop,
                         targets=[(tp1, 0.40)],
                         move_stop_to_after_first_tp=move_stop,
                         runner_target=tt, max_hold_bars=MAX_HOLD, risk_mult=rmult,
                         meta={"variant": "struct", "tp1": float(tp1), "tt": float(tt)})
