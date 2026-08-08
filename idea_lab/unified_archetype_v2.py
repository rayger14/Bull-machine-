"""
THE UNIFIED ARCHETYPE v2  (LONG-ONLY, pre-registered, conjunctive; STUDY ONLY)
==============================================================================

WHY v2 EXISTS (the identity problem add.43/44 exposed)
------------------------------------------------------
The v1 unified (unified_archetype.py, add.43) is M1-ONLY: a struct_sweep_low
"spring" bought at the DRAWN range low. On the fixed sensors it produced TRAIN
PF 2.02 but OOS-A 0.73 / OOS-B 0.15 -> FAIL 1/3, and the live-forward run (add.44)
was PF 0.89 / -$822 with ALL 10 entries BELOW ema_200. DIAGNOSIS (add.43/44): it is
structurally a below-EMA200 CAPITULATION buyer (a falling-knife catcher). The levels
it buys DO bounce (avg MFE +5.6R, all 10 reached >=+1R within 168h) but the tight
structural stop is stabbed FIRST 7/10 in high-vol markdowns. NOTHING makes the
strategy stand down when the regime is a genuine markdown, so it keeps catching knives.
The root cause, campaign-final (add.44): a long-biased entry with no regime stand-down.

THE v2 HYPOTHESIS (pre-registered; test it, report the verdict even if REJECT)
-----------------------------------------------------------------------------
Add a SECOND long door that STRUCTURALLY CANNOT fire deep in a markdown, and
tighten the first door with the WI failing-spring filter:

  * M1 pathway (spring), now RTZ-FILTERED. Keep the corrected struct_sweep_low
    spring, but do NOT enter the raw reclaim. Require the WI failing-spring filter:
    a structure break (SOS / "jump across the creek") AFTER the spring, THEN a
    return-to-zone pullback that HOLDS above the creek (LPS). Enter the pullback
    that holds, not the raw reclaim. (WI: "the spring alone is a data point; the
    jump + the back-up that holds is the trade.")

  * M2 pathway (LPS after SOS) -- THE IDENTITY CHANGE. Using the causal HTF eye
    state machine (engine/features/eye_state.py, commit 81c9b1f; states IN_RANGE,
    MANIPULATION, MODEL_FORMING, CONFIRMED_BREAK, TRENDING; no-repaint verified),
    enter the LPS return-to-zone that HOLDS AFTER a bull CONFIRMED_BREAK, into a
    FORMED model -- i.e. the ALIGNED_FORMING context (eye_state==MODEL_FORMING with
    a persistent bull eye_dir). CRITICAL DESIGN CONSTRAINT (add.44 / eye Phase-2 at
    81c9b1f): do NOT enter the breakout EXTENSION (the CONFIRMED_BREAK / TRENDING
    tier) -- that tier was best in TRAIN but INVERTED OOS (PF 1.74 -> 0.47) because
    trend-extension mean-reverts. The ONE tier with STABLE cross-era ordering was
    ALIGNED_FORMING (accumulation MODEL_FORMING inside a range, bull bias) > COUNTER.
    So M2 enters the ALIGNED_FORMING pullback, never the extension. Because M2
    REQUIRES a prior bull CONFIRMED_BREAK (a real HTF up-break, which sets eye_dir
    bull), it CANNOT fire deep in a markdown -- eye_dir there is bear/neutral and
    there is no recent up-break. THAT self-regime-filtering is the whole point of v2.

  * Time-as-validity. fib_time_confluence stays the CONVICTION booster (1.5x) AND
    each setup carries a VALIDITY WINDOW: if the confirmation (M1 SOS+RTZ / M2 LPS)
    does not arrive within the window, the setup EXPIRES and is skipped. The window
    lengths are pre-registered below (OURS). (WI add.11: "time-based expiry of the
    expected window" is itself an invalidation.)

  * Regime permission. Keep the VALIDATED stables_rot_rising==0 flag (do NOT rebuild
    the dominance reader -- CONFIRMED REJECT, commit 6e518c0). Keep the v1 not-markdown
    permission (close>ema200 OR struct_range_state != broken_down).

  * Long-only (distribution mirror out of scope).

THE ACID TEST (the one that decides whether v2 changed the DNA)
--------------------------------------------------------------
Report the below/above-EMA200 split PER PATHWAY PER ERA. v1 was ~76% below-EMA200
multi-era and 100% below live. If M2 finally takes a materially higher share of
ABOVE-EMA200 (trend-aligned) trades, the M2/LPS door broke the falling-knife DNA.
If M2 is still ~100% below-EMA200, it FAILED to change the identity -- say so.

-----------------------------------------------------------------------------
PRE-REGISTERED SPEC  (fixed BEFORE any outcome is measured; ALL magnitudes OURS,
                      per the add.35 provenance discipline; NO grids, NO iterating
                      against results)
-----------------------------------------------------------------------------
SHARED
  R0 REGIME (permission, both pathways): stables_rot_rising==0
     AND (close>ema_200 OR struct_range_state != broken_down).
  R-SIZE conviction (both pathways): base risk 1%; 1.5x if
     (fib_time_confluence>0) OR (a Bojan-low zone is active at entry).
  R-STOP (both pathways): stop = created_structural_low - 0.25*ATR14(entry),
     where created_low is the min low over the setup's own cluster (see each door).
  R-EXIT variants (identical entries+stops): 'struct' = TP1 40% at struct_range_high
     (else swing_high_50, else entry+1R; must clear >=0.5R) -> stop to BREAKEVEN ->
     runner 60% to measured move struct_range_high+(high-low), floored entry+2R.
     'naive' = plain 1R/2R/3R equal thirds. max_hold=168h. HEADLINE = struct/flat.
  R-DEDUP: no re-entry within DEDUP_K bars of the last entry (episode dedup).
  R-CONFLICT: if BOTH doors fire on the same bar, take M2 (trend-aligned LPS is
     pre-registered as the higher-quality door; the confound is minimized by
     ALWAYS reporting per-pathway attribution).

M1 pathway (RTZ-filtered spring)  [struct sensor = idea_lab 1D N=5 drawn range]
  M1.1 context: struct_range_state==active AND struct_range_pos<=0.4 at the spring.
  M1.2 spring: most-recent struct_sweep_low bar s within trailing SPRING_WIN bars.
       creek = high[s]; spring_low = low[s].
  M1.3 not invalidated: no bar in [s .. i-1] CLOSED below spring_low.
  M1.4 SOS ("jump across the creek"): some bar in (s .. i-1] CLOSED above
       creek + SOS_BUF_ATR*ATR (a real reclaim of supply, not the raw spring bar).
  M1.5 RTZ hold + entry at bar i (the LPS): low[i] <= creek + RTZ_ATR*ATR (pulled
       back INTO the creek zone) AND close[i] >= creek (HELD above the creek) AND
       close[i] > spring_low. Enter at close[i]. created_low = min low over [s..i].
  M1 VALIDITY WINDOW = SPRING_WIN (spring->SOS->LPS must complete inside it).

M2 pathway (LPS after SOS -- the identity change)  [eye sensor = 1D N=40 HTF states]
  M2.1 SOS precondition + validity window: a bull CONFIRMED_BREAK
       (eye_state==CONFIRMED_BREAK AND eye_dir=='bull') occurred within trailing
       M2_SOS_WIN bars. break_level = range_upper_1d at that break bar. If no such
       break in the window, the setup does not exist (window expiry = skip).
  M2.2 ALIGNED_FORMING context at entry (the stable tier; NOT the extension):
       eye_dir[i]=='bull' AND eye_state[i]=='MODEL_FORMING'.
  M2.3 LPS holds above reclaimed structure: close[i] >= break_level.
       (MODEL_FORMING requires price in the low third of the LIVE range = pulled
        back to demand; close>=break_level = the up-break was NOT given back.)
  M2.4 entry at close[i]. created_low = min low over trailing LPS_LOOKBACK bars
       (the pullback leg). Stop = created_low - 0.25*ATR (under the created LPS low).
  M2 VALIDITY WINDOW = M2_SOS_WIN (LPS must form within it of the SOS).

  M2-broad (DIAGNOSTIC ONLY, reported separately, NOT the headline): relax M2.2 to
    eye_dir=='bull' AND eye_state in {MODEL_FORMING, IN_RANGE, MANIPULATION} (non-
    extension) AND add an explicit RTZ test low[i] <= break_level + RTZ_ATR*ATR.
    Shows whether widening past the strict ALIGNED_FORMING tier changes the picture.

-----------------------------------------------------------------------------
FLAGGED HYPOTHESES (OURS; none are WI numbers; pre-registered single values, NOT swept)
-----------------------------------------------------------------------------
  SPRING_WIN=72   SOS_BUF_ATR=0.10   RTZ_ATR=0.5   DISCOUNT_MAX=0.4 (v1 inherit)
  M2_SOS_WIN=360  LPS_LOOKBACK=48    STOP_BUF_ATR=0.25 (v1 inherit)
  CONVICTION_MULT=1.5 (v1 inherit)   DEDUP_K=3   BOJAN_W=0.5 (v1 inherit)   MAX_HOLD=168
  HTF for the drawn range = 1D N=5 (v1 inherit); HTF for the eye = 1D N=40 (81c9b1f).

CAUSALITY: every read at bar i uses only arrays at indices <= i. struct/bojan/eye are
all causal, no-repaint (eye verified by tests/test_eye_state_causality.py). Entry is
this bar's close. No future data anywhere.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester import EntryPlan
from htf_pivots import reanchor_frame
from structural_range import build_structural_range, ACTIVE, BROKEN_DOWN
from bojan_detector import build_bojan
from engine.features.eye_state import (
    compute_eye_features, MODEL_FORMING, CONFIRMED_BREAK, TRENDING,
    IN_RANGE, MANIPULATION,
)

# ---- FLAGGED PARAMS (OURS; pre-registered single values, NOT swept) ----
HTF_FREQ = "1D"
HTF_N = 5                 # idea_lab drawn-range anchor (M1 door)
DISCOUNT_MAX = 0.4        # M1 discount-half threshold
SPRING_WIN = 72           # M1 spring->SOS->LPS validity window (bars)
SOS_BUF_ATR = 0.10        # M1 "jump across the creek" buffer above the spring high
RTZ_ATR = 0.5             # return-to-zone proximity (ATR) for M1 creek / M2-broad
M2_SOS_WIN = 360          # M2 SOS lookback + LPS validity window (bars ~ 15 days)
LPS_LOOKBACK = 48         # M2 created-LPS-low lookback (bars)
STOP_BUF_ATR = 0.25       # stop below the created structural low
CONVICTION_MULT = 1.5
DEDUP_K = 3
BOJAN_W = 0.5
MAX_HOLD = 168
MIN_TP1_R = 0.5


def build_sensors_v2(df: pd.DataFrame):
    """Compute all fixed sensors once (causal): idea_lab 1D N=5 structural range +
    Bojan-low (for M1), and the HTF eye state machine (for M2)."""
    reanch = reanchor_frame(df, HTF_FREQ, HTF_N)
    sr = build_structural_range(reanch)
    bj = build_bojan(reanch, sr, BOJAN_W)
    eye = compute_eye_features(df)
    return sr, bj, eye


class UnifiedArchetypeV2:
    """Callable entry_fn generating its OWN trades from TWO long doors (M1 RTZ-spring,
    M2 LPS-after-SOS). variant in {'struct','naive'}; conviction toggles R-SIZE.
    m2_mode in {'aligned','broad'} selects the strict ALIGNED_FORMING M2 (headline)
    or the diagnostic M2-broad relaxation."""

    def __init__(self, df, sr, bj, eye, variant="struct", conviction=False,
                 m2_mode="aligned"):
        self.variant = variant
        self.conviction = conviction
        self.m2_mode = m2_mode
        # ohlc / context
        self.opens = df["open"].to_numpy(float)
        self.highs = df["high"].to_numpy(float)
        self.lows = df["low"].to_numpy(float)
        self.closes = df["close"].to_numpy(float)
        self.atr = df["atr_14"].to_numpy(float)
        self.ema200 = df["ema_200"].to_numpy(float)
        self.stables = np.nan_to_num(df["stables_rot_rising"].to_numpy(float))
        self.fib = np.nan_to_num(df["fib_time_confluence"].to_numpy(float))
        # struct sensor (M1)
        self.state = sr["struct_range_state"].to_numpy(dtype=object)
        self.pos = sr["struct_range_pos"].to_numpy(float)
        self.rhigh = sr["struct_range_high"].to_numpy(float)
        self.rlow = sr["struct_range_low"].to_numpy(float)
        self.sweep_lo = sr["struct_sweep_low"].to_numpy(np.int8)
        self.swh50 = df["swing_high_50"].to_numpy(float)
        self.bojan_active = bj["bojan_low_active"].to_numpy(np.int8)
        # eye sensor (M2)
        self.eye_state = eye["eye_state"].to_numpy(dtype=object)
        self.eye_dir = eye["eye_dir"].to_numpy(dtype=object)
        self.range_upper = eye["range_upper_1d"].to_numpy(float)
        self.index = df.index
        self.n = len(df)
        self._last_entry_i = -10_000
        self.entries_log = []

    # ---------- shared gates ----------
    def _regime_ok(self, i):
        not_markdown = (self.closes[i] > self.ema200[i]) or (self.state[i] != BROKEN_DOWN)
        return bool(not_markdown and self.stables[i] == 0)

    def _conviction(self, i):
        time_present = self.fib[i] > 0
        bojan_present = self.bojan_active[i] == 1
        rmult = CONVICTION_MULT if (self.conviction and (time_present or bojan_present)) else 1.0
        return time_present, bojan_present, rmult

    # ---------- M1 door: RTZ-filtered spring ----------
    def _m1(self, i):
        if not (self.state[i] == ACTIVE and not np.isnan(self.pos[i])
                and self.pos[i] <= DISCOUNT_MAX):
            return None
        atr = self.atr[i]
        if not np.isfinite(atr) or atr <= 0:
            return None
        # most-recent spring within window
        lo = max(0, i - SPRING_WIN + 1)
        s = None
        for c in range(i, lo - 1, -1):
            if self.sweep_lo[c] == 1 and self.state[c] == ACTIVE:
                s = c
                break
        if s is None or s >= i:
            return None
        spring_low = self.lows[s]
        creek = self.highs[s]
        if not (np.isfinite(spring_low) and np.isfinite(creek)):
            return None
        # not invalidated: no close below spring_low in [s .. i-1]
        if np.nanmin(self.closes[s:i]) < spring_low:
            return None
        # SOS: some close in (s .. i-1] above creek + buffer
        seg = self.closes[s + 1:i]  # excludes entry bar i
        if len(seg) == 0 or np.nanmax(seg) < creek + SOS_BUF_ATR * atr:
            return None
        # RTZ hold at i (the LPS)
        if not (self.lows[i] <= creek + RTZ_ATR * atr
                and self.closes[i] >= creek
                and self.closes[i] > spring_low):
            return None
        created_low = np.nanmin(self.lows[s:i + 1])
        return {"pathway": "M1", "created_low": created_low, "sweep_bar": s,
                "aligned_forming": False, "break_level": np.nan}

    # ---------- M2 door: LPS after SOS (ALIGNED_FORMING) ----------
    def _m2(self, i):
        atr = self.atr[i]
        if not np.isfinite(atr) or atr <= 0:
            return None
        if not (self.eye_dir[i] == "bull"):
            return None
        if self.m2_mode == "aligned":
            if self.eye_state[i] != MODEL_FORMING:
                return None
        else:  # broad (diagnostic)
            if self.eye_state[i] not in (MODEL_FORMING, IN_RANGE, MANIPULATION):
                return None
        # SOS precondition: a bull CONFIRMED_BREAK within the window (exclude entry bar)
        lo = max(0, i - M2_SOS_WIN)
        break_level = np.nan
        for c in range(i - 1, lo - 1, -1):
            if self.eye_state[c] == CONFIRMED_BREAK and self.eye_dir[c] == "bull":
                break_level = self.range_upper[c]
                break
        if not np.isfinite(break_level):
            return None
        # LPS holds above reclaimed structure
        if not (self.closes[i] >= break_level):
            return None
        if self.m2_mode != "aligned":
            # explicit RTZ test for the broad diagnostic
            if not (self.lows[i] <= break_level + RTZ_ATR * atr):
                return None
        lps_lo = max(0, i - LPS_LOOKBACK + 1)
        created_low = np.nanmin(self.lows[lps_lo:i + 1])
        return {"pathway": "M2", "created_low": created_low, "sweep_bar": None,
                "aligned_forming": bool(self.eye_state[i] == MODEL_FORMING),
                "break_level": float(break_level)}

    # ---------- plan builder ----------
    def _plan(self, i, sig):
        atr = self.atr[i]
        entry_raw = self.closes[i]
        created_low = sig["created_low"]
        if not np.isfinite(created_low):
            return None
        stop = created_low - STOP_BUF_ATR * atr
        R = entry_raw - stop
        if R <= 0:
            return None
        time_present, bojan_present, rmult = self._conviction(i)
        rhigh = self.rhigh[i]
        rlow = self.rlow[i]
        swh = self.swh50[i]

        self.entries_log.append({
            "entry_time": self.index[i], "pathway": sig["pathway"],
            "aligned_forming": sig["aligned_forming"],
            "below_ema200": bool(entry_raw < self.ema200[i]),
            "time_present": bool(time_present), "bojan_present": bool(bojan_present),
            "risk_mult": rmult, "R_price": float(R),
            "break_level": sig["break_level"],
        })

        if self.variant == "naive":
            targets = [(entry_raw + 1 * R, 1 / 3.0),
                       (entry_raw + 2 * R, 1 / 3.0),
                       (entry_raw + 3 * R, 1 / 3.0)]
            return EntryPlan(direction="long", stop=stop, targets=targets,
                             move_stop_to_after_first_tp=None, runner_target=None,
                             max_hold_bars=MAX_HOLD, risk_mult=rmult,
                             meta={"variant": "naive", "pathway": sig["pathway"],
                                   "_rmult": rmult})
        # struct (R7) geometry
        tp1 = None
        for cand in (rhigh, swh):
            if np.isfinite(cand) and cand >= entry_raw + MIN_TP1_R * R:
                tp1 = cand
                break
        if tp1 is None:
            tp1 = entry_raw + 1 * R
        if np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow:
            measured = rhigh + (rhigh - rlow)
        else:
            measured = -np.inf
        tt = max(entry_raw + 2 * R, measured)
        return EntryPlan(direction="long", stop=stop, targets=[(tp1, 0.40)],
                         move_stop_to_after_first_tp=entry_raw, runner_target=tt,
                         max_hold_bars=MAX_HOLD, risk_mult=rmult,
                         meta={"variant": "struct", "pathway": sig["pathway"],
                               "tp1": float(tp1), "tt": float(tt), "_rmult": rmult})

    def __call__(self, df, i):
        if i - self._last_entry_i < DEDUP_K:
            return None
        if not self._regime_ok(i):
            return None
        # R-CONFLICT: prefer M2 (trend-aligned) over M1 when both fire
        sig = self._m2(i)
        if sig is None:
            sig = self._m1(i)
        if sig is None:
            return None
        plan = self._plan(i, sig)
        if plan is None:
            return None
        self._last_entry_i = i
        return plan
