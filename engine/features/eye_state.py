"""
The All-Seeing Eye — Phase 1: HTF state machine as a causal feature pass.
========================================================================

Implements Wyckoff_Insider's operative HTF state set (wyckoff_audit.md addendum 11)
as a causal, no-repaint feature pass on the 1D grid, broadcast back to the 1H grid
with closed-HTF-bar discipline (merge_asof backward + 1h settle lag).

STUDY ONLY. This module does NOT wire any sizing/boost/gate. It emits descriptive
state features that Phase 2 tiers champion entries against. No production config or
code is touched.

Design authority (two evidence bases, both constrain the mechanization):
  - Source mechanics: wyckoff_audit.md addendum 11 (WI batch 4).
  - Spec: docs/knowledge/all_seeing_eye_spec_v1.md (P1 = features, causal, no-repaint).

--------------------------------------------------------------------------------
PRE-REGISTERED THRESHOLDS  (H = OUR HYPOTHESIS where WI gave no number;
                            S = specified/derivable from spec/source)
--------------------------------------------------------------------------------
  N_RANGE_1D        = 40    (H)  trailing 1D bars for range boundary swing extremes
  RANGE_SHIFT       = 1     (S)  boundaries use bars [t-N .. t-1] only (causal)
  ATR_PERIOD_1D     = 14    (H)  daily ATR period for retest proximity
  EXTREME_THIRD     = 1/3   (H)  MODEL_FORMING requires location in extreme third
  ACCEPT_CONSEC     = 2     (S)  acceptance = 2 consecutive 1D closes beyond
                                 (causal restatement of "the NEXT 1D close also beyond":
                                  the confirmation bar is the acceptance bar, and at its
                                  close BOTH closes are already known -> no lookahead)
  TREND_CONSEC      = 5     (S)  5 consecutive 1D closes beyond -> TRENDING
  RETEST_ATR_MULT   = 1.0   (S)  retest = revisit within 1*ATR(1D) of broken boundary
                                 and close back in trend direction
  M2_SOS_LOOKBACK   = 10    (S)  completed-M2 (C_accum + SOS within 10 1D bars) may set
                                 LOCAL bias pre-break
  SETTLE_LAG_H      = 1     (S)  broadcast at 1D close + 1h (closed-bar discipline)

Body-close semantics: for a 1D candle the "body close" IS the daily close price;
range boundaries are built from BODY extremes (max/min of open,close), not wick
extremes, per WI ("confirmed body-close extremes").

MANIPULATION is a wick beyond a boundary WITHOUT a body close beyond. It is
direction-tagged (which side was swept) but is NOT a bias flip (WI explicit).
A single body close beyond WITHOUT acceptance is also NOT yet a flip: it is held
as a pending break (state stays IN_RANGE, eye_pending_break tagged) until the
acceptance bar confirms it. Only a body close back INSIDE cancels a break
("wick-only re-entry does not cancel", per spec).
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- pre-registered constants (see header) ---
N_RANGE_1D = 40
RANGE_SHIFT = 1
ATR_PERIOD_1D = 14
EXTREME_THIRD = 1.0 / 3.0
ACCEPT_CONSEC = 2
TREND_CONSEC = 5
RETEST_ATR_MULT = 1.0
M2_SOS_LOOKBACK = 10
SETTLE_LAG_H = 1

# state enum (strings, kept human-readable for the feature store / reports)
IN_RANGE = "IN_RANGE"
MANIPULATION = "MANIPULATION"
MODEL_FORMING = "MODEL_FORMING"
CONFIRMED_BREAK = "CONFIRMED_BREAK"
TRENDING = "TRENDING"


def _resample_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV to 1D calendar bars. Daily bar D covers [D 00:00, D+1 00:00);
    its close is the last 1H close of day D (the 23:00 bar)."""
    o = df_1h["open"].resample("1D").first()
    h = df_1h["high"].resample("1D").max()
    l = df_1h["low"].resample("1D").min()
    c = df_1h["close"].resample("1D").last()
    v = df_1h["volume"].resample("1D").sum()
    daily = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    daily = daily.dropna(subset=["open", "high", "low", "close"])
    return daily


def _daily_phase_and_sos(df_1h: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate the EXISTING 1H phase machinery to the 1D grid, causally.
    - daily phase_dir = the phase_dir at the LAST 1H bar of the day (EOD read).
    - daily sos = 1 if ANY 1H bar in the day fired wyckoff_sos.
    Both are known at the daily close, so both are causal for state[t]."""
    phase = df_1h["wyckoff_phase_dir"] if "wyckoff_phase_dir" in df_1h.columns else pd.Series(index=df_1h.index, dtype=object)
    sos = df_1h["wyckoff_sos"] if "wyckoff_sos" in df_1h.columns else pd.Series(0.0, index=df_1h.index)
    d_phase = phase.resample("1D").last()
    d_sos = (sos.fillna(0).resample("1D").max() > 0).astype(int)
    out = pd.DataFrame({"phase_dir": d_phase, "sos": d_sos})
    return out.reindex(daily_index)


def _atr_1d(daily: pd.DataFrame, period: int = ATR_PERIOD_1D) -> pd.Series:
    """Wilder-style ATR on the daily bars; uses only bar t and priors (causal)."""
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        (daily["high"] - daily["low"]).abs(),
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _run_state_machine(daily: pd.DataFrame) -> pd.DataFrame:
    """Stateful forward pass over 1D bars. Every read is on bars <= t (causal).

    Design (v2 — corrects the frozen-break-level degeneracy of a naive persistence
    machine): eye_state is the LOCAL structure vs the LIVE rolling range and is driven
    by consecutive-closes-beyond-boundary run counters (so TRENDING cannot become a
    permanent absorbing state). eye_dir is a PERSISTENT standing bias that flips only
    on a confirmed opposite break (WI: body break + acceptance flips bias) or an M2
    local-bias signal, and otherwise persists.

      run_up  = consecutive 1D closes above the LIVE upper boundary (reset when not)
      run_dn  = consecutive 1D closes below the LIVE lower boundary
      state:  run>=TREND_CONSEC -> TRENDING ; run>=ACCEPT_CONSEC -> CONFIRMED_BREAK ;
              run==1 -> pending (IN_RANGE, not a flip) ; wick-only -> MANIPULATION ;
              else MODEL_FORMING (phase+extreme) or IN_RANGE.
      bias:   flips on run>=ACCEPT_CONSEC (confirmed break) in the break direction;
              M2 (C_accum + recent SOS) may lift neutral->bull pre-break; else persists.
    """
    body_high = daily[["open", "close"]].max(axis=1)
    body_low = daily[["open", "close"]].min(axis=1)

    # causal range boundaries: rolling extremes of BODY highs/lows over trailing N, shifted 1
    upper = body_high.rolling(N_RANGE_1D, min_periods=N_RANGE_1D).max().shift(RANGE_SHIFT)
    lower = body_low.rolling(N_RANGE_1D, min_periods=N_RANGE_1D).min().shift(RANGE_SHIFT)

    atr = _atr_1d(daily)
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]
    phase = daily["phase_dir"]
    sos = daily["sos"].fillna(0).astype(int)
    # rolling SOS-in-last-K flag (includes current bar; causal)
    sos_recent = sos.rolling(M2_SOS_LOOKBACK, min_periods=1).max().astype(int)

    n = len(daily)
    states = np.empty(n, dtype=object)
    dirs = np.empty(n, dtype=object)
    manip_dir = np.empty(n, dtype=object)
    pending = np.zeros(n, dtype=int)
    m2_bias = np.zeros(n, dtype=int)
    locs = np.full(n, np.nan)

    # carried state
    standing_bias = "neutral"   # persistent HTF bias; flips only on confirmed break / M2
    run_up = 0                  # consecutive closes above LIVE upper
    run_dn = 0                  # consecutive closes below LIVE lower
    break_level = np.nan        # boundary at the bar the current run STARTED (for retest)
    retest_done = False

    idx = daily.index
    for i in range(n):
        u = upper.iloc[i]
        l = lower.iloc[i]
        c = close.iloc[i]
        hi = high.iloc[i]
        lo = low.iloc[i]
        a = atr.iloc[i]
        ph = phase.iloc[i]

        warmup = not (np.isfinite(u) and np.isfinite(l) and u > l)
        loc = float(np.clip((c - l) / (u - l), 0.0, 1.0)) if not warmup else np.nan
        locs[i] = loc

        if warmup:
            states[i], dirs[i], manip_dir[i] = IN_RANGE, "neutral", "none"
            continue

        close_above = c > u
        close_below = c < l
        wick_above = (hi > u) and not close_above
        wick_below = (lo < l) and not close_below

        # ---- update run counters off the LIVE boundary (no frozen level) ----
        if close_above:
            if run_up == 0:
                break_level = u          # boundary at run start (for retest proximity)
                retest_done = False
            run_up += 1
            run_dn = 0
        elif close_below:
            if run_dn == 0:
                break_level = l
                retest_done = False
            run_dn += 1
            run_up = 0
        else:
            run_up = 0
            run_dn = 0

        # ---- retest bookkeeping (successful revisit within 1 ATR of broken level) ----
        if run_up >= ACCEPT_CONSEC and np.isfinite(break_level):
            if (lo <= break_level + RETEST_ATR_MULT * a) and close_above:
                retest_done = True
        if run_dn >= ACCEPT_CONSEC and np.isfinite(break_level):
            if (hi >= break_level - RETEST_ATR_MULT * a) and close_below:
                retest_done = True

        st, dr, md = IN_RANGE, standing_bias, "none"

        # ---- classify local structure (priority) ----
        if run_up >= ACCEPT_CONSEC:
            standing_bias = "bull"                       # confirmed break flips bias
            dr = "bull"
            st = TRENDING if (run_up >= TREND_CONSEC or retest_done) else CONFIRMED_BREAK
        elif run_dn >= ACCEPT_CONSEC:
            standing_bias = "bear"
            dr = "bear"
            st = TRENDING if (run_dn >= TREND_CONSEC or retest_done) else CONFIRMED_BREAK
        elif run_up == 1 or run_dn == 1:
            # single body close beyond -> pending break, NOT a bias flip yet
            st, dr = IN_RANGE, standing_bias
            pending[i] = 1
        elif wick_above:
            st, md = MANIPULATION, "bear"                # swept highs -> downside attention
            dr = standing_bias
        elif wick_below:
            st, md = MANIPULATION, "bull"                # swept lows -> upside attention
            dr = standing_bias
        else:
            # inside the live range: MODEL_FORMING at the matching extreme, else IN_RANGE
            if ph == "C_accum" and loc <= EXTREME_THIRD:
                st, dr = MODEL_FORMING, "bull"
            elif ph == "C_distrib" and loc >= (1.0 - EXTREME_THIRD):
                st, dr = MODEL_FORMING, "bear"
            else:
                st, dr = IN_RANGE, standing_bias

        # ---- M2 pre-break LOCAL bias (WI rule; flagged hypothesis) ----
        # completed-M2 signature: C_accum with SOS within last 10 1D bars may lift a
        # neutral standing bias to bull pre-break. Never overrides a bear break/trend.
        if (run_up < ACCEPT_CONSEC and run_dn < ACCEPT_CONSEC
                and str(ph).startswith("C_accum") and sos_recent.iloc[i] == 1):
            m2_bias[i] = 1
            if standing_bias == "neutral":
                standing_bias = "bull"
                dr = "bull"
                if st == IN_RANGE:
                    st = MODEL_FORMING

        states[i], dirs[i], manip_dir[i] = st, dr, md

    out = pd.DataFrame({
        "eye_state": states,
        "eye_dir": dirs,
        "eye_location": locs,
        "eye_manip_dir": manip_dir,
        "eye_pending_break": pending,
        "eye_m2_bias": m2_bias,
        "range_upper_1d": upper.values,
        "range_lower_1d": lower.values,
    }, index=idx)
    return out


def compute_eye_state_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Full 1D pipeline: resample -> daily phase/sos aggregation -> state machine.
    Returns a 1D-indexed frame of eye features (not yet broadcast)."""
    df_1h = df_1h.sort_index()
    daily = _resample_1d(df_1h)
    aux = _daily_phase_and_sos(df_1h, daily.index)
    daily = daily.join(aux)
    return _run_state_machine(daily)


def broadcast_to_1h(daily_eye: pd.DataFrame, df_1h_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Broadcast 1D eye features to the 1H grid with closed-bar discipline:
    a 1D state for day D becomes visible at D's close + SETTLE_LAG_H hours, and each
    1H bar receives the most recent VISIBLE 1D state (merge_asof backward)."""
    de = daily_eye.copy()
    # daily bar D's close occurs at the last 1H bar of D (D 23:00). Visible at + settle lag.
    de["available_at"] = de.index + pd.Timedelta(hours=23) + pd.Timedelta(hours=SETTLE_LAG_H)
    de = de.sort_values("available_at")

    right = de.reset_index(drop=True)
    left = pd.DataFrame(index=df_1h_index.sort_values())
    left["ts"] = left.index

    merged = pd.merge_asof(
        left.reset_index(drop=True),
        right,
        left_on="ts",
        right_on="available_at",
        direction="backward",
    )
    merged.index = left.index
    cols = ["eye_state", "eye_dir", "eye_location", "eye_manip_dir",
            "eye_pending_break", "eye_m2_bias", "range_upper_1d", "range_lower_1d"]
    return merged[cols]


def compute_eye_features(df_1h: pd.DataFrame) -> pd.DataFrame:
    """End-to-end: 1H OHLCV(+wyckoff_phase_dir, wyckoff_sos) -> 1H eye features.
    Causal and no-repaint by construction (see module header)."""
    daily_eye = compute_eye_state_1d(df_1h)
    return broadcast_to_1h(daily_eye, df_1h.index)
