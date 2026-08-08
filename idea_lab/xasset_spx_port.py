"""
CROSS-ASSET PORT: the LONG unified archetype (v2) -> SPX  (STUDY ONLY)
=====================================================================

PRE-REGISTERED PORT SPEC  (fixed BEFORE any outcome is measured; no per-asset
tuning, no grids, no iterating against results). The whole point is to run the
IDENTICAL BTC-tuned long model on SPX and ask ONE question:

    The BTC long model (M1 RTZ-spring dip-buyer) DIED in BTC's markdown
    (OOS-B 2025-2026 PF 0.29, 5/5 stops, 81% of pooled entries below EMA200).
    Wyckoff is an EQUITIES method. Does the SAME model, SAME params, also die in
    SPX BEARS (dotcom, GFC, 2022)?  If it bleeds in equity bears too -> the
    falling-knife DNA is INTRINSIC (a model problem, cross-asset). If it survives
    equity bears -> the failure is BTC-STRUCTURE-specific.

WHAT THE MODEL ACTUALLY IS (grounded, not aspirational)
-------------------------------------------------------
On BTC the headline UnifiedArchetypeV2 (struct/flat) is DRIVEN ENTIRELY by the
M1 door (RTZ-filtered spring). The M2 "all-seeing eye" ALIGNED_FORMING door fired
n=0 trades in ALL THREE BTC eras (see idea_lab/v2_eras_output.txt), because
eye_state==MODEL_FORMING requires wyckoff_phase_dir=='C_accum' -- a wyckoff-store
feature. So the deployed "eye long model" is, in practice, the M1 spring
dip-buyer + fib-time conviction + a price-based not-markdown permission. That is
exactly what we port.

PORT RULES (pre-registered)
---------------------------
DROPPED (crypto-only):
  * stables_rot_rising  -> set to 0 for all bars. This reduces the shared R0 regime
    permission to EXACTLY its price-based half:
        close > ema_200  OR  struct_range_state != broken_down
    (identical to the non-crypto portion of the v2 R0 gate; no code edit needed --
     stables[i]==0 is then always true).
  * wyckoff-store / whale features (wyckoff_phase_dir, wyckoff_sos) -> absent for
    SPX. eye_state runs degenerate (phase=NaN, sos=0), so M2-ALIGNED is INERT (=0),
    exactly as on BTC. M2-BROAD (price-only bull CONFIRMED_BREAK + LPS) can still
    fire and is reported as a DIAGNOSTIC ONLY (never the headline).

KEPT IDENTICAL (BTC-tuned, NOT retuned; all in BAR units -> timeframe-invariant):
  N=5 pivots, DISCOUNT_MAX=0.4, MIN_WIDTH_ATR=1.5, SPRING_WIN=72, SOS_BUF_ATR=0.10,
  RTZ_ATR=0.5 (the add.45 keeper), STOP_BUF_ATR=0.25, DEDUP_K=3, CONVICTION_MULT=1.5,
  MAX_HOLD=168, TP/BE/runner geometry. The ONLY change is the DATA timeframe
  (exec H->D, struct range D->W); params stay in bar units (168 daily bars, 72 daily
  bars, etc.) -- this is the "changed ONLY by timeframe scaling" discipline. A
  consequence (flagged, intentional): on daily exec, MAX_HOLD=168 bars ~ 8 months.

SENSORS (OHLCV-derived, asset-agnostic; computed CAUSALLY from SPX bars; there is
no V22_CTX store for SPX). REUSE the idea_lab modules UNCHANGED:
  * ema_200          : recursive EMA, alpha=2/(200+1), seeded at arr[0] (matches the
                       live_feature_computer EMA).
  * atr_14           : Wilder ATR (RMA of true range, period 14).
  * swing_high_50 /  : symmetric N=SWING_N fractal pivots on the EXEC timeframe,
    swing_low_50       confirmed at bar+N, forward-filled. FLAGGED (ours): only feeds
                       the M1 TP1 FALLBACK and the bojan anchor; the reanchor step
                       OVERWRITES these for the structural range, so this choice does
                       NOT touch M1 entries/stops. SWING_N=10 (~ the store's "10-bar
                       pivot lag").
  * fib_time_confluence : faithful vectorized port of live_feature_computer._fib_features
                       (centered 41-bar swing shifted +20 = causal; price_score x
                       time_score geo-mean; bar-count fib zones -> PORTABLE). Affects
                       ONLY the conviction (1.5x) tier; the FLAT headline is immune.
  * struct range (M1): reanchor exec TF one step up, then build_structural_range
                       UNCHANGED. Run 1 (hourly exec) -> 1D N=5. Run 2 (daily exec)
                       -> WEEKLY N=5 (reuses detect_fractal_pivots + _broadcast
                       unchanged; only the resample step is weekly).
  * eye (M2 diag)    : compute_eye_features UNCHANGED (degenerate without wyckoff).
  * bojan            : build_bojan UNCHANGED. CAVEAT: its round-number anchor grid is
                       BTC-scaled ($1k steps) -> mildly off for SPX<~2000; affects
                       ONLY the conviction tier, never the flat headline.

VOLUME NOTE: Yahoo ^GSPC index volume is unreliable (0 on some bars). The HEADLINE
model path (regime, M1 spring, struct range, stop, TP) uses NO volume. Volume enters
only via eye_state's sos aggregation (needs wyckoff_sos, absent -> 0) and HTF resample
sums (unused downstream). So the headline is volume-independent. Flagged and verified.

CAUSALITY: every sensor uses only bars <= i. Entry is the decision bar's close.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htf_pivots import (
    reanchor_frame, detect_fractal_pivots, _broadcast,
)
from structural_range import build_structural_range
from bojan_detector import build_bojan
from engine.features.eye_state import compute_eye_features
from unified_archetype_v2 import BOJAN_W, HTF_FREQ, HTF_N

# ---- port-local FLAGGED params (ours; NOT swept) ----
SWING_N = 10             # exec-TF fractal pivot lookback (feeds TP1 fallback + bojan only)
FIB_LOOKBACK = 20        # matches live_feature_computer centered-swing lookback
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_TIME_NUMBERS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


# ---------------------------------------------------------------------------
# Base OHLCV sensors (causal)
# ---------------------------------------------------------------------------
def load_spx(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"timestamp": "ts"}) if "timestamp" in df.columns else df
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
    df = df.sort_index()
    df.index.name = None
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]].copy()


def ema_recursive(arr: np.ndarray, period: int) -> np.ndarray:
    """alpha=2/(period+1) recursive EMA seeded at arr[0] (matches live EMA)."""
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def wilder_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ATR (RMA of true range). Causal (uses bar t and priors)."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.empty_like(tr)
    atr[0] = tr[0]
    a = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = a * tr[i] + (1.0 - a) * atr[i - 1]
    return atr


def fractal_swings(df: pd.DataFrame, N: int):
    """Symmetric N-bar fractal pivots on the exec frame; confirmed at bar+N,
    forward-filled. Causal. Returns (swing_high_50, swing_low_50) arrays."""
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(df)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for p in range(N, n - N):
        hp, lp = high[p], low[p]
        if hp > high[p - N:p].max() and hp > high[p + 1:p + 1 + N].max():
            sh[p + N] = hp   # confirmed only after N right-side bars close
        if lp < low[p - N:p].min() and lp < low[p + 1:p + 1 + N].min():
            sl[p + N] = lp
    sh = pd.Series(sh, index=df.index).ffill().to_numpy()
    sl = pd.Series(sl, index=df.index).ffill().to_numpy()
    return sh, sl


def fib_time_confluence(df: pd.DataFrame, lookback: int = FIB_LOOKBACK) -> np.ndarray:
    """Faithful causal port of live_feature_computer._fib_features fib_time_confluence:
    centered (2*lb+1) swing detection shifted +lb (causal), retracement to nearest fib
    level, bars-since-last-swing proximity to a fib number, geo-mean of the two scores."""
    high = df["high"]
    low = df["low"]
    close = df["close"].to_numpy(float)
    window = 2 * lookback + 1
    sh_mask = (high == high.rolling(window, center=True).max()).shift(lookback).fillna(False).to_numpy()
    sl_mask = (low == low.rolling(window, center=True).min()).shift(lookback).fillna(False).to_numpy()
    hi = high.to_numpy(float)
    lo = low.to_numpy(float)
    n = len(df)
    out = np.zeros(n, dtype=float)
    last_sh = last_sl = np.nan
    last_sh_pos = last_sl_pos = -1
    for i in range(n):
        if sh_mask[i]:
            last_sh = hi[i]; last_sh_pos = i
        if sl_mask[i]:
            last_sl = lo[i]; last_sl_pos = i
        if np.isnan(last_sh) or np.isnan(last_sl):
            continue
        swing_range = last_sh - last_sl
        if swing_range <= 0:
            continue
        retr = max(0.0, min(1.0, (last_sh - close[i]) / swing_range))
        nearest = min(FIB_LEVELS, key=lambda f: abs(retr - f))
        distance = abs(retr - nearest)
        last_swing_pos = max(last_sh_pos, last_sl_pos)
        bars_since = i - last_swing_pos
        min_time_diff = min(abs(bars_since - f) for f in FIB_TIME_NUMBERS)
        price_score = float(np.exp(-distance * 10.0))
        time_score = float(np.exp(-min_time_diff * 0.5))
        out[i] = float(np.sqrt(price_score * time_score))
    return out


def build_base_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ema_200, atr_14, swing_high_50/low_50, fib_time_confluence,
    stables_rot_rising(=0). All causal, OHLCV-derived. Returns a NEW frame."""
    out = df.copy()
    out["ema_200"] = ema_recursive(out["close"].to_numpy(float), 200)
    out["atr_14"] = wilder_atr(out, 14)
    sh, sl = fractal_swings(out, SWING_N)
    out["swing_high_50"] = sh
    out["swing_low_50"] = sl
    out["fib_time_confluence"] = fib_time_confluence(out)
    out["stables_rot_rising"] = 0.0    # DROP crypto rotation -> price-only regime half
    return out


# ---------------------------------------------------------------------------
# Weekly reanchor for the DAILY run (D->W), reusing detect_fractal_pivots + _broadcast
# ---------------------------------------------------------------------------
def reanchor_frame_weekly(df: pd.DataFrame, N: int = HTF_N) -> pd.DataFrame:
    """Daily-exec -> WEEKLY N-pivot reanchor (one timeframe up from the 1D->1W step).
    Mirrors htf_pivots.reanchor_frame but resamples to weekly. Reuses the UNCHANGED
    detect_fractal_pivots / _broadcast for the pivot logic. Causal: a weekly pivot at
    bar p is visible only after bar p+N closes."""
    rule = "1W"
    o = df["open"].resample(rule, label="left", closed="left").first()
    h = df["high"].resample(rule, label="left", closed="left").max()
    l = df["low"].resample(rule, label="left", closed="left").min()
    c = df["close"].resample(rule, label="left", closed="left").last()
    v = df["volume"].resample(rule, label="left", closed="left").sum()
    wk = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna(
        subset=["open", "high", "low", "close"])
    delta = pd.Timedelta(days=7)
    wk["close_time"] = wk.index + delta
    last = df.index[-1]
    wk = wk[wk["close_time"] <= last + pd.Timedelta(days=1)]
    piv = detect_fractal_pivots(wk, N)
    sh = _broadcast(piv, df.index, "pivot_high_level", "is_swing_high")
    sl = _broadcast(piv, df.index, "pivot_low_level", "is_swing_low")
    out = df.copy()
    out["swing_low_50"] = sl.to_numpy()
    out["swing_high_50"] = sh.to_numpy()
    return out


def build_sensors_spx(df: pd.DataFrame, struct_tf: str):
    """Build (sr, bj, eye) for SPX.
    struct_tf='1D' -> hourly-exec run (reanchor 1D N=5, identical to BTC).
    struct_tf='1W' -> daily-exec run (reanchor WEEKLY N=5)."""
    if struct_tf == "1D":
        reanch = reanchor_frame(df, HTF_FREQ, HTF_N)      # unchanged idea_lab reanchor
    elif struct_tf == "1W":
        reanch = reanchor_frame_weekly(df, HTF_N)
    else:
        raise ValueError(struct_tf)
    sr = build_structural_range(reanch)                    # unchanged
    bj = build_bojan(reanch, sr, BOJAN_W)                  # unchanged (round-grid caveat)
    eye = compute_eye_features(df)                         # unchanged (degenerate w/o wyckoff)
    return sr, bj, eye
