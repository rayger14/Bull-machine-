"""
HTF SWING PIVOTS  (STUDY ONLY — sensor plumbing for the structural-range re-anchor)

WHY THIS EXISTS
---------------
wyckoff_audit addendum 34 proved the structural_range machine FIXES spring location
and POC trend-contamination, but its ONE remaining defect is the ANCHOR TIMEFRAME:
fed 1H swing_50 pivots, the drawn box redraws every ~1.1 days (median lifetime 0.4d),
whereas the Wyckoff-Insider ranges a human draws live for WEEKS. Addendum 32 found the
store's HTF swing columns (tf1d_range_*, tf4h_range_*) are all-null — so we must BUILD
HTF swing pivots and re-anchor the (unchanged) machine to them.

WHAT THIS MODULE DOES
---------------------
1. Resample the causal 1H OHLCV grid up to 4H and 1D (open=first, high=max, low=min,
   close=last, vol=sum). CLOSED BARS ONLY (a partial in-progress HTF bar is never used).
2. Detect swing highs/lows on each HTF with a SYMMETRIC N-bar fractal pivot:
     swing_high at HTF bar p  iff  high[p] > high[p +/- k] for all k in 1..N
     swing_low  at HTF bar p  iff  low[p]  < low[p +/- k]  for all k in 1..N
   Right-side lag => a pivot at bar p is CONFIRMED only once bar p+N has CLOSED
   (PIT-correct, no repaint). The confirmation timestamp = close-time of bar p+N.
3. Broadcast confirmed HTF pivot LEVELS onto the 1H grid via merge_asof backward:
   each 1H bar sees the most recent pivot level whose CONFIRMATION time <= that bar.
   Emits swing_high_4h / swing_low_4h / swing_high_1d / swing_low_1d — drop-in
   replacements for the 1H swing_high_50 / swing_low_50 the machine currently uses.

HONEST CAVEATS (provenance ledger addendum 35, section C.2 — FLAGGED HYPOTHESES, OURS)
------------------------------------------------------------------------------------
  * N (pivot lookback) — WI publishes NO pivot lookback number. We pre-register N=3
    AND N=5 on each of 4H and 1D as hypotheses and report all four.
  * HTF choice (4H vs 1D vs W/M) — WI's true structural scale is Weekly/Monthly;
    1D may still be a compromise. Flagged in the verdict.
  * Everything downstream (MIN_WIDTH_ATR=1.5, break rules) is inherited UNCHANGED from
    the validated machine and remains a flagged hypothesis per the ledger.

CAUSALITY CONTRACT
------------------
  * Resampling uses only closed bars (label + close explicit; a bar labelled at time T
    covers [T, T+freq) and CLOSES at T+freq).
  * A pivot is emitted onto the 1H grid at time = close-time of the (p+N)-th HTF bar,
    i.e. AFTER all N right-side confirming bars have closed. Nothing is visible early.
  * merge_asof(direction="backward") guarantees a 1H bar only ever reads a level whose
    confirmation timestamp is <= the 1H bar's own timestamp.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HTF_FREQ = {"4H": "4h", "1D": "1D"}


def resample_htf(df1h: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Causal 1H -> HTF OHLCV resample. Returns HTF bars indexed by PERIOD-START, with a
    'close_time' column = period-start + freq (the causal moment the bar is complete).

    Only fully-formed bars are kept: the last bin is dropped if the source data does not
    extend a full `freq` beyond its start (an in-progress bar).
    """
    if freq not in HTF_FREQ:
        raise ValueError(f"freq must be one of {list(HTF_FREQ)}")
    rule = HTF_FREQ[freq]
    o = df1h["open"].resample(rule, label="left", closed="left").first()
    h = df1h["high"].resample(rule, label="left", closed="left").max()
    l = df1h["low"].resample(rule, label="left", closed="left").min()
    c = df1h["close"].resample(rule, label="left", closed="left").last()
    v = df1h["volume"].resample(rule, label="left", closed="left").sum()
    htf = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    htf = htf.dropna(subset=["open", "high", "low", "close"])  # drop empty bins (data gaps)
    delta = pd.Timedelta(rule)
    htf["close_time"] = htf.index + delta
    # drop an in-progress final bar: its close_time is beyond the last available 1H close
    last_1h = df1h.index[-1]
    htf = htf[htf["close_time"] <= last_1h + pd.Timedelta(hours=1)]
    return htf


def detect_fractal_pivots(htf: pd.DataFrame, N: int) -> pd.DataFrame:
    """Symmetric N-bar fractal pivots on an HTF OHLCV frame.

    Returns a DataFrame with columns:
      is_swing_high, is_swing_low  (bool, marked at the PIVOT bar position p)
      confirm_time                 (Timestamp the pivot becomes visible = close of bar p+N)
      pivot_high_level             (high[p]  where is_swing_high else NaN)
      pivot_low_level              (low[p]   where is_swing_low  else NaN)
    Only pivots that have N full bars on BOTH sides are emitted (edges excluded).
    """
    high = htf["high"].to_numpy(float)
    low = htf["low"].to_numpy(float)
    close_time = htf["close_time"].to_numpy()  # datetime64
    m = len(htf)
    is_sh = np.zeros(m, dtype=bool)
    is_sl = np.zeros(m, dtype=bool)
    for p in range(N, m - N):
        hp = high[p]
        lp = low[p]
        # strict Williams fractal: pivot must exceed all N neighbours on each side
        left_h = high[p - N:p]
        right_h = high[p + 1:p + 1 + N]
        left_l = low[p - N:p]
        right_l = low[p + 1:p + 1 + N]
        if hp > left_h.max() and hp > right_h.max():
            is_sh[p] = True
        if lp < left_l.min() and lp < right_l.min():
            is_sl[p] = True
    confirm_time = np.full(m, np.datetime64("NaT"), dtype="datetime64[ns]")
    # confirmation = close-time of bar p+N (the last right-side bar closes then)
    for p in range(N, m - N):
        if is_sh[p] or is_sl[p]:
            confirm_time[p] = close_time[p + N]
    out = pd.DataFrame(
        {
            "is_swing_high": is_sh,
            "is_swing_low": is_sl,
            "confirm_time": confirm_time,
            "pivot_high_level": np.where(is_sh, high, np.nan),
            "pivot_low_level": np.where(is_sl, low, np.nan),
        },
        index=htf.index,
    )
    return out


def _broadcast(piv: pd.DataFrame, one_h_index: pd.DatetimeIndex, level_col: str, flag_col: str) -> pd.Series:
    """merge_asof-backward broadcast of a confirmed pivot level onto the 1H grid.
    Each 1H bar reads the most recent pivot whose confirm_time <= the 1H timestamp."""
    events = piv.loc[piv[flag_col]].copy()
    events = events[["confirm_time", level_col]].dropna(subset=["confirm_time"])
    events = events.sort_values("confirm_time")
    left = pd.DataFrame({"ts": one_h_index})
    merged = pd.merge_asof(
        left, events.rename(columns={"confirm_time": "ts"}),
        on="ts", direction="backward",
    )
    return pd.Series(merged[level_col].to_numpy(), index=one_h_index)


def build_htf_pivots(df1h: pd.DataFrame, freq: str, N: int) -> pd.DataFrame:
    """End-to-end: resample -> fractal detect -> broadcast. Returns a DataFrame on the
    1H index with columns swing_high_<tf> and swing_low_<tf> (lower-cased freq, e.g. 4h/1d)."""
    tf = freq.lower()
    htf = resample_htf(df1h, freq)
    piv = detect_fractal_pivots(htf, N)
    sh = _broadcast(piv, df1h.index, "pivot_high_level", "is_swing_high")
    sl = _broadcast(piv, df1h.index, "pivot_low_level", "is_swing_low")
    return pd.DataFrame({f"swing_high_{tf}": sh, f"swing_low_{tf}": sl}, index=df1h.index)


def reanchor_frame(df1h: pd.DataFrame, freq: str, N: int) -> pd.DataFrame:
    """Return a COPY of df1h with swing_low_50 / swing_high_50 REPLACED by the HTF pivots,
    so the (unchanged) build_structural_range machine anchors to the HTF swings.
    All other columns (ohlc, atr_14, volume, ema, wyckoff_spring_*) are preserved."""
    piv = build_htf_pivots(df1h, freq, N)
    tf = freq.lower()
    out = df1h.copy()
    out["swing_low_50"] = piv[f"swing_low_{tf}"].to_numpy()
    out["swing_high_50"] = piv[f"swing_high_{tf}"].to_numpy()
    return out


if __name__ == "__main__":
    store = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
    df = pd.read_parquet(store)
    for freq in ("4H", "1D"):
        htf = resample_htf(df, freq)
        print(f"\n{freq}: {len(htf):,} bars  {htf.index[0]} -> {htf.index[-1]}")
        for N in (3, 5):
            piv = detect_fractal_pivots(htf, N)
            nsh = int(piv["is_swing_high"].sum())
            nsl = int(piv["is_swing_low"].sum())
            bc = build_htf_pivots(df, freq, N)
            cov_l = bc[f"swing_low_{freq.lower()}"].notna().mean()
            print(f"  N={N}: {nsh} swing highs, {nsl} swing lows | "
                  f"1H-grid low coverage {cov_l*100:.1f}% | "
                  f"~{nsl/((htf.index[-1]-htf.index[0]).days/365):.0f} lows/yr")
