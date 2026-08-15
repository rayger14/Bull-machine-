"""
N5 FAILED-COUNT — 3-POINT TRUNCATION / CAUSALITY CHECK (add.65 discipline)
==========================================================================
N5 is a NEW rolling sensor (failed-breakout count in trailing 90d). It must be
causal & non-repainting: the failed_count as-of a fire at bar i must be IDENTICAL
whether computed from data truncated at i, or i+50, or i+200 (adding future data
cannot change a past reading). This re-derives the count three ways per sampled
fire and asserts equality. STUDY ONLY.
"""
from __future__ import annotations
import os, sys, random
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_wi_batch7 as W
from xasset_spx_port import load_spx
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from backtester import run_backtest


def count_at_i_truncated(closes, es, ed, ru, ts, i, trunc_end):
    """Recompute N5 failed_count for a fire at bar i using ONLY bars [0..trunc_end].
    trunc_end >= i. A truly causal sensor gives the same answer for any trunc_end>=i."""
    ce = closes[:trunc_end + 1]; ee = es[:trunc_end + 1]; de = ed[:trunc_end + 1]
    re = ru[:trunc_end + 1]
    breaks = W.confirmed_break_events(ee, de, re)
    failed = W.precompute_failed_breaks(ce, breaks, W.N5_FAIL_BARS)
    fb = np.array([f[0] for f in failed], dtype=int)
    fr = np.array([f[1] for f in failed], dtype=int)
    if len(fb) == 0:
        return 0
    t0 = ts[i] - np.timedelta64(W.N5_WINDOW_DAYS, "D")
    m = (ts[fb] >= t0) & (ts[fb] < ts[i]) & (fr <= i)
    return int(m.sum())


def main():
    uni = W.breadth300_universe()
    random.seed(11)
    sample = random.sample(uni, 12)
    print("N5 3-POINT TRUNCATION CHECK (trunc at i, i+50, i+200) — must all equal the sensor count")
    print(f"{'asset':<8}{'fires':>6}{'checked':>8}{'mismatches':>11}{'max|Δ|':>8}")
    grand_mm = 0; grand_checked = 0
    for key, path, sec in sample:
        df_raw = load_spx(path); df, sr, bj, eye = build_daily_sensors(df_raw)
        strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
        res = run_backtest(df, strat, label=key)
        closes = df["close"].to_numpy(float)
        es = eye["eye_state"].to_numpy(dtype=object); ed = eye["eye_dir"].to_numpy(dtype=object)
        ru = eye["range_upper_1d"].to_numpy(float)
        ts = np.asarray(df.index.values, dtype="datetime64[ns]"); n = len(df)
        idx = df.index
        # sensor reference count (full-series, resolution<=i) — same as run_wi_batch7
        breaks = W.confirmed_break_events(es, ed, ru)
        failed = W.precompute_failed_breaks(closes, breaks, W.N5_FAIL_BARS)
        fb = np.array([f[0] for f in failed], dtype=int); fr = np.array([f[1] for f in failed], dtype=int)
        mm = 0; checked = 0; maxd = 0
        for t in res["trades"]:
            i = idx.get_loc(t["entry_time"])
            if len(fb):
                t0 = ts[i] - np.timedelta64(W.N5_WINDOW_DAYS, "D")
                ref = int(((ts[fb] >= t0) & (ts[fb] < ts[i]) & (fr <= i)).sum())
            else:
                ref = 0
            for te in (i, min(i + 50, n - 1), min(i + 200, n - 1)):
                cnt = count_at_i_truncated(closes, es, ed, ru, ts, i, te)
                checked += 1
                if cnt != ref:
                    mm += 1; maxd = max(maxd, abs(cnt - ref))
        grand_mm += mm; grand_checked += checked
        print(f"{key:<8}{len(res['trades']):>6}{checked:>8}{mm:>11}{maxd:>8}")
    print(f"\nGRAND: {grand_checked} checks, {grand_mm} mismatches -> "
          f"{'CAUSAL / NO REPAINT (PASS)' if grand_mm == 0 else 'REPAINT DETECTED (FAIL)'}")


if __name__ == "__main__":
    main()
