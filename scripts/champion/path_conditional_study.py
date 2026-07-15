#!/usr/bin/env python3
"""Path-conditional study — where does POST-ENTRY information live?

Entry features don't separate winners from losers (proven 5x). But winners and
losers diverge immediately after entry. This study quantifies that divergence
(Part 1: conditional outcome curves) to locate the strongest cell for the ONE
pre-registered policy family (Part 2, separate runner: early-weakness time-cut).

Part 1 (this script): for wick_trap and liquidity_compression positions across
train/mid/holdout —
  - P(win | reached +{0.25,0.5,1.0}R within {6,12,24,48}h)
  - P(win | MFE < 0.25R after {6,12,24,48}h)   (the "never got going" cells)
  - time-to-first +0.5R for winners vs losers
  - early MAE (first 12h) for winners vs losers
Paths reconstructed from V14L's own high/low between entry and exit (R = |entry-stop|).

Usage: python3 scripts/champion/path_conditional_study.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
STORE = REPO / "data/features_mtf/BTC_1H_FEATURES_V14L_LEVELS.parquet"
LOGS = {"wick_trap": REPO / "results/wick_trap_eww", "liquidity_compression": REPO / "results/lc_eww"}
WINDOWS = ["train", "mid", "holdout"]
R_LEVELS = [0.25, 0.5, 1.0]
HORIZONS = [6, 12, 24, 48]


def load_paths():
    v = pd.read_parquet(STORE, columns=["high", "low"])
    if v.index.tz is not None:
        v.index = v.index.tz_localize(None)
    return v


def positions(arch, window, px):
    df = pd.read_csv(LOGS[arch] / f"{window}_trades.csv")
    df["ts"] = pd.to_datetime(df["timestamp"], format="mixed").dt.tz_localize(None)
    df["ex"] = pd.to_datetime(df["exit_timestamp"], format="mixed").dt.tz_localize(None)
    g = df.groupby(["ts", "entry_price"]).agg(pnl=("pnl", "sum"), stop=("stop_loss", "first"),
                                              ex=("ex", "max")).reset_index()
    rows = []
    highs, lows, idx = px["high"].to_numpy(), px["low"].to_numpy(), px.index
    for _, p in g.iterrows():
        R = abs(p.entry_price - p.stop)
        if R <= 0 or pd.isna(p.ex):
            continue
        lo = idx.searchsorted(p.ts)
        hi = idx.searchsorted(p.ex, side="right")
        if hi <= lo:
            continue
        seg_hi = (highs[lo:hi] - p.entry_price) / R      # favorable excursion path (long)
        seg_lo = (p.entry_price - lows[lo:hi]) / R       # adverse excursion path
        # cumulative max favorable by hour-offset
        cum_mfe = np.maximum.accumulate(seg_hi)
        rec = {"win": p.pnl > 0, "pnl": p.pnl, "hold_h": len(seg_hi)}
        for r in R_LEVELS:
            hit = np.argmax(seg_hi >= r) if (seg_hi >= r).any() else None
            for h in HORIZONS:
                rec[f"reach{r}_by{h}"] = (hit is not None and hit < h)
        for h in HORIZONS:
            rec[f"mfe_at{h}"] = cum_mfe[min(h, len(cum_mfe)) - 1]
        rec["mae_12h"] = float(np.max(seg_lo[:12])) if len(seg_lo) else np.nan
        first_green = np.argmax(seg_hi >= 0.5) if (seg_hi >= 0.5).any() else None
        rec["t_first_halfR"] = first_green if first_green is not None else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    px = load_paths()
    for arch in LOGS:
        print(f"\n{'='*74}\n{arch.upper()}")
        frames = {w: positions(arch, w, px) for w in WINDOWS}
        for w, d in frames.items():
            print(f"  {w}: {len(d)} positions, WR {d.win.mean():.0%}")

        print(f"\n  P(win | reached +X R within H hours)   [train / mid / holdout]  (base rates above)")
        for r in R_LEVELS:
            line = f"    +{r}R:  "
            for h in HORIZONS:
                cells = []
                for w in WINDOWS:
                    d = frames[w]
                    m = d[d[f"reach{r}_by{h}"]]
                    cells.append(f"{m.win.mean():.0%}" if len(m) >= 8 else "--")
                line += f"by{h:>2}h {'/'.join(cells):<16}"
            print(line)

        print(f"\n  P(win | MFE < 0.25R after H hours)  — the 'never got going' cells")
        for h in HORIZONS:
            cells = []
            ns = []
            for w in WINDOWS:
                d = frames[w]
                m = d[d[f"mfe_at{h}"] < 0.25]
                cells.append(f"{m.win.mean():.0%}" if len(m) >= 8 else "--")
                ns.append(str(len(m)))
            print(f"    after {h:>2}h: {'/'.join(cells):<18} (n={'/'.join(ns)})")

        print(f"\n  time to first +0.5R (median hours): ", end="")
        for w in WINDOWS:
            d = frames[w]
            tw = d[d.win].t_first_halfR.median()
            tl = d[~d.win].t_first_halfR.median()
            print(f"{w} W={tw:.0f}h L={'never' if np.isnan(tl) else f'{tl:.0f}h'}  ", end="")
        print()
        print(f"  early MAE first-12h (median R): ", end="")
        for w in WINDOWS:
            d = frames[w]
            print(f"{w} W={d[d.win].mae_12h.median():.2f} L={d[~d.win].mae_12h.median():.2f}  ", end="")
        print()


if __name__ == "__main__":
    main()
