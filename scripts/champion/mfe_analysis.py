#!/usr/bin/env python3
"""Maximum Favorable Excursion (MFE) analysis — "did trades run further than we kept?"

For every entered trade, reconstruct the in-trade price path (V14 OHLC between
entry and exit) and measure, in R-multiples (R = entry-to-stop distance):
  MFE_R     how far the trade went IN OUR FAVOR before exit (the opportunity)
  MAE_R     how far it went against us (heat taken)
  realized_R what we actually captured (from pnl_pct / stop_distance_pct)
  capture   realized_R / MFE_R  (1.0 = kept it all; 0.3 = left 70% on the table)

Answers the user's question directly:
  - Were targets too far? -> look at how often MFE reached the take-profit Rs.
  - Did we not take profit enough? -> capture ratio + "gave-back" winners.
  - GAVE-BACK losers: trades that reached >=1R favorable then exited a LOSER
    (winners we mismanaged) — the clearest "tighten/trail" signal.

Per-archetype + overall. Read-only. Hypothesis-generating for the exit study.
Usage: python3 scripts/champion/mfe_analysis.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CHAMP = REPO / "results/champion_v14"
V14 = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
OUT = REPO / "results/champion_v14/mfe_analysis.json"


def main():
    ohlc = pd.read_parquet(V14, columns=["high", "low"])
    if getattr(ohlc.index, "tz", None) is not None:
        ohlc.index = ohlc.index.tz_localize(None)
    highs, lows, idx = ohlc["high"].to_numpy(), ohlc["low"].to_numpy(), ohlc.index

    archetypes = sorted({Path(p).parents[1].name for p in glob.glob(str(CHAMP / "*/full/trade_log.csv"))})
    results, all_rows = [], []

    for a in archetypes:
        df = pd.read_csv(CHAMP / a / "full" / "trade_log.csv")
        if df.empty:
            continue
        df["en"] = pd.to_datetime(df["timestamp"], format="mixed").dt.tz_localize(None)
        df["ex"] = pd.to_datetime(df["exit_timestamp"], format="mixed").dt.tz_localize(None)
        rows = []
        for _, t in df.iterrows():
            entry, stop = float(t["entry_price"]), float(t["stop_loss"])
            R = abs(entry - stop)
            if R <= 0 or pd.isna(t["ex"]):
                continue
            lo = idx.searchsorted(t["en"]); hi = idx.searchsorted(t["ex"], side="right")
            if hi <= lo:
                continue
            seg_hi, seg_lo = highs[lo:hi], lows[lo:hi]
            exitp = float(t["exit_price"])
            if t["direction"] == "long":
                mfe = (np.nanmax(seg_hi) - entry) / R
                mae = (entry - np.nanmin(seg_lo)) / R
                realized = (exitp - entry) / R   # price-based, same units as MFE
            else:
                mfe = (entry - np.nanmin(seg_lo)) / R
                mae = (np.nanmax(seg_hi) - entry) / R
                realized = (entry - exitp) / R
            # per-trade capture of the available favorable move (clip to [0,1] view)
            capture = realized / mfe if mfe > 0.05 else np.nan
            giveback = mfe - realized   # R left on the table vs the high-water mark
            rows.append({"win": float(t["pnl_pct"]) > 0, "mfe": mfe, "mae": mae,
                         "realized": realized, "capture": capture, "giveback": giveback})
        if len(rows) < 20:
            continue
        r = pd.DataFrame(rows)
        winners, losers = r[r.win], r[~r.win]
        rec = {
            "archetype": a, "n": len(r), "win_rate": round(float(r.win.mean()), 3),
            "median_MFE_R_winners": round(float(winners.mfe.median()), 2) if len(winners) else None,
            "median_realized_R_winners": round(float(winners.realized.median()), 2) if len(winners) else None,
            "median_capture_winners": round(float(winners.capture.median()), 2) if len(winners) else None,
            "median_giveback_R_winners": round(float(winners.giveback.median()), 2) if len(winners) else None,
            # losers that ran >=1R / >=2R in our favor before exiting red = mismanaged winners
            "losers_reached_1R_pct": round(float((losers.mfe >= 1.0).mean()) * 100, 1) if len(losers) else None,
            # how often did ANY trade have the room for the 2R / 3R scale-out targets?
            "pct_reached_2R": round(float((r.mfe >= 2.0).mean()) * 100, 1),
            "pct_reached_3R": round(float((r.mfe >= 3.0).mean()) * 100, 1),
            "median_MAE_R": round(float(r.mae.median()), 2),
        }
        results.append(rec)
        all_rows.extend(rows)

    # overall
    R = pd.DataFrame(all_rows)
    winners = R[R.win]; losers = R[~R.win]
    overall = {
        "archetype": "ALL", "n": len(R), "win_rate": round(float(R.win.mean()), 3),
        "median_MFE_R_winners": round(float(winners.mfe.median()), 2),
        "median_realized_R_winners": round(float(winners.realized.median()), 2),
        "median_capture_winners": round(float(winners.capture.median()), 2),
        "median_giveback_R_winners": round(float(winners.giveback.median()), 2),
        "losers_reached_1R_pct": round(float((losers.mfe >= 1.0).mean()) * 100, 1),
        "pct_reached_2R": round(float((R.mfe >= 2.0).mean()) * 100, 1),
        "pct_reached_3R": round(float((R.mfe >= 3.0).mean()) * 100, 1),
    }

    print(f"{'archetype':<24}{'n':>5}{'WR':>5}{'MFE_w':>7}{'real_w':>7}{'capt%':>6}{'givbk':>7}"
          f"{'L>=1R%':>7}{'any2R%':>7}{'any3R%':>7}")
    for r in results + [overall]:
        cap = r.get('median_capture_winners')
        print(f"{r['archetype']:<24}{r['n']:>5}{r['win_rate']*100:>4.0f}%"
              f"{str(r.get('median_MFE_R_winners')):>7}{str(r.get('median_realized_R_winners')):>7}"
              f"{(str(round(cap*100))+'%' if cap is not None else 'n/a'):>6}{str(r.get('median_giveback_R_winners')):>7}"
              f"{str(r.get('losers_reached_1R_pct')):>7}{str(r.get('pct_reached_2R')):>7}{str(r.get('pct_reached_3R')):>7}")

    OUT.write_text(json.dumps({"per_archetype": results, "overall": overall}, indent=2, default=str))
    print(f"\nLegend: MFE_w=median high-water mark of WINNERS (R) | real_w=median R KEPT | "
          f"capt%=per-trade kept/available | givbk=R given back from peak | "
          f"L>=1R%=losers that were +1R before losing | any2R/3R%=trades with room for far targets")
    print(f"report -> {OUT}")


if __name__ == "__main__":
    main()
