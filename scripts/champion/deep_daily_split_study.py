#!/usr/bin/env python3
"""Deep-daily alignment split study — Phase 1 of the HTF-done-right plan.

V16 (feature REPLACEMENT) failed the deploy gate; this tests the same
information ADDITIVELY: do the deep-daily Wyckoff direction scores separate
winners from losers on the champions' UNCHANGED V15 trade populations?

PRE-REGISTERED (2026-07-21, before results):
  split   = sign(deep_bullish - deep_bearish) at the entry hour
            Aligned (bull>bear) / Opposed (bear>bull) / Neutral (equal)
  bar     = Aligned PF >= Opposed PF in BOTH wfo_train AND holdout, n>=30/cell
  next    = pass -> V17 additive store + 1.25x aligned sizing boost (Rule 9)
  fail    = campaign ends, verdict recorded, deep-daily gate stays OFF

Usage: python3 scripts/champion/deep_daily_split_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V16 = REPO / "data/features_mtf/BTC_1H_FEATURES_V16_DEEPDAILY.parquet"
CHAMPIONS = ["wick_trap", "order_block_retest", "liquidity_compression"]
WINDOWS = ["wfo_train", "holdout_2025_26"]
MIN_N = 30


def pf(x: pd.Series) -> float:
    gp = x[x > 0].sum()
    gl = -x[x < 0].sum()
    return gp / gl if gl > 0 else float("inf")


def main() -> None:
    deep = pd.read_parquet(
        V16, columns=["tf1d_wyckoff_bullish_score", "tf1d_wyckoff_bearish_score"])
    if deep.index.tz is not None:
        deep.index = deep.index.tz_localize(None)
    diff = deep["tf1d_wyckoff_bullish_score"].fillna(0) - \
        deep["tf1d_wyckoff_bearish_score"].fillna(0)
    state = pd.Series("neutral", index=deep.index)
    state[diff > 0] = "aligned"
    state[diff < 0] = "opposed"

    overall_pass = {}
    for arch in CHAMPIONS:
        print(f"\n=== {arch} ===")
        window_pass = []
        for w in WINDOWS:
            f = REPO / f"results/champion_v15/{arch}/{w}/trade_log.csv"
            t = pd.read_csv(f)
            t["ts"] = pd.to_datetime(t["timestamp"], format="mixed") \
                .dt.tz_localize(None).dt.floor("h")
            pos = t.groupby("position_id").agg(pnl=("pnl", "sum"), ts=("ts", "first"))
            pos["state"] = pos["ts"].map(state).fillna("neutral")
            base = pf(pos["pnl"])
            cells = {}
            for s in ["aligned", "opposed", "neutral"]:
                sub = pos[pos["state"] == s]["pnl"]
                cells[s] = (len(sub), pf(sub) if len(sub) else float("nan"),
                            sub.sum())
            line = f"  {w:16s} base n={len(pos):>3} PF={base:.2f}"
            for s in ["aligned", "opposed", "neutral"]:
                n, p, tot = cells[s]
                p_s = f"{p:.2f}" if p == p and p != float("inf") else str(p)
                line += f" | {s} n={n:>3} PF={p_s} ${tot:>7,.0f}"
            a_n, a_pf, _ = cells["aligned"]
            o_n, o_pf, _ = cells["opposed"]
            ok = (a_n >= MIN_N and o_n >= MIN_N and a_pf >= o_pf)
            line += f"  -> {'PASS' if ok else 'fail'}"
            window_pass.append(ok)
            print(line)
        overall_pass[arch] = all(window_pass)

    print("\n" + "=" * 70)
    print("PRE-REGISTERED VERDICT (aligned>=opposed in BOTH windows, n>=30/cell):")
    for arch, p in overall_pass.items():
        print(f"  {arch:24s} {'PASS -> Phase 2 candidate' if p else 'FAIL/insufficient'}")
    if not any(overall_pass.values()):
        print("  Campaign ends here per kill condition; deep-daily gate stays OFF.")


if __name__ == "__main__":
    main()
