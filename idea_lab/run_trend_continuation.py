"""
TREND-CONTINUATION DOOR -- cross-asset validation harness (STUDY ONLY, add.48).
See trend_continuation_door.py for the pre-registered spec. NOTHING SHIPS.

Assets (all DAILY exec, WEEKLY N=5 struct range, PRICE-ONLY eye, IDENTICAL params):
  BTC (1H V22_CTX -> resampled 1D), SPX 1D, NDX 1D, GOLD 1D.  SPX 1H = bonus.
Per asset: self-test parity (must be 0.00%), per-regime table with above/below-EMA200
and fires-per-bear, CPCV subsample stability, + the DEAD-SPRING baseline contrast.
Forward test: BTC 2026-02-15 -> data end (2026-06-10).
"""
from __future__ import annotations
import os
import sys
import itertools
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
from trend_continuation_door import (
    TrendContinuationDoor, DeadSpringDoor, build_daily_sensors, resample_daily,
)
from run_xasset_spx import selftest_on
from xasset_spx_port import load_spx

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
XA = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
      "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/xasset")
BTC_1H = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"

# ------------------------------------------------------------------ regimes
SPX_REGIMES = [
    ("1990s bull",     "1990-01-01", "2000-03-23", "bull"),
    ("dotcom bear",    "2000-03-24", "2002-10-09", "BEAR"),
    ("2003-07 bull",   "2002-10-10", "2007-10-09", "bull"),
    ("GFC bear",       "2007-10-10", "2009-03-09", "BEAR"),
    ("2009-2020 bull", "2009-03-10", "2020-02-19", "bull"),
    ("COVID crash",    "2020-02-20", "2020-04-30", "BEAR"),
    ("2020-21 bull",   "2020-05-01", "2021-12-31", "bull"),
    ("2022 bear",      "2022-01-01", "2022-10-12", "BEAR"),
    ("2023-2026 bull", "2022-10-13", "2026-12-31", "bull"),
]
NDX_REGIMES = SPX_REGIMES  # same US-equity regime calendar
GOLD_REGIMES = [
    ("2000-08 bull",   "2000-08-30", "2008-03-17", "bull"),
    ("2008 GFC drop",  "2008-03-18", "2008-11-20", "BEAR"),
    ("2009-11 bull",   "2008-11-21", "2011-09-05", "bull"),
    ("2011-15 bear",   "2011-09-06", "2015-12-17", "BEAR"),
    ("2016-19 base",   "2015-12-18", "2018-08-16", "bull"),
    ("2018-20 bull",   "2018-08-17", "2020-08-06", "bull"),
    ("2020-22 correct","2020-08-07", "2022-11-03", "BEAR"),
    ("2023-26 bull",   "2022-11-04", "2026-12-31", "bull"),
]
BTC_REGIMES = [
    ("2018 bear",      "2018-01-01", "2018-12-15", "BEAR"),
    ("2019-21 bull",   "2018-12-16", "2021-11-10", "bull"),
    ("2022 bear",      "2021-11-11", "2022-12-31", "BEAR"),
    ("2023-24 bull",   "2023-01-01", "2024-12-31", "bull"),
    ("2025-26H1 mkdn", "2025-01-01", "2026-06-10", "BEAR"),
]
BTC_ERAS = [  # coarse 3-era headline view
    ("2018-2022",  "2018-01-01", "2022-12-31"),
    ("2023-24",    "2023-01-01", "2024-12-31"),
    ("2025-26H1",  "2025-01-01", "2026-06-10"),
]


# ------------------------------------------------------------------ helpers
def fmt(s):
    pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  "
            f"avgR={s['avgR']:+.3f}  PnL=${s['PnL']:>11,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def exits(trades):
    return dict(Counter(t["exit_reason"] for t in trades))


def window_stats(trades, elog, a, b):
    ent = pd.DatetimeIndex([e["entry_time"] for e in elog])
    m = (ent >= a) & (ent <= b)
    idxs = np.where(m)[0]
    tr = [trades[i] for i in idxs]; lg = [elog[i] for i in idxs]
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in tr:
        e += t["pnl"]; eq.append(e)
    s = compute_stats(tr, eq, INITIAL_CASH)
    bl = [0, 0.0]; ab = [0, 0.0]
    for t, el in zip(tr, lg):
        k = bl if el["below_ema200"] else ab
        k[0] += 1; k[1] += t["pnl"]
    return s, bl, ab, tr


def run_door(strat_cls, df, sr, bj, eye, variant="struct", conviction=False):
    strat = strat_cls(df, sr, bj, eye, variant=variant, conviction=conviction)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label=strat_cls.__name__)
    res["entries_log"] = strat.entries_log
    return res


# ------------------------------------------------------------------ CPCV
def cpcv_stability(trades, K=6, test_m=2):
    """No-fit CPCV = subsample stability. Partition trades (by entry order) into K
    contiguous time-blocks; for every C(K, test_m) combination, pool the test-block
    trades and compute PF. Report mean/std PF, and fraction of folds with PF>1 / >=1.5.
    (No parameter fitting -> this measures cross-subperiod robustness, the relevant
    anti-overfit check for a FIXED-param door. Purge/embargo is moot with no training.)
    Returns a dict; None if n too small."""
    n = len(trades)
    if n < K:              # cannot even fill K blocks
        return None
    blocks = np.array_split(np.arange(n), K)
    pfs = []; ns = []
    for combo in itertools.combinations(range(K), test_m):
        idx = np.concatenate([blocks[c] for c in combo])
        tr = [trades[i] for i in idx]
        pnls = np.array([t["pnl"] for t in tr])
        w = pnls[pnls > 0].sum(); l = -pnls[pnls < 0].sum()
        pf = (w / l) if l > 1e-9 else (float("inf") if w > 0 else 0.0)
        pfs.append(pf); ns.append(len(tr))
    fin = [p for p in pfs if np.isfinite(p)]
    return {
        "K": K, "test_m": test_m, "n_folds": len(pfs),
        "n_per_fold": int(np.median(ns)),
        "mean_PF": float(np.mean(fin)) if fin else float("inf"),
        "std_PF": float(np.std(fin)) if fin else 0.0,
        "frac_PF_gt1": float(np.mean([p > 1 for p in pfs])),
        "frac_PF_ge1p5": float(np.mean([p >= 1.5 for p in pfs])),
        "n_inf": len(pfs) - len(fin),
    }


# ------------------------------------------------------------------ per-asset
def run_asset(name, df, sr, bj, eye, regimes, selftest_label):
    print("\n" + "#" * 100)
    print(f"# {name}")
    print("#" * 100)
    ok = selftest_on(df, selftest_label)
    print(f"  vol_zero% = {100*(df['volume']==0).mean():.2f}%  (door is volume-independent)")

    # --- headline door: struct/flat ---
    tc = run_door(TrendContinuationDoor, df, sr, bj, eye, "struct", False)
    tc_naive = run_door(TrendContinuationDoor, df, sr, bj, eye, "naive", False)
    tc_conv = run_door(TrendContinuationDoor, df, sr, bj, eye, "struct", True)
    ds = run_door(DeadSpringDoor, df, sr, bj, eye, "struct", False)

    print("\n-- TREND-CONTINUATION door (breakout-retest) --")
    print(f"  struct/flat (HEADLINE): {fmt(tc['stats'])}")
    print(f"                          exits {exits(tc['trades'])}")
    print(f"  naive/flat            : {fmt(tc_naive['stats'])}")
    print(f"  struct/conv (1.5x)    : {fmt(tc_conv['stats'])}")
    bl = sum(1 for e in tc["entries_log"] if e["below_ema200"])
    ab = len(tc["entries_log"]) - bl
    tot = bl + ab
    print(f"  EMA200 split          : above={ab} below={bl}  "
          f"ABOVE-share={100*ab/tot:.0f}%" if tot else "  (no trades)")

    print("\n-- DEAD-SPRING baseline (M1 dip-buyer, contrast) --")
    print(f"  struct/flat           : {fmt(ds['stats'])}")
    dbl = sum(1 for e in ds["entries_log"] if e["below_ema200"])
    dab = len(ds["entries_log"]) - dbl
    dtot = dbl + dab
    print(f"  EMA200 split          : above={dab} below={dbl}  "
          f"ABOVE-share={100*dab/dtot:.0f}%" if dtot else "  (no trades)")

    # --- per-regime table (headline door) + fires-per-bear ---
    print("\n" + "=" * 100)
    print(f"PER-REGIME (headline struct/flat)  --  BEAR windows test the self-regime-filter claim")
    print("=" * 100)
    print(f"{'regime':<16}{'kind':<6}{'n':>4}{'WR':>7}{'PF':>7}{'avgR':>8}{'PnL':>12}"
          f"{'MaxDD':>8}  {'below-EMA200':>20}  {'above-EMA200':>20}")
    bear_fires = 0; bear_windows = 0; bear_trades = []
    for rn, a, b, kind in regimes:
        s, blk, abk, tr = window_stats(tc["trades"], tc["entries_log"], a, b)
        pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"{rn:<16}{kind:<6}{s['n']:>4}{s['WR']*100:>6.1f}%{pf_s:>7}{s['avgR']:>+8.3f}"
              f"{s['PnL']:>12,.0f}{s['MaxDD_pct']:>7.1f}%  "
              f"{'n='+str(blk[0]):>6} ${blk[1]:>+10,.0f}  {'n='+str(abk[0]):>6} ${abk[1]:>+10,.0f}")
        if kind == "BEAR":
            bear_windows += 1
            bear_fires += s["n"]
            bear_trades += tr
            if s["n"] == 0:
                pass
    # bear stand-down summary
    stood_down = sum(1 for rn, a, b, kind in regimes if kind == "BEAR"
                     and window_stats(tc["trades"], tc["entries_log"], a, b)[0]["n"] == 0)
    print(f"\n  SELF-REGIME-FILTER: {stood_down}/{bear_windows} bear windows had ZERO fires "
          f"({100*stood_down/bear_windows:.0f}% stand-down).  total bear fires={bear_fires}")
    if bear_trades:
        eq = [INITIAL_CASH]; e = INITIAL_CASH
        for t in bear_trades:
            e += t["pnl"]; eq.append(e)
        sb = compute_stats(bear_trades, eq, INITIAL_CASH)
        print(f"  pooled BEAR-window trades: {fmt(sb)}  (if it fires+loses here, claim is FALSE)")

    # --- CPCV ---
    cp = cpcv_stability(tc["trades"], K=6, test_m=2)
    print("\n-- CPCV (no-fit subsample stability, K=6 blocks, test_m=2) --")
    if cp is None:
        print(f"  n={len(tc['trades'])} too small for K=6 -> CPCV NOT computable; report is directional only.")
        cp2 = cpcv_stability(tc["trades"], K=3, test_m=1)
        if cp2:
            print(f"  fallback K=3/test_m=1: mean_PF={cp2['mean_PF']:.2f} std={cp2['std_PF']:.2f} "
                  f"frac_PF>1={cp2['frac_PF_gt1']:.0%} n/fold~{cp2['n_per_fold']} (inf folds={cp2['n_inf']})")
    else:
        print(f"  folds={cp['n_folds']}  n/fold~{cp['n_per_fold']}  mean_PF={cp['mean_PF']:.2f} "
              f"std={cp['std_PF']:.2f}  frac_PF>1={cp['frac_PF_gt1']:.0%}  "
              f"frac_PF>=1.5={cp['frac_PF_ge1p5']:.0%}  (inf folds={cp['n_inf']})")
    return {"tc": tc, "ds": ds, "stood_down": stood_down, "bear_windows": bear_windows,
            "bear_fires": bear_fires, "cpcv": cp}


# ------------------------------------------------------------------ loaders
def load_daily_asset(path):
    raw = load_spx(path)                      # generic OHLCV loader (ts index)
    return build_daily_sensors(raw)


def load_btc_daily():
    btc = pd.read_parquet(BTC_1H)
    daily = resample_daily(btc[["open", "high", "low", "close", "volume"]])
    return build_daily_sensors(daily)


# ------------------------------------------------------------------ forward
def forward_btc(df, sr, bj, eye):
    print("\n" + "#" * 100)
    print("# FORWARD TEST -- BTC 2026-02-15 -> data end (2026-06-10). No-look-ahead: sensors")
    print("#   are built on the FULL causal history; we only REPORT entries in the window.")
    print("#   Expect near-zero fires in a markdown; any fire must be above-EMA200.")
    print("#" * 100)
    tc = run_door(TrendContinuationDoor, df, sr, bj, eye, "struct", False)
    a, b = pd.Timestamp("2026-02-15"), pd.Timestamp("2026-06-10")
    s, blk, abk, tr = window_stats(tc["trades"], tc["entries_log"], a, b)
    print(f"  fires in window: {fmt(s)}")
    print(f"  below-EMA200 n={blk[0]} ${blk[1]:+,.0f}   above-EMA200 n={abk[0]} ${abk[1]:+,.0f}")
    if s["n"] == 0:
        print("  -> ZERO fires = correct stand-down in the markdown (self-regime-filter holds).")


# ------------------------------------------------------------------ main
def main():
    print("COSTS: commission 2bps/side, slippage 3bps/side. Risk 1%/trade, start "
          f"${INITIAL_CASH:,.0f}. One position at a time. HEADLINE = struct/flat (rmult=1.0).")
    print("PARAMS IDENTICAL ACROSS ALL ASSETS. Daily exec, weekly N=5 struct range, price-only eye.")

    results = {}
    # BTC (resampled daily)
    dfb, srb, bjb, eyeb = load_btc_daily()
    results["BTC"] = run_asset("BTC daily (1H V22_CTX -> 1D)", dfb, srb, bjb, eyeb,
                               BTC_REGIMES, "BTC_1D")
    # coarse BTC eras
    print("\n  -- BTC coarse 3-era (headline struct/flat) --")
    tc = results["BTC"]["tc"]
    for en, a, b in BTC_ERAS:
        s, blk, abk, _ = window_stats(tc["trades"], tc["entries_log"], a, b)
        print(f"    {en:<12}: {fmt(s)}  above-EMA200 n={abk[0]} below n={blk[0]}")

    # SPX / NDX / GOLD daily
    dfs, srs, bjs, eyes = load_daily_asset(f"{XA}/SPX_1D.parquet")
    results["SPX"] = run_asset("SPX daily 1990-2026", dfs, srs, bjs, eyes, SPX_REGIMES, "SPX_1D")

    dfn, srn, bjn, eyen = load_daily_asset(f"{XA}/NDX_1D.parquet")
    results["NDX"] = run_asset("NDX daily 1990-2026", dfn, srn, bjn, eyen, NDX_REGIMES, "NDX_1D")

    dfg, srg, bjg, eyeg = load_daily_asset(f"{XA}/GOLD_1D.parquet")
    results["GOLD"] = run_asset("GOLD daily 2000-2026 (KEY uncorrelated test)",
                                dfg, srg, bjg, eyeg, GOLD_REGIMES, "GOLD_1D")

    # forward test
    forward_btc(dfb, srb, bjb, eyeb)

    # ---- cross-asset verdict summary ----
    print("\n" + "=" * 100)
    print("CROSS-ASSET SUMMARY (headline struct/flat, IDENTICAL params)")
    print("=" * 100)
    print(f"{'asset':<8}{'n':>5}{'WR':>7}{'PF':>7}{'PnL':>13}{'above-EMA200':>14}"
          f"{'bear stand-down':>18}{'CPCV meanPF':>13}")
    n_pf_ge = 0; n_assets = 0
    for k in ("BTC", "SPX", "NDX", "GOLD"):
        r = results[k]; s = r["tc"]["stats"]
        el = r["tc"]["entries_log"]
        ab = sum(1 for e in el if not e["below_ema200"]); tot = len(el)
        absh = f"{100*ab/tot:.0f}%" if tot else "n/a"
        sd = f"{r['stood_down']}/{r['bear_windows']}"
        pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        cpm = f"{r['cpcv']['mean_PF']:.2f}" if r["cpcv"] else "n<K"
        print(f"{k:<8}{s['n']:>5}{s['WR']*100:>6.1f}%{pf_s:>7}{s['PnL']:>13,.0f}{absh:>14}"
              f"{sd:>18}{cpm:>13}")
        n_assets += 1
        if pf != float("inf") and pf >= 1.5:
            n_pf_ge += 1
        elif pf == float("inf") and s["n"] > 0:
            n_pf_ge += 1
    print(f"\nPRE-REGISTERED PASS RULE: PF>=1.5 on >=3/4 assets AND >=80% bear stand-down "
          f"AND CPCV meanPF>1 where computable.")
    print(f"  -> PF>=1.5 assets: {n_pf_ge}/4")


if __name__ == "__main__":
    main()
