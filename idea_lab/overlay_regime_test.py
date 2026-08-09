"""
REGIME-OVERLAY REUSE TEST (study only) — does a coarse causal bear flag we ALREADY own
plug the trend-continuation door's one defect (bear-market-RALLY breakout leak, add.48)?

We do NOT build a new detector. We test two coarse, causal, higher-timeframe bear flags:
  FLAG-SLOPE-K : ema200 is FALLING over the trailing K daily bars (ema200[i] < ema200[i-K]).
                 Targets exactly the leak — price ABOVE ema200 (door fires) but the medium
                 trend is DOWN (bear-market rally). Pre-registered K in {20, 60}. Causal.
  FLAG-STORE   : BTC only — the existing V22_CTX regime_label / regime_risk_off_prob column
                 (the CMI regime service we already have). Answers "does the thing we own work?"

For each asset: door baseline vs door∩(regime NOT bear). Success = the removed trades are the
bear-window LOSERS and the kept trades retain the winners (PF stays >=1.5, ideally improves).
NO threshold fishing: flags fixed before looking at results.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
from trend_continuation_door import TrendContinuationDoor, build_daily_sensors, resample_daily
from xasset_spx_port import load_spx

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
BTC_1H = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
XA = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
      "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/xasset")

BEAR_WINDOWS = {
    "BTC":  [("2018", "2018-01-01", "2018-12-15"), ("2022", "2021-11-11", "2022-12-31"),
             ("2025-26H1", "2025-01-01", "2026-06-10")],
    "SPX":  [("dotcom", "2000-03-24", "2002-10-09"), ("GFC", "2007-10-10", "2009-03-09"),
             ("COVID", "2020-02-19", "2020-03-23"), ("2022", "2022-01-01", "2022-10-12")],
    "NDX":  [("dotcom", "2000-03-24", "2002-10-09"), ("GFC", "2007-10-10", "2009-03-09"),
             ("COVID", "2020-02-19", "2020-03-23"), ("2022", "2022-01-01", "2022-10-12")],
    "GOLD": [("2011-15", "2011-09-06", "2015-12-17"), ("2020-22", "2020-08-07", "2022-09-26")],
}


def stats_of(trades):
    if not trades:
        return dict(n=0, WR=0, PF=0.0, PnL=0.0)
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in trades:
        e += t["pnl"]; eq.append(e)
    s = compute_stats(trades, eq, INITIAL_CASH)
    return dict(n=s["n"], WR=s["WR"], PF=s["PF"], PnL=s["PnL"])


def fmt(s):
    pf = "inf" if s["PF"] == float("inf") else f"{s['PF']:.2f}"
    return f"n={s['n']:>3}  WR={s['WR']*100:>4.0f}%  PF={pf:>5}  PnL=${s['PnL']:>+10,.0f}"


def in_any_bear(ts, windows):
    for _, a, b in windows:
        if pd.Timestamp(a) <= ts <= pd.Timestamp(b):
            return True
    return False


def run_one(name, df, sr, bj, eye, store_regime=None):
    print("\n" + "=" * 96)
    print(f"# {name}")
    print("=" * 96)
    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label="TC")
    trades = res["trades"]
    idx = df.index
    ecol = "ema_200" if "ema_200" in df.columns else ("ema200" if "ema200" in df.columns else None)
    ema = df[ecol].to_numpy(dtype=float) if ecol else None
    pos_of = {t: i for i, t in enumerate(idx)}
    windows = BEAR_WINDOWS[name]

    print(f"  door baseline           : {fmt(stats_of(trades))}")
    bl = [t for t in trades if in_any_bear(t['entry_time'], windows)]
    print(f"  of which in BEAR windows: {fmt(stats_of(bl))}   <- the leak (should be net-losers)")

    for K in (20, 60):
        kept, removed = [], []
        for t in trades:
            p = pos_of.get(t["entry_time"])
            bear = (p is not None and p >= K and ema is not None
                    and ema[p] < ema[p - K])   # 200-EMA falling over trailing K bars
            (removed if bear else kept).append(t)
        sk, sr_ = stats_of(kept), stats_of(removed)
        rb = [t for t in removed if in_any_bear(t['entry_time'], windows)]
        kb = [t for t in kept if in_any_bear(t['entry_time'], windows)]
        print(f"\n  FLAG-SLOPE-{K} (ema200 falling over {K}d = bear):")
        print(f"    KEPT (regime OK)      : {fmt(sk)}")
        print(f"    REMOVED (regime bear) : {fmt(sr_)}   ({len(rb)}/{len(removed)} were in bear windows)")
        print(f"    bear-window leak after: {fmt(stats_of(kb))}   (was {fmt(stats_of(bl))})")

    if store_regime is not None:
        # existing CMI regime_label / risk_off from the store, mapped to entry_time
        kept, removed = [], []
        for t in trades:
            lab = store_regime.get(t["entry_time"])
            bear = lab in ("bear", "risk_off", "crisis")
            (removed if bear else kept).append(t)
        kb = [t for t in kept if in_any_bear(t['entry_time'], windows)]
        print(f"\n  FLAG-STORE (existing CMI regime_label == bear/risk_off/crisis):")
        print(f"    labels seen at entries: {sorted(set(store_regime.get(t['entry_time'],'?') for t in trades))}")
        print(f"    KEPT                  : {fmt(stats_of(kept))}")
        print(f"    REMOVED               : {fmt(stats_of(removed))}")
        print(f"    bear-window leak after: {fmt(stats_of(kb))}")


def main():
    print("REGIME-OVERLAY REUSE TEST — plug the door's bear-rally leak with a flag we already own.")
    print("Pre-registered flags: ema200-falling (K=20,60) all assets; existing CMI regime_label (BTC).\n")

    # BTC (daily) + existing store regime_label
    btc = pd.read_parquet(BTC_1H)
    dfb, srb, bjb, eyeb = build_daily_sensors(
        resample_daily(btc[["open", "high", "low", "close", "volume"]]))
    store_reg = None
    if "regime_label" in btc.columns:
        lab = btc["regime_label"].dropna()
        print(f"  [store regime_label values: {lab.astype(str).value_counts().head(6).to_dict()}]")
        # as-of map: for each daily sensor bar, last known 1H regime_label at/before it
        aligned = lab.reindex(lab.index.union(dfb.index)).ffill().reindex(dfb.index)
        store_reg = {ts: aligned.get(ts) for ts in dfb.index}
    run_one("BTC", dfb, srb, bjb, eyeb, store_regime=store_reg)

    for tag, fn in (("SPX", "SPX_1D.parquet"), ("NDX", "NDX_1D.parquet"), ("GOLD", "GOLD_1D.parquet")):
        raw = load_spx(f"{XA}/{fn}")
        df, sr, bj, eye = build_daily_sensors(raw)
        run_one(tag, df, sr, bj, eye)


if __name__ == "__main__":
    main()
