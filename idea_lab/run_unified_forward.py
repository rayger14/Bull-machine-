"""
THE UNIFIED ARCHETYPE — LIVE FORWARD-TEST (no look-ahead).  STUDY ONLY.

Descriptive forensics: how would the standalone unified archetype (add.43, branch
study/unified-archetype) have ACTUALLY behaved over the live-collected window
(~2026-02 -> 2026-08), generating its OWN trades on CAUSAL fixed sensors + REAL
klines, with NO look-ahead?  This is NOT a re-optimization; the spec is frozen
exactly as add.43.  OOS-B (2025-01 -> 2026-06-10, PF 0.15) already covers most of
this window; this shows the freshest data concretely (incl. the post-store tail).

NO-LOOK-AHEAD DATA CONSTRUCTION
------------------------------
  * Sensors (1D N=5 structural range + Bojan-low) are built ONCE, forward, over the
    FULL combined causal history (2018-01-01 -> 2026-08-06).  A value at bar t uses
    only data <= t (the modules are causal; slicing after a full forward pass ==
    building per-slice, without any era-boundary reset).
  * FEATURE PROVENANCE per column (combined frame):
      OHLC/volume/atr_14 : STORE (V22_CTX) through 2026-06-10; Binance.US klines_1h
                           (matches store to ~0.01% mean in the 4,392-bar overlap)
                           for the 2026-06-11 -> 2026-08-06 TAIL.
      ema_200, swing_50  : STORE through 2026-06-10; computed causally over the
                           combined series for the TAIL only (store values untouched).
      fib_time_confluence: STORE through 2026-06-10; the SERVER'S REAL LIVE feature
                           logs (results/coinbase_paper/live_features/*.jsonl, the
                           true no-look-ahead record of what the engine computed live)
                           for 2026-06-18 -> tail; the 06-11..06-17 gap -> 0 (FLAGGED).
                           R4/time is a CONVICTION TIER (sizing only), never a gate.
      stables_rot_rising : STORE through 2026-06-10; NOT logged live (only USDT.D /
                           USDC.D) -> TAIL defaults to 0 (permissive). FLAGGED; a
                           sensitivity (stables=1 on tail) is printed.
"""
from __future__ import annotations
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from backtester import run_backtest, run_selftest, INITIAL_CASH, RISK_PCT
from unified_archetype import UnifiedArchetype, build_sensors

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
STORE = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
SCRATCH = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
           "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/poc_live_forensics")
KLINES = f"{SCRATCH}/klines_1h.parquet"
SERVER_TAIL = f"{SCRATCH}/server_tail_features.json"
CHAMP_TRADES = f"{SCRATCH}/live_trades.json"

STORE_END = pd.Timestamp("2026-06-10 23:00:00")
LIVE_START = pd.Timestamp("2026-02-15 00:00:00")   # champion live paper began 2026-02-15
LIVE_END = pd.Timestamp("2026-08-06 23:00:00")     # klines end
FEAT_COLS = ["open", "high", "low", "close", "volume", "atr_14",
             "swing_high_50", "swing_low_50", "ema_200",
             "stables_rot_rising", "fib_time_confluence"]


def build_combined_frame():
    store = pd.read_parquet(STORE)
    store = store.loc[store.index <= STORE_END, FEAT_COLS].copy()

    kl = pd.read_parquet(KLINES)
    kl.index = kl.index.tz_convert("UTC").tz_localize(None)
    kl = kl[kl.index > STORE_END].copy()          # tail only

    # combined index = store bars + tail kline bars
    comb = pd.DataFrame(index=store.index.union(kl.index))
    for c in FEAT_COLS:
        comb[c] = np.nan
    comb.loc[store.index, FEAT_COLS] = store.values
    # tail OHLCV + atr from klines
    for c in ["open", "high", "low", "close", "volume", "atr_14"]:
        comb.loc[kl.index, c] = kl[c].values

    tail = comb.index > STORE_END

    # ema_200 continuous causal ewm; fill TAIL only (store values kept intact)
    ema_full = comb["close"].ewm(span=200, adjust=False).mean()
    comb.loc[tail, "ema_200"] = ema_full[tail]
    # swing_50 rolling causal over combined; fill TAIL only
    sh = comb["high"].rolling(50, min_periods=10).max()
    sl = comb["low"].rolling(50, min_periods=10).min()
    comb.loc[tail, "swing_high_50"] = sh[tail]
    comb.loc[tail, "swing_low_50"] = sl[tail]

    # fib_time_confluence: real server live values for the tail
    srv = pd.DataFrame(json.load(open(SERVER_TAIL)))
    srv["ts"] = pd.to_datetime(srv["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
    srv = srv.dropna(subset=["ts"]).drop_duplicates("ts").set_index("ts").sort_index()
    fib_srv = srv["fib_time_confluence"].reindex(comb.index)
    comb.loc[tail, "fib_time_confluence"] = fib_srv[tail]
    fib_gap = tail & comb["fib_time_confluence"].isna()
    comb.loc[fib_gap, "fib_time_confluence"] = 0.0    # 06-11..06-17 gap -> tier default
    # stables not logged live -> permissive default 0 on tail (flagged; sensitivity below)
    comb.loc[tail, "stables_rot_rising"] = 0.0

    # cross-check server-logged ema_200 vs our continuous ema on the tail overlap
    ema_srv = srv["ema_200"].reindex(comb.index)
    both = tail & ema_srv.notna() & comb["ema_200"].notna()
    if both.sum():
        d = ((comb.loc[both, "ema_200"] - ema_srv[both]) / ema_srv[both] * 100).abs()
        print(f"[cross-check] tail ema_200 lab-vs-server-live: n={int(both.sum())} "
              f"mean|Δ|={d.mean():.3f}%  max|Δ|={d.max():.3f}%")

    n_tail = int(tail.sum())
    n_fibgap = int(fib_gap.sum())
    print(f"[coverage] combined bars={len(comb)}  store<= {STORE_END.date()}  "
          f"tail(klines)={n_tail}  fib-gap(06-11..06-17)={n_fibgap} bars -> fib=0")
    return comb


def fmt(s):
    pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  "
            f"avgR={s['avgR']:+.3f}  PnL=${s['PnL']:>10,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def run_variant(sub, sr_s, bj_s, variant, conv):
    strat = UnifiedArchetype(sub, sr_s, bj_s, variant=variant, conviction=conv)
    res = run_backtest(sub, strat, risk_pct=RISK_PCT,
                       label=f"{variant}|{'conv' if conv else 'flat'}")
    res["entries_log"] = strat.entries_log
    return res


def falling_knife_check(trades, sub):
    """For each entry: below ema_200? did it bounce (V-reversal: reclaim entry within
    168h) or keep falling (stopped)? Report entry high-water & low-water in R."""
    closes = sub["close"]; highs = sub["high"]; lows = sub["low"]; ema = sub["ema_200"]
    rows = []
    for t in trades:
        et = t["entry_time"]; ef = t["entry_fill"]; stop = t["stop0"]
        R = ef - stop
        pos = sub.index.get_loc(et)
        below = bool(closes.iloc[pos] < ema.iloc[pos])
        fwd = slice(pos + 1, min(pos + 1 + 168, len(sub)))
        fh = highs.iloc[fwd]; fl = lows.iloc[fwd]
        mfe_R = (fh.max() - ef) / R if len(fh) and R > 0 else np.nan   # max fav excursion
        mae_R = (ef - fl.min()) / R if len(fl) and R > 0 else np.nan   # max adverse excursion
        reclaimed = bool(len(fh) and fh.max() >= ef + 1.0 * R)         # +1R "bounce"
        rows.append({
            "entry_time": et, "below_ema200": below, "R_price": R,
            "mfe_R": mfe_R, "mae_R": mae_R, "vrev_1R_168h": reclaimed,
            "exit_reason": t["exit_reason"], "R": t["R"], "pnl": t["pnl"],
            "bars_held": t["bars_held"], "risk_mult": t["meta"].get("_rmult"),
        })
    return pd.DataFrame(rows)


def champion_actual(a, b):
    """Aggregate the champion book's ACTUAL live paper trades (fill records) to
    net positions over [a,b], keyed by position_id (fallback: entry+archetype)."""
    d = json.load(open(CHAMP_TRADES))
    df = pd.DataFrame(d)
    df["te"] = pd.to_datetime(df["timestamp_entry"]).dt.tz_localize(None)
    df["tx"] = pd.to_datetime(df["timestamp_exit"]).dt.tz_localize(None)
    df = df[(df["te"] >= a) & (df["te"] <= b)].copy()
    df["pid"] = df["position_id"].astype(str)
    blank = df["pid"].str.len() == 0
    df.loc[blank, "pid"] = (df.loc[blank, "archetype"] + "|" +
                            df.loc[blank, "te"].astype(str) + "|" +
                            df.loc[blank, "direction"])
    grp = df.groupby("pid").agg(
        entry=("te", "min"), exit=("tx", "max"),
        arch=("archetype", "first"), dirn=("direction", "first"),
        pnl=("pnl", "sum"),
    ).reset_index()
    n = len(grp)
    pnls = grp["pnl"].to_numpy()
    wins = pnls[pnls > 0].sum(); losses = -pnls[pnls < 0].sum()
    wr = float((pnls > 0).mean()) if n else 0.0
    pf = float(wins / losses) if losses > 1e-9 else float("inf")
    return {
        "n_positions": n, "n_fills": len(df), "WR": wr, "PF": pf,
        "PnL": float(pnls.sum()), "longs": int((grp.dirn == "long").sum()),
        "shorts": int((grp.dirn == "short").sum()),
    }, grp


def main():
    comb = build_combined_frame()
    if not run_selftest(comb):
        raise SystemExit("SELF-TEST FAILED")

    print("\nBuilding fixed sensors (1D N=5 structural range + Bojan-low) on full causal history...")
    sr, bj = build_sensors(comb)

    mask = (comb.index >= LIVE_START) & (comb.index <= LIVE_END)
    sub = comb.loc[mask]; sr_s = sr.loc[mask]; bj_s = bj.loc[mask]

    print("\n" + "=" * 96)
    print(f"THE UNIFIED ARCHETYPE — LIVE FORWARD-TEST  [{LIVE_START.date()} -> {LIVE_END.date()}]")
    print("=" * 96)
    print(f"Costs: commission 2bps/side, slippage 3bps/side (once in fill). Base risk "
          f"{RISK_PCT*100:.0f}%/trade, start ${INITIAL_CASH:,.0f}. One position at a time.")
    print("Conviction = 1.5x risk when fib_time_confluence>0 OR a Bojan-low zone is active.")

    variants = {}
    for variant in ("struct", "naive"):
        for conv in (False, True):
            variants[(variant, conv)] = run_variant(sub, sr_s, bj_s, variant, conv)

    print("\n### Live-window aggregates (all four spec variants)")
    for (variant, conv), res in variants.items():
        tag = f"{variant:6s} {'conv' if conv else 'flat'}"
        print(f"  {tag}: {fmt(res['stats'])}")
        print(f"           exits {dict(Counter(t['exit_reason'] for t in res['trades']))}")

    # attach conviction risk_mult into trade meta for the KO table
    head = variants[("struct", False)]      # headline = struct/flat (matches add.43 PF)
    conv_res = variants[("struct", True)]
    for t, e in zip(head["trades"], head["entries_log"]):
        t["meta"]["_rmult"] = e["risk_mult"]

    # ---- TRADE-BY-TRADE LIST (headline struct/flat) ----
    print("\n" + "=" * 96)
    print("TRADE-BY-TRADE LIST  (headline: struct / flat — the add.43 PF geometry)")
    print("=" * 96)
    hdr = (f"{'#':>2} {'entry_time':<16} {'entry':>8} {'stop':>8} "
           f"{'exit_time':<16} {'exit_reason':>14} {'barsH':>5} {'R':>6} {'pnl$':>9} "
           f"{'<ema200':>7} {'sweep_pos':>9} {'fib>0':>5} {'boj':>3}")
    print(hdr)
    for k, (t, e) in enumerate(zip(head["trades"], head["entries_log"]), 1):
        below = "yes" if sub["close"].loc[t["entry_time"]] < sub["ema_200"].loc[t["entry_time"]] else "no"
        print(f"{k:>2} {str(t['entry_time']):<16} {t['entry_fill']:>8.0f} {t['stop0']:>8.0f} "
              f"{str(t['exit_time']):<16} {t['exit_reason']:>14} {t['bars_held']:>5} "
              f"{t['R']:>6.2f} {t['pnl']:>9.0f} {below:>7} {e['struct_range_pos']:>9.2f} "
              f"{('Y' if e['time_present'] else '-'):>5} {('Y' if e['bojan_present'] else '-'):>3}")
    if not head["trades"]:
        print("  (no unified entries in the live window)")

    # ---- FALLING-KNIFE CHECK ----
    print("\n" + "=" * 96)
    print("FALLING-KNIFE CHECK (headline struct/flat): below-EMA200 entries, V-reversal (>=+1R "
          "within 168h) vs kept falling")
    print("=" * 96)
    fk = falling_knife_check(head["trades"], sub)
    if len(fk):
        below = fk[fk.below_ema200]
        print(f"  entries total={len(fk)}  below_ema200={len(below)} "
              f"({100*len(below)/len(fk):.0f}%)")
        print(f"  below-EMA200 subset: V-reversal(+1R/168h)={int(below.vrev_1R_168h.sum())}  "
              f"kept-falling(stopped)={int((below.exit_reason.str.startswith('stop')).sum())}")
        print(f"  avg MFE={fk.mfe_R.mean():+.2f}R  avg MAE={fk.mae_R.mean():+.2f}R  "
              f"win-rate={(fk.R>0).mean()*100:.0f}%")
        print(fk[["entry_time", "below_ema200", "vrev_1R_168h", "mfe_R", "mae_R",
                  "exit_reason", "R"]].to_string(index=False))

    # ---- REGIME CHARACTER ----
    print("\n" + "=" * 96)
    print("REGIME CHARACTER of the live window (BTC 1H close path)")
    print("=" * 96)
    for lbl, a, b in [("warmup   2025-12-10..2026-02-14", "2025-12-10", "2026-02-14"),
                      ("live-core 2026-02-15..2026-06-10", "2026-02-15", "2026-06-10"),
                      ("tail     2026-06-11..2026-08-06", "2026-06-11", "2026-08-06")]:
        s = comb.loc[a:b, "close"]
        e = comb.loc[a:b, "ema_200"]
        if len(s):
            below = (s < e).mean() * 100
            print(f"  {lbl}: first={s.iloc[0]:>7.0f} last={s.iloc[-1]:>7.0f} "
                  f"min={s.min():>7.0f} max={s.max():>7.0f} chg={100*(s.iloc[-1]/s.iloc[0]-1):+6.1f}%  "
                  f"bars<EMA200={below:.0f}%")

    # ---- CHAMPION ACTUAL (same window) ----
    print("\n" + "=" * 96)
    print(f"CHAMPION BOOK — ACTUAL live paper trades over [{LIVE_START.date()} -> {LIVE_END.date()}]")
    print("  (full 16-archetype junk book at full live size; aggregated fill-records -> positions)")
    print("=" * 96)
    champ, grp = champion_actual(LIVE_START, LIVE_END)
    print(f"  positions={champ['n_positions']} (from {champ['n_fills']} fill-records)  "
          f"longs={champ['longs']} shorts={champ['shorts']}")
    print(f"  WR={champ['WR']*100:.1f}%  PF={champ['PF']:.2f}  PnL=${champ['PnL']:,.0f}")

    # ---- SIDE BY SIDE ----
    print("\n" + "=" * 96)
    print("SIDE BY SIDE (same live window)")
    print("=" * 96)
    h = head["stats"]
    print(f"  UNIFIED (hypothetical, struct/flat): "
          f"n={h['n']}  WR={h['WR']*100:.1f}%  PF={h['PF']:.2f}  PnL=${h['PnL']:,.0f}  "
          f"MaxDD={h['MaxDD_pct']:.1f}%")
    print(f"  CHAMPION (actual live, full book)  : "
          f"n={champ['n_positions']}  WR={champ['WR']*100:.1f}%  PF={champ['PF']:.2f}  "
          f"PnL=${champ['PnL']:,.0f}")

    # sensitivity: stables=1 on tail (blocks tail entries) — does it change anything?
    comb2 = comb.copy()
    tailm = comb2.index > STORE_END
    comb2.loc[tailm, "stables_rot_rising"] = 1.0
    sr2, bj2 = build_sensors(comb2)
    m2 = (comb2.index >= LIVE_START) & (comb2.index <= LIVE_END)
    res2 = run_variant(comb2.loc[m2], sr2.loc[m2], bj2.loc[m2], "struct", False)
    print("\n[sensitivity] TAIL stables_rot_rising=1 (blocks tail longs): "
          f"n={res2['stats']['n']} PF={res2['stats']['PF']:.2f} "
          f"PnL=${res2['stats']['PnL']:,.0f}  (vs default-0 headline n={h['n']})")


if __name__ == "__main__":
    main()
