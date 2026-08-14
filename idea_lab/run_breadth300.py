"""
BREADTH-AT-SCALE (add.64; STUDY ONLY; nothing ships)
====================================================
Runs the FROZEN add.48 trend-continuation door, IDENTICAL params, on ALL current
S&P 500 constituents with >=8yr adjusted daily history. Answers the owner's "many
trades that are good" via BREADTH (the free parameter), and characterizes the
PORTFOLIO the owner would actually experience (concurrency, episodes, monthly
cadence, bear behavior) under pre-registered operational caps.

See breadth300_PREREGISTRATION.txt. NO per-asset tuning; run ALL, report ALL.
Headline = struct/flat rmult=1.0. Costs 2bps+3bps/side, risk fixed 1% ($1000/trade), $100k.
"""
from __future__ import annotations
import os, sys, json, itertools, warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from backtester import run_backtest, INITIAL_CASH, RISK_PCT
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from xasset_spx_port import load_spx
from run_xasset_spx import selftest_on

SCROOT = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
B3 = os.path.join(SCROOT, "breadth300")
RAW = json.load(open(os.path.join(SCROOT, "sp500_raw.json")))
SECTORS = RAW["sectors"]

MIN_YEARS = 8.0
FIXED = 0.01 * INITIAL_CASH          # fixed 1% risk = $1000/trade (matches add.62 pooling)
RECENT = pd.Timestamp("2018-01-01")  # survivorship-primary recent window
CAP_TOTAL = 20                       # P1 pre-registered
CAP_SECTOR = 5                       # P2 pre-registered

EQUITY_BEARS = [
    ("2018Q4", "2018-10-01", "2018-12-24"),
    ("COVID",  "2020-02-19", "2020-04-07"),
    ("2022",   "2022-01-03", "2022-10-13"),
]


def integrity(path):
    df = pd.read_parquet(path)
    ts = pd.DatetimeIndex(pd.to_datetime(df["ts"]))
    gaps = ts.to_series().diff().dt.days.dropna()
    big = int((gaps > 7).sum())
    nan_ohlc = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    return {"n": len(df), "span_yrs": (ts.max() - ts.min()).days / 365.25,
            "gaps_gt7d": big, "nan_ohlc": nan_ohlc,
            "start": str(ts.min().date()), "end": str(ts.max().date())}


def boot_meanR_ci(Rs, n_boot=10000, seed=42):
    Rs = np.asarray(Rs, float)
    if len(Rs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = Rs[rng.integers(0, len(Rs), size=(n_boot, len(Rs)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def pf_of(Rs):
    Rs = np.asarray(Rs, float)
    w = Rs[Rs > 0].sum(); l = -Rs[Rs < 0].sum()
    return (w / l) if l > 1e-9 else (float("inf") if w > 0 else 0.0)


def dd_of(pnls_ordered):
    eq = INITIAL_CASH + np.cumsum(np.asarray(pnls_ordered, float))
    eq = np.concatenate([[INITIAL_CASH], eq])
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def run_asset(key, path, do_selftest):
    df_raw = load_spx(path)
    df, sr, bj, eye = build_daily_sensors(df_raw)
    st = selftest_on(df, key) if do_selftest else None
    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label=key)
    elog = strat.entries_log
    trades = res["trades"]; stats = res["stats"]
    tot = len(elog)
    ab = sum(1 for e in elog if not e["below_ema200"])
    Rs = [t["R"] for t in trades]
    tr = [{"entry_time": t["entry_time"], "exit_time": t["exit_time"], "R": t["R"],
           "below_ema200": e["below_ema200"]} for t, e in zip(trades, elog)]
    return {"key": key, "sector": SECTORS.get(key, "?"),
            "n": stats["n"], "WR": stats["WR"], "PF": stats["PF"], "avgR": stats["avgR"],
            "PnL_fixed": float(np.sum(Rs) * FIXED),
            "above_share": (100 * ab / tot if tot else 0.0),
            "selftest": st, "trades": tr,
            "span_yrs": (df.index[-1] - df.index[0]).days / 365.25}


def main():
    print("=" * 100)
    print("BREADTH-AT-SCALE (add.64) — FROZEN add.48 door on S&P 500 constituents (>=8yr, auto_adjust)")
    print("Costs 2bps+3bps/side, fixed 1% risk ($1000/trade), $100k. NO per-asset tuning. Run ALL.")
    print("=" * 100)

    files = sorted(f[:-8] for f in os.listdir(B3) if f.endswith(".parquet"))
    # ----- screens -----
    universe = []; excluded = []
    for key in files:
        integ = integrity(os.path.join(B3, f"{key}.parquet"))
        if integ["span_yrs"] < MIN_YEARS:
            excluded.append((key, f"{integ['span_yrs']:.1f}yr")); continue
        if integ["gaps_gt7d"] > 3 or integ["nan_ohlc"] > 0:
            excluded.append((key, f"integ gaps={integ['gaps_gt7d']} nan={integ['nan_ohlc']}")); continue
        universe.append(key)
    print(f"\nSCREENS: {len(files)} fetched -> {len(universe)} pass (>= {MIN_YEARS}yr + integrity); "
          f"{len(excluded)} excluded")

    # ----- referee parity on a 10-name sample -----
    import random
    random.seed(7)
    sample = sorted(random.sample(universe, min(10, len(universe))))
    print(f"\nREFEREE PARITY self-test sample (10 names): {sample}")

    results = {}
    for i, key in enumerate(universe):
        do_st = key in sample
        try:
            r = run_asset(key, os.path.join(B3, f"{key}.parquet"), do_st)
        except Exception as e:
            excluded.append((key, f"run err {e}")); continue
        results[key] = r
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(universe)} run", flush=True)

    st_results = {k: results[k]["selftest"] for k in sample if k in results}
    st_fail = [k for k, v in st_results.items() if v is False]
    print(f"\n  REFEREE PARITY: {len(st_results)} tested; "
          f"{'ALL 0.00% PASS' if not st_fail else 'FAIL: ' + ','.join(st_fail)}")

    # ============================================================ PART 1: per-name
    per_pf = []; per_pnl = []; per_n = []; above_all = []
    for k, r in results.items():
        if r["n"] > 0 and np.isfinite(r["PF"]):
            per_pf.append(r["PF"])
        per_pnl.append(r["PnL_fixed"]); per_n.append(r["n"]); above_all.append(r["above_share"])
    per_pf = np.array(per_pf); per_pnl = np.array(per_pnl); per_n = np.array(per_n)
    n_names = len(results)
    n_traded = sum(1 for r in results.values() if r["n"] > 0)
    pf_finite = np.array([r["PF"] for r in results.values() if r["n"] > 0 and np.isfinite(r["PF"])])
    pf_all_incl_inf = [r["PF"] for r in results.values() if r["n"] > 0]
    frac_pf_gt1 = np.mean([p > 1 for p in pf_all_incl_inf])
    frac_pf_gt15 = np.mean([p >= 1.5 for p in pf_all_incl_inf])

    print("\n" + "=" * 100)
    print("PART 1 — PER-NAME DISTRIBUTION (headline struct/flat, identical params)")
    print("=" * 100)
    print(f"  names run                 : {n_names}   (traded >=1: {n_traded})")
    print(f"  total trades (pooled)     : {int(per_n.sum())}")
    print(f"  per-name n: mean {per_n.mean():.1f}  median {np.median(per_n):.0f}  "
          f"max {per_n.max():.0f}  min {per_n.min():.0f}")
    print(f"  per-name PF (finite): median {np.median(pf_finite):.2f}  mean {pf_finite.mean():.2f}")
    print(f"  %% names PF>1  : {100*frac_pf_gt1:.1f}%   %% names PF>1.5: {100*frac_pf_gt15:.1f}%")
    print(f"  per-name PnL(1%R): median ${np.median(per_pnl):,.0f}  mean ${per_pnl.mean():,.0f}  "
          f"%% names PnL>0: {100*np.mean(per_pnl>0):.1f}%")
    # worst decile
    order = np.argsort(per_pnl)
    wd = order[:max(1, len(order)//10)]
    print(f"  worst PnL decile ({len(wd)} names): sum ${per_pnl[wd].sum():,.0f}  "
          f"mean ${per_pnl[wd].mean():,.0f}")
    # PF histogram
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    labels = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5", ">5(incl inf)"]
    hist = [0]*len(labels)
    for p in pf_all_incl_inf:
        for bi in range(len(bins)-1):
            if bins[bi] <= p < bins[bi+1]:
                hist[bi] += 1; break
    print("  PF histogram (traded names):")
    for lab, h in zip(labels, hist):
        print(f"      {lab:>13}: {h:4d}  {'#'*int(60*h/max(1,max(hist)))}")

    # ============================================================ PART 2: pooled
    def pooled_stats(trades):
        Rs = np.array([t["R"] for t in trades])
        pnls = Rs * FIXED
        ci = boot_meanR_ci(Rs)
        # equity curve ordered by exit
        order = sorted(range(len(trades)), key=lambda i: trades[i]["exit_time"])
        dd = dd_of([pnls[i] for i in order])
        return {"n": len(Rs), "PF": pf_of(Rs), "PnL": float(pnls.sum()),
                "meanR": float(Rs.mean()) if len(Rs) else 0.0, "ci": ci, "MaxDD": dd}

    all_trades = []
    for k, r in results.items():
        for t in r["trades"]:
            all_trades.append({**t, "key": k, "sector": r["sector"]})
    full = pooled_stats(all_trades)
    recent = pooled_stats([t for t in all_trades if pd.Timestamp(t["entry_time"]) >= RECENT])

    ev = sorted(all_trades, key=lambda x: x["entry_time"])
    span_yrs = (pd.Timestamp(max(t["exit_time"] for t in all_trades))
                - pd.Timestamp(min(t["entry_time"] for t in all_trades))).days / 365.25
    recent_yrs = (pd.Timestamp("2026-08-14") - RECENT).days / 365.25

    print("\n" + "=" * 100)
    print("PART 2 — POOLED (fixed 1% risk). FULL window + RECENT 2018-2026 (survivorship-primary)")
    print("=" * 100)
    for tag, s, yrs, nt in [("FULL   ", full, span_yrs, len(all_trades)),
                            ("RECENT ", recent, recent_yrs,
                             sum(1 for t in all_trades if pd.Timestamp(t["entry_time"]) >= RECENT))]:
        print(f"  {tag}: n={s['n']:5d}  PF {s['PF']:.2f}  PnL ${s['PnL']:>10,.0f}  "
              f"meanR {s['meanR']:+.3f}  boot95%CI [{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]  "
              f"MaxDD {s['MaxDD']:.2f}%  trades/yr {nt/yrs:.1f}")

    # per-year counts
    yc = Counter(pd.Timestamp(t["entry_time"]).year for t in all_trades)
    print("\n  PER-YEAR trade counts (pooled, uncapped):")
    yrs_sorted = sorted(yc)
    line = "   ".join(f"{y}:{yc[y]}" for y in yrs_sorted)
    print("    " + line)

    # per-sector clustering
    print("\n  PER-SECTOR (pooled): n, PF, PnL(1%R)")
    sec_tr = defaultdict(list)
    for t in all_trades:
        sec_tr[t["sector"]].append(t["R"])
    for sec in sorted(sec_tr, key=lambda s: -len(sec_tr[s])):
        Rs = np.array(sec_tr[sec])
        print(f"      {sec:<26}{len(Rs):5d}  PF {pf_of(Rs):5.2f}  ${Rs.sum()*FIXED:>10,.0f}")

    # mega-cap trio vs broad (survivorship check b)
    trio = ["AAPL", "MSFT", "NVDA"]
    trio_R = [t["R"] for t in all_trades if t["key"] in trio]
    broad_R = [t["R"] for t in all_trades if t["key"] not in trio]
    print(f"\n  SURVIVORSHIP CHECK (b) mega-cap trio vs broad universe:")
    print(f"      AAPL+MSFT+NVDA : n={len(trio_R):4d}  PF {pf_of(trio_R):.2f}  meanR {np.mean(trio_R):+.3f}")
    print(f"      broad (rest)   : n={len(broad_R):4d}  PF {pf_of(broad_R):.2f}  meanR {np.mean(broad_R):+.3f}")

    # ============================================================ PART 3: portfolio
    print("\n" + "=" * 100)
    print("PART 3 — PORTFOLIO REALITY (concurrency, episodes, monthly cadence, bear, caps)")
    print("=" * 100)

    # --- episode clustering (+-5d) ---
    episodes = []; cur = []
    for t in ev:
        if not cur:
            cur = [t]; continue
        if (pd.Timestamp(t["entry_time"]) - pd.Timestamp(cur[-1]["entry_time"])).days <= 5:
            cur.append(t)
        else:
            episodes.append(cur); cur = [t]
    if cur:
        episodes.append(cur)
    print(f"  EPISODES (+-5d cross-name): raw entries {len(all_trades)} -> "
          f"{len(episodes)} independent episodes")
    ep_recent = [ep for ep in episodes if pd.Timestamp(ep[0]["entry_time"]) >= RECENT]
    print(f"      recent-era (2018-26): raw {recent['n']} -> {len(ep_recent)} episodes "
          f"({recent['n']/max(1,recent_yrs):.1f} raw/yr, {len(ep_recent)/max(1,recent_yrs):.1f} ep/yr)")

    # --- concurrency on daily grid (UNCAPPED) ---
    def concurrency_series(trades, start, end):
        grid = pd.date_range(start, end, freq="D")
        cnt = np.zeros(len(grid), dtype=int)
        gi = {d: k for k, d in enumerate(grid)}
        ev2 = np.zeros(len(grid)+1, dtype=int)
        for t in trades:
            a = pd.Timestamp(t["entry_time"]).normalize()
            b = pd.Timestamp(t["exit_time"]).normalize()
            if b < start or a > end:
                continue
            a = max(a, start); b = min(b, end)
            ia = (a - start).days; ib = (b - start).days
            ev2[ia] += 1; ev2[ib+1] -= 1
        occ = np.cumsum(ev2)[:len(grid)]
        return grid, occ

    gstart = pd.Timestamp("2018-01-01"); gend = pd.Timestamp("2026-08-14")
    grid, occ = concurrency_series(all_trades, gstart, gend)
    print(f"\n  CONCURRENCY (UNCAPPED, daily grid 2018-2026): peak {occ.max()}  "
          f"median {int(np.median(occ))}  mean {occ.mean():.1f}  "
          f"95th pct {int(np.percentile(occ,95))}")
    print(f"      at 1% risk each, peak {occ.max()} concurrent => ~{occ.max()}% of equity at risk "
          f"simultaneously (gross), median ~{int(np.median(occ))}%")

    # --- capped portfolio (P1 max 20 total FCFS + P2 max 5/sector FCFS) ---
    open_pos = []  # (exit_ts, sector)
    capped = []; skipped_total = 0; skipped_sector = 0
    for t in ev:
        a = pd.Timestamp(t["entry_time"])
        open_pos = [op for op in open_pos if op[0] > a]
        n_open = len(open_pos)
        n_sec = sum(1 for op in open_pos if op[1] == t["sector"])
        if n_open >= CAP_TOTAL:
            skipped_total += 1; continue
        if n_sec >= CAP_SECTOR:
            skipped_sector += 1; continue
        capped.append(t); open_pos.append((pd.Timestamp(t["exit_time"]), t["sector"]))
    # hold-time (explains WHY the cap binds: long holds saturate slots)
    holds = np.array([(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).days
                      for t in all_trades])
    print(f"\n  HOLD TIME (calendar days/trade): mean {holds.mean():.0f}  median {int(np.median(holds))}  "
          f"90th {int(np.percentile(holds,90))}  max {holds.max()}  "
          f"(8-mo max_hold + measured-move runners => long holds saturate the book)")

    cap_R = np.array([t["R"] for t in capped])
    cap_pnls = cap_R * FIXED
    cap_order = sorted(range(len(capped)), key=lambda i: capped[i]["exit_time"])
    cap_dd = dd_of([cap_pnls[i] for i in cap_order])
    cap_ci = boot_meanR_ci(cap_R)
    cap_recent = [t for t in capped if pd.Timestamp(t["entry_time"]) >= RECENT]
    print(f"\n  CAPPED PORTFOLIO (P1 max {CAP_TOTAL} total + P2 max {CAP_SECTOR}/sector, FCFS):")
    print(f"      accepted {len(capped)} / {len(all_trades)}  (skipped total-cap {skipped_total}, "
          f"sector-cap {skipped_sector})")
    print(f"      PF {pf_of(cap_R):.2f}  PnL ${cap_pnls.sum():,.0f}  meanR {cap_R.mean():+.3f}  "
          f"CI [{cap_ci[0]:+.3f},{cap_ci[1]:+.3f}]  MaxDD {cap_dd:.2f}%")
    print(f"      cadence: full {len(capped)/span_yrs:.1f}/yr ; recent(2018-26) "
          f"{len(cap_recent)/recent_yrs:.1f}/yr")
    _, occ_cap = concurrency_series(capped, gstart, gend)
    print(f"      capped concurrency: peak {occ_cap.max()}  median {int(np.median(occ_cap))}")

    # --- cap-level sensitivity (what total-cap reaches 100/yr, at what DD/exposure) ---
    def cap_sim(cap_total, cap_sector=CAP_SECTOR):
        op = []; acc = []
        for t in ev:
            a = pd.Timestamp(t["entry_time"])
            op = [x for x in op if x[0] > a]
            if len(op) >= cap_total: continue
            if sum(1 for x in op if x[1] == t["sector"]) >= cap_sector: continue
            acc.append(t); op.append((pd.Timestamp(t["exit_time"]), t["sector"]))
        rec = [t for t in acc if pd.Timestamp(t["entry_time"]) >= RECENT]
        Rs = np.array([t["R"] for t in acc])
        o = sorted(range(len(acc)), key=lambda i: acc[i]["exit_time"])
        return len(rec)/recent_yrs, dd_of([Rs[i]*FIXED for i in o]), pf_of(Rs)
    print(f"\n  CAP-LEVEL SENSITIVITY (recent entries/yr | MaxDD | PF), per-sector cap held at {CAP_SECTOR}:")
    print(f"      (each open slot = 1%% gross risk; cap N => up to N%% of equity at risk)")
    for ct in [10, 20, 30, 40, 60, 100]:
        cyr, cdd, cpf = cap_sim(ct)
        print(f"      cap={ct:<4} => {cyr:5.1f}/yr   MaxDD {cdd:6.2f}%   PF {cpf:.2f}")

    # --- monthly cadence experience (capped, 2018-2026) ---
    mc = Counter(pd.Timestamp(t["entry_time"]).to_period("M") for t in cap_recent)
    months = pd.period_range("2018-01", "2026-08", freq="M")
    counts = np.array([mc.get(m, 0) for m in months])
    ge5 = int((counts >= 5).sum()); ge10 = int((counts >= 10).sum()); zero = int((counts == 0).sum())
    last12 = pd.period_range("2025-09", "2026-08", freq="M")
    last12_n = sum(mc.get(m, 0) for m in last12)
    last_month = mc.get(pd.Period("2026-07", freq="M"), 0) + mc.get(pd.Period("2026-08", freq="M"), 0)
    print(f"\n  MONTHLY CADENCE (CAPPED portfolio, {len(months)} months 2018-01..2026-08):")
    print(f"      trades/month: mean {counts.mean():.1f}  median {int(np.median(counts))}  "
          f"max {counts.max()}")
    print(f"      months >=5 entries: {ge5} ({100*ge5/len(months):.0f}%)  "
          f">=10: {ge10}  ZERO: {zero} ({100*zero/len(months):.0f}%)")
    print(f"      last 12 months entries: {last12_n}   last ~month(2026-07/08): {last_month}")

    # --- bear behavior ---
    print(f"\n  BEAR BEHAVIOR (uncapped door fires + capped-portfolio DD):")
    for name, a, b in EQUITY_BEARS:
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        fires = [t for t in all_trades if a <= pd.Timestamp(t["entry_time"]) <= b]
        fr = np.array([t["R"] for t in fires])
        pf_b = pf_of(fr) if len(fr) else 0.0
        print(f"      {name:<8}{a.date()}..{b.date()}: fires {len(fires):4d}  "
              f"PF {pf_b:5.2f}  PnL ${fr.sum()*FIXED:>9,.0f}  meanR {(fr.mean() if len(fr) else 0):+.3f}")
    # capped MaxDD through 2022
    cap_2022 = [t for t in capped if pd.Timestamp(t["entry_time"]).year <= 2022
                and pd.Timestamp(t["entry_time"]).year >= 2021]
    if cap_2022:
        c22o = sorted(range(len(cap_2022)), key=lambda i: cap_2022[i]["exit_time"])
        dd22 = dd_of([cap_2022[i]["R"]*FIXED for i in c22o])
        print(f"      capped MaxDD through 2021-2022 bear (fires in-window): {dd22:.2f}% (n={len(cap_2022)})")

    # ============================================================ PART 4: verdict
    V1 = recent["PF"] >= 1.5 and recent["ci"][0] > 0 and recent["n"] >= 300
    V2 = np.median(pf_finite) > 1
    V3 = frac_pf_gt1 >= 0.55
    V4 = cap_dd >= -15.0
    V5 = (len(cap_recent) / recent_yrs) >= 100
    print("\n" + "=" * 100)
    print("PART 4 — VERDICT vs PRE-REGISTERED RULE")
    print("=" * 100)
    print(f"  V1 recent PF>=1.5 & CI-lo>0 & n>=300 : {recent['PF']:.2f}, CI-lo {recent['ci'][0]:+.3f}, "
          f"n={recent['n']}  -> {'PASS' if V1 else 'FAIL'}")
    print(f"  V2 median per-name PF > 1            : {np.median(pf_finite):.2f}  -> {'PASS' if V2 else 'FAIL'}")
    print(f"  V3 >=55%% of names PF>1              : {100*frac_pf_gt1:.1f}%  -> {'PASS' if V3 else 'FAIL'}")
    print(f"  V4 capped-portfolio MaxDD <= 15%%    : {cap_dd:.2f}%  -> {'PASS' if V4 else 'FAIL'}")
    print(f"  V5 cadence >= 100 tr/yr (capped)     : {len(cap_recent)/recent_yrs:.1f}/yr  -> {'PASS' if V5 else 'FAIL'}")
    allpass = V1 and V2 and V3 and V4 and V5
    print(f"\n  BREADTH-AT-SCALE VALIDATES: {'YES (all 5)' if allpass else 'NO'}")

    # dump
    dump = {
        "screens": {"fetched": len(files), "universe": len(universe), "excluded_n": len(excluded)},
        "per_name": {k: {kk: vv for kk, vv in v.items() if kk != "trades"} for k, v in results.items()},
        "excluded": excluded[:60],
        "pooled_full": full, "pooled_recent": recent,
        "per_year": dict(yc),
        "per_sector": {sec: {"n": len(sec_tr[sec]), "PF": pf_of(sec_tr[sec]),
                             "PnL": float(np.sum(sec_tr[sec])*FIXED)} for sec in sec_tr},
        "trio_vs_broad": {"trio": {"n": len(trio_R), "PF": pf_of(trio_R), "meanR": float(np.mean(trio_R))},
                          "broad": {"n": len(broad_R), "PF": pf_of(broad_R), "meanR": float(np.mean(broad_R))}},
        "episodes": {"raw": len(all_trades), "independent": len(episodes),
                     "recent_raw": recent["n"], "recent_ep": len(ep_recent)},
        "concurrency_uncapped": {"peak": int(occ.max()), "median": int(np.median(occ)),
                                 "mean": float(occ.mean()), "p95": int(np.percentile(occ, 95))},
        "capped": {"accepted": len(capped), "skipped_total": skipped_total,
                   "skipped_sector": skipped_sector, "PF": pf_of(cap_R),
                   "PnL": float(cap_pnls.sum()), "meanR": float(cap_R.mean()),
                   "ci": cap_ci, "MaxDD": cap_dd, "peak_conc": int(occ_cap.max()),
                   "median_conc": int(np.median(occ_cap)),
                   "cadence_recent": len(cap_recent)/recent_yrs},
        "monthly": {"ge5": ge5, "ge10": ge10, "zero": zero, "n_months": len(months),
                    "last12": last12_n, "mean_per_month": float(counts.mean())},
        "verdict": {"V1": bool(V1), "V2": bool(V2), "V3": bool(V3), "V4": bool(V4),
                    "V5": bool(V5), "PASS": bool(allpass)},
        "selftest_sample": {k: st_results.get(k) for k in sample},
    }
    outp = os.path.join(B3, "breadth300_results.json")
    with open(outp, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\n  results dumped -> {outp}")


if __name__ == "__main__":
    main()
