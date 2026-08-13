"""
WIDE-BASKET VALIDATION of THE ONE STRATEGY door (add.62; STUDY ONLY; nothing ships)
===================================================================================
Runs the EXACT add.48 trend-continuation door, IDENTICAL params, on the widest honest
universe (29 markets across 6 families) and defines the pre-registered forward basket.
Reproduces the existing 13 (10 crypto cached + GOLD/NDX cached) and adds 16 new markets
(+SPX retry) fetched via yfinance (idea_lab/fetch_wide.py). NO per-asset tuning; run ALL,
report ALL. See idea_lab/wide_basket_PREREGISTRATION.txt (frozen before any fetch/outcome).

Headline = struct/flat rmult=1.0, exactly as add.48/54/59. Costs 2bps+3bps, risk 1%, $100k.
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

from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from xasset_spx_port import load_spx
from run_xasset_spx import selftest_on

SCROOT = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
WB  = os.path.join(SCROOT, "wide_basket")
OS_ = os.path.join(SCROOT, "one_strategy")   # cached crypto (add.54)
XA  = os.path.join(SCROOT, "xasset")         # cached GOLD/NDX (add.59)

MIN_YEARS = 4.0

# key -> (family, parquet_path).  PRE-REGISTERED (frozen).
UNIVERSE = {}
for s in ["BTC-USD","ETH-USD","SOL-USD","LTC-USD","XRP-USD","ADA-USD",
          "DOGE-USD","DOT-USD","AVAX-USD","LINK-USD"]:
    UNIVERSE[s] = ("crypto", os.path.join(OS_, f"{s}.parquet"))
UNIVERSE["GOLD"] = ("metal",        os.path.join(XA, "GOLD_1D.parquet"))
UNIVERSE["NDX"]  = ("equity-index", os.path.join(XA, "NDX_1D.parquet"))
NEW_FAMILY = {
    "SPX":"equity-index","DJI":"equity-index","RUT":"equity-index",
    "N225":"equity-index","DAX":"equity-index","FTSE":"equity-index",
    "SILVER":"metal","COPPER":"metal","PLATINUM":"metal",
    "OIL":"energy",
    "EURUSD":"fx","GBPUSD":"fx","USDJPY":"fx","AUDUSD":"fx",
    "AAPL":"single-stock","MSFT":"single-stock","NVDA":"single-stock",
}
for k, fam in NEW_FAMILY.items():
    UNIVERSE[k] = (fam, os.path.join(WB, f"{k}.parquet"))

# assets already mined in prior addenda (reproduced) vs genuinely NEW (self-test required)
REPRODUCED = set(["BTC-USD","ETH-USD","SOL-USD","LTC-USD","XRP-USD","ADA-USD",
                  "DOGE-USD","DOT-USD","AVAX-USD","LINK-USD","GOLD","NDX"])

# equity bear windows (index + single-stock); crypto uses its own calendar.
EQUITY_BEARS = [
    ("dotcom",  "2000-03-01", "2002-10-15"),
    ("GFC",     "2007-10-09", "2009-03-09"),
    ("COVID",   "2020-02-19", "2020-04-07"),
    ("2022",    "2022-01-03", "2022-10-13"),
]
CRYPTO_BEARS = [
    ("2018 bear", "2018-01-01", "2018-12-15"),
    ("2022 bear", "2021-11-10", "2022-12-31"),
    ("2025-26 mkdn", "2025-01-01", "2026-12-31"),
]


# ---------------------------------------------------------------------------- integrity
def integrity(df_raw: pd.DataFrame):
    if "ts" in df_raw.columns:
        ts = pd.to_datetime(df_raw["ts"])
    elif "timestamp" in df_raw.columns:
        ts = pd.to_datetime(df_raw["timestamp"])
    else:
        ts = df_raw.index
    ts = pd.DatetimeIndex(ts)
    gaps = ts.to_series().diff().dt.days.dropna()
    # business-day expectation ~ up to 4-day gaps (weekend+holiday) normal; >7 = real hole
    big = int((gaps > 7).sum())
    vol = df_raw["volume"].to_numpy(float)
    zerovol = float(np.mean(vol == 0))
    nan_ohlc = int(df_raw[["open","high","low","close"]].isna().any(axis=1).sum())
    return {"n": len(df_raw), "span_yrs": (ts.max()-ts.min()).days/365.25,
            "gaps_gt7d": big, "zerovol_pct": 100*zerovol, "nan_ohlc": nan_ohlc,
            "start": str(ts.min().date()), "end": str(ts.max().date())}


# ---------------------------------------------------------------------------- window
def window_pnl(trades, elog, a, b):
    ent = pd.DatetimeIndex([e["entry_time"] for e in elog])
    m = (ent >= pd.Timestamp(a)) & (ent <= pd.Timestamp(b))
    idx = np.where(m)[0]
    tr = [trades[i] for i in idx]
    pnl = sum(t["pnl"] for t in tr)
    return len(tr), pnl


def kblock_stability(trades, K=6, test_m=2):
    n = len(trades)
    if n < K:
        return None
    blocks = np.array_split(np.arange(n), K)
    pfs = []
    for combo in itertools.combinations(range(K), test_m):
        idx = np.concatenate([blocks[c] for c in combo])
        pnls = np.array([trades[i]["pnl"] for i in idx])
        w = pnls[pnls > 0].sum(); l = -pnls[pnls < 0].sum()
        pf = (w/l) if l > 1e-9 else (float("inf") if w > 0 else 0.0)
        pfs.append(pf)
    return {"n_folds": len(pfs),
            "frac_PF_gt1": float(np.mean([p > 1 for p in pfs])),
            "frac_PF_ge1p5": float(np.mean([p >= 1.5 for p in pfs]))}


def boot_meanR_ci(Rs, n_boot=10000, seed=42):
    Rs = np.asarray(Rs, float)
    if len(Rs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = Rs[rng.integers(0, len(Rs), size=(n_boot, len(Rs)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ---------------------------------------------------------------------------- per-asset
def run_asset(key, fam, path, do_selftest):
    df_raw = load_spx(path)
    integ = integrity(pd.read_parquet(path))
    df, sr, bj, eye = build_daily_sensors(df_raw)
    st_pass = None
    if do_selftest:
        st_pass = selftest_on(df, key)

    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res2 = run_backtest(df, strat, risk_pct=RISK_PCT, label=key)
    elog = strat.entries_log
    trades = res2["trades"]; stats = res2["stats"]

    ab = sum(1 for e in elog if not e["below_ema200"]); tot = len(elog)
    above_share = 100*ab/tot if tot else 0.0
    ab_pnl = sum(t["pnl"] for t,e in zip(trades,elog) if not e["below_ema200"])
    bl_pnl = sum(t["pnl"] for t,e in zip(trades,elog) if e["below_ema200"])

    bears = CRYPTO_BEARS if fam == "crypto" else (
            EQUITY_BEARS if fam in ("equity-index","single-stock") else None)
    bear_n = bear_pnl = 0; bear_defined = bears is not None
    if bears:
        for _, a, b in bears:
            n_, p_ = window_pnl(trades, elog, a, b)
            bear_n += n_; bear_pnl += p_

    Rs = [t["R"] for t in trades]
    return {
        "key": key, "family": fam, "integrity": integ, "selftest": st_pass,
        "n": stats["n"], "WR": stats["WR"], "PF": stats["PF"], "avgR": stats["avgR"],
        "PnL": stats["PnL"], "MaxDD": stats["MaxDD_pct"], "above_share": above_share,
        "above_pnl": ab_pnl, "below_pnl": bl_pnl, "below_n": tot-ab,
        "bear_defined": bear_defined, "bear_n": bear_n, "bear_pnl": bear_pnl,
        "exits": dict(Counter(t["exit_reason"] for t in trades)),
        "trades": [{"entry_time": e["entry_time"], "exit_time": t["exit_time"],
                    "R": t["R"], "pnl": t["pnl"], "below_ema200": e["below_ema200"]}
                   for t, e in zip(trades, elog)],
        "span_yrs": integ["span_yrs"],
    }


def main():
    print("="*100)
    print("WIDE-BASKET VALIDATION (add.62) — identical add.48 door on 29 markets, 6 families")
    print("Costs 2bps+3bps/side, risk 1%, $100k. Headline struct/flat rmult=1.0. NO per-asset tuning.")
    print("="*100)

    results = {}; excluded = []
    for key, (fam, path) in UNIVERSE.items():
        if not os.path.exists(path):
            excluded.append((key, "no data")); print(f"MISSING {key}"); continue
        do_st = key not in REPRODUCED   # self-test parity referee on every NEW asset
        r = run_asset(key, fam, path, do_st)
        if r["span_yrs"] < MIN_YEARS:
            excluded.append((key, f"{r['span_yrs']:.1f}yr"));
            print(f"EXCLUDE {key}: only {r['span_yrs']:.1f}yr"); continue
        results[key] = r

    # ----- Part 1: per-asset table -----
    print("\n" + "="*100)
    print("PART 1 — PER-ASSET (headline struct/flat, IDENTICAL params). integrity + door.")
    print("="*100)
    hdr = (f"{'asset':<9}{'fam':<13}{'yrs':>5}{'gap>7':>6}{'zVol%':>7}{'n':>4}"
           f"{'WR':>6}{'PF':>7}{'PnL':>10}{'MaxDD':>7}{'ab200':>7}{'bearN':>6}{'bearPnL':>9}{'ST':>5}")
    print(hdr)
    for key in UNIVERSE:
        if key not in results: continue
        r = results[key]; i = r["integrity"]
        pf = r["PF"]; pf_s = "inf" if pf==float("inf") else f"{pf:.2f}"
        st = "-" if r["selftest"] is None else ("0.00%" if r["selftest"] else "FAIL")
        bn = f"{r['bear_n']}" if r["bear_defined"] else "n/d"
        bp = f"{r['bear_pnl']:,.0f}" if r["bear_defined"] else "n/d"
        print(f"{key:<9}{r['family']:<13}{r['span_yrs']:>5.1f}{i['gaps_gt7d']:>6}"
              f"{i['zerovol_pct']:>7.1f}{r['n']:>4}{r['WR']*100:>5.0f}%{pf_s:>7}"
              f"{r['PnL']:>10,.0f}{r['MaxDD']:>6.1f}%{r['above_share']:>6.0f}%"
              f"{bn:>6}{bp:>11}{st:>7}")
    if excluded:
        print(f"\n  EXCLUDED (history/data): {excluded}")

    # ----- referee parity summary -----
    st_fail = [k for k in results if results[k]["selftest"] is False]
    st_new  = [k for k in results if results[k]["selftest"] is not None]
    print(f"\n  REFEREE PARITY: {len(st_new)} new assets self-tested; "
          f"{'ALL 0.00% PASS' if not st_fail else 'FAIL: '+','.join(st_fail)}")

    # ----- Part 2: family verdicts -----
    print("\n" + "="*100)
    print("PART 2 — FAMILY BREAKDOWN (pooled per family, fixed 1% risk additive)")
    print("="*100)
    FIXED = 0.01 * INITIAL_CASH
    fams = {}
    for key, r in results.items():
        fams.setdefault(r["family"], []).append(key)
    print(f"{'family':<14}{'assets':>7}{'n':>5}{'PF':>7}{'PnL(1%R)':>11}{'meanR':>8}"
          f"{'ab200%':>8}{'bearN':>7}{'bearPnL':>10}")
    fam_summ = {}
    for fam in ["crypto","equity-index","single-stock","metal","energy","fx"]:
        keys = fams.get(fam, [])
        if not keys: continue
        Rs=[]; below=0; above=0; bn=0; bp=0.0; bdef=False
        for k in keys:
            for t in results[k]["trades"]:
                Rs.append(t["R"])
                if t["below_ema200"]: below+=1
                else: above+=1
            if results[k]["bear_defined"]:
                bdef=True; bn+=results[k]["bear_n"]; bp+=results[k]["bear_pnl"]
        Rs=np.array(Rs)
        pnl = Rs.sum()*FIXED
        w=Rs[Rs>0].sum(); l=-Rs[Rs<0].sum()
        pf=(w/l) if l>1e-9 else (float("inf") if w>0 else 0.0)
        pf_s="inf" if pf==float("inf") else f"{pf:.2f}"
        absh=100*above/(above+below) if (above+below) else 0
        fam_summ[fam]={"n":len(Rs),"PF":pf,"PnL":pnl,"meanR":float(Rs.mean()) if len(Rs) else 0,
                       "above_share":absh,"assets":len(keys)}
        bns=f"{bn}" if bdef else "n/d"; bps=f"{bp:,.0f}" if bdef else "n/d"
        print(f"{fam:<14}{len(keys):>7}{len(Rs):>5}{pf_s:>7}{pnl:>11,.0f}"
              f"{(Rs.mean() if len(Rs) else 0):>8.3f}{absh:>7.0f}%{bns:>7}{bps:>10}")

    # ----- Part 2: wide basket pooled -----
    print("\n" + "="*100)
    print("PART 2 — WIDE BASKET (all families pooled, fixed 1% risk)")
    print("="*100)
    all_entries=[]; basket=[]
    for key, r in results.items():
        for t in r["trades"]:
            all_entries.append((t["entry_time"], key, t["pnl"], t["R"]))
            basket.append({"exit_time": t["exit_time"], "R": t["R"],
                           "pnl_fixed": t["R"]*FIXED, "entry_time": t["entry_time"]})
    raw_n=len(all_entries)
    # calendar span across the pooled trade set
    span_days=(max(t["exit_time"] for t in basket)-min(e for e,_,_,_ in all_entries)).days
    span_yrs=span_days/365.25

    # episode clustering (+-5d): any two fires within 5 calendar days = same episode
    ev=sorted(all_entries, key=lambda x:x[0]); episodes=[]; cur=[]
    for e in ev:
        if not cur: cur=[e]; continue
        if (e[0]-cur[-1][0]).days<=5: cur.append(e)
        else: episodes.append(cur); cur=[e]
    if cur: episodes.append(cur)
    ep_pnls=np.array([sum(R*FIXED for (_,_,_,R) in ep) for ep in episodes])
    ep_w=ep_pnls[ep_pnls>0].sum(); ep_l=-ep_pnls[ep_pnls<0].sum()
    ep_pf=ep_w/ep_l if ep_l>1e-9 else float("inf")

    basket.sort(key=lambda x:x["exit_time"])
    beq=[INITIAL_CASH]; e=INITIAL_CASH
    for t in basket: e+=t["pnl_fixed"]; beq.append(e)
    bpnls=np.array([t["pnl_fixed"] for t in basket])
    bw=bpnls[bpnls>0].sum(); bl=-bpnls[bpnls<0].sum()
    bpf=bw/bl if bl>1e-9 else float("inf")
    beq=np.array(beq); peak=np.maximum.accumulate(beq); bdd=((beq-peak)/peak).min()*100
    allR=np.array([R for (_,_,_,R) in all_entries])
    ci=boot_meanR_ci(allR)

    print(f"  RAW pooled trades          : {raw_n}")
    print(f"  INDEPENDENT episodes(+-5d) : {len(episodes)}  (cross-asset clustered; honest n)")
    print(f"  calendar span              : {span_yrs:.1f}yr")
    print(f"  RAW trades / year          : {raw_n/span_yrs:.1f}   <-- THE cadence number (target 10-15)")
    print(f"  episodes / year            : {len(episodes)/span_yrs:.1f}")
    print(f"  basket PF (fixed 1%R)      : {bpf:.2f}   PnL ${bpnls.sum():,.0f}   MaxDD {bdd:.2f}%")
    print(f"  episode-additive PF        : {ep_pf:.2f}   PnL ${ep_pnls.sum():,.0f}")
    print(f"  pooled meanR               : {allR.mean():+.3f}  sd {allR.std():.3f}  "
          f"bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    kb2=kblock_stability([{"pnl":t["pnl_fixed"]}
                          for t in sorted(basket,key=lambda x:x['entry_time'])],6,2)
    if kb2:
        print(f"  K-block (K6,m2) stability  : folds={kb2['n_folds']} "
              f"frac>1={kb2['frac_PF_gt1']:.0%} frac>=1.5={kb2['frac_PF_ge1p5']:.0%}")

    # ----- pass-only forward basket (drop families that FAIL Part 1: fx, energy) -----
    PASS_FAMS = {"crypto","equity-index","single-stock","metal"}
    FAIL_FAMS = {"fx","energy"}
    pass_entries=[]; pass_basket=[]
    for key, r in results.items():
        if r["family"] in FAIL_FAMS: continue
        for t in r["trades"]:
            pass_entries.append((t["entry_time"], key, t["R"]))
            pass_basket.append({"exit_time": t["exit_time"], "pnl_fixed": t["R"]*FIXED,
                                "entry_time": t["entry_time"], "R": t["R"]})
    pass_basket.sort(key=lambda x:x["exit_time"])
    peq=[INITIAL_CASH]; e=INITIAL_CASH
    for t in pass_basket: e+=t["pnl_fixed"]; peq.append(e)
    ppnls=np.array([t["pnl_fixed"] for t in pass_basket])
    pw=ppnls[ppnls>0].sum(); pl=-ppnls[ppnls<0].sum(); ppf=pw/pl if pl>1e-9 else float("inf")
    peq=np.array(peq); ppeak=np.maximum.accumulate(peq); pdd=((peq-ppeak)/ppeak).min()*100
    passR=np.array([R for (_,_,R) in pass_entries]); pci=boot_meanR_ci(passR)
    p_span=(max(t["exit_time"] for t in pass_basket)-min(e for e,_,_ in pass_entries)).days/365.25

    # recent-era cadence: 2021-01-01 -> now, when ALL 29 assets are live (the forward reality)
    RECENT="2021-01-01"
    def recent_rate(entries):
        rec=[x for x in entries if x[0]>=pd.Timestamp(RECENT)]
        yrs=(pd.Timestamp("2026-08-13")-pd.Timestamp(RECENT)).days/365.25
        return len(rec), len(rec)/yrs
    full_rec_n, full_rec_rate = recent_rate([(t,k,R) for (t,k,_,R) in all_entries])
    pass_rec_n, pass_rec_rate = recent_rate(pass_entries)

    print("\n" + "="*100)
    print("PART 2b — PASS-ONLY FORWARD BASKET (drop fx+energy that FAIL Part 1) + RECENT-ERA cadence")
    print("="*100)
    print(f"  PASS-only basket (crypto+equity-index+single-stock+metal, 25 assets):")
    print(f"    n={len(pass_basket)}  PF {ppf:.2f}  PnL ${ppnls.sum():,.0f}  MaxDD {pdd:.2f}%  "
          f"meanR {passR.mean():+.3f}  CI [{pci[0]:+.3f},{pci[1]:+.3f}]")
    print(f"    span {p_span:.1f}yr  raw {len(pass_basket)/p_span:.1f}/yr (full-history, era-inflated)")
    print(f"  RECENT-ERA cadence (2021-01-01 -> 2026-08, ALL assets live = the forward reality):")
    print(f"    FULL 29-asset basket : {full_rec_n} trades / 5.6yr = {full_rec_rate:.1f}/yr")
    print(f"    PASS 25-asset basket : {pass_rec_n} trades / 5.6yr = {pass_rec_rate:.1f}/yr")

    # ----- deflation -----
    print("\n" + "-"*100)
    print("  add.60 DEFLATION: honest forward expectation ~ HALF the in-sample edge.")
    print(f"    in-sample basket meanR {allR.mean():+.3f} -> deflated ~{allR.mean()/2:+.3f}R/trade")
    print(f"    in-sample basket PF {bpf:.2f} -> deflated forward PF ~1.6-1.7 (K20-50 door haircut, add.60 M3)")
    print("-"*100)

    # dump for report
    dump={"results":{k:{kk:vv for kk,vv in v.items() if kk!='trades'} for k,v in results.items()},
          "excluded":excluded, "fam_summ":fam_summ,
          "basket":{"raw_n":raw_n,"episodes":len(episodes),"span_yrs":span_yrs,
                    "raw_per_yr":raw_n/span_yrs,"ep_per_yr":len(episodes)/span_yrs,
                    "basket_pf":bpf,"basket_pnl":float(bpnls.sum()),"maxdd":bdd,
                    "ep_pf":ep_pf,"meanR":float(allR.mean()),"meanR_sd":float(allR.std()),
                    "meanR_ci":ci},
          "pass_basket":{"n":len(pass_basket),"PF":ppf,"PnL":float(ppnls.sum()),
                    "MaxDD":pdd,"meanR":float(passR.mean()),"meanR_ci":pci,
                    "span_yrs":p_span,"raw_per_yr":len(pass_basket)/p_span,
                    "recent_full_n":full_rec_n,"recent_full_rate":full_rec_rate,
                    "recent_pass_n":pass_rec_n,"recent_pass_rate":pass_rec_rate}}
    outp=os.path.join(WB,"wide_basket_results.json")
    with open(outp,"w") as f: json.dump(dump,f,indent=2,default=str)
    print(f"\n  results dumped -> {outp}")


if __name__ == "__main__":
    main()
