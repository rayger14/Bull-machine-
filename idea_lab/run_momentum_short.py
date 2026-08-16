"""
MOMENTUM-SHORT DOOR — MEASUREMENT RUN (add.68; STUDY ONLY; nothing ships)
========================================================================
Runs the pre-registered mirror-short door (momentum_short_door.MomentumShortDoor) on
the SAME wide universe / families / costs as the validated add.48/62 LONG door, and
evaluates the pre-registered pass rule (idea_lab/momentum_short_PREREGISTRATION.txt §6):

  (A) EDGE : pooled SHORT PF>=1.5 at n>=50, bootstrap meanR CI-low>0, on the CLEAN
             families (crypto + equity-index + metal).
  (B) MONEY: pooled SHORT PF>1.2 in the pooled BEAR windows (crypto=CRYPTO_BEARS,
             equity-index=EQUITY_BEARS; metals have no pre-defined bear calendar in the
             harness -> excluded from the money pool, flagged).
  (C) HEDGE: combined LONG+SHORT book MaxDD <= LONG-only MaxDD (clean families).

Also reports (per pre-reg §4/§5): the T5 retest fire-rate per bear window (the markdown-
resilience asymmetry acid test), above/below-EMA200 split, K-block CPCV stability,
episode clustering, and the single-stock (borrow-APPROXIMATE) sleeve separately.

Headline = struct/flat rmult=1.0. Costs 2bps+3bps/side, risk 1%, $100k. NO per-asset tuning.
"""
from __future__ import annotations
import os, sys, warnings
from collections import Counter
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))

from backtester import run_backtest, INITIAL_CASH, RISK_PCT
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from momentum_short_door import MomentumShortDoor, t5_scan
from xasset_spx_port import load_spx
from run_xasset_spx import selftest_on
from run_wide_basket import (integrity, window_pnl, kblock_stability, boot_meanR_ci,
                             EQUITY_BEARS, CRYPTO_BEARS)

SCROOT = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
OS_ = os.path.join(SCROOT, "one_strategy")   # fresh crypto (Coinbase)
WB  = os.path.join(SCROOT, "wide_basket")    # yfinance families (+ GOLD/NDX refetched)
MIN_YEARS = 4.0
FIXED = 0.01 * INITIAL_CASH

UNIVERSE = {}
for s in ["BTC-USD","ETH-USD","SOL-USD","LTC-USD","XRP-USD","ADA-USD",
          "DOGE-USD","DOT-USD","AVAX-USD","LINK-USD"]:
    UNIVERSE[s] = ("crypto", os.path.join(OS_, f"{s}.parquet"))
FAM = {"GOLD":"metal","NDX":"equity-index","SPX":"equity-index","DJI":"equity-index",
       "RUT":"equity-index","N225":"equity-index","DAX":"equity-index","FTSE":"equity-index",
       "SILVER":"metal","COPPER":"metal","PLATINUM":"metal","OIL":"energy",
       "EURUSD":"fx","GBPUSD":"fx","USDJPY":"fx","AUDUSD":"fx",
       "AAPL":"single-stock","MSFT":"single-stock","NVDA":"single-stock"}
for k, fam in FAM.items():
    UNIVERSE[k] = (fam, os.path.join(WB, f"{k}.parquet"))

CLEAN_FAMS = {"crypto", "equity-index", "metal"}
BEAR_FAMS = {"crypto": CRYPTO_BEARS, "equity-index": EQUITY_BEARS}  # metals: no calendar


def bears_for(fam):
    if fam == "crypto": return CRYPTO_BEARS
    if fam in ("equity-index", "single-stock"): return EQUITY_BEARS
    return None


def run_asset(key, fam, path, do_selftest):
    df_raw = load_spx(path)
    integ = integrity(pd.read_parquet(path))
    df, sr, bj, eye = build_daily_sensors(df_raw)
    st = selftest_on(df, key) if do_selftest else None

    short = MomentumShortDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res_s = run_backtest(df, short, risk_pct=RISK_PCT, label=f"{key}_short")
    elog_s = short.entries_log
    armed = t5_scan(short)   # causal, independent pass (dedup-free)

    long = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res_l = run_backtest(df, long, risk_pct=RISK_PCT, label=f"{key}_long")

    s_tr, s_st = res_s["trades"], res_s["stats"]
    below = sum(1 for e in elog_s if e["below_ema200"]); tot = len(elog_s)
    bl_share = 100*below/tot if tot else 0.0
    bears = bears_for(fam); bear_n = bear_pnl = 0
    if bears:
        for _, a, b in bears:
            n_, p_ = window_pnl(s_tr, elog_s, a, b); bear_n += n_; bear_pnl += p_
    return {
        "key": key, "family": fam, "integrity": integ, "selftest": st,
        "n": s_st["n"], "WR": s_st["WR"], "PF": s_st["PF"], "avgR": s_st["avgR"],
        "PnL": s_st["PnL"], "MaxDD": s_st["MaxDD_pct"], "below_share": bl_share,
        "bear_defined": bears is not None, "bear_n": bear_n, "bear_pnl": bear_pnl,
        "exits": dict(Counter(t["exit_reason"] for t in s_tr)),
        "short_trades": [{"entry_time": e["entry_time"], "exit_time": t["exit_time"],
                          "R": t["R"], "pnl": t["pnl"],
                          "below_ema200": e["below_ema200"]}
                         for t, e in zip(s_tr, elog_s)],
        "long_trades": [{"entry_time": lt["entry_time"], "exit_time": lt["exit_time"],
                         "R": lt["R"]} for lt in res_l["trades"]],
        "armed": armed, "span_yrs": integ["span_yrs"],
    }


def pooled_pf_meanR(trades):
    Rs = np.array([t["R"] for t in trades], float)
    if len(Rs) == 0: return 0, 0.0, 0.0, (float("nan"), float("nan"))
    w = Rs[Rs > 0].sum(); l = -Rs[Rs < 0].sum()
    pf = (w/l) if l > 1e-9 else (float("inf") if w > 0 else 0.0)
    return len(Rs), pf, float(Rs.mean()), boot_meanR_ci(Rs)


def book_maxdd(trades):
    """Fixed-1%R additive equity curve sorted by exit_time -> MaxDD %."""
    if not trades: return 0.0, 0.0
    tr = sorted(trades, key=lambda x: x["exit_time"])
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in tr: e += t["R"]*FIXED; eq.append(e)
    eq = np.array(eq); peak = np.maximum.accumulate(eq)
    return float(((eq-peak)/peak).min()*100), float(eq[-1]-INITIAL_CASH)


def main():
    print("="*104)
    print("MOMENTUM-SHORT DOOR (add.68) — pre-registered mirror of the add.48 long door, INVERTED")
    print("Costs 2bps+3bps/side, risk 1%, $100k. Headline struct/flat rmult=1.0. NO per-asset tuning.")
    print("="*104)

    results = {}; excluded = []
    for key, (fam, path) in UNIVERSE.items():
        if not os.path.exists(path):
            excluded.append((key, "no data")); print(f"MISSING {key} ({path})"); continue
        r = run_asset(key, fam, path, do_selftest=True)
        if r["span_yrs"] < MIN_YEARS:
            excluded.append((key, f"{r['span_yrs']:.1f}yr")); print(f"EXCLUDE {key}"); continue
        results[key] = r

    # -------- Part 1: per-asset short table --------
    print("\n" + "="*104)
    print("PART 1 — PER-ASSET SHORT DOOR (headline struct/flat). integrity + parity + door.")
    print("="*104)
    print(f"{'asset':<9}{'fam':<13}{'yrs':>5}{'n':>4}{'WR':>6}{'PF':>7}{'PnL':>10}"
          f"{'MaxDD':>7}{'below200':>9}{'bearN':>6}{'bearPnL':>10}{'ST':>7}")
    st_fail = []
    for key in UNIVERSE:
        if key not in results: continue
        r = results[key]; pf = r["PF"]; pf_s = "inf" if pf==float("inf") else f"{pf:.2f}"
        st = "-" if r["selftest"] is None else ("0.00%" if r["selftest"] else "FAIL")
        if r["selftest"] is False: st_fail.append(key)
        bn = f"{r['bear_n']}" if r["bear_defined"] else "n/d"
        bp = f"{r['bear_pnl']:,.0f}" if r["bear_defined"] else "n/d"
        print(f"{key:<9}{r['family']:<13}{r['span_yrs']:>5.1f}{r['n']:>4}{r['WR']*100:>5.0f}%"
              f"{pf_s:>7}{r['PnL']:>10,.0f}{r['MaxDD']:>6.1f}%{r['below_share']:>8.0f}%"
              f"{bn:>6}{bp:>10}{st:>7}")
    if excluded: print(f"\n  EXCLUDED: {excluded}")
    print(f"  REFEREE PARITY: {'ALL 0.00% PASS' if not st_fail else 'FAIL: '+','.join(st_fail)}")

    # -------- Part 2: family pooled --------
    print("\n" + "="*104)
    print("PART 2 — FAMILY POOLED (fixed 1% risk additive)")
    print("="*104)
    fams = {}
    for key, r in results.items(): fams.setdefault(r["family"], []).append(key)
    print(f"{'family':<14}{'assets':>7}{'n':>5}{'PF':>7}{'PnL':>11}{'meanR':>8}{'below200%':>10}")
    for fam in ["crypto","equity-index","metal","single-stock","energy","fx"]:
        keys = fams.get(fam, [])
        if not keys: continue
        tr = [t for k in keys for t in results[k]["short_trades"]]
        n, pf, meanR, _ = pooled_pf_meanR(tr)
        pnl = sum(t["R"]*FIXED for t in tr)
        below = sum(1 for t in tr if t["below_ema200"]); bshare = 100*below/n if n else 0
        pf_s = "inf" if pf==float("inf") else f"{pf:.2f}"
        tag = "  [CLEAN]" if fam in CLEAN_FAMS else ("  [refute]" if fam in ("fx","energy") else "  [borrow~approx]")
        print(f"{fam:<14}{len(keys):>7}{n:>5}{pf_s:>7}{pnl:>11,.0f}{meanR:>8.3f}{bshare:>9.0f}%{tag}")

    # -------- Part 3: PASS RULE --------
    print("\n" + "="*104)
    print("PART 3 — PRE-REGISTERED PASS RULE (A edge / B money / C hedge)")
    print("="*104)
    clean_short = [t for k, r in results.items() if r["family"] in CLEAN_FAMS
                   for t in r["short_trades"]]
    nA, pfA, meanRA, ciA = pooled_pf_meanR(clean_short)
    A = (pfA >= 1.5) and (nA >= 50) and (ciA[0] > 0)
    print(f"(A) EDGE   clean pooled short: n={nA}  PF={pfA:.2f}  meanR={meanRA:+.3f}  "
          f"CI[{ciA[0]:+.3f},{ciA[1]:+.3f}]")
    print(f"           need PF>=1.5 AND n>=50 AND CI-lo>0  ->  {'PASS' if A else 'FAIL'}")

    bear_pool = []
    for k, r in results.items():
        if r["family"] not in CLEAN_FAMS: continue
        bears = BEAR_FAMS.get(r["family"])
        if not bears: continue   # metals: no bear calendar -> excluded from money pool
        ents = pd.DatetimeIndex([t["entry_time"] for t in r["short_trades"]])
        for _, a, b in bears:
            m = (ents >= pd.Timestamp(a)) & (ents <= pd.Timestamp(b))
            bear_pool += [r["short_trades"][i] for i in np.where(m)[0]]
    nB, pfB, meanRB, ciB = pooled_pf_meanR(bear_pool)
    B = (nB > 0) and (pfB > 1.2)
    bpnl = sum(t["R"]*FIXED for t in bear_pool)
    print(f"(B) MONEY  bear-window pooled short (crypto+equity-idx bears): n={nB}  "
          f"PF={pfB:.2f}  PnL=${bpnl:,.0f}  meanR={meanRB:+.3f}")
    print(f"           need PF>1.2 in pooled bear windows  ->  {'PASS' if B else 'FAIL'}")

    long_clean = [t for k, r in results.items() if r["family"] in CLEAN_FAMS
                  for t in r["long_trades"] if t["exit_time"] is not None]
    dd_long, pnl_long = book_maxdd(long_clean)
    combined = long_clean + [{"exit_time": t["exit_time"], "R": t["R"]} for t in clean_short]
    dd_both, pnl_both = book_maxdd(combined)
    C = dd_both >= dd_long   # less-negative (shallower) MaxDD; DD values are negative %
    print(f"(C) HEDGE  clean book MaxDD: long-only={dd_long:.2f}%  long+short={dd_both:.2f}%  "
          f"(long PnL ${pnl_long:,.0f} -> +short ${pnl_both:,.0f})")
    print(f"           need long+short MaxDD <= long-only (shallower)  ->  {'PASS' if C else 'FAIL'}")

    verdict = "VALIDATE" if (A and B and C) else "REJECT / watch-item"
    print(f"\n  >>> PASS RULE (A AND B AND C): {verdict}   [A={A} B={B} C={C}] <<<")

    # -------- Part 4: T5 fire-rate per bear window (the asymmetry acid test) --------
    print("\n" + "="*104)
    print("PART 4 — T5 RETEST FIRE-RATE per bear window (markdown-resilience asymmetry)")
    print("  armed = regime_ok AND T1-T4 (real prior down-break, held below).  fired = T5 (up-retest).")
    print("="*104)
    print(f"{'window':<16}{'armed_bars':>11}{'fired(T5)':>11}{'fire_rate':>11}")
    def t5_window(a, b, families):
        A_=F_=0
        for k, r in results.items():
            if r["family"] not in families: continue
            for row in r["armed"]:
                if pd.Timestamp(a) <= pd.Timestamp(row["time"]) <= pd.Timestamp(b):
                    A_ += 1; F_ += 1 if row["fired"] else 0
        return A_, F_
    all_a = all_f = 0
    for name, a, b in CRYPTO_BEARS:
        A_, F_ = t5_window(a, b, {"crypto"}); all_a += A_; all_f += F_
        fr = f"{100*F_/A_:.1f}%" if A_ else "n/a"
        print(f"{'crypto '+name:<16}{A_:>11}{F_:>11}{fr:>11}")
    for name, a, b in EQUITY_BEARS:
        A_, F_ = t5_window(a, b, {"equity-index"}); all_a += A_; all_f += F_
        fr = f"{100*F_/A_:.1f}%" if A_ else "n/a"
        print(f"{'eqidx '+name:<16}{A_:>11}{F_:>11}{fr:>11}")
    fr = f"{100*all_f/all_a:.1f}%" if all_a else "n/a"
    print(f"{'POOLED bears':<16}{all_a:>11}{all_f:>11}{fr:>11}")

    # -------- Part 5: robustness on the clean short pool --------
    print("\n" + "="*104)
    print("PART 5 — ROBUSTNESS (clean short pool): K-block CPCV + episode clustering")
    print("="*104)
    kb = kblock_stability([{"pnl": t["R"]*FIXED}
                           for t in sorted(clean_short, key=lambda x: x["entry_time"])], 6, 2)
    if kb:
        print(f"  K-block (K6,m2): folds={kb['n_folds']}  frac PF>1={kb['frac_PF_gt1']:.0%}  "
              f"frac PF>=1.5={kb['frac_PF_ge1p5']:.0%}")
    ev = sorted(clean_short, key=lambda x: x["entry_time"])
    episodes = []; cur = []
    for e in ev:
        if not cur: cur = [e]; continue
        if (pd.Timestamp(e["entry_time"])-pd.Timestamp(cur[-1]["entry_time"])).days <= 5:
            cur.append(e)
        else: episodes.append(cur); cur = [e]
    if cur: episodes.append(cur)
    print(f"  raw clean trades={len(clean_short)}  independent episodes(+-5d)={len(episodes)}")
    print(f"  overall below-EMA200 share (clean short): "
          f"{100*sum(1 for t in clean_short if t['below_ema200'])/max(1,len(clean_short)):.0f}%  "
          f"(mirror acid test: shorts SHOULD be below-EMA200)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
