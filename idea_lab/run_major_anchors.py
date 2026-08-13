"""
MAJOR-ANCHOR GANN + GENTLER-LPS-REENTRY STUDY  (add.59 -- FINAL ROUND for both hypotheses)
==========================================================================================
Round-2 corrected re-test of the TWO implementation defects the add.58 study flagged in
its OWN Gann/campaign build. ONE pre-registered corrected definition each, NO grids, FINAL
round: fail again -> both hypotheses CLOSE permanently.

PART A -- MAJOR-anchor Gann (fixes add.58 D1 "anchors saturated"):
  anchors = weekly N=5 confirmed pivots that are ALSO the trailing-365-calendar-day extreme
  at formation (causal) + crypto halvings. Counts/tol UNCHANGED from add.58.
  SATURATION GUARD (pre-registered): the test is meaningful only if entry-window coverage
  < 25%. If it STILL saturates -> report and STOP Part A (that closes the hypothesis:
  "major" cannot be mechanized without discretion).
  If it clears: G-ENTRY split (in vs out mean R, bootstrap CI) + G-EXIT (major danger tier).

PART B -- gentler LPS re-entry (fixes add.58 D2 "second-entry too strict"):
  CampaignV2bDoor: entries 2-3 fire on a NEW higher confirmed N=10 LPS that is retested and
  HOLDS (no new breakout). v1 vs CAMPAIGN-v2b on the same market set.

PASS RULES (pre-registered, IDENTICAL to add.58):
  G-ENTRY passes iff in-window mean R >= out-window with CI lower bound > 0.
  G-EXIT  passes iff improves basket MaxDD or PnL WITHOUT runner-tail collapse > 30%.
  CAMPAIGN-v2b passes iff PF >= v1PF-0.3 AND PnL >= v1 AND cadence >= 1.5x AND MaxDD <= 1.5x v1.

Costs 2bps+3bps, 1% risk, $100k. Referee parity reported. STUDY ONLY -- nothing ships.
"""
from __future__ import annotations
import os, sys
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from backtester import run_backtest, INITIAL_CASH, RISK_PCT
from campaign_backtester import run_campaign_backtest
from campaign_backtester import run_selftest as campaign_referee_selftest
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from campaign_strategy import CampaignV2bDoor
from gann_time import compute_gann_windows, run_selftest as gann_selftest, _anchors
from xasset_spx_port import load_spx
from run_xasset_spx import selftest_on
from run_campaign_v2 import v1_gann_exit_resim, basket_stats, boot_ci, pf_of, FIXED_RISK

SC = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
      "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD", "XRP-USD", "ADA-USD",
          "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD"]
XASSET = [("GOLD", f"{SC}/xasset/GOLD_1D.parquet"),
          ("NDX",  f"{SC}/xasset/NDX_1D.parquet")]
RNG = np.random.default_rng(20260813)
SAT_THRESH = 25.0     # pre-registered saturation guard (entry-window coverage %)


def load_all():
    out = {}
    for sym in CRYPTO:
        p = os.path.join(SC, "one_strategy", f"{sym}.parquet")
        if not os.path.exists(p):
            print(f"  MISSING {sym}"); continue
        df, sr, bj, eye = build_daily_sensors(load_spx(p))
        out[sym] = dict(df=df, sr=sr, bj=bj, eye=eye, hv=True)
    for name, p in XASSET:
        if not os.path.exists(p):
            print(f"  MISSING {name}"); continue
        df, sr, bj, eye = build_daily_sensors(load_spx(p))
        out[name] = dict(df=df, sr=sr, bj=bj, eye=eye, hv=False)
    return out


def part_A(assets):
    print("\n" + "=" * 100)
    print("PART A -- MAJOR-anchor Gann re-test (fixes add.58 D1). Anchors = trailing-365-day "
          "extreme N=5 pivots + halvings.")
    print("=" * 100)
    print("(1) ANCHOR CENSUS + COVERAGE (saturation guard: meaningful only if entry_w < 25%)\n")
    hdr = (f"{'asset':<9}{'yrs':>5}{'bars':>6} | {'v58_anc':>8}{'v58_ent%':>9} | "
           f"{'MAJ_anc':>8}{'MAJ_hi':>7}{'MAJ_lo':>7}{'MAJ_hv':>7}{'ent%':>7}{'dgr%':>7}  guard")
    print(hdr)
    covs = {}
    danger = {}
    for name, a in assets.items():
        df, hv = a["df"], a["hv"]
        yrs = (df.index.max() - df.index.min()).days / 365.25
        g0 = compute_gann_windows(df, use_halvings=hv, major_only=False)
        g1 = compute_gann_windows(df, use_halvings=hv, major_only=True)
        anc = _anchors(df, hv, major_only=True)
        nhi = sum(1 for x in anc if x["source"] == "pivot" and x["kind"] == "high")
        nlo = sum(1 for x in anc if x["source"] == "pivot" and x["kind"] == "low")
        nhv = sum(1 for x in anc if x["source"] == "halving")
        e1 = 100 * g1["entry_window"].mean(); d1 = 100 * g1["danger_window"].mean()
        covs[name] = e1
        danger[name] = g1["danger_window"]
        assets[name]["entry_w_major"] = g1["entry_window"]
        guard = "PASS<25" if e1 < SAT_THRESH else "SATURATES"
        print(f"{name:<9}{yrs:>5.1f}{len(df):>6} | {g0['n_anchors']:>8}{100*g0['entry_window'].mean():>9.1f} | "
              f"{g1['n_anchors']:>8}{nhi:>7}{nlo:>7}{nhv:>7}{e1:>7.1f}{d1:>7.1f}  {guard}")
    cv = np.array(list(covs.values()))
    n_pass = int((cv < SAT_THRESH).sum())
    print(f"\n  MAJOR entry_w coverage: mean {cv.mean():.1f}%  median {np.median(cv):.1f}%  "
          f"range [{cv.min():.1f}, {cv.max():.1f}]  assets<25%: {n_pass}/{len(cv)}")
    flagship = [c for k, c in covs.items() if k in ("BTC-USD", "GOLD", "NDX")]
    flagship_ok = all(c < SAT_THRESH for c in flagship)
    guard_cleared = (cv.mean() < SAT_THRESH) and flagship_ok and (n_pass > len(cv) / 2)
    print(f"  flagships BTC/GOLD/NDX coverage {['%.1f'%c for c in flagship]}  "
          f"all<25%: {flagship_ok}")
    print(f"\n  >> SATURATION GUARD {'CLEARED' if guard_cleared else 'NOT CLEARED'} "
          f"(need basket-mean<25% AND flagships<25% AND majority<25%).")

    # DIAGNOSTIC (reported regardless, but NON-DISPOSITIVE if guard not cleared): is there any
    # latent G-ENTRY / G-EXIT edge being discarded? Uses the MAJOR anchors.
    print("\n(2) DIAGNOSTIC G-ENTRY / G-EXIT on MAJOR anchors "
          + ("(PRIMARY -- guard cleared)" if guard_cleared else
             "(NON-DISPOSITIVE -- guard NOT cleared; shown only to prove no hidden edge)") + ":")
    v1_res = {}
    for name, a in assets.items():
        v1 = run_backtest(a["df"], TrendContinuationDoor(a["df"], a["sr"], a["bj"], a["eye"],
                                                         variant="struct", conviction=False), label="v1")
        pos = {t: k for k, t in enumerate(a["df"].index)}
        ew = a["entry_w_major"]
        for t in v1["trades"]:
            t["in_gann_entry"] = bool(ew[pos[t["entry_time"]]])
        v1_res[name] = v1
        a["v1"] = v1
    inR = [t["R"] for r in v1_res.values() for t in r["trades"] if t["in_gann_entry"]]
    outR = [t["R"] for r in v1_res.values() for t in r["trades"] if not t["in_gann_entry"]]
    mi = np.mean(inR) if inR else float("nan"); mo = np.mean(outR) if outR else float("nan")
    if inR and outR:
        bi = RNG.integers(0, len(inR), size=(10000, len(inR)))
        bo = RNG.integers(0, len(outR), size=(10000, len(outR)))
        dstat = np.asarray(inR)[bi].mean(1) - np.asarray(outR)[bo].mean(1)
        dci = (float(np.percentile(dstat, 2.5)), float(np.percentile(dstat, 97.5)))
    else:
        dci = (float("nan"), float("nan"))
    print(f"    G-ENTRY (v1 door): in-window n={len(inR)} meanR={mi:+.3f}  |  "
          f"out-window n={len(outR)} meanR={mo:+.3f}  |  Δ={mi-mo:+.3f} "
          f"CI[{dci[0]:+.3f},{dci[1]:+.3f}]")
    gentry_pass = (mi >= mo) and (dci[0] > 0)
    print(f"      G-ENTRY {'PASS' if gentry_pass else 'FAIL'} (needs in>=out, CI lower>0)")

    # G-EXIT on v1 door with MAJOR danger windows (paired per-trade re-sim, add.56 method)
    allpairs = []
    for name, a in assets.items():
        allpairs += v1_gann_exit_resim(a["df"], a["sr"], a["bj"], a["eye"], danger[name])
    dR = [g - b for (b, g) in allpairs]
    Rb = [b for (b, g) in allpairs]; Rg = [g for (b, g) in allpairs]
    changed = [(b, g) for (b, g) in allpairs if abs(g - b) > 1e-9]
    ciΔ = boot_ci(dR)
    tailb = sum(x for x in Rb if x >= 3); tailg = sum(x for x in Rg if x >= 3)
    print(f"    G-EXIT (v1 door, MAJOR danger): n={len(allpairs)}  altered={len(changed)}  "
          f"meanΔR={np.mean(dR):+.3f} CI[{ciΔ[0]:+.3f},{ciΔ[1]:+.3f}]")
    print(f"      sumR {sum(Rb):+.1f}->{sum(Rg):+.1f}  maxR {max(Rb):.2f}->{max(Rg):.2f}  "
          f"runner-tail(>=3) {tailb:.1f}->{tailg:.1f}")
    tail_collapse = (tailb > 0 and (tailb - tailg) / abs(tailb) > 0.30)
    gexit_pass = (sum(Rg) >= sum(Rb)) and not tail_collapse
    print(f"      G-EXIT {'PASS' if gexit_pass else 'FAIL/INERT'} "
          f"(altered {len(changed)} trades)")
    return guard_cleared, gentry_pass, gexit_pass


def part_B(assets):
    print("\n" + "=" * 100)
    print("PART B -- gentler LPS re-entry (fixes add.58 D2). v1 vs CAMPAIGN-v2b (LPS-hold re-entries).")
    print("=" * 100)
    hdr = (f"{'asset':<9}{'yrs':>5} | {'v1_n':>5}{'v1_PF':>7}{'v1_PnL':>10}{'v1_DD':>7} | "
           f"{'v2b_n':>6}{'camp':>5}{'reent':>6}{'e/cmp':>6}{'v2b_PF':>7}{'v2b_PnL':>10}{'v2b_DD':>7}"
           f"{'v1t/y':>7}{'v2t/y':>7}")
    print(hdr)
    v1_R = []; v2_R = []; v1_raw = 0; v2_raw = 0
    per_camp_delta = []
    all_entry_times = []; all_exit_times = []
    total_reent = 0
    for name, a in assets.items():
        df = a["df"]
        yrs = (df.index.max() - df.index.min()).days / 365.25
        v1 = a["v1"]                                  # already computed in Part A
        strat = CampaignV2bDoor(df, a["sr"], a["bj"], a["eye"])
        v2 = run_campaign_backtest(df, strat, label="v2b")
        s1, s2 = v1["stats"], v2["stats"]
        camps = defaultdict(list)
        for t in v2["trades"]:
            camps[t["meta"]["campaign_id"]].append(t)
        n_camp = len(camps); n_ent = len(v2["trades"])
        reent = strat.reentry_fires
        total_reent += reent
        epc = n_ent / n_camp if n_camp else 0.0
        pf1 = "inf" if s1["PF"] == float("inf") else f"{s1['PF']:.2f}"
        pf2 = "inf" if s2["PF"] == float("inf") else f"{s2['PF']:.2f}"
        print(f"{name:<9}{yrs:>5.1f} | {s1['n']:>5}{pf1:>7}{s1['PnL']:>10,.0f}{s1['MaxDD_pct']:>7.1f} | "
              f"{s2['n']:>6}{n_camp:>5}{reent:>6}{epc:>6.2f}{pf2:>7}{s2['PnL']:>10,.0f}"
              f"{s2['MaxDD_pct']:>7.1f}{s1['n']/yrs:>7.2f}{n_ent/yrs:>7.2f}")
        # basket accumulation
        for t in v1["trades"]:
            v1_R.append((t["exit_time"], t["R"])); v1_raw += 1
            all_entry_times.append(t["entry_time"]); all_exit_times.append(t["exit_time"])
        for t in v2["trades"]:
            v2_R.append((t["exit_time"], t["R"])); v2_raw += 1
            all_entry_times.append(t["entry_time"]); all_exit_times.append(t["exit_time"])
        # paired campaign comparison (same campaigns, two managements)
        v1_by_time = sorted(v1["trades"], key=lambda t: t["entry_time"])
        for cid, ts in camps.items():
            c_start = min(t["entry_time"] for t in ts); c_end = max(t["exit_time"] for t in ts)
            v2R = sum(t["R"] for t in ts)
            v1R = sum(t["R"] for t in v1_by_time if c_start <= t["entry_time"] <= c_end)
            per_camp_delta.append((v2R - v1R) * FIXED_RISK)

    b1 = basket_stats(v1_R); b2 = basket_stats(v2_R)
    span = (max(all_exit_times) - min(all_entry_times)).days / 365.25
    cadence_mult = (v2_raw / v1_raw) if v1_raw else float("nan")
    print("\n" + "-" * 100)
    print(f"BASKET (fixed 1% risk, additive; span {span:.1f}yr):")
    print(f"  v1 : PF {b1['PF']:.2f}  PnL ${b1['PnL']:,.0f}  MaxDD {b1['MaxDD']:.2f}%  "
          f"n={b1['n']}  cadence {v1_raw/span:.2f} tr/yr")
    print(f"  v2b: PF {b2['PF']:.2f}  PnL ${b2['PnL']:,.0f}  MaxDD {b2['MaxDD']:.2f}%  "
          f"n={b2['n']}  cadence {v2_raw/span:.2f} tr/yr")
    print(f"  gentler RE-ENTRIES that actually fired across all campaigns: {total_reent}")
    print(f"  CADENCE MULTIPLIER v2b/v1 = {cadence_mult:.2f}x")

    n_c = len(per_camp_delta)
    mean_d = float(np.mean(per_camp_delta)) if per_camp_delta else 0.0
    ci = boot_ci(per_camp_delta)
    wins = sum(1 for d in per_camp_delta if d > 0)
    print(f"\n  PAIRED campaign comparison: {n_c} campaigns  v2b>v1: {wins}/{n_c}")
    print(f"  mean ΔPnL/campaign (v2b-v1): ${mean_d:,.0f}  bootstrap 95% CI [${ci[0]:,.0f}, ${ci[1]:,.0f}]  "
          f"total ${sum(per_camp_delta):,.0f}")

    v1PF = b1["PF"] if np.isfinite(b1["PF"]) else 99
    v2PF = b2["PF"] if np.isfinite(b2["PF"]) else 99
    c1 = v2PF >= v1PF - 0.3
    c2 = b2["PnL"] >= b1["PnL"]
    c3 = cadence_mult >= 1.5
    c4 = b2["MaxDD"] >= 1.5 * b1["MaxDD"]
    print("\n  PART B PRE-REGISTERED VERDICT:")
    print(f"    C1 PF >= v1PF-0.3 ({v1PF-0.3:.2f}): v2bPF {v2PF:.2f} -> {'PASS' if c1 else 'FAIL'}")
    print(f"    C2 PnL >= v1 (${b1['PnL']:,.0f}): v2b ${b2['PnL']:,.0f} -> {'PASS' if c2 else 'FAIL'}")
    print(f"    C3 cadence >= 1.5x: {cadence_mult:.2f}x -> {'PASS' if c3 else 'FAIL'}")
    print(f"    C4 MaxDD <= 1.5x v1 ({1.5*b1['MaxDD']:.2f}%): v2b {b2['MaxDD']:.2f}% -> {'PASS' if c4 else 'FAIL'}")
    partB_pass = c1 and c2 and c3 and c4
    print(f"    => CAMPAIGN-v2b {'PASSES' if partB_pass else 'FAILS'} the pre-registered rule.")
    return partB_pass, total_reent


def main():
    print("=" * 100)
    print("add.59 MAJOR-ANCHOR + GENTLER-REENTRY STUDY (FINAL round). Costs 2bps+3bps, risk 1%, "
          "$100k. STUDY ONLY.")
    print("=" * 100)
    # referee / causality parity
    ref_df = load_spx(f"{SC}/one_strategy/BTC-USD.parquet")
    print("\n--- REFEREE PARITY (campaign engine vs textbook) + GANN causality/no-repaint ---")
    campaign_referee_selftest(ref_df)
    gann_selftest(ref_df, use_halvings=True, major_only=True)

    assets = load_all()
    print(f"\n  markets loaded: {len(assets)} ({', '.join(assets.keys())})")
    print("  NOTE: SPX unavailable (Yahoo v8/Stooq gated, per add.54/58). Non-crypto "
          "independent evidence = GOLD (uncorrelated) + NDX.")

    guard_cleared, gentry_pass, gexit_pass = part_A(assets)
    partB_pass, total_reent = part_B(assets)

    print("\n" + "=" * 100)
    print("FINAL DISPOSITION (add.59)")
    print("=" * 100)
    if not guard_cleared:
        print("  PART A (Gann MAJOR anchor): saturation guard NOT cleared -> STOP per pre-registration.")
        print("    HYPOTHESIS CLOSED. 'Major' cannot be mechanized causally without discretion.")
    else:
        print(f"  PART A (Gann MAJOR anchor): guard cleared. G-ENTRY {'PASS' if gentry_pass else 'FAIL'}, "
              f"G-EXIT {'PASS' if gexit_pass else 'FAIL/INERT'}.")
        if not (gentry_pass or gexit_pass):
            print("    HYPOTHESIS CLOSED (round 2 = last): no edge even with major anchors.")
    print(f"  PART B (gentler LPS re-entry): re-entries fired = {total_reent}; "
          f"CAMPAIGN-v2b {'PASSES' if partB_pass else 'FAILS'}.")
    if not partB_pass:
        print("    HYPOTHESIS CLOSED (round 2 = last): gentler re-entry does not beat v1.")
    print("=" * 100)


if __name__ == "__main__":
    main()
