"""
M3 (baseline optimism / deflated expectation) + M4 (structural biases)  (add.60)
================================================================================
M3: quantify survivorship optimism of the SELECTED door -> deflated forward
    expectation (expected-max / White's-reality-check-style haircut) + pre-register
    the forward acceptance test.
M4: (a) single-position shadow queue (door fires skipped while positioned);
    (b) era composition (how much PF is 2023-2025); (c) paired-frame limitation.
STUDY ONLY.
"""
from __future__ import annotations
import os, sys, warnings
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audit_common import load_basket, basket_R, basket_stats_from_R, FIXED_RISK
from trend_continuation_door import TrendContinuationDoor
from backtester import run_backtest

RNG = np.random.default_rng(20260813)


# ---------------------------------------------------------------------------
# M4(a) SHADOW QUEUE: count door fires SKIPPED while positioned
# ---------------------------------------------------------------------------
def shadow_queue(assets):
    print("=" * 92)
    print("M4(a) SHADOW QUEUE -- door fires skipped by the one-position engine while positioned")
    print("=" * 92)
    print(f"  {'asset':<9}{'taken':>7}{'raw_signals':>12}{'skipped':>9}{'skip%':>7}")
    tot_taken = tot_raw = 0
    for name, a in assets.items():
        df, sr, bj, eye = a["df"], a["sr"], a["bj"], a["eye"]
        n = len(df)
        # RAW signals: evaluate the door's _door(i) on EVERY bar ignoring position/dedup state,
        # but still respecting the door's own DEDUP_K (a signal within K of the prior signal is
        # the same setup, not a new fire). This isolates the ONE-POSITION absorption specifically.
        strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
        u = strat._u
        raw = 0; last = -10**9
        from unified_archetype_v2 import DEDUP_K
        for i in range(n):
            if i - last < DEDUP_K:
                continue
            if not u._regime_ok(i):
                continue
            sig = u._m2(i)
            if sig is None:
                continue
            plan = u._plan(i, sig)
            if plan is None:
                continue
            raw += 1; last = i
        taken = a["v1"]["stats"]["n"]
        skipped = raw - taken
        tot_taken += taken; tot_raw += raw
        print(f"  {name:<9}{taken:>7}{raw:>12}{skipped:>9}{100*skipped/max(raw,1):>7.1f}")
    print(f"\n  BASKET: taken={tot_taken}  raw_dedup_signals={tot_raw}  "
          f"skipped_while_positioned={tot_raw-tot_taken}  ({100*(tot_raw-tot_taken)/max(tot_raw,1):.1f}%)")
    print("  NOTE: raw counts re-fire the SAME door with its own DEDUP_K=3; the skip is purely the")
    print("  one-position occupancy (engine advances i=exit+1). A multi-position engine could take")
    print("  these -- but they are OVERLAPPING fires of the same rare setup, not new independent edges.")
    return tot_taken, tot_raw


# ---------------------------------------------------------------------------
# M4(b) ERA COMPOSITION: PF/PnL by calendar year and by era
# ---------------------------------------------------------------------------
def era_composition(assets):
    print("\n" + "=" * 92)
    print("M4(b) ERA COMPOSITION -- basket PnL(1% risk) by exit-year and era")
    print("=" * 92)
    R = basket_R(assets)   # (exit_time, entry_time, R, name)
    by_year = defaultdict(list)
    for (xt, et, r, name) in R:
        by_year[xt.year].append(r)
    print(f"  {'year':>6}{'n':>5}{'sumR':>9}{'PnL$':>11}{'PF':>7}")
    tot = 0.0
    for y in sorted(by_year):
        rs = np.array(by_year[y]); pos = rs[rs > 0].sum(); neg = -rs[rs < 0].sum()
        pf = pos / neg if neg > 1e-9 else float("inf")
        pnl = rs.sum() * FIXED_RISK; tot += pnl
        print(f"  {y:>6}{len(rs):>5}{rs.sum():>9.2f}{pnl:>11,.0f}{pf:>7.2f}")
    allR = np.array([r for (_, _, r, _) in R])
    total_pnl = allR.sum() * FIXED_RISK
    # era split
    def era_pnl(lo, hi):
        rs = np.array([r for (xt, _, r, _) in R if lo <= xt.year <= hi])
        return rs.sum() * FIXED_RISK, len(rs)
    p2325, n2325 = era_pnl(2023, 2025)
    p_all = total_pnl
    print(f"\n  TOTAL basket PnL = ${p_all:,.0f}  (n={len(allR)})")
    print(f"  2023-2025 slice: ${p2325:,.0f}  ({100*p2325/p_all:.0f}% of total PnL, n={n2325})")
    p2324, n2324 = era_pnl(2023, 2024)
    print(f"  2023-2024 slice: ${p2324:,.0f}  ({100*p2324/p_all:.0f}% of total PnL, n={n2324})")
    # concentration: top asset share
    by_asset = defaultdict(float)
    for (_, _, r, name) in R:
        by_asset[name] += r * FIXED_RISK
    gold = by_asset.get("GOLD", 0.0)
    print(f"  GOLD alone: ${gold:,.0f} ({100*gold/p_all:.0f}% of total PnL) -- single-asset concentration")


# ---------------------------------------------------------------------------
# M3 DEFLATED EXPECTATION
# ---------------------------------------------------------------------------
def deflated_expectation(assets):
    print("\n" + "=" * 92)
    print("M3 -- DEFLATED FORWARD EXPECTATION (expected-max / reality-check haircut)")
    print("=" * 92)
    allR = np.array([r for (_, _, r, _) in basket_R(assets)])
    n = len(allR); mu = allR.mean(); sd = allR.std(ddof=1); se = sd / np.sqrt(n)
    t = mu / se
    b = basket_stats_from_R([r for (_, _, r, _) in basket_R(assets)])
    print(f"  in-sample basket: n={n}  meanR={mu:+.3f}  sdR={sd:.3f}  SE={se:.3f}  t={t:.2f}  PF={b['PF']:.2f}")

    # Expected max of K independent N(0,1) (Gumbel approx) = the selection hurdle.
    def emax(K):
        if K <= 1: return 0.0
        a = np.sqrt(2 * np.log(K))
        return a - (np.log(np.log(K)) + np.log(4 * np.pi)) / (2 * a)
    print(f"\n  Garden of forking paths (door-shaped variants searched add.45-59), deflation by K:")
    print(f"  {'K':>4}{'E[max_K]':>10}{'defl_t':>8}{'defl_meanR':>11}{'implied_PF':>11}")
    # crude PF map: hold the loser distribution, scale winners so mean shifts to defl_mu.
    win = allR[allR > 0]; los = allR[allR < 0]
    for K in [1, 5, 10, 20, 30, 50]:
        hurdle = emax(K)
        dt = t - hurdle
        dmu = dt * se
        # implied PF: shift every R by (dmu - mu) i.e. lower the whole distribution to the deflated mean,
        # then recompute PF (conservative: treats the haircut as a uniform expectancy cut).
        shifted = allR + (dmu - mu)
        pos = shifted[shifted > 0].sum(); neg = -shifted[shifted < 0].sum()
        pf = pos / neg if neg > 1e-9 else float("inf")
        print(f"  {K:>4}{hurdle:>10.2f}{dt:>8.2f}{dmu:>11.3f}{pf:>11.2f}")
    print("\n  READ: even at K=50 forking paths the deflated t stays ~1.8 and deflated meanR ~0.25-0.30R")
    print("  (~half the in-sample 0.50R). The CROSS-ASSET identical-param survival (Gold/NDX/SPX, add.47/48)")
    print("  is the main mitigant -- params were NOT fit per asset, so most forking was at door-CONCEPT")
    print("  selection, partially controlled by the independent asset classes.")

    # Pre-registered forward test
    print("\n  " + "-" * 84)
    print("  PRE-REGISTERED FORWARD ACCEPTANCE TEST (lock now):")
    print("  Unit = the cross-asset BASKET (fixed 1% risk, >=10 Coinbase-INTX-tradeable assets).")
    print("  Deflated H1 expectation: meanR ~= 0.26R/trade, PF ~= 1.7; H0 (dead): meanR<=0, PF<=1.0.")
    cad = 4.5   # basket trades/yr (add.54/58/59)
    for target_n in [30, 40, 60]:
        yrs = target_n / cad
        # detectable meanR at n with the paired 1-sample bootstrap ~ 1.96*sd/sqrt(n)
        mde = 1.96 * sd / np.sqrt(target_n)
        print(f"    n={target_n} (~{yrs:.1f}yr @ {cad:.1f} basket-tr/yr): CONFIRM if basket PF>=1.5 AND "
              f"1-sample meanR bootstrap CI-lo>0 (MDE on meanR ~= {mde:.2f}R);")
    print("    REFUTE if PF<1.0 with CI-hi<0; else KEEP COLLECTING. Bear-rally (below-EMA200) fires")
    print("    logged as a flag, never filtered (add.48-50). Finish line is MULTI-YEAR by construction:")
    print("    at ~4.5 basket-trades/yr a 30-trade verdict needs ~7 years -- this is itself a finding")
    print("    about why 'nothing works' is empirically un-adjudicable on the available history.")


# ---------------------------------------------------------------------------
# M4(c) PAIRED-FRAME LIMITATION (analytical enumeration, printed)
# ---------------------------------------------------------------------------
def paired_frame_note():
    print("\n" + "=" * 92)
    print("M4(c) PAIRED-FRAME STRUCTURAL LIMITATION")
    print("=" * 92)
    print("  The overlay/paired machinery can only RE-WEIGHT or RE-EXIT the same ~106 fires. Any")
    print("  hypothesis whose value is NEW TRADE GENERATION cannot pass this frame by construction:")
    print("   - campaign topology re-entries (add.57/58): value claim = MORE at-size entries -> the")
    print("     paired-campaign frame compares the SAME campaigns, so extra cadence can only show as")
    print("     ΔPnL/campaign, and the door emits ~1 entry/campaign -> structurally ~0 to measure.")
    print("   - LTF generator / 4H fractal (add.52-55): value claim = MORE setups at faster scale ->")
    print("     the door produced the SAME count at 4H; a paired re-weight of 8-9 fires cannot create")
    print("     the cadence the hypothesis is about.")
    print("   - Gann entry windows (add.58/59): re-weighting 106 fires by an in/out flag -> the in/out")
    print("     SPLIT machinery (M1b) has MDE=inf here, so even a real +0.3R window is invisible.")
    print("  => these were UN-PASSABLE given the frame + n, independent of whether an effect exists.")


def main():
    print("Loading basket for M3/M4 ...")
    assets = load_basket(with_campaigns=False, verbose=False)
    shadow_queue(assets)
    era_composition(assets)
    deflated_expectation(assets)
    paired_frame_note()


if __name__ == "__main__":
    main()
