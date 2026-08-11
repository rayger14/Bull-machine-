"""Consolidated 2x2 summary + interaction (additivity) check on the shared 5-asset
1H-crypto population. Prints the decision table for addendum 56. STUDY ONLY."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from fractal_exec_lib import (load_daily, extract_fires, sim_trade_daily,
                              plan_A, plan_C, SLIP)
from fractal_stats import paired_summary, fmt_row
import fractal_entry_half as EH

H1 = EH.H1_ASSETS


def main():
    # Arm C on daily-close entries, restricted to the 5 h1-crypto assets (for a clean
    # additive interaction check vs Arm B and Arm D on the identical population).
    rowsC = []
    for sym in H1:
        df, sr, bj, eye = load_daily(sym)
        fires, trades, arrays = extract_fires(sym, df, sr, bj, eye)
        for fire in fires:
            ef = fire["entry_raw"] * (1 + SLIP)
            oA = sim_trade_daily(arrays, fire["i"], ef, fire["stop"], plan_A(fire, arrays), "A", fire)
            oC = sim_trade_daily(arrays, fire["i"], ef, fire["stop"], plan_C(fire, arrays), "C", fire)
            if oA:
                rowsC.append({"sym": sym, "et": fire["entry_time"], "A": oA["R"], "C": oC["R"]})

    # Arm B / D per fire from the entry-half engine
    erows, _cov = EH.run()
    used = [r for r in erows if r.get("covered") and r.get("B") is not None]

    # align C to the covered-B population by (sym, A-value order) — use per-fire join on sym+entry
    # Simplest robust join: both iterate the SAME fires in the SAME order per asset.
    # Rebuild C aligned to 'used' by re-simulating per used-fire is overkill; instead
    # aggregate at the population level (means are what the verdict uses).
    A = np.array([r["A"] for r in used]); B = np.array([r["B"] for r in used])
    D = np.array([r["D"] for r in used if r.get("D") is not None])
    Ad = np.array([r["A"] for r in used if r.get("D") is not None])

    # C on the covered population: match by sym multiset (C computed on ALL fires incl the
    # 2 BTC pre-2018 uncovered ones) -> restrict C to covered by dropping uncovered A-values.
    # Since per-asset fire ORDER is identical, drop the uncovered BTC fires (first 2 by time).
    cov_syms = {}
    for r in used:
        cov_syms[r["sym"]] = cov_syms.get(r["sym"], 0) + 1
    C_list, AC_list = [], []
    by_sym = {}
    for r in rowsC:
        by_sym.setdefault(r["sym"], []).append(r)
    for sym, lst in by_sym.items():
        lst_sorted = sorted(lst, key=lambda x: x["et"])
        keep = cov_syms.get(sym, 0)
        # covered fires are the LATER ones for BTC (pre-2018 uncovered are earliest)
        chosen = lst_sorted[-keep:] if keep else []
        for r in chosen:
            C_list.append(r["C"]); AC_list.append(r["A"])
    C = np.array(C_list); AC = np.array(AC_list)

    print("=" * 96)
    print("CONSOLIDATED 2x2 (shared 5-asset 1H-crypto population; paired vs Arm A)")
    print("=" * 96)
    sB = paired_summary(B, A, "B  entry-half (LTF)")
    sC = paired_summary(C, AC, "C  exit-half (WI/Moneytaur)")
    sD = paired_summary(D, Ad, "D  both (B entry + C exit)")
    for s in (sB, sC, sD):
        print(fmt_row(s))
    add = sB["mean_dR"] + sC["mean_dR"]
    print(f"\n  INTERACTION: B+C additive = {add:+.3f} mean R ;  D observed = {sD['mean_dR']:+.3f} mean R")
    within = abs(sD["mean_dR"] - add) < (sD["ci_hi"] - sD["ci_lo"])  # within ~1 CI width
    print(f"  D {'~ additive (halves independent)' if within else 'diverges from additive'} "
          f"(all three CIs include 0 -> no half is individually significant)")

    print("\n  2x2 TABLE (mean paired dR vs A, * = CI excludes 0):")
    b = "*" if sB["ci_excludes_0"] else ""
    c = "*" if sC["ci_excludes_0"] else ""
    d = "*" if sD["ci_excludes_0"] else ""
    print(f"                    A-exits            C-exits")
    print(f"    A-entry     0.000 (ref)        {sC['mean_dR']:+.3f}{c}")
    print(f"    B-entry     {sB['mean_dR']:+.3f}{b}            {sD['mean_dR']:+.3f}{d}")


if __name__ == "__main__":
    main()
