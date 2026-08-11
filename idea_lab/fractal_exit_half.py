"""EXIT HALF (Arm A vs Arm C vs C-trail) — paired per-trade deltas on the SAME
door fires across ALL 13 markets (crypto + equities + gold). STUDY ONLY.

Arm A  = daily-close entry + add.54 struct exits (headline reference).
Arm C  = daily-close entry + WI/Moneytaur exit engine (bojan/wick-magnet/negfib TP1,
         negfib-ext runner + Moneytaur structure trail).
C-trail= baseline struct TP1 + Moneytaur trail replacing measured-move runner (ablation).

PASS rule (pre-registered): C mean dR > 0 with bootstrap CI excluding 0 AND runner
tail preserved (max-R not collapsed >30%) AND consistent sign in >=2/3 families."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import Counter

from fractal_exec_lib import (load_daily, extract_fires, sim_trade_daily, plan_A,
                              plan_C, plan_C_trail, CRYPTO, XASSET, SLIP, FAMILY, CFG)
from fractal_stats import paired_summary, fmt_row

ALL = list(CRYPTO) + XASSET


def run():
    rows = []   # per fire: dict(sym, family, A, C, Ctrail, maxR_A, maxR_C, tp1_src...)
    for sym in ALL:
        df, sr, bj, eye = load_daily(sym)
        fires, trades, arrays = extract_fires(sym, df, sr, bj, eye)
        for fire in fires:
            ef = fire["entry_raw"] * (1 + SLIP)
            oA = sim_trade_daily(arrays, fire["i"], ef, fire["stop"], plan_A(fire, arrays), "A", fire)
            oC = sim_trade_daily(arrays, fire["i"], ef, fire["stop"], plan_C(fire, arrays), "C", fire)
            oT = sim_trade_daily(arrays, fire["i"], ef, fire["stop"], plan_C_trail(fire, arrays), "Ct", fire)
            if oA is None:
                continue
            rows.append({
                "sym": sym, "family": fire["family"],
                "A": oA["R"], "C": oC["R"], "Ct": oT["R"],
                "maxR_A": oA["max_R"], "maxR_C": oC["max_R"],
                "rA": oA["reason"], "rC": oC["reason"], "rT": oT["reason"],
                "below_ema200": fire["below_ema200"],
            })
    return rows


def verdict_block(rows):
    A = np.array([r["A"] for r in rows]); C = np.array([r["C"] for r in rows])
    Ct = np.array([r["Ct"] for r in rows])
    print("\n" + "=" * 100)
    print(f"EXIT HALF — paired per-trade deltas (n={len(rows)} fires, all 13 markets)")
    print("=" * 100)
    sC = paired_summary(C, A, "C (full WI engine) vs A")
    sT = paired_summary(Ct, A, "C-trail (trail only) vs A")
    print(fmt_row(sC))
    print(fmt_row(sT))

    # runner-tail preservation: max single-trade REALIZED R
    maxA, maxC = A.max(), C.max()
    tail_ok = maxC >= 0.70 * maxA
    print(f"\n  RUNNER TAIL: max single-trade realized R  A={maxA:.2f}  C={maxC:.2f}  "
          f"(C/A={maxC/maxA:.2f})  -> {'PRESERVED' if tail_ok else 'COLLAPSED >30% (FAIL)'}")

    # per-family sign consistency
    print("\n  PER-FAMILY mean dR (C vs A):")
    fam_signs = []
    for fam in ["crypto", "equity", "gold"]:
        idx = [k for k, r in enumerate(rows) if r["family"] == fam]
        if not idx:
            print(f"    {fam:<8} n=0"); continue
        d = C[idx] - A[idx]
        sign = "+" if d.mean() > 0 else ("-" if d.mean() < 0 else "0")
        fam_signs.append(d.mean() > 0)
        print(f"    {fam:<8} n={len(idx):>3}  meanDR={d.mean():+.3f}  "
              f"medDR={np.median(d):+.3f}  totDR={d.sum():+.1f}  sign={sign}")
    fam_pos = sum(fam_signs)
    fam_ok = fam_pos >= 2

    # attribution via C-trail
    print("\n  ATTRIBUTION (C-trail isolates the Moneytaur trail alone):")
    print(f"    C full   meanDR={sC['mean_dR']:+.3f}  (targets + trail)")
    print(f"    C-trail  meanDR={sT['mean_dR']:+.3f}  (trail only, baseline targets)")
    print(f"    => target-craft contribution ~ {sC['mean_dR']-sT['mean_dR']:+.3f} mean R")

    # exit-reason mix
    print("\n  EXIT-REASON MIX:")
    print(f"    A : {dict(Counter(r['rA'] for r in rows))}")
    print(f"    C : {dict(Counter(r['rC'] for r in rows))}")

    ci_ok = sC["ci_excludes_0"] and sC["mean_dR"] > 0
    print("\n  PRE-REGISTERED EXIT-HALF VERDICT:")
    print(f"    (a) C meanDR>0 & CI excl 0 : {sC['mean_dR']:+.3f}, CI[{sC['ci_lo']:+.3f},{sC['ci_hi']:+.3f}] -> {'PASS' if ci_ok else 'FAIL'}")
    print(f"    (b) runner tail preserved  : C/A={maxC/maxA:.2f} -> {'PASS' if tail_ok else 'FAIL'}")
    print(f"    (c) >=2/3 families same sign: {fam_pos}/3 positive -> {'PASS' if fam_ok else 'FAIL'}")
    overall = ci_ok and tail_ok and fam_ok
    print(f"    => EXIT HALF {'PASSES' if overall else 'FAILS'}")
    return sC, sT


def main():
    rows = run()
    # per-asset quick table
    print("PER-ASSET total R (A vs C vs C-trail):")
    print(f"{'asset':<9}{'fam':<8}{'n':>4}{'totR_A':>9}{'totR_C':>9}{'totR_Ct':>9}{'dC':>8}{'dCt':>8}")
    for sym in ALL:
        rr = [r for r in rows if r["sym"] == sym]
        if not rr:
            print(f"{sym:<9}{'':<8}{0:>4}"); continue
        A = sum(r["A"] for r in rr); C = sum(r["C"] for r in rr); Ct = sum(r["Ct"] for r in rr)
        print(f"{sym:<9}{FAMILY[sym]:<8}{len(rr):>4}{A:>9.2f}{C:>9.2f}{Ct:>9.2f}{C-A:>+8.2f}{Ct-A:>+8.2f}")
    verdict_block(rows)


if __name__ == "__main__":
    main()
