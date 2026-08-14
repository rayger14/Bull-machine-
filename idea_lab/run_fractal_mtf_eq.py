"""PHASE 3 second family: crypto-INDEPENDENT equity/metal 4H fractal (~2.9y intraday).
Same throttle-fixed native-eye machinery as run_fractal_mtf.py. Short window (yfinance
730d cap) => DIRECTIONAL power only, and 2023-26 is BULL-skewed (flagged)."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from run_fractal_mtf import (build_asset_4h, run_arm, daily_ref, stats_from_trades,
                             boot_ci, pf_of, kblock, in_window, OHLCV)

EQDIR = "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/fractal_mtf_eq"


def load_eq(sym):
    df = pd.read_parquet(f"{EQDIR}/{sym}_1H.parquet")
    return df[OHLCV].sort_index()


def main():
    syms = [os.path.basename(f).replace("_1H.parquet", "")
            for f in sorted(glob.glob(f"{EQDIR}/*_1H.parquet"))]
    pool = {"naked": [], "gated": []}
    daily_pool = []
    dpy = 0.0; lpy = {"naked": 0.0, "gated": 0.0}; na = 0
    for sym in syms:
        df_1h = load_eq(sym)
        dref, dyrs = daily_ref(df_1h)
        df, sr, bj, eye4, htf = build_asset_4h(df_1h)
        yrs4 = (df.index[-1] - df.index[0]).days / 365.25
        tn, _ = run_arm(df, sr, bj, eye4, htf, "naked")
        tg, _ = run_arm(df, sr, bj, eye4, htf, "gated")
        for t in tn: t["_sym"] = sym
        for t in tg: t["_sym"] = sym
        pool["naked"] += tn; pool["gated"] += tg; daily_pool += dref
        sd = stats_from_trades(dref, dyrs); sn = stats_from_trades(tn, yrs4); sg = stats_from_trades(tg, yrs4)
        dpy += sd["perYr"]; lpy["naked"] += sn["perYr"]; lpy["gated"] += sg["perYr"]; na += 1
        print(f"{sym:<6} DAILY n={sd['n']:<3}({sd['perYr']:.2f}/yr) PF={sd['PF']:.2f}  |  "
              f"NAKED n={sn['n']:<3}({sn['perYr']:.2f}/yr) PF={sn['PF']:.2f} mR={sn['meanR']:+.2f}  |  "
              f"GATED n={sg['n']:<3}({sg['perYr']:.2f}/yr) PF={sg['PF']:.2f} mR={sg['meanR']:+.2f}")
    print("\n" + "=" * 78)
    print(f"POOLED EQUITY/METAL 4H FAMILY ({na} assets, ~2.9y intraday, BULL-skewed)")
    print(f"  DAILY door ref: n={len(daily_pool)} PF={pf_of(daily_pool):.2f} cadence={dpy:.2f}/yr total")
    for arm in ("naked", "gated"):
        tp = pool[arm]; Rs = [t["R"] for t in tp]
        lo, hi = boot_ci(Rs); k1, k15 = kblock(tp)
        ab = np.mean([t.get("entry_above_ema200", True) for t in tp]) * 100 if tp else 0
        print(f"\n  --- LTF-{arm.upper()} ---")
        print(f"    n={len(tp)} PF={pf_of(tp):.2f} PnL(1%R)=${sum(t['pnl'] for t in tp):+,.0f} "
              f"meanR={np.mean(Rs):+.3f} bootCI[{lo:+.3f},{hi:+.3f}]")
        print(f"    above-EMA200={ab:.0f}% K6 frac>1={k1:.0%} frac>=1.5={k15:.0%} "
              f"cadence={lpy[arm]:.1f}/yr total mult={lpy[arm]/max(dpy,1e-9):.1f}x")


if __name__ == "__main__":
    main()
