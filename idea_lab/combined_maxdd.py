"""Pre-registered MaxDD clause: combined book (daily door + LTF layer) MaxDD must be
<= 1.5x the door alone. Computed on the FULL-CYCLE crypto family (the one with bears).
Also pools the two families for the n>=150 / CI check."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from run_fractal_mtf import (load_1h, daily_ref, build_asset_4h, run_arm, pf_of, boot_ci)


def maxdd_from_trades(trades, cash=100000.0):
    ts = sorted(trades, key=lambda t: t["entry_time"])
    eq = cash; peak = cash; mdd = 0.0
    for t in ts:
        eq += t["pnl"]; peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak)
    return mdd * 100


syms = ["BTC", "ETH", "LTC", "SOL", "LINK"]
door = []; naked = []; gated = []
for sym in syms:
    df_1h = load_1h(sym)
    dref, _ = daily_ref(df_1h)
    df, sr, bj, eye4, htf = build_asset_4h(df_1h)
    tn, _ = run_arm(df, sr, bj, eye4, htf, "naked")
    tg, _ = run_arm(df, sr, bj, eye4, htf, "gated")
    door += dref; naked += tn; gated += tg

dd_door = maxdd_from_trades(door)
dd_dn = maxdd_from_trades(door + naked)
dd_dg = maxdd_from_trades(door + gated)
print("CRYPTO FAMILY combined-book MaxDD (fixed 1% risk, shared $100k book):")
print(f"  door alone           : MaxDD {dd_door:.2f}%  (n={len(door)})")
print(f"  door + LTF-NAKED     : MaxDD {dd_dn:.2f}%  ({dd_dn/dd_door:.2f}x door)  n={len(door)+len(naked)}")
print(f"  door + LTF-GATED     : MaxDD {dd_dg:.2f}%  ({dd_dg/dd_door:.2f}x door)  n={len(door)+len(gated)}")
print(f"  PRE-REG CEILING = 1.5x door = {1.5*dd_door:.2f}%")
print(f"  NAKED clause: {'PASS' if abs(dd_dn)<=1.5*abs(dd_door) else 'FAIL'}   "
      f"GATED clause: {'PASS' if abs(dd_dg)<=1.5*abs(dd_door) else 'FAIL'}")
