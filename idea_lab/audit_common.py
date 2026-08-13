"""
AUDIT COMMON  (wyckoff_audit add.60 -- ADVERSARIAL AUDIT OF THE TESTING METHODOLOGY)
===================================================================================
Shared loader: reproduce the v1 trend-continuation door across the 12-market basket
(add.54/58/59 set), collect per-trade R with entry/exit times, and the CampaignV2
campaign segmentation used by the paired-campaign verdict machinery.

STUDY ONLY. No production code touched. Reuses the exact idea_lab building blocks so
the audit measures the SAME apparatus the add.48-59 verdicts used.
"""
from __future__ import annotations
import os, sys, warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester import run_backtest, INITIAL_CASH, RISK_PCT
from campaign_backtester import run_campaign_backtest
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from campaign_strategy import CampaignV2Door
from xasset_spx_port import load_spx

SC = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
      "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD", "XRP-USD", "ADA-USD",
          "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD"]
XASSET = [("GOLD", f"{SC}/xasset/GOLD_1D.parquet"),
          ("NDX",  f"{SC}/xasset/NDX_1D.parquet")]
FIXED_RISK = 0.01 * INITIAL_CASH


def load_basket(with_campaigns: bool = True, verbose: bool = True):
    """Return dict name -> {df, sr, bj, eye, v1 (dict), camps (dict cid->trades)}.
    v1 = the add.54 headline trend-continuation door (single entry, struct exits).
    camps = CampaignV2Door segmentation (for the paired-campaign cadence machinery)."""
    assets = {}
    specs = [(s, os.path.join(SC, "one_strategy", f"{s}.parquet"), True) for s in CRYPTO]
    specs += [(n, p, False) for n, p in XASSET]
    for name, p, hv in specs:
        if not os.path.exists(p):
            if verbose: print(f"  MISSING {name}")
            continue
        df, sr, bj, eye = build_daily_sensors(load_spx(p))
        v1 = run_backtest(df, TrendContinuationDoor(df, sr, bj, eye, variant="struct",
                                                    conviction=False), label="v1")
        rec = dict(df=df, sr=sr, bj=bj, eye=eye, hv=hv, v1=v1)
        if with_campaigns:
            v2 = run_campaign_backtest(df, CampaignV2Door(df, sr, bj, eye), label="v2")
            camps = defaultdict(list)
            for t in v2["trades"]:
                camps[t["meta"]["campaign_id"]].append(t)
            rec["v2"] = v2
            rec["camps"] = camps
        assets[name] = rec
        if verbose:
            print(f"  {name:<9} v1 n={v1['stats']['n']:>3} PF={v1['stats']['PF']} "
                  f"PnL={v1['stats']['PnL']:>10,.0f}")
    return assets


def basket_R(assets):
    """Pooled list of (exit_time, entry_time, R, name) for the v1 door."""
    out = []
    for name, a in assets.items():
        for t in a["v1"]["trades"]:
            out.append((t["exit_time"], t["entry_time"], t["R"], name))
    return out


def basket_stats_from_R(rlist):
    """Fixed-1%-risk basket over [(exit_time,...,R,...)] or [R]. Ordered by exit if tuples."""
    if not rlist:
        return {"PF": 0.0, "PnL": 0.0, "MaxDD": 0.0, "n": 0}
    if isinstance(rlist[0], tuple):
        tr = sorted(rlist, key=lambda x: x[0])
        Rs = [x[2] for x in tr]
    else:
        Rs = list(rlist)
    eq = [INITIAL_CASH]; e = INITIAL_CASH; pnls = []
    for R in Rs:
        d = R * FIXED_RISK; e += d; eq.append(e); pnls.append(d)
    eq = np.array(eq); peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min() * 100
    p = np.asarray(pnls, float)
    w = p[p > 0].sum(); l = -p[p < 0].sum()
    pf = (w / l) if l > 1e-9 else (float("inf") if w > 0 else 0.0)
    return {"PF": pf, "PnL": float(np.sum(pnls)), "MaxDD": float(dd), "n": len(Rs)}


if __name__ == "__main__":
    print("Loading 12-market basket (v1 door + CampaignV2 segmentation)...")
    a = load_basket()
    R = basket_R(a)
    b = basket_stats_from_R(R)
    allR = np.array([x[2] for x in R])
    print(f"\nBASKET: n={b['n']}  PF={b['PF']:.2f}  PnL=${b['PnL']:,.0f}  MaxDD={b['MaxDD']:.2f}%")
    print(f"R distribution: mean={allR.mean():+.3f}  sd={allR.std(ddof=1):.3f}  "
          f"min={allR.min():+.2f}  max={allR.max():+.2f}  WR={(allR>0).mean():.1%}")
    ncamp = sum(len(a[n]["camps"]) for n in a if "camps" in a[n])
    print(f"CampaignV2 campaigns total: {ncamp}")
