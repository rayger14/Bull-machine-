#!/usr/bin/env python3
"""Phase-1 tick-flow splits — PRE-REGISTERED in backfill_aggtrades_flow.py.

S1  cvd_true 24h divergence at entry (price new 24h low, cvd_true holds)
S2  delta_at_low <= 0 at entry hour (sellers aggressed the extreme)
S3  delta_after_low > 0 at entry hour (aggression flipped to buyers post-low)

Populations: the CLEAN wick_trap trade logs (wt_train / wt_hold).
Bar: consistent direction in BOTH eras, n>=30/cell for claims.
Usage: python3 scripts/champion/tick_flow_splits.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FLOW_DIR = REPO / "results/rebuild/aggtrades_flow"
CLEAN = Path("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-"
             "/da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/clean")


def pf(x: pd.Series) -> float:
    gp = x[x > 0].sum()
    gl = -x[x < 0].sum()
    return round(gp / gl, 2) if gl > 0 else float("inf")


def main() -> None:
    flow = pd.concat([pd.read_parquet(f) for f in sorted(FLOW_DIR.glob("*.parquet"))])
    flow = flow[~flow.index.duplicated(keep="last")].sort_index()
    print(f"flow hours: {len(flow):,} ({flow.index[0]} -> {flow.index[-1]})")

    v = pd.read_parquet(REPO / "data/features_mtf/BTC_1H_FEATURES_V15_STRUCTURE.parquet",
                        columns=["low"])
    if v.index.tz is not None:
        v.index = v.index.tz_localize(None)
    df = v.join(flow, how="left")
    df["cvd_true"] = df["true_delta"].fillna(0).cumsum()
    prior_low = df["low"].rolling(24).min().shift(1)
    prior_cvd = df["cvd_true"].rolling(24).min().shift(1)
    df["s1_div"] = (df["low"] < prior_low) & (df["cvd_true"] > prior_cvd)
    df["s2_sellers_at_low"] = df["delta_at_low"] <= 0
    df["s3_recovery_flip"] = df["delta_after_low"] > 0

    for w, tag in [("wt_train", "TRAIN"), ("wt_hold", "HOLDOUT")]:
        t = pd.read_csv(CLEAN / w / "trade_log.csv")
        t["ts"] = pd.to_datetime(t["timestamp"], format="mixed") \
            .dt.tz_localize(None).dt.floor("h")
        pos = t.groupby("position_id").agg(pnl=("pnl", "sum"), ts=("ts", "first"))
        f = df.reindex(pos["ts"]).set_index(pos.index)
        miss = int(f["true_delta"].isna().sum())
        print(f"\n=== wick_trap {tag} (n={len(pos)}, base PF {pf(pos.pnl)},"
              f" flow-missing {miss}) ===")
        for name in ["s1_div", "s2_sellers_at_low", "s3_recovery_flip"]:
            m = f[name].fillna(False).values
            a, b = pos.pnl[m], pos.pnl[~m & f["true_delta"].notna().values]
            print(f"  {name:20s} YES n={len(a):>3} PF={pf(a) if len(a) else '--'} "
                  f"${a.sum():>9,.0f} | NO n={len(b):>3} PF={pf(b) if len(b) else '--'} "
                  f"${b.sum():>9,.0f}")


if __name__ == "__main__":
    main()
