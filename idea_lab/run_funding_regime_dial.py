"""
FORM B — FUNDING AS REGIME DIAL on the add.48 door (boost-shaped; STUDY ONLY; add.70)
=====================================================================================
Implements the pre-registration Form B. Runs the FROZEN trend-continuation door on
BTC-USD daily (resampled causally from 1H V22), flags each door entry by whether the
Form-A expanding-85th funding percentile says "crowded long" at entry, and reports the
PAIRED split (crowded vs clean). Sizing-tier framing, NEVER a gate.

Pre-flagged UNDERPOWERED: funding is BTC-only and the daily door fires ~10x on BTC.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
WT = f"{REPO}/.claude/worktrees/funding-carry"
sys.path.insert(0, f"{WT}/idea_lab")

from backtester import run_backtest, RISK_PCT  # noqa: E402
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor  # noqa: E402

PRICE = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
DERIV = f"{REPO}/data/cache/derivatives_hourly_full.parquet"
F3_WIN = 9
WARMUP = 270


def btc_daily():
    p = pd.read_parquet(PRICE)[["open", "high", "low", "close", "volume"]].copy()
    d = p.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                              "close": "last", "volume": "sum"}).dropna()
    d.index.name = None
    return d


def expanding_pct(x, q):
    out = np.full(len(x), np.nan); buf = []
    for k in range(len(x)):
        v = x[k]
        if v == v:
            buf.append(v)
        if buf:
            out[k] = np.percentile(buf, q)
    return out


def funding_crowded_series():
    """8h f3 and expanding-85th, as a Series indexed by 8h stamp (naive UTC)."""
    dv = pd.read_parquet(DERIV)["binance_funding_rate"].copy()
    dv.index = dv.index.tz_localize(None)
    f8 = dv.resample("8h").last().dropna()
    f3 = f8.rolling(F3_WIN).mean()
    p85 = expanding_pct(f3.to_numpy(), 85)
    crowded = (f3.to_numpy() > p85)
    return pd.Series(crowded, index=f8.index), pd.Series(f3.to_numpy(), index=f8.index), \
        pd.Series(p85, index=f8.index)


def flag_entry(entry_time, crowded_s):
    """crowded_long at entry = nearest settled 8h value at/just before the entry day."""
    te = pd.Timestamp(entry_time)
    sub = crowded_s[crowded_s.index <= te]
    if len(sub) == 0:
        return None
    return bool(sub.iloc[-1])


def boot_ci(a, b, n=10000, seed=7):
    """bootstrap 95% CI on mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.array(a); b = np.array(b)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    diffs = np.empty(n)
    for i in range(n):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def pf(nets):
    nets = np.array(nets)
    gp = nets[nets > 0].sum(); gl = -nets[nets < 0].sum()
    return float(gp / gl) if gl > 0 else float("inf")


def main():
    d = btc_daily()
    df, sr, bj, eye = build_daily_sensors(d)
    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label="BTC-door")
    trades = res["trades"]

    crowded_s, f3_s, p85_s = funding_crowded_series()
    # restrict to fires within the funding window
    fmin, fmax = crowded_s.index[0], crowded_s.index[-1]

    rows = []
    for t in trades:
        te = pd.Timestamp(t["entry_time"])
        if te < fmin or te > fmax:
            continue
        fl = flag_entry(t["entry_time"], crowded_s)
        if fl is None:
            continue
        rows.append(dict(entry=str(te.date()), R=t["R"], crowded=fl))

    crowded_R = [r["R"] for r in rows if r["crowded"]]
    clean_R = [r["R"] for r in rows if not r["crowded"]]

    def blk(name, arr):
        return dict(tier=name, n=len(arr),
                    meanR=round(float(np.mean(arr)), 4) if arr else None,
                    pf=round(pf(arr), 3) if arr else None)

    delta = (np.mean(clean_R) - np.mean(crowded_R)) if (clean_R and crowded_R) else float("nan")
    ci = boot_ci(clean_R, crowded_R) if (clean_R and crowded_R) else (float("nan"), float("nan"))

    powered = (len(crowded_R) >= 30 and len(clean_R) >= 30)
    adopt = bool(powered and delta > 0 and ci[0] > 0)

    out = dict(
        door_total_fires=len(trades),
        door_fires_in_funding_window=len(rows),
        funding_window=[str(fmin), str(fmax)],
        clean_tier=blk("clean (not crowded-long)", clean_R),
        crowded_tier=blk("crowded-long at entry", crowded_R),
        delta_clean_minus_crowded=round(float(delta), 4) if delta == delta else None,
        boot95_CI_on_delta=[round(ci[0], 4) if ci[0] == ci[0] else None,
                            round(ci[1], 4) if ci[1] == ci[1] else None],
        powered_n_ge_30_each=powered,
        VERDICT=("ADOPT sizing-tier candidate" if adopt else
                 ("DIRECTIONAL ONLY — underpowered (n<30 per tier); cannot power on one asset"
                  if not powered else "REJECT — no CI-supported crowded penalty")),
    )
    print("REGIME_DIAL_JSON_START")
    print(json.dumps(out, indent=2, default=str))
    print("REGIME_DIAL_JSON_END")


if __name__ == "__main__":
    main()
