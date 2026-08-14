"""
PHASE 3 — POWERED FRACTAL MTF TEST (study only, pre-registered).
================================================================
The doctrine-faithful role-separated architecture with the add.55 throttle FIXED
(native-resolution eye, see eye_native.py; diagnosed-bug repair, not a tuning choice).

ROLE SEPARATION (Phase-2 research verdict = role iii; Elder/ICT/Wyckoff converge):
  HTF = WHERE/WHETHER  -> the validated door's campaign context = DAILY eye bull
        standing bias active (a confirmed daily up-break not yet given back).
  LTF = WHEN           -> a FRESH 4H break-retest-hold structure inside that context.
Two arms, reported separately (task: funnel-fix finding vs edge finding):
  LTF-NAKED   = 4H door, native eye, NO HTF gate           (role i: "same pattern faster")
  LTF-GATED   = 4H door fires ONLY while DAILY campaign on  (role iii: HTF context + LTF struct)
Reference = the validated DAILY door (cadence-multiplier denominator).

Costs 2bps commission + 3bps slippage (backtester defaults), fixed 1% risk (and 0.5%
reported per WI size doctrine for the tactical layer). NO per-asset tuning. NO grids.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd

from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
from trend_continuation_door import (TrendContinuationDoor, build_daily_sensors,
                                     build_base_sensors, resample_daily)
import trend_continuation_door as tcd
from structural_range import build_structural_range
from probe_fractal_4h import resample_4h, reanchor_frame_daily
from xasset_spx_port import HTF_N
from eye_native import compute_eye_native
from engine.features import eye_state as ES

SCRATCH = "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad"
BTC_1H = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
H1DIR = f"{SCRATCH}/fractal_execution/h1"
OHLCV = ["open", "high", "low", "close", "volume"]

TRAIN = ("2018-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-08-31")
# bear windows (crypto), for regime stratification
BEARS = [("2018-01-01", "2018-12-15"), ("2021-11-11", "2022-12-31"), ("2025-01-01", "2025-06-30")]


def load_1h(sym):
    if sym == "BTC":
        df = pd.read_parquet(BTC_1H)[OHLCV].copy()
    else:
        f = f"{H1DIR}/{sym}-USD_1H.parquet"
        df = pd.read_parquet(f)
        df = df.set_index(pd.DatetimeIndex(df["timestamp"]))[OHLCV].sort_index()
    return df


class HTFGatedDoor:
    """Wrap TrendContinuationDoor; only allow a fire when HTF campaign bull-bias is on
    at bar i (causal: htf_bull already lagged to the exec grid)."""
    def __init__(self, df, sr, bj, eye, htf_bull, variant="struct"):
        self._d = TrendContinuationDoor(df, sr, bj, eye, variant=variant, conviction=False)
        self.entries_log = self._d.entries_log
        self._htf = htf_bull.astype(bool).to_numpy()

    def __call__(self, df, i):
        if not self._htf[i]:
            return None
        return self._d(df, i)


def build_asset_4h(df_1h):
    """4H exec, native 4H eye (throttle fixed), daily-N5 anchor struct range,
    plus the DAILY HTF campaign context (daily native eye bull bias) lagged to 4H."""
    d4 = resample_4h(df_1h)
    df = build_base_sensors(d4)
    reanch = reanchor_frame_daily(df, HTF_N)
    sr = build_structural_range(reanch)
    bj = tcd.build_bojan(reanch, sr, tcd.BOJAN_W)
    eye4 = compute_eye_native(d4)                       # LTF (throttle fixed)
    # HTF context = DAILY native eye bull bias, on this asset's own daily bars
    d1 = resample_daily(df_1h)
    eye1 = compute_eye_native(d1)
    htf_bull_daily = (eye1["eye_dir"] == "bull")
    # lag daily->4H causally: daily state for day D visible from D 24:00 (merge_asof backward)
    av = htf_bull_daily.copy()
    av.index = av.index + pd.Timedelta(days=1)
    htf_bull_4h = av.reindex(df.index, method="ffill").fillna(False)
    return df, sr, bj, eye4, htf_bull_4h


def run_arm(df, sr, bj, eye, htf_bull, arm, risk=RISK_PCT):
    if arm == "naked":
        strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    else:
        strat = HTFGatedDoor(df, sr, bj, eye, htf_bull, variant="struct")
    res = run_backtest(df, strat, risk_pct=risk, label=arm)
    trades = res["trades"]
    below = {e["entry_time"]: e["below_ema200"] for e in strat.entries_log}
    for t in trades:
        t["entry_above_ema200"] = not below.get(t["entry_time"], False)
    return trades, strat


def daily_ref(df_1h):
    d1 = resample_daily(df_1h)
    dfd, srd, bjd, eyed = build_daily_sensors(d1)
    strat = TrendContinuationDoor(dfd, srd, bjd, eyed, variant="struct", conviction=False)
    res = run_backtest(dfd, strat, risk_pct=RISK_PCT, label="dailyref")
    yrs = (dfd.index[-1] - dfd.index[0]).days / 365.25
    return res["trades"], yrs


def stats_from_trades(trades, span_yrs):
    if not trades:
        return dict(n=0, WR=0, PF=0, PnL=0, meanR=0, MaxDD=0, ab=0, perYr=0)
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in trades: e += t["pnl"]; eq.append(e)
    s = compute_stats(trades, eq, INITIAL_CASH)
    ab = sum(1 for t in trades if t.get("entry_above_ema200", True))
    return dict(n=s["n"], WR=s["WR"], PF=s["PF"], PnL=s["PnL"],
                meanR=float(np.mean([t["R"] for t in trades])),
                MaxDD=s["MaxDD_pct"], perYr=s["n"]/max(span_yrs, 1e-9))


def boot_ci(Rs, nboot=5000, seed=42):
    if len(Rs) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    Rs = np.asarray(Rs)
    means = [rng.choice(Rs, len(Rs), replace=True).mean() for _ in range(nboot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def pf_of(trades):
    g = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    l = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    return (g / l) if l > 0 else float("inf")


def kblock(trades, k=6):
    """K-block stability: split timeline into k blocks, frac with PF>1 and >=1.5."""
    if len(trades) < k:
        return (np.nan, np.nan)
    ts = sorted(trades, key=lambda t: t["entry_time"])
    blocks = np.array_split(ts, k)
    pfs = [pf_of(list(b)) for b in blocks if len(b)]
    return (np.mean([p > 1 for p in pfs]), np.mean([p >= 1.5 for p in pfs]))


def in_window(t, a, b):
    return pd.Timestamp(a) <= t["entry_time"] <= pd.Timestamp(b)


def main():
    syms = ["BTC", "ETH", "LTC", "SOL", "LINK"]
    pool = {"naked": [], "gated": []}
    daily_pool = []
    per_asset = []
    daily_total_peryr = 0.0
    ltf_total_peryr = {"naked": 0.0, "gated": 0.0}
    n_assets = 0

    for sym in syms:
        df_1h = load_1h(sym)
        dref, dyrs = daily_ref(df_1h)
        df, sr, bj, eye4, htf = build_asset_4h(df_1h)
        yrs4 = (df.index[-1] - df.index[0]).days / 365.25
        tn, _ = run_arm(df, sr, bj, eye4, htf, "naked")
        tg, _ = run_arm(df, sr, bj, eye4, htf, "gated")
        for t in tn: t["_sym"] = sym
        for t in tg: t["_sym"] = sym
        pool["naked"] += tn; pool["gated"] += tg; daily_pool += dref
        sd = stats_from_trades(dref, dyrs)
        sn = stats_from_trades(tn, yrs4); sg = stats_from_trades(tg, yrs4)
        per_asset.append((sym, sd, sn, sg))
        daily_total_peryr += sd["perYr"]; n_assets += 1
        ltf_total_peryr["naked"] += sn["perYr"]; ltf_total_peryr["gated"] += sg["perYr"]
        print(f"{sym:<5} DAILY n={sd['n']:<3}({sd['perYr']:.2f}/yr) PF={sd['PF']:.2f}  |  "
              f"NAKED n={sn['n']:<3}({sn['perYr']:.2f}/yr) PF={sn['PF']:.2f} meanR={sn['meanR']:+.2f}  |  "
              f"GATED n={sg['n']:<3}({sg['perYr']:.2f}/yr) PF={sg['PF']:.2f} meanR={sg['meanR']:+.2f}")

    print("\n" + "=" * 78)
    print("POOLED CRYPTO 4H FAMILY (5 assets)")
    d_n = len(daily_pool)
    print(f"  DAILY door ref: n={d_n}  PF={pf_of(daily_pool):.2f}  "
          f"cadence(sum/asset avg)={daily_total_peryr:.2f}/yr total, {daily_total_peryr/n_assets:.2f}/yr avg")
    for arm in ("naked", "gated"):
        tp = pool[arm]
        Rs = [t["R"] for t in tp]
        lo, hi = boot_ci(Rs)
        k1, k15 = kblock(tp)
        ab = np.mean([t.get("entry_above_ema200", True) for t in tp]) * 100 if tp else 0
        cad = ltf_total_peryr[arm]
        print(f"\n  --- LTF-{arm.upper()} ---")
        print(f"    n={len(tp)}  PF={pf_of(tp):.2f}  PnL(1%R)=${sum(t['pnl'] for t in tp):+,.0f}  "
              f"meanR={np.mean(Rs):+.3f}  bootCI[{lo:+.3f},{hi:+.3f}]")
        print(f"    above-EMA200={ab:.0f}%  K6 frac>1={k1:.0%} frac>=1.5={k15:.0%}  "
              f"cadence total={cad:.1f}/yr avg={cad/n_assets:.1f}/yr")
        print(f"    CADENCE MULTIPLIER vs daily = {cad/max(daily_total_peryr,1e-9):.1f}x")
        # train/OOS
        for label, (a, b) in [("TRAIN", TRAIN), ("OOS", OOS)]:
            sub = [t for t in tp if in_window(t, a, b)]
            if sub:
                print(f"    {label:<5} n={len(sub):<4} PF={pf_of(sub):.2f} "
                      f"PnL=${sum(t['pnl'] for t in sub):+,.0f} meanR={np.mean([t['R'] for t in sub]):+.3f}")
        # regime: bear windows
        bear = [t for t in tp if any(in_window(t, a, b) for a, b in BEARS)]
        if bear:
            print(f"    BEAR  n={len(bear):<4} PF={pf_of(bear):.2f} PnL=${sum(t['pnl'] for t in bear):+,.0f}")

    # 0.5% risk restatement (linear in risk for PnL; PF/meanR invariant)
    print("\n  (0.5% risk: PnL halves, PF & meanR & multiplier unchanged — R is scale-free.)")


if __name__ == "__main__":
    main()
