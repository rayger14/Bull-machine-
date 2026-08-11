"""
FRACTAL PROBE (study only): the EXACT validated door geometry, one timeframe down.
Daily-exec/weekly-anchor (validated, add.48/54) -> 4H-exec/DAILY-anchor.
ALL params stay in BAR units (no retune): N_range=40 bars, ACCEPT_CONSEC=2,
M2_SOS_WIN=360 bars, LPS_LOOKBACK=48, RTZ_ATR=0.5, STOP_BUF=0.25, MAX_HOLD=168 bars.
On 4H: 40 bars ~ 6.7d ceiling, SOS window ~ 60d, max_hold ~ 28d.
Question: does the door's edge survive at the tactical timeframe (more trades), or
is this the third death of "speed it up"? Pre-registered: report BTC full-history
stats + per-era + above-EMA200 + bear-window fires vs the daily reference.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
from trend_continuation_door import TrendContinuationDoor, DeadSpringDoor
from xasset_spx_port import build_base_sensors, detect_fractal_pivots, _broadcast, HTF_N
from structural_range import build_structural_range
from trend_continuation_door import build_daily_sensors, resample_daily
import trend_continuation_door as tcd

BTC_1H = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
OHLCV = ["open", "high", "low", "close", "volume"]


def resample_4h(df_1h):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df_1h[OHLCV].resample("4h").agg(agg).dropna(subset=["open", "high", "low", "close"])


def reanchor_frame_daily(df_4h, N=HTF_N):
    """4H-exec -> DAILY N-pivot reanchor. Byte-for-byte mirror of reanchor_frame_weekly
    one step down (rule 1W->1D, delta 7d->1d). Causal: pivot visible after p+N closes."""
    rule = "1D"
    o = df_4h["open"].resample(rule, label="left", closed="left").first()
    h = df_4h["high"].resample(rule, label="left", closed="left").max()
    l = df_4h["low"].resample(rule, label="left", closed="left").min()
    c = df_4h["close"].resample(rule, label="left", closed="left").last()
    v = df_4h["volume"].resample(rule, label="left", closed="left").sum()
    dd = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna(
        subset=["open", "high", "low", "close"])
    delta = pd.Timedelta(days=1)
    dd["close_time"] = dd.index + delta
    last = df_4h.index[-1]
    dd = dd[dd["close_time"] <= last + pd.Timedelta(hours=4)]
    piv = detect_fractal_pivots(dd, N)
    sh = _broadcast(piv, df_4h.index, "pivot_high_level", "is_swing_high")
    sl = _broadcast(piv, df_4h.index, "pivot_low_level", "is_swing_low")
    out = df_4h.copy()
    out["swing_low_50"] = sl.to_numpy()
    out["swing_high_50"] = sh.to_numpy()
    return out


def build_4h_sensors(df_4h):
    df = build_base_sensors(df_4h)                       # ema200/atr14 in 4H bars
    reanch = reanchor_frame_daily(df, HTF_N)             # 4H -> daily N=5 reanchor
    sr = build_structural_range(reanch)
    bj = tcd.build_bojan(reanch, sr, tcd.BOJAN_W)
    eye = tcd.compute_eye_features(df[OHLCV].copy())     # 40-bar rolling on 4H bars
    return df, sr, bj, eye


BTC_BEARS = [("2018", "2018-01-01", "2018-12-15"), ("2022", "2021-11-11", "2022-12-31"),
             ("2025-26H1", "2025-01-01", "2026-06-10")]
ERAS = [("TRAIN 18-22", "2018-01-01", "2022-12-31"), ("OOS-A 23-24", "2023-01-01", "2024-12-31"),
        ("OOS-B 25-26", "2025-01-01", "2026-06-10")]


def report(tag, df, sr, bj, eye):
    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label="TC")
    tr, el = res["trades"], strat.entries_log
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in tr: e += t["pnl"]; eq.append(e)
    s = compute_stats(tr, eq, INITIAL_CASH)
    ab = sum(1 for x in el if not x["below_ema200"]); tot = len(el)
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    pf = "inf" if s["PF"] == float("inf") else f"{s['PF']:.2f}"
    print(f"\n== {tag} ==  span {df.index[0].date()}->{df.index[-1].date()} ({yrs:.1f}y)")
    print(f"  n={s['n']}  ({s['n']/yrs:.1f}/yr)  WR={s['WR']*100:.0f}%  PF={pf}  "
          f"PnL=${s['PnL']:+,.0f}  MaxDD={s['MaxDD_pct']:.1f}%  above-EMA200={100*ab/max(tot,1):.0f}%")
    for name, a, b in ERAS:
        sub = [t for t in tr if pd.Timestamp(a) <= t["entry_time"] <= pd.Timestamp(b)]
        if not sub:
            print(f"    {name:<14} n=0"); continue
        eq2 = [INITIAL_CASH]; e2 = INITIAL_CASH
        for t in sub: e2 += t["pnl"]; eq2.append(e2)
        ss = compute_stats(sub, eq2, INITIAL_CASH)
        pf2 = "inf" if ss["PF"] == float("inf") else f"{ss['PF']:.2f}"
        print(f"    {name:<14} n={ss['n']:<4} WR={ss['WR']*100:>3.0f}%  PF={pf2:>5}  PnL=${ss['PnL']:+,.0f}")
    bear = [t for t in tr if any(pd.Timestamp(a) <= t["entry_time"] <= pd.Timestamp(b)
                                 for _, a, b in BTC_BEARS)]
    bp = sum(t["pnl"] for t in bear)
    print(f"    BEAR-window fires: n={len(bear)}  PnL=${bp:+,.0f}")
    return s


def main():
    btc = pd.read_parquet(BTC_1H)
    # reference: validated daily door
    d1 = resample_daily(btc[OHLCV])
    report("DAILY door (validated add.48 reference)", *build_daily_sensors(d1))
    # fractal: 4H door, identical bar-unit params
    d4 = resample_4h(btc)
    report("4H FRACTAL door (same params, one TF down)", *build_4h_sensors(d4))


if __name__ == "__main__":
    main()
