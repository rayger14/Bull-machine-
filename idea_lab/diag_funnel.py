"""PHASE 1 funnel table: reproduce add.55 trade counts + stage-by-stage funnel."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from probe_fractal_4h import (resample_4h, resample_daily, build_4h_sensors,
                              build_daily_sensors, BTC_1H, OHLCV)
from trend_continuation_door import TrendContinuationDoor
from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
from engine.features import eye_state as ES
import unified_archetype_v2 as U

btc = pd.read_parquet(BTC_1H)
d1 = resample_daily(btc[OHLCV]); d4 = resample_4h(btc)

def funnel(tag, df, sr, bj, eye):
    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label="TC")
    tr = res["trades"]
    eq=[INITIAL_CASH]; e=INITIAL_CASH
    for t in tr: e+=t["pnl"]; eq.append(e)
    s = compute_stats(tr, eq, INITIAL_CASH)
    # funnel stages by walking the door logic manually
    st = eye["eye_state"].to_numpy(object); dr = eye["eye_dir"].to_numpy(object)
    ru = eye["range_upper_1d"].to_numpy(float)
    is_cb = np.array([(a==ES.CONFIRMED_BREAK and b=="bull") for a,b in zip(st,dr)])
    cb_enters = int((is_cb & ~np.r_[False, is_cb[:-1]]).sum())
    # bars where a bull CB is within the trailing SOS window (setup 'live')
    sos_live = 0; t1t2 = 0
    for i in range(len(df)):
        lo = max(0, i-U.M2_SOS_WIN)
        has_cb = any(st[c]==ES.CONFIRMED_BREAK and dr[c]=="bull" for c in range(lo,i))
        if has_cb: sos_live += 1
        if dr[i]=="bull" and st[i] in (ES.MODEL_FORMING, ES.IN_RANGE, ES.MANIPULATION) and has_cb:
            t1t2 += 1
    pf = "inf" if s["PF"]==float("inf") else f"{s['PF']:.2f}"
    yrs = (df.index[-1]-df.index[0]).days/365.25
    print(f"\n== {tag} ==")
    print(f"  exec bars                         : {len(df):,}")
    print(f"  eye CONFIRMED_BREAK enter events  : {cb_enters}")
    print(f"  bars with live SOS-window setup   : {sos_live:,}")
    print(f"  bars passing T1(bull)+T2(state)+T3: {t1t2:,}")
    print(f"  ENTERED trades                    : {s['n']}  ({s['n']/yrs:.2f}/yr)")
    print(f"  WR={s['WR']*100:.0f}%  PF={pf}  PnL=${s['PnL']:+,.0f}")
    return s

df1,sr1,bj1,eye1 = build_daily_sensors(d1)
df4,sr4,bj4,eye4 = build_4h_sensors(d4)
funnel("DAILY door (add.55 ref: expect n=9)", df1,sr1,bj1,eye1)
funnel("4H fractal AS-BUILT (add.55: expect n=8)", df4,sr4,bj4,eye4)
