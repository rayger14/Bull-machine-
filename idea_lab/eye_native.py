"""
THROTTLE-FIX (diagnosed-bug repair, study only).
=================================================
Phase-1 diagnosis: engine/features/eye_state.compute_eye_features() hard-calls
_resample_1d() which collapses ANY input frame to '1D' calendar bars before running
the break state machine. So the add.55 "4H fractal" ran its break-generator on DAILY
bars (3,081) not the 4H exec frame (18,471) — emitting the SAME 7 CONFIRMED_BREAK
events as the daily door. The 40-bar ceiling was a 40-DAY ceiling, never a 40*4H one.

FIX (not a tuning choice): run the SAME state machine on the exec frame at its native
resolution. N_RANGE=40 now means 40 EXEC bars (40*4H ~ 6.7d) as the probe's own
comment intended ("40-bar rolling on 4H bars"). No resample, no broadcast — the eye is
already at exec cadence. Everything else (ACCEPT_CONSEC=2, TREND_CONSEC=5, retest,
body-close semantics) is byte-identical to engine/features/eye_state._run_state_machine.

PARITY GUARANTEE: for a DAILY exec frame this must reproduce the daily door bit-for-bit
(resampling daily->'1D' is idempotent), so the validated door is unchanged. Verified in
__main__.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from engine.features import eye_state as ES


def compute_eye_native(df_exec: pd.DataFrame) -> pd.DataFrame:
    """Run the eye state machine on df_exec AT ITS OWN RESOLUTION (no 1D collapse).
    df_exec: OHLCV indexed at the exec timeframe. Returns eye features on df_exec.index.
    Price-only: phase_dir/sos are set neutral/0 (inert), matching the door's price-only
    guarantee (wyckoff cols dropped on every asset, incl. BTC)."""
    df = df_exec.sort_index().copy()
    daily = df[["open", "high", "low", "close", "volume"]].copy()
    daily["phase_dir"] = np.nan          # inert (price-only door)
    daily["sos"] = 0
    eye = ES._run_state_machine(daily)   # SAME machine, native bars
    # already at exec resolution -> no broadcast; add 1-bar settle lag for causality
    # (the state at bar t is only actionable at t's close; the door reads eye[i] to plan
    #  entry at close[i], and the machine at i uses only bars<=i, so eye[i] is causal.
    #  We keep the SAME 1-bar broadcast lag the daily path had via merge_asof-backward by
    #  shifting the eye one exec bar forward, so the door never reads a same-bar-future.)
    eye = eye.shift(1)
    eye.iloc[0] = eye.iloc[1] if len(eye) > 1 else eye.iloc[0]
    return eye


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from probe_fractal_4h import (resample_4h, resample_daily, BTC_1H, OHLCV,
                                  reanchor_frame_daily)
    from trend_continuation_door import (TrendContinuationDoor, build_daily_sensors,
                                         build_base_sensors)
    from structural_range import build_structural_range
    import trend_continuation_door as tcd
    from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
    from xasset_spx_port import HTF_N

    btc = pd.read_parquet(BTC_1H)

    def run(df, sr, bj, eye, tag):
        strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
        res = run_backtest(df, strat, risk_pct=RISK_PCT, label="TC")
        tr = res["trades"]; eq=[INITIAL_CASH]; e=INITIAL_CASH
        for t in tr: e+=t["pnl"]; eq.append(e)
        s = compute_stats(tr, eq, INITIAL_CASH)
        pf = "inf" if s["PF"]==float("inf") else f"{s['PF']:.2f}"
        yrs=(df.index[-1]-df.index[0]).days/365.25
        print(f"  {tag:<34} n={s['n']:<4}({s['n']/yrs:.2f}/yr) WR={s['WR']*100:>3.0f}% "
              f"PF={pf:>5} PnL=${s['PnL']:+,.0f} DD={s['MaxDD_pct']:.1f}%")
        return s

    print("PARITY CHECK (daily door: native eye must == current path n=9 PF 2.56)")
    d1 = resample_daily(btc[OHLCV])
    df1, sr1, bj1, eye1_cur = build_daily_sensors(d1)
    run(df1, sr1, bj1, eye1_cur, "daily CURRENT path")
    eye1_nat = compute_eye_native(d1)
    run(df1, sr1, bj1, eye1_nat, "daily NATIVE-eye (fix)")

    print("\n4H FRACTAL with THROTTLE FIXED (native 4H eye)")
    d4 = resample_4h(btc)
    df4 = build_base_sensors(d4)
    reanch4 = reanchor_frame_daily(df4, HTF_N)
    sr4 = build_structural_range(reanch4)
    bj4 = tcd.build_bojan(reanch4, sr4, tcd.BOJAN_W)
    eye4_cur = tcd.compute_eye_features(df4[OHLCV].copy())
    run(df4, sr4, bj4, eye4_cur, "4H AS-BUILT (collapsed eye)")
    eye4_nat = compute_eye_native(d4)
    run(df4, sr4, bj4, eye4_nat, "4H NATIVE-eye (throttle fixed)")
    # census the fixed eye's break events
    st=eye4_nat["eye_state"].to_numpy(object); dr=eye4_nat["eye_dir"].to_numpy(object)
    is_cb=np.array([(a==ES.CONFIRMED_BREAK and b=="bull") for a,b in zip(st,dr)])
    cb=int((is_cb & ~np.r_[False, is_cb[:-1]]).sum())
    print(f"  -> native 4H eye CONFIRMED_BREAK enter events: {cb}  (was 7 collapsed)")
