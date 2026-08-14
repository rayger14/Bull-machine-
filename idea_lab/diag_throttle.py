"""
PHASE 1 THROTTLE DIAGNOSIS (study only, no production touch).
Proves WHERE add.55's 4H fractal door was throttled by instrumenting the eye
break-generator at 4H vs what a TRUE 4H eye would emit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from probe_fractal_4h import (resample_4h, resample_daily, build_4h_sensors,
                              build_daily_sensors, BTC_1H, OHLCV)
import trend_continuation_door as tcd
from engine.features import eye_state as ES

btc = pd.read_parquet(BTC_1H)

# ---- 1) The frames as the probe builds them ----
d1 = resample_daily(btc[OHLCV])
d4 = resample_4h(btc)
print("FRAME SIZES")
print(f"  1H store bars      : {len(btc):,}")
print(f"  daily exec frame   : {len(d1):,} bars")
print(f"  4H exec frame      : {len(d4):,} bars")

# ---- 2) What does compute_eye_features actually run on for each? ----
# It calls ES._resample_1d internally. Show the internal daily grid it collapses to.
daily_from_d1 = ES._resample_1d(d1)     # daily door: daily->'1D' (idempotent)
daily_from_d4 = ES._resample_1d(d4)     # 4H fractal: 4H->'1D'  (COLLAPSE?)
print("\nEYE INTERNAL _resample_1d('1D') GRID  (the frame the break-machine runs on)")
print(f"  daily door  -> {len(daily_from_d1):,} daily bars  (state-machine steps)")
print(f"  4H fractal  -> {len(daily_from_d4):,} daily bars  (state-machine steps)")
print(f"  4H exec bars fed in were {len(d4):,}; eye collapsed them to {len(daily_from_d4):,}.")

# ---- 3) Build sensors both ways and count eye CONFIRMED_BREAK events + cadence ----
def eye_break_census(tag, eye, exec_index):
    st = eye["eye_state"].to_numpy(object)
    dr = eye["eye_dir"].to_numpy(object)
    # CONFIRMED_BREAK bull ENTER events (transition into CB from non-CB), on the exec grid
    is_cb = np.array([(s == ES.CONFIRMED_BREAK and d == "bull") for s, d in zip(st, dr)])
    enters = np.where(is_cb & ~np.r_[False, is_cb[:-1]])[0]
    # unique underlying eye values (broadcast repeats each daily state across exec bars)
    n_unique_states = eye[["eye_state","eye_dir","range_upper_1d"]].drop_duplicates().shape[0]
    print(f"\n[{tag}] exec bars={len(exec_index):,}")
    print(f"  unique (state,dir,range_upper) rows in eye : {n_unique_states:,}"
          f"   <- distinct underlying HTF bars behind the broadcast")
    print(f"  bull CONFIRMED_BREAK bars (exec grid)      : {int(is_cb.sum()):,}")
    print(f"  bull CONFIRMED_BREAK *enter events*        : {len(enters)}")
    if len(enters) >= 2:
        gaps = np.diff([exec_index[e] for e in enters])
        med = pd.to_timedelta(np.median(gaps))
        print(f"  median spacing between CB enters           : {med}")
    return len(enters)

df1, sr1, bj1, eye1 = build_daily_sensors(d1)
df4, sr4, bj4, eye4 = build_4h_sensors(d4)
n1 = eye_break_census("DAILY door eye", eye1, df1.index)
n4 = eye_break_census("4H fractal eye", eye4, df4.index)

# ---- 4) What would a TRUE 4H eye see? raw 4H body-close breaks of the 40-bar ceiling ----
def raw_4h_breaks(d4, N=ES.N_RANGE_1D, accept=ES.ACCEPT_CONSEC):
    body_high = d4[["open","close"]].max(axis=1)
    upper = body_high.rolling(N, min_periods=N).max().shift(1)
    c = d4["close"]
    close_above = (c > upper).to_numpy()
    # count runs reaching >=accept consecutive closes above (a CONFIRMED break at 4H scale)
    run = 0; confirmed = 0; raw_single = 0
    for x in close_above:
        if x:
            run += 1
            if run == 1: raw_single += 1
            if run == accept: confirmed += 1
        else:
            run = 0
    return raw_single, confirmed

raw_single, raw_conf = raw_4h_breaks(d4)
print("\nWHAT A TRUE 4H EYE WOULD EMIT (40 *4H-bar* ceiling, body-close, accept=2)")
print(f"  raw single-close 4H breaks of ceiling      : {raw_single}")
print(f"  CONFIRMED (>=2 consec 4H closes) 4H breaks  : {raw_conf}")
print(f"\nTHROTTLE RATIO: true-4H confirmed breaks {raw_conf}  vs  actual eye CB-enters {n4}"
      f"  = {raw_conf/max(n4,1):.1f}x suppressed")
