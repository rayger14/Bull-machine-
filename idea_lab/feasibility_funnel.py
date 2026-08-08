"""
FEASIBILITY FUNNEL for the UNIFIED ARCHETYPE (study only).

Counts how many bars survive each conjunctive gate of the pre-registered spec,
and how many TRIGGER bars (candidate entries) exist per year. This is the
trap-reset-lesson feasibility check: if pooled candidate fires < 25, flag
underpowered but still run.

Gates (all causal, computed on the 1D N=5 re-anchored structural range):
  R1 regime permission : (close > ema_200 OR struct_state != broken_down) AND stables_rot_rising==0
  R2 structural context: struct_state == active AND struct_range_pos <= 0.4 (discount half)
  R3 trigger           : struct_sweep_low within trailing K=3 bars (incl this bar)
  TIME (conviction)    : fib_time_confluence > 0 at the bar
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from htf_pivots import reanchor_frame
from structural_range import build_structural_range, ACTIVE, BROKEN_DOWN

STORE = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
K_TRIGGER = 3


def sweep_recent(sweep_low: np.ndarray, k: int) -> np.ndarray:
    n = len(sweep_low); out = np.zeros(n, dtype=np.int8); last = -10_000
    for i in range(n):
        if sweep_low[i] == 1:
            last = i
        if i - last < k:
            out[i] = 1
    return out


def main():
    df = pd.read_parquet(STORE)
    reanch = reanchor_frame(df, "1D", 5)
    sr = build_structural_range(reanch)

    close = df["close"].to_numpy(float)
    ema200 = df["ema_200"].to_numpy(float)
    stables = np.nan_to_num(df["stables_rot_rising"].to_numpy(float))
    fib = np.nan_to_num(df["fib_time_confluence"].to_numpy(float))
    state = sr["struct_range_state"].to_numpy(dtype=object)
    pos = sr["struct_range_pos"].to_numpy(float)
    sweep_lo = sr["struct_sweep_low"].to_numpy(np.int8)

    active = state == ACTIVE
    not_markdown = (close > ema200) | (state != BROKEN_DOWN)
    R1 = not_markdown & (stables == 0)
    R2 = active & (pos <= 0.4)
    trig = sweep_recent(sweep_lo, K_TRIGGER) == 1
    R3 = R2 & trig
    fire = R1 & R3                          # full conjunctive candidate bar
    fire_time = fire & (fib > 0)

    n = len(df)
    print(f"total bars: {n}  ({df.index[0]} -> {df.index[-1]})")
    print(f"active range bars           : {active.sum():>6} ({100*active.mean():.1f}%)")
    print(f"R1 regime permission        : {R1.sum():>6} ({100*R1.mean():.1f}%)")
    print(f"R2 discount-in-active-range : {R2.sum():>6} ({100*R2.mean():.1f}%)")
    print(f"struct_sweep_low bars (raw) : {int(sweep_lo.sum()):>6}")
    print(f"R3 (R2 & sweep<=K3)         : {R3.sum():>6} ({100*R3.mean():.1f}%)")
    print(f"FIRE (R1 & R3) bar-level    : {fire.sum():>6}")
    print(f"FIRE & fib_time>0           : {fire_time.sum():>6}")

    # candidate ENTRY EVENTS: collapse consecutive fire bars into episodes (a
    # trigger cluster = one candidate). Count events per year.
    fire_i = np.where(fire)[0]
    events = []
    prev = -100
    for i in fire_i:
        if i - prev > 6:                    # new episode if >6 bars since last fire
            events.append(i)
        prev = i
    ev_idx = df.index[events]
    yrs = pd.Series(ev_idx.year)
    print(f"\ncandidate ENTRY EVENTS (episodes, >6-bar gap) : {len(events)}")
    print("per-year event count:")
    print(yrs.value_counts().sort_index().to_string())
    pooled = len(events)
    print(f"\nPOOLED candidate events = {pooled}  "
          f"{'[UNDERPOWERED <25 -> directional only]' if pooled < 25 else '[n>=25 OK]'}")
    fibpos = sum(1 for i in events if fib[i] > 0)
    print(f"  of which fib_time_confluence>0 at event bar: {fibpos} ({100*fibpos/max(pooled,1):.0f}%)")


if __name__ == "__main__":
    main()
