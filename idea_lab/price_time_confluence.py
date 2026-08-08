"""
PRICE x TIME CONFLUENCE  (STUDY ONLY — step-3 re-test on the fixed sensors)

wyckoff_audit addendum 36 recovered the FOUNDING doctrine: a good long entry needs the
right PLACE **and** the right TIME, combined MULTIPLICATIVELY (conjunctive / weakest-link),
NOT additively (the Lesson-#54 fusion anti-pattern). The founding
hidden_fibs.detect_price_time_confluence used price_strength x time_strength x 1.5.

This module rebuilds that founding shape on the NOW-FIXED sensors:
  - PLACE  = idea_lab structural range (add.37: 1D N=5 pivots -> weeks-long drawn box,
             the validated WHERE sensor). place_strength is HIGH in the discount half of
             an ACTIVE range, with a recent-sweep-low bonus.
  - TIME   = the store's fib_time_confluence (add.38: the deployed temporal layer,
             validated NET-POSITIVE 3/3). Bounded [0,1], real spread.
  - CONFLUENCE = place_strength x time_strength  (multiplicative, the founding shape).

-------------------------------------------------------------------------------
PRE-REGISTERED FORMULAS (fixed before any validation; magnitudes are OURS/flagged, add.35)
-------------------------------------------------------------------------------
PLACE (bounded [0,1]):
    if struct_range_state == 'active':
        discount   = clip(1 - struct_range_pos, 0, 1)      # 1 at range_low, 0 at range_high
        sweep_recent = 1 if struct_sweep_low==1 within trailing K bars (inclusive) else 0
        place_strength = clip(discount + SWEEP_BONUS * sweep_recent, 0, 1)
    else:
        place_strength = 0.0
  FLAGGED PARAMS (ours, no WI number): K = 6 bars, SWEEP_BONUS = 0.25.

TIME (bounded [0,1], causal store column):
    time_strength = fib_time_confluence
  WHY fib_time_confluence over temporal_confluence_score:
    (1) it is the exact feature of the deployed order_block_retest gate validated
        PROTECTIVE 3/3 in add.38 (temporal_confluence_score's gate is only on
        retest_cluster, MIXED 2/3);
    (2) it has genuine dynamic range [0,1] (mean ~0.20, ~65% nonzero) whereas
        temporal_confluence_score is a narrow ~[0.41,0.69] band that cannot tier;
    (3) already bounded [0,1] -> correct scale for a multiplicative combo with place.

CONFLUENCE:
    PRIMARY   confluence_mult = place_strength * time_strength
    COMPARE   confluence_geo  = sqrt(place_strength * time_strength)   (founding geo-mean)

CAUSALITY: place uses only the causal structural_range machine (confirmed 1D N=5 pivots,
closed bars); time is a causal store column; the recent-sweep window looks only BACKWARD.
No future data anywhere.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from htf_pivots import reanchor_frame
from structural_range import build_structural_range, ACTIVE

# ---- FLAGGED PARAMS (OURS, add.35) -------------------------------------------
HTF_FREQ = "1D"          # add.37 verdict: 1D N=5 is the structural-range anchor
HTF_N = 5
SWEEP_K = 6              # trailing bars a struct_sweep_low still counts as "recent"
SWEEP_BONUS = 0.25       # additive place bonus for a recent sweep-low (pre-clip)
TIME_COL = "fib_time_confluence"


def _sweep_recent(sweep_low: np.ndarray, k: int) -> np.ndarray:
    """1 if struct_sweep_low==1 anywhere in trailing window [i-k+1 .. i] (causal)."""
    n = len(sweep_low)
    out = np.zeros(n, dtype=np.int8)
    run = 0  # bars since last sweep (inclusive counter)
    last = -10_000
    for i in range(n):
        if sweep_low[i] == 1:
            last = i
        if i - last < k:
            out[i] = 1
    return out


def build_price_time_confluence(store_path: str) -> pd.DataFrame:
    """Compute PLACE/TIME/CONFLUENCE on the full 1H grid. Returns a DataFrame indexed by
    the 1H timestamp with the sensor columns. Study-only, causal, no production write."""
    df = pd.read_parquet(store_path)

    # PLACE: re-anchor to 1D N=5 HTF pivots, then run the UNCHANGED structural-range machine
    reanch = reanchor_frame(df, HTF_FREQ, HTF_N)
    sr = build_structural_range(reanch)  # pre-registered defaults (buffer 0, confirm 1)

    state = sr["struct_range_state"].to_numpy(dtype=object)
    pos = sr["struct_range_pos"].to_numpy(dtype=float)
    sweep_lo = sr["struct_sweep_low"].to_numpy(dtype=np.int8)

    active = state == ACTIVE
    discount = np.clip(1.0 - pos, 0.0, 1.0)
    discount = np.where(active & ~np.isnan(pos), discount, 0.0)
    sweep_recent = _sweep_recent(sweep_lo, SWEEP_K)
    place = np.clip(discount + SWEEP_BONUS * sweep_recent, 0.0, 1.0)
    place = np.where(active, place, 0.0)  # only ever nonzero inside an active range

    # TIME: causal store column, already bounded [0,1]
    time_s = df[TIME_COL].to_numpy(dtype=float)
    time_s = np.nan_to_num(np.clip(time_s, 0.0, 1.0), nan=0.0)

    conf_mult = place * time_s
    conf_geo = np.sqrt(np.clip(place * time_s, 0.0, None))

    out = pd.DataFrame(
        {
            "struct_range_state": state,
            "struct_range_pos": pos,
            "struct_sweep_low": sweep_lo,
            "sweep_recent_k": sweep_recent,
            "place_strength": place,
            "time_strength": time_s,
            "confluence_mult": conf_mult,
            "confluence_geo": conf_geo,
        },
        index=df.index,
    )
    return out


if __name__ == "__main__":
    store = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
    s = build_price_time_confluence(store)
    print("rows", len(s))
    for c in ("place_strength", "time_strength", "confluence_mult", "confluence_geo"):
        col = s[c]
        print(f"{c:16s} mean={col.mean():.4f} nonzero={100*(col>0).mean():.1f}% p75={col.quantile(.75):.4f} p90={col.quantile(.9):.4f} max={col.max():.4f}")
    print("active frac", f"{(s['struct_range_state']=='active').mean()*100:.1f}%")
