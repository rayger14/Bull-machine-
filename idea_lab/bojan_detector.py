"""
BOJAN DETECTOR  (STUDY ONLY — step-3 re-test on the fixed structural sensors)

WHY THIS EXISTS
---------------
The prior Bojan work "failed" because no real detector existed. wyckoff_audit
addendum 33 proved the only thing carrying the name "Bojan" was a bare
`wick_lower_ratio >= 0.5` size-stub on wick_trap — and that bare wick% fires
DEAD-CENTER of range (median range_position 0.496) and is UNDER-represented at
range lows (9.6% at the bottom quintile vs 20% random). Zero structural enrichment.
The addendum-10 Bojan-gate failure was therefore partly a sensor misfire: there was
no structural extreme to anchor the wick to.

Addendum 37 built that missing extreme: the 1D N=5 structural range (median level-dwell
13.7 days — WI's "weeks"). NOW a real, WI-faithful Bojan detector can be built:
anchor the wick tip to a structural extreme.

WI GRAMMAR HONORED (addendum 8, verbatim source)
------------------------------------------------
  * "Bojan = multi-timeframe precision. M1 = highest wick tags major liquidity /
    MM supply-demand; M2 = same, connected to mid/HTF range highs/lows."
  * Zone CENTERED ON THE WICK TIP (body/open secondary)  -> zone = [wick tip, body edge].
  * INVALIDATION: close beyond the zone -> CONVERTS to an unfinished candle (UNF).
  * GRAMMAR: Bojan LOWS = demand/entry + SL anchors; Bojan HIGHS = TP anchors/short triggers.
  * NO public wick-% rule (WI VERBATIM: "our legacy 70% is OUR invention, not his").
    -> W is a FLAGGED HYPOTHESIS, swept over {0.5, 0.6, 0.7}.

WHAT IS *NOT* BUILT (flagged for WI v2)
---------------------------------------
  * The M1/M2 candle-ordinal rule ("M2 = third candle + M1 = fourth candle") is
    under-specified in the public corpus. This detector builds the CORE
    structural-anchored wick only. M1/M2 sequencing is a v2 item pending WI clarification.
  * TT (take-profit) formula: WI explicitly disclaims a public spec — not built.

CORE DEFINITION (BOJAN-LOW; mirror for HIGH)
--------------------------------------------
A 1H bar i is a BOJAN-LOW candidate iff ALL of:
  1. lower_wick_ratio_i = (min(open,close) - low) / (high - low)  >=  W    (W flagged)
  2. its WICK TIP (the bar low) is at/through a STRUCTURAL EXTREME E:
        low_i  <=  E + PROX_ATR * atr_i          (within PROX_ATR, or pierces)
     where E is the NEAREST of:
        struct_range_low (1D N=5)   |  swing_low_50 (1H)  |  nearest round number
  3. the bar CLOSES BACK ABOVE that extreme:  close_i >= E
        (the "wick tags liquidity, body reclaims" requirement)
Zone = [wick tip = low_i , body low = min(open_i, close_i)].

PERSISTENCE / UNF CONVERSION (WI)
---------------------------------
  * A formed Bojan-low zone stays ACTIVE (a demand level) bar-to-bar.
  * INVALIDATION: the first later bar that CLOSES below the zone bottom (the wick tip)
    invalidates the Bojan -> it becomes an UNF magnet at that level (emit unf_from_bojan_low).
  * bojan_low_active = 1 while >=1 un-invalidated low zone exists.

EMITTED COLUMNS (per 1H bar)
----------------------------
  bojan_low_form / bojan_high_form        1 at the formation bar
  bojan_low_active / bojan_high_active     1 while >=1 zone of that side is un-invalidated
  bojan_low_zone_top / _bottom             edges of the MOST RECENT active low zone (NaN if none)
  bojan_high_zone_top / _bottom            edges of the MOST RECENT active high zone
  bojan_low_anchor / bojan_high_anchor     which extreme the wick tagged {struct/swing/round}
  unf_from_bojan_low / _high               level emitted on invalidation bar (NaN otherwise)
  bojan_low_n_active / bojan_high_n_active  concurrency count

FLAGGED HYPOTHESES (provenance ledger, addendum 35 — ALL OURS, none WI numbers)
------------------------------------------------------------------------------
  W in {0.5, 0.6, 0.7}   PROX_ATR = 0.5   NEAR_ATR = 1.0 (Stage-1 proximity)
  round-number grid (below)   1D N=5 anchor (inherited from addendum 37, also ours)

CAUSALITY
---------
Formation at bar i uses only bar i's own OHLC and already-confirmed (lagged) levels
(struct range from confirmed HTF pivots; swing_50 is 10-bar-lagged; atr_14 causal;
round numbers from the bar's own price). Invalidation is decided at the later bar only.
No future data anywhere. (Verified by the 3-cut no-repaint test in the validator.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---- FLAGGED HYPOTHESES (ours; see addendum 35 provenance ledger) ----
PROX_ATR = 0.5          # wick-tip proximity to the structural extreme
NEAR_ATR = 1.0          # Stage-1: entry within this many ATR of an active zone = "near"
W_SWEEP = (0.5, 0.6, 0.7)


def round_step(price: float) -> float:
    """Psychological round-number grid, scale-aware (FLAGGED, ours).
    <$20k -> $1,000 ; <$100k -> $5,000 ; else $10,000. These are the levels a human
    (WI) actually draws (20k/25k/30k... 100k). Coarse by design to avoid over-firing."""
    if price < 20_000:
        return 1_000.0
    if price < 100_000:
        return 5_000.0
    return 10_000.0


def _nearest_round(price: float) -> float:
    s = round_step(price)
    return round(price / s) * s


def wick_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """lower_wick_ratio / upper_wick_ratio = wick length / full bar range (0..1)."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    rng = np.where((h - l) > 0, h - l, np.nan)
    body_low = np.minimum(o, c)
    body_high = np.maximum(o, c)
    lower = (body_low - l) / rng
    upper = (h - body_high) / rng
    return pd.DataFrame({"lower_wick_ratio": lower, "upper_wick_ratio": upper}, index=df.index)


def build_bojan(
    df: pd.DataFrame,
    sr: pd.DataFrame,
    W: float,
    prox_atr: float = PROX_ATR,
) -> pd.DataFrame:
    """Causal forward pass. Returns per-1H-bar Bojan columns + attaches a zone roster
    as .attrs['low_zones'] / .attrs['high_zones'] for the Stage-1 proximity test.

    Required df columns: open, high, low, close, swing_low_50, swing_high_50, atr_14.
    `sr` must carry struct_range_low / struct_range_high (from build_structural_range on
    the 1D N=5 re-anchored frame, per addendum 37).
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr_14"].to_numpy(float)
    swl = df["swing_low_50"].to_numpy(float)
    swh = df["swing_high_50"].to_numpy(float)
    srl = sr["struct_range_low"].to_numpy(float)
    srh = sr["struct_range_high"].to_numpy(float)
    wr = wick_ratios(df)
    lwr = wr["lower_wick_ratio"].to_numpy(float)
    uwr = wr["upper_wick_ratio"].to_numpy(float)
    n = len(df)
    ts = df.index

    low_form = np.zeros(n, np.int8)
    high_form = np.zeros(n, np.int8)
    low_active = np.zeros(n, np.int8)
    high_active = np.zeros(n, np.int8)
    low_zt = np.full(n, np.nan); low_zb = np.full(n, np.nan)
    high_zt = np.full(n, np.nan); high_zb = np.full(n, np.nan)
    low_anchor = np.array([""] * n, dtype=object)
    high_anchor = np.array([""] * n, dtype=object)
    unf_low = np.full(n, np.nan)
    unf_high = np.full(n, np.nan)
    low_nact = np.zeros(n, np.int32)
    high_nact = np.zeros(n, np.int32)

    # active zone rosters: list of dicts {bottom, top, anchor, formed_i}
    low_zones: list[dict] = []
    high_zones: list[dict] = []
    low_zones_all: list[dict] = []   # full roster w/ invalidation for Stage-1
    high_zones_all: list[dict] = []

    def low_extreme_candidates(i):
        cands = []
        if not np.isnan(srl[i]):
            cands.append(("struct", srl[i]))
        if not np.isnan(swl[i]):
            cands.append(("swing", swl[i]))
        cands.append(("round", _nearest_round(c[i])))
        return cands

    def high_extreme_candidates(i):
        cands = []
        if not np.isnan(srh[i]):
            cands.append(("struct", srh[i]))
        if not np.isnan(swh[i]):
            cands.append(("swing", swh[i]))
        cands.append(("round", _nearest_round(c[i])))
        return cands

    for i in range(n):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            # still process invalidations below with prior state, but no new formation
            a = np.nan

        # ---- INVALIDATION FIRST (a later bar closing beyond a zone) ----
        if not np.isnan(c[i]):
            still_low = []
            for z in low_zones:
                if c[i] < z["bottom"]:                # close below wick tip -> UNF
                    z["invalidated_i"] = i
                    z["unf_level"] = z["bottom"]
                    if np.isnan(unf_low[i]):
                        unf_low[i] = z["bottom"]
                else:
                    still_low.append(z)
            low_zones = still_low
            still_high = []
            for z in high_zones:
                if c[i] > z["top"]:                   # close above wick tip -> UNF
                    z["invalidated_i"] = i
                    z["unf_level"] = z["top"]
                    if np.isnan(unf_high[i]):
                        unf_high[i] = z["top"]
                else:
                    still_high.append(z)
            high_zones = still_high

        # ---- FORMATION (bar i's own completed candle) ----
        if not np.isnan(a):
            # BOJAN-LOW
            if lwr[i] >= W:
                best = None
                for name, E in low_extreme_candidates(i):
                    if E is None or np.isnan(E):
                        continue
                    tags = l[i] <= E + prox_atr * a          # tip at/through extreme
                    reclaims = c[i] >= E                      # body closes back above
                    if tags and reclaims:
                        d = abs(l[i] - E)
                        if best is None or d < best[2]:
                            best = (name, E, d)
                if best is not None:
                    name, E, _ = best
                    bottom = l[i]
                    top = min(o[i], c[i])
                    if top <= bottom:
                        top = bottom + 1e-9
                    z = {"bottom": bottom, "top": top, "anchor": name,
                         "formed_i": i, "formed_ts": ts[i], "invalidated_i": None,
                         "unf_level": np.nan, "side": "low"}
                    low_zones.append(z)
                    low_zones_all.append(z)
                    low_form[i] = 1
                    low_anchor[i] = name

            # BOJAN-HIGH
            if uwr[i] >= W:
                best = None
                for name, E in high_extreme_candidates(i):
                    if E is None or np.isnan(E):
                        continue
                    tags = h[i] >= E - prox_atr * a
                    reclaims = c[i] <= E
                    if tags and reclaims:
                        d = abs(h[i] - E)
                        if best is None or d < best[2]:
                            best = (name, E, d)
                if best is not None:
                    name, E, _ = best
                    top = h[i]
                    bottom = max(o[i], c[i])
                    if bottom >= top:
                        bottom = top - 1e-9
                    z = {"bottom": bottom, "top": top, "anchor": name,
                         "formed_i": i, "formed_ts": ts[i], "invalidated_i": None,
                         "unf_level": np.nan, "side": "high"}
                    high_zones.append(z)
                    high_zones_all.append(z)
                    high_form[i] = 1
                    high_anchor[i] = name

        # ---- EMIT active state ----
        low_nact[i] = len(low_zones)
        high_nact[i] = len(high_zones)
        if low_zones:
            low_active[i] = 1
            z = low_zones[-1]                          # most recent active
            low_zt[i] = z["top"]; low_zb[i] = z["bottom"]
        if high_zones:
            high_active[i] = 1
            z = high_zones[-1]
            high_zt[i] = z["top"]; high_zb[i] = z["bottom"]

    out = pd.DataFrame(
        {
            "bojan_low_form": low_form,
            "bojan_high_form": high_form,
            "bojan_low_active": low_active,
            "bojan_high_active": high_active,
            "bojan_low_zone_top": low_zt,
            "bojan_low_zone_bottom": low_zb,
            "bojan_high_zone_top": high_zt,
            "bojan_high_zone_bottom": high_zb,
            "bojan_low_anchor": low_anchor,
            "bojan_high_anchor": high_anchor,
            "unf_from_bojan_low": unf_low,
            "unf_from_bojan_high": unf_high,
            "bojan_low_n_active": low_nact,
            "bojan_high_n_active": high_nact,
        },
        index=df.index,
    )
    # convert index positions to timestamps for the Stage-1 roster
    for z in low_zones_all + high_zones_all:
        z["formed_ts"] = ts[z["formed_i"]]
        z["invalidated_ts"] = ts[z["invalidated_i"]] if z["invalidated_i"] is not None else None
    out.attrs["low_zones"] = low_zones_all
    out.attrs["high_zones"] = high_zones_all
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from structural_range import build_structural_range
    from htf_pivots import reanchor_frame

    STORE = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
    df = pd.read_parquet(STORE)
    dfa = reanchor_frame(df, "1D", 5)          # addendum-37 anchor
    sr = build_structural_range(dfa)
    for W in W_SWEEP:
        bj = build_bojan(dfa, sr, W)
        nlow = int(bj["bojan_low_form"].sum())
        nhigh = int(bj["bojan_high_form"].sum())
        print(f"W={W}: {nlow} bojan-low forms, {nhigh} bojan-high forms | "
              f"low_active {bj['bojan_low_active'].mean()*100:.1f}% of bars | "
              f"median concurrency {int(bj['bojan_low_n_active'][bj['bojan_low_active']==1].median() or 0)}")
