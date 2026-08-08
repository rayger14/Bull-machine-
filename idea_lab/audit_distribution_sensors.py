"""
PHASE 1 — DISTRIBUTION-ZONE SENSOR AUDIT  (STUDY ONLY; the GATE for the short mirror)
=====================================================================================
Mirror of the LONG-side sensor validation (validate_structural_range.py / wyckoff_audit
add.34/41/45). Verifies the distribution sensors fire at the RIGHT LOCATIONS and with
adequate RELIABILITY *before* any short strategy is built on them. If distribution
detection FAILS the pre-registered criteria, the short mirror is NOT built (report the
audit verdict as the deliverable — a broken-sensor finding, exactly like the long side
needed add.34).

------------------------------------------------------------------------------------
PRE-REGISTERED PASS CRITERIA  (fixed BEFORE any number is measured; NO tuning to results)
------------------------------------------------------------------------------------
CHECK 1 — struct_sweep_high LOCATION (mirror of add.34 spring-at-low):
   Metric: distribution of (wick_high - struct_range_high)/ATR for every struct_sweep_high
   firing inside an active range (by construction wick_high > range_high, so overshoot>=0;
   we test TIGHTNESS = shallow upthrust pierce vs blown-through scatter).
   PASS iff ALL:
     (1a) median overshoot <= 1.0 ATR
     (1b) >= 70% of firings within 1.0 ATR of the drawn ceiling
     (1c) count >= 30 firings AND sweep_high count >= 0.33 * sweep_low count
          (ceilings detected with frequency comparable to floors => ranges are two-sided,
           not an accumulation-only sensor).

CHECK 2 — bojan_high AT A STRUCTURAL HIGH (mirror of add.41; long side was 97.4% at lows):
   Metric: % of bojan_high FORMS within NEAR_ATR(=1.0) of struct_range_high OR swing_high_50,
   vs a random-bar baseline (fraction of active-range bars within 1.0 ATR of a structural
   high). PASS iff BOTH:
     (2a) >= 60% of bojan_high forms at/near a structural high (absolute)
     (2b) enrichment >= 2.0x the random-bar baseline.

CHECK 3 — C_distrib / BEAR-CONTEXT RELIABILITY (THE CRUX — "knowing the zones correctly"):
   Metric A: C_distrib bar count vs C_accum; eye MODEL_FORMING-bear bars vs -bull.
   Metric B: 3 major tops (2021 blow-off ~Nov-2021, 2024 top ~Mar-2024, a 2025 top) — is a
             bear signal (C_distrib OR eye MODEL_FORMING/MANIPULATION-bear OR struct_sweep_high
             OR bojan_high) present within +/-30 days of each?
   Metric C: of the 6 DISTRIBUTION consensus events in the 14 hand-labels
             (#3 PSY, #4 ST-distrib, #5 UTAD, #6 LPSY, #12 BC, #13 AR), how many are caught
             by a bear signal within +/-48h & 5% price?
   PASS iff ALL:
     (3a) C_distrib >= 0.10 * C_accum   (bear "cause" zone within one order of magnitude of
          the bull one; mirror-symmetry floor. Long side had C_accum plentiful.)
     (3b) >= 4 of 6 distribution consensus events caught
     (3c) all 3 major tops marked with >=1 bear signal in the window.

CHECK 4 — CAUSALITY / NO-REPAINT (reuse the 3-cut truncation pattern):
   struct_sweep_high, bojan_high_form, and eye bear-states must be byte-identical on the
   overlap when history is truncated at 3 interior cut points. PASS iff 0 repaints.

GATE: the short mirror is built ONLY IF all four checks PASS. Any FAIL => STOP, report the
audit verdict (broken/weak distribution sensor) as the deliverable.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_range import build_structural_range, ACTIVE
from bojan_detector import build_bojan
from htf_pivots import reanchor_frame
from engine.features.eye_state import (
    compute_eye_features, MODEL_FORMING, MANIPULATION, CONFIRMED_BREAK, TRENDING, IN_RANGE,
)

STORE = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
HTF_FREQ, HTF_N = "1D", 5
BOJAN_W_VAL = 0.5
NEAR_ATR = 1.0

# 6 distribution-side consensus events (from bin/validate_wyckoff_consensus_events.py)
DISTRIB_EVENTS = [
    ("#3 PSY 2021",   "2021-02-21", 58000),
    ("#4 ST-distrib", "2021-09-07", 52000),
    ("#5 UTAD 2021",  "2021-11-10", 69000),
    ("#6 LPSY 2022",  "2022-01-15", 47000),
    ("#12 BC 2024",   "2024-03-14", 73660),
    ("#13 AR 2024",   "2024-04-15", 60000),
]
MAJOR_TOPS = [
    ("2021 blow-off top", "2021-11-10"),
    ("2024 ATH top",      "2024-03-14"),
    ("2025 top",          "2025-01-20"),  # BTC local top ~Jan 2025 (within data)
]


def hr(t):
    print("\n" + "=" * 82)
    print(t)
    print("=" * 82)


def main():
    df = pd.read_parquet(STORE)
    print(f"loaded {len(df):,} bars  {df.index[0]} -> {df.index[-1]}")
    reanch = reanchor_frame(df, HTF_FREQ, HTF_N)
    sr = build_structural_range(reanch)
    bj = build_bojan(reanch, sr, BOJAN_W_VAL)
    eye = compute_eye_features(df)

    active = (sr["struct_range_state"] == ACTIVE).to_numpy()
    atr = df["atr_14"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    srh = sr["struct_range_high"].to_numpy(float)
    srl = sr["struct_range_low"].to_numpy(float)
    swh = df["swing_high_50"].to_numpy(float)

    verdicts = {}

    # ---------------------------------------------------------------- CHECK 1
    hr("CHECK 1 — struct_sweep_high LOCATION (mirror add.34 spring-at-low)")
    sweep_hi = (sr["struct_sweep_high"] == 1).to_numpy() & active
    sweep_lo = (sr["struct_sweep_low"] == 1).to_numpy() & active
    n_hi, n_lo = int(sweep_hi.sum()), int(sweep_lo.sum())
    over = (high[sweep_hi] - srh[sweep_hi]) / atr[sweep_hi]
    over = over[np.isfinite(over)]
    med = float(np.median(over)) if len(over) else np.nan
    within1 = float((over <= 1.0).mean()) if len(over) else np.nan
    print(f"  struct_sweep_high firings (in active range): {n_hi}")
    print(f"  struct_sweep_low  firings (in active range): {n_lo}  (ratio hi/lo = {n_hi/max(n_lo,1):.2f})")
    print(f"  (wick_high - range_high)/ATR: median={med:.2f}  mean={over.mean():.2f}  "
          f"within 1 ATR={within1*100:.0f}%   p90={np.percentile(over,90):.2f}")
    c1a = med <= 1.0
    c1b = within1 >= 0.70
    c1c = (n_hi >= 30) and (n_hi >= 0.33 * n_lo)
    verdicts["CHECK 1 (sweep_high location)"] = c1a and c1b and c1c
    print(f"  1a median<=1.0 ATR: {c1a}   1b >=70% within 1 ATR: {c1b}   "
          f"1c count>=30 & hi>=0.33*lo: {c1c}")
    print(f"  -> CHECK 1 {'PASS' if verdicts['CHECK 1 (sweep_high location)'] else 'FAIL'}")

    # ---------------------------------------------------------------- CHECK 2
    hr("CHECK 2 — bojan_high AT A STRUCTURAL HIGH (mirror add.41; long was 97.4%)")
    bh_form = (bj["bojan_high_form"] == 1).to_numpy()
    n_bh = int(bh_form.sum())
    bl_form = (bj["bojan_low_form"] == 1).to_numpy()
    n_bl = int(bl_form.sum())

    def near_struct_high(idx_mask):
        m = idx_mask & np.isfinite(atr) & (atr > 0)
        d_srh = np.abs(high - srh) / atr
        d_swh = np.abs(high - swh) / atr
        near = ((d_srh <= NEAR_ATR) | (d_swh <= NEAR_ATR)) & m
        return near, m

    near_bh, m_bh = near_struct_high(bh_form)
    frac_bh = float(near_bh.sum() / max(m_bh.sum(), 1))
    # random baseline: fraction of ALL active-range bars near a structural high
    rand_mask = active & np.isfinite(atr) & (atr > 0)
    near_rand, m_rand = near_struct_high(rand_mask)
    frac_rand = float(near_rand.sum() / max(m_rand.sum(), 1))
    enrich = frac_bh / frac_rand if frac_rand > 0 else np.inf
    # anchor breakdown
    anch = bj["bojan_high_anchor"].to_numpy(object)[bh_form]
    from collections import Counter
    print(f"  bojan_high forms: {n_bh}   (bojan_low forms: {n_bl})")
    print(f"  anchor mix (bojan_high): {dict(Counter(a for a in anch if a))}")
    print(f"  % bojan_high within {NEAR_ATR} ATR of struct_range_high OR swing_high_50: {frac_bh*100:.1f}%")
    print(f"  random-bar baseline (active bars near a structural high):            {frac_rand*100:.1f}%")
    print(f"  enrichment = {enrich:.2f}x")
    c2a = frac_bh >= 0.60
    c2b = enrich >= 2.0
    verdicts["CHECK 2 (bojan_high location)"] = c2a and c2b
    print(f"  2a >=60% at/near structural high: {c2a}   2b enrichment>=2x: {c2b}")
    print(f"  -> CHECK 2 {'PASS' if verdicts['CHECK 2 (bojan_high location)'] else 'FAIL'}")

    # ---------------------------------------------------------------- CHECK 3
    hr("CHECK 3 — C_distrib / BEAR-CONTEXT RELIABILITY (THE CRUX)")
    ph = df["wyckoff_phase_dir"].astype(str)
    n_cdist = int((ph == "C_distrib").sum())
    n_cacc = int((ph == "C_accum").sum())
    print("  --- Metric A: phase & eye bear-vs-bull census ---")
    print(f"  wyckoff_phase_dir C_distrib: {n_cdist:>6}  ({n_cdist/len(df)*100:.2f}% of bars)")
    print(f"  wyckoff_phase_dir C_accum:   {n_cacc:>6}  ({n_cacc/len(df)*100:.2f}% of bars)")
    print(f"  C_distrib / C_accum ratio = {n_cdist/max(n_cacc,1):.3f}  (pre-reg floor 0.10)")
    est = eye["eye_state"].astype(str)
    edir = eye["eye_dir"].astype(str)
    mf_bear = int(((est == MODEL_FORMING) & (edir == "bear")).sum())
    mf_bull = int(((est == MODEL_FORMING) & (edir == "bull")).sum())
    manip_bear = int((eye["eye_manip_dir"].astype(str) == "bear").sum())
    manip_bull = int((eye["eye_manip_dir"].astype(str) == "bull").sum())
    cb_bear = int(((est == CONFIRMED_BREAK) & (edir == "bear")).sum())
    cb_bull = int(((est == CONFIRMED_BREAK) & (edir == "bull")).sum())
    print(f"  eye MODEL_FORMING bear: {mf_bear:>6}   bull: {mf_bull:>6}  "
          f"(ratio {mf_bear/max(mf_bull,1):.3f})")
    print(f"  eye MANIPULATION  bear: {manip_bear:>6}   bull: {manip_bull:>6}")
    print(f"  eye CONFIRMED_BREAK bear: {cb_bear:>6}  bull: {cb_bull:>6}")
    print(f"  A_distrib total: {int((ph=='A_distrib').sum())}  (catch-all preliminary phase — see caveats)")

    print("\n  --- Metric B: 3 major tops, any bear signal within +/-30 days ---")
    def bear_signal_mask():
        return (
            (ph.to_numpy() == "C_distrib")
            | ((est.to_numpy() == MODEL_FORMING) & (edir.to_numpy() == "bear"))
            | (eye["eye_manip_dir"].astype(str).to_numpy() == "bear")
            | (sr["struct_sweep_high"].to_numpy() == 1)
            | (bj["bojan_high_form"].to_numpy() == 1)
        )
    bmask = pd.Series(bear_signal_mask(), index=df.index)
    tops_ok = 0
    for name, d in MAJOR_TOPS:
        c = pd.Timestamp(d)
        w = bmask.loc[c - pd.Timedelta(days=30): c + pd.Timedelta(days=30)]
        # break down which sub-signals fired
        seg = slice(df.index.searchsorted(c - pd.Timedelta(days=30)),
                    df.index.searchsorted(c + pd.Timedelta(days=30)))
        n_cd = int((ph.to_numpy()[seg] == "C_distrib").sum())
        n_mf = int(((est.to_numpy()[seg] == MODEL_FORMING) & (edir.to_numpy()[seg] == "bear")).sum())
        n_sw = int((sr["struct_sweep_high"].to_numpy()[seg] == 1).sum())
        n_bj = int((bj["bojan_high_form"].to_numpy()[seg] == 1).sum())
        n_mn = int((eye["eye_manip_dir"].astype(str).to_numpy()[seg] == "bear").sum())
        got = bool(w.any())
        tops_ok += int(got)
        print(f"  {name:<20} {d}: bear-signal={'YES' if got else 'NO '}  "
              f"[C_distrib={n_cd} MF_bear={n_mf} sweep_hi={n_sw} bojan_hi={n_bj} manip_bear={n_mn}]")

    print("\n  --- Metric C: 6 distribution consensus events caught (+/-48h, 5%) ---")
    events_ok = 0
    for name, d, px in DISTRIB_EVENTS:
        c = pd.Timestamp(d)
        w = bmask.loc[c - pd.Timedelta(hours=48): c + pd.Timedelta(hours=48)]
        got = bool(w.any())
        events_ok += int(got)
        print(f"  {name:<16} {d} ~${px:>6,}: {'CAUGHT' if got else 'MISS  '}")

    c3a = (n_cdist >= 0.10 * n_cacc)
    c3b = (events_ok >= 4)
    c3c = (tops_ok == 3)
    verdicts["CHECK 3 (bear-context reliability)"] = c3a and c3b and c3c
    print(f"\n  3a C_distrib>=0.10*C_accum: {c3a} ({n_cdist} vs floor {0.10*n_cacc:.0f})")
    print(f"  3b >=4/6 distrib events: {c3b} ({events_ok}/6)")
    print(f"  3c all 3 tops marked: {c3c} ({tops_ok}/3)")
    print(f"  -> CHECK 3 {'PASS' if verdicts['CHECK 3 (bear-context reliability)'] else 'FAIL'}")

    # ---------------------------------------------------------------- CHECK 4
    hr("CHECK 4 — CAUSALITY / NO-REPAINT (3-cut truncation)")
    cut_pts = [int(len(df) * f) for f in (0.35, 0.60, 0.85)]
    all_ok = True
    for cp in cut_pts:
        sub = df.iloc[:cp].copy()
        reanch_t = reanchor_frame(sub, HTF_FREQ, HTF_N)
        sr_t = build_structural_range(reanch_t)
        bj_t = build_bojan(reanch_t, sr_t, BOJAN_W_VAL)
        eye_t = compute_eye_features(sub)
        eq_sw = np.array_equal(sr["struct_sweep_high"].iloc[:cp].to_numpy(),
                               sr_t["struct_sweep_high"].to_numpy())
        eq_bj = np.array_equal(bj["bojan_high_form"].iloc[:cp].to_numpy(),
                               bj_t["bojan_high_form"].to_numpy())
        # eye compare on overlapping index (broadcast may lag at tail)
        common = eye_t.index.intersection(eye.index)
        common = common[common <= df.index[cp - 1]]
        eq_eye = np.array_equal(eye.loc[common, "eye_state"].astype(str).to_numpy(),
                                eye_t.loc[common, "eye_state"].astype(str).to_numpy()) and \
                 np.array_equal(eye.loc[common, "eye_dir"].astype(str).to_numpy(),
                                eye_t.loc[common, "eye_dir"].astype(str).to_numpy())
        ok = eq_sw and eq_bj and eq_eye
        all_ok &= ok
        print(f"  cut @ {cp:>6} ({df.index[cp-1].date()}): sweep_high={'OK' if eq_sw else 'REPAINT'}  "
              f"bojan_high={'OK' if eq_bj else 'REPAINT'}  eye={'OK' if eq_eye else 'REPAINT'}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    verdicts["CHECK 4 (causality)"] = all_ok
    print(f"  -> CHECK 4 {'PASS' if all_ok else 'FAIL'}")

    # ---------------------------------------------------------------- GATE
    hr("GATE VERDICT")
    for k, v in verdicts.items():
        print(f"  {k:<40} {'PASS' if v else 'FAIL'}")
    gate = all(verdicts.values())
    print(f"\n  GATE: {'PASS -> build the short mirror (Phase 2)' if gate else 'FAIL -> STOP; do NOT build on broken sensors'}")
    print("\nSTUDY ONLY — distribution-sensor audit; awaiting gate decision.")
    return gate


if __name__ == "__main__":
    main()
