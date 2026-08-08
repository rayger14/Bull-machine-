"""
DIAGNOSTIC (STUDY ONLY): why median episode lifetime stays ~0.4d while change-frac
says the drawn level is weeks-stable, and whether break-acceptance closes the gap.

Two measurements on the 1D N=5 anchor (the most stable variant):
 1. LEVEL-DWELL: how long a given (range_low, range_high) pair persists, BRIDGING brief
    forming/broken gaps if the SAME box resumes. This is the human "how long did the box
    last" number, immune to 1H boundary-tap fragmentation.
 2. BREAK-ACCEPTANCE sweep: re-run the machine's diagnostic knobs (break_buffer_atr,
    break_confirm_bars) — does requiring real acceptance beyond the HTF boundary collapse
    the micro-episodes into weeks-long ones, and what does it cost in POC trend-phi?
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from structural_range import build_structural_range, build_structural_poc, ACTIVE
from htf_pivots import reanchor_frame
from validate_structural_range import phi_coef, episodes

STORE = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"


def level_dwell_days(sr):
    """Dwell time of the drawn box KEYED ON range_high (the ceiling only redraws, never
    tightens). Consecutive bars sharing the same non-NaN range_high = one drawn box,
    even across brief forming gaps that resume the same ceiling. Returns dwell days."""
    rh = sr["struct_range_high"].to_numpy(float)
    # compress to runs of equal (non-nan) range_high; forming gaps (nan) split only if
    # the ceiling changes when it resumes
    dwell = []
    cur = None
    count = 0
    for i in range(len(rh)):
        v = rh[i]
        if np.isnan(v):
            continue  # forming/broken: don't reset yet; bridge to next active
        if cur is None:
            cur = v; count = 1
        elif v == cur:
            count += 1
        else:
            dwell.append(count); cur = v; count = 1
    if cur is not None:
        dwell.append(count)
    return np.array(dwell) / 24.0


def run(df, freq, N, buffer, confirm):
    df_re = reanchor_frame(df, freq, N)
    sr = build_structural_range(df_re, break_buffer_atr=buffer, break_confirm_bars=confirm)
    active = (sr["struct_range_state"] == ACTIVE).to_numpy()
    eps = episodes(sr)
    lt = np.array([e["bars"] for e in eps]) / 24.0 if eps else np.array([0.0])
    dwell = level_dwell_days(sr)
    # POC contamination
    spoc = build_structural_poc(df, sr).to_numpy()
    close = df["close"].to_numpy(); ema200 = df["ema_200"].to_numpy(); atr = df["atr_14"].to_numpy()
    up = close > ema200
    valid = np.isfinite(spoc) & active & np.isfinite(ema200)
    phi = phi_coef((close > spoc)[valid], up[valid])
    # weekly drift
    w = 168; d = []
    for i in range(w, len(spoc)):
        if active[i] and active[i-w] and np.isfinite(spoc[i]) and np.isfinite(spoc[i-w]) and atr[i] > 0:
            d.append(abs(spoc[i]-spoc[i-w])/atr[i])
    wd = np.median(d) if d else float("nan")
    return dict(buffer=buffer, confirm=confirm, n_eps=len(eps),
                ep_med=float(np.median(lt)), ep_mean=float(lt.mean()),
                dwell_med=float(np.median(dwell)), dwell_mean=float(dwell.mean()),
                dwell_max=float(dwell.max()), n_boxes=len(dwell),
                active_cov=float(active.mean()), phi=float(phi), wd=float(wd))


def main():
    df = pd.read_parquet(STORE)
    print("DIAGNOSTIC 1 — LEVEL-DWELL (bridging 1H break/reform churn), pre-registered knobs (buf=0,confirm=1)")
    print(f"{'variant':<10}{'nEps':>7}{'epMed_d':>9}{'epMean_d':>10}"
          f"{'nBoxes':>8}{'dwellMed_d':>12}{'dwellMean_d':>13}{'dwellMax_d':>12}")
    for freq, N in [("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)]:
        r = run(df, freq, N, 0.0, 1)
        print(f"{freq+' N='+str(N):<10}{r['n_eps']:>7}{r['ep_med']:>9.2f}{r['ep_mean']:>10.2f}"
              f"{r['n_boxes']:>8}{r['dwell_med']:>12.1f}{r['dwell_mean']:>13.1f}{r['dwell_max']:>12.1f}")

    print("\nDIAGNOSTIC 2 — BREAK-ACCEPTANCE sweep on 1D N=5 (does acceptance collapse micro-episodes?)")
    print("  (diagnostic knobs, NOT the pre-registered spec; watch phi = trend contamination)")
    print(f"{'buf_atr':>8}{'confirm':>8}{'nEps':>7}{'epMed_d':>9}{'epMean_d':>10}"
          f"{'dwellMed_d':>12}{'actCov%':>9}{'phi':>8}{'pocWkDrift':>12}")
    for buf, conf in [(0.0,1),(0.25,1),(0.5,1),(0.0,2),(0.0,3),(0.5,2),(1.0,2)]:
        r = run(df, "1D", 5, buf, conf)
        print(f"{buf:>8.2f}{conf:>8}{r['n_eps']:>7}{r['ep_med']:>9.2f}{r['ep_mean']:>10.2f}"
              f"{r['dwell_med']:>12.1f}{r['active_cov']*100:>9.1f}{r['phi']:>+8.3f}{r['wd']:>12.2f}")

    print("\nSTUDY ONLY — sensor-fix (HTF anchor) diagnostic, awaiting user decision.")


if __name__ == "__main__":
    main()
