"""
DIAGNOSTIC (beyond pre-registration): the pre-registered single-1H-body-close break
rule produces ranges that whipsaw (median lifetime ~0.4 days), NOT the weeks-long
boxes a human draws. This script characterizes how much BREAK ACCEPTANCE (a buffer
in ATR and/or consecutive-close confirmation) is needed to reach human-like
persistence, and what it costs in coverage. It does NOT change the pre-registered
design; it tells the user what an acceptance rule would buy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from structural_range import build_structural_range, build_structural_poc, ACTIVE
from validate_structural_range import episodes, rolling_poc_60d, phi_coef

STORE = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"


def summarize(df, buf, conf):
    sr = build_structural_range(df, break_buffer_atr=buf, break_confirm_bars=conf)
    active = (sr["struct_range_state"] == ACTIVE).to_numpy()
    eps = episodes(sr)
    life = np.array([e["bars"] for e in eps]) / 24.0 if eps else np.array([0.0])
    # struct POC trend contamination + weekly drift
    close = df["close"].to_numpy(); atr = df["atr_14"].to_numpy()
    ema200 = df["ema_200"].to_numpy()
    spoc = build_structural_poc(df, sr).to_numpy()
    valid = np.isfinite(spoc) & active & np.isfinite(ema200)
    phi = phi_coef((close > spoc)[valid], (close > ema200)[valid]) if valid.sum() else np.nan
    # weekly drift
    w = 168; d = []
    for i in range(w, len(spoc)):
        if active[i] and active[i-w] and np.isfinite(spoc[i]) and np.isfinite(spoc[i-w]) and atr[i] > 0:
            d.append(abs(spoc[i]-spoc[i-w])/atr[i])
    wd = np.median(d) if d else np.nan
    return dict(buf=buf, conf=conf, n_eps=len(eps),
                med_life=np.median(life), p90_life=np.percentile(life, 90),
                max_life=life.max(), cover=active.mean(), phi=phi, wk_drift=wd)


def main():
    df = pd.read_parquet(STORE)
    print("DIAGNOSTIC — break-acceptance sweep (pre-registered = buf 0.0, conf 1)\n")
    print(f"{'buf_atr':>7} {'conf':>4} | {'#eps':>5} {'med_life_d':>10} {'p90_d':>7} "
          f"{'max_d':>6} {'cover%':>7} {'poc_phi':>8} {'poc_wkdrift':>11}")
    print("-" * 82)
    grid = [
        (0.0, 1),   # <-- PRE-REGISTERED
        (0.25, 1), (0.5, 1), (1.0, 1),
        (0.0, 2), (0.0, 3),
        (0.5, 2), (0.5, 3), (1.0, 3),
    ]
    for buf, conf in grid:
        r = summarize(df, buf, conf)
        tag = "  <-- PRE-REG" if (buf, conf) == (0.0, 1) else ""
        print(f"{r['buf']:>7.2f} {r['conf']:>4} | {r['n_eps']:>5} {r['med_life']:>10.2f} "
              f"{r['p90_life']:>7.1f} {r['max_life']:>6.1f} {r['cover']*100:>6.1f}% "
              f"{r['phi']:>+8.3f} {r['wk_drift']:>11.2f}{tag}")
    print("\nrolling-60d POC baseline: phi~+0.515, wk_drift~0.44, change-frac~100%")
    print("STUDY ONLY — sensor-fix prototype, awaiting user decision.")


if __name__ == "__main__":
    main()
