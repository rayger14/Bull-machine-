#!/usr/bin/env python3
"""Parity gate: LFC-replayed segment vs V12 store on the 2024 H2 overlap.

The segment (built on the LIVE feature path) and V12 (built on the deleted
batch pipeline) coexist on 2024-07-01..2024-12-31. Champion-critical features
must correlate strongly; documented-divergent features (chop_score etc.) are
reported but not gating.

Writes results/rebuild/overlap_validation.json and prints a verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEG = REPO / "results/rebuild/segment_lfc.parquet"
V12 = REPO / "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"
REPORT = REPO / "results/rebuild/overlap_validation.json"

WINDOW = ("2024-07-01", "2024-12-31")

# Gating features: the champion's hard gates + fusion + sizing inputs.
# (corr_min chosen per feature class: levels vs z-scores vs ranks)
GATING = {
    "close":           0.999,
    "rsi_14":          0.85,
    "volume_zscore":   0.70,
    "atr_percentile":  0.80,
    "adx":             0.85,
    "oi_change_4h":    0.70,
    "taker_imbalance": 0.70,
    "ls_ratio_extreme": 0.60,
}
# Reported only (known live/backtest divergence or different conventions)
REPORTED = ["chop_score", "wyckoff_score", "fusion_smc", "funding_Z",
            "atr_pct", "volume_ratio", "ema_slope_50", "instability_score"]


def main():
    seg = pd.read_parquet(SEG)
    v12 = pd.read_parquet(V12)
    if getattr(v12.index, "tz", None) is not None:
        v12.index = v12.index.tz_localize(None)
    a, b = seg.loc[WINDOW[0]:WINDOW[1]], v12.loc[WINDOW[0]:WINDOW[1]]
    common = a.index.intersection(b.index)
    print(f"overlap: {len(common)} bars {common[0]} → {common[-1]}\n")

    out, failures = {}, []
    print(f"{'feature':<18}{'corr':>7}{'seg_mean':>10}{'v12_mean':>10}{'gate':>7}{'verdict':>9}")
    for feat, cmin in {**GATING, **{f: None for f in REPORTED}}.items():
        if feat not in a.columns or feat not in b.columns:
            print(f"{feat:<18}{'—':>7}{'—':>10}{'—':>10}{'—':>7}{'MISSING':>9}")
            if cmin is not None:
                failures.append(f"{feat}: missing")
            continue
        x = pd.to_numeric(a.loc[common, feat], errors="coerce")
        y = pd.to_numeric(b.loc[common, feat], errors="coerce")
        ok = x.notna() & y.notna()
        corr = float(x[ok].corr(y[ok])) if ok.sum() > 100 and x[ok].std() > 0 else float("nan")
        rec = {"corr": None if np.isnan(corr) else round(corr, 3),
               "seg_mean": round(float(x[ok].mean()), 4),
               "v12_mean": round(float(y[ok].mean()), 4),
               "gating": cmin is not None}
        out[feat] = rec
        if cmin is not None:
            verdict = "PASS" if (corr == corr and corr >= cmin) else "FAIL"
            if verdict == "FAIL":
                failures.append(f"{feat}: corr={corr:.3f} < {cmin}")
        else:
            verdict = "info"
        print(f"{feat:<18}{corr:>7.3f}{rec['seg_mean']:>10}{rec['v12_mean']:>10}"
              f"{str(cmin or '—'):>7}{verdict:>9}")

    # wick anomaly agreement (derived: lower/upper wick > 35% of range)
    def wick_anom(df):
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        lower = (np.minimum(df["close"], df["open"]) - df["low"]) / rng
        upper = (df["high"] - np.maximum(df["close"], df["open"])) / rng
        return ((lower > 0.35) | (upper > 0.35)).astype(float)

    wa, wb = wick_anom(a.loc[common]), wick_anom(b.loc[common])
    agree = float((wa == wb).mean())
    out["wick_anomaly_agreement"] = round(agree, 4)
    print(f"\nwick_anomaly gate agreement: {agree:.1%} (gate fires seg {wa.mean():.1%} vs v12 {wb.mean():.1%})")
    if agree < 0.95:
        failures.append(f"wick_anomaly agreement {agree:.1%} < 95%")

    REPORT.write_text(json.dumps({"window": WINDOW, "n_bars": len(common),
                                  "features": out, "failures": failures}, indent=2))
    print(f"\nreport → {REPORT}")
    print("VERDICT:", "PASS" if not failures else f"FAIL ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
