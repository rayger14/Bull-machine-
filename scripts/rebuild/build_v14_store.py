#!/usr/bin/env python3
"""Build the V14 FULL LIVE-PATH feature store from the stitched full-history segment.

Unlike merge_extend_store.py (which spliced 2025+ onto V12), V14 is ENTIRELY
live-path: every bar 2018-01-01 → present computed by LiveFeatureComputer.
V12 is kept untouched for reproducibility of historical results.

Output: data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet
        (V12 295-col schema for backtester compatibility; live-path-only
         columns dropped, V12-only columns NaN — backtester defaults handle them)

Sanity gates before writing:
  - hourly continuity 2018-01-01 → present (no gaps)
  - champion-critical features present and non-degenerate
  - close parity vs V12 on overlap (corr >= 0.999; Binance-vs-Coinbase level
    differences of a few bps are expected and fine)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V12 = REPO / "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"
SEGMENT = REPO / "results/rebuild/segment_lfc.parquet"
MACRO = REPO / "data/cache/macro_daily_history.parquet"
OUT = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
REPORT = REPO / "results/rebuild/v14_report.json"

START = "2018-01-01"

RENAME = {
    "FEAR_GREED": "fear_greed",
    "OIL_Z": "WTI_Z",
    "oil_price": "WTI",
    "gold_price": "GOLD",
}

CRITICAL = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "atr_percentile", "volume_zscore", "adx", "liquidity_score",
    "wyckoff_score", "fusion_smc",
    "oi_change_4h", "taker_imbalance", "ls_ratio_extreme",
    "fear_greed", "VIX_Z", "DXY_Z",
]


def main():
    v12 = pd.read_parquet(V12)
    if getattr(v12.index, "tz", None) is not None:
        v12.index = v12.index.tz_localize(None)
    seg = pd.read_parquet(SEGMENT)
    print(f"segment: {seg.shape}  {seg.index[0]} → {seg.index[-1]}")

    seg = seg.rename(columns=RENAME)
    if "fear_greed" in seg.columns:
        fg = pd.to_numeric(seg["fear_greed"], errors="coerce")
        seg["extreme_fear"] = (fg <= 25).astype(float)
        seg["extreme_greed"] = (fg >= 75).astype(float)
    if MACRO.exists():
        m = pd.read_parquet(MACRO)
        daily = m.reindex(seg.index.normalize(), method="ffill")
        daily.index = seg.index
        for src, dst in [("VIX", "VIX"), ("DXY", "DXY"), ("GOLD", "GOLD"),
                         ("OIL", "WTI"), ("YIELD_10Y", "YIELD_10Y")]:
            if dst not in seg.columns or pd.to_numeric(seg[dst], errors="coerce").isna().all():
                seg[dst] = daily[src].values
    if "regime_label" in seg.columns:
        seg["regime_label"] = np.nan  # backtester re-derives consistently

    for c in seg.columns:
        if seg[c].dtype == object:
            coerced = pd.to_numeric(seg[c], errors="coerce")
            if coerced.notna().mean() > 0.5:
                seg[c] = coerced

    v14 = seg.loc[seg.index >= START].reindex(columns=v12.columns)

    # --- Sanity gates ---------------------------------------------------
    diffs = v14.index.to_series().diff().dropna()
    gaps = diffs[diffs != pd.Timedelta(hours=1)]
    print(f"rows: {len(v14)}, gaps: {len(gaps)}")
    if len(gaps):
        print(gaps.head(10).to_string())
        raise SystemExit("ABORT: gaps in V14 index")

    crit_bad = []
    for c in CRITICAL:
        col = pd.to_numeric(v14[c], errors="coerce") if c in v14.columns else None
        if col is None or col.notna().mean() < 0.5 or col.std() == 0:
            crit_bad.append(c)
    print(f"critical features degenerate/missing: {crit_bad or 'NONE'}")
    if crit_bad:
        raise SystemExit(f"ABORT: {crit_bad}")

    common = v14.index.intersection(v12.index)
    close_corr = float(pd.to_numeric(v14.loc[common, "close"], errors="coerce")
                       .corr(pd.to_numeric(v12.loc[common, "close"], errors="coerce")))
    print(f"close parity vs V12 on {len(common)} overlap bars: corr={close_corr:.5f}")
    if close_corr < 0.999:
        raise SystemExit("ABORT: close parity failed")

    coverage = {c: float(pd.to_numeric(v14[c], errors='coerce').notna().mean())
                for c in v14.columns}
    populated = sum(1 for v in coverage.values() if v > 0)

    for c in v14.columns:
        if v14[c].dtype == object:
            mask = v14[c].notna()
            v14.loc[mask, c] = v14.loc[mask, c].astype(str)

    v14.to_parquet(OUT)
    print(f"\nWROTE {OUT}: {v14.shape}  {v14.index[0]} → {v14.index[-1]}")
    print(f"schema coverage: {populated}/{len(v14.columns)} columns populated")

    REPORT.write_text(json.dumps({
        "shape": list(v14.shape),
        "start": str(v14.index[0]), "end": str(v14.index[-1]),
        "close_corr_vs_v12": close_corr,
        "populated_cols": populated,
        "empty_cols": [c for c, v in coverage.items() if v == 0],
    }, indent=2))
    print(f"report → {REPORT}")


if __name__ == "__main__":
    main()
