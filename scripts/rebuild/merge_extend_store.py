#!/usr/bin/env python3
"""Splice the LFC-replayed segment (2025-01-01 → present) onto the V12 store.

Output: data/features_mtf/BTC_1H_FEATURES_V13_EXTENDED.parquet
(symlink NOT touched — backtests must opt in explicitly via --feature-store)

Steps:
  1. Rename segment columns to V12 conventions (FEAR_GREED → fear_greed, etc.)
  2. Derive V12-only convenience columns where trivially computable
     (extreme_fear/extreme_greed from fear_greed; raw VIX/DXY from macro cache)
  3. Keep only V12-schema columns; missing ones → NaN (backtester defaults
     handle absent features per-row via .get(col, default))
  4. Blank regime_label on segment rows → backtester's _derive_regime_labels
     re-derives ALL labels consistently from price (it already does this for
     V12, whose stored labels are degenerate)
  5. Concat at the 2025-01-01 boundary, validate continuity, write V13

Also writes a coverage report: results/rebuild/v13_coverage.json
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
OUT = REPO / "data/features_mtf/BTC_1H_FEATURES_V13_EXTENDED.parquet"
REPORT = REPO / "results/rebuild/v13_coverage.json"

SPLICE = "2025-01-01"

RENAME = {
    "FEAR_GREED": "fear_greed",
    # fear_greed_norm matches; *_Z names match V12 (VIX_Z, DXY_Z, GOLD_Z, OIL_Z→?)
    "OIL_Z": "WTI_Z",          # live OIL = CL=F = WTI in V12
    "oil_price": "WTI",
    "gold_price": "GOLD",
}

# Champion-critical features that MUST be present in the segment
CRITICAL = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "atr_percentile", "volume_zscore", "adx",
    "wyckoff_score", "fusion_smc",
    "oi_change_4h", "taker_imbalance", "ls_ratio_extreme", "funding_Z",
    "fear_greed", "VIX_Z", "DXY_Z",
]


def main():
    v12 = pd.read_parquet(V12)
    if getattr(v12.index, "tz", None) is not None:
        v12.index = v12.index.tz_localize(None)
    seg = pd.read_parquet(SEGMENT)
    print(f"V12: {v12.shape}  {v12.index[0]} → {v12.index[-1]}")
    print(f"segment: {seg.shape}  {seg.index[0]} → {seg.index[-1]}")

    seg = seg.rename(columns=RENAME)

    # Derived V12 conveniences
    if "fear_greed" in seg.columns:
        fg = pd.to_numeric(seg["fear_greed"], errors="coerce")
        seg["extreme_fear"] = (fg <= 25).astype(float)
        seg["extreme_greed"] = (fg >= 75).astype(float)
    # Raw VIX/DXY/yields from the daily macro cache (ffilled to 1H)
    if MACRO.exists():
        m = pd.read_parquet(MACRO)
        daily = m.reindex(seg.index.normalize(), method="ffill")
        daily.index = seg.index
        for src, dst in [("VIX", "VIX"), ("DXY", "DXY"), ("GOLD", "GOLD"),
                         ("OIL", "WTI"), ("YIELD_10Y", "YIELD_10Y")]:
            if dst not in seg.columns or seg[dst].isna().all():
                seg[dst] = daily[src].values

    # Blank regime labels → force consistent re-derivation in the backtester
    if "regime_label" in seg.columns:
        seg["regime_label"] = np.nan

    # numeric coercion for any object columns that are really numeric
    for c in seg.columns:
        if seg[c].dtype == object:
            coerced = pd.to_numeric(seg[c], errors="coerce")
            if coerced.notna().mean() > 0.5:
                seg[c] = coerced

    new_rows = seg.loc[seg.index >= SPLICE]
    aligned = new_rows.reindex(columns=v12.columns)  # V12 schema; missing → NaN

    covered = [c for c in v12.columns if c in new_rows.columns and new_rows[c].notna().any()]
    missing = [c for c in v12.columns if c not in covered]
    crit_missing = [c for c in CRITICAL if c not in covered]
    print(f"\nschema coverage: {len(covered)}/{len(v12.columns)} V12 columns populated")
    print(f"champion-critical missing: {crit_missing or 'NONE'}")
    if crit_missing:
        raise SystemExit(f"ABORT: critical features missing from segment: {crit_missing}")

    # dtype harmonization (avoid object/float clashes on concat)
    for c in v12.columns:
        if v12[c].dtype != object:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
        else:
            # object col in V12 (labels) — keep as object
            pass

    v13 = pd.concat([v12, aligned]).sort_index()
    v13 = v13[~v13.index.duplicated(keep="first")]  # V12 wins on any overlap

    # mixed-type object columns (e.g. str labels in V12, ints/NaN in segment)
    # break pyarrow — normalize to str, preserving NaN
    for c in v13.columns:
        if v13[c].dtype == object:
            mask = v13[c].notna()
            v13.loc[mask, c] = v13.loc[mask, c].astype(str)

    # Boundary continuity check
    pre = v13.loc["2024-12-30":"2024-12-31", "close"].iloc[-1]
    post = v13.loc["2025-01-01":"2025-01-02", "close"].iloc[0]
    gap_pct = abs(post - pre) / pre * 100
    print(f"boundary close: {pre:.0f} → {post:.0f} ({gap_pct:.2f}% gap)")

    v13.to_parquet(OUT)
    print(f"\nWROTE {OUT}: {v13.shape}  {v13.index[0]} → {v13.index[-1]}")

    REPORT.write_text(json.dumps({
        "v13_shape": list(v13.shape),
        "segment_rows_added": int(len(aligned)),
        "covered_cols": len(covered),
        "missing_cols": missing,
        "critical_missing": crit_missing,
        "boundary_gap_pct": float(gap_pct),
    }, indent=2))
    print(f"coverage report → {REPORT}")


if __name__ == "__main__":
    main()
