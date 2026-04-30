#!/usr/bin/env python3
"""
Validate SOUL Restoration
=========================

Compare broken feature store vs restored feature store to verify all macro
features are now functioning with real data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_feature_store(path: Path, name: str):
    """Analyze a feature store and return stats."""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    print(f"Path: {path}")

    df = pd.read_parquet(path)

    # Get timestamp info
    if df.index.name == 'timestamp':
        ts_min = df.index.min()
        ts_max = df.index.max()
    else:
        ts_min = df['timestamp'].min()
        ts_max = df['timestamp'].max()

    print(f"Bars: {len(df):,}")
    print(f"Features: {len(df.columns)}")
    print(f"Period: {ts_min} to {ts_max}")
    print(f"File size: {path.stat().st_size / (1024*1024):.1f} MB")

    # Analyze macro features
    macro_features = {
        'VIX_Z': 'VIX Volatility Z-Score',
        'DXY_Z': 'Dollar Index Z-Score',
        'YIELD_10Y_Z': '10Y Treasury Z-Score',
        'YIELD_2Y_Z': '2Y Treasury Z-Score',
        'BTC.D_Z': 'BTC Dominance Z-Score',
        'USDT.D_Z': 'USDT Dominance Z-Score',
        'funding_Z': 'Funding Rate Z-Score',
    }

    print(f"\nMACRO FEATURES:")
    print(f"{'-' * 80}")

    stats = {}
    for feat, desc in macro_features.items():
        if feat in df.columns or (df.index.name and feat in df.index):
            col = df[feat] if feat in df.columns else df.index
            unique = col.nunique()
            nan_pct = (col.isna().sum() / len(col)) * 100
            mean = col.mean() if not col.isna().all() else np.nan
            std = col.std() if not col.isna().all() else np.nan

            status = "✅" if unique > 1000 else ("⚠️ " if unique > 1 else "🔴")

            print(f"{status} {feat:20s} | {unique:8,} unique | {nan_pct:5.1f}% NaN | μ={mean:7.2f} σ={std:7.2f}")

            stats[feat] = {
                'unique': unique,
                'nan_pct': nan_pct,
                'mean': mean,
                'std': std,
                'working': unique > 100  # Consider working if >100 unique values
            }
        else:
            print(f"🔴 {feat:20s} | MISSING")
            stats[feat] = {'working': False}

    return stats


def main():
    data_root = Path("/Users/raymondghandchi/Bull-machine-/Bull-machine-/data/features_mtf")

    broken_path = data_root / "BTC_1H_REGIME_COMPLETE_WITH_FUNDING_2018-01-01_to_2024-12-31_20260122_151606.parquet"
    restored_path = data_root / "BTC_1H_SOUL_RESTORED_2018_2024.parquet"

    # Analyze both
    broken_stats = analyze_feature_store(broken_path, "BEFORE: Broken Feature Store (47.8% constant)")
    restored_stats = analyze_feature_store(restored_path, "AFTER: SOUL Restored Feature Store")

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("RESTORATION SUMMARY")
    print(f"{'=' * 80}")

    macro_features = ['VIX_Z', 'DXY_Z', 'YIELD_10Y_Z', 'YIELD_2Y_Z', 'BTC.D_Z', 'USDT.D_Z', 'funding_Z']

    before_working = sum(1 for f in macro_features if broken_stats.get(f, {}).get('working', False))
    after_working = sum(1 for f in macro_features if restored_stats.get(f, {}).get('working', False))

    print(f"\nMacro Features Working:")
    print(f"  BEFORE: {before_working}/{len(macro_features)} ({before_working/len(macro_features)*100:.1f}%)")
    print(f"  AFTER:  {after_working}/{len(macro_features)} ({after_working/len(macro_features)*100:.1f}%)")
    print(f"  IMPROVEMENT: +{after_working - before_working} features restored")

    # Feature-by-feature comparison
    print(f"\nFeature-by-Feature Comparison:")
    print(f"{'-' * 80}")
    print(f"{'Feature':<20s} | {'Before Unique':>15s} | {'After Unique':>15s} | {'Status':>10s}")
    print(f"{'-' * 80}")

    for feat in macro_features:
        before_unique = broken_stats.get(feat, {}).get('unique', 0)
        after_unique = restored_stats.get(feat, {}).get('unique', 0)

        if after_unique > before_unique * 10:
            status = "✅ FIXED"
        elif after_unique > before_unique:
            status = "⚠️  IMPROVED"
        elif after_unique == before_unique and after_unique > 100:
            status = "✅ OK"
        else:
            status = "🔴 SAME"

        print(f"{feat:<20s} | {before_unique:>15,} | {after_unique:>15,} | {status:>10s}")

    print(f"\n{'=' * 80}")
    if after_working == len(macro_features):
        print("✅ SOUL RESTORATION SUCCESSFUL - ALL MACRO FEATURES WORKING")
    elif after_working > before_working:
        print(f"⚠️  PARTIAL SUCCESS - {after_working}/{len(macro_features)} features working")
    else:
        print("🔴 RESTORATION FAILED - No improvement")
    print(f"{'=' * 80}")

    print(f"\nRecommended Action:")
    print(f"  1. Update backtest configs to use:")
    print(f"     feature_store_path: '{restored_path}'")
    print(f"  2. Run validation backtest on Jan 2022")
    print(f"  3. Verify regime detection working (should see crisis signals)")
    print(f"  4. Check archetype diversity (expect >2 archetypes now)")


if __name__ == "__main__":
    main()
