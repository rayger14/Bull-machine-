#!/usr/bin/env python3
"""
Patch GMM regime features from macro history into feature stores.
"""

import pandas as pd
from pathlib import Path

def patch_gmm_features(store_path: Path):
    """Patch GMM features into feature store."""

    print(f"\n{'='*80}")
    print(f"Patching GMM Features: {store_path.name}")
    print(f"{'='*80}")

    # Load feature store
    print(f"\n📂 Loading feature store...")
    features = pd.read_parquet(store_path)
    features['timestamp'] = pd.to_datetime(features['timestamp'], utc=True)

    print(f"   Shape: {features.shape}")
    print(f"   Range: {features['timestamp'].min()} to {features['timestamp'].max()}")

    # Load macro history
    print(f"\n📊 Loading macro history...")
    macro = pd.read_parquet('data/macro/macro_history.parquet')
    macro['timestamp'] = pd.to_datetime(macro['timestamp'], utc=True)

    print(f"   Shape: {macro.shape}")
    print(f"   Range: {macro['timestamp'].min()} to {macro['timestamp'].max()}")

    # GMM features to merge
    gmm_features = ['VIX_Z', 'DXY_Z', 'YC_SPREAD', 'YC_Z', 'BTC.D_Z', 'USDT.D_Z',
                    'RV_7', 'RV_20', 'RV_30', 'RV_60', 'TOTAL_RET', 'TOTAL2_RET']

    # Drop existing GMM features if present
    for feat in gmm_features:
        if feat in features.columns:
            features = features.drop(columns=[feat])

    # Merge GMM features
    print(f"\n🔗 Merging {len(gmm_features)} GMM features...")

    macro_subset = macro[['timestamp'] + gmm_features].copy()

    features = features.merge(macro_subset, on='timestamp', how='left')

    # Fill NaN with 0
    for feat in gmm_features:
        features[feat] = features[feat].fillna(0)

    # Check coverage
    print(f"\n✅ GMM features added:")
    for feat in gmm_features:
        nonzero = (features[feat] != 0.0).sum()
        coverage = nonzero / len(features) * 100
        print(f"   {feat:15s} {coverage:5.1f}% non-zero")

    # Save
    print(f"\n💾 Saving patched feature store...")
    features.to_parquet(store_path, index=False)

    print(f"   ✅ Saved: {store_path}")


def main():
    print(f"\n{'='*80}")
    print(f"🚀 PATCHING GMM REGIME FEATURES")
    print(f"{'='*80}")

    stores = [
        Path("data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet"),
        Path("data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet"),
    ]

    for store in stores:
        if store.exists():
            patch_gmm_features(store)
        else:
            print(f"❌ Not found: {store}")

    print(f"\n{'='*80}")
    print(f"✅ GMM FEATURES PATCHING COMPLETE")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    exit(main())
