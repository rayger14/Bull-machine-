#!/usr/bin/env python3
"""
Add GMM Features to Existing Feature Store

Adds the 19 GMM features required by RegimeDetector to an existing feature store.
Uses RegimeDetector.engineer_features() with the store's macro data.

Usage:
    python3 bin/add_gmm_features.py --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.regime_detector import RegimeDetector


def main():
    parser = argparse.ArgumentParser(description='Add GMM features to feature store')
    parser.add_argument('--input', type=str, required=True, help='Path to input feature store')
    parser.add_argument('--output', type=str, help='Path to output (default: overwrites input)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print("\n" + "="*80)
    print("ADD GMM FEATURES TO FEATURE STORE")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Load existing feature store
    print(f"\n📊 Loading existing feature store...")
    df = pd.read_parquet(input_path)
    print(f"   Shape: {df.shape}")

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print(f"   Index type: {type(df.index)}")
    else:
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Initialize RegimeDetector
    print(f"\n🔮 Initializing RegimeDetector...")
    detector = RegimeDetector()
    print(f"   Required GMM features: {len(detector.features)}")

    # Check which features are missing
    missing = [f for f in detector.features if f not in df.columns]
    print(f"\n❌ Missing features ({len(missing)}): {missing[:5]}...")

    if len(missing) == 0:
        print("\n✅ All GMM features already present!")
        return 0

    # Engineer GMM features
    print(f"\n⚙️  Engineering GMM features...")
    print(f"   This may take a few minutes...")

    df_with_gmm = detector.engineer_features(df)

    # Verify all features were added
    still_missing = [f for f in detector.features if f not in df_with_gmm.columns]
    if still_missing:
        print(f"\n⚠️  Warning: Still missing {len(still_missing)} features: {still_missing}")

    # Count how many were added
    added_features = [f for f in detector.features if f in df_with_gmm.columns and f not in df.columns]
    print(f"\n✅ Added {len(added_features)} GMM features:")
    for feat in added_features[:10]:
        nan_count = df_with_gmm[feat].isna().sum()
        print(f"   - {feat}: {nan_count} NaN values ({100*nan_count/len(df_with_gmm):.1f}%)")
    if len(added_features) > 10:
        print(f"   ... and {len(added_features) - 10} more")

    # Save updated feature store
    print(f"\n💾 Saving updated feature store...")
    df_with_gmm.to_parquet(output_path)

    # Verify
    df_check = pd.read_parquet(output_path)
    print(f"\n✅ Verification:")
    print(f"   Shape: {df_check.shape}")
    print(f"   Features added: {df_check.shape[1] - df.shape[1]}")

    print("\n" + "="*80)
    print("✅ GMM FEATURES ADDED SUCCESSFULLY")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
