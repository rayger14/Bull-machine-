#!/usr/bin/env python3
"""
Append macro features to existing feature stores for regime classification.

Loads macro_history.parquet and merges the 13 regime features into existing
feature stores without rebuilding from scratch.

Usage:
    python3 bin/append_macro_to_feature_store.py --asset BTC --start 2024-01-01 --end 2024-12-31
    python3 bin/append_macro_to_feature_store.py --asset BTC --start 2022-01-01 --end 2023-12-31
"""

import pandas as pd
import argparse
from pathlib import Path

# 13 features required by regime classifier (from configs/btc_v8_adaptive.json)
REGIME_FEATURES = [
    'VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y',
    'USDT.D', 'BTC.D', 'TOTAL', 'TOTAL2',
    'funding', 'oi', 'rv_20d', 'rv_60d'
]

def main():
    parser = argparse.ArgumentParser(description='Append macro features to feature store')
    parser.add_argument('--asset', required=True, help='Asset symbol (e.g., BTC, ETH)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--macro-file', default='data/macro/macro_history.parquet',
                       help='Path to macro history file')

    args = parser.parse_args()

    # Paths
    feature_store_path = Path(f'data/features_mtf/{args.asset}_1H_{args.start}_to_{args.end}.parquet')
    macro_path = Path(args.macro_file)

    print(f"\n{'='*60}")
    print(f"Appending Macro Features to Feature Store")
    print(f"{'='*60}")

    # 1. Load existing feature store
    if not feature_store_path.exists():
        print(f"❌ Feature store not found: {feature_store_path}")
        return 1

    print(f"\n1. Loading feature store: {feature_store_path}")
    features = pd.read_parquet(feature_store_path)
    print(f"   Shape: {features.shape}")

    # Reset index to get timestamp as column (index is named 'time')
    if 'timestamp' not in features.columns:
        features = features.reset_index()
        # Rename 'time' to 'timestamp' for consistency with macro data
        if 'time' in features.columns:
            features = features.rename(columns={'time': 'timestamp'})

    print(f"   Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")

    # 2. Load macro history
    if not macro_path.exists():
        print(f"❌ Macro history not found: {macro_path}")
        return 1

    print(f"\n2. Loading macro history: {macro_path}")
    macro = pd.read_parquet(macro_path)
    print(f"   Shape: {macro.shape}")
    print(f"   Date range: {macro['timestamp'].min()} to {macro['timestamp'].max()}")

    # 3. Filter macro to same date range as feature store (with some buffer)
    start_dt = pd.to_datetime(args.start).tz_localize('UTC') - pd.Timedelta(days=7)  # 7-day buffer
    end_dt = pd.to_datetime(args.end).tz_localize('UTC') + pd.Timedelta(days=1)

    macro_filtered = macro[
        (macro['timestamp'] >= start_dt) &
        (macro['timestamp'] <= end_dt)
    ][['timestamp'] + REGIME_FEATURES].copy()

    print(f"\n3. Filtered macro to {start_dt.date()} - {end_dt.date()}")
    print(f"   Rows: {len(macro_filtered)}")

    # 4. Merge with feature store (asof merge for forward-fill behavior)
    print(f"\n4. Merging macro features...")

    # Ensure both are sorted by timestamp
    features = features.sort_values('timestamp').reset_index(drop=True)
    macro_filtered = macro_filtered.sort_values('timestamp').reset_index(drop=True)

    # Use merge_asof to get closest macro value for each feature timestamp
    # (forward-fill behavior: use most recent macro value)
    merged = pd.merge_asof(
        features,
        macro_filtered,
        on='timestamp',
        direction='backward',  # Use most recent macro value
        tolerance=pd.Timedelta(hours=2)  # Max 2-hour gap
    )

    # Check for any missing values
    missing_counts = merged[REGIME_FEATURES].isna().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠️  Warning: Some macro features have missing values:")
        for feat, count in missing_counts.items():
            if count > 0:
                print(f"   {feat}: {count} missing ({count/len(merged)*100:.1f}%)")

    print(f"\n5. Merged shape: {merged.shape}")
    print(f"   Added {len(REGIME_FEATURES)} macro features")

    # 5. Backup original
    backup_path = feature_store_path.with_suffix('.parquet.bak_pre_macro')
    if not backup_path.exists():
        print(f"\n6. Creating backup: {backup_path.name}")
        features.to_parquet(backup_path, compression='snappy')
    else:
        print(f"\n6. Backup already exists: {backup_path.name}")

    # 6. Save updated feature store
    print(f"\n7. Saving updated feature store...")
    # Set timestamp back as index to match original format (named 'time')
    merged_indexed = merged.set_index('timestamp')
    merged_indexed.index.name = 'time'  # Original index was named 'time', not 'timestamp'
    merged_indexed.to_parquet(feature_store_path, compression='snappy')

    print(f"\n✅ SUCCESS!")
    print(f"\nFeature store updated:")
    print(f"  Path: {feature_store_path}")
    print(f"  Shape: {features.shape} → {merged.shape}")
    print(f"  New columns: {', '.join(REGIME_FEATURES)}")
    print(f"\nBackup saved to: {backup_path}")
    print(f"{'='*60}\n")

    return 0

if __name__ == '__main__':
    exit(main())
