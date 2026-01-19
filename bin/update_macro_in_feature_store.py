#!/usr/bin/env python3
"""
Update existing macro features in feature store with real data from macro_history.

This script replaces the placeholder macro data that was previously appended
with the real data from the updated macro_history.parquet file.
"""

import pandas as pd
import argparse
from pathlib import Path

REGIME_FEATURES = [
    'VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y',
    'USDT.D', 'BTC.D', 'TOTAL', 'TOTAL2',
    'funding', 'oi', 'rv_20d', 'rv_60d'
]

def main():
    parser = argparse.ArgumentParser(description='Update macro features in feature store')
    parser.add_argument('--asset', required=True, help='Asset symbol (e.g., BTC, ETH)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--macro-file', default='data/macro/macro_history.parquet',
                       help='Path to macro history file')

    args = parser.parse_args()

    # Paths
    feature_store_path = Path(f'data/features_mtf/{args.asset}_1H_{args.start}_to_{args.end}.parquet')
    macro_path = Path(args.macro_file)

    print(f"\\n{'='*60}")
    print(f"Updating Macro Features in Feature Store")
    print(f"{'='*60}\\n")

    # 1. Load feature store
    if not feature_store_path.exists():
        print(f"❌ Feature store not found: {feature_store_path}")
        return 1

    print(f"1. Loading feature store: {feature_store_path}")
    features = pd.read_parquet(feature_store_path)
    print(f"   Shape: {features.shape}")

    # Reset index to get timestamp as column
    if 'timestamp' not in features.columns:
        features = features.reset_index()
        if 'time' in features.columns:
            features = features.rename(columns={'time': 'timestamp'})

    print(f"   Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")

    # 2. Load macro history
    if not macro_path.exists():
        print(f"❌ Macro history not found: {macro_path}")
        return 1

    print(f"\\n2. Loading macro history: {macro_path}")
    macro = pd.read_parquet(macro_path)
    print(f"   Shape: {macro.shape}")

    # 3. Filter macro to date range
    start_dt = pd.to_datetime(args.start).tz_localize('UTC') - pd.Timedelta(days=7)
    end_dt = pd.to_datetime(args.end).tz_localize('UTC') + pd.Timedelta(days=1)

    macro_filtered = macro[
        (macro['timestamp'] >= start_dt) &
        (macro['timestamp'] <= end_dt)
    ][['timestamp'] + REGIME_FEATURES].copy()

    print(f"\\n3. Filtered macro to {start_dt.date()} - {end_dt.date()}")
    print(f"   Rows: {len(macro_filtered)}")

    # 4. Drop existing macro columns
    existing_macro_cols = [col for col in REGIME_FEATURES if col in features.columns]
    if existing_macro_cols:
        print(f"\\n4. Dropping existing macro columns: {len(existing_macro_cols)} columns")
        features = features.drop(columns=existing_macro_cols)
    else:
        print(f"\\n4. No existing macro columns to drop")

    # 5. Merge with feature store
    print(f"\\n5. Merging updated macro features...")

    features = features.sort_values('timestamp').reset_index(drop=True)
    macro_filtered = macro_filtered.sort_values('timestamp').reset_index(drop=True)

    merged = pd.merge_asof(
        features,
        macro_filtered,
        on='timestamp',
        direction='backward',
        tolerance=pd.Timedelta(hours=2)
    )

    # Check coverage
    missing_counts = merged[REGIME_FEATURES].isna().sum()
    print(f"\\n6. Data quality:")
    for feat in REGIME_FEATURES:
        count = len(merged) - missing_counts[feat]
        pct = count / len(merged) * 100
        if missing_counts[feat] > 0:
            print(f"   ⚠️  {feat}: {count}/{len(merged)} ({pct:.1f}% coverage)")
        else:
            print(f"   ✅ {feat}: {count}/{len(merged)} (100% coverage)")

    # 7. Backup original
    backup_path = feature_store_path.with_suffix('.parquet.bak_pre_real_macro')
    if not backup_path.exists():
        print(f"\\n7. Creating backup: {backup_path.name}")
        pd.read_parquet(feature_store_path).to_parquet(backup_path, compression='snappy')
    else:
        print(f"\\n7. Backup already exists: {backup_path.name}")

    # 8. Save updated feature store
    print(f"\\n8. Saving updated feature store...")
    merged_indexed = merged.set_index('timestamp')
    merged_indexed.index.name = 'time'
    merged_indexed.to_parquet(feature_store_path, compression='snappy')

    print(f"\\n✅ SUCCESS!")
    print(f"\\n   Updated: {feature_store_path}")
    print(f"   Shape: {merged.shape}")
    print(f"   Macro columns: {', '.join(REGIME_FEATURES)}")
    print(f"\\n{'='*60}\\n")

    return 0

if __name__ == '__main__':
    exit(main())
