#!/usr/bin/env python3
"""
Patch Missing Funding Column into Existing Feature Store

Adds funding_Z to an existing feature store without full rebuild.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def compute_funding_z(funding_df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Compute z-score normalized funding rate.

    Args:
        funding_df: DataFrame with 'timestamp' and 'value' columns
        window: Rolling window for z-score (default 90 days)

    Returns:
        DataFrame with timestamp and funding_Z
    """
    df = funding_df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Compute rolling z-score
    rolling_mean = df['value'].rolling(window=window, min_periods=1).mean()
    rolling_std = df['value'].rolling(window=window, min_periods=1).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    df['funding_Z'] = (df['value'] - rolling_mean) / rolling_std

    # Fill NaN with 0
    df['funding_Z'] = df['funding_Z'].fillna(0)

    return df[['timestamp', 'funding_Z']]


def patch_feature_store(input_path: Path, funding_path: Path, output_path: Path = None):
    """
    Add funding_Z column to existing feature store.

    Args:
        input_path: Path to existing feature store parquet
        funding_path: Path to FUNDING_1D.csv
        output_path: Output path (defaults to overwriting input)
    """
    print(f"\n{'='*80}")
    print("PATCHING FEATURE STORE WITH FUNDING_Z")
    print(f"{'='*80}")

    # Load existing feature store
    print(f"\n📂 Loading feature store: {input_path}")
    features = pd.read_parquet(input_path)

    # Reset index if time is in index
    if features.index.name == 'time':
        features = features.reset_index()
        features = features.rename(columns={'time': 'timestamp'})

    print(f"   Shape: {features.shape}")
    print(f"   Columns: {len(features.columns)}")
    print(f"   Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")

    # Check if funding_Z already exists
    if 'funding_Z' in features.columns:
        print(f"\n⚠️  funding_Z already exists")
        print(f"   Current NaN%: {100 * features['funding_Z'].isna().sum() / len(features):.1f}%")
        print(f"   Dropping and recomputing...")
        features = features.drop(columns=['funding_Z'])

    # Load funding data
    print(f"\n📊 Loading funding data: {funding_path}")
    funding = pd.read_csv(funding_path)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'])
    print(f"   Records: {len(funding)}")
    print(f"   Date range: {funding['timestamp'].min()} to {funding['timestamp'].max()}")

    # Compute funding_Z
    print(f"\n🔢 Computing funding_Z (90-day rolling z-score)...")
    funding_z = compute_funding_z(funding)
    print(f"   Stats: min={funding_z['funding_Z'].min():.2f}, "
          f"max={funding_z['funding_Z'].max():.2f}, "
          f"mean={funding_z['funding_Z'].mean():.2f}")

    # Resample to hourly to match feature store
    print(f"\n⏰ Resampling to hourly...")
    funding_z = funding_z.set_index('timestamp')
    funding_z = funding_z.resample('1h').ffill()  # Forward fill daily values
    funding_z = funding_z.reset_index()

    # Ensure timezone-aware to match feature store
    if funding_z['timestamp'].dt.tz is None:
        funding_z['timestamp'] = funding_z['timestamp'].dt.tz_localize('UTC')

    print(f"   Hourly records: {len(funding_z)}")

    # Merge with feature store
    print(f"\n🔗 Merging with feature store...")
    features = features.merge(
        funding_z,
        on='timestamp',
        how='left'
    )

    # Check coverage
    nan_count = features['funding_Z'].isna().sum()
    nan_pct = 100 * nan_count / len(features)
    print(f"   NaN values: {nan_count} ({nan_pct:.1f}%)")

    if nan_count > 0:
        print(f"\n   ⚠️  Missing funding data for:")
        missing_dates = features[features['funding_Z'].isna()]['timestamp']
        print(f"      First missing: {missing_dates.min()}")
        print(f"      Last missing: {missing_dates.max()}")
        print(f"\n   Filling with 0 (neutral funding)...")
        features['funding_Z'] = features['funding_Z'].fillna(0)

    # Save
    if output_path is None:
        output_path = input_path

    print(f"\n💾 Saving patched feature store: {output_path}")
    features.to_parquet(output_path, index=False)

    print(f"\n{'='*80}")
    print("✅ FEATURE STORE PATCHED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nFinal shape: {features.shape}")
    print(f"Columns with funding_Z: {len(features.columns)}")
    print(f"funding_Z coverage: {100 - nan_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Patch funding_Z into feature store')
    parser.add_argument('--input', type=str, required=True, help='Input feature store parquet')
    parser.add_argument('--funding', type=str, default='data/FUNDING_1D.csv', help='Funding CSV')
    parser.add_argument('--output', type=str, help='Output path (default: overwrite input)')

    args = parser.parse_args()

    input_path = Path(args.input)
    funding_path = Path(args.funding)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        raise FileNotFoundError(f"Feature store not found: {input_path}")

    if not funding_path.exists():
        raise FileNotFoundError(f"Funding data not found: {funding_path}")

    patch_feature_store(input_path, funding_path, output_path)


if __name__ == '__main__':
    main()
