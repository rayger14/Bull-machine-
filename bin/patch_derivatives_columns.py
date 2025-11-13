#!/usr/bin/env python3
"""
Patch Derivatives Columns into Existing Feature Store

Adds OI, funding, and derived features to feature store without full rebuild.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_oi_data(oi_path: Path) -> pd.DataFrame:
    """Load and normalize OI data."""
    print(f"\n📊 Loading OI data: {oi_path}")

    if not oi_path.exists():
        print(f"   ⚠️  File not found, skipping OI")
        return None

    oi = pd.read_csv(oi_path)

    # Convert Unix timestamp to datetime if needed
    if 'time' in oi.columns:
        oi['timestamp'] = pd.to_datetime(oi['time'], unit='s', utc=True)
        oi = oi[['timestamp', 'close']].rename(columns={'close': 'oi'})
    elif 'timestamp' in oi.columns:
        oi['timestamp'] = pd.to_datetime(oi['timestamp'], utc=True)
        oi = oi[['timestamp', 'oi']]

    oi = oi.sort_values('timestamp').reset_index(drop=True)

    print(f"   Records: {len(oi)}")
    print(f"   Date range: {oi['timestamp'].min()} to {oi['timestamp'].max()}")
    print(f"   OI range: {oi['oi'].min():.0f} to {oi['oi'].max():.0f}")

    return oi


def compute_oi_features(oi_df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Compute OI-derived features.

    Args:
        oi_df: DataFrame with timestamp and oi columns
        window: Rolling window for z-score (default 252 hours = ~10.5 days)

    Returns:
        DataFrame with OI features
    """
    df = oi_df.copy()

    # OI change over 24 hours
    df['oi_change_24h'] = df['oi'].diff(24)
    df['oi_change_pct_24h'] = df['oi'].pct_change(24) * 100

    # OI z-score (252h rolling window)
    rolling_mean = df['oi'].rolling(window=window, min_periods=100).mean()
    rolling_std = df['oi'].rolling(window=window, min_periods=100).std()
    rolling_std = rolling_std.replace(0, np.nan)

    df['oi_z'] = (df['oi'] - rolling_mean) / rolling_std
    df['oi_z'] = df['oi_z'].fillna(0)

    return df[['timestamp', 'oi', 'oi_change_24h', 'oi_change_pct_24h', 'oi_z']]


def patch_derivatives(input_path: Path, oi_path: Path = None, output_path: Path = None):
    """
    Add derivatives columns to existing feature store.

    Args:
        input_path: Path to existing feature store parquet
        oi_path: Path to OI CSV file
        output_path: Output path (defaults to overwriting input)
    """
    print(f"\n{'='*80}")
    print("PATCHING FEATURE STORE WITH DERIVATIVES")
    print(f"{'='*80}")

    # Load existing feature store
    print(f"\n📂 Loading feature store: {input_path}")
    features = pd.read_parquet(input_path)

    # Ensure timestamp column exists
    if 'timestamp' not in features.columns:
        if features.index.name in ['time', 'timestamp']:
            features = features.reset_index()
            if features.columns[0] in ['time', 'index']:
                features = features.rename(columns={features.columns[0]: 'timestamp'})

    print(f"   Shape: {features.shape}")
    print(f"   Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")

    added_columns = []

    # Add OI features
    if oi_path and oi_path.exists():
        oi_data = load_oi_data(oi_path)

        if oi_data is not None:
            print(f"\n🔢 Computing OI features...")
            oi_features = compute_oi_features(oi_data)

            print(f"\n🔗 Merging OI features...")
            features = features.merge(
                oi_features,
                on='timestamp',
                how='left'
            )

            # Check coverage
            oi_coverage = features['oi'].notna().sum() / len(features) * 100
            oi_z_min = features['oi_z'].min()
            oi_z_max = features['oi_z'].max()
            print(f"   OI coverage: {oi_coverage:.1f}%")
            print(f"   oi_z range: [{oi_z_min:.2f}, {oi_z_max:.2f}]")

            added_columns.extend(['oi', 'oi_change_24h', 'oi_change_pct_24h', 'oi_z'])

    # Save
    if output_path is None:
        output_path = input_path

    print(f"\n💾 Saving patched feature store: {output_path}")
    features.to_parquet(output_path, index=False)

    print(f"\n{'='*80}")
    print("✅ DERIVATIVES PATCHED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nFinal shape: {features.shape}")
    print(f"Added columns: {', '.join(added_columns) if added_columns else 'none'}")


def main():
    parser = argparse.ArgumentParser(description='Patch derivatives into feature store')
    parser.add_argument('--input', type=str, required=True, help='Input feature store parquet')
    parser.add_argument('--oi', type=str, help='OI CSV file')
    parser.add_argument('--output', type=str, help='Output path (default: overwrite input)')

    args = parser.parse_args()

    input_path = Path(args.input)
    oi_path = Path(args.oi) if args.oi else None
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        raise FileNotFoundError(f"Feature store not found: {input_path}")

    patch_derivatives(input_path, oi_path, output_path)


if __name__ == '__main__':
    main()
