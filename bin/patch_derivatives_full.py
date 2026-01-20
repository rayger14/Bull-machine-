#!/usr/bin/env python3
"""
Patch ALL Derivatives Columns into Feature Store

Adds funding, OI, long/short, and liquidations to feature store.

Usage:
    python3 bin/patch_derivatives_full.py \
        --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \
        --funding data/derivatives/BTC_funding_2022_2023.csv \
        --oi data/derivatives/BTC_oi_2022_2023.csv \
        --ls-ratio data/derivatives/BTC_ls_ratio_2022_2023.csv \
        --liquidations data/derivatives/BTC_liquidations_2022_2023.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def compute_z_score(series: pd.Series, window: int = 252, min_periods: int = 100) -> pd.Series:
    """Compute rolling z-score."""
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    return z.fillna(0)


def patch_all_derivatives(
    input_path: Path,
    funding_path: Path = None,
    oi_path: Path = None,
    ls_ratio_path: Path = None,
    liq_path: Path = None,
    output_path: Path = None
):
    """
    Add all derivatives columns to feature store.

    Args:
        input_path: Path to existing feature store parquet
        funding_path: Path to funding CSV
        oi_path: Path to OI CSV
        ls_ratio_path: Path to long/short ratio CSV
        liq_path: Path to liquidations CSV
        output_path: Output path (defaults to overwriting input)
    """
    print(f"\n{'='*80}")
    print("PATCHING FEATURE STORE WITH ALL DERIVATIVES")
    print(f"{'='*80}")

    # Load existing feature store
    print(f"\n📂 Loading feature store: {input_path}")
    features = pd.read_parquet(input_path)

    if 'timestamp' not in features.columns:
        if features.index.name in ['time', 'timestamp']:
            features = features.reset_index()
            if features.columns[0] in ['time', 'index']:
                features = features.rename(columns={features.columns[0]: 'timestamp'})

    features['timestamp'] = pd.to_datetime(features['timestamp'], utc=True)

    print(f"   Shape: {features.shape}")
    print(f"   Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")

    added_columns = []

    # ========================================================================
    # 1. FUNDING RATE
    # ========================================================================
    if funding_path and funding_path.exists():
        print(f"\n{'='*80}")
        print(f"📊 Processing Funding Rate")
        print(f"{'='*80}")
        print(f"Loading: {funding_path}")

        funding = pd.read_csv(funding_path)
        funding['timestamp'] = pd.to_datetime(funding['timestamp'], utc=True)

        # Resample to hourly (funding is 8-hourly, forward fill)
        funding = funding.set_index('timestamp')['funding_rate'].resample('1h').ffill().reset_index()

        # Compute funding_Z (252h rolling = ~10.5 days)
        funding['funding_Z'] = compute_z_score(funding['funding_rate'], window=252)

        print(f"   Records: {len(funding)}")
        print(f"   Range: {funding['timestamp'].min()} to {funding['timestamp'].max()}")

        # Merge
        features = features.merge(funding, on='timestamp', how='left')

        coverage = features['funding_rate'].notna().sum() / len(features) * 100
        print(f"   Coverage: {coverage:.1f}%")
        print(f"   funding_Z range: [{features['funding_Z'].min():.2f}, {features['funding_Z'].max():.2f}]")

        added_columns.extend(['funding_rate', 'funding_Z'])

    # ========================================================================
    # 2. OPEN INTEREST
    # ========================================================================
    if oi_path and oi_path.exists():
        print(f"\n{'='*80}")
        print(f"📊 Processing Open Interest")
        print(f"{'='*80}")
        print(f"Loading: {oi_path}")

        oi = pd.read_csv(oi_path)
        oi['timestamp'] = pd.to_datetime(oi['timestamp'], utc=True)

        # Compute derived features
        oi['oi_change_24h'] = oi['oi'].diff(24)
        oi['oi_change_pct_24h'] = oi['oi'].pct_change(24) * 100
        oi['oi_Z'] = compute_z_score(oi['oi'], window=252)

        print(f"   Records: {len(oi)}")
        print(f"   Range: {oi['timestamp'].min()} to {oi['timestamp'].max()}")
        print(f"   OI range: ${oi['oi'].min():,.0f} to ${oi['oi'].max():,.0f}")

        # Merge
        features = features.merge(oi, on='timestamp', how='left')

        coverage = features['oi'].notna().sum() / len(features) * 100
        print(f"   Coverage: {coverage:.1f}%")

        added_columns.extend(['oi', 'oi_change_24h', 'oi_change_pct_24h', 'oi_Z'])

    # ========================================================================
    # 3. LONG/SHORT RATIO
    # ========================================================================
    if ls_ratio_path and ls_ratio_path.exists():
        print(f"\n{'='*80}")
        print(f"📊 Processing Long/Short Ratio")
        print(f"{'='*80}")
        print(f"Loading: {ls_ratio_path}")

        ls = pd.read_csv(ls_ratio_path)
        ls['timestamp'] = pd.to_datetime(ls['timestamp'], utc=True)

        # Compute z-score for ratio
        ls['ls_ratio_Z'] = compute_z_score(ls['ls_ratio'], window=252)

        print(f"   Records: {len(ls)}")
        print(f"   Range: {ls['timestamp'].min()} to {ls['timestamp'].max()}")
        print(f"   L/S ratio range: [{ls['ls_ratio'].min():.2f}, {ls['ls_ratio'].max():.2f}]")

        # Merge
        features = features.merge(ls, on='timestamp', how='left')

        coverage = features['ls_ratio'].notna().sum() / len(features) * 100
        print(f"   Coverage: {coverage:.1f}%")

        added_columns.extend(['ls_ratio', 'long_pct', 'short_pct', 'ls_ratio_Z'])

    # ========================================================================
    # 4. LIQUIDATIONS
    # ========================================================================
    if liq_path and liq_path.exists():
        print(f"\n{'='*80}")
        print(f"📊 Processing Liquidations")
        print(f"{'='*80}")
        print(f"Loading: {liq_path}")

        liq = pd.read_csv(liq_path)
        liq['timestamp'] = pd.to_datetime(liq['timestamp'], utc=True)

        # Compute rolling sums (24h) and z-scores
        liq['liq_long_24h'] = liq['liq_long_usd'].rolling(24).sum()
        liq['liq_short_24h'] = liq['liq_short_usd'].rolling(24).sum()
        liq['liq_total_24h'] = liq['liq_total_usd'].rolling(24).sum()

        # Z-scores for hourly liquidations
        liq['liq_long_Z'] = compute_z_score(liq['liq_long_usd'], window=168)  # 1 week
        liq['liq_short_Z'] = compute_z_score(liq['liq_short_usd'], window=168)

        print(f"   Records: {len(liq)}")
        print(f"   Range: {liq['timestamp'].min()} to {liq['timestamp'].max()}")
        print(f"   Total liquidations: ${liq['liq_total_usd'].sum():,.0f}")

        # Merge
        features = features.merge(liq, on='timestamp', how='left')

        coverage = features['liq_total_usd'].notna().sum() / len(features) * 100
        print(f"   Coverage: {coverage:.1f}%")

        added_columns.extend([
            'liq_long_usd', 'liq_short_usd', 'liq_total_usd',
            'liq_long_24h', 'liq_short_24h', 'liq_total_24h',
            'liq_long_Z', 'liq_short_Z'
        ])

    # Save
    if output_path is None:
        output_path = input_path

    print(f"\n{'='*80}")
    print(f"💾 Saving patched feature store")
    print(f"{'='*80}")
    print(f"Output: {output_path}")

    features.to_parquet(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"✅ DERIVATIVES PATCHED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nFinal shape: {features.shape}")
    print(f"Added columns ({len(added_columns)}): {', '.join(added_columns)}")

    # Summary statistics
    print(f"\n📊 Coverage Summary:")
    for col in added_columns:
        if col in features.columns:
            coverage = features[col].notna().sum() / len(features) * 100
            print(f"   {col:25s} {coverage:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Patch all derivatives into feature store')
    parser.add_argument('--input', type=str, required=True, help='Input feature store parquet')
    parser.add_argument('--funding', type=str, help='Funding CSV file')
    parser.add_argument('--oi', type=str, help='OI CSV file')
    parser.add_argument('--ls-ratio', type=str, help='Long/Short ratio CSV file')
    parser.add_argument('--liquidations', type=str, help='Liquidations CSV file')
    parser.add_argument('--output', type=str, help='Output path (default: overwrite input)')

    args = parser.parse_args()

    input_path = Path(args.input)
    funding_path = Path(args.funding) if args.funding else None
    oi_path = Path(args.oi) if args.oi else None
    ls_ratio_path = Path(args.ls_ratio) if args.ls_ratio else None
    liq_path = Path(args.liquidations) if args.liquidations else None
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        raise FileNotFoundError(f"Feature store not found: {input_path}")

    patch_all_derivatives(input_path, funding_path, oi_path, ls_ratio_path, liq_path, output_path)


if __name__ == '__main__':
    main()
