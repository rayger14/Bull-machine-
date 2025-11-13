#!/usr/bin/env python3
"""
Patch Funding Rate from TradingView Exports

Takes 4H and 1H TradingView exports and patches feature stores.

Usage:
    python3 bin/patch_funding_from_tv.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_z_score(series: pd.Series, window: int = 252, min_periods: int = 100) -> pd.Series:
    """Compute rolling z-score."""
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    return z.fillna(0)


def load_tv_funding(path_4h: Path, path_1h: Path) -> pd.DataFrame:
    """
    Load and merge TradingView funding rate exports.

    Args:
        path_4h: 4H funding data (2022-2025)
        path_1h: 1H funding data (2024-2025)

    Returns:
        Combined 1H funding rate DataFrame
    """
    print(f"\n{'='*80}")
    print(f"Loading TradingView Funding Rate Data")
    print(f"{'='*80}")

    # Load 4H data
    print(f"\n📊 Loading 4H data: {path_4h.name}")
    df_4h = pd.read_csv(path_4h)
    df_4h['timestamp'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h = df_4h[['timestamp', 'close']].rename(columns={'close': 'funding_rate'})
    df_4h = df_4h.sort_values('timestamp').reset_index(drop=True)

    print(f"   Records: {len(df_4h)}")
    print(f"   Range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")

    # Resample 4H to 1H (forward-fill)
    print(f"\n⚡ Resampling 4H → 1H (forward-fill)...")
    df_4h_1h = df_4h.set_index('timestamp').resample('1h').ffill().reset_index()
    print(f"   Resampled to {len(df_4h_1h)} hourly records")

    # Load 1H data
    print(f"\n📊 Loading 1H data: {path_1h.name}")
    df_1h = pd.read_csv(path_1h)
    df_1h['timestamp'] = pd.to_datetime(df_1h['time'], unit='s', utc=True)
    df_1h = df_1h[['timestamp', 'close']].rename(columns={'close': 'funding_rate'})
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)

    print(f"   Records: {len(df_1h)}")
    print(f"   Range: {df_1h['timestamp'].min()} to {df_1h['timestamp'].max()}")

    # Merge: Use 1H data where available, 4H elsewhere
    print(f"\n🔗 Merging datasets...")

    # Split 4H data: before 2024
    df_4h_pre2024 = df_4h_1h[df_4h_1h['timestamp'] < '2024-01-01'].copy()

    # Combine
    df_combined = pd.concat([df_4h_pre2024, df_1h], ignore_index=True)
    df_combined = df_combined.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

    print(f"   Combined: {len(df_combined)} records")
    print(f"   Final range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")

    # Compute z-score
    print(f"\n📈 Computing funding_Z (252h rolling)...")
    df_combined['funding_Z'] = compute_z_score(df_combined['funding_rate'], window=252)

    print(f"   funding_rate range: [{df_combined['funding_rate'].min():.6f}, {df_combined['funding_rate'].max():.6f}]")
    print(f"   funding_Z range: [{df_combined['funding_Z'].min():.2f}, {df_combined['funding_Z'].max():.2f}]")

    return df_combined


def patch_feature_store(store_path: Path, funding_df: pd.DataFrame) -> None:
    """Patch funding_Z into feature store."""

    print(f"\n{'='*80}")
    print(f"Patching Feature Store: {store_path.name}")
    print(f"{'='*80}")

    # Load feature store
    print(f"\n📂 Loading feature store...")
    features = pd.read_parquet(store_path)

    if 'timestamp' not in features.columns:
        if features.index.name in ['time', 'timestamp']:
            features = features.reset_index()
            if features.columns[0] in ['time', 'index']:
                features = features.rename(columns={features.columns[0]: 'timestamp'})

    features['timestamp'] = pd.to_datetime(features['timestamp'], utc=True)

    print(f"   Shape: {features.shape}")
    print(f"   Range: {features['timestamp'].min()} to {features['timestamp'].max()}")

    # Check existing funding_Z
    if 'funding_Z' in features.columns:
        old_nonzero = (features['funding_Z'] != 0.0).sum()
        old_coverage = old_nonzero / len(features) * 100
        print(f"\n⚠️  Existing funding_Z: {old_coverage:.1f}% non-zero ({old_nonzero}/{len(features)} bars)")
        print(f"   Will be REPLACED with new data")

    # Merge funding data
    print(f"\n🔗 Merging funding data...")

    # Drop old funding columns if they exist
    funding_cols = ['funding_rate', 'funding_Z']
    for col in funding_cols:
        if col in features.columns:
            features = features.drop(columns=[col])

    # Merge new data
    features = features.merge(
        funding_df[['timestamp', 'funding_rate', 'funding_Z']],
        on='timestamp',
        how='left'
    )

    # Fill NaN with 0 (for any missing timestamps)
    features['funding_rate'] = features['funding_rate'].fillna(0)
    features['funding_Z'] = features['funding_Z'].fillna(0)

    # Check coverage
    nonzero = (features['funding_Z'] != 0.0).sum()
    coverage = nonzero / len(features) * 100

    print(f"\n✅ New funding_Z coverage: {coverage:.1f}% ({nonzero}/{len(features)} bars)")
    print(f"   Range: [{features['funding_Z'].min():.2f}, {features['funding_Z'].max():.2f}]")
    print(f"   Mean: {features['funding_Z'].mean():.2f}, Std: {features['funding_Z'].std():.2f}")

    # Save
    print(f"\n💾 Saving patched feature store...")
    features.to_parquet(store_path, index=False)

    print(f"   ✅ Saved: {store_path}")


def main():
    """Patch both 2022-2023 and 2024 feature stores."""

    print(f"\n{'='*80}")
    print(f"🚀 PATCHING FUNDING RATE FROM TRADINGVIEW")
    print(f"{'='*80}")

    # Paths
    tv_4h = Path("/Users/raymondghandchi/Downloads/BINANCE_BTCUSDT_PREMIUM, 240_ef612.csv")
    tv_1h = Path("/Users/raymondghandchi/Downloads/BINANCE_BTCUSDT_PREMIUM, 60_44cd0.csv")

    store_2022 = Path("data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet")
    store_2024 = Path("data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet")

    # Check files exist
    for path in [tv_4h, tv_1h, store_2022, store_2024]:
        if not path.exists():
            print(f"❌ File not found: {path}")
            return 1

    # Load and merge TradingView data
    funding_df = load_tv_funding(tv_4h, tv_1h)

    # Patch 2022-2023 store
    patch_feature_store(store_2022, funding_df)

    # Patch 2024 store
    patch_feature_store(store_2024, funding_df)

    print(f"\n{'='*80}")
    print(f"✅ FUNDING RATE PATCHING COMPLETE")
    print(f"{'='*80}")
    print(f"\nBoth feature stores now have full funding_Z coverage:")
    print(f"  - 2022-2023: data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet")
    print(f"  - 2024:      data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet")

    print(f"\nNext steps:")
    print(f"  1. For OI (optional): Export BINANCE:BTCUSDT.P OI indicator")
    print(f"  2. Validate: python3 bin/validate_feature_store_v10.py --input <store>")
    print(f"  3. Run backtests with Router v10")

    return 0


if __name__ == '__main__':
    exit(main())
