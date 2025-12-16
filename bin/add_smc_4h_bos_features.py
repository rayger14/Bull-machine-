#!/usr/bin/env python3
"""
Add SMC 4H Break of Structure (BOS) Features
============================================

Generates:
- tf4h_bos_bearish: 4H bearish break of structure (institutional sell signal)
- tf4h_bos_bullish: 4H bullish break of structure (institutional buy signal)

BOS Detection Logic:
- Bullish BOS: Price breaks above recent swing high with volume confirmation
- Bearish BOS: Price breaks below recent swing low with volume confirmation
- Uses 20-period lookback for swing identification
- Requires volume > 20-period average for confirmation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def detect_bos_vectorized(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    Detect Break of Structure (BOS) using vectorized operations.

    Returns:
        (bos_bullish, bos_bearish): Two boolean Series
    """
    # Calculate swing highs/lows over lookback window
    swing_high = df['high'].rolling(window=lookback, min_periods=1).max()
    swing_low = df['low'].rolling(window=lookback, min_periods=1).min()

    # Volume confirmation: current volume > rolling average
    vol_avg = df['volume'].rolling(window=lookback, min_periods=1).mean()
    vol_confirm = df['volume'] > vol_avg

    # Bullish BOS: Break above swing high with volume
    bos_bullish = (df['close'] > swing_high.shift(1)) & vol_confirm

    # Bearish BOS: Break below swing low with volume
    bos_bearish = (df['close'] < swing_low.shift(1)) & vol_confirm

    return bos_bullish.fillna(False), bos_bearish.fillna(False)


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1H data to 4H for BOS detection.

    Args:
        df_1h: 1H DataFrame with OHLCV data

    Returns:
        4H DataFrame with OHLCV
    """
    # Ensure timestamp index
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        df_1h = df_1h.set_index('timestamp')

    # Resample to 4H
    df_4h = pd.DataFrame({
        'open': df_1h['open'].resample('4H').first(),
        'high': df_1h['high'].resample('4H').max(),
        'low': df_1h['low'].resample('4H').min(),
        'close': df_1h['close'].resample('4H').last(),
        'volume': df_1h['volume'].resample('4H').sum()
    })

    return df_4h.dropna()


def forward_fill_to_1h(df_4h_feature: pd.Series, df_1h_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward fill 4H feature values to 1H timestamps.

    Args:
        df_4h_feature: 4H feature series
        df_1h_index: Target 1H index

    Returns:
        1H series with forward-filled values
    """
    # Reindex to 1H and forward fill
    return df_4h_feature.reindex(df_1h_index, method='ffill').fillna(False)


def add_4h_bos_features(mtf_store_path: str, dry_run: bool = False) -> dict:
    """
    Add tf4h_bos_bearish and tf4h_bos_bullish to MTF feature store.

    Args:
        mtf_store_path: Path to MTF parquet file
        dry_run: If True, compute but don't write

    Returns:
        dict with statistics
    """
    print(f"Loading MTF store: {mtf_store_path}")
    df = pd.read_parquet(mtf_store_path)

    original_shape = df.shape
    print(f"Original shape: {original_shape}")

    # Check if features already exist
    if 'tf4h_bos_bearish' in df.columns and 'tf4h_bos_bullish' in df.columns:
        print("⚠️  WARNING: tf4h_bos features already exist!")
        print("   Recalculating and overwriting...")

    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure timestamp index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("Cannot find timestamp column or index")

    print("\n1. Resampling 1H data to 4H...")
    df_4h = resample_to_4h(df)
    print(f"   Created 4H DataFrame: {df_4h.shape}")

    print("\n2. Detecting BOS on 4H timeframe...")
    bos_4h_bullish, bos_4h_bearish = detect_bos_vectorized(df_4h, lookback=20)

    bullish_events = bos_4h_bullish.sum()
    bearish_events = bos_4h_bearish.sum()
    print(f"   Found {bullish_events} bullish BOS events (4H)")
    print(f"   Found {bearish_events} bearish BOS events (4H)")

    print("\n3. Forward-filling 4H signals to 1H timestamps...")
    df['tf4h_bos_bullish'] = forward_fill_to_1h(bos_4h_bullish, df.index)
    df['tf4h_bos_bearish'] = forward_fill_to_1h(bos_4h_bearish, df.index)

    # Convert to boolean type
    df['tf4h_bos_bullish'] = df['tf4h_bos_bullish'].astype(bool)
    df['tf4h_bos_bearish'] = df['tf4h_bos_bearish'].astype(bool)

    bullish_1h = df['tf4h_bos_bullish'].sum()
    bearish_1h = df['tf4h_bos_bearish'].sum()
    print(f"   Propagated to {bullish_1h} bullish 1H rows")
    print(f"   Propagated to {bearish_1h} bearish 1H rows")

    stats = {
        'original_shape': original_shape,
        'new_shape': df.shape,
        'bullish_4h_events': int(bullish_events),
        'bearish_4h_events': int(bearish_events),
        'bullish_1h_rows': int(bullish_1h),
        'bearish_1h_rows': int(bearish_1h),
        'total_rows': len(df)
    }

    if not dry_run:
        print(f"\n4. Writing updated MTF store...")
        # Create backup first
        backup_path = mtf_store_path.replace('.parquet', '_backup_bos.parquet')
        import shutil
        shutil.copy2(mtf_store_path, backup_path)
        print(f"   Backup created: {backup_path}")

        df.to_parquet(mtf_store_path)
        print(f"   ✅ Written {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        print(f"\n4. DRY RUN - Not writing to disk")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add SMC 4H BOS features to MTF store')
    parser.add_argument(
        '--mtf-store',
        default='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet',
        help='Path to MTF feature store'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Compute but do not write to disk'
    )

    args = parser.parse_args()

    # Resolve path relative to project root
    project_root = Path(__file__).parent.parent
    mtf_path = project_root / args.mtf_store

    if not mtf_path.exists():
        print(f"❌ ERROR: MTF store not found: {mtf_path}")
        return 1

    try:
        stats = add_4h_bos_features(str(mtf_path), dry_run=args.dry_run)

        print("\n" + "="*60)
        print("SMC 4H BOS FEATURE GENERATION COMPLETE")
        print("="*60)
        print(f"Original columns: {stats['original_shape'][1]}")
        print(f"New columns: {stats['new_shape'][1]}")
        print(f"Added: +{stats['new_shape'][1] - stats['original_shape'][1]} columns")
        print(f"\nBullish BOS (4H): {stats['bullish_4h_events']} events")
        print(f"Bearish BOS (4H): {stats['bearish_4h_events']} events")
        print(f"Coverage: {stats['total_rows']} total rows")

        if not args.dry_run:
            print(f"\n✅ STATUS: Features written to MTF store")
        else:
            print(f"\n⚠️  STATUS: DRY RUN - No changes written")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
