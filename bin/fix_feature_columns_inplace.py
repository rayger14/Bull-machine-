#!/usr/bin/env python3
"""
Fix broken feature columns in existing BTC 2024 feature store.

This loads the existing 80-column feature store and recomputes ONLY the P0 columns:
- tf4h_boms_displacement
- tf1d_boms_strength
- tf4h_fusion_score

Much faster than full rebuild since we're just updating specific columns in-place.
PTI trap types are skipped - we'll use existing PTI score numerically.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.structure.boms_detector import detect_boms

def load_feature_store():
    path = "data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet"
    print(f"Loading: {path}")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    return df

def resample_to_timeframe(df_1h, timeframe):
    """Quick resample helper"""
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df_1h.resample(timeframe).agg(agg_dict).dropna()

def fix_boms_displacement(df):
    """Fix tf4h_boms_displacement"""
    print("\n[1/3] Fixing tf4h_boms_displacement...")

    # Resample to 4H
    df_4h = resample_to_timeframe(df[['open', 'high', 'low', 'close', 'volume']], '4H')

    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"  Processing bar {idx}/{len(df)}...")

        timestamp = df.index[idx]
        window_4h = df_4h[df_4h.index <= timestamp].tail(100)

        if len(window_4h) >= 30:
            boms_4h = detect_boms(window_4h, timeframe='4H')
            df.iloc[idx, df.columns.get_loc('tf4h_boms_displacement')] = boms_4h.displacement

    non_zero = (df['tf4h_boms_displacement'] > 0).sum()
    print(f"  ✓ Non-zero: {non_zero}/{len(df)} ({non_zero/len(df)*100:.1f}%)")
    return df

def fix_boms_strength(df):
    """Fix tf1d_boms_strength"""
    print("\n[2/3] Fixing tf1d_boms_strength...")

    # Resample to 1D
    df_1d = resample_to_timeframe(df[['open', 'high', 'low', 'close', 'volume']], '1D')

    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"  Processing bar {idx}/{len(df)}...")

        timestamp = df.index[idx]
        window_1d = df_1d[df_1d.index <= timestamp].tail(100)

        if len(window_1d) >= 30:
            boms_1d = detect_boms(window_1d, timeframe='1D')

            # Calculate ATR
            atr_1d = window_1d['close'].pct_change().abs().rolling(14).mean().iloc[-1] * window_1d['close'].iloc[-1]

            if atr_1d > 0 and boms_1d.displacement > 0:
                strength = min(boms_1d.displacement / (2.0 * atr_1d), 1.0)
                df.iloc[idx, df.columns.get_loc('tf1d_boms_strength')] = strength

    non_zero = (df['tf1d_boms_strength'] > 0).sum()
    print(f"  ✓ Non-zero: {non_zero}/{len(df)} ({non_zero/len(df)*100:.1f}%)")
    return df

def fix_tf4h_fusion(df):
    """Fix tf4h_fusion_score"""
    print("\n[3/3] Fixing tf4h_fusion_score...")

    # Calculate from available 4H features
    tf4h_fusion = pd.Series(0.0, index=df.index)

    if 'tf4h_structure_alignment' in df.columns:
        tf4h_fusion += df['tf4h_structure_alignment'].astype(float) * 0.30

    if 'tf4h_squiggle_entry_window' in df.columns:
        tf4h_fusion += df['tf4h_squiggle_entry_window'].astype(float) * 0.20

        if 'tf4h_squiggle_confidence' in df.columns:
            tf4h_fusion += df['tf4h_squiggle_confidence'] * 0.20

    if 'tf4h_choch_flag' in df.columns:
        tf4h_fusion += df['tf4h_choch_flag'].astype(float) * 0.30

    df['tf4h_fusion_score'] = tf4h_fusion.clip(upper=1.0)

    non_zero = (df['tf4h_fusion_score'] > 0).sum()
    print(f"  ✓ Non-zero: {non_zero}/{len(df)} ({non_zero/len(df)*100:.1f}%)")
    return df

def save_with_backup(df):
    """Save with backup"""
    path = "data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet"
    backup_path = "data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet.backup"

    print(f"\nSaving...")
    print(f"  Backup: {backup_path}")
    Path(path).rename(backup_path)

    print(f"  Writing: {path}")
    df.to_parquet(path)

    print(f"✅ Complete! Shape: {df.shape}")

def main():
    print("=" * 60)
    print("P0 Column Patcher - BTC 2024 Feature Store")
    print("=" * 60)
    print("Patching only critical columns:")
    print("  1. tf4h_boms_displacement")
    print("  2. tf1d_boms_strength")
    print("  3. tf4h_fusion_score")
    print()
    print("Skipping PTI trap types - using existing PTI score numerically")
    print("=" * 60)
    print()

    df = load_feature_store()

    df = fix_boms_displacement(df)
    df = fix_boms_strength(df)
    df = fix_tf4h_fusion(df)

    save_with_backup(df)

if __name__ == '__main__':
    main()
