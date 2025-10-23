#!/usr/bin/env python3
"""
Patch specific columns in an existing feature store without rebuilding everything.

Usage:
    python bin/patch_feature_columns.py --asset BTC --year 2024 \
        --cols tf4h_boms_displacement,tf1d_boms_strength,tf4h_fusion_score

This loads the existing feature store, recomputes ONLY the specified columns,
and atomically replaces them while preserving all other columns.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.structure.boms_detector import detect_boms
from engine.structure.pti_detector import detect_pti


def load_feature_store(asset: str, year: int) -> pd.DataFrame:
    """Load existing feature store."""
    path = f"data/features_mtf/{asset}_1H_{year}-01-01_to_{year}-12-31.parquet"
    if not Path(path).exists():
        raise FileNotFoundError(f"Feature store not found: {path}")

    print(f"Loading feature store: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_raw_ohlcv(asset: str, start: str, end: str) -> pd.DataFrame:
    """Load raw OHLCV data for recomputation."""
    # This would load from your raw data source
    # For now, placeholder - you'd implement based on your data pipeline
    raise NotImplementedError("Implement your OHLCV loading logic here")


def patch_boms_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute BOMS displacement columns with fixed calculation."""
    print("\nPatching BOMS displacement columns...")

    # Need to resample to 4H and run BOMS detector
    # This is a simplified example - you'd need proper resampling logic
    from engine.data.resample import resample_to_timeframe

    df_4h = resample_to_timeframe(df, '4H')

    for idx in range(len(df)):
        # Get 4H window for this 1H bar
        # Simplified - you'd need proper alignment logic
        timestamp = df.index[idx]
        window_4h = df_4h[df_4h.index <= timestamp].tail(100)

        if len(window_4h) >= 30:
            boms_4h = detect_boms(window_4h, timeframe='4H')
            df.loc[df.index[idx], 'tf4h_boms_displacement'] = boms_4h.displacement

    non_zero_pct = (df['tf4h_boms_displacement'] > 0).mean() * 100
    print(f"  ✓ tf4h_boms_displacement: {non_zero_pct:.1f}% non-zero")

    return df


def patch_boms_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute BOMS strength with proper normalization."""
    print("\nPatching BOMS strength columns...")

    # Normalize displacement to 0.0-1.0 range
    df_1d = resample_to_timeframe(df, '1D')

    for idx in range(len(df)):
        timestamp = df.index[idx]
        window_1d = df_1d[df_1d.index <= timestamp].tail(100)

        if len(window_1d) >= 30:
            boms_1d = detect_boms(window_1d, timeframe='1D')

            # Calculate ATR
            atr_1d = window_1d['close'].pct_change().abs().rolling(14).mean().iloc[-1] * window_1d['close'].iloc[-1]

            if atr_1d > 0 and boms_1d.displacement > 0:
                strength = min(boms_1d.displacement / (2.0 * atr_1d), 1.0)
                df.loc[df.index[idx], 'tf1d_boms_strength'] = strength

    non_zero_pct = (df['tf1d_boms_strength'] > 0).mean() * 100
    print(f"  ✓ tf1d_boms_strength: {non_zero_pct:.1f}% non-zero")

    return df


def patch_pti_trap_type(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute PTI trap type classification."""
    print("\nPatching PTI trap type column...")

    for idx in range(len(df)):
        window_1h = df.iloc[max(0, idx-100):idx+1]

        if len(window_1h) >= 30:
            pti_result = detect_pti(window_1h, timeframe='1H')

            # Classify trap type based on PTI score and components
            if pti_result.score > 0.6:
                # Simple classification - enhance based on your PTI detector output
                trap_type = classify_trap_type(pti_result, window_1h)
                df.loc[df.index[idx], 'tf1h_pti_trap_type'] = trap_type

    non_none_pct = (df['tf1h_pti_trap_type'] != 'none').mean() * 100
    print(f"  ✓ tf1h_pti_trap_type: {non_none_pct:.1f}% classified")

    return df


def patch_tf4h_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute tf4h_fusion_score from components."""
    print("\nPatching tf4h_fusion_score column...")

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

    non_zero_pct = (df['tf4h_fusion_score'] > 0).mean() * 100
    print(f"  ✓ tf4h_fusion_score: {non_zero_pct:.1f}% non-zero")

    return df


def classify_trap_type(pti_result, window):
    """Classify PTI trap type - placeholder for your logic."""
    # Implement based on your PTI detector output structure
    return 'none'


def validate_patch(df: pd.DataFrame, cols: list):
    """Validate patched columns meet health checks."""
    print("\nValidating patched columns...")

    health_checks = {
        'tf4h_boms_displacement': lambda s: (s > 0).mean() > 0.10,
        'tf1d_boms_strength': lambda s: (s > 0).mean() > 0.10,
        'tf4h_fusion_score': lambda s: (s != 0).mean() > 0.25,
        'tf1h_pti_trap_type': lambda s: (s != 'none').mean() > 0.05,
    }

    for col in cols:
        if col in health_checks and col in df.columns:
            if not health_checks[col](df[col]):
                raise ValueError(f"Health check FAILED for {col} - still mostly zeros/defaults!")
            print(f"  ✓ {col} passed health check")


def atomic_save(df: pd.DataFrame, asset: str, year: int, cols_patched: list):
    """Atomically save patched feature store."""
    path = f"data/features_mtf/{asset}_1H_{year}-01-01_to_{year}-12-31.parquet"
    tmp_path = f"{path}.tmp"
    backup_path = f"{path}.backup"

    print(f"\nSaving patched feature store...")

    # Add metadata about patch
    df.attrs['__patched_cols'] = ','.join(cols_patched)
    df.attrs['__patched_at'] = datetime.now().isoformat()

    # Write to temp file
    df.to_parquet(tmp_path)

    # Backup original
    if Path(path).exists():
        Path(path).rename(backup_path)

    # Atomic replace
    Path(tmp_path).rename(path)

    print(f"  ✓ Saved to {path}")
    print(f"  ✓ Backup at {backup_path}")


def main():
    parser = argparse.ArgumentParser(description='Patch specific columns in feature store')
    parser.add_argument('--asset', required=True, help='Asset symbol (BTC, ETH, etc.)')
    parser.add_argument('--year', type=int, required=True, help='Year (2024, etc.)')
    parser.add_argument('--cols', required=True, help='Comma-separated column names to patch')

    args = parser.parse_args()

    cols_to_patch = [c.strip() for c in args.cols.split(',')]

    print(f"=== Patching Feature Store ===")
    print(f"Asset: {args.asset}")
    print(f"Year: {args.year}")
    print(f"Columns: {cols_to_patch}")

    # Load existing feature store
    df = load_feature_store(args.asset, args.year)

    # Patch requested columns
    patch_funcs = {
        'tf4h_boms_displacement': patch_boms_displacement,
        'tf1d_boms_strength': patch_boms_strength,
        'tf1h_pti_trap_type': patch_pti_trap_type,
        'tf4h_fusion_score': patch_tf4h_fusion,
    }

    for col in cols_to_patch:
        if col in patch_funcs:
            df = patch_funcs[col](df)
        else:
            print(f"⚠️  No patch function for '{col}' - skipping")

    # Validate
    validate_patch(df, cols_to_patch)

    # Save atomically
    atomic_save(df, args.asset, args.year, cols_to_patch)

    print("\n✅ Patch complete!")


if __name__ == '__main__':
    main()
