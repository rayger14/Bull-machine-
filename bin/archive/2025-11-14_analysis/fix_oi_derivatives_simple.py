#!/usr/bin/env python3
"""
Simple OI Derivatives Fix - Calculate from Existing Data

REALITY CHECK:
- 2024: Has raw OI (100% coverage) → CAN calculate derivatives
- 2022-2023: NO raw OI data → CANNOT calculate (leave as NaN)
- OKX API 404 → Cannot backfill 2022-2023

This script:
1. Loads MTF store
2. Calculates oi_change_24h, oi_change_pct_24h, oi_z for rows with OI
3. Leaves 2022-2023 as NaN (graceful degradation)
4. Creates backup before modifying
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

def calculate_oi_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate OI derivatives for rows that have OI data.

    Handles:
    - oi_change_24h: Absolute change vs 24H ago
    - oi_change_pct_24h: Percentage change vs 24H ago
    - oi_z: Z-score (30-day rolling window)
    """
    print("\n" + "="*80)
    print("Calculating OI Derivatives")
    print("="*80)

    # Only calculate where OI exists
    has_oi = df['oi'].notna()
    n_valid = has_oi.sum()

    print(f"\nOI data availability:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Has OI: {n_valid:,} ({n_valid/len(df)*100:.1f}%)")
    print(f"  Missing OI: {(~has_oi).sum():,}")

    # Calculate absolute change
    df['oi_change_24h'] = df['oi'].diff(24)

    # Calculate percentage change
    df['oi_change_pct_24h'] = df['oi'].pct_change(24) * 100

    # Calculate z-score (30-day rolling)
    rolling_mean = df['oi'].rolling(window=24*30, min_periods=24).mean()
    rolling_std = df['oi'].rolling(window=24*30, min_periods=24).std()
    df['oi_z'] = (df['oi'] - rolling_mean) / rolling_std

    # Report results
    print(f"\nDerivatives calculated:")
    for col in ['oi_change_24h', 'oi_change_pct_24h', 'oi_z']:
        valid = df[col].notna().sum()
        print(f"  {col:20s}: {valid:,} / {len(df):,} ({valid/len(df)*100:.1f}%)")

    return df

def validate_oi_derivatives(df: pd.DataFrame):
    """Validate OI derivatives look reasonable."""
    print("\n" + "="*80)
    print("Validation")
    print("="*80)

    # Check 2024 coverage
    df_2024 = df[df.index.year == 2024]

    print(f"\n2024 OI Derivatives:")
    print(f"  oi_change_pct_24h:")
    print(f"    Mean: {df_2024['oi_change_pct_24h'].mean():.2f}%")
    print(f"    Std: {df_2024['oi_change_pct_24h'].std():.2f}%")
    print(f"    Min: {df_2024['oi_change_pct_24h'].min():.2f}%")
    print(f"    Max: {df_2024['oi_change_pct_24h'].max():.2f}%")

    print(f"\n  oi_z:")
    print(f"    Mean: {df_2024['oi_z'].mean():.3f}")
    print(f"    Std: {df_2024['oi_z'].std():.3f}")
    print(f"    Min: {df_2024['oi_z'].min():.3f}")
    print(f"    Max: {df_2024['oi_z'].max():.3f}")

    # Check for extreme values (potential issues)
    extreme_changes = df_2024['oi_change_pct_24h'].abs() > 50
    if extreme_changes.any():
        print(f"\n⚠️  Warning: {extreme_changes.sum()} rows with extreme OI changes (>50%)")
        print(f"    This might indicate data quality issues")

    # Check distribution
    print(f"\nDistribution check:")
    print(f"  P05: {df_2024['oi_change_pct_24h'].quantile(0.05):.2f}%")
    print(f"  P25: {df_2024['oi_change_pct_24h'].quantile(0.25):.2f}%")
    print(f"  P50: {df_2024['oi_change_pct_24h'].quantile(0.50):.2f}%")
    print(f"  P75: {df_2024['oi_change_pct_24h'].quantile(0.75):.2f}%")
    print(f"  P95: {df_2024['oi_change_pct_24h'].quantile(0.95):.2f}%")

def main():
    print("\n" + "="*80)
    print("OI DERIVATIVES FIX (Simple Version)")
    print("="*80)
    print("\nReality: Only 2024 has OI data, 2022-2023 unavailable (API 404)")
    print("Action: Calculate derivatives for 2024, leave 2022-2023 as NaN")

    # Load MTF store
    mtf_path = Path("data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")

    if not mtf_path.exists():
        print(f"\n❌ ERROR: MTF store not found: {mtf_path}")
        return 1

    print(f"\nLoading: {mtf_path}")
    df = pd.read_parquet(mtf_path)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Check OI column exists
    if 'oi' not in df.columns:
        print(f"\n❌ ERROR: 'oi' column not found in MTF store")
        return 1

    # Create backup
    backup_path = mtf_path.parent / f"{mtf_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    print(f"\nCreating backup: {backup_path.name}")
    shutil.copy2(mtf_path, backup_path)

    # Calculate derivatives
    df = calculate_oi_derivatives(df)

    # Validate
    validate_oi_derivatives(df)

    # Save
    print(f"\n" + "="*80)
    print("Saving Updated MTF Store")
    print("="*80)
    print(f"\nWriting: {mtf_path}")
    df.to_parquet(mtf_path, engine='pyarrow')

    print(f"\n✅ SUCCESS!")
    print(f"\nUpdated MTF store:")
    print(f"  File: {mtf_path}")
    print(f"  Features: {len(df.columns)} (was {len(df.columns) - 3} before derivatives)")
    print(f"  Backup: {backup_path}")

    print(f"\nLimitations:")
    print(f"  ⚠️  2022-2023 OI data unavailable (API 404)")
    print(f"  ⚠️  S5 validation on 2022 will use fallback logic")
    print(f"  ✅ 2024 OI derivatives fully functional")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
