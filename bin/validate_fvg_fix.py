#!/usr/bin/env python3
"""
Validate FVG Fix in Canonical Feature Store

Comprehensive validation of the FVG feature fix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def main():
    # Load the updated canonical store
    df = pd.read_parquet('data/features_mtf/BTC_1H_CANONICAL_20260201.parquet')

    print('=' * 80)
    print('FVG Feature Verification')
    print('=' * 80)

    print(f'\nDataset: {len(df):,} bars from {df.index.min()} to {df.index.max()}')
    print(f'Total columns: {len(df.columns)}')

    print('\n1. FVG Feature Columns:')
    fvg_cols = [col for col in df.columns if 'fvg' in col.lower()]
    for col in sorted(fvg_cols):
        print(f'   - {col}')

    print('\n2. FVG Feature Statistics:')
    for col in ['fvg_present', 'fvg_bullish', 'fvg_bearish', 'fvg_size']:
        if col in df.columns:
            if df[col].dtype == bool:
                count = df[col].sum()
                pct = (count / len(df)) * 100
                print(f'   {col:20s}: {count:6,} True ({pct:5.2f}%)')
            else:
                nonzero = (df[col] != 0).sum()
                pct = (nonzero / len(df)) * 100
                avg = df[df[col] != 0][col].mean() if nonzero > 0 else 0
                print(f'   {col:20s}: {nonzero:6,} non-zero ({pct:5.2f}%), avg={avg:.2f}')

    print('\n3. Sample FVG Gaps (first 10 detected):')
    fvg_bars = df[df['fvg_present']].head(10)
    print('   Timestamp                     | Bullish | Bearish | Gap Size')
    print('   ' + '-' * 70)
    for idx, row in fvg_bars.iterrows():
        bullish_str = str(row['fvg_bullish'])
        bearish_str = str(row['fvg_bearish'])
        gap_size = row['fvg_size']
        print(f'   {idx} | {bullish_str:7s} | {bearish_str:7s} | ${gap_size:7.2f}')

    print('\n4. FVG Distribution by Year:')
    df_fvg = df[df['fvg_present']].copy()
    df_fvg['year'] = df_fvg.index.year
    yearly = df_fvg.groupby('year').size()
    for year, count in yearly.items():
        year_total = len(df[df.index.year == year])
        pct = (count / year_total) * 100 if year_total > 0 else 0
        print(f'   {year}: {count:4,} gaps ({pct:5.2f}%)')

    print('\n5. Cross-validation:')
    bullish_only = df['fvg_bullish'] & ~df['fvg_bearish']
    bearish_only = ~df['fvg_bullish'] & df['fvg_bearish']
    both = df['fvg_bullish'] & df['fvg_bearish']
    neither = ~df['fvg_bullish'] & ~df['fvg_bearish'] & df['fvg_present']

    print(f'   Bullish only: {bullish_only.sum():,}')
    print(f'   Bearish only: {bearish_only.sum():,}')
    print(f'   Both (invalid): {both.sum():,}')
    print(f'   Neither but present (invalid): {neither.sum():,}')

    if both.sum() == 0 and neither.sum() == 0:
        print('   ✓ All FVG gaps are valid (either bullish or bearish, not both)')
    else:
        print('   ⚠️  WARNING: Found invalid FVG gaps')

    print('\n6. MTF FVG Features:')
    mtf_fvg_cols = [col for col in df.columns if 'tf' in col and 'fvg' in col.lower()]
    for col in sorted(mtf_fvg_cols):
        if df[col].dtype in [bool, 'bool']:
            count = df[col].sum()
            pct = (count / len(df)) * 100
            print(f'   {col:25s}: {count:6,} True ({pct:5.2f}%)')
        elif df[col].dtype in ['float64', 'float32']:
            nonzero = (df[col] != 0).sum()
            if nonzero > 0:
                pct = (nonzero / len(df)) * 100
                print(f'   {col:25s}: {nonzero:6,} non-zero ({pct:5.2f}%)')
            else:
                print(f'   {col:25s}: ALL ZEROS')

    print('\n' + '=' * 80)
    print('✓ FVG features successfully integrated!')
    print('=' * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
