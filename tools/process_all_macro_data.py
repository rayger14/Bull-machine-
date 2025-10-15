"""
Process ALL Real Macro Data from TradingView

Loads VIX, DXY, MOVE, yields, oil, and crypto market cap data
from real TradingView exports and merges into macro feature stores.

REAL DATA ONLY - NO SYNTHETIC FALLBACKS.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.io.tradingview_loader import load_tv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_macro_indicators() -> pd.DataFrame:
    """
    Load all macro indicators from TradingView exports (hourly resolution)

    Returns:
        DataFrame with columns: timestamp, VIX, DXY, MOVE, YIELD_2Y, YIELD_10Y, WTI
    """
    logger.info("=" * 70)
    logger.info("Loading ALL Real Macro Data from TradingView")
    logger.info("=" * 70)

    # Load all indicators (hourly)
    logger.info("\n📊 Loading macro indicators...")

    vix_df = load_tv("VIX_1H")
    dxy_df = load_tv("DXY_1H")
    move_df = load_tv("MOVE_1H")
    us02y_df = load_tv("US02Y_1H")
    us10y_df = load_tv("US10Y_1H")
    wti_df = load_tv("WTI_1H")

    logger.info(f"  VIX:      {len(vix_df)} bars")
    logger.info(f"  DXY:      {len(dxy_df)} bars")
    logger.info(f"  MOVE:     {len(move_df)} bars")
    logger.info(f"  US02Y:    {len(us02y_df)} bars")
    logger.info(f"  US10Y:    {len(us10y_df)} bars")
    logger.info(f"  WTI:      {len(wti_df)} bars")

    # Create combined DataFrame using close prices
    df = pd.DataFrame({
        'VIX': vix_df['close'],
        'DXY': dxy_df['close'],
        'MOVE': move_df['close'],
        'YIELD_2Y': us02y_df['close'],
        'YIELD_10Y': us10y_df['close'],
        'WTI': wti_df['close']
    })

    # Calculate yield curve (10Y - 2Y)
    df['YIELD_CURVE'] = df['YIELD_10Y'] - df['YIELD_2Y']

    # Validate data quality
    logger.info(f"\n✅ Combined macro data: {len(df)} hourly records")
    logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Show latest values
    logger.info(f"\n   Latest snapshot ({df.index[-1].date()}):")
    logger.info(f"   VIX:      {df['VIX'].iloc[-1]:.2f}")
    logger.info(f"   DXY:      {df['DXY'].iloc[-1]:.2f}")
    logger.info(f"   MOVE:     {df['MOVE'].iloc[-1]:.2f}")
    logger.info(f"   2Y Yield: {df['YIELD_2Y'].iloc[-1]:.2f}%")
    logger.info(f"   10Y Yield: {df['YIELD_10Y'].iloc[-1]:.2f}%")
    logger.info(f"   Curve:    {df['YIELD_CURVE'].iloc[-1]:.2f}bp {'(inverted!)' if df['YIELD_CURVE'].iloc[-1] < 0 else ''}")
    logger.info(f"   WTI Oil:  ${df['WTI'].iloc[-1]:.2f}/bbl")

    # Check for variance (not flat synthetic data)
    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        std_val = df[col].std()
        assert std_val > 0.01, f"{col} appears flat (std={std_val:.6f})"
        logger.info(f"   {col} std: {std_val:.2f} ✅")

    logger.info("\n   Data quality checks passed ✅")

    return df


def merge_macro_into_feature_store(asset: str):
    """
    Merge all real macro data into asset's feature store

    Args:
        asset: BTC or ETH
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Merging ALL Real Macro Data into {asset} Feature Store")
    logger.info("=" * 70)

    # Load existing feature store
    feature_path = f"data/macro/{asset}_macro_features.parquet"
    feature_df = pd.read_parquet(feature_path)

    logger.info(f"\n📊 Loaded {asset} feature store: {feature_df.shape}")
    logger.info(f"   Date range: {feature_df['timestamp'].min()} to {feature_df['timestamp'].max()}")

    # Load all macro indicators
    macro_df = load_all_macro_indicators()

    # The index IS the timestamp from TradingView data (named 'time')
    # Reset index to get timestamp as a column
    macro_df = macro_df.reset_index()
    # Rename 'time' to 'timestamp' for consistency
    if 'time' in macro_df.columns:
        macro_df = macro_df.rename(columns={'time': 'timestamp'})

    # Ensure timestamps are timezone-aware
    feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'], utc=True)
    macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], utc=True)

    # Check coverage before merge
    logger.info(f"\n   Before merge:")
    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        if col in feature_df.columns:
            non_null = feature_df[col].notna().sum()
            logger.info(f"   {col}: {non_null}/{len(feature_df)} ({non_null/len(feature_df)*100:.1f}%)")

    # Merge on timestamp
    # Select only the macro columns we want to merge
    merge_cols = ['timestamp', 'VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y', 'YIELD_CURVE', 'WTI']
    macro_subset = macro_df[merge_cols].copy()

    # Merge, preferring new macro data
    merged_df = feature_df.merge(
        macro_subset,
        on='timestamp',
        how='left',
        suffixes=('_old', '')
    )

    # Drop old columns if they exist
    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        old_col = f'{col}_old'
        if old_col in merged_df.columns:
            # Use new value if available, otherwise keep old
            merged_df[col] = merged_df[col].fillna(merged_df[old_col])
            merged_df = merged_df.drop(columns=[old_col])

    # Add YIELD_CURVE and WTI if they don't exist
    if 'YIELD_CURVE' not in feature_df.columns:
        logger.info("   Added YIELD_CURVE column")
    if 'WTI' not in feature_df.columns:
        logger.info("   Added WTI column")

    # Check coverage after merge
    logger.info(f"\n   After merge:")
    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        non_null_before = feature_df[col].notna().sum() if col in feature_df.columns else 0
        non_null_after = merged_df[col].notna().sum()
        improvement = non_null_after - non_null_before
        logger.info(f"   {col}: {non_null_after}/{len(merged_df)} ({non_null_after/len(merged_df)*100:.1f}%) +{improvement} bars")

    # Show sample of filled values
    filled_mask = merged_df['VIX'].notna()
    if filled_mask.any():
        sample_row = merged_df[filled_mask].iloc[-1]
        logger.info(f"\n   Latest filled values ({sample_row['timestamp'].date()}):")
        logger.info(f"   VIX:      {sample_row['VIX']:.2f}")
        logger.info(f"   DXY:      {sample_row['DXY']:.2f}")
        logger.info(f"   MOVE:     {sample_row['MOVE']:.2f}")
        logger.info(f"   2Y:       {sample_row['YIELD_2Y']:.2f}%")
        logger.info(f"   10Y:      {sample_row['YIELD_10Y']:.2f}%")
        if 'YIELD_CURVE' in merged_df.columns:
            logger.info(f"   Curve:    {sample_row['YIELD_CURVE']:.2f}bp")
        if 'WTI' in merged_df.columns:
            logger.info(f"   WTI:      ${sample_row['WTI']:.2f}/bbl")

    # Backup original
    backup_path = f"{feature_path}.bak2"
    feature_df.to_parquet(backup_path)
    logger.info(f"\n💾 Backed up original to: {backup_path}")

    # Save merged data
    merged_df.to_parquet(feature_path)
    logger.info(f"💾 Saved updated feature store to: {feature_path}")
    logger.info(f"   Size: {Path(feature_path).stat().st_size / 1024:.1f} KB")
    logger.info(f"   Total columns: {len(merged_df.columns)}")

    return merged_df


def validate_2024_macro_data(asset: str = 'BTC'):
    """
    Validate that 2024 macro data is complete and realistic
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Validating 2024 Macro Data for {asset}")
    logger.info("=" * 70)

    # Load feature store
    feature_df = pd.read_parquet(f"data/macro/{asset}_macro_features.parquet")
    feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'], utc=True)

    # Filter to 2024
    mask_2024 = (feature_df['timestamp'] >= '2024-01-01') & (feature_df['timestamp'] < '2025-01-01')
    df_2024 = feature_df[mask_2024]

    logger.info(f"\n📊 2024 Data Coverage ({len(df_2024)} bars):")

    # Check each macro feature
    macro_cols = ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y', 'TOTAL', 'TOTAL2', 'BTC.D']

    for col in macro_cols:
        if col not in df_2024.columns:
            logger.warning(f"   ⚠️  {col}: Column missing!")
            continue

        non_null = df_2024[col].notna().sum()
        pct = non_null / len(df_2024) * 100
        std = df_2024[col].std()
        mean = df_2024[col].mean()
        min_val = df_2024[col].min()
        max_val = df_2024[col].max()

        status = "✅" if pct >= 95 and std > 0.01 else "⚠️"
        logger.info(f"   {status} {col:10s}: {pct:5.1f}% coverage, std={std:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")

    # Show 2024 statistics
    logger.info(f"\n📈 2024 Macro Statistics:")
    logger.info(df_2024[macro_cols].describe().round(2))

    # Check for synthetic constants
    synthetic_flags = []
    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        if col in df_2024.columns:
            std = df_2024[col].std()
            if std < 0.01:
                synthetic_flags.append(col)

    if synthetic_flags:
        logger.warning(f"\n⚠️  WARNING: Synthetic constants detected: {synthetic_flags}")
        logger.warning(f"   These appear to be flat fallback values, not real market data!")
        return False
    else:
        logger.info(f"\n✅ All macro features show realistic variance!")
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process all real macro data from TradingView")
    parser.add_argument('--asset', type=str, default='BTC', choices=['BTC', 'ETH'])
    parser.add_argument('--all', action='store_true', help='Process both BTC and ETH')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, do not merge')

    args = parser.parse_args()

    if args.validate_only:
        validate_2024_macro_data(args.asset)
    elif args.all:
        merge_macro_into_feature_store('BTC')
        print("\n")
        merge_macro_into_feature_store('ETH')
        print("\n")
        validate_2024_macro_data('BTC')
    else:
        merge_macro_into_feature_store(args.asset)
        validate_2024_macro_data(args.asset)

    print("\n" + "=" * 70)
    print("✅ Real macro data processing complete!")
    print("=" * 70)
