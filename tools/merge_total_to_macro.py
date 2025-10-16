"""
Merge TOTAL/TOTAL2/TOTAL3 data into macro feature stores

Updates BTC_macro_features.parquet and ETH_macro_features.parquet
with real crypto market cap data.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_total_data_to_macro_features(asset: str):
    """
    Merge TOTAL/TOTAL2/TOTAL3 data into asset's macro feature store

    Args:
        asset: BTC or ETH
    """
    logger.info("=" * 70)
    logger.info(f"Merging TOTAL/TOTAL2/TOTAL3 data into {asset} macro features")
    logger.info("=" * 70)

    # Load existing macro features
    macro_path = f"data/macro/{asset}_macro_features.parquet"
    macro_df = pd.read_parquet(macro_path)

    logger.info(f"\n📊 Loaded {asset} macro features: {macro_df.shape}")
    logger.info(f"   Date range: {macro_df['timestamp'].min()} to {macro_df['timestamp'].max()}")
    logger.info(f"   Existing columns: {list(macro_df.columns)}")

    # Load TOTAL/TOTAL2/TOTAL3 data
    total_path = "data/macro/crypto_marketcap_hourly.parquet"
    total_df = pd.read_parquet(total_path)

    # Rename 'time' to 'timestamp' for merging
    if 'time' in total_df.columns:
        total_df = total_df.rename(columns={'time': 'timestamp'})

    logger.info(f"\n📊 Loaded TOTAL/TOTAL2/TOTAL3 data: {total_df.shape}")
    logger.info(f"   Date range: {total_df['timestamp'].min()} to {total_df['timestamp'].max()}")

    # Check TOTAL/TOTAL2 columns before merge
    total_before = macro_df['TOTAL'].notna().sum()
    total2_before = macro_df['TOTAL2'].notna().sum()

    logger.info(f"\n   Before merge:")
    logger.info(f"   TOTAL non-null: {total_before}/{len(macro_df)}")
    logger.info(f"   TOTAL2 non-null: {total2_before}/{len(macro_df)}")

    # Merge TOTAL/TOTAL2/TOTAL3 and BTC.D data
    # Select only the columns we want to merge
    merge_cols = ['timestamp', 'TOTAL', 'TOTAL2', 'TOTAL3', 'BTC.D']
    available_merge_cols = [col for col in merge_cols if col in total_df.columns]

    total_subset = total_df[available_merge_cols].copy()

    # Ensure timestamps are timezone-aware for both
    macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], utc=True)
    total_subset['timestamp'] = pd.to_datetime(total_subset['timestamp'], utc=True)

    # Merge on timestamp, preferring TOTAL data values
    merged_df = macro_df.merge(
        total_subset,
        on='timestamp',
        how='left',
        suffixes=('_old', '')
    )

    # If columns were renamed with _old suffix, drop the old ones
    for col in ['TOTAL', 'TOTAL2', 'TOTAL3', 'BTC.D']:
        old_col = f'{col}_old'
        if old_col in merged_df.columns:
            # Use new value if available, otherwise keep old
            merged_df[col] = merged_df[col].fillna(merged_df[old_col])
            merged_df = merged_df.drop(columns=[old_col])

    # Add TOTAL3 column if it doesn't exist
    if 'TOTAL3' not in macro_df.columns and 'TOTAL3' in merged_df.columns:
        logger.info("   Added TOTAL3 column")

    # Check after merge
    total_after = merged_df['TOTAL'].notna().sum()
    total2_after = merged_df['TOTAL2'].notna().sum()
    total3_after = merged_df['TOTAL3'].notna().sum() if 'TOTAL3' in merged_df.columns else 0
    btcd_after = merged_df['BTC.D'].notna().sum()

    logger.info(f"\n   After merge:")
    logger.info(f"   TOTAL non-null: {total_after}/{len(merged_df)} (+{total_after - total_before})")
    logger.info(f"   TOTAL2 non-null: {total2_after}/{len(merged_df)} (+{total2_after - total2_before})")
    logger.info(f"   TOTAL3 non-null: {total3_after}/{len(merged_df)}")
    logger.info(f"   BTC.D non-null: {btcd_after}/{len(merged_df)}")

    # Show sample of filled values
    filled_mask = merged_df['TOTAL'].notna()
    if filled_mask.any():
        sample_row = merged_df[filled_mask].iloc[-1]
        logger.info(f"\n   Latest filled values ({sample_row['timestamp'].date()}):")
        logger.info(f"   TOTAL:  ${sample_row['TOTAL']/1e9:.1f}B")
        logger.info(f"   TOTAL2: ${sample_row['TOTAL2']/1e9:.1f}B")
        if 'TOTAL3' in merged_df.columns:
            logger.info(f"   TOTAL3: ${sample_row['TOTAL3']/1e9:.1f}B")
        logger.info(f"   BTC.D:  {sample_row['BTC.D']:.2f}%")

    # Save updated macro features
    backup_path = f"{macro_path}.bak"
    Path(backup_path).parent.mkdir(parents=True, exist_ok=True)

    # Backup original
    macro_df.to_parquet(backup_path)
    logger.info(f"\n💾 Backed up original to: {backup_path}")

    # Save merged data
    merged_df.to_parquet(macro_path)
    logger.info(f"💾 Saved updated macro features to: {macro_path}")
    logger.info(f"   Size: {Path(macro_path).stat().st_size / 1024:.1f} KB")

    return merged_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge TOTAL/TOTAL2/TOTAL3 into macro features")
    parser.add_argument('--asset', type=str, default='BTC',
                        choices=['BTC', 'ETH'], help='Asset to update')
    parser.add_argument('--all', action='store_true',
                        help='Update both BTC and ETH')

    args = parser.parse_args()

    if args.all:
        merge_total_data_to_macro_features('BTC')
        print("\n")
        merge_total_data_to_macro_features('ETH')
    else:
        merge_total_data_to_macro_features(args.asset)

    print("\n" + "=" * 70)
    print("✅ Merge complete!")
    print("=" * 70)
