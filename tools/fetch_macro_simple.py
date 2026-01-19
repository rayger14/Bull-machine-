"""
Simple macro feature fetcher - uses existing price data and adds minimal external features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_feature_store(asset: str = "BTC"):
    """Load existing feature store"""
    path = f"data/features/v18/{asset}_1H.parquet"
    logger.info(f"Loading feature store: {path}")

    df = pd.read_parquet(path)
    df = df.reset_index()  # Move 'time' index to column
    df = df.rename(columns={'time': 'timestamp'})  # Rename to timestamp
    logger.info(f"  ✅ Loaded {len(df)} bars")

    return df


def compute_realized_volatility(df: pd.DataFrame):
    """Compute realized volatility from close prices"""
    logger.info("Computing realized volatility...")

    # Log returns
    returns = np.log(df['close'] / df['close'].shift(1))

    # 20 hour and 60 hour rolling windows
    df['rv_20d'] = returns.rolling(20).std() * np.sqrt(365 * 24) * 100
    df['rv_60d'] = returns.rolling(60).std() * np.sqrt(365 * 24) * 100

    logger.info("  ✅ Computed rv_20d and rv_60d")

    return df


def add_synthetic_macro(df: pd.DataFrame):
    """Add synthetic macro features"""
    logger.info("Adding synthetic macro features...")

    # Use reasonable defaults based on 2024 conditions
    df['VIX'] = 20.0  # Moderate volatility
    df['DXY'] = 104.0  # Dollar strength
    df['MOVE'] = 80.0  # Bond volatility
    df['YIELD_2Y'] = 4.5  # 2Y Treasury
    df['YIELD_10Y'] = 4.3  # 10Y Treasury

    # Crypto metrics (computed from price action)
    df['funding'] = 0.01  # 1% annualized (neutral)
    df['oi'] = df['volume'] * df['close']  # OI proxy from volume

    # Market cap dominance (approximate 2024 values)
    df['TOTAL'] = np.nan  # Total crypto market cap (not critical)
    df['TOTAL2'] = np.nan  # Ex-BTC market cap
    df['USDT.D'] = 4.5  # USDT dominance ~4-5%
    df['BTC.D'] = 55.0  # BTC dominance ~55%

    logger.info("  ✅ Added synthetic macro features")

    return df


def build_macro_dataset(asset: str = "BTC"):
    """
    Build macro dataset from existing feature store

    Args:
        asset: BTC or ETH

    Returns:
        DataFrame with macro features
    """
    logger.info("=" * 70)
    logger.info(f"Building macro dataset for {asset}")
    logger.info("=" * 70)

    # Load feature store
    df = load_feature_store(asset)

    # Compute RV
    df = compute_realized_volatility(df)

    # Add synthetic macro
    df = add_synthetic_macro(df)

    # Select macro columns
    macro_cols = [
        'timestamp', 'close', 'volume',
        'rv_20d', 'rv_60d',
        'VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y',
        'funding', 'oi',
        'TOTAL', 'TOTAL2', 'USDT.D', 'BTC.D'
    ]

    macro_df = df[macro_cols].copy()

    # Forward fill NaNs
    macro_df = macro_df.fillna(method='ffill')

    logger.info(f"\n✅ Macro dataset complete: {len(macro_df)} records")
    logger.info(f"   Date range: {macro_df['timestamp'].min()} to {macro_df['timestamp'].max()}")

    return macro_df


def save_macro_dataset(df: pd.DataFrame, output_path: str):
    """Save macro dataset to parquet"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"\n💾 Saved to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', type=str, default='BTC', choices=['BTC', 'ETH'])
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/macro/{args.asset}_macro_features.parquet"

    # Build dataset
    macro_df = build_macro_dataset(args.asset)

    # Save
    save_macro_dataset(macro_df, args.output)

    # Show sample
    print("\n" + "=" * 70)
    print("MACRO FEATURE SAMPLE (last 5 rows)")
    print("=" * 70)
    print(macro_df.tail())

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(macro_df[['rv_20d', 'rv_60d', 'funding', 'oi']].describe())
