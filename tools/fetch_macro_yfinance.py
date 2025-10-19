"""
Fetch Historical Macro Data using yfinance (better rate limiting)

Downloads real historical data for 2024 regime classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import time

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_yfinance_data(symbol: str, start_date: str, end_date: str, delay: float = 1.0) -> pd.DataFrame:
    """
    Fetch historical data using yfinance library

    Args:
        symbol: Yahoo Finance symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        delay: Delay between requests to avoid rate limiting

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"\n📊 Fetching {symbol}...")
    logger.info(f"   Date range: {start_date} to {end_date}")

    try:
        time.sleep(delay)  # Rate limiting delay

        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if df.empty:
            logger.error(f"   ❌ No data returned for {symbol}")
            return pd.DataFrame()

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Ensure timezone aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        logger.info(f"   ✅ Fetched {len(df)} daily bars")
        logger.info(f"   Range: {df.index.min().date()} to {df.index.max().date()}")
        logger.info(f"   Latest close: {df['close'].iloc[-1]:.2f}")
        logger.info(f"   Mean: {df['close'].mean():.2f}, Std: {df['close'].std():.2f}")

        return df

    except Exception as e:
        logger.error(f"   ❌ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to hourly using forward fill"""
    if df.empty:
        return df

    # Ensure timezone aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Resample to hourly
    df_hourly = df.resample('1h').ffill()

    return df_hourly


def fetch_all_indicators(start_date: str = '2023-01-01', end_date: str = '2025-10-14') -> dict:
    """
    Fetch all macro indicators with rate limiting

    Returns:
        Dictionary of indicator DataFrames (hourly)
    """
    logger.info("=" * 70)
    logger.info("FETCHING HISTORICAL MACRO DATA (yfinance)")
    logger.info("=" * 70)

    indicators = {}
    delay = 2.0  # 2 second delay between requests

    # 1. VIX - CBOE Volatility Index
    vix_df = fetch_yfinance_data('^VIX', start_date, end_date, delay)
    if not vix_df.empty:
        indicators['VIX'] = resample_to_hourly(vix_df[['close']])

    # 2. DXY - US Dollar Index
    dxy_df = fetch_yfinance_data('DX-Y.NYB', start_date, end_date, delay)
    if dxy_df.empty:
        logger.info("   Trying UUP ETF as DXY proxy...")
        dxy_df = fetch_yfinance_data('UUP', start_date, end_date, delay)
    if not dxy_df.empty:
        # If UUP, scale to approximate DXY range (typically 100-110)
        if 'UUP' in str(dxy_df):
            dxy_df['close'] = dxy_df['close'] * 4  # Approximate scaling
        indicators['DXY'] = resample_to_hourly(dxy_df[['close']])

    # 3. Treasury Yields - use ETF proxies
    # SHY = 1-3 Year Treasury ETF (proxy for 2Y)
    # IEF = 7-10 Year Treasury ETF (proxy for 10Y)
    logger.info("\n   Using Treasury ETFs as yield proxies...")

    shy_df = fetch_yfinance_data('SHY', start_date, end_date, delay)
    if not shy_df.empty:
        # Convert ETF price to approximate yield (inverse relationship)
        # 2Y yield typically 2-5%, inversely related to price
        shy_yield = 4.5 - (shy_df['close'] - shy_df['close'].mean()) * 0.05
        indicators['YIELD_2Y'] = resample_to_hourly(pd.DataFrame({'close': shy_yield}))

    ief_df = fetch_yfinance_data('IEF', start_date, end_date, delay)
    if not ief_df.empty:
        # 10Y yield typically 3-5%
        ief_yield = 4.0 - (ief_df['close'] - ief_df['close'].mean()) * 0.04
        indicators['YIELD_10Y'] = resample_to_hourly(pd.DataFrame({'close': ief_yield}))

    # 4. WTI Oil
    wti_df = fetch_yfinance_data('CL=F', start_date, end_date, delay)
    if not wti_df.empty:
        indicators['WTI'] = resample_to_hourly(wti_df[['close']])

    # 5. MOVE Index proxy - TLT volatility
    logger.info("\n   Creating MOVE proxy from TLT volatility...")
    tlt_df = fetch_yfinance_data('TLT', start_date, end_date, delay)
    if not tlt_df.empty:
        # Calculate 20-day rolling volatility
        tlt_returns = tlt_df['close'].pct_change()
        tlt_vol = tlt_returns.rolling(20).std() * np.sqrt(252) * 100
        # Scale to MOVE range (typically 80-150)
        move_proxy = 80 + (tlt_vol - tlt_vol.min()) / (tlt_vol.max() - tlt_vol.min()) * 70
        move_proxy = move_proxy.fillna(method='ffill')
        indicators['MOVE'] = resample_to_hourly(pd.DataFrame({'close': move_proxy}))
        logger.info(f"   ✅ Created MOVE proxy (range: {move_proxy.min():.1f} - {move_proxy.max():.1f})")

    return indicators


def merge_into_feature_store(indicators: dict, asset: str = 'BTC'):
    """
    Merge indicators into feature store, OVERWRITING synthetic data

    Args:
        indicators: Dict of indicator DataFrames
        asset: BTC or ETH
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"MERGING INTO {asset} FEATURE STORE")
    logger.info("=" * 70)

    # Load feature store
    feature_path = f"data/macro/{asset}_macro_features.parquet"
    feature_df = pd.read_parquet(feature_path)
    feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'], utc=True)

    logger.info(f"\n📊 Feature store: {feature_df.shape}")

    # Check 2024 coverage before
    mask_2024 = (feature_df['timestamp'] >= '2024-01-01') & (feature_df['timestamp'] < '2025-01-01')
    df_2024_before = feature_df[mask_2024]

    logger.info(f"\n   2024 BEFORE merge:")
    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        if col in df_2024_before.columns:
            std = df_2024_before[col].std()
            mean = df_2024_before[col].mean()
            status = "✅" if std > 0.01 else "⚠️"
            logger.info(f"   {status} {col:10s}: std={std:.3f}, mean={mean:.2f}")

    # Merge each indicator - REPLACE old values
    for name, indicator_df in indicators.items():
        if indicator_df.empty:
            continue

        # Prepare for merge
        indicator_df = indicator_df.reset_index()
        indicator_df.columns = ['timestamp', name]
        indicator_df['timestamp'] = pd.to_datetime(indicator_df['timestamp'], utc=True)

        # Drop old column entirely if exists
        if name in feature_df.columns:
            feature_df = feature_df.drop(columns=[name])

        # Merge to add new column
        feature_df = feature_df.merge(
            indicator_df[['timestamp', name]],
            on='timestamp',
            how='left'
        )

        logger.info(f"   ✅ Merged {name}: {indicator_df[name].notna().sum()} bars")

    # Check 2024 coverage after
    df_2024_after = feature_df[mask_2024]

    logger.info(f"\n   2024 AFTER merge:")
    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        if col in df_2024_after.columns:
            non_null = df_2024_after[col].notna().sum()
            std = df_2024_after[col].std()
            mean = df_2024_after[col].mean()
            min_val = df_2024_after[col].min()
            max_val = df_2024_after[col].max()
            status = "✅" if std > 0.1 else "⚠️"
            logger.info(f"   {status} {col:10s}: {non_null}/{len(df_2024_after)} bars, std={std:.2f}, mean={mean:.2f}, range=[{min_val:.1f}, {max_val:.1f}]")

    # Backup and save
    backup_path = f"{feature_path}.bak_yfinance"
    pd.read_parquet(feature_path).to_parquet(backup_path)
    logger.info(f"\n💾 Backed up to: {backup_path}")

    feature_df.to_parquet(feature_path)
    logger.info(f"💾 Saved to: {feature_path}")
    logger.info(f"   Size: {Path(feature_path).stat().st_size / 1024:.1f} KB")

    return feature_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2025-10-14')
    parser.add_argument('--asset', default='BTC', choices=['BTC', 'ETH'])
    parser.add_argument('--all', action='store_true')

    args = parser.parse_args()

    # Fetch indicators
    indicators = fetch_all_indicators(args.start, args.end)

    # Merge
    if args.all:
        merge_into_feature_store(indicators, 'BTC')
        print("\n")
        merge_into_feature_store(indicators, 'ETH')
    else:
        merge_into_feature_store(indicators, args.asset)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, df in indicators.items():
        if not df.empty:
            print(f"✅ {name:15s}: {len(df):6d} hourly bars")
    print("=" * 70)
    print("✅ Complete!")
    print("=" * 70)
