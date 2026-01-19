"""
Fetch Historical Macro Data from Yahoo Finance and FRED

Downloads real historical data for:
- VIX (CBOE Volatility Index)
- DXY (US Dollar Index)
- US Treasury Yields (2Y, 10Y)
- Oil (WTI)
- MOVE Index (Bond volatility - proxy if not available)

Covers full 2024 history for regime classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import requests
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_yahoo_finance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance

    Args:
        symbol: Yahoo Finance symbol (e.g., ^VIX, DX-Y.NYB)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"\n📊 Fetching {symbol} from Yahoo Finance...")
    logger.info(f"   Date range: {start_date} to {end_date}")

    # Convert dates to Unix timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())

    # Yahoo Finance download URL
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
    params = {
        'period1': start_ts,
        'period2': end_ts,
        'interval': '1d',
        'events': 'history',
        'includeAdjustedClose': 'true'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse CSV
        df = pd.read_csv(StringIO(response.text))

        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        logger.info(f"   ✅ Fetched {len(df)} daily bars")
        logger.info(f"   Range: {df.index.min().date()} to {df.index.max().date()}")
        logger.info(f"   Latest close: {df['close'].iloc[-1]:.2f}")

        return df

    except Exception as e:
        logger.error(f"   ❌ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def fetch_fred_series(series_id: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    """
    Fetch economic data from FRED (Federal Reserve Economic Data)

    Args:
        series_id: FRED series ID (e.g., DGS2, DGS10)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: FRED API key (optional - can use public endpoint for some data)

    Returns:
        DataFrame with series data
    """
    logger.info(f"\n📊 Fetching {series_id} from FRED...")

    # Try direct CSV download (works for some public series)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {
        'id': series_id,
        'cosd': start_date,
        'coed': end_date
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
        df.columns = ['close']

        # Remove NaN values
        df = df.dropna()

        logger.info(f"   ✅ Fetched {len(df)} daily observations")
        logger.info(f"   Range: {df.index.min().date()} to {df.index.max().date()}")
        logger.info(f"   Latest value: {df['close'].iloc[-1]:.2f}")

        return df

    except Exception as e:
        logger.error(f"   ❌ Failed to fetch {series_id}: {e}")
        return pd.DataFrame()


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to hourly using forward fill

    Args:
        df: Daily DataFrame

    Returns:
        Hourly DataFrame
    """
    # Ensure timezone aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Resample to hourly
    df_hourly = df.resample('1h').ffill()

    return df_hourly


def fetch_all_macro_indicators(start_date: str = '2022-01-01', end_date: str = '2025-10-14') -> dict:
    """
    Fetch all macro indicators from various sources

    Returns:
        Dictionary of DataFrames with hourly data
    """
    logger.info("=" * 70)
    logger.info("FETCHING HISTORICAL MACRO DATA")
    logger.info("=" * 70)

    indicators = {}

    # 1. VIX - CBOE Volatility Index
    vix_df = fetch_yahoo_finance('^VIX', start_date, end_date)
    if not vix_df.empty:
        indicators['VIX'] = resample_to_hourly(vix_df[['close']])

    # 2. DXY - US Dollar Index (use UUP ETF as proxy, or try DX-Y.NYB)
    dxy_df = fetch_yahoo_finance('DX-Y.NYB', start_date, end_date)
    if dxy_df.empty:
        logger.info("   Trying UUP ETF as DXY proxy...")
        dxy_df = fetch_yahoo_finance('UUP', start_date, end_date)
    if not dxy_df.empty:
        indicators['DXY'] = resample_to_hourly(dxy_df[['close']])

    # 3. Treasury Yields from FRED
    us02y_df = fetch_fred_series('DGS2', start_date, end_date)  # 2-Year
    if not us02y_df.empty:
        indicators['YIELD_2Y'] = resample_to_hourly(us02y_df)

    us10y_df = fetch_fred_series('DGS10', start_date, end_date)  # 10-Year
    if not us10y_df.empty:
        indicators['YIELD_10Y'] = resample_to_hourly(us10y_df)

    # 4. WTI Oil
    wti_df = fetch_yahoo_finance('CL=F', start_date, end_date)  # WTI Crude
    if not wti_df.empty:
        indicators['WTI'] = resample_to_hourly(wti_df[['close']])

    # 5. MOVE Index proxy - use TLT (Treasury Bond ETF) volatility
    # MOVE is not publicly available, so we'll use TLT as a proxy
    logger.info("\n📊 MOVE Index not directly available - using TLT volatility as proxy...")
    tlt_df = fetch_yahoo_finance('TLT', start_date, end_date)
    if not tlt_df.empty:
        # Calculate 20-day historical volatility as MOVE proxy
        tlt_returns = tlt_df['close'].pct_change()
        tlt_vol = tlt_returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized vol
        move_proxy = pd.DataFrame({'close': tlt_vol}, index=tlt_df.index)
        move_proxy = move_proxy.dropna()
        # Scale to approximate MOVE range (typically 80-150)
        move_proxy['close'] = 80 + (move_proxy['close'] - move_proxy['close'].min()) * 50
        indicators['MOVE'] = resample_to_hourly(move_proxy)
        logger.info(f"   ✅ Created MOVE proxy from TLT volatility")

    return indicators


def merge_into_feature_store(indicators: dict, asset: str = 'BTC'):
    """
    Merge fetched indicators into asset's macro feature store

    Args:
        indicators: Dict of indicator DataFrames
        asset: BTC or ETH
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"MERGING HISTORICAL DATA INTO {asset} FEATURE STORE")
    logger.info("=" * 70)

    # Load existing feature store
    feature_path = f"data/macro/{asset}_macro_features.parquet"
    feature_df = pd.read_parquet(feature_path)
    feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'], utc=True)

    logger.info(f"\n📊 Loaded {asset} feature store: {feature_df.shape}")
    logger.info(f"   Date range: {feature_df['timestamp'].min()} to {feature_df['timestamp'].max()}")

    # Check coverage before merge
    logger.info(f"\n   Before merge (2024 only):")
    mask_2024 = (feature_df['timestamp'] >= '2024-01-01') & (feature_df['timestamp'] < '2025-01-01')
    df_2024 = feature_df[mask_2024]

    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        if col in df_2024.columns:
            non_null = df_2024[col].notna().sum()
            std = df_2024[col].std()
            status = "✅" if std > 0.01 else "⚠️"
            logger.info(f"   {status} {col:10s}: {non_null}/{len(df_2024)} bars, std={std:.2f}")

    # Merge each indicator
    merged_df = feature_df.copy()

    for name, indicator_df in indicators.items():
        if indicator_df.empty:
            continue

        # Prepare for merge
        indicator_df = indicator_df.reset_index()
        indicator_df = indicator_df.rename(columns={indicator_df.columns[0]: 'timestamp', 'close': name})
        indicator_df['timestamp'] = pd.to_datetime(indicator_df['timestamp'], utc=True)

        # Merge on timestamp, OVERWRITING existing values
        merged_df = merged_df.merge(
            indicator_df[['timestamp', name]],
            on='timestamp',
            how='left',
            suffixes=('_old', '')
        )

        # Drop old column if exists
        old_col = f'{name}_old'
        if old_col in merged_df.columns:
            # Prefer new data over old
            merged_df[name] = merged_df[name].fillna(merged_df[old_col])
            # Actually, let's REPLACE old data with new data where available
            mask_new = merged_df[name].notna()
            merged_df.loc[mask_new, name] = merged_df.loc[mask_new, name]
            merged_df = merged_df.drop(columns=[old_col])

        logger.info(f"   ✅ Merged {name}: {indicator_df[name].notna().sum()} bars")

    # Check coverage after merge
    logger.info(f"\n   After merge (2024 only):")
    df_2024_new = merged_df[mask_2024]

    for col in ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y']:
        if col in df_2024_new.columns:
            non_null = df_2024_new[col].notna().sum()
            std = df_2024_new[col].std()
            mean = df_2024_new[col].mean()
            status = "✅" if std > 0.01 else "⚠️"
            logger.info(f"   {status} {col:10s}: {non_null}/{len(df_2024_new)} bars, std={std:.2f}, mean={mean:.2f}")

    # Backup and save
    backup_path = f"{feature_path}.bak_historical"
    feature_df.to_parquet(backup_path)
    logger.info(f"\n💾 Backed up original to: {backup_path}")

    merged_df.to_parquet(feature_path)
    logger.info(f"💾 Saved updated feature store to: {feature_path}")
    logger.info(f"   Size: {Path(feature_path).stat().st_size / 1024:.1f} KB")

    return merged_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical macro data")
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2025-10-14', help='End date')
    parser.add_argument('--asset', type=str, default='BTC', choices=['BTC', 'ETH'])
    parser.add_argument('--all', action='store_true', help='Process both BTC and ETH')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch data, do not merge')

    args = parser.parse_args()

    # Fetch all indicators
    indicators = fetch_all_macro_indicators(args.start, args.end)

    if not args.fetch_only:
        if args.all:
            merge_into_feature_store(indicators, 'BTC')
            print("\n")
            merge_into_feature_store(indicators, 'ETH')
        else:
            merge_into_feature_store(indicators, args.asset)

    # Show summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, df in indicators.items():
        if not df.empty:
            print(f"✅ {name:15s}: {len(df):6d} hourly bars")
        else:
            print(f"❌ {name:15s}: Failed to fetch")

    print("\n" + "=" * 70)
    print("✅ Historical macro data fetch complete!")
    print("=" * 70)
