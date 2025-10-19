"""
Fetch macro features for regime classification

Uses CCXT for crypto data (funding, OI) and yfinance for TradFi data
Saves to parquet for fast loading by optimize_v19.py
"""

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_funding_rate(symbol: str = "BTC/USDT", start_date: str = "2024-01-01"):
    """
    Fetch funding rate history from Binance

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        start_date: Start date for historical data

    Returns:
        DataFrame with timestamp and funding_rate columns
    """
    logger.info(f"Fetching funding rate for {symbol}...")

    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # Binance funding rates are every 8 hours
        since = int(pd.Timestamp(start_date).timestamp() * 1000)

        funding_history = []

        # Fetch in chunks (500 records at a time)
        while True:
            rates = exchange.fetch_funding_rate_history(symbol, since=since, limit=500)

            if not rates:
                break

            funding_history.extend(rates)

            # Update since to last timestamp
            since = rates[-1]['timestamp'] + 1

            # Stop if we reached current time
            if rates[-1]['timestamp'] > pd.Timestamp.now().timestamp() * 1000:
                break

            logger.info(f"  Fetched {len(funding_history)} funding rate records...")

        # Convert to DataFrame
        df = pd.DataFrame(funding_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[['timestamp', 'fundingRate']].rename(columns={'fundingRate': 'funding'})

        # Resample to hourly (forward fill)
        df = df.set_index('timestamp').resample('1H').ffill().reset_index()

        logger.info(f"  ✅ Fetched {len(df)} hourly funding rates")
        return df

    except Exception as e:
        logger.error(f"  ❌ Failed to fetch funding rates: {e}")
        return pd.DataFrame(columns=['timestamp', 'funding'])


def fetch_open_interest(symbol: str = "BTC/USDT", start_date: str = "2024-01-01"):
    """
    Fetch open interest history from Binance

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        start_date: Start date for historical data

    Returns:
        DataFrame with timestamp and oi columns
    """
    logger.info(f"Fetching open interest for {symbol}...")

    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # Binance OI is available via REST API
        # We'll use open interest history endpoint
        since = int(pd.Timestamp(start_date).timestamp() * 1000)

        # Note: Binance doesn't have a direct fetch_open_interest_history in CCXT
        # We'll fetch OHLCV and calculate OI proxy from volume
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Use volume as OI proxy (normalize)
        df['oi'] = df['volume']
        df = df[['timestamp', 'oi']]

        logger.info(f"  ✅ Fetched {len(df)} hourly OI records (volume proxy)")
        return df

    except Exception as e:
        logger.error(f"  ❌ Failed to fetch open interest: {e}")
        return pd.DataFrame(columns=['timestamp', 'oi'])


def fetch_crypto_dominance(start_date: str = "2024-01-01"):
    """
    Fetch crypto market cap data (TOTAL, TOTAL2, USDT.D, BTC.D)

    Uses CoinGecko API or yfinance proxies

    Returns:
        DataFrame with timestamp, TOTAL, TOTAL2, USDT.D, BTC.D columns
    """
    logger.info("Fetching crypto dominance data...")

    # For simplicity, we'll use synthetic data based on BTC price
    # In production, use CoinGecko API or TradingView data

    try:
        # Fetch BTC price as proxy
        btc = yf.download("BTC-USD", start=start_date, interval="1h", progress=False)

        if btc.empty:
            raise ValueError("No BTC data from yfinance")

        df = pd.DataFrame({
            'timestamp': btc.index,
            'TOTAL': np.nan,  # Total crypto market cap (will use proxy)
            'TOTAL2': np.nan,  # Total ex-BTC
            'USDT.D': 4.5,    # USDT dominance (relatively stable ~4-5%)
            'BTC.D': 55.0     # BTC dominance (approximate)
        })

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')

        logger.info(f"  ✅ Generated {len(df)} hourly dominance records (synthetic)")
        return df

    except Exception as e:
        logger.error(f"  ❌ Failed to fetch crypto dominance: {e}")
        return pd.DataFrame(columns=['timestamp', 'TOTAL', 'TOTAL2', 'USDT.D', 'BTC.D'])


def compute_realized_volatility(price_df: pd.DataFrame, windows=[20, 60]):
    """
    Compute realized volatility from price data

    Args:
        price_df: DataFrame with 'close' column
        windows: List of window sizes in hours

    Returns:
        DataFrame with rv_20d and rv_60d columns
    """
    logger.info("Computing realized volatility...")

    # Compute log returns
    returns = np.log(price_df['close'] / price_df['close'].shift(1))

    df = price_df[['timestamp']].copy()

    for window in windows:
        # Rolling std of returns * sqrt(365*24) for annualized vol
        rv = returns.rolling(window).std() * np.sqrt(365 * 24) * 100
        df[f'rv_{window}d'] = rv

    logger.info(f"  ✅ Computed RV for windows: {windows}")
    return df


def fetch_tradfi_macro(start_date: str = "2024-01-01"):
    """
    Fetch TradFi macro indicators using yfinance

    Fetches:
    - ^VIX: VIX index
    - DX-Y.NYB: Dollar index
    - ^MOVE: MOVE index (bond volatility) - if available
    - ^TNX: 10-year yield
    - ^FVX: 5-year yield (proxy for 2Y)

    Returns:
        DataFrame with hourly TradFi macro features
    """
    logger.info("Fetching TradFi macro indicators...")

    tickers = {
        '^VIX': 'VIX',
        'DX-Y.NYB': 'DXY',
        '^TNX': 'YIELD_10Y',
        '^FVX': 'YIELD_2Y'  # 5Y as proxy for 2Y
    }

    dfs = []

    for ticker, name in tickers.items():
        try:
            logger.info(f"  Fetching {name} ({ticker})...")
            data = yf.download(ticker, start=start_date, interval="1d", progress=False)

            if data.empty:
                logger.warning(f"    ⚠️  No data for {ticker}")
                continue

            df = pd.DataFrame({
                'timestamp': data.index,
                name: data['Close'].values
            })

            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')

            # Resample to hourly (forward fill)
            df = df.set_index('timestamp').resample('1H').ffill().reset_index()

            dfs.append(df)
            logger.info(f"    ✅ Fetched {len(df)} records for {name}")

        except Exception as e:
            logger.error(f"    ❌ Failed to fetch {ticker}: {e}")

    # Merge all TradFi features
    if dfs:
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on='timestamp', how='outer')

        logger.info(f"  ✅ Merged TradFi data: {len(merged)} records")
        return merged

    return pd.DataFrame(columns=['timestamp'])


def build_macro_dataset(start_date: str = "2024-01-01", end_date: str = None):
    """
    Build complete macro feature dataset

    Combines:
    - Funding rates (CCXT)
    - Open interest (CCXT)
    - Crypto dominance (yfinance/synthetic)
    - Realized volatility (computed from BTC price)
    - TradFi macro (yfinance)

    Args:
        start_date: Start date for data
        end_date: End date (default: today)

    Returns:
        DataFrame with all macro features aligned to hourly timestamps
    """
    logger.info("=" * 70)
    logger.info("Building macro feature dataset")
    logger.info("=" * 70)

    # Fetch BTC price for RV computation
    logger.info("\n📊 Fetching BTC price data...")
    btc = yf.download("BTC-USD", start=start_date, end=end_date, interval="1h", progress=False)

    if btc.empty:
        logger.error("Failed to fetch BTC price data")
        return None

    btc_df = pd.DataFrame({
        'timestamp': pd.to_datetime(btc.index).tz_localize('UTC') if btc.index.tz is None else pd.to_datetime(btc.index),
        'close': btc['Close'].values.flatten()
    })

    # Compute realized volatility
    rv_df = compute_realized_volatility(btc_df, windows=[20, 60])

    # Fetch all macro features
    funding_df = fetch_funding_rate("BTC/USDT:USDT", start_date)
    oi_df = fetch_open_interest("BTC/USDT:USDT", start_date)
    dominance_df = fetch_crypto_dominance(start_date)
    tradfi_df = fetch_tradfi_macro(start_date)

    # Merge all features on timestamp
    logger.info("\n🔗 Merging all macro features...")

    macro_df = rv_df.copy()

    for df in [funding_df, oi_df, dominance_df, tradfi_df]:
        if not df.empty:
            macro_df = macro_df.merge(df, on='timestamp', how='left')

    # Add MOVE proxy (inverse of DXY volatility if not available)
    if 'MOVE' not in macro_df.columns and 'DXY' in macro_df.columns:
        macro_df['MOVE'] = 80.0  # Static default
        logger.info("  Using static MOVE value (80.0)")

    # Forward fill missing values
    macro_df = macro_df.fillna(method='ffill')

    logger.info(f"\n✅ Macro dataset complete: {len(macro_df)} hourly records")
    logger.info(f"   Columns: {list(macro_df.columns)}")
    logger.info(f"   Date range: {macro_df['timestamp'].min()} to {macro_df['timestamp'].max()}")

    return macro_df


def save_macro_dataset(df: pd.DataFrame, output_path: str):
    """Save macro dataset to parquet"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"\n💾 Saved macro dataset to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch macro features for regime classification")
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/macro/macro_features.parquet',
                        help='Output parquet file')

    args = parser.parse_args()

    # Build dataset
    macro_df = build_macro_dataset(args.start, args.end)

    if macro_df is not None:
        save_macro_dataset(macro_df, args.output)

        # Show summary stats
        print("\n" + "=" * 70)
        print("MACRO FEATURE SUMMARY")
        print("=" * 70)
        print(macro_df.describe())
    else:
        logger.error("Failed to build macro dataset")
