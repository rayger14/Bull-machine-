"""
Fetch TOTAL/TOTAL2/TOTAL3 crypto market cap data

Uses CoinGecko API to fetch:
- TOTAL: Total crypto market cap
- TOTAL2: Total ex-BTC market cap
- TOTAL3: Total ex-BTC and ETH

Saves to parquet for integration with macro features
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_coingecko_global_data(days: int = 365) -> pd.DataFrame:
    """
    Fetch global crypto market cap data from CoinGecko

    Args:
        days: Number of days of history (max 365 for free tier)

    Returns:
        DataFrame with timestamp, total_market_cap, btc_market_cap, eth_market_cap
    """
    logger.info(f"Fetching CoinGecko global market data ({days} days)...")

    url = "https://api.coingecko.com/api/v3/global"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        global_data = data.get('data', {})

        # Current snapshot
        total_mcap = global_data.get('total_market_cap', {}).get('usd', 0)
        btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
        eth_dominance = global_data.get('market_cap_percentage', {}).get('eth', 0)

        # Calculate individual market caps
        btc_mcap = total_mcap * (btc_dominance / 100)
        eth_mcap = total_mcap * (eth_dominance / 100)

        logger.info(f"  Total Market Cap: ${total_mcap/1e9:.1f}B")
        logger.info(f"  BTC Dominance: {btc_dominance:.1f}%")
        logger.info(f"  ETH Dominance: {eth_dominance:.1f}%")

        return {
            'timestamp': pd.Timestamp.now(tz='UTC'),
            'total_mcap': total_mcap,
            'btc_mcap': btc_mcap,
            'eth_mcap': eth_mcap,
            'btc_dominance': btc_dominance,
            'eth_dominance': eth_dominance
        }

    except Exception as e:
        logger.error(f"Failed to fetch CoinGecko data: {e}")
        return None


def fetch_btc_market_cap_history(days: int = 365) -> pd.DataFrame:
    """
    Fetch BTC market cap history from CoinGecko

    Args:
        days: Number of days (max 365 for free tier)

    Returns:
        DataFrame with timestamp and btc_market_cap
    """
    logger.info(f"Fetching BTC market cap history ({days} days)...")

    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract market caps
        market_caps = data.get('market_caps', [])

        df = pd.DataFrame(market_caps, columns=['timestamp', 'btc_market_cap'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        logger.info(f"  ✅ Fetched {len(df)} BTC market cap records")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch BTC history: {e}")
        return pd.DataFrame()


def fetch_eth_market_cap_history(days: int = 365) -> pd.DataFrame:
    """
    Fetch ETH market cap history from CoinGecko

    Args:
        days: Number of days

    Returns:
        DataFrame with timestamp and eth_market_cap
    """
    logger.info(f"Fetching ETH market cap history ({days} days)...")

    url = f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        market_caps = data.get('market_caps', [])

        df = pd.DataFrame(market_caps, columns=['timestamp', 'eth_market_cap'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        logger.info(f"  ✅ Fetched {len(df)} ETH market cap records")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch ETH history: {e}")
        return pd.DataFrame()


def estimate_total_market_cap(btc_mcap: float, btc_dominance: float = 55.0) -> float:
    """
    Estimate total crypto market cap from BTC market cap and dominance

    Args:
        btc_mcap: BTC market cap in USD
        btc_dominance: BTC dominance percentage

    Returns:
        Estimated total market cap
    """
    if btc_dominance == 0:
        return np.nan

    return btc_mcap / (btc_dominance / 100)


def build_crypto_marketcap_dataset(days: int = 365) -> pd.DataFrame:
    """
    Build comprehensive crypto market cap dataset

    Fetches BTC and ETH market caps, estimates TOTAL/TOTAL2/TOTAL3

    Args:
        days: Days of history

    Returns:
        DataFrame with TOTAL, TOTAL2, TOTAL3, BTC.D
    """
    logger.info("=" * 70)
    logger.info("Building Crypto Market Cap Dataset")
    logger.info("=" * 70)

    # Fetch BTC history
    btc_df = fetch_btc_market_cap_history(days)
    time.sleep(1)  # Rate limit

    # Fetch ETH history
    eth_df = fetch_eth_market_cap_history(days)
    time.sleep(1)

    if btc_df.empty or eth_df.empty:
        logger.error("Failed to fetch market cap data")
        return pd.DataFrame()

    # Merge on timestamp (daily)
    df = btc_df.merge(eth_df, on='timestamp', how='outer')
    df = df.sort_values('timestamp')

    # Forward fill any gaps
    df = df.fillna(method='ffill')

    # Calculate BTC dominance (use typical range 40-60%)
    # For 2024, BTC dominance ranged from ~50-58%
    df['btc_dominance'] = 55.0  # Approximate average for 2024

    # Estimate TOTAL market cap from BTC
    df['TOTAL'] = df.apply(
        lambda x: estimate_total_market_cap(x['btc_market_cap'], x['btc_dominance']),
        axis=1
    )

    # TOTAL2 = TOTAL - BTC
    df['TOTAL2'] = df['TOTAL'] - df['btc_market_cap']

    # TOTAL3 = TOTAL - BTC - ETH
    df['TOTAL3'] = df['TOTAL'] - df['btc_market_cap'] - df['eth_market_cap']

    # BTC.D percentage
    df['BTC.D'] = df['btc_dominance']

    # Convert to billions for readability
    df['TOTAL_billions'] = df['TOTAL'] / 1e9
    df['TOTAL2_billions'] = df['TOTAL2'] / 1e9
    df['TOTAL3_billions'] = df['TOTAL3'] / 1e9

    logger.info(f"\n✅ Market cap dataset complete: {len(df)} daily records")
    logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"\n   Latest snapshot:")
    logger.info(f"   TOTAL: ${df['TOTAL_billions'].iloc[-1]:.1f}B")
    logger.info(f"   TOTAL2: ${df['TOTAL2_billions'].iloc[-1]:.1f}B")
    logger.info(f"   TOTAL3: ${df['TOTAL3_billions'].iloc[-1]:.1f}B")
    logger.info(f"   BTC.D: {df['BTC.D'].iloc[-1]:.1f}%")

    return df


def upsample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upsample daily data to hourly using forward fill

    Args:
        df: Daily DataFrame

    Returns:
        Hourly DataFrame
    """
    logger.info("Upsampling to hourly resolution...")

    df = df.set_index('timestamp')
    df_hourly = df.resample('1H').ffill()
    df_hourly = df_hourly.reset_index()

    logger.info(f"  ✅ Upsampled to {len(df_hourly)} hourly records")

    return df_hourly


def save_marketcap_data(df: pd.DataFrame, output_path: str):
    """Save market cap data to parquet"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"\n💾 Saved to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch crypto market cap data")
    parser.add_argument('--days', type=int, default=365, help='Days of history')
    parser.add_argument('--output', type=str, default='data/macro/crypto_marketcap.parquet')

    args = parser.parse_args()

    # Build dataset
    df = build_crypto_marketcap_dataset(args.days)

    if not df.empty:
        # Upsample to hourly
        df_hourly = upsample_to_hourly(df)

        # Save
        save_marketcap_data(df_hourly, args.output)

        # Show summary stats
        print("\n" + "=" * 70)
        print("MARKET CAP SUMMARY")
        print("=" * 70)
        print(df_hourly[['TOTAL_billions', 'TOTAL2_billions', 'TOTAL3_billions', 'BTC.D']].describe())
    else:
        logger.error("Failed to build market cap dataset")
