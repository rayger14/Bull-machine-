#!/usr/bin/env python3
"""
Fetch Historical Funding Rates from CoinGlass API

Fetches BTC funding rates from 2022-2023 for Router v10 regime detection.

Usage:
    python3 bin/fetch_coinglass_funding.py --api-key YOUR_KEY --start 2022-01-01 --end 2024-12-31
"""

import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
import time
from pathlib import Path


def fetch_funding_history(api_key: str, symbol: str, exchange: str,
                          start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical funding rates from CoinGlass API.

    Args:
        api_key: CoinGlass API key
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (e.g., "Binance")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with timestamp and funding rate
    """
    url = "https://open-api-v4.coinglass.com/api/futures/funding-rate/history"

    headers = {
        "CG-API-KEY": api_key,
        "accept": "application/json"
    }

    # Convert dates to timestamps (milliseconds)
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

    print(f"\n{'='*80}")
    print(f"FETCHING FUNDING RATES FROM COINGLASS")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Exchange: {exchange}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timestamps: {start_ts} to {end_ts}")

    all_data = []
    batch_size = 1000  # CoinGlass max limit
    current_end = end_ts

    while current_end > start_ts:
        params = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": "1d",  # Daily data
            "limit": batch_size,
            "endTime": current_end
        }

        print(f"\n📡 Fetching batch (endTime={datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')})")

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                break

            data = response.json()

            if data.get("code") != "0":
                print(f"❌ API Error: {data}")
                break

            records = data.get("data", [])

            if not records:
                print(f"⚠️  No more data available")
                break

            print(f"   ✅ Got {len(records)} records")

            # Add to collection
            all_data.extend(records)

            # Get oldest timestamp from this batch
            oldest_ts = min(int(r['time']) for r in records)
            oldest_date = datetime.fromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')
            print(f"   Oldest in batch: {oldest_date}")

            # Stop if we've gone back far enough
            if oldest_ts <= start_ts:
                print(f"   ✅ Reached start date")
                break

            # Move window back
            current_end = oldest_ts - 1

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    print(f"\n{'='*80}")
    print(f"📊 Total records fetched: {len(all_data)}")

    if not all_data:
        raise ValueError("No data fetched!")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='ms')

    # Extract funding rate (use 'close' as the funding rate value)
    df['funding'] = df['close'].astype(float)

    # Keep only time and funding
    df = df[['time', 'funding']].copy()
    df = df.rename(columns={'time': 'timestamp'})

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Filter to requested date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()

    print(f"\n📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📊 Final records: {len(df)}")
    print(f"💰 Funding rate stats:")
    print(f"   Min:  {df['funding'].min():.6f}")
    print(f"   Max:  {df['funding'].max():.6f}")
    print(f"   Mean: {df['funding'].mean():.6f}")
    print(f"   NaN:  {df['funding'].isna().sum()} ({100*df['funding'].isna().sum()/len(df):.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch historical funding rates from CoinGlass')
    parser.add_argument('--api-key', type=str, required=True, help='CoinGlass API key')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--exchange', type=str, default='Binance', help='Exchange (default: Binance)')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--output', type=str, default='data/FUNDING_1D.csv', help='Output CSV file')

    args = parser.parse_args()

    # Fetch data
    df = fetch_funding_history(
        api_key=args.api_key,
        symbol=args.symbol,
        exchange=args.exchange,
        start_date=args.start,
        end_date=args.end
    )

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    print(f"   {len(df)} records")

    print(f"\n{'='*80}")
    print(f"✅ FUNDING RATE DATA FETCH COMPLETE")
    print(f"{'='*80}")

    print(f"\n📝 Next steps:")
    print(f"1. Rebuild 2022-2023 feature store:")
    print(f"   python3 bin/build_mtf_feature_store.py --asset BTC --start 2022-01-01 --end 2023-12-31")
    print(f"\n2. Run Router v10 backtest:")
    print(f"   python3 bin/backtest_router_v10.py --asset BTC --start 2022-01-01 --end 2023-12-31")

    return 0


if __name__ == '__main__':
    exit(main())
