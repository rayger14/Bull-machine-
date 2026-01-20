#!/usr/bin/env python3
"""
Download historical OHLCV data from CryptoCompare API (FREE, no geo-restrictions)
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys


class CryptoCompareDownloader:
    """Download historical data from CryptoCompare API"""

    BASE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"

    def __init__(self, output_dir: str = "chart_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_hourly(
        self,
        symbol: str,
        start_date: str,
        end_date: str = None,
        limit: int = 2000
    ) -> pd.DataFrame:
        """
        Download hourly OHLCV data

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            limit: Bars per request (max 2000)

        Returns:
            DataFrame with OHLCV data
        """
        # Parse dates
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        else:
            end_ts = int(datetime.now().timestamp())

        all_data = []
        current_ts = start_ts

        print(f"📊 Downloading {symbol}USDT hourly data...")
        print(f"   Period: {start_date} to {end_date or 'now'}")

        request_count = 0
        while current_ts < end_ts:
            # Build request
            params = {
                'fsym': symbol,
                'tsym': 'USDT',
                'limit': limit,
                'toTs': min(current_ts + (limit * 3600), end_ts)
            }

            try:
                response = requests.get(self.BASE_URL, params=params, timeout=15)
                response.raise_for_status()
                result = response.json()

                if result['Response'] != 'Success':
                    print(f"\n❌ API Error: {result.get('Message', 'Unknown error')}")
                    break

                data = result['Data']['Data']
                if not data:
                    break

                all_data.extend(data)
                current_ts = data[-1]['time'] + 3600  # Next hour

                request_count += 1
                print(f"   Fetched {len(all_data)} hours...", end='\r')

                # Rate limiting (avoid hitting limits)
                time.sleep(0.2)

                # Stop if we've reached the end
                if len(data) < limit:
                    break

            except requests.exceptions.RequestException as e:
                print(f"\n❌ Network Error: {e}")
                if request_count > 0:
                    print(f"   Partial data saved: {len(all_data)} bars")
                    break
                else:
                    return None

        print(f"\n   ✅ Downloaded {len(all_data)} hours")

        if not all_data:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Rename columns to match TradingView format
        df = df.rename(columns={
            'time': 'timestamp',
            'volumefrom': 'volume'
        })

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')

        # Keep essential columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str = "60"):
        """Save DataFrame to CSV in TradingView format"""
        if df is None or df.empty:
            print(f"❌ No data to save for {symbol}")
            return None

        # Convert time to Unix timestamp
        df_export = df.copy()
        df_export['time'] = (df_export['time'].astype(int) / 10**9).astype(int)

        # Create filename
        filename = self.output_dir / f"CRYPTOCOMPARE_{symbol}USDT, {interval}.csv"

        df_export.to_csv(filename, index=False)
        print(f"💾 Saved to: {filename}")

        return filename


def main():
    parser = argparse.ArgumentParser(description="Download CryptoCompare OHLCV data")
    parser.add_argument("--symbol", type=str, required=True, help="Crypto symbol (BTC, ETH, SOL)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--output", type=str, default="chart_logs", help="Output directory")

    args = parser.parse_args()

    print("=" * 70)
    print("📥 CryptoCompare Historical Data Downloader (FREE API)")
    print("=" * 70)

    downloader = CryptoCompareDownloader(output_dir=args.output)

    # Download hourly data
    df = downloader.download_hourly(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end
    )

    if df is not None and not df.empty:
        downloader.save_to_csv(df, args.symbol, interval="60")
        print(f"\n✅ Download complete! Got {len(df)} hours of data")
        print(f"   Duration: {(df['time'].max() - df['time'].min()).days} days")
    else:
        print("\n❌ Download failed or no data available")
        sys.exit(1)


if __name__ == "__main__":
    main()
