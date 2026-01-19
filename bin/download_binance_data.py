#!/usr/bin/env python3
"""
Download historical OHLCV data from Binance API
Supports crypto pairs and macro data with automatic rate limiting
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys


class BinanceDataDownloader:
    """Download historical klines from Binance API"""

    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"  # Use Futures API (no geo-restrictions)

    def __init__(self, output_dir: str = "chart_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Download klines from Binance API

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1h', '4h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            limit: Bars per request (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        # Parse dates
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        if end_date:
            end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ms = int(datetime.now().timestamp() * 1000)

        all_data = []
        current_ms = start_ms

        print(f"📊 Downloading {symbol} {interval} data...")
        print(f"   Period: {start_date} to {end_date or 'now'}")

        request_count = 0
        while current_ms < end_ms:
            # Build request
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ms,
                'endTime': end_ms,
                'limit': limit
            }

            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_data.extend(data)
                current_ms = data[-1][0] + 1  # Next request starts after last timestamp

                request_count += 1
                print(f"   Fetched {len(all_data)} bars...", end='\r')

                # Rate limiting (1200 requests per minute = ~20 per second)
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"\n❌ API Error: {e}")
                if request_count > 0:
                    print(f"   Partial data saved: {len(all_data)} bars")
                    break
                else:
                    return None

        print(f"\n   ✅ Downloaded {len(all_data)} bars")

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert prices to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Keep essential columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save DataFrame to CSV in TradingView format"""
        if df is None or df.empty:
            print(f"❌ No data to save for {symbol}")
            return

        # Convert time to Unix timestamp for compatibility
        df_export = df.copy()
        df_export['time'] = (df_export['time'].astype(int) / 10**9).astype(int)

        # Create filename (match existing pattern)
        interval_map = {'1h': '60', '4h': '240', '1d': '1D'}
        interval_str = interval_map.get(interval, interval)

        filename = self.output_dir / f"BINANCE_{symbol}, {interval_str}.csv"

        df_export.to_csv(filename, index=False)
        print(f"💾 Saved to: {filename}")

        return filename


def main():
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data")
    parser.add_argument("--symbol", type=str, required=True, help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1h", choices=['1h', '4h', '1d'], help="Timeframe")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--output", type=str, default="chart_logs", help="Output directory")

    args = parser.parse_args()

    print("=" * 70)
    print("📥 Binance Historical Data Downloader")
    print("=" * 70)

    downloader = BinanceDataDownloader(output_dir=args.output)

    # Download data
    df = downloader.download_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end
    )

    if df is not None:
        downloader.save_to_csv(df, args.symbol, args.interval)
        print("\n✅ Download complete!")
    else:
        print("\n❌ Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
