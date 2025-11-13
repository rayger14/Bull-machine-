#!/usr/bin/env python3
"""
Fetch Historical Funding Rates from CoinGlass API (Rate-Limited)

Respects Hobbyist plan limits: 30 req/min (we use 25 req/min to be safe).
Implements exponential backoff for rate limit errors.

Usage:
    python3 bin/fetch_coinglass_funding_v2.py --api-key YOUR_KEY --start 2022-01-01 --end 2024-12-31
"""

import requests
import pandas as pd
import argparse
from datetime import datetime
import time
from pathlib import Path


def fetch_funding_history_throttled(api_key: str, symbol: str, exchange: str,
                                     start_date: str, end_date: str,
                                     req_per_min: int = 25) -> pd.DataFrame:
    """
    Fetch historical funding rates with rate limiting and exponential backoff.

    Args:
        api_key: CoinGlass API key
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (e.g., "Binance")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        req_per_min: Max requests per minute (default 25, limit is 30)

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
    print(f"RATE-LIMITED FUNDING RATE FETCH")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Exchange: {exchange}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Rate limit: {req_per_min} requests/min")

    # Calculate sleep time between requests to stay under limit
    sleep_time = 60.0 / req_per_min  # seconds between requests
    print(f"Sleep between requests: {sleep_time:.2f}s")

    all_data = []
    batch_size = 1000
    current_end = end_ts
    request_count = 0
    start_time = time.time()

    while current_end > start_ts:
        params = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": "1d",
            "limit": batch_size,
            "endTime": current_end
        }

        current_date = datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')
        print(f"\n📡 Request #{request_count + 1} (endTime={current_date})")

        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)

                if response.status_code != 200:
                    print(f"   ❌ HTTP {response.status_code}")
                    retry_count += 1
                    backoff = min(60, 2 ** retry_count)  # Exponential backoff, max 60s
                    print(f"   ⏳ Backing off {backoff}s (retry {retry_count}/{max_retries})")
                    time.sleep(backoff)
                    continue

                data = response.json()

                # Check for rate limit error
                if data.get("code") == "400" and "Too Many Requests" in data.get("msg", ""):
                    print(f"   ⚠️  Rate limited!")
                    retry_count += 1
                    backoff = min(60, 2 ** retry_count * 10)  # Longer backoff for rate limits
                    print(f"   ⏳ Backing off {backoff}s (retry {retry_count}/{max_retries})")
                    time.sleep(backoff)
                    continue

                # Check for API error
                if data.get("code") != "0":
                    print(f"   ❌ API Error: {data}")
                    break

                records = data.get("data", [])

                if not records:
                    print(f"   ℹ️  No more data available")
                    break

                # Success!
                request_count += 1
                elapsed = time.time() - start_time
                rate = request_count / (elapsed / 60)  # requests per minute

                print(f"   ✅ Got {len(records)} records")
                print(f"   📊 Rate: {rate:.1f} req/min (target: {req_per_min})")

                # Add to collection
                all_data.extend(records)

                # Get oldest timestamp from this batch
                oldest_ts = min(int(r['time']) for r in records)
                oldest_date = datetime.fromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')
                print(f"   📅 Oldest in batch: {oldest_date}")

                # Stop if we've gone back far enough
                if oldest_ts <= start_ts:
                    print(f"   ✅ Reached target start date")
                    break

                # Check if we're stuck (same oldest date)
                if oldest_ts >= current_end - 86400000:  # Less than 1 day progress
                    print(f"   ⚠️  Pagination stalled - may have hit data availability limit")
                    break

                # Move window back
                current_end = oldest_ts - 1

                # Rate limiting: sleep to maintain target rate
                time.sleep(sleep_time)
                break  # Break retry loop on success

            except Exception as e:
                print(f"   ❌ Error: {e}")
                retry_count += 1
                backoff = min(30, 2 ** retry_count)
                print(f"   ⏳ Backing off {backoff}s (retry {retry_count}/{max_retries})")
                time.sleep(backoff)

        # If max retries exceeded, stop
        if retry_count >= max_retries:
            print(f"\n❌ Max retries exceeded, stopping")
            break

    print(f"\n{'='*80}")
    print(f"📊 Fetch Summary:")
    print(f"   Total records: {len(all_data)}")
    print(f"   Total requests: {request_count}")
    print(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")

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

    # Remove duplicates (keep most recent)
    df = df.drop_duplicates(subset=['timestamp'], keep='last')

    # Resample to daily (take mean if multiple readings per day)
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date')['funding'].mean().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.rename(columns={'date': 'timestamp'})

    # Filter to requested date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    daily = daily[(daily['timestamp'] >= start_dt) & (daily['timestamp'] <= end_dt)].copy()

    print(f"\n📅 Final dataset:")
    print(f"   Date range: {daily['timestamp'].min()} to {daily['timestamp'].max()}")
    print(f"   Daily records: {len(daily)}")
    print(f"   Funding rate stats:")
    print(f"      Min:  {daily['funding'].min():.6f}")
    print(f"      Max:  {daily['funding'].max():.6f}")
    print(f"      Mean: {daily['funding'].mean():.6f}")
    print(f"      NaN:  {daily['funding'].isna().sum()}")

    return daily


def main():
    parser = argparse.ArgumentParser(description='Fetch historical funding rates (rate-limited)')
    parser.add_argument('--api-key', type=str, required=True, help='CoinGlass API key')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--exchange', type=str, default='Binance', help='Exchange')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--output', type=str, default='data/FUNDING_1D.csv', help='Output CSV file')
    parser.add_argument('--req-per-min', type=int, default=25, help='Max requests/min (default: 25)')

    args = parser.parse_args()

    # Fetch data with rate limiting
    df = fetch_funding_history_throttled(
        api_key=args.api_key,
        symbol=args.symbol,
        exchange=args.exchange,
        start_date=args.start,
        end_date=args.end,
        req_per_min=args.req_per_min
    )

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")

    print(f"\n{'='*80}")
    print(f"✅ FUNDING RATE FETCH COMPLETE")
    print(f"{'='*80}")

    # Check what we got vs what we wanted
    start_dt = pd.to_datetime(args.start)
    actual_start = df['timestamp'].min()

    if actual_start > start_dt:
        gap_days = (actual_start - start_dt).days
        print(f"\n⚠️  DATA GAP DETECTED:")
        print(f"   Requested: {args.start}")
        print(f"   Oldest fetched: {actual_start.strftime('%Y-%m-%d')}")
        print(f"   Gap: {gap_days} days")
        print(f"\n   This may indicate:")
        print(f"   1. Hobbyist plan historical data limit")
        print(f"   2. CoinGlass data retention policy")
        print(f"   3. Consider manual TradingView export for missing period")

    return 0


if __name__ == '__main__':
    exit(main())
