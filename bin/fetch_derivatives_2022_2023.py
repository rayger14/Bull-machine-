#!/usr/bin/env python3
"""
Fetch ALL Derivatives Data for BTC 2022-2023

Fetches from CoinGlass:
1. Funding rates (8-hourly)
2. Open Interest (hourly)
3. Long/Short ratio (hourly)
4. Liquidations (hourly)

Usage:
    export COINGLASS_API_KEY="your_key_here"
    python3 bin/fetch_derivatives_2022_2023.py
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path


class CoinGlassFetcher:
    """Rate-limited CoinGlass API fetcher."""

    def __init__(self, api_key: str, req_per_min: int = 10):
        self.api_key = api_key
        self.req_per_min = req_per_min
        self.sleep_time = 60.0 / req_per_min
        self.request_count = 0
        self.start_time = time.time()

        self.base_url = "https://open-api-v4.coinglass.com/api/futures"
        self.headers = {
            "CG-API-KEY": api_key,
            "accept": "application/json"
        }

    def _make_request(self, endpoint: str, params: dict, max_retries: int = 5) -> dict:
        """Make rate-limited API request with exponential backoff."""
        url = f"{self.base_url}/{endpoint}"

        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)

                if response.status_code != 200:
                    print(f"   ❌ HTTP {response.status_code}")
                    backoff = min(60, 2 ** retry)
                    print(f"   ⏳ Retry {retry+1}/{max_retries}, waiting {backoff}s")
                    time.sleep(backoff)
                    continue

                data = response.json()

                # Check for rate limit
                if data.get("code") == "400" and "Too Many Requests" in data.get("msg", ""):
                    print(f"   ⚠️  Rate limited!")
                    backoff = min(120, 2 ** retry * 10)
                    print(f"   ⏳ Backing off {backoff}s")
                    time.sleep(backoff)
                    continue

                # Check for API error
                if data.get("code") != "0":
                    error_msg = data.get('msg', 'Unknown error')
                    print(f"   ❌ API Error: {error_msg}")
                    print(f"   📋 Full response: {data}")
                    if retry == max_retries - 1:
                        return None
                    time.sleep(2 ** retry)
                    continue

                # Success
                self.request_count += 1
                elapsed = time.time() - self.start_time
                rate = self.request_count / (elapsed / 60) if elapsed > 0 else 0
                print(f"   ✅ Success (Rate: {rate:.1f} req/min)")

                # Rate limit sleep
                time.sleep(self.sleep_time)
                return data

            except Exception as e:
                print(f"   ❌ Error: {e}")
                if retry == max_retries - 1:
                    return None
                time.sleep(min(30, 2 ** retry))

        return None

    def fetch_funding_rates(self, symbol: str, exchange: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical funding rates."""
        print(f"\n{'='*80}")
        print(f"📊 Fetching Funding Rates")
        print(f"{'='*80}")
        print(f"Symbol: {symbol} | Exchange: {exchange}")
        print(f"Period: {start_date} to {end_date}")

        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_data = []
        current_end = end_ts
        batch_num = 0

        while current_end > start_ts:
            batch_num += 1
            params = {
                "symbol": symbol,
                "exchange": exchange,
                "interval": "h8",  # 8-hour funding rate intervals
                "limit": 1000,
                "endTime": current_end
            }

            current_date = datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')
            print(f"\n📡 Batch #{batch_num} (endTime={current_date})")

            result = self._make_request("funding-rate/history", params)

            if not result or not result.get("data"):
                print(f"   ℹ️  No more data available")
                break

            records = result["data"]
            print(f"   Got {len(records)} records")
            all_data.extend(records)

            # Get oldest timestamp
            oldest_ts = min(int(r['time']) for r in records)
            oldest_date = datetime.fromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')
            print(f"   Oldest: {oldest_date}")

            if oldest_ts <= start_ts:
                print(f"   ✅ Reached target start date")
                break

            # Check for pagination stall
            if oldest_ts >= current_end - 86400000:
                print(f"   ⚠️  Pagination stalled - may have hit data limit")
                break

            current_end = oldest_ts - 1

        if not all_data:
            print(f"   ❌ No funding data fetched!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['time'].astype(int), unit='ms', utc=True)
        df['funding_rate'] = df['close'].astype(float)
        df = df[['timestamp', 'funding_rate']].copy()
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        print(f"\n✅ Fetched {len(df)} funding rate records")
        print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def fetch_open_interest(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical open interest (aggregated across exchanges)."""
        print(f"\n{'='*80}")
        print(f"📊 Fetching Open Interest (Aggregated)")
        print(f"{'='*80}")
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")

        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_data = []
        current_end = end_ts
        batch_num = 0

        while current_end > start_ts:
            batch_num += 1
            params = {
                "symbol": symbol,
                "interval": "h1",  # Hourly OI
                "limit": 1000,
                "endTime": current_end
            }

            current_date = datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')
            print(f"\n📡 Batch #{batch_num} (endTime={current_date})")

            result = self._make_request("openInterest/ohlc-aggregated-history", params)

            if not result or not result.get("data"):
                print(f"   ℹ️  No more data available")
                break

            records = result["data"]
            print(f"   Got {len(records)} records")
            all_data.extend(records)

            # Get oldest timestamp
            oldest_ts = min(int(r['t']) for r in records)
            oldest_date = datetime.fromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')
            print(f"   Oldest: {oldest_date}")

            if oldest_ts <= start_ts:
                print(f"   ✅ Reached target start date")
                break

            if oldest_ts >= current_end - 86400000:
                print(f"   ⚠️  Pagination stalled")
                break

            current_end = oldest_ts - 1

        if not all_data:
            print(f"   ❌ No OI data fetched!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['t'].astype(int), unit='ms', utc=True)
        df['oi'] = df['c'].astype(float)  # Close value = OI in USD
        df = df[['timestamp', 'oi']].copy()
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        print(f"\n✅ Fetched {len(df)} OI records")
        print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   OI range: ${df['oi'].min():,.0f} to ${df['oi'].max():,.0f}")

        return df

    def fetch_long_short_ratio(self, symbol: str, exchange: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch long/short account ratio."""
        print(f"\n{'='*80}")
        print(f"📊 Fetching Long/Short Ratio")
        print(f"{'='*80}")
        print(f"Symbol: {symbol} | Exchange: {exchange}")
        print(f"Period: {start_date} to {end_date}")

        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_data = []
        current_end = end_ts
        batch_num = 0

        while current_end > start_ts:
            batch_num += 1
            params = {
                "symbol": symbol,
                "exchange": exchange,
                "interval": "h1",
                "limit": 1000,
                "endTime": current_end
            }

            current_date = datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')
            print(f"\n📡 Batch #{batch_num} (endTime={current_date})")

            result = self._make_request("longShortRatio/history", params)

            if not result or not result.get("data"):
                print(f"   ℹ️  No more data available")
                break

            records = result["data"]
            print(f"   Got {len(records)} records")
            all_data.extend(records)

            oldest_ts = min(int(r['t']) for r in records)
            oldest_date = datetime.fromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')
            print(f"   Oldest: {oldest_date}")

            if oldest_ts <= start_ts:
                print(f"   ✅ Reached target start date")
                break

            if oldest_ts >= current_end - 86400000:
                print(f"   ⚠️  Pagination stalled")
                break

            current_end = oldest_ts - 1

        if not all_data:
            print(f"   ❌ No L/S ratio data fetched!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['t'].astype(int), unit='ms', utc=True)
        df['ls_ratio'] = df['longShortRatio'].astype(float)
        df['long_pct'] = df['longRate'].astype(float)
        df['short_pct'] = df['shortRate'].astype(float)
        df = df[['timestamp', 'ls_ratio', 'long_pct', 'short_pct']].copy()
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        print(f"\n✅ Fetched {len(df)} L/S ratio records")
        print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def fetch_liquidations(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch liquidation data (aggregated across exchanges)."""
        print(f"\n{'='*80}")
        print(f"📊 Fetching Liquidations (Aggregated)")
        print(f"{'='*80}")
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")

        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_data = []
        current_end = end_ts
        batch_num = 0

        while current_end > start_ts:
            batch_num += 1
            params = {
                "symbol": symbol,
                "interval": "h1",
                "limit": 1000,
                "endTime": current_end
            }

            current_date = datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')
            print(f"\n📡 Batch #{batch_num} (endTime={current_date})")

            result = self._make_request("liquidation/history", params)

            if not result or not result.get("data"):
                print(f"   ℹ️  No more data available")
                break

            records = result["data"]
            print(f"   Got {len(records)} records")
            all_data.extend(records)

            oldest_ts = min(int(r['t']) for r in records)
            oldest_date = datetime.fromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')
            print(f"   Oldest: {oldest_date}")

            if oldest_ts <= start_ts:
                print(f"   ✅ Reached target start date")
                break

            if oldest_ts >= current_end - 86400000:
                print(f"   ⚠️  Pagination stalled")
                break

            current_end = oldest_ts - 1

        if not all_data:
            print(f"   ❌ No liquidation data fetched!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['t'].astype(int), unit='ms', utc=True)
        df['liq_long_usd'] = df['longLiquidationUsd'].astype(float)
        df['liq_short_usd'] = df['shortLiquidationUsd'].astype(float)
        df['liq_total_usd'] = df['liq_long_usd'] + df['liq_short_usd']
        df = df[['timestamp', 'liq_long_usd', 'liq_short_usd', 'liq_total_usd']].copy()
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        print(f"\n✅ Fetched {len(df)} liquidation records")
        print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df


def main():
    """Fetch all derivatives data for BTC 2022-2023."""

    # Try to load from .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"📄 Loading API key from .env file...")
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    if key.strip() == 'COINGLASS_API_KEY':
                        os.environ['COINGLASS_API_KEY'] = value.strip()
                        break

    # Get API key from environment
    api_key = os.getenv("COINGLASS_API_KEY")
    if not api_key:
        print("❌ Error: COINGLASS_API_KEY not found")
        print("\nOptions:")
        print("  1. Create .env file with: COINGLASS_API_KEY=your_key")
        print("  2. Export: export COINGLASS_API_KEY='your_key_here'")
        return 1

    print(f"\n{'='*80}")
    print(f"🚀 FETCHING BTC DERIVATIVES DATA (2022-2023)")
    print(f"{'='*80}")

    # Conservative rate limiting for paid plan (10 req/min)
    fetcher = CoinGlassFetcher(api_key, req_per_min=10)

    # Try both symbol formats
    symbol = "BTCUSDT"  # Changed from "BTC" to full pair format
    exchange = "Binance"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    output_dir = Path("data/derivatives")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Funding Rates
    funding_df = fetcher.fetch_funding_rates(symbol, exchange, start_date, end_date)
    if not funding_df.empty:
        funding_df.to_csv(output_dir / "BTC_funding_2022_2023.csv", index=False)
        print(f"\n💾 Saved: {output_dir}/BTC_funding_2022_2023.csv")

    # 2. Fetch Open Interest
    oi_df = fetcher.fetch_open_interest(symbol, start_date, end_date)
    if not oi_df.empty:
        oi_df.to_csv(output_dir / "BTC_oi_2022_2023.csv", index=False)
        print(f"\n💾 Saved: {output_dir}/BTC_oi_2022_2023.csv")

    # 3. Fetch Long/Short Ratio
    ls_df = fetcher.fetch_long_short_ratio(symbol, exchange, start_date, end_date)
    if not ls_df.empty:
        ls_df.to_csv(output_dir / "BTC_ls_ratio_2022_2023.csv", index=False)
        print(f"\n💾 Saved: {output_dir}/BTC_ls_ratio_2022_2023.csv")

    # 4. Fetch Liquidations
    liq_df = fetcher.fetch_liquidations(symbol, start_date, end_date)
    if not liq_df.empty:
        liq_df.to_csv(output_dir / "BTC_liquidations_2022_2023.csv", index=False)
        print(f"\n💾 Saved: {output_dir}/BTC_liquidations_2022_2023.csv")

    print(f"\n{'='*80}")
    print(f"✅ DERIVATIVES FETCH COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal API requests: {fetcher.request_count}")
    print(f"Total time: {(time.time() - fetcher.start_time)/60:.1f} minutes")
    print(f"\nNext step:")
    print(f"  python3 bin/patch_derivatives_full.py \\")
    print(f"    --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet")

    return 0


if __name__ == '__main__':
    exit(main())
