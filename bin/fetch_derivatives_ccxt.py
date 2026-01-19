#!/usr/bin/env python3
"""
Fetch Funding Rate & OI for BTC 2022-2023 via CCXT

Uses Binance USDT-M futures (most liquid, reliable historical data).

Usage:
    python3 bin/fetch_derivatives_ccxt.py
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path


def fetch_funding_rate_history(exchange, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch funding rate history using CCXT standardized methods.

    Funding rate is 8-hourly (00:00, 08:00, 16:00 UTC).
    """
    print(f"\n{'='*80}")
    print(f"📊 Fetching Funding Rate History")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")

    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

    all_data = []
    current_ts = start_ts
    batch = 0

    while current_ts < end_ts:
        batch += 1
        try:
            print(f"\n📡 Batch #{batch} (from {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d')})")

            # Use CCXT's fetch_funding_rate_history method
            funding_rates = exchange.fetch_funding_rate_history(
                symbol,
                since=current_ts,
                limit=1000
            )

            if not funding_rates:
                print(f"   ℹ️  No more data")
                break

            print(f"   ✅ Got {len(funding_rates)} records")
            all_data.extend(funding_rates)

            # Get latest timestamp for next batch
            latest_ts = funding_rates[-1]['timestamp']
            latest_date = datetime.fromtimestamp(latest_ts/1000).strftime('%Y-%m-%d')
            print(f"   Latest: {latest_date}")

            if latest_ts >= end_ts:
                print(f"   ✅ Reached end date")
                break

            current_ts = latest_ts + 1
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"   ❌ Error: {e}")
            if "Too many requests" in str(e) or "429" in str(e):
                print(f"   ⏳ Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            else:
                break

    if not all_data:
        print(f"   ❌ No funding data fetched!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['funding_rate'] = df['fundingRate'].astype(float)
    df = df[['timestamp', 'funding_rate']].copy()
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

    print(f"\n✅ Fetched {len(df)} funding rate records")
    print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Funding rate range: {df['funding_rate'].min():.4f} to {df['funding_rate'].max():.4f}")

    return df


def fetch_open_interest_history(exchange, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Open Interest history from Binance.

    Binance provides 5-minute OI data. We'll resample to 1H.
    """
    print(f"\n{'='*80}")
    print(f"📊 Fetching Open Interest History")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")

    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

    all_data = []
    current_ts = start_ts
    batch = 0

    while current_ts < end_ts:
        batch += 1
        try:
            # Binance OI endpoint (5m intervals, max 500 records per request)
            print(f"\n📡 Batch #{batch} (from {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d')})")

            oi_data = exchange.fapiPublicGetOpenInterestHist({
                'symbol': symbol.replace('/', ''),  # BTCUSDT
                'period': '5m',
                'startTime': current_ts,
                'endTime': min(current_ts + 300000 * 500, end_ts),  # 500 * 5min = ~1.7 days
                'limit': 500
            })

            if not oi_data:
                print(f"   ℹ️  No more data")
                break

            print(f"   ✅ Got {len(oi_data)} records")
            all_data.extend(oi_data)

            # Get latest timestamp for next batch
            latest_ts = int(oi_data[-1]['timestamp'])
            latest_date = datetime.fromtimestamp(latest_ts/1000).strftime('%Y-%m-%d')
            print(f"   Latest: {latest_date}")

            if latest_ts >= end_ts:
                print(f"   ✅ Reached end date")
                break

            current_ts = latest_ts + 1
            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            print(f"   ❌ Error: {e}")
            if "Too many requests" in str(e):
                print(f"   ⏳ Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            else:
                break

    if not all_data:
        print(f"   ❌ No OI data fetched!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
    df['oi'] = df['sumOpenInterest'].astype(float)  # Total OI in contracts
    df['oi_value'] = df['sumOpenInterestValue'].astype(float)  # OI in USD

    # Resample to 1H (taking last value in each hour)
    df = df.set_index('timestamp')
    df_1h = df[['oi', 'oi_value']].resample('1h').last().reset_index()
    df_1h = df_1h.dropna()

    print(f"\n✅ Fetched {len(df)} OI records (5m)")
    print(f"   Resampled to {len(df_1h)} hourly records")
    print(f"   Range: {df_1h['timestamp'].min()} to {df_1h['timestamp'].max()}")
    print(f"   OI range: {df_1h['oi_value'].min():,.0f} to {df_1h['oi_value'].max():,.0f} USD")

    return df_1h


def main():
    """Fetch BTC funding rate and OI for 2022-2023."""

    print(f"\n{'='*80}")
    print(f"🚀 FETCHING BTC DERIVATIVES VIA CCXT (2022-2023)")
    print(f"{'='*80}")

    # Try multiple exchanges (in case of geo-restrictions)
    exchanges_to_try = [
        ('bybit', ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'future'}})),
        ('okx', ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})),
        ('binance', ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})),
    ]

    exchange = None
    for name, exch in exchanges_to_try:
        try:
            print(f"\n🔍 Trying {name.upper()}...")
            # Test connection
            exch.fetch_ticker('BTC/USDT')
            exchange = exch
            print(f"   ✅ {name.upper()} accessible")
            break
        except Exception as e:
            print(f"   ❌ {name.upper()} failed: {str(e)[:100]}")
            continue

    if not exchange:
        print("\n❌ No accessible exchange found!")
        print("   All major exchanges blocked or unavailable.")
        return 1

    start_date = "2022-01-01"
    end_date = "2023-12-31"

    # Symbol format varies by exchange
    symbol_map = {
        'bybit': 'BTC/USDT:USDT',
        'okx': 'BTC/USDT:USDT',
        'binance': 'BTC/USDT',
    }

    symbol = symbol_map.get(exchange.id, 'BTC/USDT')
    print(f"\nUsing symbol format: {symbol} for {exchange.id}")

    output_dir = Path("data/derivatives")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Funding Rate
    funding_df = fetch_funding_rate_history(exchange, symbol, start_date, end_date)
    if not funding_df.empty:
        output_path = output_dir / "BTC_funding_2022_2023_ccxt.csv"
        funding_df.to_csv(output_path, index=False)
        print(f"\n💾 Saved: {output_path}")

    # 2. Fetch Open Interest (if supported by exchange)
    print(f"\n{'='*80}")
    print(f"📊 Open Interest - Checking if supported...")
    print(f"{'='*80}")

    if hasattr(exchange, 'fetch_open_interest_history'):
        try:
            oi_df = fetch_open_interest_history(exchange, symbol, start_date, end_date)
            if not oi_df.empty:
                output_path = output_dir / "BTC_oi_2022_2023_ccxt.csv"
                oi_df.to_csv(output_path, index=False)
                print(f"\n💾 Saved: {output_path}")
        except Exception as e:
            print(f"   ⚠️  OI fetch failed: {e}")
            print(f"   Continuing with funding rate only...")
            oi_df = pd.DataFrame()
    else:
        print(f"   ⚠️  OI history not supported by {exchange.id}")
        print(f"   Continuing with funding rate only (primary signal)...")
        oi_df = pd.DataFrame()

    print(f"\n{'='*80}")
    print(f"✅ DERIVATIVES FETCH COMPLETE (CCXT)")
    print(f"{'='*80}")

    if not funding_df.empty and not oi_df.empty:
        print(f"\n📊 Summary:")
        print(f"   Funding Rate: {len(funding_df)} records")
        print(f"   Open Interest: {len(oi_df)} records")
        print(f"\nNext step:")
        print(f"  python3 bin/patch_derivatives_full.py \\")
        print(f"    --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \\")
        print(f"    --funding data/derivatives/BTC_funding_2022_2023_ccxt.csv \\")
        print(f"    --oi data/derivatives/BTC_oi_2022_2023_ccxt.csv")

    return 0


if __name__ == '__main__':
    exit(main())
