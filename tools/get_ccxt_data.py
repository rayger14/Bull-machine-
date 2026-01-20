#!/usr/bin/env python3
"""
Download historical OHLCV using CCXT (better geo-handling, retry logic)
"""
import ccxt
import pandas as pd
import time
import argparse
from datetime import datetime, timezone
from dateutil import parser as dtp
from tqdm import tqdm


def to_ms(dt_str):
    """Convert date string to UTC milliseconds"""
    dt = dtp.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_ohlcv(exchange_name, symbol, timeframe, start_ms, end_ms, limit=1000):
    """
    Fetch OHLCV data using CCXT

    Args:
        exchange_name: Exchange name (binance, bybit, coinbase, kraken)
        symbol: Trading pair (BTC/USDT, ETH/USDT)
        timeframe: Timeframe (1h, 4h, 1d)
        start_ms: Start timestamp in ms
        end_ms: End timestamp in ms
        limit: Bars per request
    """
    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}  # Use spot market
    })

    all_ohlcv = []
    current_ms = start_ms

    print(f"📊 Downloading {symbol} {timeframe} from {exchange_name}...")

    with tqdm(total=0, unit="bars", desc=f"{symbol} {timeframe}") as pbar:
        while current_ms < end_ms:
            try:
                # Fetch data
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ms,
                    limit=limit
                )

                if not ohlcv:
                    break

                # Filter to end date
                ohlcv = [candle for candle in ohlcv if candle[0] <= end_ms]

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                pbar.update(len(ohlcv))

                # Move to next batch
                last_timestamp = ohlcv[-1][0]
                if last_timestamp == current_ms:
                    # Avoid infinite loop
                    break

                current_ms = last_timestamp + 1

                # Rate limiting (CCXT handles this but add safety)
                time.sleep(exchange.rateLimit / 1000)

                # Stop if we've caught up
                if last_timestamp >= end_ms:
                    break

            except ccxt.NetworkError as e:
                print(f"\n⚠️  Network error: {e}, retrying...")
                time.sleep(5)
                continue

            except ccxt.ExchangeError as e:
                print(f"\n❌ Exchange error: {e}")
                break

            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                break

    if not all_ohlcv:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    return df


def main():
    ap = argparse.ArgumentParser(description="Download OHLCV via CCXT")
    ap.add_argument("--exchange", required=True, help="Exchange: binance, bybit, coinbase, kraken")
    ap.add_argument("--symbol", required=True, help="Trading pair: BTC/USDT, ETH/USDT")
    ap.add_argument("--timeframe", default="1h", help="Timeframe: 1h, 4h, 1d")
    ap.add_argument("--start", required=True, help="Start date: 2023-01-01")
    ap.add_argument("--end", required=True, help="End date: 2025-10-13")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)

    df = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, start_ms, end_ms)

    if df.empty:
        print("❌ No data returned")
        return

    # Save to CSV
    df_export = df.copy()
    df_export['timestamp'] = (df_export['timestamp'].astype(int) / 10**9).astype(int)
    df_export = df_export.rename(columns={'timestamp': 'time'})
    df_export.to_csv(args.out, index=False)

    # Also save parquet
    df.set_index('timestamp').to_parquet(args.out.replace('.csv', '.parquet'))

    print(f"\n✅ Saved {len(df):,} bars → {args.out}")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")


if __name__ == "__main__":
    main()
