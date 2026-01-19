#!/usr/bin/env python3
"""
Backfill Missing Macro Features for GMM Regime Classifier

Fetches historical data and computes 6 missing features:
1. OI_CHANGE - Open Interest change from OKX
2. PERP_BASIS - Perpetual basis (OKX perp - Coinbase spot)
3. VOL_TERM - Volatility term structure (RV_30 / RV_7)
4. ALT_ROTATION - Altcoin rotation (TOTAL2_RET - TOTAL_RET)
5. TOTAL3_RET - Set to 0 for now (approximation)
6. SKEW_25D - Set to 0 for now (approximation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from tqdm import tqdm
import requests

def fetch_okx_historical_perp(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OKX perpetual futures data (hourly)"""
    print("\n=== Fetching OKX Perpetual Futures (BTC/USDT:USDT) ===")

    exchange = ccxt.okx()
    since = exchange.parse8601(f'{start_date}T00:00:00Z')
    end_ts = exchange.parse8601(f'{end_date}T23:59:59Z')

    all_data = []
    current_since = since

    while current_since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', since=current_since, limit=1000)
            if not ohlcv:
                break

            all_data.extend(ohlcv)
            current_since = ohlcv[-1][0] + 3600000  # Move to next hour

            print(f"  Fetched {len(all_data)} candles (up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc)})")
            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
            continue

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'close': 'perp_price'}, inplace=True)

    print(f"✅ Fetched {len(df)} OKX perp candles")
    return df[['perp_price']]

def fetch_coinbase_historical_spot(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Coinbase spot data (hourly)"""
    print("\n=== Fetching Coinbase Spot (BTC/USD) ===")

    exchange = ccxt.coinbase()
    since = exchange.parse8601(f'{start_date}T00:00:00Z')
    end_ts = exchange.parse8601(f'{end_date}T23:59:59Z')

    all_data = []
    current_since = since

    while current_since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', since=current_since, limit=300)
            if not ohlcv:
                break

            all_data.extend(ohlcv)
            current_since = ohlcv[-1][0] + 3600000

            print(f"  Fetched {len(all_data)} candles (up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc)})")
            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
            continue

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'close': 'spot_price'}, inplace=True)

    print(f"✅ Fetched {len(df)} Coinbase spot candles")
    return df[['spot_price']]

def fetch_okx_historical_oi(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OKX Open Interest history using REST API

    OKX provides historical OI through their REST API endpoint:
    GET /api/v5/public/open-interest-history
    """
    print("\n=== Fetching OKX Historical Open Interest ===")

    # OKX API endpoint
    base_url = "https://www.okx.com"
    endpoint = "/api/v5/public/open-interest-history"

    # Convert dates to timestamps (milliseconds)
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_data = []
    current_after = None  # Pagination cursor

    while True:
        try:
            # Build request parameters
            params = {
                'instId': 'BTC-USDT-SWAP',  # OKX perpetual contract ID
                'period': '1H',              # Hourly data
                'limit': '100'               # Max per request
            }

            if current_after:
                params['after'] = current_after

            # Make request
            response = requests.get(f"{base_url}{endpoint}", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data['code'] != '0':
                print(f"  ⚠️ API error: {data.get('msg', 'Unknown error')}")
                break

            records = data.get('data', [])
            if not records:
                break

            # Filter by date range
            filtered = [r for r in records if start_ts <= int(r['ts']) <= end_ts]
            all_data.extend(filtered)

            print(f"  Fetched {len(all_data)} OI records (up to {datetime.fromtimestamp(int(records[-1]['ts'])/1000, tz=timezone.utc)})")

            # Check if we've reached the start date
            if int(records[-1]['ts']) < start_ts:
                break

            # Pagination: OKX uses 'after' cursor (timestamp of last record)
            current_after = records[-1]['ts']
            time.sleep(0.3)  # Rate limit

        except Exception as e:
            print(f"  ⚠️ Error fetching OI: {e}")
            break

    if not all_data:
        print("  ⚠️ No OI data fetched - will compute OI_CHANGE as 0")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['ts'].astype(int), unit='ms', utc=True)
    df['oi'] = df['oi'].astype(float)  # Open interest in contracts
    df['oiCcy'] = df['oiCcy'].astype(float)  # Open interest in BTC

    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    print(f"✅ Fetched {len(df)} hourly OI records")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Mean OI: {df['oiCcy'].mean():.2f} BTC")

    return df[['oiCcy']]

def compute_perp_basis(perp_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.Series:
    """Compute perpetual basis in basis points"""
    print("\n=== Computing PERP_BASIS ===")

    # Merge on timestamp
    merged = pd.merge(perp_df, spot_df, left_index=True, right_index=True, how='inner')

    # Compute basis in basis points
    merged['PERP_BASIS'] = ((merged['perp_price'] - merged['spot_price']) / merged['spot_price']) * 10000

    print(f"✅ Computed PERP_BASIS for {len(merged)} timestamps")
    print(f"   Mean basis: {merged['PERP_BASIS'].mean():.2f} bps")
    print(f"   Std basis: {merged['PERP_BASIS'].std():.2f} bps")

    return merged['PERP_BASIS']

def compute_oi_change(oi_df: pd.DataFrame) -> pd.Series:
    """Compute OI change as percentage change from previous period"""
    print("\n=== Computing OI_CHANGE ===")

    # Compute percentage change
    oi_change = oi_df['oiCcy'].pct_change().fillna(0.0)

    print(f"✅ Computed OI_CHANGE for {len(oi_change)} timestamps")
    print(f"   Mean change: {oi_change.mean():.4f} ({oi_change.mean()*100:.2f}%)")
    print(f"   Std change: {oi_change.std():.4f} ({oi_change.std()*100:.2f}%)")
    print(f"   Max increase: {oi_change.max():.4f} ({oi_change.max()*100:.2f}%)")
    print(f"   Max decrease: {oi_change.min():.4f} ({oi_change.min()*100:.2f}%)")

    return oi_change

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from existing data"""
    print("\n=== Computing Derived Features ===")

    df_new = df.copy()

    # 1. VOL_TERM - Volatility term structure (RV_30 / RV_7 - 1)
    if 'RV_30' in df.columns and 'RV_7' in df.columns:
        df_new['VOL_TERM'] = (df['RV_30'] / (df['RV_7'] + 1e-10)) - 1.0
        print(f"   ✅ VOL_TERM computed (mean: {df_new['VOL_TERM'].mean():.3f})")
    else:
        print("   ⚠️ RV features missing - setting VOL_TERM to 0")
        df_new['VOL_TERM'] = 0.0

    # 2. ALT_ROTATION - Altcoin rotation (TOTAL2_RET - TOTAL_RET)
    if 'TOTAL2_RET' in df.columns and 'TOTAL_RET' in df.columns:
        df_new['ALT_ROTATION'] = df['TOTAL2_RET'] - df['TOTAL_RET']
        print(f"   ✅ ALT_ROTATION computed (mean: {df_new['ALT_ROTATION'].mean():.3f})")
    else:
        print("   ⚠️ TOTAL features missing - setting ALT_ROTATION to 0")
        df_new['ALT_ROTATION'] = 0.0

    # 3. TOTAL3_RET - Set to 0 for now (need CoinGecko PRO or alternative)
    df_new['TOTAL3_RET'] = 0.0
    print("   ⚠️ TOTAL3_RET set to 0 (historical market cap not available)")

    # 4. SKEW_25D - Set to 0 for now (need Deribit options data)
    df_new['SKEW_25D'] = 0.0
    print("   ⚠️ SKEW_25D set to 0 (options data not available)")

    return df_new

def patch_feature_store(feature_store_path: str, new_features: pd.DataFrame):
    """Patch existing feature store with new features"""
    print(f"\n=== Patching Feature Store: {feature_store_path} ===")

    # Load existing feature store
    df_store = pd.read_parquet(feature_store_path)
    print(f"  Loaded feature store: {len(df_store)} rows, {len(df_store.columns)} columns")

    # Check if timestamp column exists
    if 'timestamp' in df_store.columns:
        print(f"  Date range: {df_store['timestamp'].min()} to {df_store['timestamp'].max()}")
    else:
        print(f"  Index range: {df_store.index[0]} to {df_store.index[-1]}")

    # Merge new features
    for col in new_features.columns:
        if col in df_store.columns:
            print(f"  ⚠️ Overwriting existing column: {col}")

        # Ensure alignment - new_features should have same length as df_store
        if len(new_features) != len(df_store):
            print(f"  ⚠️ Length mismatch: store={len(df_store)}, new={len(new_features)} for column {col}")
            # Reset index to align by position
            df_store[col] = new_features[col].values
        else:
            df_store[col] = new_features[col].values

        print(f"  ✅ Added/updated column: {col}")

    # Save patched feature store
    output_path = feature_store_path.replace('.parquet', '_with_macro.parquet')
    df_store.to_parquet(output_path)
    print(f"✅ Saved patched feature store: {output_path}")
    print(f"   Total columns: {len(df_store.columns)}")

    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backfill missing macro features')
    parser.add_argument('--feature-store', required=True, help='Path to feature store parquet file')
    parser.add_argument('--start-date', default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip fetching, only compute derived features')
    args = parser.parse_args()

    print("=" * 80)
    print("BACKFILLING MISSING MACRO FEATURES FOR GMM REGIME CLASSIFIER")
    print("=" * 80)

    # Load existing feature store to get timestamps
    df_store = pd.read_parquet(args.feature_store)
    print(f"\nLoaded feature store: {len(df_store)} rows")

    # Check if timestamp column exists
    if 'timestamp' not in df_store.columns:
        raise ValueError("Feature store must have a 'timestamp' column")

    # Convert timestamp to datetime if needed
    df_store['timestamp'] = pd.to_datetime(df_store['timestamp'], utc=True)

    print(f"Date range: {df_store['timestamp'].min()} to {df_store['timestamp'].max()}")
    print(f"Existing columns: {len(df_store.columns)}")

    # Check which features are missing
    missing_features = ['OI_CHANGE', 'TOTAL3_RET', 'ALT_ROTATION', 'VOL_TERM', 'SKEW_25D', 'PERP_BASIS']
    present = [f for f in missing_features if f in df_store.columns]
    missing = [f for f in missing_features if f not in df_store.columns]

    print(f"\nPresent features: {present}")
    print(f"Missing features: {missing}")

    # Initialize new features DataFrame aligned with feature store
    new_features = pd.DataFrame()

    if not args.skip_fetch:
        # Fetch perp and spot data
        try:
            perp_df = fetch_okx_historical_perp(args.start_date, args.end_date)
            spot_df = fetch_coinbase_historical_spot(args.start_date, args.end_date)

            # Compute PERP_BASIS
            basis_series = compute_perp_basis(perp_df, spot_df)

            # Create DataFrame with timestamp index for merging
            basis_df = pd.DataFrame({'PERP_BASIS': basis_series})
            basis_df.reset_index(inplace=True)
            basis_df.rename(columns={'index': 'timestamp'}, inplace=True)

            # Merge with feature store timestamps
            merged = pd.merge(
                df_store[['timestamp']],
                basis_df,
                on='timestamp',
                how='left'
            )

            # Fill missing values with forward fill then zero
            new_features['PERP_BASIS'] = merged['PERP_BASIS'].ffill().fillna(0.0)

        except Exception as e:
            print(f"\n⚠️ Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            print("  Setting PERP_BASIS to 0")
            new_features['PERP_BASIS'] = 0.0

        # Fetch OI data and compute OI_CHANGE
        try:
            oi_df = fetch_okx_historical_oi(args.start_date, args.end_date)

            if oi_df is not None:
                # Compute OI_CHANGE from historical OI
                oi_change_series = compute_oi_change(oi_df)

                # Create DataFrame with timestamp index for merging
                oi_df_merge = pd.DataFrame({'OI_CHANGE': oi_change_series})
                oi_df_merge.reset_index(inplace=True)
                oi_df_merge.rename(columns={'index': 'timestamp'}, inplace=True)

                # Merge with feature store timestamps
                merged = pd.merge(
                    df_store[['timestamp']],
                    oi_df_merge,
                    on='timestamp',
                    how='left'
                )

                # Fill missing values with forward fill then zero
                new_features['OI_CHANGE'] = merged['OI_CHANGE'].ffill().fillna(0.0)
            else:
                print("  Setting OI_CHANGE to 0")
                new_features['OI_CHANGE'] = 0.0

        except Exception as e:
            print(f"\n⚠️ Error fetching OI: {e}")
            import traceback
            traceback.print_exc()
            print("  Setting OI_CHANGE to 0")
            new_features['OI_CHANGE'] = 0.0

    else:
        print("\n⚠️ Skipping fetch - setting PERP_BASIS and OI_CHANGE to 0")
        new_features['PERP_BASIS'] = 0.0
        new_features['OI_CHANGE'] = 0.0

    # Compute derived features
    df_with_derived = compute_derived_features(df_store)

    # Add derived features to new_features (OI_CHANGE handled separately above)
    for col in ['VOL_TERM', 'ALT_ROTATION', 'TOTAL3_RET', 'SKEW_25D']:
        new_features[col] = df_with_derived[col]

    # Ensure OI_CHANGE exists (in case skip-fetch was used or fetch failed)
    if 'OI_CHANGE' not in new_features.columns:
        new_features['OI_CHANGE'] = 0.0

    # Patch feature store
    output_path = patch_feature_store(args.feature_store, new_features)

    print("\n" + "=" * 80)
    print("✅ BACKFILL COMPLETE!")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    print("\nNext steps:")
    print("1. Update config to use new feature store path")
    print("2. Test regime detection with complete features")
    print("3. Run backtest with bear archetypes enabled")

if __name__ == '__main__':
    main()
