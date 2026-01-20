#!/usr/bin/env python3
"""
Process TradingView Funding Rate Export

Converts TradingView CSV export to the format needed for feature engineering.

Usage:
    1. Export BINANCE:BTCUSDTPERP_FUNDING.RATE from TradingView (2022-2024, 1D timeframe)
    2. Save as data/FUNDING_1D_HISTORICAL.csv
    3. Run: python3 bin/process_funding_export.py
"""

import pandas as pd
from pathlib import Path


def main():
    input_file = Path('data/FUNDING_1D_HISTORICAL.csv')
    output_file = Path('data/FUNDING_1D.csv')

    print("\n" + "="*80)
    print("PROCESS TRADINGVIEW FUNDING EXPORT")
    print("="*80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    if not input_file.exists():
        print(f"\n❌ ERROR: {input_file} not found!")
        print("\n📋 Instructions:")
        print("1. Open TradingView: BINANCE:BTCUSDTPERP_FUNDING.RATE")
        print("2. Set timeframe: 1D (Daily)")
        print("3. Set date range: 2022-01-01 to 2024-12-31")
        print("4. Right-click chart → 'Export chart data...'")
        print("5. Save as: data/FUNDING_1D_HISTORICAL.csv")
        print("6. Run this script again")
        return 1

    # Load TradingView export
    print(f"\n📊 Loading TradingView export...")
    df = pd.read_csv(input_file)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # TradingView exports typically have: time, open, high, low, close
    # For funding rate, the 'close' column is the actual rate

    # Parse timestamp
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Rename close to funding (this is the funding rate value)
    if 'close' in df.columns:
        df_funding = df[['close']].rename(columns={'close': 'funding'})
    else:
        print(f"   Available columns: {list(df.columns)}")
        print(f"   ⚠️  Warning: 'close' column not found, using first numeric column")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_funding = df[[numeric_cols[0]]].rename(columns={numeric_cols[0]: 'funding'})

    print(f"\n📈 Funding Rate Stats:")
    print(f"   Date range: {df_funding.index.min()} to {df_funding.index.max()}")
    print(f"   Total days: {len(df_funding)}")
    print(f"   NaN count: {df_funding['funding'].isna().sum()}")
    print(f"   Min: {df_funding['funding'].min():.6f}")
    print(f"   Max: {df_funding['funding'].max():.6f}")
    print(f"   Mean: {df_funding['funding'].mean():.6f}")

    # Save processed data
    print(f"\n💾 Saving processed funding data...")
    df_funding.to_csv(output_file)

    # Verify
    df_check = pd.read_csv(output_file, index_col=0, parse_dates=True)
    print(f"\n✅ Verification:")
    print(f"   Shape: {df_check.shape}")
    print(f"   Index: {df_check.index.min()} to {df_check.index.max()}")

    print("\n" + "="*80)
    print("✅ FUNDING DATA PROCESSED SUCCESSFULLY")
    print("="*80)
    print("\n📝 Next steps:")
    print("1. Rebuild 2022-2023 feature store with new funding data:")
    print("   python3 bin/build_mtf_feature_store.py --asset BTC --start 2022-01-01 --end 2023-12-31")
    print("\n2. Re-run Router v10 backtest:")
    print("   python3 bin/backtest_router_v10.py --asset BTC --start 2022-01-01 --end 2023-12-31")

    return 0


if __name__ == '__main__':
    exit(main())
