#!/usr/bin/env python3
"""
Download VIX 2024 data from Yahoo Finance and convert to TradingView format
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

print("=" * 80)
print("Downloading VIX 2024 Data from Yahoo Finance")
print("=" * 80)

# Download VIX data (^VIX is the Yahoo Finance symbol)
print("\nDownloading ^VIX from 2023-10-01 to 2024-12-31...")
vix = yf.Ticker("^VIX")
df = vix.history(start='2023-10-01', end='2024-12-31', interval='1d')

print(f"Downloaded {len(df)} rows")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Convert to TradingView CSV format
# TradingView format: time (unix timestamp), open, high, low, close, volume
df_tv = pd.DataFrame({
    'time': [int(ts.timestamp()) for ts in df.index],
    'value': df['Close'].values,  # 'value' is used by macro loader
    'open': df['Open'].values,
    'high': df['High'].values,
    'low': df['Low'].values,
    'close': df['Close'].values,
    'volume': df['Volume'].values
})

# Save to data directory
output_path = 'data/VIX_1D.csv'
df_tv.to_csv(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")
print(f"   Rows: {len(df_tv)}")
print(f"   Columns: {df_tv.columns.tolist()}")
print(f"   Sample (first 3 rows):")
print(df_tv.head(3).to_string())

print("\n" + "=" * 80)
print("VIX Data Download Complete!")
print("=" * 80)

# Verify the data
print("\nVerifying data...")
print(f"First date: {pd.Timestamp(df_tv['time'].iloc[0], unit='s')}")
print(f"Last date: {pd.Timestamp(df_tv['time'].iloc[-1], unit='s')}")
print(f"VIX range: [{df_tv['close'].min():.2f}, {df_tv['close'].max():.2f}]")
print(f"Mean VIX: {df_tv['close'].mean():.2f}")

# Check Q3 2024 specifically
q3_mask = (df_tv['time'] >= 1719792000) & (df_tv['time'] <= 1727654400)
q3_data = df_tv[q3_mask]
print(f"\nQ3 2024 (Jul-Sep) data: {len(q3_data)} rows")
if len(q3_data) > 0:
    print(f"  VIX range: [{q3_data['close'].min():.2f}, {q3_data['close'].max():.2f}]")
    print(f"  Mean VIX: {q3_data['close'].mean():.2f}")
