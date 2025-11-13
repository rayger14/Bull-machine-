#!/usr/bin/env python3
"""
Debug trap archetype to see WHY no detections occur.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Load REAL data
df = pd.read_parquet('data/features/v18/BTC_1H.parquet')
df = df.tail(100)  # Just 100 bars for debug

print("="*70)
print("TRAP DEBUG - WHY NO DETECTIONS?")
print("="*70)

# Check data completeness
required_fields = [
    'tf4h_fusion_score', 'adx_14', 'tf1h_bos_flag',
    'close', 'open', 'high', 'low'
]

print("\nData completeness check:")
for field in required_fields:
    if field in df.columns:
        non_null = df[field].notna().sum()
        print(f"  ✓ {field}: {non_null}/{len(df)} non-null")
        if non_null > 0:
            print(f"      Range: [{df[field].min():.3f}, {df[field].max():.3f}]")
    else:
        print(f"  ❌ {field}: MISSING!")

# Check if ANY bar would pass the checks
print("\nChecking if ANY bar could pass trap conditions...")
print("(Using relaxed params: quality=0.40, liq=0.40, adx=15, fusion=0.25)")

if 'tf4h_fusion_score' in df.columns:
    pass_quality = (df['tf4h_fusion_score'] > 0.40).sum()
    print(f"  Pass quality check (tf4h_fusion > 0.40): {pass_quality}/{len(df)}")

if 'adx_14' in df.columns:
    pass_adx = (df['adx_14'] > 15.0).sum()
    print(f"  Pass ADX check (adx_14 > 15): {pass_adx}/{len(df)}")

if 'tf1h_bos_flag' in df.columns:
    pass_bos = (df['tf1h_bos_flag'] != 0).sum()
    print(f"  Pass BOS check (tf1h_bos_flag != 0): {pass_bos}/{len(df)}")

# Try to manually check one bar
print("\nManual check of ONE bar:")
row = df.iloc[50]
print(f"  tf4h_fusion_score: {row.get('tf4h_fusion_score', 'MISSING')}")
print(f"  adx_14: {row.get('adx_14', 'MISSING')}")
print(f"  tf1h_bos_flag: {row.get('tf1h_bos_flag', 'MISSING')}")
print(f"  close: {row.get('close', 'MISSING')}")
print(f"  open: {row.get('open', 'MISSING')}")
print(f"  high: {row.get('high', 'MISSING')}")
print(f"  low: {row.get('low', 'MISSING')}")

# Calculate wick
if all(k in row.index for k in ['close', 'open', 'high', 'low']):
    close = row['close']
    open_price = row['open']
    high = row['high']
    low = row['low']

    body = abs(close - open_price)
    upper_wick = high - max(close, open_price)
    lower_wick = min(close, open_price) - low

    print(f"\n  Wick analysis:")
    print(f"    Body: {body:.2f}")
    print(f"    Upper wick: {upper_wick:.2f}")
    print(f"    Lower wick: {lower_wick:.2f}")
    print(f"    Lower wick > 1.0 * body? {lower_wick > 1.0 * body}")
    print(f"    Upper wick > 1.0 * body? {upper_wick > 1.0 * body}")

print("\n" + "="*70)
