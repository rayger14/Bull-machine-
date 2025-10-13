#!/usr/bin/env python3
"""
Trace exact pandas ADX calculation step by step.
"""

import pandas as pd
import numpy as np

# Small test dataset
np.random.seed(42)
n = 50
prices = 2000 + np.cumsum(np.random.randn(n) * 10)

df = pd.DataFrame({
    'high': prices + np.abs(np.random.randn(n) * 5),
    'low': prices - np.abs(np.random.randn(n) * 5),
    'close': prices
})

period = 14

print("Pandas ADX Calculation Trace")
print("="*70)

high = df['high']
low = df['low']
close = df['close']

# Step 1: Directional Movement
plus_dm = high.diff()
minus_dm = -low.diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm < 0] = 0

print("\nStep 1: Directional Movement (first 20 values)")
print(f"  plus_dm:  {plus_dm.head(20).values}")
print(f"  minus_dm: {minus_dm.head(20).values}")

# Step 2: True Range
tr = pd.concat([
    high - low,
    (high - close.shift()).abs(),
    (low - close.shift()).abs()
], axis=1).max(axis=1)

print("\nStep 2: True Range (first 20 values)")
print(f"  tr: {tr.head(20).values}")

# Step 3: ATR
atr = tr.rolling(period).mean()

print(f"\nStep 3: ATR (rolling {period})")
print(f"  First non-NaN at index {period-1}: {atr.iloc[period-1]}")
print(f"  ATR[{period-1}]: {atr.iloc[period-1]}")
print(f"  ATR[{period}]: {atr.iloc[period]}")
print(f"  ATR[{period+1}]: {atr.iloc[period+1]}")

# Step 4: DM rolling means
plus_dm_sma = plus_dm.rolling(period).mean()
minus_dm_sma = minus_dm.rolling(period).mean()

print(f"\nStep 4: DM SMAs")
print(f"  plus_dm_sma[{period-1}]: {plus_dm_sma.iloc[period-1]}")
print(f"  minus_dm_sma[{period-1}]: {minus_dm_sma.iloc[period-1]}")

# Step 5: Directional Indicators
plus_di = 100 * (plus_dm_sma / atr)
minus_di = 100 * (minus_dm_sma / atr)

print(f"\nStep 5: Directional Indicators")
print(f"  +DI[{period-1}]: {plus_di.iloc[period-1]}")
print(f"  -DI[{period-1}]: {minus_di.iloc[period-1]}")
print(f"  +DI[{period}]: {plus_di.iloc[period]}")
print(f"  -DI[{period}]: {minus_di.iloc[period]}")

# Step 6: DX
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)

print(f"\nStep 6: DX")
print(f"  DX[{period-1}]: {dx.iloc[period-1]}")
print(f"  DX[{period}]: {dx.iloc[period]}")
print(f"  DX[{period*2-2}]: {dx.iloc[period*2-2]}")

# Step 7: ADX
adx = dx.rolling(period).mean()

print(f"\nStep 7: ADX (rolling {period} of DX)")
print(f"  First non-NaN ADX at index {period*2-2}: {adx.iloc[period*2-2]}")
print(f"  ADX[{period*2-2}]: {adx.iloc[period*2-2]}")
print(f"  ADX[{period*2-1}]: {adx.iloc[period*2-1]}")
print(f"  ADX[{period*2}]: {adx.iloc[period*2]}")
print(f"  ADX[{n-1}]: {adx.iloc[-1]}")

print("\n" + "="*70)
print("Key Insights:")
print(f"  - TR starts with NaN at index 0 (due to shift)")
print(f"  - ATR first valid at index {period-1} (rolling needs {period} values)")
print(f"  - DX first valid at index {period-1}")
print(f"  - ADX first valid at index {period*2-2} (needs {period} DX values)")
print("="*70)
