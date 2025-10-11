#!/usr/bin/env python3
"""
Test ADX parity between pandas and Numba implementations.
"""

import pandas as pd
import numpy as np

# Create synthetic test data
np.random.seed(42)
n = 200
prices = 2000 + np.cumsum(np.random.randn(n) * 10)

df = pd.DataFrame({
    'high': prices + np.abs(np.random.randn(n) * 5),
    'low': prices - np.abs(np.random.randn(n) * 5),
    'close': prices,
    'volume': np.random.randint(1000, 10000, n)
})

print("Testing ADX calculation parity...")
print(f"Dataset: {len(df)} bars\n")

# Pandas implementation (from fast_signals.py)
def calc_adx_pandas(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period * 2:
        return 0.0

    high = df['high']
    low = df['low']
    close = df['close']

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    # Average True Range
    atr = tr.rolling(period).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # Directional Index
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)

    # ADX (smoothed DX)
    adx = dx.rolling(period).mean()

    return adx.iloc[-1] if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else 0.0

# Numba implementation
from engine.indicators.fast_indicators import get_adx_scalar

# Test 1: Single value comparison
adx_pandas = calc_adx_pandas(df, 14)
adx_numba = get_adx_scalar(df, 14)

print(f"Final ADX value:")
print(f"  Pandas: {adx_pandas:.6f}")
print(f"  Numba:  {adx_numba:.6f}")
print(f"  Diff:   {abs(adx_pandas - adx_numba):.6f}")

if abs(adx_pandas - adx_numba) < 0.01:
    print("  ✅ Match!")
else:
    print("  ❌ MISMATCH")

# Test 2: Full series comparison
print("\n" + "="*70)
print("Full Series Comparison (every 10 bars)")
print("="*70)

pandas_series = []
numba_series = []

for i in range(50, len(df), 10):
    df_slice = df.iloc[:i].copy()
    pandas_series.append(calc_adx_pandas(df_slice, 14))
    numba_series.append(get_adx_scalar(df_slice, 14))

pandas_arr = np.array(pandas_series)
numba_arr = np.array(numba_series)

print(f"\nSamples: {len(pandas_series)}")
print(f"Max diff: {np.abs(pandas_arr - numba_arr).max():.6f}")
print(f"Mean diff: {np.abs(pandas_arr - numba_arr).mean():.6f}")
print(f"Correlation: {np.corrcoef(pandas_arr, numba_arr)[0,1]:.6f}")

# Find worst case
worst_idx = np.argmax(np.abs(pandas_arr - numba_arr))
print(f"\nWorst mismatch at index {worst_idx}:")
print(f"  Pandas: {pandas_arr[worst_idx]:.6f}")
print(f"  Numba:  {numba_arr[worst_idx]:.6f}")
print(f"  Diff:   {abs(pandas_arr[worst_idx] - numba_arr[worst_idx]):.6f}")

# Test 3: Check threshold impact (ADX > 20)
pandas_above = (pandas_arr > 20).sum()
numba_above = (numba_arr > 20).sum()

print(f"\nThreshold impact (ADX > 20):")
print(f"  Pandas: {pandas_above}/{len(pandas_arr)} ({100*pandas_above/len(pandas_arr):.1f}%)")
print(f"  Numba:  {numba_above}/{len(numba_arr)} ({100*numba_above/len(numba_arr):.1f}%)")

if abs(pandas_above - numba_above) > len(pandas_arr) * 0.05:
    print("  ❌ CRITICAL: >5% difference in signal generation!")
else:
    print("  ✅ Acceptable threshold agreement")

# Test 4: Detailed step-by-step comparison on a small window
print("\n" + "="*70)
print("Detailed Calculation Breakdown (50-bar window)")
print("="*70)

df_small = df.iloc[100:150].copy().reset_index(drop=True)

# Pandas step-by-step
high = df_small['high']
low = df_small['low']
close = df_small['close']

plus_dm_pd = high.diff()
minus_dm_pd = -low.diff()
plus_dm_pd[plus_dm_pd < 0] = 0
minus_dm_pd[minus_dm_pd < 0] = 0

tr_pd = pd.concat([
    high - low,
    (high - close.shift()).abs(),
    (low - close.shift()).abs()
], axis=1).max(axis=1)

atr_pd = tr_pd.rolling(14).mean()
plus_di_pd = 100 * (plus_dm_pd.rolling(14).mean() / atr_pd)
minus_di_pd = 100 * (minus_dm_pd.rolling(14).mean() / atr_pd)
dx_pd = 100 * (plus_di_pd - minus_di_pd).abs() / (plus_di_pd + minus_di_pd + 1e-10)
adx_pd = dx_pd.rolling(14).mean()

# Numba
from engine.indicators.fast_indicators import calc_adx_numba
adx_nb, plus_di_nb, minus_di_nb = calc_adx_numba(
    df_small['high'].values,
    df_small['low'].values,
    df_small['close'].values,
    14
)

print(f"\nFinal values (bar 49):")
print(f"  +DI  - Pandas: {plus_di_pd.iloc[-1]:.4f}, Numba: {plus_di_nb[-1]:.4f}, Diff: {abs(plus_di_pd.iloc[-1] - plus_di_nb[-1]):.6f}")
print(f"  -DI  - Pandas: {minus_di_pd.iloc[-1]:.4f}, Numba: {minus_di_nb[-1]:.4f}, Diff: {abs(minus_di_pd.iloc[-1] - minus_di_nb[-1]):.6f}")
print(f"  ADX  - Pandas: {adx_pd.iloc[-1]:.4f}, Numba: {adx_nb[-1]:.4f}, Diff: {abs(adx_pd.iloc[-1] - adx_nb[-1]):.6f}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
