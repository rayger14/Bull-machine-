#!/usr/bin/env python3
"""
Diagnose why Numba indicators produce different results.
Compare pandas vs Numba ADX calculation on real data.
"""

import pandas as pd
import numpy as np
from engine.context.loader import RealDataLoader

# Load real ETH data
loader = RealDataLoader()
df_1h, df_4h, df_1d = loader.load_ohlcv('ETH', '2025-06-15', '2025-06-22')

print(f"Loaded {len(df_1h)} 1H bars, {len(df_4h)} 4H bars")

# Test 1: Compare ADX calculations
print("\n" + "="*70)
print("TEST 1: ADX Calculation Comparison")
print("="*70)

# Pandas implementation (original)
def calc_adx_pandas(df: pd.DataFrame, period: int = 14) -> float:
    """Original pandas implementation from fast_signals.py"""
    if len(df) < period * 2:
        return 0.0

    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()

    return adx.iloc[-1] if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else 0.0

# Numba implementation
try:
    from engine.indicators.fast_indicators import get_adx_scalar
    NUMBA_AVAILABLE = True
except ImportError:
    print("❌ Numba indicators not available")
    NUMBA_AVAILABLE = False
    exit(1)

# Compare on multiple sample windows
for window_size in [50, 100, len(df_4h)]:
    df_sample = df_4h.tail(window_size).copy()

    adx_pandas = calc_adx_pandas(df_sample, period=14)
    adx_numba = get_adx_scalar(df_sample, period=14)
    diff = abs(adx_pandas - adx_numba)

    print(f"\nWindow: {window_size} bars")
    print(f"  Pandas ADX: {adx_pandas:.6f}")
    print(f"  Numba ADX:  {adx_numba:.6f}")
    print(f"  Difference: {diff:.6f}")

    if diff > 0.01:
        print(f"  ⚠️  SIGNIFICANT DIFFERENCE (>{0.01})")
    else:
        print(f"  ✅ Match within tolerance")

# Test 2: Check if ADX values are systematically different
print("\n" + "="*70)
print("TEST 2: Statistical Comparison Over Full Dataset")
print("="*70)

pandas_values = []
numba_values = []

for i in range(50, len(df_4h), 10):  # Sample every 10th bar
    df_slice = df_4h.iloc[:i]
    pandas_values.append(calc_adx_pandas(df_slice, 14))
    numba_values.append(get_adx_scalar(df_slice, 14))

pandas_arr = np.array(pandas_values)
numba_arr = np.array(numba_values)

print(f"\nSamples: {len(pandas_values)}")
print(f"Pandas - Mean: {pandas_arr.mean():.2f}, Std: {pandas_arr.std():.2f}, Range: [{pandas_arr.min():.2f}, {pandas_arr.max():.2f}]")
print(f"Numba  - Mean: {numba_arr.mean():.2f}, Std: {numba_arr.std():.2f}, Range: [{numba_arr.min():.2f}, {numba_arr.max():.2f}]")
print(f"\nMax absolute diff: {np.abs(pandas_arr - numba_arr).max():.6f}")
print(f"Mean absolute diff: {np.abs(pandas_arr - numba_arr).mean():.6f}")

correlation = np.corrcoef(pandas_arr, numba_arr)[0, 1]
print(f"Correlation: {correlation:.6f}")

if correlation < 0.99:
    print("\n⚠️  PROBLEM: Low correlation suggests systematic difference in calculation")
elif np.abs(pandas_arr - numba_arr).max() > 1.0:
    print("\n⚠️  PROBLEM: Large differences detected")
else:
    print("\n✅ Calculations match within acceptable tolerance")

# Test 3: Find specific divergence points
print("\n" + "="*70)
print("TEST 3: Divergence Analysis")
print("="*70)

diffs = np.abs(pandas_arr - numba_arr)
if diffs.max() > 0.1:
    worst_idx = np.argmax(diffs)
    print(f"\nWorst divergence at sample {worst_idx}:")
    print(f"  Pandas: {pandas_arr[worst_idx]:.6f}")
    print(f"  Numba:  {numba_arr[worst_idx]:.6f}")
    print(f"  Diff:   {diffs[worst_idx]:.6f}")

    # Check if this affects signal generation (ADX > 20 threshold)
    pandas_above = (pandas_arr > 20).sum()
    numba_above = (numba_arr > 20).sum()
    print(f"\nSignal threshold impact (ADX > 20):")
    print(f"  Pandas triggers: {pandas_above} / {len(pandas_arr)} ({100*pandas_above/len(pandas_arr):.1f}%)")
    print(f"  Numba triggers:  {numba_above} / {len(numba_arr)} ({100*numba_above/len(numba_arr):.1f}%)")

    if abs(pandas_above - numba_above) > len(pandas_arr) * 0.05:
        print(f"  ⚠️  CRITICAL: Significant difference in signal generation!")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
