#!/usr/bin/env python3
"""Debug pandas rolling window behavior."""

import pandas as pd
import numpy as np

# Small test case
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
s = pd.Series(data)

print("Original data:", data)
print("\nPandas rolling(3).mean():")
rolling = s.rolling(3).mean()
print(rolling.values)

print("\nManual SMA calculation:")
manual = np.zeros(len(data))
for i in range(3, len(data) + 1):
    manual[i-1] = np.mean(data[i-3:i])
    print(f"  i={i-1}: mean({data[i-3:i]}) = {manual[i-1]}")

print("\nComparison:")
for i, (p, m) in enumerate(zip(rolling.values, manual)):
    match = "✓" if (pd.isna(p) and m == 0) or abs(p - m) < 1e-10 else "✗"
    print(f"  [{i}] Pandas: {p}, Manual: {m} {match}")

# Now test with real difference calculation
print("\n" + "="*70)
print("Testing diff() behavior")
print("="*70)

prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0])
s_prices = pd.Series(prices)

print("\nPrices:", prices)
print("Pandas diff():", s_prices.diff().values)

manual_diff = np.zeros(len(prices))
for i in range(1, len(prices)):
    manual_diff[i] = prices[i] - prices[i-1]
print("Manual diff: ", manual_diff)
