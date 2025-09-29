#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
from bull_machine.modules.bojan.bojan import compute_bojan_score, detect_wick_magnet, BojanConfig

print("=== Debugging Bojan Implementation ===")

# Test 1: Upper wick magnet
print("\nTest 1: Upper wick magnet")
df1 = pd.DataFrame({
    'open': [100],
    'high': [110],    # High at 110
    'low': [99],      # Low at 99
    'close': [101],   # Close at 101
    'volume': [1000]
})

print("Data:", df1.iloc[0].to_dict())

config = BojanConfig()
print(f"Config: wick_magnet_threshold = {config.wick_magnet_threshold}")

# Test wick magnet detection directly
wick_result = detect_wick_magnet(df1, config)
print("Wick magnet result:", wick_result)

# Calculate manual metrics
total_range = 110 - 99  # 11
body_high = max(100, 101)  # 101
upper_wick = 110 - body_high  # 9
upper_wick_ratio = upper_wick / total_range  # 9/11 = 0.818
print(f"Manual calculation: upper_wick_ratio = {upper_wick_ratio:.3f} >= {config.wick_magnet_threshold} = {upper_wick_ratio >= config.wick_magnet_threshold}")

# Test full Bojan score
bojan_result = compute_bojan_score(df1)
print("Bojan score result:", bojan_result)

# Test 2: Trap reset
print("\n\nTest 2: Trap reset")
df2 = pd.DataFrame({
    'open': [100, 99],
    'high': [101, 104],   # Current bar sweeps higher
    'low': [98, 97],      # Then sweeps lower first
    'close': [99, 103],   # But closes bullish with large body
    'volume': [1000, 1500]
})

print("Data:")
print(df2)

bojan_result2 = compute_bojan_score(df2, {'trap_body_min': 1.0})
print("Bojan score result:", bojan_result2)

print("\nTrap reset signals:")
trap_signals = bojan_result2['signals']['trap_reset']
print(trap_signals)