#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
from bull_machine.strategy.po3_detection import detect_po3, reverses, has_bojan_high

# Test 1: Basic PO3 detection volume issue
df = pd.DataFrame({
    'high': [100, 105, 110, 108, 112],
    'low': [95, 100, 105, 102, 107],
    'close': [98, 104, 108, 106, 111],
    'volume': [1000, 1200, 2000, 1800, 2500]
})

print("Volume debugging:")
print(f"All volumes: {list(df['volume'])}")

# Test rolling mean calculation
vol_rolling = df['volume'].rolling(min(10, len(df)))
print(f"Rolling mean window: {min(10, len(df))}")
vol_mean = vol_rolling.mean().iloc[-1]
print(f"Rolling volume mean: {vol_mean}")

# Test simple mean
vol_simple = df['volume'].mean()
print(f"Simple volume mean: {vol_simple}")

# Test recent volumes
recent_vols = df['volume'].tail(min(3, len(df)))
print(f"Recent volumes (last 3): {list(recent_vols)}")

# Test spike detection with both thresholds
vol_spike_threshold = 1.4
print(f"Volume spike threshold: {vol_spike_threshold}")
print(f"Threshold value: {vol_spike_threshold * vol_mean}")

for i, vol in enumerate(recent_vols):
    spike = vol > vol_spike_threshold * vol_mean
    print(f"Volume {i}: {vol} > {vol_spike_threshold * vol_mean} = {spike}")

vol_spike = any(vol > vol_spike_threshold * vol_mean for vol in recent_vols)
print(f"Any volume spike: {vol_spike}")

# Test with stricter threshold like in tests
vol_spike_15 = any(vol > 1.5 * vol_mean for vol in recent_vols)
print(f"Volume spike (1.5x): {vol_spike_15}")

# Run PO3 detection
irh = 110
irl = 95
po3_result = detect_po3(df, irh, irl, vol_spike_threshold=1.4)
print(f"PO3 Result (1.4x): {po3_result}")

po3_result_15 = detect_po3(df, irh, irl, vol_spike_threshold=1.5)
print(f"PO3 Result (1.5x): {po3_result_15}")