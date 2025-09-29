#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
from bull_machine.strategy.po3_detection import detect_po3, reverses, has_bojan_high

# Create a more realistic PO3 test case
print("=== PO3 Working Test ===\n")

# Test 1: Clear high sweep with reversal
df1 = pd.DataFrame({
    'high': [100, 102, 104, 118, 108],  # Clear sweep above 110 at bar 4
    'low': [95, 97, 99, 107, 90],       # Then break below 95 at bar 5
    'close': [98, 100, 102, 112, 92],   # Close shows the pattern
    'volume': [1000, 1100, 1200, 3000, 2800]  # Clear volume spike
})

irh = 110
irl = 95

print("Test 1: High sweep + low break")
print(df1)
print(f"IRH: {irh}, IRL: {irl}")

vol_mean = df1['volume'].mean()
print(f"Volume mean: {vol_mean}")
print(f"Volume spike threshold: {1.5 * vol_mean}")
print(f"Volume at sweep: {df1['volume'].iloc[-2]} > {1.5 * vol_mean} = {df1['volume'].iloc[-2] > 1.5 * vol_mean}")

# Debug Test 1
recent_bars = df1.tail(5)
sweep_high_detected = any(bar['high'] > irh for _, bar in recent_bars.iterrows())
current_close = df1['close'].iloc[-1]
print(f"Sweep high detected: {sweep_high_detected}")
print(f"Current close: {current_close} < IRL {irl}: {current_close < irl}")
print(f"Reversal: {reverses(df1, 3)}")

po3_result = detect_po3(df1, irh, irl, vol_spike_threshold=1.3)  # Lower threshold
print(f"PO3 Result: {po3_result}\n")

# Test 2: Low sweep with reversal
df2 = pd.DataFrame({
    'high': [105, 107, 109, 108, 115],  # Break above 110 at bar 5
    'low': [100, 102, 104, 88, 103],    # Sweep below 95 at bar 4
    'close': [103, 105, 107, 92, 112],  # Clear reversal pattern
    'volume': [1000, 1100, 1200, 2800, 2500]
})

print("Test 2: Low sweep + high break")
print(df2)
po3_result2 = detect_po3(df2, irh, irl, vol_spike_threshold=1.3)
print(f"PO3 Result: {po3_result2}\n")

# Test 3: With Bojan high
df3 = pd.DataFrame({
    'open': [100, 102, 104, 106, 108],
    'high': [102, 120, 110, 118, 115],  # Bojan high at bar 2
    'low': [98, 101, 103, 88, 103],     # Sweep at bar 4
    'close': [101, 105, 107, 92, 112],  # Reversal
    'volume': [1000, 1100, 1200, 2800, 2500]
})

print("Test 3: With Bojan high pattern")
print(df3)
bojan_detected = has_bojan_high(df3)
print(f"Bojan high detected: {bojan_detected}")
po3_result3 = detect_po3(df3, irh, irl, vol_spike_threshold=1.3)
print(f"PO3 Result: {po3_result3}")

# Test 4: Working conditions check
print("\n=== Testing Individual Components ===")

# Test reversal function
test_reversal_df = pd.DataFrame({
    'high': [105],
    'low': [95],
    'close': [103]  # Close at 80% of range
})
print(f"Reversal test: {reverses(test_reversal_df)}")

# Test volume spike detection
test_vols = [1000, 1100, 1200, 2800, 2500]
vol_mean_test = sum(test_vols) / len(test_vols)
spikes = [vol > 1.3 * vol_mean_test for vol in test_vols[-3:]]
print(f"Volume spikes in last 3: {spikes}, Any spike: {any(spikes)}")