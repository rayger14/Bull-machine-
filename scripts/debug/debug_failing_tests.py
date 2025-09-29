#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
from bull_machine.strategy.po3_detection import detect_po3, reverses, has_bojan_high

print("=== Debugging Failing PO3 Tests ===\n")

# Test 1: Basic PO3 detection (FAILING)
print("TEST 1: Basic PO3 detection")
df1 = pd.DataFrame({
    'high': [100, 105, 110, 108, 112],
    'low': [95, 100, 105, 102, 107],
    'close': [98, 104, 108, 106, 111],
    'volume': [1000, 1200, 2000, 1800, 2500]
})

irh = 110
irl = 95

print("DataFrame:")
print(df1)
print(f"IRH: {irh}, IRL: {irl}")

# Debug step by step
vol_mean = df1['volume'].rolling(min(10, len(df1))).mean().iloc[-1]
print(f"Volume mean: {vol_mean}")

recent_vols = df1['volume'].tail(min(3, len(df1)))
print(f"Recent volumes: {list(recent_vols)}")
vol_spike = any(vol > 1.5 * vol_mean for vol in recent_vols)
print(f"Volume spike detected: {vol_spike}")

recent_bars = df1.tail(min(5, len(df1)))
sweep_low_detected = any(bar['low'] < irl for _, bar in recent_bars.iterrows())
sweep_high_detected = any(bar['high'] > irh for _, bar in recent_bars.iterrows())
print(f"Sweep low detected: {sweep_low_detected}")
print(f"Sweep high detected: {sweep_high_detected}")

current_close = df1['close'].iloc[-1]
print(f"Current close: {current_close}")
print(f"Close > IRH ({irh}): {current_close > irh}")
print(f"Close < IRL ({irl}): {current_close < irl}")

reversal = reverses(df1, 3)
print(f"Reversal detected: {reversal}")

po3_result = detect_po3(df1, irh, irl, vol_spike_threshold=1.4)
print(f"PO3 Result: {po3_result}\n")

# Test 2: High sweep low break (FAILING)
print("TEST 2: High sweep low break")
df2 = pd.DataFrame({
    'high': [105, 107, 109, 112, 98],  # Sweep above IRH at bar 4
    'low': [100, 102, 104, 107, 92],   # Break below IRL at bar 5
    'close': [103, 105, 107, 98, 95],
    'volume': [1000, 1200, 1400, 2500, 2000]
})

irh2 = 109
irl2 = 100

print("DataFrame:")
print(df2)
print(f"IRH: {irh2}, IRL: {irl2}")

vol_mean2 = df2['volume'].rolling(min(10, len(df2))).mean().iloc[-1]
print(f"Volume mean: {vol_mean2}")

recent_vols2 = df2['volume'].tail(min(3, len(df2)))
print(f"Recent volumes: {list(recent_vols2)}")
vol_spike2 = any(vol > 1.5 * vol_mean2 for vol in recent_vols2)
print(f"Volume spike detected: {vol_spike2}")

recent_bars2 = df2.tail(min(5, len(df2)))
sweep_low_detected2 = any(bar['low'] < irl2 for _, bar in recent_bars2.iterrows())
sweep_high_detected2 = any(bar['high'] > irh2 for _, bar in recent_bars2.iterrows())
print(f"Sweep low detected: {sweep_low_detected2}")
print(f"Sweep high detected: {sweep_high_detected2}")

current_close2 = df2['close'].iloc[-1]
print(f"Current close: {current_close2}")
print(f"Close > IRH ({irh2}): {current_close2 > irh2}")
print(f"Close < IRL ({irl2}): {current_close2 < irl2}")

reversal2 = reverses(df2, 3)
print(f"Reversal detected: {reversal2}")

po3_result2 = detect_po3(df2, irh2, irl2, vol_spike_threshold=1.5)
print(f"PO3 Result: {po3_result2}\n")

# Test 3: Bojan high boost (FAILING)
print("TEST 3: Bojan high boost")
df3 = pd.DataFrame({
    'open': [98, 100, 102, 104, 106],
    'high': [101, 115, 107, 109, 111],  # Bojan high at bar 2
    'low': [95, 99, 101, 94, 105],     # Sweep at bar 4
    'close': [99, 103, 105, 108, 108],  # Break at bar 4
    'volume': [1000, 1200, 1400, 2500, 2000]
})

irh3 = 107
irl3 = 95

print("DataFrame:")
print(df3)
print(f"IRH: {irh3}, IRL: {irl3}")

bojan_detected = has_bojan_high(df3)
print(f"Bojan high detected: {bojan_detected}")

vol_mean3 = df3['volume'].rolling(min(10, len(df3))).mean().iloc[-1]
print(f"Volume mean: {vol_mean3}")

recent_vols3 = df3['volume'].tail(min(3, len(df3)))
print(f"Recent volumes: {list(recent_vols3)}")
vol_spike3 = any(vol > 1.5 * vol_mean3 for vol in recent_vols3)
print(f"Volume spike detected: {vol_spike3}")

recent_bars3 = df3.tail(min(5, len(df3)))
sweep_low_detected3 = any(bar['low'] < irl3 for _, bar in recent_bars3.iterrows())
sweep_high_detected3 = any(bar['high'] > irh3 for _, bar in recent_bars3.iterrows())
print(f"Sweep low detected: {sweep_low_detected3}")
print(f"Sweep high detected: {sweep_high_detected3}")

current_close3 = df3['close'].iloc[-1]
print(f"Current close: {current_close3}")
print(f"Close > IRH ({irh3}): {current_close3 > irh3}")
print(f"Close < IRL ({irl3}): {current_close3 < irl3}")

reversal3 = reverses(df3, 3)
print(f"Reversal detected: {reversal3}")

po3_result3 = detect_po3(df3, irh3, irl3, vol_spike_threshold=1.5)
print(f"PO3 Result: {po3_result3}")