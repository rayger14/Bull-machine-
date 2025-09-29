#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
from bull_machine.strategy.po3_detection import detect_po3, reverses, has_bojan_high

# Test the specific failing case: high_sweep_low_break
df = pd.DataFrame({
    'high': [105, 107, 109, 112, 98],  # Sweep above IRH at bar 4
    'low': [100, 102, 104, 107, 92],   # Break below IRL at bar 5
    'close': [103, 105, 107, 98, 95],
    'volume': [1000, 1200, 1400, 2500, 2000]  # Volume spike during sweep
})

irh = 109  # Initial Range High
irl = 100  # Initial Range Low

print("=== High Sweep Low Break Test ===")
print("DataFrame:")
print(df)
print(f"IRH: {irh}, IRL: {irl}")

# Debug step by step
recent_bars = df.tail(min(5, len(df)))
print("\nRecent bars analysis:")
for i, (idx, bar) in enumerate(recent_bars.iterrows()):
    print(f"Bar {i} (idx {idx}): high={bar['high']}, low={bar['low']}, close={bar['close']}")
    print(f"  Sweeps low IRH ({irh}): {bar['high'] > irh}")
    print(f"  Sweeps below IRL ({irl}): {bar['low'] < irl}")

sweep_low_detected = any(bar['low'] < irl for _, bar in recent_bars.iterrows())
sweep_high_detected = any(bar['high'] > irh for _, bar in recent_bars.iterrows())
print(f"\nOverall sweep detection:")
print(f"Sweep low detected: {sweep_low_detected}")
print(f"Sweep high detected: {sweep_high_detected}")

current_close = df['close'].iloc[-1]
print(f"Current close: {current_close}")
print(f"Close < IRL ({irl}): {current_close < irl}")
print(f"Close > IRH ({irh}): {current_close > irh}")

# Run detection
po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.4)
print(f"\nPO3 result: {po3}")
print(f"Expected: high_sweep_low_break")

# The issue: both sweeps detected, but the high sweep came first (bar 3: 112 > 109)
# The low sweep came later (bar 4: 92 < 100)
# We should prioritize the most recent decisive pattern
print(f"\nDetailed analysis:")
print(f"Bar 3: high={df.iloc[3]['high']}, sweeps high: {df.iloc[3]['high'] > irh}")
print(f"Bar 4: low={df.iloc[4]['low']}, sweeps low: {df.iloc[4]['low'] < irl}")
print(f"Most recent action is low sweep + low break")