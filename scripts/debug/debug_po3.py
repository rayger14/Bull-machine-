#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
from bull_machine.strategy.po3_detection import detect_po3, reverses, has_bojan_high

# Test data from failing test
df = pd.DataFrame({
    'high': [100, 105, 110, 108, 112],
    'low': [95, 100, 105, 102, 107],
    'close': [98, 104, 108, 106, 111],
    'volume': [1000, 1200, 2000, 1800, 2500]
})

irh = 110
irl = 95

print("Testing PO3 detection...")
print("DataFrame:")
print(df)
print(f"\nIRH: {irh}, IRL: {irl}")

# Debug volume calculation
vol_mean_old = df['volume'].rolling(min(20, len(df))).mean().iloc[-1]
vol_mean_new = df['volume'].rolling(min(10, len(df))).mean().iloc[-1]
print(f"Volume mean (old): {vol_mean_old}")
print(f"Volume mean (new): {vol_mean_new}")

# Use simple mean for small datasets
vol_mean = df['volume'].mean()
print(f"Volume mean (simple): {vol_mean}")

current_vol = df['volume'].iloc[-1]
prev_vol = df['volume'].iloc[-2]
print(f"Current volume: {current_vol}, Previous volume: {prev_vol}")

vol_spike = (current_vol > 1.5 * vol_mean) or (prev_vol > 1.5 * vol_mean)
print(f"Volume spike detected: {vol_spike}")
print(f"  Current vol check: {current_vol} > {1.5 * vol_mean} = {current_vol > 1.5 * vol_mean}")
print(f"  Previous vol check: {prev_vol} > {1.5 * vol_mean} = {prev_vol > 1.5 * vol_mean}")

# Debug price levels
current_high = df['high'].iloc[-1]
current_low = df['low'].iloc[-1]
current_close = df['close'].iloc[-1]

print(f"Current high: {current_high}, low: {current_low}, close: {current_close}")

# New logic: Check sweeps in recent history
recent_bars = df.tail(min(5, len(df)))
sweep_low_detected = any(bar['low'] < irl for _, bar in recent_bars.iterrows())
sweep_high_detected = any(bar['high'] > irh for _, bar in recent_bars.iterrows())

# Volume spike check with new logic
recent_vols = df['volume'].tail(min(3, len(df)))
vol_spike_new = any(vol > 1.5 * vol_mean for vol in recent_vols)

print(f"Recent volumes: {list(recent_vols)}")
print(f"Volume spike (new logic): {vol_spike_new}")

print(f"Sweep low detected: {sweep_low_detected}")
print(f"Sweep high detected: {sweep_high_detected}")

# Check PO3 patterns
if sweep_high_detected and vol_spike_new:
    if current_close > irh:
        print("PO3 Pattern: High sweep + high break detected!")
    elif current_close < irl:
        print("PO3 Pattern: High sweep + low break detected!")
    elif current_close < (irl + (irh - irl) * 0.4):
        print("PO3 Pattern: High sweep + reversal detected!")

# Check reversal
reversal = reverses(df, 3)
print(f"Reversal detected: {reversal}")

# Check Bojan high
bojan = has_bojan_high(df)
print(f"Bojan high detected: {bojan}")

# Run PO3 detection
po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.5)
print(f"\nPO3 result: {po3}")