#!/usr/bin/env python3

import sys
sys.path.append('.')

import pandas as pd
import numpy as np

# Copy the exact PO3 detection logic to debug
def debug_detect_po3(df, irh, irl, vol_spike_threshold=1.4, reverse_bars=3):
    """Debug version of detect_po3 with verbose output."""
    print(f"\n=== PO3 Detection Debug ===")
    print(f"IRH: {irh}, IRL: {irl}")
    print(f"Volume spike threshold: {vol_spike_threshold}")

    if len(df) < 2:
        print("Insufficient data: len(df) < 2")
        return None

    # Calculate volume statistics
    vol_mean = df['volume'].rolling(min(10, len(df))).mean().iloc[-1]
    if vol_mean == 0 or pd.isna(vol_mean):
        vol_mean = df['volume'].mean()
    print(f"Volume mean: {vol_mean}")

    # Look for volume spikes in recent bars
    recent_vols = df['volume'].tail(min(3, len(df)))
    vol_spike = any(vol > vol_spike_threshold * vol_mean for vol in recent_vols)
    print(f"Recent volumes: {list(recent_vols)}")
    print(f"Volume spike detected: {vol_spike}")

    # Look for sweep patterns
    recent_bars = df.tail(min(5, len(df)))
    sweep_low_detected = any(bar['low'] < irl for _, bar in recent_bars.iterrows())
    sweep_high_detected = any(bar['high'] > irh for _, bar in recent_bars.iterrows())
    print(f"Sweep low detected: {sweep_low_detected}")
    print(f"Sweep high detected: {sweep_high_detected}")

    # Current price levels
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    current_close = df['close'].iloc[-1]
    print(f"Current: high={current_high}, low={current_low}, close={current_close}")

    # Check reversal
    from bull_machine.strategy.po3_detection import reverses, has_bojan_high
    reversal = reverses(df, reverse_bars)
    print(f"Reversal detected: {reversal}")

    bojan = has_bojan_high(df)
    print(f"Bojan high detected: {bojan}")

    # Pattern checks
    print(f"\n=== Pattern Checks ===")

    # Pattern 1: Low sweep followed by high break
    print(f"Pattern 1 check: sweep_low_detected={sweep_low_detected} AND vol_spike={vol_spike}")
    if sweep_low_detected and vol_spike:
        print(f"  Current close > IRH: {current_close} > {irh} = {current_close > irh}")
        range_60_pct = irl + (irh - irl) * 0.6
        print(f"  60% range level: {range_60_pct}")
        print(f"  Close > 60% range: {current_close} > {range_60_pct} = {current_close > range_60_pct}")
        print(f"  Reversal: {reversal}")

        if current_close > irh or (current_close > range_60_pct and reversal):
            bojan_boost = 0.10 if bojan else 0
            result = {'po3_type': 'low_sweep_high_break', 'strength': 0.70 + bojan_boost}
            print(f"  MATCH! Pattern 1: {result}")
            return result

    # Pattern 2: High sweep followed by low break
    print(f"Pattern 2 check: sweep_high_detected={sweep_high_detected} AND vol_spike={vol_spike}")
    if sweep_high_detected and vol_spike:
        print(f"  Current close < IRL: {current_close} < {irl} = {current_close < irl}")
        range_40_pct = irl + (irh - irl) * 0.4
        print(f"  40% range level: {range_40_pct}")
        print(f"  Close < 40% range: {current_close} < {range_40_pct} = {current_close < range_40_pct}")
        print(f"  Reversal: {reversal}")

        if current_close < irl or (current_close < range_40_pct and reversal):
            bojan_boost = 0.10 if bojan else 0
            result = {'po3_type': 'high_sweep_low_break', 'strength': 0.70 + bojan_boost}
            print(f"  MATCH! Pattern 2: {result}")
            return result

    # Pattern 3: Low sweep with strong reversal in range
    print(f"Pattern 3 check: sweep_low_detected={sweep_low_detected} AND vol_spike={vol_spike}")
    range_60_pct = irl + (irh - irl) * 0.6
    if sweep_low_detected and vol_spike and current_close > range_60_pct:
        print(f"  Close > 60% range AND reversal: {current_close} > {range_60_pct} AND {reversal}")
        if reversal:
            bojan_boost = 0.10 if bojan else 0
            result = {'po3_type': 'low_sweep_reversal', 'strength': 0.65 + bojan_boost}
            print(f"  MATCH! Pattern 3: {result}")
            return result

    # Pattern 4: High sweep with strong reversal in range
    print(f"Pattern 4 check: sweep_high_detected={sweep_high_detected} AND vol_spike={vol_spike}")
    range_40_pct = irl + (irh - irl) * 0.4
    if sweep_high_detected and vol_spike and current_close < range_40_pct:
        print(f"  Close < 40% range AND reversal: {current_close} < {range_40_pct} AND {reversal}")
        if reversal:
            bojan_boost = 0.10 if bojan else 0
            result = {'po3_type': 'high_sweep_reversal', 'strength': 0.65 + bojan_boost}
            print(f"  MATCH! Pattern 4: {result}")
            return result

    print("No PO3 pattern matched")
    return None

# Test the problematic case
df = pd.DataFrame({
    'high': [100, 105, 110, 108, 112],
    'low': [95, 100, 105, 102, 107],
    'close': [98, 104, 108, 106, 111],
    'volume': [1000, 1200, 2000, 1800, 2500]
})

irh = 110
irl = 95

result = debug_detect_po3(df, irh, irl, vol_spike_threshold=1.4)
print(f"\nFinal result: {result}")