#!/usr/bin/env python3
"""Debug BOMS detection with real live data."""
import sys
sys.path.insert(0, '.')

from engine.structure.boms_detector import detect_boms, find_swing_points
from bin.live.coinbase_client import CoinbaseAdapter
import pandas as pd
import numpy as np

adapter = CoinbaseAdapter()
bars = adapter.fetch_ohlcv_1h(limit=500)
print(f"Fetched {len(bars)} 1H bars")
print(f"Price range: {bars.close.min():.0f} - {bars.close.max():.0f}")

# 4H resampled
bars_4h = bars.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
print(f"4H bars: {len(bars_4h)}")

# Run BOMS on each timeframe
for label, df, tf in [("1H", bars, "1H"), ("4H", bars_4h, "4H")]:
    boms = detect_boms(df, timeframe=tf)
    print(f"\n=== {label} BOMS ===")
    print(f"  detected: {boms.boms_detected}")
    print(f"  direction: {boms.direction}")
    print(f"  displacement: {boms.displacement:.1f}")
    print(f"  break_level: {boms.break_level:.0f}")
    print(f"  volume_surge: {boms.volume_surge:.2f}")
    print(f"  fvg_present: {boms.fvg_present}")

# Debug swing points
swings_4h = find_swing_points(bars_4h, window=20)
sh = swings_4h['swing_high']
sl = swings_4h['swing_low']
cur = bars_4h.close.iloc[-1]
print(f"\n=== 4H Swing Analysis ===")
print(f"  swing_high: ${sh:,.0f}")
print(f"  swing_low:  ${sl:,.0f}")
print(f"  current:    ${cur:,.0f}")
print(f"  above high? {cur > sh}")
print(f"  below low?  {cur < sl}")

# Volume analysis
vol_mean = bars_4h.volume.rolling(20).mean().iloc[-1]
print(f"\n=== Last 10 4H Bars ===")
for i in range(-10, 0):
    row = bars_4h.iloc[i]
    vs = row.volume / vol_mean if vol_mean > 0 else 0
    above = "ABOVE HIGH" if row.close > sh else ""
    below = "BELOW LOW" if row.close < sl else ""
    flag = above or below or "between"
    print(f"  {bars_4h.index[i]}: close=${row.close:,.0f} vol={vs:.2f}x {flag}")

# Why is BOMS returning 0?
print("\n=== ROOT CAUSE ANALYSIS ===")
print("BOMS requires ALL of:")
print(f"  1. Close beyond swing high ({sh:,.0f}) or swing low ({sl:,.0f})")
if cur < sh and cur > sl:
    print(f"     FAIL: Current ${cur:,.0f} is BETWEEN swings (no structure break)")
    print(f"     This is WHY displacement=0. Price hasn't broken structure.")
elif cur > sh:
    print(f"     PASS: Close ${cur:,.0f} > swing_high ${sh:,.0f}")
    print("  2. Volume > 1.5x mean")
    print("  3. FVG trail behind the move")
    print("  4. No immediate reversal")
elif cur < sl:
    print(f"     PASS: Close ${cur:,.0f} < swing_low ${sl:,.0f}")

# Also check what the feature computer produces for boms_strength
print("\n=== Simulated boms_strength computation ===")
boms_4h = detect_boms(bars_4h, timeframe='4H')
atr_14 = bars['close'].diff().abs().rolling(14).mean().iloc[-1]
disp = boms_4h.displacement
if atr_14 > 0 and disp > 0:
    boms_strength = min(disp / (2.0 * atr_14), 1.0)
else:
    boms_strength = 0.0
print(f"  boms_4h.displacement: {disp:.1f}")
print(f"  atr_14: {atr_14:.1f}")
print(f"  boms_strength: {boms_strength:.4f}")
print(f"  liquidity_score (used by archetypes): {boms_strength:.4f} (boms component)")
