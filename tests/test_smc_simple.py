#!/usr/bin/env python3
"""
Simple SMC Engine Test - Minimal thresholds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from engine.io.tradingview_loader import load_tv
import pandas as pd
import numpy as np

def test_basic_smc():
    """Test basic SMC pattern recognition"""

    # Load data
    df = load_tv('ETH_4H')
    start_date = pd.to_datetime('2025-05-10', utc=True)
    end_date = pd.to_datetime('2025-05-25', utc=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"📊 Testing SMC on {len(df)} bars")

    # Simple Break of Structure detection
    print("\n🔍 Testing Break of Structure (BOS)...")

    # Calculate swing highs and lows (simplified)
    df['swing_high'] = df['high'].rolling(window=3, center=True).max() == df['high']
    df['swing_low'] = df['low'].rolling(window=3, center=True).min() == df['low']

    swing_highs = df[df['swing_high'] == True]['high']
    swing_lows = df[df['swing_low'] == True]['low']

    print(f"  Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")

    # Simple BOS detection - price breaking above recent swing high
    bos_signals = 0
    for i in range(10, len(df)):
        current_close = df.iloc[i]['close']
        recent_highs = swing_highs[swing_highs.index <= df.index[i]]

        if len(recent_highs) >= 2:
            latest_high = recent_highs.iloc[-1]
            prev_high = recent_highs.iloc[-2]

            # BOS: current close breaks above recent swing high
            if current_close > latest_high * 1.001:  # 0.1% break threshold
                bos_signals += 1
                print(f"    BOS Signal at {df.index[i]}: Close ${current_close:.2f} > High ${latest_high:.2f}")
                if bos_signals >= 3:  # Show max 3 examples
                    break

    print(f"  📈 BOS Signals found: {bos_signals}")

    # Simple Fair Value Gap detection
    print("\n🔍 Testing Fair Value Gaps (FVG)...")

    fvg_signals = 0
    for i in range(2, len(df)):
        # 3-bar FVG pattern: gap between bar[i-2] and bar[i]
        bar_minus_2 = df.iloc[i-2]
        bar_current = df.iloc[i]

        # Bullish FVG: gap between previous low and current high
        gap_size = bar_current['low'] - bar_minus_2['high']
        gap_pct = gap_size / bar_current['close']

        if gap_pct > 0.002:  # 0.2% minimum gap
            fvg_signals += 1
            print(f"    FVG Signal at {df.index[i]}: Gap {gap_pct*100:.2f}%")
            if fvg_signals >= 3:
                break

    print(f"  📈 FVG Signals found: {fvg_signals}")

    # Simple Order Block detection
    print("\n🔍 Testing Order Blocks (OB)...")

    ob_signals = 0
    for i in range(5, len(df)):
        # Look for displacement (strong move) followed by consolidation
        lookback = df.iloc[i-5:i+1]
        price_move = (lookback['close'].iloc[-1] - lookback['close'].iloc[0]) / lookback['close'].iloc[0]

        if abs(price_move) > 0.01:  # 1% displacement
            ob_signals += 1
            print(f"    OB Signal at {df.index[i]}: Displacement {price_move*100:.2f}%")
            if ob_signals >= 3:
                break

    print(f"  📈 Order Block Signals found: {ob_signals}")

    print("\n" + "="*40)
    print("🎯 SIMPLE SMC SUMMARY")
    print("="*40)
    print(f"BOS Signals: {bos_signals}")
    print(f"FVG Signals: {fvg_signals}")
    print(f"Order Block Signals: {ob_signals}")

    total_signals = bos_signals + fvg_signals + ob_signals
    print(f"Total SMC Signals: {total_signals}")

    if total_signals == 0:
        print("❌ No SMC signals detected - may need threshold adjustment")
    else:
        print(f"✅ Found {total_signals} potential SMC signals")

if __name__ == "__main__":
    test_basic_smc()