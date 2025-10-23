#!/usr/bin/env python3
"""
BOMS Diagnostic Tool - Show why BOMS detector returns all False

Tests each of the 4 BOMS conditions separately to identify bottlenecks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from engine.io.tradingview_loader import load_tv
from engine.structure.boms_detector import (
    detect_boms,
    find_swing_points,
    detect_fvg_trail,
    check_no_immediate_reversal
)

def test_boms_conditions():
    """Test BOMS detector conditions step-by-step."""

    print("=" * 80)
    print("BOMS Diagnostic: BTC 1D Q3 2024")
    print("=" * 80)

    # Load BTC 1D with warmup
    df_1d_raw = load_tv("BTC_1D")
    df_1d = df_1d_raw.copy()
    df_1d.columns = [c.lower() for c in df_1d.columns]

    # Filter to Q3 2024
    start_ts = pd.Timestamp('2024-07-01', tz='UTC')
    end_ts = pd.Timestamp('2024-09-30', tz='UTC')
    warmup_start = start_ts - pd.Timedelta(days=360)

    df_full = df_1d[(df_1d.index >= warmup_start) & (df_1d.index <= end_ts)].copy()
    df_q3 = df_1d[(df_1d.index >= start_ts) & (df_1d.index <= end_ts)].copy()

    print(f"\nLoaded {len(df_full)} bars (with warmup), {len(df_q3)} bars in Q3 2024")
    print(f"Date range: {df_q3.index[0].date()} to {df_q3.index[-1].date()}")

    # Track condition failures
    stats = {
        'total_bars_tested': 0,
        'swing_breaks': 0,  # Condition 1: Price broke swing
        'volume_confirmed': 0,  # Condition 2: Volume > 1.8x
        'fvg_present': 0,  # Condition 3: FVG trail
        'no_reversal': 0,  # Condition 4: No reversal
        'boms_detected': 0  # All 4 conditions met
    }

    detections = []

    # Test each Q3 bar with expanding window
    for i, timestamp in enumerate(df_q3.index):
        stats['total_bars_tested'] += 1

        # Use ALL history up to this timestamp
        window = df_full[df_full.index <= timestamp].tail(50)

        if len(window) < 40:
            continue

        # Find swing points (30-bar window for 1D)
        swings = find_swing_points(window, window=30)
        swing_high = swings['swing_high']
        swing_low = swings['swing_low']

        # Check last 5 bars
        for j in range(len(window) - 5, len(window)):
            if j < 0:
                continue

            close = window['close'].iloc[j]
            volume = window['volume'].iloc[j]
            vol_mean = window['volume'].rolling(20).mean().iloc[j]

            if vol_mean == 0 or np.isnan(vol_mean):
                continue

            volume_surge = volume / vol_mean

            # CONDITION 1: Check swing break
            if close > swing_high:
                stats['swing_breaks'] += 1
                direction = 'bullish'
                swing_level = swing_high

                # CONDITION 2: Check volume
                if volume_surge > 1.8:
                    stats['volume_confirmed'] += 1

                    # CONDITION 3: Check FVG
                    fvg_present = detect_fvg_trail(window, j - 3, 'bullish')
                    if fvg_present:
                        stats['fvg_present'] += 1

                        # CONDITION 4: Check no reversal
                        no_reversal = check_no_immediate_reversal(window, j, 'bullish', bars=3)
                        if no_reversal:
                            stats['no_reversal'] += 1
                            stats['boms_detected'] += 1

                            detections.append({
                                'timestamp': timestamp,
                                'direction': 'bullish',
                                'close': close,
                                'swing_level': swing_level,
                                'volume_surge': volume_surge,
                                'displacement': (close - swing_level) / swing_level
                            })

            elif close < swing_low:
                stats['swing_breaks'] += 1
                direction = 'bearish'
                swing_level = swing_low

                # CONDITION 2: Check volume
                if volume_surge > 1.8:
                    stats['volume_confirmed'] += 1

                    # CONDITION 3: Check FVG
                    fvg_present = detect_fvg_trail(window, j - 3, 'bearish')
                    if fvg_present:
                        stats['fvg_present'] += 1

                        # CONDITION 4: Check no reversal
                        no_reversal = check_no_immediate_reversal(window, j, 'bearish', bars=3)
                        if no_reversal:
                            stats['no_reversal'] += 1
                            stats['boms_detected'] += 1

                            detections.append({
                                'timestamp': timestamp,
                                'direction': 'bearish',
                                'close': close,
                                'swing_level': swing_level,
                                'volume_surge': volume_surge,
                                'displacement': (swing_level - close) / swing_level
                            })

    # Print results
    print("\n" + "=" * 80)
    print("BOMS Condition Funnel Analysis")
    print("=" * 80)
    print(f"Total bars tested: {stats['total_bars_tested']}")
    print(f"\nCondition 1 - Swing break (close > swing_high or < swing_low):")
    print(f"  ✓ {stats['swing_breaks']} bars ({stats['swing_breaks']/stats['total_bars_tested']*100:.1f}%)")

    if stats['swing_breaks'] > 0:
        print(f"\nCondition 2 - Volume surge (> 1.8x mean):")
        print(f"  ✓ {stats['volume_confirmed']} bars ({stats['volume_confirmed']/stats['swing_breaks']*100:.1f}% of swing breaks)")

        if stats['volume_confirmed'] > 0:
            print(f"\nCondition 3 - FVG present (imbalance trail):")
            print(f"  ✓ {stats['fvg_present']} bars ({stats['fvg_present']/stats['volume_confirmed']*100:.1f}% of volume-confirmed)")

            if stats['fvg_present'] > 0:
                print(f"\nCondition 4 - No immediate reversal (3 bars):")
                print(f"  ✓ {stats['no_reversal']} bars ({stats['no_reversal']/stats['fvg_present']*100:.1f}% of FVG-confirmed)")

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT: {stats['boms_detected']} BOMS detected")
    print(f"{'=' * 80}")

    if detections:
        print("\nDetected BOMS:")
        for det in detections:
            print(f"  {det['timestamp'].date()}: {det['direction'].upper()} BOMS")
            print(f"    Close: ${det['close']:.2f}, Swing: ${det['swing_level']:.2f}")
            print(f"    Volume surge: {det['volume_surge']:.2f}x")
            print(f"    Displacement: {det['displacement']:.2%}")
    else:
        print("\n❌ No BOMS detected - analyzing bottleneck...")

        # Identify the tightest constraint
        if stats['swing_breaks'] == 0:
            print("\n⚠️  BOTTLENECK: No swing breaks detected")
            print("    → Price never closed beyond 30-bar swing high/low in Q3 2024")
            print("    → Consider shorter swing_window (e.g., 20 bars instead of 30)")

        elif stats['volume_confirmed'] == 0:
            print("\n⚠️  BOTTLENECK: Volume threshold too strict (1.8x)")
            print(f"    → {stats['swing_breaks']} swing breaks, but none had volume > 1.8x mean")
            print("    → Consider lowering to 1.5x or 1.3x for 1D timeframe")

        elif stats['fvg_present'] == 0:
            print("\n⚠️  BOTTLENECK: FVG requirement too strict")
            print(f"    → {stats['volume_confirmed']} volume-confirmed breaks, but none had FVG trail")
            print("    → FVG requires 0.1% gap (high[i] < low[i+2])")
            print("    → Consider relaxing FVG gap threshold or making it optional")

        elif stats['no_reversal'] == 0:
            print("\n⚠️  BOTTLENECK: Reversal check too strict")
            print(f"    → {stats['fvg_present']} FVG-confirmed breaks, but all reversed within 3 bars")
            print("    → Consider shorter confirmation window (1-2 bars instead of 3)")

if __name__ == '__main__':
    test_boms_conditions()
