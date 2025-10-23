#!/usr/bin/env python3
"""
Test BOMS on 4H timeframe - less strict than 1D
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from engine.io.tradingview_loader import load_tv
from engine.structure.boms_detector import detect_boms

def test_boms_4h():
    """Test BOMS detector on 4H timeframe."""

    print("=" * 80)
    print("BOMS Test: BTC 4H Q3 2024")
    print("=" * 80)

    # Load BTC 4H
    df_4h_raw = load_tv("BTC_4H")
    df_4h = df_4h_raw.copy()
    df_4h.columns = [c.lower() for c in df_4h.columns]

    # Filter to Q3 2024 with warmup
    start_ts = pd.Timestamp('2024-07-01', tz='UTC')
    end_ts = pd.Timestamp('2024-09-30', tz='UTC')
    warmup_start = start_ts - pd.Timedelta(days=90)

    df_full = df_4h[(df_4h.index >= warmup_start) & (df_4h.index <= end_ts)].copy()
    df_q3 = df_4h[(df_4h.index >= start_ts) & (df_4h.index <= end_ts)].copy()

    print(f"\nLoaded {len(df_full)} bars (with warmup), {len(df_q3)} bars in Q3 2024")

    # Test BOMS detection per timestamp
    detections = []
    for timestamp in df_q3.index:
        window = df_full[df_full.index <= timestamp].tail(50)

        if len(window) < 30:
            continue

        result = detect_boms(window, timeframe='4H', config={})

        if result.boms_detected:
            detections.append({
                'timestamp': timestamp,
                'direction': result.direction,
                'volume_surge': result.volume_surge,
                'displacement': result.displacement,
                'fvg_present': result.fvg_present
            })

    print(f"\n{'=' * 80}")
    print(f"BOMS Detected (4H): {len(detections)}")
    print(f"{'=' * 80}")

    if detections:
        print("\nDetected BOMS:")
        for det in detections[:10]:  # Show first 10
            print(f"  {det['timestamp']}: {det['direction'].upper()}")
            print(f"    Volume: {det['volume_surge']:.2f}x, Disp: {det['displacement']:.2%}, FVG: {det['fvg_present']}")
    else:
        print("\n❌ No 4H BOMS detected in Q3 2024")
        print("   → BOMS is a rare, high-conviction signal")
        print("   → Feature store returning all False is VALID if no BOMS occurred")

if __name__ == '__main__':
    test_boms_4h()
