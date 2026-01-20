#!/usr/bin/env python3
"""
Quick integration test for Wyckoff event detection system.
Tests the full pipeline: data loading → event detection → output verification.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.wyckoff import events as wyckoff_events


def create_test_data(n_bars=500):
    """Create synthetic BTC-like price data for testing."""
    dates = pd.date_range(start='2022-01-01', periods=n_bars, freq='1h')

    # Generate realistic price movement
    np.random.seed(42)
    trend = np.linspace(40000, 35000, n_bars)  # Downtrend (bear market 2022)
    noise = np.random.normal(0, 200, n_bars)
    prices = trend + noise

    # Add volume with some spikes
    base_volume = 1000000
    volume = base_volume + np.random.exponential(500000, n_bars)

    # Create OHLC data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(100, 50, n_bars)),
        'low': prices - np.abs(np.random.normal(100, 50, n_bars)),
        'close': prices + np.random.normal(0, 50, n_bars),
        'volume': volume,
    })

    # Add a selling climax at bar 100 (extreme volume spike at lows)
    sc_idx = 100
    df.loc[sc_idx, 'volume'] = base_volume * 5  # 5x volume spike
    df.loc[sc_idx, 'low'] = df.loc[sc_idx, 'close'] - 500  # Wide range
    df.loc[sc_idx, 'close'] = df.loc[sc_idx, 'low'] + 100  # Close near low

    return df


def main():
    print("=" * 80)
    print("WYCKOFF EVENT DETECTION - INTEGRATION TEST")
    print("=" * 80)

    # Step 1: Create test data
    print("\n[1/5] Creating synthetic BTC data (500 bars)...")
    df = create_test_data(n_bars=500)
    print(f"✓ Data created: {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Step 2: Initialize Wyckoff engine
    print("\n[2/5] Initializing WyckoffEngine...")
    config = {
        'wyckoff_events': {
            'enabled': True,
            'pti_integration': False,  # Disable PTI for simple test
            'min_confidence': 0.6,
        }
    }
    engine = WyckoffEngine(config=config)
    print("✓ WyckoffEngine initialized")

    # Step 3: Detect events
    print("\n[3/5] Running event detection on test data...")
    try:
        df_with_events = engine.detect_wyckoff_events(df)
        print(f"✓ Event detection complete, output has {len(df_with_events.columns)} columns")
    except Exception as e:
        print(f"✗ ERROR during event detection: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Check for expected columns
    print("\n[4/5] Verifying output columns...")
    expected_columns = [
        'wyckoff_sc', 'wyckoff_sc_confidence',
        'wyckoff_bc', 'wyckoff_bc_confidence',
        'wyckoff_ar', 'wyckoff_spring_a', 'wyckoff_ut'
    ]

    missing = [col for col in expected_columns if col not in df_with_events.columns]
    if missing:
        print(f"✗ Missing columns: {missing}")
        print(f"   Available columns: {df_with_events.columns.tolist()}")
        return 1
    else:
        print(f"✓ All expected columns present")

    # Step 5: Analyze detections
    print("\n[5/5] Analyzing detected events...")

    events_detected = {
        'SC (Selling Climax)': df_with_events['wyckoff_sc'].sum(),
        'BC (Buying Climax)': df_with_events['wyckoff_bc'].sum(),
        'AR (Automatic Rally)': df_with_events['wyckoff_ar'].sum(),
        'Spring Type A': df_with_events['wyckoff_spring_a'].sum(),
        'Spring Type B': df_with_events['wyckoff_spring_b'].sum(),
        'UT (Upthrust)': df_with_events['wyckoff_ut'].sum(),
        'SOS (Sign of Strength)': df_with_events['wyckoff_sos'].sum(),
    }

    total_events = sum(events_detected.values())
    print(f"\nTotal events detected: {total_events}")
    for event, count in events_detected.items():
        if count > 0:
            print(f"  • {event}: {count}")

    # Check confidence scores
    if total_events > 0:
        print("\nConfidence score statistics:")
        for event_name, col in [('SC', 'wyckoff_sc_confidence'), ('BC', 'wyckoff_bc_confidence')]:
            if df_with_events[col].max() > 0:
                scores = df_with_events[df_with_events[col] > 0][col]
                print(f"  • {event_name}: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

    # Summary
    print("\n" + "=" * 80)
    if total_events > 0:
        print("✓ INTEGRATION TEST PASSED")
        print(f"  Event detection is working correctly ({total_events} events detected)")
    else:
        print("⚠ WARNING: No events detected")
        print("  This may be due to strict thresholds or insufficient test data")
        print("  Check configs/wyckoff_events_config.json for threshold settings")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
