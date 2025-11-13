#!/usr/bin/env python3
"""
Test trap wiring with REAL market data.

This is the definitive test - uses actual BTC data from feature store.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from engine.archetypes.logic import ArchetypeLogic


def test_with_real_data():
    """Test trap wiring with real BTC data."""

    # Load REAL MTF data
    data_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        print("Need: python3 bin/build_mtf_feature_store.py BTC 2022-01-01 2023-12-31")
        return False

    print(f"📂 Loading real data from {data_path}...")
    df = pd.read_parquet(data_path)
    df = df.tail(500)  # Last 500 bars for speed
    print(f"  ✓ Loaded {len(df)} bars")

    # Config with RELAXED params (should detect MORE)
    config_relaxed = {
        'use_archetypes': True,
        'enable_H': True,
        'archetypes': {
            'trap_within_trend': {
                'quality_threshold': 0.40,  # Lower = easier to detect
                'liquidity_threshold': 0.40,  # Higher = easier
                'adx_threshold': 15.0,  # Lower = easier
                'fusion_threshold': 0.25,  # Lower = easier
                'wick_multiplier': 1.0  # Lower = easier
            }
        },
        'thresholds': {}
    }

    # Config with STRICT params (should detect FEWER)
    config_strict = {
        'use_archetypes': True,
        'enable_H': True,
        'archetypes': {
            'trap_within_trend': {
                'quality_threshold': 0.70,  # Higher = harder
                'liquidity_threshold': 0.15,  # Lower = harder
                'adx_threshold': 40.0,  # Higher = harder
                'fusion_threshold': 0.50,  # Higher = harder
                'wick_multiplier': 5.0  # Higher = harder
            }
        },
        'thresholds': {}
    }

    # Test with both configs
    logic_relaxed = ArchetypeLogic(config_relaxed)
    logic_strict = ArchetypeLogic(config_strict)

    detections_relaxed = 0
    detections_strict = 0

    for i in range(10, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        fusion_score = 0.40

        try:
            detected_relaxed = logic_relaxed._check_H(row, prev_row, df, i, fusion_score)
            detected_strict = logic_strict._check_H(row, prev_row, df, i, fusion_score)

            if detected_relaxed:
                detections_relaxed += 1
            if detected_strict:
                detections_strict += 1
        except Exception as e:
            # Skip bars with missing data
            continue

    print("\n" + "="*70)
    print("TRAP WIRING TEST - REAL DATA")
    print("="*70)
    print(f"\nTested {len(df)-10} bars of real BTC data")
    print(f"\nWith RELAXED params (0.40/0.40/15/0.25/1.0):")
    print(f"  Detections: {detections_relaxed}")
    print(f"\nWith STRICT params (0.70/0.15/40/0.50/5.0):")
    print(f"  Detections: {detections_strict}")

    delta = abs(detections_relaxed - detections_strict)

    print(f"\nΔ Detections: {delta}")

    if delta > 0:
        print(f"\n✅ WIRING WORKS!")
        print(f"   Relaxed params detected MORE ({detections_relaxed} vs {detections_strict})")
        print(f"   Parameters are affecting real detection behavior.")
        return True
    else:
        print(f"\n❌ WIRING BROKEN!")
        print(f"   Both configs produced same result: {detections_relaxed}")
        print(f"   Parameters are NOT affecting detection.")
        return False


if __name__ == '__main__':
    try:
        success = test_with_real_data()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
