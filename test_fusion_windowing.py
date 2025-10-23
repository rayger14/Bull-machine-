#!/usr/bin/env python3
"""
Test analyze_fusion() with different window sizes to diagnose the root cause
of constant domain scores in feature store builds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from engine.io.tradingview_loader import load_tv
from engine.fusion.domain_fusion import analyze_fusion

def test_fusion_windowing():
    """Test fusion with progressively larger windows"""

    print("=" * 80)
    print("Fusion Windowing Test - BTC Q3 2024")
    print("=" * 80)

    # Load data
    df_1h = load_tv('BTC_1H')
    df_4h = load_tv('BTC_4H')
    df_1d = load_tv('BTC_1D')

    # Filter to Q3 2024
    start = pd.Timestamp('2024-07-01', tz='UTC')
    end = pd.Timestamp('2024-09-30', tz='UTC')

    df_1h = df_1h[(df_1h.index >= start) & (df_1h.index <= end)]
    df_4h = df_4h[(df_4h.index >= start) & (df_4h.index <= end)]
    df_1d = df_1d[(df_1d.index >= start) & (df_1d.index <= end)]

    # Test 1: Full period (like direct call)
    print("\n" + "=" * 80)
    print("TEST 1: Full Period (All Q3 2024 Data)")
    print("=" * 80)

    config = {'fusion': {'weights': {'wyckoff': 0.30, 'smc': 0.15, 'liquidity': 0.25, 'momentum': 0.30}}}

    result = analyze_fusion(df_1h, df_4h, df_1d, config)

    print(f"\nDomain Scores:")
    print(f"  Wyckoff:  {result.wyckoff_score:.4f}")
    print(f"  SMC:      {result.smc_score:.4f}")
    print(f"  HOB:      {result.hob_score:.4f}")
    print(f"  Momentum: {result.momentum_score:.4f}")
    print(f"\nFusion Score: {result.score:.4f}")
    print(f"Direction: {result.direction}")

    # Test 2: Small window (like first timestamp in feature store build)
    print("\n" + "=" * 80)
    print("TEST 2: Small Window (First 50/100/200 bars)")
    print("=" * 80)

    # Simulate what happens on first timestamp
    timestamp_1d = df_1d.index[20]  # 20th 1D bar

    window_1d = df_1d[df_1d.index <= timestamp_1d].tail(50)
    window_4h = df_4h[df_4h.index <= timestamp_1d].tail(100)
    window_1h = df_1h[df_1h.index <= timestamp_1d].tail(200)

    print(f"\nWindow sizes:")
    print(f"  1D: {len(window_1d)} bars")
    print(f"  4H: {len(window_4h)} bars")
    print(f"  1H: {len(window_1h)} bars")

    result_small = analyze_fusion(window_1h, window_4h, window_1d, config)

    print(f"\nDomain Scores:")
    print(f"  Wyckoff:  {result_small.wyckoff_score:.4f}")
    print(f"  SMC:      {result_small.smc_score:.4f}")
    print(f"  HOB:      {result_small.hob_score:.4f}")
    print(f"  Momentum: {result_small.momentum_score:.4f}")
    print(f"\nFusion Score: {result_small.score:.4f}")
    print(f"Direction: {result_small.direction}")

    # Test 3: Medium window (mid-period)
    print("\n" + "=" * 80)
    print("TEST 3: Medium Window (Mid-period)")
    print("=" * 80)

    timestamp_1d_mid = df_1d.index[len(df_1d)//2]  # Middle timestamp

    window_1d_mid = df_1d[df_1d.index <= timestamp_1d_mid].tail(50)
    window_4h_mid = df_4h[df_4h.index <= timestamp_1d_mid].tail(100)
    window_1h_mid = df_1h[df_1h.index <= timestamp_1d_mid].tail(200)

    print(f"\nWindow sizes:")
    print(f"  1D: {len(window_1d_mid)} bars")
    print(f"  4H: {len(window_4h_mid)} bars")
    print(f"  1H: {len(window_1h_mid)} bars")

    result_mid = analyze_fusion(window_1h_mid, window_4h_mid, window_1d_mid, config)

    print(f"\nDomain Scores:")
    print(f"  Wyckoff:  {result_mid.wyckoff_score:.4f}")
    print(f"  SMC:      {result_mid.smc_score:.4f}")
    print(f"  HOB:      {result_mid.hob_score:.4f}")
    print(f"  Momentum: {result_mid.momentum_score:.4f}")
    print(f"\nFusion Score: {result_mid.score:.4f}")
    print(f"Direction: {result_mid.direction}")

    # Test 4: Check why structure alignment is 0
    print("\n" + "=" * 80)
    print("TEST 4: Structure Detector Investigation")
    print("=" * 80)

    from engine.structure.internal_external import detect_structure_state

    # Test with full data
    structure_full = detect_structure_state(df_1h, df_4h, df_1d, config)
    print(f"\nFull period structure:")
    print(f"  Internal phase: {structure_full.internal_phase}")
    print(f"  External trend: {structure_full.external_trend}")
    print(f"  Alignment: {structure_full.alignment}")
    print(f"  Conflict score: {structure_full.conflict_score}")

    # Test with small window
    structure_small = detect_structure_state(window_1h, window_4h, window_1d, config)
    print(f"\nSmall window structure:")
    print(f"  Internal phase: {structure_small.internal_phase}")
    print(f"  External trend: {structure_small.external_trend}")
    print(f"  Alignment: {structure_small.alignment}")
    print(f"  Conflict score: {structure_small.conflict_score}")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    test_fusion_windowing()
