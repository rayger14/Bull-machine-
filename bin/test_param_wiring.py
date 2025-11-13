#!/usr/bin/env python3
"""
Comprehensive parameter wiring tests.

Tests that all optimizable parameters actually affect backtest outcomes.
This prevents the 8-hour zero-variance optimization failure from happening again.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from engine.archetypes.logic import ArchetypeLogic


def create_test_data(n_bars=200):
    """Create synthetic test data for wire tests."""
    np.random.seed(42)

    return pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'open': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'high': 102 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'low': 98 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'tf4h_fusion_score': 0.55 + np.random.randn(n_bars) * 0.1,
        'adx_14': 28 + np.random.randn(n_bars) * 5,
        'tf1h_bos_flag': np.random.choice([0, 1, -1], n_bars),
        'liquidity_score': 0.28 + np.random.randn(n_bars) * 0.05,
        'volume': 1000 + np.random.randn(n_bars) * 100,
        'atr_20': 2.0 + np.random.randn(n_bars) * 0.3,
        # Add fields needed by helper methods
        'wyckoff_score': 0.4 + np.random.randn(n_bars) * 0.1,
        'momentum_score': 0.5 + np.random.randn(n_bars) * 0.1,
        'smc_score': 0.45 + np.random.randn(n_bars) * 0.1,
    })


def test_param_wiring(archetype: str, param: str, val_min, val_max, archetype_code: str = 'H'):
    """
    Test that a parameter affects archetype detection.

    Args:
        archetype: Archetype name (e.g., 'trap_within_trend')
        param: Parameter key (e.g., 'quality_threshold')
        val_min: Minimum value to test
        val_max: Maximum value to test
        archetype_code: Archetype code letter (e.g., 'H')

    Returns:
        True if parameter is wired, False otherwise
    """
    df = create_test_data(200)

    # Base config
    base_config = {
        'use_archetypes': True,
        f'enable_{archetype_code}': True,
        'archetypes': {
            archetype: {}
        },
        'thresholds': {}
    }

    # Config with MIN value
    config_min = base_config.copy()
    config_min['archetypes'] = {archetype: {param: val_min}}

    # Config with MAX value
    config_max = base_config.copy()
    config_max['archetypes'] = {archetype: {param: val_max}}

    # Test detection with both configs
    logic_min = ArchetypeLogic(config_min)
    logic_max = ArchetypeLogic(config_max)

    detections_min = 0
    detections_max = 0
    fusion_sum_min = 0.0
    fusion_sum_max = 0.0

    for i in range(10, len(df)):  # Start at 10 to have history
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        fusion_score = 0.40

        # Test with MIN config
        if archetype_code == 'H':
            detected_min = logic_min._check_H(row, prev_row, df, i, fusion_score)
            detected_max = logic_max._check_H(row, prev_row, df, i, fusion_score)
        else:
            # For other archetypes, add their check methods
            detected_min = False
            detected_max = False

        if detected_min:
            detections_min += 1
            fusion_sum_min += fusion_score

        if detected_max:
            detections_max += 1
            fusion_sum_max += fusion_score

    # Check if anything changed
    detection_delta = abs(detections_min - detections_max)
    is_wired = detection_delta > 0

    print(f"\n{'='*70}")
    print(f"WIRE TEST: {archetype}.{param}")
    print(f"{'='*70}")
    print(f"Range: {val_min} → {val_max}")
    print(f"Detections: {detections_min} → {detections_max} (Δ{detection_delta})")
    print(f"Status: {'✅ WIRED' if is_wired else '❌ NOT WIRED'}")
    print(f"{'='*70}\n")

    return is_wired


def main():
    """Run comprehensive wire tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE PARAMETER WIRING TESTS")
    print("="*70)
    print("Testing all optimizable parameters...")
    print()

    tests = [
        # Trap archetype parameters
        ('trap_within_trend', 'quality_threshold', 0.40, 0.70, 'H'),
        ('trap_within_trend', 'liquidity_threshold', 0.20, 0.40, 'H'),
        ('trap_within_trend', 'adx_threshold', 20.0, 35.0, 'H'),
        ('trap_within_trend', 'fusion_threshold', 0.25, 0.45, 'H'),
        ('trap_within_trend', 'wick_multiplier', 1.5, 3.0, 'H'),
    ]

    results = {}
    for archetype, param, val_min, val_max, code in tests:
        try:
            is_wired = test_param_wiring(archetype, param, val_min, val_max, code)
            results[f"{archetype}.{param}"] = is_wired
        except Exception as e:
            print(f"❌ ERROR testing {archetype}.{param}: {e}")
            import traceback
            traceback.print_exc()
            results[f"{archetype}.{param}"] = False

    # Summary
    print("\n" + "="*70)
    print("WIRE TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for key, status in results.items():
        status_icon = '✅' if status else '❌'
        print(f"  {status_icon} {key}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed < total:
        print("\n❌ WIRE TESTS FAILED")
        print("Some parameters are not properly connected.")
        print("DO NOT run Optuna optimization until this is fixed!")
        return 1
    else:
        print("\n✅ ALL WIRE TESTS PASSED")
        print("Parameters are properly wired and affecting behavior.")
        print("Safe to run optimization!")
        return 0


if __name__ == '__main__':
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
