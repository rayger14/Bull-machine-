#!/usr/bin/env python3
"""
Quick smoke test to verify trap parameters are properly wired.

This test creates two configs with extreme param values and verifies
that they produce different detection behavior.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from engine.archetypes.logic import ArchetypeLogic


def test_trap_wiring():
    """Verify that changing trap params actually changes behavior."""

    # Create minimal test data (just a few bars)
    test_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'open': [99, 100, 101, 102, 103],
        'high': [102, 103, 104, 105, 106],
        'low': [98, 99, 100, 101, 102],
        'tf4h_fusion_score': [0.6, 0.6, 0.6, 0.6, 0.6],
        'adx_14': [30, 30, 30, 30, 30],
        'tf1h_bos_flag': [1, 1, 1, 1, 1],
        'liquidity_score': [0.25, 0.25, 0.25, 0.25, 0.25]
    })

    # Config with TIGHT params (should pass checks)
    config_tight = {
        'use_archetypes': True,
        'enable_H': True,
        'archetypes': {
            'trap_within_trend': {
                'quality_threshold': 0.40,  # Lower = easier to pass
                'liquidity_threshold': 0.40,  # Higher = easier to pass
                'adx_threshold': 20.0,  # Lower = easier to pass
                'fusion_threshold': 0.20,  # Lower = easier to pass
                'wick_multiplier': 1.0  # Lower = easier to pass
            }
        },
        'thresholds': {}  # Empty to isolate new accessor
    }

    # Config with STRICT params (should fail checks)
    config_strict = {
        'use_archetypes': True,
        'enable_H': True,
        'archetypes': {
            'trap_within_trend': {
                'quality_threshold': 0.70,  # Higher = harder to pass
                'liquidity_threshold': 0.15,  # Lower = harder to pass
                'adx_threshold': 40.0,  # Higher = harder to pass
                'fusion_threshold': 0.50,  # Higher = harder to pass
                'wick_multiplier': 5.0  # Higher = harder to pass
            }
        },
        'thresholds': {}
    }

    # Test with tight params
    logic_tight = ArchetypeLogic(config_tight)
    row = test_data.iloc[2]
    prev_row = test_data.iloc[1]

    result_tight = logic_tight._check_H(row, prev_row, test_data, 2, fusion_score=0.40)

    # Test with strict params
    logic_strict = ArchetypeLogic(config_strict)
    result_strict = logic_strict._check_H(row, prev_row, test_data, 2, fusion_score=0.40)

    print("\n" + "="*60)
    print("TRAP WIRING SMOKE TEST")
    print("="*60)
    print(f"\nWith TIGHT params (0.40 quality_th, 0.40 liq_th, 20 adx_th):")
    print(f"  Result: {result_tight}")
    print(f"\nWith STRICT params (0.70 quality_th, 0.15 liq_th, 40 adx_th):")
    print(f"  Result: {result_strict}")

    if result_tight != result_strict:
        print(f"\n✅ WIRING WORKS! Parameters affect detection behavior.")
        print(f"   Tight: {result_tight}, Strict: {result_strict}")
        return True
    else:
        print(f"\n❌ WIRING BROKEN! Both configs produced same result: {result_tight}")
        print(f"   Parameters are not being read correctly.")
        return False


if __name__ == '__main__':
    try:
        success = test_trap_wiring()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
