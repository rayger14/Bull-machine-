#!/usr/bin/env python3
"""
Test S1 V2 Regime Filter Implementation

Validates that regime filter correctly blocks bull market false positives
while allowing bear market capitulation signals.

Tests:
1. Regime-based blocking (risk_on regime blocked)
2. Regime-based allowance (risk_off, crisis regimes allowed)
3. Drawdown override (severe drawdown bypasses regime check)
4. Fallback to crisis_composite when regime_label missing
5. Backward compatibility (use_regime_filter=false)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext


def create_test_config(use_regime_filter: bool) -> dict:
    """Create test config with regime filter enabled/disabled."""
    return {
        'use_archetypes': True,
        'enable_S1': True,
        'thresholds': {
            'liquidity_vacuum': {
                'use_v2_logic': True,
                'use_regime_filter': use_regime_filter,
                'allowed_regimes': ['risk_off', 'crisis'],
                'drawdown_override_pct': 0.10,
                'require_regime_or_drawdown': True,

                # V2 thresholds (VERY loose to ensure pattern passes when regime allows)
                'capitulation_depth_max': -0.05,  # Only need 5% drawdown
                'crisis_composite_min': 0.20,     # Low crisis threshold
                'volume_climax_3b_min': 0.20,
                'wick_exhaustion_3b_min': 0.20,
                'fusion_threshold': 0.20,
            }
        }
    }


def create_test_row(**kwargs) -> pd.Series:
    """Create test data row with pattern that would normally trigger S1."""
    defaults = {
        # V2 features (strong capitulation pattern BUT below drawdown override)
        'capitulation_depth': -0.08,   # 8% drawdown (below 10% override threshold)
        'crisis_composite': 0.45,      # High crisis
        'volume_climax_last_3b': 0.50, # Strong volume
        'wick_exhaustion_last_3b': 0.60, # Strong wick
        'liquidity_drain_pct': -0.30,
        'liquidity_velocity': -0.05,
        'liquidity_persistence': 5,

        # Confluence features
        'funding_Z': -1.5,
        'rsi_14': 25,
        'atr_percentile': 0.85,

        # Regime label (this is what we'll vary)
        'regime_label': 'risk_off',
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_regime_filter_blocks_bull_market():
    """Test that regime filter blocks trades in bull markets (risk_on)."""
    print("\nTest 1: Regime filter blocks bull market trades")
    print("=" * 60)

    config = create_test_config(use_regime_filter=True)
    logic = ArchetypeLogic(config)

    # Create row with risk_on regime (bull market)
    row = create_test_row(regime_label='risk_on')

    # Create context with regime info
    context = RuntimeContext(
        ts=pd.Timestamp('2024-01-01'),
        row=row,
        regime_probs={'risk_on': 0.8, 'neutral': 0.1, 'risk_off': 0.05, 'crisis': 0.05},
        regime_label='risk_on',
        adapted_params={},
        thresholds=config.get('thresholds', {})
    )

    matched, score, meta = logic._check_S1(context)

    print(f"Regime: {row['regime_label']}")
    print(f"Matched: {matched}")
    print(f"Score: {score:.3f}")
    print(f"Reason: {meta.get('reason', 'N/A')}")

    assert not matched, "Should block bull market trade"
    assert meta['reason'] == 'regime_filter_blocked', "Should be blocked by regime filter"
    assert meta['current_regime'] == 'risk_on', "Should detect risk_on regime"
    assert not meta['regime_ok'], "risk_on should not be in allowed regimes"
    print("✓ PASS: Bull market trade correctly blocked")

    return True


def test_regime_filter_allows_bear_market():
    """Test that regime filter allows trades in bear markets (risk_off)."""
    print("\nTest 2: Regime filter allows bear market trades")
    print("=" * 60)

    config = create_test_config(use_regime_filter=True)
    logic = ArchetypeLogic(config)

    # Create row with risk_off regime (bear market)
    row = create_test_row(regime_label='risk_off')

    context = RuntimeContext(
        ts=pd.Timestamp('2022-06-01'),
        row=row,
        regime_probs={'risk_off': 0.8, 'crisis': 0.1, 'neutral': 0.05, 'risk_on': 0.05},
        regime_label='risk_off',
        adapted_params={},
        thresholds=config.get('thresholds', {})
    )

    matched, score, meta = logic._check_S1(context)

    print(f"Regime: {row['regime_label']}")
    print(f"Matched: {matched}")
    print(f"Score: {score:.3f}")
    print(f"Meta keys: {list(meta.keys())}")

    assert matched, "Should allow bear market trade"
    assert 'regime_filter_blocked' not in meta.get('reason', ''), "Should not be blocked"
    print("✓ PASS: Bear market trade correctly allowed")

    return True


def test_regime_filter_allows_crisis():
    """Test that regime filter allows trades during crisis."""
    print("\nTest 3: Regime filter allows crisis trades")
    print("=" * 60)

    config = create_test_config(use_regime_filter=True)
    logic = ArchetypeLogic(config)

    # Create row with crisis regime
    row = create_test_row(regime_label='crisis')

    context = RuntimeContext(
        ts=pd.Timestamp('2022-11-01'),
        row=row,
        regime_probs={'crisis': 0.9, 'risk_off': 0.08, 'neutral': 0.01, 'risk_on': 0.01},
        regime_label='crisis',
        adapted_params={},
        thresholds=config.get('thresholds', {})
    )

    matched, score, meta = logic._check_S1(context)

    print(f"Regime: {row['regime_label']}")
    print(f"Matched: {matched}")
    print(f"Score: {score:.3f}")

    assert matched, "Should allow crisis trade"
    assert 'regime_filter_blocked' not in meta.get('reason', ''), "Should not be blocked"
    print("✓ PASS: Crisis trade correctly allowed")

    return True


def test_drawdown_override():
    """Test that severe drawdown overrides regime filter."""
    print("\nTest 4: Drawdown override bypasses regime filter")
    print("=" * 60)

    config = create_test_config(use_regime_filter=True)
    logic = ArchetypeLogic(config)

    # Create row with risk_on regime BUT severe drawdown (>10%)
    row = create_test_row(
        regime_label='risk_on',
        capitulation_depth=-0.15  # 15% drawdown (exceeds 10% override)
    )

    context = RuntimeContext(
        ts=pd.Timestamp('2024-08-05'),
        row=row,
        regime_probs={'risk_on': 0.6, 'neutral': 0.3, 'risk_off': 0.08, 'crisis': 0.02},
        regime_label='risk_on',
        adapted_params={},
        thresholds=config.get('thresholds', {})
    )

    matched, score, meta = logic._check_S1(context)

    print(f"Regime: {row['regime_label']}")
    print(f"Drawdown: {row['capitulation_depth']:.1%}")
    print(f"Matched: {matched}")
    print(f"Score: {score:.3f}")

    assert matched, "Should allow trade due to drawdown override"
    print("✓ PASS: Drawdown override correctly bypassed regime filter")

    return True


def test_fallback_to_crisis_composite():
    """Test fallback to crisis_composite when regime_label missing."""
    print("\nTest 5: Fallback to crisis_composite when regime_label missing")
    print("=" * 60)

    config = create_test_config(use_regime_filter=True)
    logic = ArchetypeLogic(config)

    # Create row WITHOUT regime_label, but high crisis_composite
    row = create_test_row()
    row = row.drop('regime_label')  # Remove regime_label
    row['crisis_composite'] = 0.50  # High crisis = should infer risk_off

    context = RuntimeContext(
        ts=pd.Timestamp('2022-05-01'),
        row=row,
        regime_probs={'unknown': 1.0},
        regime_label='unknown',
        adapted_params={},
        thresholds=config.get('thresholds', {})
    )

    matched, score, meta = logic._check_S1(context)

    print(f"Regime label: {row.get('regime_label', 'MISSING')}")
    print(f"Crisis composite: {row['crisis_composite']:.2f}")
    print(f"Matched: {matched}")
    print(f"Score: {score:.3f}")

    assert matched, "Should infer risk_off from high crisis_composite and allow trade"
    print("✓ PASS: Fallback to crisis_composite works correctly")

    return True


def test_backward_compatibility():
    """Test that regime filter is disabled when use_regime_filter=false."""
    print("\nTest 6: Backward compatibility (use_regime_filter=false)")
    print("=" * 60)

    config = create_test_config(use_regime_filter=False)
    logic = ArchetypeLogic(config)

    # Create row with risk_on regime (would be blocked if filter enabled)
    row = create_test_row(regime_label='risk_on')

    context = RuntimeContext(
        ts=pd.Timestamp('2024-01-01'),
        row=row,
        regime_probs={'risk_on': 0.8, 'neutral': 0.1, 'risk_off': 0.05, 'crisis': 0.05},
        regime_label='risk_on',
        adapted_params={},
        thresholds=config.get('thresholds', {})
    )

    matched, score, meta = logic._check_S1(context)

    print(f"Regime: {row['regime_label']}")
    print(f"Regime filter enabled: False")
    print(f"Matched: {matched}")
    print(f"Score: {score:.3f}")

    assert matched, "Should allow trade when regime filter disabled"
    assert 'regime_filter_blocked' not in str(meta), "Should not have regime filter in metadata"
    print("✓ PASS: Backward compatibility maintained")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("S1 V2 REGIME FILTER VALIDATION")
    print("=" * 60)

    tests = [
        test_regime_filter_blocks_bull_market,
        test_regime_filter_allows_bear_market,
        test_regime_filter_allows_crisis,
        test_drawdown_override,
        test_fallback_to_crisis_composite,
        test_backward_compatibility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED")
        print("\nRegime filter implementation validated:")
        print("  - Blocks bull market false positives")
        print("  - Allows bear market and crisis capitulations")
        print("  - Drawdown override works correctly")
        print("  - Fallback to crisis_composite when regime missing")
        print("  - Backward compatible (filter can be disabled)")
        sys.exit(0)


if __name__ == '__main__':
    main()
