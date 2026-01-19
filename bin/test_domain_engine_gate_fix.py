#!/usr/bin/env python3
"""
Domain Engine Gate Fix Validation Test

OBJECTIVE:
Prove that domain engines now correctly boost marginal signals to pass threshold.

TEST CASES:
1. S1 V2: score=0.38, threshold=0.40, wyckoff_spring_a boost=2.5x → PASS (0.95)
2. S1 V1: score=0.30, threshold=0.35, wyckoff_spring boost=1.25x → PASS (0.375)
3. S4: score=0.35, threshold=0.40, wyckoff_accumulation boost=2.0x → PASS (0.70)
4. S5: score=0.30, threshold=0.35, wyckoff_utad boost=2.5x → PASS (0.75)
5. Veto test: wyckoff_distribution should still block regardless of score

EXPECTED BEHAVIOR:
- WITHOUT domain engines: All signals REJECTED
- WITH domain engines: All signals ACCEPTED (except veto case)
- Vetoes execute BEFORE boosts (safety first)
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext


def create_test_context(archetype_name: str, base_features: dict, domain_features: dict = None) -> RuntimeContext:
    """
    Create a synthetic RuntimeContext for testing.

    Args:
        archetype_name: Name of archetype being tested
        base_features: Base signal features (for scoring)
        domain_features: Domain engine features (wyckoff, smc, etc.)

    Returns:
        RuntimeContext with synthetic data
    """
    # Merge features
    all_features = {**base_features}
    if domain_features:
        all_features.update(domain_features)

    # Create synthetic row
    row = pd.Series(all_features)

    # Create minimal config with feature flags
    config = {
        'feature_flags': {
            'enable_wyckoff': True,
            'enable_smc': True,
            'enable_temporal': True,
            'enable_hob': False,
            'enable_macro': False
        },
        'thresholds': {
            archetype_name: {}  # Use defaults
        }
    }

    # Create context
    context = RuntimeContext(
        row=row,
        config=config,
        metadata={'feature_flags': config['feature_flags']}
    )

    return context


def test_s1_v2_marginal_signal():
    """
    TEST 1: S1 V2 - Marginal signal boosted by wyckoff_spring_a

    Setup:
    - Base score ≈ 0.38 (below threshold=0.40)
    - wyckoff_spring_a = True (2.5x boost)
    - Expected: 0.38 * 2.5 = 0.95 > 0.40 → PASS
    """
    print("\n" + "="*80)
    print("TEST 1: S1 V2 - Marginal Signal with Wyckoff Spring A Boost")
    print("="*80)

    # Create base features to generate score ≈ 0.38
    base_features = {
        # V2 required features
        'capitulation_depth': -0.25,  # 25% drawdown
        'crisis_composite': 0.60,     # Crisis environment
        'volume_climax_3b': 0.35,     # Multi-bar volume exhaustion
        'wick_exhaustion_3b': 0.30,   # Multi-bar wick exhaustion
        'liquidity_drain_pct': -0.30, # Liquidity drained
        'liquidity_velocity': -0.08,  # Negative velocity
        'liquidity_persistence': 5,   # Persistent drain
        'funding_Z': -0.6,            # Some funding reversal
        'rsi_14': 35,                 # Somewhat oversold
        'atr_percentile': 0.85,       # High volatility

        # Domain features (NO boost initially)
        'wyckoff_spring_a': False,
        'wyckoff_distribution': False,
        'wyckoff_phase_abc': 'accumulation'
    }

    # Test WITHOUT boost
    print("\n1a. WITHOUT domain boost:")
    context_no_boost = create_test_context('liquidity_vacuum', base_features)
    logic = ArchetypeLogic(context_no_boost.config)
    matched_no_boost, score_no_boost, meta_no_boost = logic._check_S1(context_no_boost)

    print(f"  Score: {score_no_boost:.3f}")
    print(f"  Threshold: 0.40")
    print(f"  Matched: {matched_no_boost}")
    print(f"  Reason: {meta_no_boost.get('reason', 'N/A')}")

    # Test WITH boost
    print("\n1b. WITH wyckoff_spring_a (2.5x boost):")
    base_features['wyckoff_spring_a'] = True
    context_with_boost = create_test_context('liquidity_vacuum', base_features)
    logic = ArchetypeLogic(context_with_boost.config)
    matched_with_boost, score_with_boost, meta_with_boost = logic._check_S1(context_with_boost)

    print(f"  Score: {score_with_boost:.3f}")
    print(f"  Score before domain: {meta_with_boost.get('score_before_domain', 'N/A')}")
    print(f"  Domain boost: {meta_with_boost.get('domain_boost', 1.0):.2f}x")
    print(f"  Domain signals: {meta_with_boost.get('domain_signals', [])}")
    print(f"  Matched: {matched_with_boost}")

    # Validation
    assert not matched_no_boost, "Signal should be REJECTED without domain boost"
    assert matched_with_boost, "Signal should be ACCEPTED with domain boost"
    assert meta_with_boost.get('domain_boost', 1.0) == 2.5, "Domain boost should be 2.5x"

    print("\n✅ TEST 1 PASSED: Domain engines correctly boost marginal S1 signals")


def test_s1_v1_fallback():
    """
    TEST 2: S1 V1 Fallback - Marginal signal boosted by wyckoff_spring

    Setup:
    - Base score ≈ 0.30 (below threshold=0.35)
    - wyckoff_spring_a = True (1.25x boost in V1 mode)
    - Expected: 0.30 * 1.25 = 0.375 > 0.35 → PASS
    """
    print("\n" + "="*80)
    print("TEST 2: S1 V1 Fallback - Marginal Signal with Wyckoff Spring Boost")
    print("="*80)

    # V1 features (no V2 features available)
    base_features = {
        # V1 REQUIRED features
        'bid_ask_volume_ratio_mean': 0.15,  # Low liquidity
        'volume_zscore': 2.5,               # Volume panic
        'open': 40000,
        'close': 39000,
        'high': 41000,
        'low': 38500,                       # Wick rejection

        # V1 OPTIONAL features
        'funding_Z': -0.8,
        'VIX_Z': 1.5,
        'DXY_Z': 0.5,
        'rsi_14': 28,
        'atr_percentile': 0.90,
        'tf4h_external_trend': 'down',

        # Domain features
        'wyckoff_spring_a': False
    }

    # Test WITHOUT boost
    print("\n2a. WITHOUT domain boost:")
    context_no_boost = create_test_context('liquidity_vacuum', base_features)
    logic = ArchetypeLogic(context_no_boost.config)
    matched_no_boost, score_no_boost, meta_no_boost = logic._check_S1(context_no_boost)

    print(f"  Score: {score_no_boost:.3f}")
    print(f"  Mode: {meta_no_boost.get('mode', 'N/A')}")
    print(f"  Matched: {matched_no_boost}")

    # Test WITH boost
    print("\n2b. WITH wyckoff_spring_a (1.25x boost):")
    base_features['wyckoff_spring_a'] = True
    context_with_boost = create_test_context('liquidity_vacuum', base_features)
    logic = ArchetypeLogic(context_with_boost.config)
    matched_with_boost, score_with_boost, meta_with_boost = logic._check_S1(context_with_boost)

    print(f"  Score: {score_with_boost:.3f}")
    print(f"  Score before domain: {meta_with_boost.get('score_before_domain', 'N/A'):.3f}")
    print(f"  Domain boost: {meta_with_boost.get('domain_boost', 1.0):.2f}x")
    print(f"  Matched: {matched_with_boost}")

    # Validation
    assert not matched_no_boost, "V1 signal should be REJECTED without domain boost"
    assert matched_with_boost, "V1 signal should be ACCEPTED with domain boost"

    print("\n✅ TEST 2 PASSED: Domain engines work in V1 fallback mode")


def test_s4_funding_divergence():
    """
    TEST 3: S4 - Marginal signal boosted by wyckoff_accumulation

    Setup:
    - Base score ≈ 0.35 (below threshold=0.40)
    - wyckoff_accumulation = True (2.0x boost)
    - Expected: 0.35 * 2.0 = 0.70 > 0.40 → PASS
    """
    print("\n" + "="*80)
    print("TEST 3: S4 Funding Divergence - Wyckoff Accumulation Boost")
    print("="*80)

    base_features = {
        # S4 required features
        'funding_Z': -1.5,                   # Extreme negative funding
        'bid_ask_volume_ratio_mean': 0.20,  # Low liquidity
        'price_resilience': 0.65,           # Price holding up
        'volume_quiet': True,               # Coiled spring

        # Domain features
        'wyckoff_phase_abc': 'neutral',
        'wyckoff_accumulation': False,
        'tf4h_bos_bearish': False  # No veto
    }

    # Test WITHOUT boost
    print("\n3a. WITHOUT domain boost:")
    context_no_boost = create_test_context('funding_divergence', base_features)
    logic = ArchetypeLogic(context_no_boost.config)
    matched_no_boost, score_no_boost, meta_no_boost = logic._check_S4(context_no_boost)

    print(f"  Score: {score_no_boost:.3f}")
    print(f"  Matched: {matched_no_boost}")

    # Test WITH boost
    print("\n3b. WITH wyckoff_accumulation (2.0x boost):")
    base_features['wyckoff_phase_abc'] = 'accumulation'
    context_with_boost = create_test_context('funding_divergence', base_features)
    logic = ArchetypeLogic(context_with_boost.config)
    matched_with_boost, score_with_boost, meta_with_boost = logic._check_S4(context_with_boost)

    print(f"  Score: {score_with_boost:.3f}")
    print(f"  Score before domain: {meta_with_boost.get('score_before_domain', 'N/A'):.3f}")
    print(f"  Domain boost: {meta_with_boost.get('domain_boost', 1.0):.2f}x")
    print(f"  Matched: {matched_with_boost}")

    # Validation
    assert not matched_no_boost, "S4 signal should be REJECTED without domain boost"
    assert matched_with_boost, "S4 signal should be ACCEPTED with domain boost"
    assert meta_with_boost.get('domain_boost', 1.0) == 2.0, "Domain boost should be 2.0x"

    print("\n✅ TEST 3 PASSED: Domain engines work for S4")


def test_s5_long_squeeze():
    """
    TEST 4: S5 - Marginal signal boosted by wyckoff_utad

    Setup:
    - Base score ≈ 0.30 (below threshold=0.35)
    - wyckoff_utad = True (2.5x boost)
    - Expected: 0.30 * 2.5 = 0.75 > 0.35 → PASS
    """
    print("\n" + "="*80)
    print("TEST 4: S5 Long Squeeze - Wyckoff UTAD Boost")
    print("="*80)

    base_features = {
        # S5 required features
        'funding_Z': 1.5,                    # Positive funding extreme
        'rsi_14': 75,                        # Overbought
        'bid_ask_volume_ratio_mean': 0.20,  # Low liquidity
        'oi_change_24h': 0.12,              # OI spike

        # Domain features
        'wyckoff_utad': False,
        'wyckoff_accumulation': False,
        'wyckoff_spring_a': False,
        'tf1h_bos_bullish': False  # No veto
    }

    # Test WITHOUT boost
    print("\n4a. WITHOUT domain boost:")
    context_no_boost = create_test_context('long_squeeze', base_features)
    logic = ArchetypeLogic(context_no_boost.config)
    matched_no_boost, score_no_boost, meta_no_boost = logic._check_S5(context_no_boost)

    print(f"  Score: {score_no_boost:.3f}")
    print(f"  Matched: {matched_no_boost}")

    # Test WITH boost
    print("\n4b. WITH wyckoff_utad (2.5x boost):")
    base_features['wyckoff_utad'] = True
    context_with_boost = create_test_context('long_squeeze', base_features)
    logic = ArchetypeLogic(context_with_boost.config)
    matched_with_boost, score_with_boost, meta_with_boost = logic._check_S5(context_with_boost)

    print(f"  Score: {score_with_boost:.3f}")
    print(f"  Score before domain: {meta_with_boost.get('score_before_domain', 'N/A'):.3f}")
    print(f"  Domain boost: {meta_with_boost.get('domain_boost', 1.0):.2f}x")
    print(f"  Matched: {matched_with_boost}")

    # Validation
    assert not matched_no_boost, "S5 signal should be REJECTED without domain boost"
    assert matched_with_boost, "S5 signal should be ACCEPTED with domain boost"
    assert meta_with_boost.get('domain_boost', 1.0) == 2.5, "Domain boost should be 2.5x"

    print("\n✅ TEST 4 PASSED: Domain engines work for S5")


def test_veto_priority():
    """
    TEST 5: Veto Priority - Vetoes execute BEFORE boosts (safety)

    Setup:
    - High base score (would pass threshold)
    - wyckoff_distribution = True (VETO)
    - wyckoff_spring_a = True (BOOST)
    - Expected: VETO blocks trade regardless of boost
    """
    print("\n" + "="*80)
    print("TEST 5: Veto Priority - Safety First")
    print("="*80)

    base_features = {
        # S1 V2 features (high score to pass threshold)
        'capitulation_depth': -0.40,
        'crisis_composite': 0.80,
        'volume_climax_3b': 0.50,
        'wick_exhaustion_3b': 0.45,
        'liquidity_drain_pct': -0.40,
        'liquidity_velocity': -0.10,
        'liquidity_persistence': 8,
        'funding_Z': -1.2,
        'rsi_14': 25,
        'atr_percentile': 0.95,

        # BOTH veto AND boost present
        'wyckoff_distribution': True,  # VETO (should win)
        'wyckoff_spring_a': True,      # BOOST (should be ignored)
        'wyckoff_phase_abc': 'distribution'
    }

    print("\n5. Testing veto priority:")
    context = create_test_context('liquidity_vacuum', base_features)
    logic = ArchetypeLogic(context.config)
    matched, score, meta = logic._check_S1(context)

    print(f"  Score: {score:.3f}")
    print(f"  Matched: {matched}")
    print(f"  Reason: {meta.get('reason', 'N/A')}")
    print(f"  Wyckoff Distribution: {base_features['wyckoff_distribution']}")
    print(f"  Wyckoff Spring A: {base_features['wyckoff_spring_a']}")

    # Validation
    assert not matched, "Veto should BLOCK trade even with boost"
    assert 'wyckoff_distribution_veto' in meta.get('reason', ''), "Reason should mention veto"

    print("\n✅ TEST 5 PASSED: Vetoes execute before boosts (safety guaranteed)")


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("DOMAIN ENGINE GATE FIX - VALIDATION TEST SUITE")
    print("="*80)
    print("\nOBJECTIVE: Prove domain engines boost marginal signals BEFORE threshold gate")
    print("SAFETY: Verify vetoes execute BEFORE boosts")

    try:
        # Run all tests
        test_s1_v2_marginal_signal()
        test_s1_v1_fallback()
        test_s4_funding_divergence()
        test_s5_long_squeeze()
        test_veto_priority()

        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print("\nSUMMARY:")
        print("1. S1 V2: Marginal signals correctly boosted by domain engines")
        print("2. S1 V1: Domain engines work in fallback mode")
        print("3. S4: Funding divergence signals correctly boosted")
        print("4. S5: Long squeeze signals correctly boosted")
        print("5. Vetoes: Execute BEFORE boosts (safety guaranteed)")
        print("\nDomain engine gate fix is PRODUCTION READY.")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
