#!/usr/bin/env python3
"""
Test S5 Graceful Degradation: Verify OI-optional behavior

Tests:
1. 2024 scenario: OI data available → 4-component scoring
2. 2022 scenario: OI data missing → 3-component scoring with weight redistribution
3. Both: Core logic (funding + RSI + liquidity) fires pattern consistently
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.archetypes.logic_v2_adapter import ArchetypeLogicV2Adapter
from engine.runtime.context import RuntimeContext
from engine.archetypes.threshold_policy import ThresholdPolicy


def create_test_context(row_data: dict, timeframe: str = "1h") -> RuntimeContext:
    """Create a minimal RuntimeContext for testing"""
    df = pd.DataFrame([row_data])

    # Create threshold policy
    policy = ThresholdPolicy(
        regime="neutral",
        base_thresholds={},
        regime_adjustments={}
    )

    return RuntimeContext(
        row=df.iloc[0],
        df=df,
        symbol="BTCUSDT",
        timeframe=timeframe,
        index=0,
        threshold_policy=policy
    )


def test_s5_with_oi_data():
    """Test S5 with OI data available (2024 scenario)"""
    print("\n" + "="*80)
    print("TEST 1: S5 with OI Data Available (2024 Scenario)")
    print("="*80)

    # Create row with OI data
    row_data = {
        'funding_Z': 1.5,              # High positive funding ✅
        'rsi_14': 75,                  # Overbought ✅
        'liquidity_score': 0.20,       # Low liquidity ✅
        'oi_change_24h': 0.15,         # 15% OI spike ✅
        'vwap_dist': 0.01,
        'volume_Z': 1.0
    }

    context = create_test_context(row_data)
    logic = ArchetypeLogicV2Adapter("BTCUSDT", "1h")

    matched, score, meta = logic._check_S5(context)

    print(f"\nInput Features:")
    print(f"  funding_Z: {row_data['funding_Z']}")
    print(f"  rsi_14: {row_data['rsi_14']}")
    print(f"  liquidity_score: {row_data['liquidity_score']}")
    print(f"  oi_change_24h: {row_data['oi_change_24h']}")

    print(f"\nResults:")
    print(f"  Matched: {matched}")
    print(f"  Score: {score:.4f}")
    print(f"  Has OI Data: {meta.get('has_oi_data', 'N/A')}")

    if 'components' in meta:
        print(f"\nScore Components:")
        for k, v in meta['components'].items():
            print(f"  {k}: {v:.4f}")

    if 'weights' in meta:
        print(f"\nWeights:")
        for k, v in meta['weights'].items():
            print(f"  {k}: {v:.4f}")

    assert matched, "Pattern should match with all gates passing"
    assert meta['has_oi_data'], "OI data should be detected"
    assert meta['components']['oi_spike'] > 0, "OI spike should contribute to score"
    assert meta['weights']['oi_spike'] == 0.15, "OI weight should be 0.15"

    print("\n✅ Test 1 PASSED: 4-component scoring active")


def test_s5_without_oi_data():
    """Test S5 without OI data (2022-2023 scenario)"""
    print("\n" + "="*80)
    print("TEST 2: S5 without OI Data (2022-2023 Scenario)")
    print("="*80)

    # Create row WITHOUT OI data (None or NaN)
    row_data = {
        'funding_Z': 1.5,              # High positive funding ✅
        'rsi_14': 75,                  # Overbought ✅
        'liquidity_score': 0.20,       # Low liquidity ✅
        'oi_change_24h': None,         # No OI data ❌
        'vwap_dist': 0.01,
        'volume_Z': 1.0
    }

    context = create_test_context(row_data)
    logic = ArchetypeLogicV2Adapter("BTCUSDT", "1h")

    matched, score, meta = logic._check_S5(context)

    print(f"\nInput Features:")
    print(f"  funding_Z: {row_data['funding_Z']}")
    print(f"  rsi_14: {row_data['rsi_14']}")
    print(f"  liquidity_score: {row_data['liquidity_score']}")
    print(f"  oi_change_24h: {row_data['oi_change_24h']} (NO DATA)")

    print(f"\nResults:")
    print(f"  Matched: {matched}")
    print(f"  Score: {score:.4f}")
    print(f"  Has OI Data: {meta.get('has_oi_data', 'N/A')}")

    if 'components' in meta:
        print(f"\nScore Components:")
        for k, v in meta['components'].items():
            print(f"  {k}: {v:.4f}")

    if 'weights' in meta:
        print(f"\nWeights:")
        for k, v in meta['weights'].items():
            print(f"  {k}: {v:.4f}")

    assert matched, "Pattern should match even without OI data"
    assert not meta['has_oi_data'], "OI data should NOT be detected"
    assert meta['components']['oi_spike'] == 0, "OI spike should be 0"
    assert meta['weights']['oi_spike'] == 0.0, "OI weight should be 0.0"
    assert meta['weights']['funding_extreme'] == 0.50, "Funding weight should increase to 0.50"
    assert meta['weights']['rsi_exhaustion'] == 0.35, "RSI weight should increase to 0.35"

    print("\n✅ Test 2 PASSED: 3-component scoring with weight redistribution")


def test_s5_gate_failures():
    """Test S5 gate failures"""
    print("\n" + "="*80)
    print("TEST 3: S5 Gate Failures")
    print("="*80)

    # Test funding gate failure
    row_data = {
        'funding_Z': 0.5,              # LOW funding ❌
        'rsi_14': 75,
        'liquidity_score': 0.20,
        'oi_change_24h': 0.15,
        'vwap_dist': 0.01,
        'volume_Z': 1.0
    }

    context = create_test_context(row_data)
    logic = ArchetypeLogicV2Adapter("BTCUSDT", "1h")

    matched, score, meta = logic._check_S5(context)

    print(f"\nTest 3a: Funding gate failure")
    print(f"  Matched: {matched}")
    print(f"  Reason: {meta.get('reason', 'N/A')}")

    assert not matched, "Pattern should NOT match with low funding"
    assert meta['reason'] == 'funding_not_extreme', "Should fail on funding gate"

    # Test RSI gate failure
    row_data['funding_Z'] = 1.5
    row_data['rsi_14'] = 50  # LOW RSI ❌

    context = create_test_context(row_data)
    matched, score, meta = logic._check_S5(context)

    print(f"\nTest 3b: RSI gate failure")
    print(f"  Matched: {matched}")
    print(f"  Reason: {meta.get('reason', 'N/A')}")

    assert not matched, "Pattern should NOT match with low RSI"
    assert meta['reason'] == 'rsi_not_overbought', "Should fail on RSI gate"

    # Test liquidity gate failure
    row_data['rsi_14'] = 75
    row_data['liquidity_score'] = 0.50  # HIGH liquidity ❌

    context = create_test_context(row_data)
    matched, score, meta = logic._check_S5(context)

    print(f"\nTest 3c: Liquidity gate failure")
    print(f"  Matched: {matched}")
    print(f"  Reason: {meta.get('reason', 'N/A')}")

    assert not matched, "Pattern should NOT match with high liquidity"
    assert meta['reason'] == 'liquidity_not_thin', "Should fail on liquidity gate"

    print("\n✅ Test 3 PASSED: All gate failures working correctly")


def test_s5_score_comparison():
    """Compare scores with and without OI data"""
    print("\n" + "="*80)
    print("TEST 4: Score Comparison (OI vs No OI)")
    print("="*80)

    # Identical conditions except OI
    base_row = {
        'funding_Z': 1.5,
        'rsi_14': 75,
        'liquidity_score': 0.20,
        'vwap_dist': 0.01,
        'volume_Z': 1.0
    }

    # Scenario A: With OI
    row_with_oi = base_row.copy()
    row_with_oi['oi_change_24h'] = 0.15

    context_a = create_test_context(row_with_oi)
    logic = ArchetypeLogicV2Adapter("BTCUSDT", "1h")
    matched_a, score_a, meta_a = logic._check_S5(context_a)

    # Scenario B: Without OI
    row_no_oi = base_row.copy()
    row_no_oi['oi_change_24h'] = None

    context_b = create_test_context(row_no_oi)
    matched_b, score_b, meta_b = logic._check_S5(context_b)

    print(f"\nScenario A (with OI):")
    print(f"  Score: {score_a:.4f}")
    print(f"  Components: {meta_a['components']}")

    print(f"\nScenario B (no OI):")
    print(f"  Score: {score_b:.4f}")
    print(f"  Components: {meta_b['components']}")

    print(f"\nScore Difference: {score_a - score_b:.4f}")

    assert matched_a and matched_b, "Both should match"
    assert score_a > score_b, "Score with OI should be higher"
    assert 0.05 < (score_a - score_b) < 0.20, "Score difference should be moderate (5-20%)"

    print("\n✅ Test 4 PASSED: Score comparison validates graceful degradation")


if __name__ == "__main__":
    try:
        test_s5_with_oi_data()
        test_s5_without_oi_data()
        test_s5_gate_failures()
        test_s5_score_comparison()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED: S5 Graceful Degradation Working!")
        print("="*80)
        print("\nKey Findings:")
        print("  ✅ 4-component scoring works when OI available (2024)")
        print("  ✅ 3-component scoring works when OI missing (2022-2023)")
        print("  ✅ Weight redistribution preserves pattern detection")
        print("  ✅ All required gates enforce correctly")
        print("  ✅ Score differences are reasonable and expected")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
