#!/usr/bin/env python3
"""
Test OI/Funding Graceful Degradation

Validates that bear archetypes handle missing OI and funding data gracefully.

Usage:
    python3 bin/test_oi_degradation.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.strategies.archetypes.bear.feature_fallback import (
    FeatureFallbackManager,
    safe_get_oi_spike,
    safe_get_oi_change,
    safe_get_funding_z,
    enrich_with_all_fallbacks,
)
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import apply_liquidity_vacuum_enrichment
from engine.strategies.archetypes.bear.funding_divergence_runtime import apply_s4_enrichment
from engine.strategies.archetypes.bear.long_squeeze_runtime import apply_s5_enrichment
from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment


def create_minimal_df(n_bars=100):
    """Create minimal OHLCV DataFrame (no OI, no funding)."""
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='1H')

    df = pd.DataFrame({
        'open': np.random.randn(n_bars).cumsum() + 20000,
        'high': np.random.randn(n_bars).cumsum() + 20100,
        'low': np.random.randn(n_bars).cumsum() + 19900,
        'close': np.random.randn(n_bars).cumsum() + 20000,
        'volume': np.random.rand(n_bars) * 1000 + 500,
    }, index=dates)

    # Ensure OHLC relationships
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df


def create_partial_df(n_bars=100):
    """Create DataFrame with partial features (volume but no OI/funding)."""
    df = create_minimal_df(n_bars)

    # Add volume-based features
    df['volume_zscore'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    df['volume_panic'] = np.where(df['volume_zscore'] > 2.0, 1.0, 0.0)

    # Add liquidity score
    df['liquidity_score'] = np.random.rand(n_bars) * 0.5 + 0.2

    # Add RSI
    df['rsi_14'] = 50 + np.random.randn(n_bars) * 15

    return df


def create_full_df(n_bars=100):
    """Create DataFrame with full features (including OI and funding)."""
    df = create_partial_df(n_bars)

    # Add OI features
    df['oi_change_spike_24h'] = np.random.rand(n_bars) * 0.5
    df['oi_change_24h'] = np.random.randn(n_bars) * 0.1
    df['oi_z'] = np.random.randn(n_bars)

    # Add funding features
    df['funding_rate'] = np.random.randn(n_bars) * 0.01
    df['funding_Z'] = (df['funding_rate'] - df['funding_rate'].mean()) / df['funding_rate'].std()

    return df


def test_fallback_manager():
    """Test FeatureFallbackManager basic functionality."""
    print("="*80)
    print("TEST 1: FeatureFallbackManager Basic Functionality")
    print("="*80)

    manager = FeatureFallbackManager(log_fallbacks=True)

    # Test with minimal data
    row = pd.Series({
        'close': 20000,
        'volume': 1000,
        'volume_zscore': 1.5,
    })

    # Should fallback to volume_zscore
    oi_spike = manager.safe_get(
        row,
        'oi_change_spike_24h',
        ['volume_panic', 'volume_zscore'],
        default=0.0
    )

    print(f"\nResult: oi_spike = {oi_spike:.4f} (should be 1.5 from volume_zscore)")

    # Should use default
    funding = manager.safe_get(
        row,
        'funding_Z',
        ['funding_rate'],
        default=0.0
    )

    print(f"Result: funding_Z = {funding:.4f} (should be 0.0 default)")

    # Check stats
    print("\nFallback usage statistics:")
    stats = manager.get_stats()
    for key, count in stats.items():
        print(f"  {key}: {count} uses")

    assert oi_spike == 1.5, "Should fallback to volume_zscore"
    assert funding == 0.0, "Should use default"

    print("\n✅ TEST 1 PASSED")
    return True


def test_batch_enrichment_minimal():
    """Test batch enrichment with minimal OHLCV data."""
    print("\n" + "="*80)
    print("TEST 2: Batch Enrichment - Minimal Data (OHLCV only)")
    print("="*80)

    df = create_minimal_df(100)

    print(f"\nBefore enrichment: {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Apply enrichment
    df_enriched = enrich_with_all_fallbacks(df)

    print(f"\nAfter enrichment: {len(df_enriched.columns)} columns")

    # Check that fallback columns were created
    expected_fallbacks = ['oi_change_spike_24h', 'oi_z', 'funding_Z']

    print("\nFallback features created:")
    for feat in expected_fallbacks:
        present = "✅" if feat in df_enriched.columns else "❌"
        print(f"  {present} {feat}")

    # Verify no crashes
    assert all(feat in df_enriched.columns for feat in expected_fallbacks), \
        "All fallback features should be created"

    print("\n✅ TEST 2 PASSED")
    return True


def test_batch_enrichment_partial():
    """Test batch enrichment with partial features."""
    print("\n" + "="*80)
    print("TEST 3: Batch Enrichment - Partial Data (Volume but no OI)")
    print("="*80)

    df = create_partial_df(100)

    print(f"\nBefore enrichment: {len(df.columns)} columns")
    print(f"Has volume_panic: {'volume_panic' in df.columns}")
    print(f"Has funding_Z: {'funding_Z' in df.columns}")

    # Apply enrichment
    df_enriched = enrich_with_all_fallbacks(df)

    print(f"\nAfter enrichment: {len(df_enriched.columns)} columns")
    print(f"Has oi_change_spike_24h: {'oi_change_spike_24h' in df_enriched.columns}")
    print(f"Has funding_Z: {'funding_Z' in df_enriched.columns}")

    # Check sample values
    sample = df_enriched.iloc[50]
    print("\nSample values (row 50):")
    print(f"  volume_panic: {sample.get('volume_panic', 'N/A'):.4f}")
    print(f"  oi_change_spike_24h (fallback): {sample['oi_change_spike_24h']:.4f}")
    print(f"  funding_Z (created): {sample['funding_Z']:.4f}")

    assert 'oi_change_spike_24h' in df_enriched.columns
    assert 'funding_Z' in df_enriched.columns

    print("\n✅ TEST 3 PASSED")
    return True


def test_batch_enrichment_full():
    """Test batch enrichment with full features (should not modify)."""
    print("\n" + "="*80)
    print("TEST 4: Batch Enrichment - Full Data (No Fallbacks Needed)")
    print("="*80)

    df = create_full_df(100)

    print(f"\nBefore enrichment: {len(df.columns)} columns")
    print(f"Has oi_change_spike_24h: {'oi_change_spike_24h' in df.columns}")
    print(f"Has funding_Z: {'funding_Z' in df.columns}")

    # Store original values
    orig_oi = df['oi_change_spike_24h'].copy()
    orig_funding = df['funding_Z'].copy()

    # Apply enrichment
    df_enriched = enrich_with_all_fallbacks(df)

    print(f"\nAfter enrichment: {len(df_enriched.columns)} columns")

    # Verify values unchanged
    oi_unchanged = (df_enriched['oi_change_spike_24h'] == orig_oi).all()
    funding_unchanged = (df_enriched['funding_Z'] == orig_funding).all()

    print(f"\nOriginal values preserved:")
    print(f"  oi_change_spike_24h: {oi_unchanged}")
    print(f"  funding_Z: {funding_unchanged}")

    assert oi_unchanged, "Should not modify existing OI features"
    assert funding_unchanged, "Should not modify existing funding features"

    print("\n✅ TEST 4 PASSED")
    return True


def test_archetype_runtime_enrichment():
    """Test that archetype runtime enrichment works with degraded data."""
    print("\n" + "="*80)
    print("TEST 5: Archetype Runtime Enrichment (Degraded Mode)")
    print("="*80)

    df = create_partial_df(100)

    # Test S1 (Liquidity Vacuum)
    print("\n[S1] Testing Liquidity Vacuum enrichment...")
    try:
        df_s1 = apply_liquidity_vacuum_enrichment(df.copy())
        s1_features = [
            'wick_lower_ratio', 'liquidity_vacuum_score', 'volume_panic',
            'liquidity_vacuum_fusion'
        ]
        s1_ok = all(f in df_s1.columns for f in s1_features)
        print(f"  Result: {'✅ PASS' if s1_ok else '❌ FAIL'}")
    except Exception as e:
        print(f"  Result: ❌ FAIL - {e}")
        s1_ok = False

    # Test S4 (Funding Divergence)
    print("\n[S4] Testing Funding Divergence enrichment...")
    try:
        df_s4 = apply_s4_enrichment(df.copy())
        s4_features = [
            'funding_z_negative', 'price_resilience', 'volume_quiet',
            's4_fusion_score'
        ]
        s4_ok = all(f in df_s4.columns for f in s4_features)
        print(f"  Result: {'✅ PASS' if s4_ok else '❌ FAIL'}")
    except Exception as e:
        print(f"  Result: ❌ FAIL - {e}")
        s4_ok = False

    # Test S5 (Long Squeeze)
    print("\n[S5] Testing Long Squeeze enrichment...")
    try:
        df_s5 = apply_s5_enrichment(df.copy())
        s5_features = [
            'funding_z_score', 'oi_change', 'rsi_overbought',
            's5_fusion_score'
        ]
        s5_ok = all(f in df_s5.columns for f in s5_features)
        print(f"  Result: {'✅ PASS' if s5_ok else '❌ FAIL'}")
    except Exception as e:
        print(f"  Result: ❌ FAIL - {e}")
        s5_ok = False

    # Test S2 (Failed Rally)
    print("\n[S2] Testing Failed Rally enrichment...")
    try:
        df_s2 = apply_runtime_enrichment(df.copy())
        s2_features = [
            'wick_upper_ratio', 'volume_fade_flag', 'rsi_bearish_div'
        ]
        s2_ok = all(f in df_s2.columns for f in s2_features)
        print(f"  Result: {'✅ PASS' if s2_ok else '❌ FAIL'}")
    except Exception as e:
        print(f"  Result: ❌ FAIL - {e}")
        s2_ok = False

    all_ok = s1_ok and s4_ok and s5_ok and s2_ok

    if all_ok:
        print("\n✅ TEST 5 PASSED - All archetypes enriched successfully in degraded mode")
    else:
        print("\n❌ TEST 5 FAILED - Some archetypes failed enrichment")

    return all_ok


def main():
    """Run all tests."""
    print("OI/FUNDING GRACEFUL DEGRADATION TEST SUITE")
    print("=" * 80)

    tests = [
        ("Fallback Manager", test_fallback_manager),
        ("Minimal Data Enrichment", test_batch_enrichment_minimal),
        ("Partial Data Enrichment", test_batch_enrichment_partial),
        ("Full Data Enrichment", test_batch_enrichment_full),
        ("Archetype Runtime Enrichment", test_archetype_runtime_enrichment),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "="*80)
    print(f"FINAL RESULT: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Graceful degradation working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - Review logs above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
