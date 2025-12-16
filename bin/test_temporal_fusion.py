#!/usr/bin/env python3
"""
Test Temporal Fusion Engine Integration

Validates that temporal timing features work correctly
and can be used for confluence-based signal generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def test_feature_presence():
    """Test 1: Verify all temporal features exist."""
    print("TEST 1: Feature Presence Check")
    print("-" * 70)

    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    expected_features = [
        # Wyckoff timing
        'bars_since_sc', 'bars_since_ar', 'bars_since_st',
        'bars_since_sos_long', 'bars_since_sos_short', 'bars_since_spring',
        'bars_since_utad', 'bars_since_ps', 'bars_since_bc',
        # Fibonacci
        'fib_time_cluster', 'fib_time_score', 'fib_time_target',
        # Cycles
        'gann_cycle', 'volatility_cycle'
    ]

    missing = []
    for feat in expected_features:
        if feat not in df.columns:
            missing.append(feat)
            print(f"  ❌ {feat} - MISSING")
        else:
            print(f"  ✅ {feat}")

    if not missing:
        print("\n✅ TEST PASSED: All 14 features present")
        return True
    else:
        print(f"\n❌ TEST FAILED: {len(missing)} features missing")
        return False


def test_data_quality():
    """Test 2: Verify data quality and distributions."""
    print("\n\nTEST 2: Data Quality Check")
    print("-" * 70)

    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    checks_passed = True

    # Check bars_since_* features
    print("\nWyckoff Timing Features:")
    bars_since_cols = [c for c in df.columns if c.startswith('bars_since_')]
    for col in bars_since_cols:
        non_null_pct = (df[col].notna().sum() / len(df)) * 100
        mean_val = df[col].mean()

        # Check for reasonable values
        if non_null_pct < 50:
            print(f"  ⚠️  {col}: Low coverage ({non_null_pct:.1f}%)")
        elif pd.isna(mean_val) or mean_val < 0:
            print(f"  ❌ {col}: Invalid mean ({mean_val})")
            checks_passed = False
        else:
            print(f"  ✅ {col}: {non_null_pct:.1f}% coverage, mean={mean_val:.1f}")

    # Check Fibonacci features
    print("\nFibonacci Features:")
    cluster_count = df['fib_time_cluster'].sum()
    cluster_pct = (cluster_count / len(df)) * 100

    if cluster_count == 0:
        print(f"  ❌ fib_time_cluster: No events found")
        checks_passed = False
    else:
        print(f"  ✅ fib_time_cluster: {cluster_count:,} events ({cluster_pct:.1f}%)")

    score_mean = df['fib_time_score'].mean()
    score_max = df['fib_time_score'].max()

    if score_max < 0.9 or score_max > 1.0:
        print(f"  ❌ fib_time_score: Invalid max ({score_max})")
        checks_passed = False
    else:
        print(f"  ✅ fib_time_score: Mean={score_mean:.3f}, Max={score_max:.3f}")

    target_count = df['fib_time_target'].notna().sum()
    if target_count == 0:
        print(f"  ❌ fib_time_target: No values set")
        checks_passed = False
    else:
        print(f"  ✅ fib_time_target: {target_count:,} non-null values")

    # Check cycle features
    print("\nCycle Features:")
    gann_count = df['gann_cycle'].sum()
    if gann_count == 0:
        print(f"  ⚠️  gann_cycle: No events found")
    else:
        print(f"  ✅ gann_cycle: {gann_count:,} events")

    vol_mean = df['volatility_cycle'].mean()
    vol_std = df['volatility_cycle'].std()
    if pd.isna(vol_mean) or vol_mean < 0 or vol_mean > 1:
        print(f"  ❌ volatility_cycle: Invalid mean ({vol_mean})")
        checks_passed = False
    else:
        print(f"  ✅ volatility_cycle: Mean={vol_mean:.3f}, Std={vol_std:.3f}")

    if checks_passed:
        print("\n✅ TEST PASSED: Data quality checks passed")
    else:
        print("\n❌ TEST FAILED: Data quality issues detected")

    return checks_passed


def test_confluence_detection():
    """Test 3: Verify confluence detection logic."""
    print("\n\nTEST 3: Confluence Detection Logic")
    print("-" * 70)

    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    # Find high-quality confluences
    high_quality = df[
        (df['fib_time_cluster'] == True) &
        (df['fib_time_score'] >= 0.667)
    ]

    print(f"\nHigh-quality confluence events: {len(high_quality):,}")

    if len(high_quality) == 0:
        print("❌ TEST FAILED: No high-quality confluences found")
        return False

    # Verify that these events actually have aligned Fibonacci levels
    bars_since_cols = [c for c in df.columns if c.startswith('bars_since_')]
    FIB_LEVELS = [13, 21, 34, 55, 89, 144]

    verification_passed = True
    sample_size = min(10, len(high_quality))

    print(f"\nVerifying {sample_size} sample events:")

    for i, (idx, row) in enumerate(high_quality.head(sample_size).iterrows(), 1):
        # Count actual Fibonacci alignments
        actual_matches = 0
        aligned_events = []

        for col in bars_since_cols:
            val = row[col]
            if pd.notna(val):
                for fib in FIB_LEVELS:
                    if abs(val - fib) <= 1:
                        actual_matches += 1
                        event_name = col.replace('bars_since_', '').upper()
                        aligned_events.append(f"{event_name}={int(val)}")
                        break

        expected_score = min(actual_matches / 3.0, 1.0)
        actual_score = row['fib_time_score']

        if abs(expected_score - actual_score) > 0.01:
            print(f"  ❌ Event {i} ({idx}): Score mismatch")
            print(f"     Expected: {expected_score:.3f}, Got: {actual_score:.3f}")
            verification_passed = False
        else:
            print(f"  ✅ Event {i} ({idx}): Score={actual_score:.3f}, Aligned={len(aligned_events)}")
            print(f"     Events: {' | '.join(aligned_events[:3])}")

    if verification_passed:
        print("\n✅ TEST PASSED: Confluence detection logic correct")
    else:
        print("\n❌ TEST FAILED: Confluence scoring issues detected")

    return verification_passed


def test_temporal_signal_generation():
    """Test 4: Generate temporal confluence signals."""
    print("\n\nTEST 4: Temporal Signal Generation")
    print("-" * 70)

    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    # Define signal generation logic
    def generate_temporal_signal(row):
        """
        Generate temporal confluence signal.

        Returns:
            dict with signal strength and reason
        """
        if not row['fib_time_cluster']:
            return {'signal': 0.0, 'reason': 'No Fib confluence'}

        score = row['fib_time_score']

        # Base signal from Fib score
        signal_strength = score

        # Boost for Gann cycle alignment
        if row['gann_cycle']:
            signal_strength *= 1.2
            signal_strength = min(signal_strength, 1.0)

        # Adjust for volatility cycle
        if row['volatility_cycle'] < 0.5:
            # Low vol = better breakout potential
            signal_strength *= 1.1
            signal_strength = min(signal_strength, 1.0)

        # Determine quality
        if signal_strength >= 0.8:
            quality = 'EXCELLENT'
        elif signal_strength >= 0.6:
            quality = 'GOOD'
        elif signal_strength >= 0.4:
            quality = 'FAIR'
        else:
            quality = 'WEAK'

        return {
            'signal': signal_strength,
            'quality': quality,
            'fib_score': score,
            'gann': row['gann_cycle'],
            'vol': row['volatility_cycle']
        }

    # Generate signals
    print("\nGenerating temporal confluence signals...")

    signals = []
    for idx, row in df.iterrows():
        sig = generate_temporal_signal(row)
        if sig['signal'] >= 0.6:  # Only track good+ signals
            signals.append({
                'timestamp': idx,
                'price': row['close'],
                **sig
            })

    signals_df = pd.DataFrame(signals)

    print(f"Total signals generated: {len(signals_df):,}")

    if len(signals_df) == 0:
        print("❌ TEST FAILED: No signals generated")
        return False

    # Show signal distribution
    print("\nSignal Quality Distribution:")
    quality_dist = signals_df['quality'].value_counts()
    for quality, count in quality_dist.items():
        pct = (count / len(signals_df)) * 100
        print(f"  {quality}: {count:,} ({pct:.1f}%)")

    # Show top signals
    print("\nTop 5 Temporal Confluence Signals:")
    top_signals = signals_df.nlargest(5, 'signal')

    for i, row in enumerate(top_signals.iterrows(), 1):
        _, sig = row
        print(f"\n  {i}. {sig['timestamp']}")
        print(f"     Price: ${sig['price']:,.2f}")
        print(f"     Signal Strength: {sig['signal']:.3f} ({sig['quality']})")
        print(f"     Fib Score: {sig['fib_score']:.3f}")
        print(f"     Gann Cycle: {'Yes' if sig['gann'] else 'No'}")
        print(f"     Vol Cycle: {sig['vol']:.3f}")

    print("\n✅ TEST PASSED: Temporal signal generation working")
    return True


def test_integration_readiness():
    """Test 5: Check integration readiness."""
    print("\n\nTEST 5: Integration Readiness Check")
    print("-" * 70)

    # Check that feature store is up to date
    feature_store_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    if not feature_store_path.exists():
        print("❌ TEST FAILED: Feature store not found")
        return False

    df = pd.read_parquet(feature_store_path)

    print("\nFeature Store Status:")
    print(f"  Path: {feature_store_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date Range: {df.index.min()} to {df.index.max()}")

    # Check for recent data
    latest_date = df.index.max()
    days_old = (pd.Timestamp.now(tz='UTC') - latest_date).days

    if days_old > 30:
        print(f"  ⚠️  Data is {days_old} days old - consider updating")
    else:
        print(f"  ✅ Data is recent ({days_old} days old)")

    # Check column count
    if len(df.columns) < 200:
        print(f"  ❌ Expected 200+ columns, found {len(df.columns)}")
        return False
    else:
        print(f"  ✅ Column count correct (200+)")

    # Check for all temporal features
    temporal_features = [c for c in df.columns if 'bars_since' in c or 'fib_time' in c or 'cycle' in c]
    if len(temporal_features) < 14:
        print(f"  ❌ Expected 14 temporal features, found {len(temporal_features)}")
        return False
    else:
        print(f"  ✅ All 14 temporal features present")

    print("\n✅ TEST PASSED: System ready for temporal fusion integration")
    return True


def run_all_tests():
    """Run complete test suite."""
    print("=" * 70)
    print("TEMPORAL FUSION ENGINE - TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        ("Feature Presence", test_feature_presence),
        ("Data Quality", test_data_quality),
        ("Confluence Detection", test_confluence_detection),
        ("Signal Generation", test_temporal_signal_generation),
        ("Integration Readiness", test_integration_readiness),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {test_name}")
            print(f"   Error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")

    print()
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print()

    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED - TEMPORAL FUSION ENGINE READY")
        print()
        print("Next Steps:")
        print("1. Integrate temporal features into signal generation")
        print("2. Run backtest with temporal filters enabled")
        print("3. Measure impact on win rate and Sharpe ratio")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW ISSUES ABOVE")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
