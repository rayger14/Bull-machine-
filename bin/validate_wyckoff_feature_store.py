#!/usr/bin/env python3
"""
Validate Wyckoff Feature Store

Loads the patched feature store and performs comprehensive validation:
1. Schema validation (all expected columns present)
2. Event firing validation (events actually trigger)
3. Confidence score validation (non-zero, bounded 0-1)
4. Phase distribution analysis
5. Time-series consistency checks

Usage:
    python3 bin/validate_wyckoff_feature_store.py \\
        --input data/features_mtf/BTC_1H_wyckoff_2022-01-01_to_2024-12-31.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from typing import List, Dict


def validate_wyckoff_feature_store(input_path: str) -> Dict:
    """
    Validate Wyckoff feature store comprehensively.

    Args:
        input_path: Path to patched feature store

    Returns:
        Dict with validation results
    """
    print("=" * 80)
    print("Wyckoff Feature Store Validation")
    print("=" * 80)
    print(f"Input: {input_path}")
    print("=" * 80)

    # Load feature store
    print("\n📂 Loading feature store...")
    df = pd.read_parquet(input_path)

    print(f"   Shape: {df.shape[0]} bars × {df.shape[1]} features")
    print(f"   Date range: {df.index[0]} → {df.index[-1]}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    results = {
        'passed': [],
        'warnings': [],
        'failures': []
    }

    # Test 1: Schema validation
    print("\n" + "=" * 80)
    print("TEST 1: Schema Validation")
    print("=" * 80)

    expected_base_features = ['open', 'high', 'low', 'close', 'volume', 'volume_z']
    expected_wyckoff_events = [
        'wyckoff_sc', 'wyckoff_bc', 'wyckoff_ar', 'wyckoff_as', 'wyckoff_st',
        'wyckoff_sos', 'wyckoff_sow', 'wyckoff_lps', 'wyckoff_lpsy',
        'wyckoff_spring_a', 'wyckoff_spring_b', 'wyckoff_ut', 'wyckoff_utad'
    ]
    expected_metadata = ['wyckoff_phase_abc', 'wyckoff_sequence_position']

    # Check base features
    missing_base = [f for f in expected_base_features if f not in df.columns]
    if missing_base:
        results['failures'].append(f"Missing base features: {missing_base}")
        print(f"   ❌ FAIL: Missing base features: {missing_base}")
    else:
        results['passed'].append("Base features present")
        print(f"   ✅ PASS: All base features present")

    # Check Wyckoff events
    missing_events = [e for e in expected_wyckoff_events if e not in df.columns]
    if missing_events:
        results['failures'].append(f"Missing Wyckoff events: {missing_events}")
        print(f"   ❌ FAIL: Missing Wyckoff events: {missing_events}")
    else:
        results['passed'].append("Wyckoff event columns present")
        print(f"   ✅ PASS: All {len(expected_wyckoff_events)} Wyckoff event columns present")

    # Check confidence columns
    missing_confidence = []
    for event in expected_wyckoff_events:
        conf_col = f"{event}_confidence"
        if conf_col not in df.columns:
            missing_confidence.append(conf_col)

    if missing_confidence:
        results['failures'].append(f"Missing confidence columns: {missing_confidence}")
        print(f"   ❌ FAIL: Missing confidence columns: {missing_confidence}")
    else:
        results['passed'].append("Confidence columns present")
        print(f"   ✅ PASS: All confidence columns present")

    # Check metadata columns
    missing_metadata = [m for m in expected_metadata if m not in df.columns]
    if missing_metadata:
        results['warnings'].append(f"Missing metadata: {missing_metadata}")
        print(f"   ⚠️  WARN: Missing metadata columns: {missing_metadata}")
    else:
        results['passed'].append("Metadata columns present")
        print(f"   ✅ PASS: Metadata columns present")

    # Test 2: Event Firing Validation
    print("\n" + "=" * 80)
    print("TEST 2: Event Firing Validation")
    print("=" * 80)

    event_counts = {}
    zero_events = []

    for event in expected_wyckoff_events:
        if event not in df.columns:
            continue

        count = df[event].sum() if df[event].dtype == bool else (df[event] > 0).sum()
        event_counts[event] = count

        if count == 0:
            zero_events.append(event)

    # Report
    print(f"\n   Event Firing Summary:")
    sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)

    for event, count in sorted_events:
        pct = (count / len(df)) * 100
        status = "✅" if count > 0 else "❌"
        print(f"      {status} {event:25s}: {count:5d} ({pct:5.2f}%)")

    # Expected at least 50% of events to fire
    events_with_data = len([c for c in event_counts.values() if c > 0])
    expected_min_events = len(expected_wyckoff_events) * 0.5

    if events_with_data >= expected_min_events:
        results['passed'].append(f"Event firing: {events_with_data}/{len(expected_wyckoff_events)} events have data")
        print(f"\n   ✅ PASS: {events_with_data}/{len(expected_wyckoff_events)} events are firing")
    else:
        results['failures'].append(f"Too few events firing: {events_with_data}/{len(expected_wyckoff_events)}")
        print(f"\n   ❌ FAIL: Only {events_with_data}/{len(expected_wyckoff_events)} events firing (expected ≥{int(expected_min_events)})")

    if zero_events:
        results['warnings'].append(f"Events with zero detections: {zero_events}")
        print(f"   ⚠️  WARN: Events with zero detections: {', '.join(zero_events)}")

    # Test 3: Confidence Score Validation
    print("\n" + "=" * 80)
    print("TEST 3: Confidence Score Validation")
    print("=" * 80)

    confidence_issues = []

    for event in expected_wyckoff_events:
        conf_col = f"{event}_confidence"
        if conf_col not in df.columns:
            continue

        conf_series = df[conf_col]

        # Check bounds [0, 1]
        if conf_series.min() < 0 or conf_series.max() > 1:
            confidence_issues.append(f"{conf_col} out of bounds [{conf_series.min():.3f}, {conf_series.max():.3f}]")

        # Check variance (should not be constant)
        non_zero = conf_series[conf_series > 0]
        if len(non_zero) > 0:
            std = non_zero.std()
            if std < 0.001:
                confidence_issues.append(f"{conf_col} has very low variance (std={std:.6f})")

    if confidence_issues:
        results['failures'].append(f"Confidence score issues: {confidence_issues}")
        print(f"   ❌ FAIL: Confidence score issues:")
        for issue in confidence_issues:
            print(f"      - {issue}")
    else:
        results['passed'].append("Confidence scores valid")
        print(f"   ✅ PASS: All confidence scores in [0, 1] with reasonable variance")

    # Show confidence statistics for top 5 events
    print(f"\n   Top 5 Events - Confidence Statistics:")
    top_5_events = sorted_events[:5]

    for event, count in top_5_events:
        conf_col = f"{event}_confidence"
        if conf_col in df.columns:
            conf = df[conf_col]
            non_zero_conf = conf[conf > 0]
            if len(non_zero_conf) > 0:
                print(f"      {event:25s}: mean={non_zero_conf.mean():.3f}, "
                      f"std={non_zero_conf.std():.3f}, "
                      f"max={non_zero_conf.max():.3f}")

    # Test 4: Phase Distribution Analysis
    print("\n" + "=" * 80)
    print("TEST 4: Phase Distribution Analysis")
    print("=" * 80)

    if 'wyckoff_phase_abc' in df.columns:
        phase_counts = df['wyckoff_phase_abc'].value_counts()

        print(f"\n   Phase Distribution:")
        for phase, count in phase_counts.items():
            pct = (count / len(df)) * 100
            print(f"      Phase {phase:8s}: {count:6d} bars ({pct:5.1f}%)")

        # Check for excessive neutral phase
        neutral_pct = (phase_counts.get('neutral', 0) / len(df)) * 100
        if neutral_pct > 50:
            results['warnings'].append(f"High neutral phase percentage: {neutral_pct:.1f}%")
            print(f"\n   ⚠️  WARN: High neutral phase percentage: {neutral_pct:.1f}%")
        else:
            results['passed'].append(f"Phase distribution reasonable (neutral: {neutral_pct:.1f}%)")
            print(f"\n   ✅ PASS: Phase distribution reasonable")

    else:
        results['failures'].append("wyckoff_phase_abc column missing")
        print(f"   ❌ FAIL: wyckoff_phase_abc column missing")

    # Test 5: Time Series Consistency
    print("\n" + "=" * 80)
    print("TEST 5: Time Series Consistency")
    print("=" * 80)

    # Check for NaN values
    wyckoff_cols = [c for c in df.columns if 'wyckoff' in c.lower()]
    nan_counts = df[wyckoff_cols].isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]

    if len(cols_with_nans) > 0:
        results['warnings'].append(f"NaN values in {len(cols_with_nans)} Wyckoff columns")
        print(f"   ⚠️  WARN: Found NaN values in {len(cols_with_nans)} columns:")
        for col, count in cols_with_nans.items():
            pct = (count / len(df)) * 100
            print(f"      - {col}: {count} NaNs ({pct:.2f}%)")
    else:
        results['passed'].append("No NaN values in Wyckoff columns")
        print(f"   ✅ PASS: No NaN values in Wyckoff columns")

    # Check index monotonicity
    if df.index.is_monotonic_increasing:
        results['passed'].append("Index is monotonically increasing")
        print(f"   ✅ PASS: Index is monotonically increasing")
    else:
        results['failures'].append("Index is not monotonically increasing")
        print(f"   ❌ FAIL: Index is not monotonically increasing")

    # Final Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\n✅ PASSED: {len(results['passed'])}")
    for p in results['passed']:
        print(f"   - {p}")

    print(f"\n⚠️  WARNINGS: {len(results['warnings'])}")
    for w in results['warnings']:
        print(f"   - {w}")

    print(f"\n❌ FAILURES: {len(results['failures'])}")
    for f in results['failures']:
        print(f"   - {f}")

    # Overall verdict
    print("\n" + "=" * 80)
    if len(results['failures']) == 0:
        print("🎉 OVERALL: VALIDATION PASSED")
        print("Feature store is ready for production use!")
    elif len(results['failures']) <= 2:
        print("⚠️  OVERALL: VALIDATION PASSED WITH WARNINGS")
        print("Feature store is usable but has minor issues.")
    else:
        print("❌ OVERALL: VALIDATION FAILED")
        print("Feature store has critical issues that must be fixed.")

    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate Wyckoff feature store'
    )
    parser.add_argument('--input', required=True, help='Input feature store path')

    args = parser.parse_args()

    # Validate input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Run validation
    results = validate_wyckoff_feature_store(args.input)

    # Exit with appropriate code
    if len(results['failures']) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
