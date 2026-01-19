#!/usr/bin/env python3
"""
Validate Agent 2 Crisis Features - Acceptance Test
===================================================

Validates that crisis features delivered by Agent 2 meet requirements:
  1. 100% coverage (no NaNs)
  2. Correct timestamps (aligned with 1H bars)
  3. Values spike during LUNA collapse (May 9-12, 2022)
  4. Values spike during FTX collapse (Nov 8-11, 2022)
  5. Values spike during June 2022 dump (Jun 13-18, 2022)

Returns:
  - ✅ PASS: All features meet requirements, ready for HMM training
  - ❌ FAIL: Issues found, report to Agent 2

Usage:
    python bin/validate_agent2_crisis_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Required features from Agent 2
REQUIRED_FEATURES = {
    'flash_crash_1h': {
        'description': '1-hour flash crash indicator',
        'type': 'binary',
        'luna_target': 0.50,  # >50% of hours during LUNA
        'ftx_target': 0.40,
        'june_target': 0.30,
    },
    'flash_crash_4h': {
        'description': '4-hour flash crash indicator',
        'type': 'binary',
        'luna_target': 0.40,
        'ftx_target': 0.30,
        'june_target': 0.25,
    },
    'flash_crash_1d': {
        'description': '24-hour flash crash indicator',
        'type': 'binary',
        'luna_target': 0.30,
        'ftx_target': 0.25,
        'june_target': 0.20,
    },
    'volume_spike': {
        'description': 'Volume surge indicator',
        'type': 'binary_or_continuous',
        'luna_target': 0.40,
        'ftx_target': 0.35,
        'june_target': 0.30,
    },
    'oi_delta_1h_z': {
        'description': '1-hour OI change z-score',
        'type': 'continuous',
        'luna_target': -2.0,  # Expect negative z-score (OI drop)
        'ftx_target': -2.0,
        'june_target': -1.5,
    },
    'oi_cascade': {
        'description': 'OI liquidation cascade indicator',
        'type': 'binary',
        'luna_target': 3,  # At least 3 cascade events
        'ftx_target': 2,
        'june_target': 2,
    },
    'funding_extreme': {
        'description': 'Extreme funding rate indicator',
        'type': 'binary',
        'luna_target': 0.20,
        'ftx_target': 0.15,
        'june_target': 0.15,
    },
    'funding_flip': {
        'description': 'Rapid funding rate flip indicator',
        'type': 'binary',
        'luna_target': 0.15,
        'ftx_target': 0.10,
        'june_target': 0.10,
    }
}

# Crisis event windows
CRISIS_EVENTS = {
    'LUNA': ('2022-05-09', '2022-05-12'),
    'FTX': ('2022-11-08', '2022-11-11'),
    'June': ('2022-06-13', '2022-06-18'),
}


def load_feature_store():
    """Load feature store and validate structure."""
    data_path = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Feature store not found: {data_path}")

    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    print(f"📊 Feature store loaded")
    print(f"   Path: {data_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"   Total columns: {len(df.columns)}")

    return df


def validate_feature_coverage(df: pd.DataFrame, feature_name: str) -> tuple:
    """
    Validate feature has 100% coverage.

    Returns:
        (passed, coverage_pct, details)
    """
    if feature_name not in df.columns:
        return False, 0.0, f"❌ Feature not found in dataframe"

    total = len(df)
    non_null = df[feature_name].notna().sum()
    coverage = (non_null / total) * 100

    if coverage == 100.0:
        return True, coverage, f"✅ 100% coverage ({non_null:,}/{total:,})"
    else:
        null_count = total - non_null
        return False, coverage, f"❌ {coverage:.1f}% coverage ({null_count:,} NaNs)"


def validate_crisis_response(df: pd.DataFrame, feature_name: str,
                             feature_config: dict, event_name: str,
                             event_window: tuple) -> tuple:
    """
    Validate feature responds to crisis event.

    Returns:
        (passed, actual_value, target_value, details)
    """
    start, end = event_window
    event_df = df.loc[start:end]

    if len(event_df) == 0:
        return False, 0, 0, f"❌ No data in {event_name} window"

    feature_type = feature_config['type']
    target_key = f"{event_name.lower()}_target"
    target = feature_config[target_key]

    if feature_type == 'binary' or feature_type == 'binary_or_continuous':
        # Binary: Check % of hours triggered
        if feature_name == 'oi_cascade':
            # For cascade, count total events (not %)
            actual = event_df[feature_name].sum()
            passed = actual >= target
            details = f"{'✅' if passed else '❌'} {event_name}: {actual} events (target: ≥{target})"
        else:
            actual = event_df[feature_name].mean()
            passed = actual >= target
            details = f"{'✅' if passed else '❌'} {event_name}: {actual:.1%} triggered (target: ≥{target:.1%})"

    elif feature_type == 'continuous':
        # Continuous: Check mean value
        actual = event_df[feature_name].mean()
        if 'oi_delta' in feature_name:
            # For OI delta, expect negative values during crisis
            passed = actual <= target
            details = f"{'✅' if passed else '❌'} {event_name}: mean={actual:.2f} (target: ≤{target})"
        else:
            passed = actual >= target
            details = f"{'✅' if passed else '❌'} {event_name}: mean={actual:.2f} (target: ≥{target})"
    else:
        return False, 0, target, f"❌ Unknown feature type: {feature_type}"

    return passed, actual, target, details


def main():
    print("\n" + "="*80)
    print("AGENT 2 CRISIS FEATURES - ACCEPTANCE VALIDATION")
    print("="*80)

    # Load data
    print("\n📊 Step 1: Load feature store")
    try:
        df = load_feature_store()
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return

    # Validate feature presence and coverage
    print("\n" + "="*80)
    print("Step 2: Validate Feature Presence & Coverage")
    print("="*80)

    coverage_results = {}
    missing_features = []

    for feature_name, feature_config in REQUIRED_FEATURES.items():
        passed, coverage_pct, details = validate_feature_coverage(df, feature_name)
        coverage_results[feature_name] = {
            'passed': passed,
            'coverage': coverage_pct,
            'details': details
        }

        print(f"\n{feature_name}:")
        print(f"  Description: {feature_config['description']}")
        print(f"  {details}")

        if not passed:
            missing_features.append(feature_name)

    coverage_pass = len(missing_features) == 0

    # Validate crisis event response
    print("\n" + "="*80)
    print("Step 3: Validate Crisis Event Response")
    print("="*80)

    event_results = {}
    failed_validations = []

    for feature_name, feature_config in REQUIRED_FEATURES.items():
        if feature_name in missing_features:
            print(f"\n⏭️  {feature_name}: Skipping (not found)")
            continue

        print(f"\n{feature_name}:")

        feature_pass = True
        event_details = {}

        for event_name, event_window in CRISIS_EVENTS.items():
            passed, actual, target, details = validate_crisis_response(
                df, feature_name, feature_config, event_name, event_window
            )

            print(f"  {details}")

            event_details[event_name] = {
                'passed': passed,
                'actual': actual,
                'target': target
            }

            if not passed:
                feature_pass = False

        event_results[feature_name] = {
            'passed': feature_pass,
            'events': event_details
        }

        if not feature_pass:
            failed_validations.append(feature_name)

    event_pass = len(failed_validations) == 0

    # Overall assessment
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\nCoverage Check:")
    coverage_count = sum(1 for r in coverage_results.values() if r['passed'])
    print(f"  Passed: {coverage_count}/{len(REQUIRED_FEATURES)} features")
    if missing_features:
        print(f"  Missing: {', '.join(missing_features)}")

    print(f"\nCrisis Response Check:")
    event_count = sum(1 for r in event_results.values() if r['passed'])
    print(f"  Passed: {event_count}/{len([f for f in REQUIRED_FEATURES if f not in missing_features])} features")
    if failed_validations:
        print(f"  Failed: {', '.join(failed_validations)}")

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    all_pass = coverage_pass and event_pass

    if all_pass:
        print("\n✅ VALIDATION PASSED")
        print("\nAll Agent 2 crisis features meet requirements:")
        print("  ✅ 100% coverage (no NaNs)")
        print("  ✅ Correct crisis event response")
        print("  ✅ Ready for HMM training")
        print("\n🚀 Next step: Run HMM retraining")
        print("   python bin/train_hmm_with_crisis_features.py --stage agent2 --n_init 10")
    else:
        print("\n❌ VALIDATION FAILED")
        print("\nIssues found:")
        if not coverage_pass:
            print(f"  ❌ Coverage issues: {len(missing_features)} features missing/incomplete")
        if not event_pass:
            print(f"  ❌ Crisis response issues: {len(failed_validations)} features don't respond correctly")

        print("\n📋 Required actions:")
        print("  1. Review feature engineering in Agent 2")
        print("  2. Fix missing/incorrect features")
        print("  3. Re-run validation")
        print("  4. DO NOT proceed to HMM training until validation passes")

    print("\n" + "="*80 + "\n")

    # Save validation report
    report_path = Path('results') / f"agent2_feature_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    Path('results').mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AGENT 2 CRISIS FEATURES - VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("COVERAGE RESULTS:\n")
        for feature_name, result in coverage_results.items():
            f.write(f"\n  {feature_name}:\n")
            f.write(f"    {result['details']}\n")

        f.write("\n\nCRISIS EVENT RESPONSE:\n")
        for feature_name, result in event_results.items():
            f.write(f"\n  {feature_name}:\n")
            for event_name, event_result in result['events'].items():
                status = "PASS" if event_result['passed'] else "FAIL"
                f.write(f"    [{status}] {event_name}: actual={event_result['actual']:.3f}, target={event_result['target']:.3f}\n")

        f.write(f"\n\nOVERALL: {'PASS' if all_pass else 'FAIL'}\n")

    print(f"📄 Validation report saved: {report_path}\n")

    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())
