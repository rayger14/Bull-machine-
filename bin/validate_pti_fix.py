#!/usr/bin/env python3
"""
Validate PTI Fix - Verify PTI features are properly calculated

This script validates that PTI features are no longer frozen at constant values
after the fix has been applied to the feature store build script.

Expected outcomes:
- tf1h_pti_score: Should have 1000+ unique values, NOT frozen at 0.5
- tf1h_pti_confidence: Should have 1000+ unique values, NOT frozen at 0.5
- tf1h_pti_trap_type: Should have multiple trap types (none, bull_trap, bear_trap, spring, utad)
- tf1d_pti_score: Should have 100+ unique values, NOT frozen at 0.5
- tf1d_pti_reversal: Should have True/False values, NOT all False

Reference: PTI_SCORES_FIX.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import argparse


def validate_pti_features(feature_store_path: str) -> bool:
    """
    Validate PTI features in the feature store.

    Args:
        feature_store_path: Path to the feature store parquet file

    Returns:
        True if all PTI features are valid, False otherwise
    """
    print(f"\n{'='*80}")
    print("PTI FIX VALIDATION")
    print(f"{'='*80}\n")
    print(f"Feature store: {feature_store_path}\n")

    # Load feature store
    try:
        df = pd.read_parquet(feature_store_path)
        print(f"✅ Loaded feature store: {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"❌ FAIL: Feature store not found at {feature_store_path}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error loading feature store: {e}")
        return False

    # Define PTI columns to validate
    pti_cols = {
        'tf1h_pti_score': {
            'type': 'float',
            'min_unique': 100,  # Should have many unique values
            'expected_range': (0.0, 1.0),
            'frozen_value': 0.5,  # Known broken value
        },
        'tf1h_pti_confidence': {
            'type': 'float',
            'min_unique': 100,
            'expected_range': (0.0, 1.0),
            'frozen_value': 0.5,
        },
        'tf1h_pti_trap_type': {
            'type': 'string',
            'min_unique': 2,  # At least 'none' and one trap type
            'expected_values': ['none', 'bull_trap', 'bear_trap', 'spring', 'utad'],
        },
        'tf1h_pti_reversal_likely': {
            'type': 'bool',
            'min_unique': 2,  # Should have both True and False
        },
        'tf1d_pti_score': {
            'type': 'float',
            'min_unique': 50,
            'expected_range': (0.0, 1.0),
            'frozen_value': 0.5,
        },
        'tf1d_pti_reversal': {
            'type': 'bool',
            'min_unique': 2,
        },
    }

    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}\n")

    all_passed = True

    for col, specs in pti_cols.items():
        # Check if column exists
        if col not in df.columns:
            print(f"❌ FAIL: {col} - Column missing from feature store")
            all_passed = False
            continue

        # Get column data
        col_data = df[col]
        unique_count = col_data.nunique()
        non_null_count = col_data.notna().sum()
        coverage_pct = (non_null_count / len(df)) * 100

        # Basic stats
        if specs['type'] in ['float', 'int']:
            mean_val = col_data.mean()
            min_val = col_data.min()
            max_val = col_data.max()
            stats_str = f"(mean={mean_val:.3f}, min={min_val:.3f}, max={max_val:.3f})"
        else:
            stats_str = ""

        # Check for frozen value issue
        if 'frozen_value' in specs:
            if unique_count == 1 and col_data.iloc[0] == specs['frozen_value']:
                print(f"❌ FAIL: {col} - STILL FROZEN at {specs['frozen_value']}")
                all_passed = False
                continue

        # Check minimum unique values
        if unique_count < specs['min_unique']:
            print(f"❌ FAIL: {col} - Only {unique_count} unique values (expected >={specs['min_unique']})")
            all_passed = False
            continue

        # Check expected range for numeric columns
        if 'expected_range' in specs:
            min_expected, max_expected = specs['expected_range']
            if min_val < min_expected or max_val > max_expected:
                print(f"⚠️  WARN: {col} - Values outside expected range {specs['expected_range']}: [{min_val:.3f}, {max_val:.3f}]")

        # Check expected values for categorical columns
        if 'expected_values' in specs:
            actual_values = set(col_data.unique())
            expected_values = set(specs['expected_values'])
            unexpected = actual_values - expected_values
            if unexpected:
                print(f"⚠️  WARN: {col} - Unexpected values found: {unexpected}")

        # Success
        print(f"✅ PASS: {col}")
        print(f"         - {unique_count:,} unique values {stats_str}")
        print(f"         - {non_null_count:,}/{len(df):,} non-null ({coverage_pct:.1f}% coverage)")

        # Additional details for categorical columns
        if specs['type'] in ['string', 'bool']:
            value_counts = col_data.value_counts()
            print(f"         - Distribution: {dict(value_counts.head(5))}")

    print(f"\n{'='*80}")

    if all_passed:
        print("✅ ALL PTI FEATURES VALIDATED - FIX SUCCESSFUL")
        print(f"{'='*80}\n")
        print("PTI features are properly calculated and ready for use.")
        print("\nNext steps:")
        print("  1. Update symlink to use this feature store")
        print("  2. Rerun archetype backtests to measure impact")
        print("  3. Verify Archetype A (Spring) signals: expected 10-20 in Q1 2023")
        print("  4. Verify Archetype K (Wick Trap) signals: expected 5-15 in Q1 2023")
        return True
    else:
        print("❌ VALIDATION FAILED - PTI STILL BROKEN")
        print(f"{'='*80}\n")
        print("PTI features are NOT properly calculated.")
        print("\nTroubleshooting:")
        print("  1. Verify add_pti_features() was called in build script")
        print("  2. Check for exceptions during PTI calculation")
        print("  3. Ensure engine/psychology/pti.py functions are working")
        print("  4. Review PTI_SCORES_FIX.md for full context")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate PTI fix in feature store'
    )
    parser.add_argument(
        '--feature-store',
        type=str,
        default='data/features_mtf/BTC_1H_complete_2022-01-01_to_2024-12-31.parquet',
        help='Path to feature store to validate'
    )

    args = parser.parse_args()

    # Run validation
    success = validate_pti_features(args.feature_store)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
