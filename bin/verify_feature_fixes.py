#!/usr/bin/env python3
"""
Quick Feature Fix Verification

Validates that broken features are now working correctly.
"""

import pandas as pd
import sys

def verify_fixes(feature_store_path: str):
    """Verify all feature fixes are working."""

    print("="*80)
    print("FEATURE FIX VERIFICATION")
    print("="*80)

    # Load data
    df = pd.read_parquet(feature_store_path)
    print(f"\nLoaded: {feature_store_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Define expected behavior
    expected = {
        'wyckoff_spring_b': {
            'type': 'bool',
            'min_pct': 0.5,  # At least 0.5% trigger rate
            'max_pct': 5.0,  # At most 5% trigger rate
            'description': 'Shallow Wyckoff spring pattern'
        },
        'temporal_confluence': {
            'type': 'bool',
            'min_pct': 15.0,  # At least 15% (multi-TF alignment common)
            'max_pct': 35.0,  # At most 35%
            'description': 'Multi-timeframe confluence'
        },
        'tf4h_fvg_present': {
            'type': 'bool',
            'min_pct': 3.0,   # At least 3%
            'max_pct': 15.0,  # At most 15%
            'description': '4H Fair Value Gap detection'
        },
        'tf4h_choch_flag': {
            'type': 'bool',
            'min_pct': 0.1,   # At least 0.1% (rare signal)
            'max_pct': 2.0,   # At most 2%
            'description': '4H Change of Character (reversal)'
        },
        'mtf_alignment_ok': {
            'type': 'bool',
            'min_pct': 20.0,  # At least 20% (trending periods)
            'max_pct': 100.0, # Can be high (permissive)
            'description': 'Multi-timeframe trend alignment'
        },
        'wyckoff_pti_confluence': {
            'type': 'bool',
            'min_pct': 0.0,   # Can be 0 (very rare)
            'max_pct': 3.0,   # At most 3%
            'description': 'Wyckoff trap + high PTI (experimental)'
        }
    }

    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)

    all_passed = True

    for feature, spec in expected.items():
        if feature not in df.columns:
            print(f"\n❌ {feature}")
            print(f"   MISSING from feature store")
            all_passed = False
            continue

        # Calculate metrics
        if df[feature].dtype == bool:
            trigger_count = df[feature].sum()
            trigger_pct = 100 * trigger_count / len(df)
        else:
            trigger_count = (df[feature] != 0).sum()
            trigger_pct = 100 * trigger_count / len(df)

        # Check if in expected range
        in_range = spec['min_pct'] <= trigger_pct <= spec['max_pct']
        status = "✅" if in_range else "⚠️"

        print(f"\n{status} {feature}")
        print(f"   {spec['description']}")
        print(f"   Triggers: {trigger_count:,} / {len(df):,} ({trigger_pct:.2f}%)")
        print(f"   Expected: {spec['min_pct']:.1f}% - {spec['max_pct']:.1f}%")

        if not in_range:
            if trigger_pct < spec['min_pct']:
                print(f"   ⚠️  WARNING: Lower than expected (may still be broken)")
            else:
                print(f"   ⚠️  WARNING: Higher than expected (may need tuning)")
            all_passed = False

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if all_passed:
        print("✅ ALL FEATURES VERIFIED")
        print("   Ready for baseline validation")
        return 0
    else:
        print("⚠️  SOME FEATURES OUT OF EXPECTED RANGE")
        print("   Review warnings above")
        print("   May still be acceptable depending on data characteristics")
        return 1

if __name__ == "__main__":
    feature_store = "data/features_2022_with_regimes.parquet"

    if len(sys.argv) > 1:
        feature_store = sys.argv[1]

    exit_code = verify_fixes(feature_store)
    sys.exit(exit_code)
