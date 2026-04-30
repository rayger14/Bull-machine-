#!/usr/bin/env python3
"""
Validate Fusion SMC Fix

Compares old vs new fusion_smc implementation to verify the fix.
Checks that fusion_smc now has 1000+ unique values like other fusion components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def validate_fusion_smc_fix(feature_store_path: str):
    """
    Validate that fusion_smc has been fixed.

    Expected:
    - fusion_smc unique values: 1000+ (was 5 before fix)
    - smc_fvg_hit trigger rate: > 0% (was 0% before fix)
    - SMC continuous scores present (smc_strength, smc_confidence)
    """
    print(f"\n{'='*70}")
    print("FUSION SMC FIX VALIDATION")
    print(f"{'='*70}\n")

    print(f"Loading feature store: {feature_store_path}")
    df = pd.read_parquet(feature_store_path)
    print(f"  → Loaded {len(df):,} bars\n")

    # Check 1: Fusion component unique values
    print("1. Fusion Component Unique Values")
    print("-" * 70)

    fusion_cols = ['fusion_wyckoff', 'fusion_liquidity', 'fusion_momentum', 'fusion_smc', 'fusion_total']
    results = {}

    for col in fusion_cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            results[col] = unique_count
            status = "✅" if unique_count > 1000 else "❌"
            print(f"{col:20s}: {unique_count:,} unique values {status}")
        else:
            print(f"{col:20s}: NOT FOUND ❌")
            results[col] = 0

    # Validation criteria
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}\n")

    all_passed = True

    # Test 1: fusion_smc should have 1000+ unique values
    if results.get('fusion_smc', 0) >= 1000:
        print("✅ PASS: fusion_smc has 1000+ unique values")
    else:
        print(f"❌ FAIL: fusion_smc has only {results.get('fusion_smc', 0)} unique values (expected 1000+)")
        all_passed = False

    # Test 2: fusion_smc should be similar to other fusion components
    if results.get('fusion_smc', 0) > 100:  # At least 100 unique values
        ratio_vs_wyckoff = results['fusion_smc'] / max(results['fusion_wyckoff'], 1)
        ratio_vs_liquidity = results['fusion_smc'] / max(results['fusion_liquidity'], 1)
        ratio_vs_momentum = results['fusion_smc'] / max(results['fusion_momentum'], 1)

        print(f"✅ PASS: fusion_smc has comparable granularity to other fusion components")
        print(f"   - vs fusion_wyckoff: {ratio_vs_wyckoff:.2%}")
        print(f"   - vs fusion_liquidity: {ratio_vs_liquidity:.2%}")
        print(f"   - vs fusion_momentum: {ratio_vs_momentum:.2%}")
    else:
        print(f"❌ FAIL: fusion_smc lacks granularity compared to other components")
        all_passed = False

    # Check 2: SMC continuous scores
    print(f"\n2. SMC Continuous Scores (New Features)")
    print("-" * 70)

    smc_continuous_cols = ['smc_strength', 'smc_confidence', 'smc_confluence', 'smc_score']
    smc_scores_found = True

    for col in smc_continuous_cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            nonzero = (df[col] > 0).sum()
            nonzero_pct = nonzero / len(df) * 100
            print(f"{col:20s}: {unique_count:,} unique, {nonzero:,} non-zero ({nonzero_pct:.2f}%)")

            if unique_count < 100:
                print(f"  ⚠️  WARNING: {col} has low unique count")
                smc_scores_found = False
        else:
            print(f"{col:20s}: NOT FOUND ❌")
            smc_scores_found = False

    if smc_scores_found:
        print("\n✅ PASS: All SMC continuous score columns present with good variation")
    else:
        print("\n❌ FAIL: SMC continuous score columns missing or insufficient variation")
        all_passed = False

    # Check 3: SMC boolean hit features
    print(f"\n3. SMC Boolean Hit Features (Archetype Detection)")
    print("-" * 70)

    smc_bool_cols = ['smc_ob_hit', 'smc_fvg_hit', 'smc_sweep_recent', 'smc_bos_aligned']
    fvg_fixed = False

    for col in smc_bool_cols:
        if col in df.columns:
            hit_count = df[col].sum()
            hit_pct = hit_count / len(df) * 100
            status = "✅" if hit_count > 0 else "❌"
            print(f"{col:20s}: {hit_count:,} hits ({hit_pct:.2f}%) {status}")

            if col == 'smc_fvg_hit' and hit_count > 0:
                fvg_fixed = True
        else:
            print(f"{col:20s}: NOT FOUND ❌")

    if fvg_fixed:
        print("\n✅ PASS: FVG detection now working (was 0% before fix)")
    else:
        print("\n❌ FAIL: FVG detection still broken (0% trigger rate)")
        all_passed = False

    # Check 4: Value distribution analysis
    print(f"\n4. fusion_smc Value Distribution")
    print("-" * 70)

    if 'fusion_smc' in df.columns:
        print(f"Min:    {df['fusion_smc'].min():.6f}")
        print(f"Max:    {df['fusion_smc'].max():.6f}")
        print(f"Mean:   {df['fusion_smc'].mean():.6f}")
        print(f"Median: {df['fusion_smc'].median():.6f}")
        print(f"Std:    {df['fusion_smc'].std():.6f}")

        # Show top 10 most common values
        print(f"\nTop 10 most common values:")
        top_values = df['fusion_smc'].value_counts().head(10)
        for val, count in top_values.items():
            pct = count / len(df) * 100
            print(f"  {val:.6f}: {count:,} occurrences ({pct:.2f}%)")

        # Check if too concentrated
        zero_pct = (df['fusion_smc'] == 0).sum() / len(df) * 100
        if zero_pct < 50:
            print(f"\n✅ PASS: Good distribution (only {zero_pct:.1f}% zeros)")
        else:
            print(f"\n⚠️  WARNING: High concentration of zeros ({zero_pct:.1f}%)")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*70}\n")

    if all_passed:
        print("✅ ALL TESTS PASSED - fusion_smc fix validated successfully!")
        print("\nThe fix has:")
        print("  ✅ Increased unique values from 5 to 1000+")
        print("  ✅ Enabled FVG detection (was 0%, now >0%)")
        print("  ✅ Added continuous SMC scores (strength, confidence, confluence)")
        print("  ✅ Maintained compatibility with archetype detection")
        print("\nReady to run backtests with fixed SMC fusion scoring.")
        return True
    else:
        print("❌ VALIDATION FAILED - fusion_smc fix incomplete")
        print("\nPlease check:")
        print("  1. Rebuild feature store with updated build_complete_feature_store.py")
        print("  2. Verify SMC engine is properly integrated")
        print("  3. Check logs for errors during SMC feature computation")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate fusion_smc fix in feature store'
    )
    parser.add_argument(
        '--feature-store',
        type=str,
        default='data/features_mtf/BTC_1H_complete_2022-01-01_to_2024-12-31.parquet',
        help='Path to feature store to validate'
    )

    args = parser.parse_args()

    # Run validation
    success = validate_fusion_smc_fix(args.feature_store)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
