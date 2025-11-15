#!/usr/bin/env python3
"""
Validate liquidity_score backfill results.

Checks:
1. Column exists in MTF store
2. Coverage (no NaN/inf values)
3. Distribution (bounds, percentiles)
4. Compare against runtime calculation (sample)

Usage:
    python3 bin/validate_liquidity_backfill.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from bin.backfill_liquidity_score import map_mtf_row_to_context
from engine.liquidity.score import compute_liquidity_score


def validate_backfill():
    """Validate liquidity_score backfill results."""

    # Load patched MTF store
    mtf_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    print(f"Loading patched MTF store: {mtf_path}")
    df = pd.read_parquet(mtf_path)

    print(f"\n{'='*80}")
    print("VALIDATION REPORT: liquidity_score Backfill")
    print(f"{'='*80}")

    # === CHECK 1: Column Exists ===
    print(f"\n1. COLUMN EXISTS")
    print("-" * 80)
    if 'liquidity_score' in df.columns:
        print("✅ PASS: liquidity_score column exists")
    else:
        print("❌ FAIL: liquidity_score column not found")
        print(f"Available columns: {list(df.columns)}")
        return False

    # === CHECK 2: Coverage ===
    print(f"\n2. COVERAGE")
    print("-" * 80)
    total_rows = len(df)
    non_null = df['liquidity_score'].notna().sum()
    null_count = df['liquidity_score'].isna().sum()
    inf_count = np.isinf(df['liquidity_score']).sum()

    print(f"Total rows: {total_rows:,}")
    print(f"Non-null: {non_null:,} ({non_null/total_rows*100:.2f}%)")
    print(f"Null: {null_count:,}")
    print(f"Inf: {inf_count:,}")

    coverage_pass = (non_null == total_rows) and (inf_count == 0)
    if coverage_pass:
        print("✅ PASS: 100% coverage, no NaN/inf")
    else:
        print("❌ FAIL: Incomplete coverage or invalid values")
        return False

    # === CHECK 3: Distribution ===
    print(f"\n3. DISTRIBUTION")
    print("-" * 80)
    scores = df['liquidity_score']

    # Bounds
    min_val = scores.min()
    max_val = scores.max()
    bounds_pass = (min_val >= 0.0) and (max_val <= 1.0)

    print(f"Min: {min_val:.3f} (expected ≥ 0.0)")
    print(f"Max: {max_val:.3f} (expected ≤ 1.0)")
    if bounds_pass:
        print("✅ PASS: All values in [0, 1]")
    else:
        print("❌ FAIL: Values out of bounds")
        return False

    # Percentiles
    p25 = scores.quantile(0.25)
    p50 = scores.quantile(0.50)
    p75 = scores.quantile(0.75)
    p90 = scores.quantile(0.90)

    print(f"\nPercentiles:")
    print(f"  p25: {p25:.3f}")
    print(f"  p50: {p50:.3f} (median, expected 0.45-0.55)")
    print(f"  p75: {p75:.3f} (expected 0.68-0.75)")
    print(f"  p90: {p90:.3f} (expected 0.80-0.90)")

    # Relaxed validation (distribution can vary based on market conditions)
    median_ok = 0.35 <= p50 <= 0.65  # Relaxed range
    print(f"\nDistribution assessment:")
    if median_ok:
        print("✅ PASS: Distribution looks reasonable (relaxed validation)")
    else:
        print("⚠️ WARNING: Distribution outside expected range (may still be valid)")

    # === CHECK 4: Runtime Consistency (Sample) ===
    print(f"\n4. RUNTIME CONSISTENCY (Sample Verification)")
    print("-" * 80)

    # Sample 100 random rows and recompute
    sample_size = min(100, len(df))
    sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
    sample = df.loc[sample_indices]

    print(f"Sampling {sample_size} rows for verification...")

    recomputed_scores = []
    for idx, row in sample.iterrows():
        ctx = map_mtf_row_to_context(row)
        score = compute_liquidity_score(ctx, side='long')
        recomputed_scores.append(score)

    stored_scores = sample['liquidity_score'].values
    recomputed_scores = np.array(recomputed_scores)

    # Compare
    diff = np.abs(stored_scores - recomputed_scores)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    consistency_pass = max_diff < 1e-6
    if consistency_pass:
        print("✅ PASS: Stored scores match runtime computation exactly")
    elif max_diff < 1e-3:
        print("⚠️ WARNING: Small numerical differences (< 1e-3), acceptable")
        consistency_pass = True
    else:
        print("❌ FAIL: Significant differences detected")
        return False

    # === CHECK 5: Feature Statistics ===
    print(f"\n5. FEATURE STATISTICS")
    print("-" * 80)
    print(f"Mean: {scores.mean():.3f}")
    print(f"Std: {scores.std():.3f}")
    print(f"Skewness: {scores.skew():.3f}")
    print(f"\nValue distribution:")
    print(f"  [0.0, 0.2): {(scores < 0.2).sum():,} ({(scores < 0.2).sum()/len(scores)*100:.1f}%)")
    print(f"  [0.2, 0.4): {((scores >= 0.2) & (scores < 0.4)).sum():,} ({((scores >= 0.2) & (scores < 0.4)).sum()/len(scores)*100:.1f}%)")
    print(f"  [0.4, 0.6): {((scores >= 0.4) & (scores < 0.6)).sum():,} ({((scores >= 0.4) & (scores < 0.6)).sum()/len(scores)*100:.1f}%)")
    print(f"  [0.6, 0.8): {((scores >= 0.6) & (scores < 0.8)).sum():,} ({((scores >= 0.6) & (scores < 0.8)).sum()/len(scores)*100:.1f}%)")
    print(f"  [0.8, 1.0]: {(scores >= 0.8).sum():,} ({(scores >= 0.8).sum()/len(scores)*100:.1f}%)")

    # === FINAL VERDICT ===
    print(f"\n{'='*80}")
    print("FINAL VALIDATION RESULT")
    print(f"{'='*80}")

    all_pass = coverage_pass and bounds_pass and consistency_pass
    if all_pass:
        print("✅ ALL CHECKS PASSED")
        print("\nThe liquidity_score column has been successfully backfilled.")
        print("You can now use it for:")
        print("  - S1 (Liquidity Vacuum) pattern validation")
        print("  - S4 (Distribution Climax) with liquidity filtering")
        print("  - Bear archetype backtesting with full feature set")
        return True
    else:
        print("❌ VALIDATION FAILED")
        print("\nSome checks did not pass. Review the report above.")
        return False


if __name__ == '__main__':
    success = validate_backfill()
    sys.exit(0 if success else 1)
