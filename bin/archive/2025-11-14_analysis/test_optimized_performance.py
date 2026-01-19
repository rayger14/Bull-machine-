#!/usr/bin/env python3
"""
Compare baseline vs optimized liquidity backfill performance.

Tests on subset to validate:
1. Correctness (scores match)
2. Performance improvement
3. Speedup metrics

Usage:
    python3 bin/test_optimized_performance.py --limit 1000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
import argparse

# Import baseline
from bin.backfill_liquidity_score import (
    compute_liquidity_scores_batch as compute_baseline
)

# Import optimized
from bin.backfill_liquidity_score_optimized import (
    compute_liquidity_scores_parallel as compute_optimized
)


def compare_performance(limit: int = 1000, workers: int = 4):
    """Compare baseline vs optimized implementation."""

    # Load MTF store
    mtf_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    print(f"Loading MTF store: {mtf_path}")
    df = pd.read_parquet(mtf_path)
    total_rows = len(df)

    # Use subset for testing
    df_test = df.head(limit)
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON TEST")
    print(f"{'='*80}")
    print(f"Total rows in MTF store: {total_rows:,}")
    print(f"Test subset: {len(df_test):,} rows")
    print(f"Workers (optimized): {workers}")

    # === BASELINE ===
    print(f"\n{'='*80}")
    print("BASELINE: compute_liquidity_scores_batch()")
    print(f"{'='*80}")

    start_baseline = time.time()
    scores_baseline = compute_baseline(df_test, side='long', show_progress=False)
    elapsed_baseline = time.time() - start_baseline

    print(f"\n✅ Baseline completed in {elapsed_baseline:.2f}s")
    print(f"   ({elapsed_baseline*1000/len(df_test):.2f}ms per row)")
    print(f"   Projected full runtime: {elapsed_baseline * total_rows / len(df_test):.2f}s")

    # === OPTIMIZED ===
    print(f"\n{'='*80}")
    print("OPTIMIZED: compute_liquidity_scores_parallel()")
    print(f"{'='*80}")

    start_optimized = time.time()
    scores_optimized = compute_optimized(
        df_test,
        side='long',
        n_workers=workers,
        chunk_size=min(1000, len(df_test)),
        show_progress=False
    )
    elapsed_optimized = time.time() - start_optimized

    print(f"\n✅ Optimized completed in {elapsed_optimized:.2f}s")
    print(f"   ({elapsed_optimized*1000/len(df_test):.2f}ms per row)")
    print(f"   Projected full runtime: {elapsed_optimized * total_rows / len(df_test):.2f}s")

    # === VALIDATION ===
    print(f"\n{'='*80}")
    print("VALIDATION: Correctness Check")
    print(f"{'='*80}")

    # Check scores match
    diff = np.abs(scores_baseline.values - scores_optimized.values)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Baseline mean: {scores_baseline.mean():.3f}")
    print(f"Optimized mean: {scores_optimized.mean():.3f}")

    if max_diff < 1e-6:
        print("✅ PASS: Scores match exactly (< 1e-6 difference)")
        correctness_pass = True
    elif max_diff < 1e-3:
        print("⚠️ WARNING: Small numerical differences (< 1e-3)")
        correctness_pass = True
    else:
        print("❌ FAIL: Significant differences detected")
        correctness_pass = False

    # === PERFORMANCE SUMMARY ===
    speedup = elapsed_baseline / elapsed_optimized
    time_saved = elapsed_baseline - elapsed_optimized
    time_saved_full = (elapsed_baseline - elapsed_optimized) * total_rows / len(df_test)

    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Method':<30} {'Time':<15} {'Per Row':<15} {'Projected (26K)':<20}")
    print("-" * 80)
    print(f"{'Baseline (iterrows)':<30} {elapsed_baseline:>10.2f}s    {elapsed_baseline*1000/len(df_test):>8.2f}ms    {elapsed_baseline * total_rows / len(df_test):>10.2f}s")
    print(f"{'Optimized (parallel)':<30} {elapsed_optimized:>10.2f}s    {elapsed_optimized*1000/len(df_test):>8.2f}ms    {elapsed_optimized * total_rows / len(df_test):>10.2f}s")
    print("-" * 80)
    print(f"{'Speedup:':<30} {speedup:>10.2f}x")
    print(f"{'Time saved (subset):':<30} {time_saved:>10.2f}s")
    print(f"{'Time saved (full):':<30} {time_saved_full:>10.2f}s")

    # === FINAL VERDICT ===
    print(f"\n{'='*80}")
    if correctness_pass and speedup > 1.0:
        print("✅ OPTIMIZATION SUCCESS")
        print(f"{'='*80}")
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Full backfill estimated: {elapsed_optimized * total_rows / len(df_test):.2f}s (~{elapsed_optimized * total_rows / len(df_test) / 60:.1f} minutes)")
        return True
    else:
        print("❌ OPTIMIZATION FAILED")
        print(f"{'='*80}")
        if not correctness_pass:
            print("Reason: Correctness check failed")
        if speedup <= 1.0:
            print(f"Reason: No speedup achieved ({speedup:.2f}x)")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test optimized performance')
    parser.add_argument('--limit', type=int, default=1000, help='Number of rows to test')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()

    success = compare_performance(limit=args.limit, workers=args.workers)
    sys.exit(0 if success else 1)
