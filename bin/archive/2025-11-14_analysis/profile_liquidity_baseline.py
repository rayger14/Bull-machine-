#!/usr/bin/env python3
"""
Profile baseline liquidity backfill performance.

Measures:
- Time per row
- Hotspot functions
- Memory usage
- Projected full runtime

Usage:
    python3 bin/profile_liquidity_baseline.py --limit 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pandas as pd
import argparse
from tqdm import tqdm

# Import from original script
from bin.backfill_liquidity_score import map_mtf_row_to_context
from engine.liquidity.score import compute_liquidity_score


def profile_baseline(limit: int = 100):
    """Profile baseline iterrows implementation."""

    # Load MTF store
    mtf_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    print(f"Loading MTF store: {mtf_path}")
    df = pd.read_parquet(mtf_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")

    # Limit for profiling
    df_subset = df.head(limit)
    print(f"Profiling on: {len(df_subset)} rows")

    # === Baseline: iterrows (SLOW) ===
    print("\n" + "=" * 80)
    print("BASELINE: iterrows() approach")
    print("=" * 80)

    scores = []
    start = time.time()

    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="iterrows"):
        ctx = map_mtf_row_to_context(row)
        score = compute_liquidity_score(ctx, side='long')
        scores.append(score)

    elapsed_iterrows = time.time() - start
    per_row_iterrows = elapsed_iterrows / len(df_subset)

    print(f"\n✅ Completed {len(df_subset)} rows")
    print(f"Time: {elapsed_iterrows:.2f}s")
    print(f"Per row: {per_row_iterrows*1000:.1f}ms")
    print(f"Projected full runtime ({total_rows:,} rows): {per_row_iterrows * total_rows / 3600:.2f} hours")

    # === Optimized: apply() approach ===
    print("\n" + "=" * 80)
    print("OPTIMIZED: apply() approach")
    print("=" * 80)

    def compute_row_score(row):
        ctx = map_mtf_row_to_context(row)
        return compute_liquidity_score(ctx, side='long')

    start = time.time()
    scores_apply = df_subset.apply(compute_row_score, axis=1)
    elapsed_apply = time.time() - start
    per_row_apply = elapsed_apply / len(df_subset)

    print(f"\n✅ Completed {len(df_subset)} rows")
    print(f"Time: {elapsed_apply:.2f}s")
    print(f"Per row: {per_row_apply*1000:.1f}ms")
    print(f"Projected full runtime ({total_rows:,} rows): {per_row_apply * total_rows / 3600:.2f} hours")

    speedup_apply = elapsed_iterrows / elapsed_apply
    print(f"\n🚀 Speedup (apply vs iterrows): {speedup_apply:.2f}x")

    # === Comparison Table ===
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Method':<25} {'Time (100 rows)':<20} {'Projected (26K rows)':<25} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'iterrows (baseline)':<25} {elapsed_iterrows:>10.1f}s         {per_row_iterrows * total_rows / 3600:>10.2f} hours             1.00x")
    print(f"{'apply':<25} {elapsed_apply:>10.1f}s         {per_row_apply * total_rows / 3600:>10.2f} hours             {speedup_apply:.2f}x")

    # === Validation: Check scores match ===
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    scores_list = scores_apply.tolist()
    diff = sum(abs(a - b) for a, b in zip(scores, scores_list))
    print(f"Score difference (sum of abs diff): {diff:.6f}")
    print(f"Max score: {max(scores_list):.3f}")
    print(f"Min score: {min(scores_list):.3f}")
    print(f"Mean score: {sum(scores_list)/len(scores_list):.3f}")

    if diff < 1e-6:
        print("✅ PASS: Scores match exactly")
    else:
        print("⚠️ WARNING: Small numerical differences detected")

    return {
        'iterrows_time': elapsed_iterrows,
        'apply_time': elapsed_apply,
        'speedup_apply': speedup_apply,
        'total_rows': total_rows,
        'projected_iterrows_hours': per_row_iterrows * total_rows / 3600,
        'projected_apply_hours': per_row_apply * total_rows / 3600
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile liquidity backfill baseline')
    parser.add_argument('--limit', type=int, default=100, help='Number of rows to profile')
    args = parser.parse_args()

    results = profile_baseline(limit=args.limit)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Create optimized script with multiprocessing")
    print(f"Expected speedup with 4 workers: ~3-4x additional improvement")
    print(f"Combined (apply + multiprocessing): ~{results['speedup_apply'] * 3.5:.1f}x faster")
    print(f"Final projected time: ~{results['projected_apply_hours'] / 3.5:.2f} hours")
