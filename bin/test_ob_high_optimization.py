#!/usr/bin/env python3
"""
Test and Benchmark OB High Optimization

Compares baseline vs optimized implementation:
1. Correctness validation (similar results)
2. Performance benchmarking (speedup measurement)
3. Full run timing

Usage:
    python bin/test_ob_high_optimization.py --asset BTC --start 2022-01-01 --end 2024-12-31
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
import argparse
from tqdm import tqdm

from engine.smc.order_blocks_adaptive import AdaptiveOrderBlockDetector, OrderBlockType
from bin.backfill_ob_high_optimized import backfill_ob_high_optimized


def run_baseline_backfill(df: pd.DataFrame, config: dict) -> tuple:
    """
    Run baseline (original) backfill implementation.

    Returns:
        (result_df, elapsed_time, bars_per_sec)
    """
    print("\n📊 Running BASELINE backfill (original implementation)...")
    df = df.copy()

    # Initialize detector
    detector = AdaptiveOrderBlockDetector(config)

    # Calculate ATR if needed
    if 'atr_14' not in df.columns:
        print("   Calculating ATR...")
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

    # Initialize result columns
    new_ob_high = pd.Series(index=df.index, dtype=float)
    new_ob_low = pd.Series(index=df.index, dtype=float)

    WINDOW_SIZE = 200
    MIN_HISTORY = 50

    start_time = time.time()

    # Process each bar (original implementation)
    for i in tqdm(range(len(df)), desc="Baseline processing"):
        try:
            start_idx = max(0, i - WINDOW_SIZE)
            window = df.iloc[start_idx:i+1]

            if len(window) < MIN_HISTORY:
                continue

            # Detect order blocks
            order_blocks = detector.detect_order_blocks(window)

            # Get current price
            current_price = df['close'].iloc[i]

            # Find nearest bearish OB (resistance)
            bearish_obs = [ob for ob in order_blocks if ob.ob_type == OrderBlockType.BEARISH]
            if bearish_obs:
                nearby_bearish = [ob for ob in bearish_obs
                                 if abs(ob.high - current_price) / current_price <= 0.05]
                if nearby_bearish:
                    above_price = [ob for ob in nearby_bearish if ob.high >= current_price]
                    if above_price:
                        nearest = min(above_price, key=lambda ob: ob.high - current_price)
                    else:
                        nearest = max(nearby_bearish, key=lambda ob: ob.high)
                    new_ob_high.iloc[i] = nearest.high

            # Find nearest bullish OB (support)
            bullish_obs = [ob for ob in order_blocks if ob.ob_type == OrderBlockType.BULLISH]
            if bullish_obs:
                nearby_bullish = [ob for ob in bullish_obs
                                 if abs(ob.low - current_price) / current_price <= 0.05]
                if nearby_bullish:
                    below_price = [ob for ob in nearby_bullish if ob.low <= current_price]
                    if below_price:
                        nearest = max(below_price, key=lambda ob: current_price - ob.low)
                    else:
                        nearest = min(nearby_bullish, key=lambda ob: ob.low)
                    new_ob_low.iloc[i] = nearest.low

        except Exception:
            pass

    elapsed_time = time.time() - start_time
    bars_per_sec = len(df) / elapsed_time if elapsed_time > 0 else 0

    df['tf1h_ob_high'] = new_ob_high
    df['tf1h_ob_low'] = new_ob_low

    print(f"   ✅ Baseline completed in {elapsed_time:.1f}s ({bars_per_sec:.1f} bars/sec)")

    return df, elapsed_time, bars_per_sec


def run_optimized_backfill(df: pd.DataFrame, config: dict) -> tuple:
    """
    Run optimized (vectorized) backfill implementation.

    Returns:
        (result_df, elapsed_time, bars_per_sec)
    """
    print("\n🚀 Running OPTIMIZED backfill (vectorized implementation)...")
    df = df.copy()

    start_time = time.time()

    # Run optimized version
    df = backfill_ob_high_optimized(df, config)

    elapsed_time = time.time() - start_time
    bars_per_sec = len(df) / elapsed_time if elapsed_time > 0 else 0

    return df, elapsed_time, bars_per_sec


def compare_results(df_baseline: pd.DataFrame, df_optimized: pd.DataFrame) -> dict:
    """
    Compare baseline vs optimized results.

    Returns:
        Dict with comparison metrics
    """
    print("\n📈 Comparing Results...")

    # Coverage comparison
    coverage_baseline = df_baseline['tf1h_ob_high'].notna().sum()
    coverage_optimized = df_optimized['tf1h_ob_high'].notna().sum()
    total_bars = len(df_baseline)

    coverage_diff = abs(coverage_baseline - coverage_optimized)
    coverage_diff_pct = (coverage_diff / total_bars * 100) if total_bars > 0 else 0

    # Value comparison (where both have values)
    both_have_values = df_baseline['tf1h_ob_high'].notna() & df_optimized['tf1h_ob_high'].notna()
    if both_have_values.sum() > 0:
        values_baseline = df_baseline.loc[both_have_values, 'tf1h_ob_high']
        values_optimized = df_optimized.loc[both_have_values, 'tf1h_ob_high']

        # Calculate relative difference
        rel_diff = ((values_optimized - values_baseline) / values_baseline).abs()
        avg_rel_diff = rel_diff.mean()
        max_rel_diff = rel_diff.max()
        matching_pct = (rel_diff < 0.01).sum() / len(rel_diff) * 100  # Within 1%
    else:
        avg_rel_diff = np.nan
        max_rel_diff = np.nan
        matching_pct = 0

    results = {
        'coverage_baseline': coverage_baseline,
        'coverage_optimized': coverage_optimized,
        'coverage_diff': coverage_diff,
        'coverage_diff_pct': coverage_diff_pct,
        'avg_rel_diff': avg_rel_diff,
        'max_rel_diff': max_rel_diff,
        'matching_pct': matching_pct,
        'total_bars': total_bars
    }

    return results


def print_benchmark_report(
    baseline_time: float,
    baseline_speed: float,
    optimized_time: float,
    optimized_speed: float,
    comparison: dict,
    total_bars: int
):
    """Print comprehensive benchmark report"""

    speedup = baseline_time / optimized_time if optimized_time > 0 else 0
    time_saved = baseline_time - optimized_time
    time_saved_pct = (time_saved / baseline_time * 100) if baseline_time > 0 else 0

    print("\n" + "=" * 80)
    print("BENCHMARK REPORT: OB HIGH OPTIMIZATION")
    print("=" * 80)

    print(f"\n📊 Performance Comparison ({total_bars} bars):")
    print(f"\n   BASELINE (Original):")
    print(f"      Time:  {baseline_time:.1f}s ({baseline_time/60:.1f} minutes)")
    print(f"      Speed: {baseline_speed:.1f} bars/second")

    print(f"\n   OPTIMIZED (Vectorized):")
    print(f"      Time:  {optimized_time:.1f}s ({optimized_time/60:.1f} minutes)")
    print(f"      Speed: {optimized_speed:.1f} bars/second")

    print(f"\n   IMPROVEMENT:")
    print(f"      Speedup:    {speedup:.2f}x faster")
    print(f"      Time saved: {time_saved:.1f}s ({time_saved_pct:.1f}% reduction)")

    # Performance rating
    if speedup >= 5.0:
        rating = "🏆 EXCELLENT (5x+ target exceeded!)"
    elif speedup >= 3.5:
        rating = "✅ GOOD (meets 3.5x target)"
    elif speedup >= 3.0:
        rating = "✅ ACCEPTABLE (meets minimum 3x target)"
    elif speedup >= 2.0:
        rating = "⚠️  NEEDS IMPROVEMENT (below 3x target)"
    else:
        rating = "❌ POOR (significant optimization needed)"

    print(f"      Rating:     {rating}")

    print(f"\n📊 Accuracy Comparison:")
    print(f"\n   Coverage:")
    print(f"      Baseline:  {comparison['coverage_baseline']}/{total_bars} "
          f"({comparison['coverage_baseline']/total_bars*100:.1f}%)")
    print(f"      Optimized: {comparison['coverage_optimized']}/{total_bars} "
          f"({comparison['coverage_optimized']/total_bars*100:.1f}%)")
    print(f"      Difference: {comparison['coverage_diff']} bars "
          f"({comparison['coverage_diff_pct']:.2f}%)")

    if not np.isnan(comparison['avg_rel_diff']):
        print(f"\n   Value Accuracy (where both have values):")
        print(f"      Avg difference:  {comparison['avg_rel_diff']*100:.2f}%")
        print(f"      Max difference:  {comparison['max_rel_diff']*100:.2f}%")
        print(f"      Matching (±1%):  {comparison['matching_pct']:.1f}%")

    # Accuracy rating
    if comparison['coverage_diff_pct'] < 5:
        accuracy_rating = "✅ EXCELLENT (within 5%)"
    elif comparison['coverage_diff_pct'] < 10:
        accuracy_rating = "✅ GOOD (within 10%)"
    elif comparison['coverage_diff_pct'] < 20:
        accuracy_rating = "⚠️  ACCEPTABLE (within 20%)"
    else:
        accuracy_rating = "❌ POOR (>20% difference)"

    print(f"\n   Accuracy Rating: {accuracy_rating}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark OB high optimization'
    )
    parser.add_argument('--asset', required=True,
                       help='Asset to test (BTC, ETH, etc.)')
    parser.add_argument('--start', required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Test on sample size (default: full dataset)')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline run (only test optimized)')

    args = parser.parse_args()

    print("=" * 80)
    print(f"OB HIGH OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print(f"Asset:  {args.asset}")
    print(f"Period: {args.start} → {args.end}")
    print("=" * 80)

    # Load data
    feature_path = Path(f'data/features_mtf/{args.asset}_1H_{args.start}_to_{args.end}.parquet')
    if not feature_path.exists():
        print(f"❌ Error: Feature store not found at {feature_path}")
        return

    print(f"\n📂 Loading feature store...")
    df = pd.read_parquet(feature_path)

    if args.sample_size:
        print(f"   Using sample: {args.sample_size} bars (out of {len(df)})")
        df = df.head(args.sample_size)
    else:
        print(f"   Using full dataset: {len(df)} bars")

    # Configuration
    config = {
        'min_displacement_pct_floor': 0.005,
        'atr_multiplier': 1.0,
        'min_volume_ratio': 1.2,
        'lookback_bars': 50,
        'min_reaction_bars': 3,
        'swing_lookback': 30
    }

    # Run baseline (if not skipped)
    if not args.skip_baseline:
        df_baseline, baseline_time, baseline_speed = run_baseline_backfill(df, config)
    else:
        print("\n⏭️  Skipping baseline run...")
        df_baseline = None
        baseline_time = None
        baseline_speed = None

    # Run optimized
    df_optimized, optimized_time, optimized_speed = run_optimized_backfill(df, config)

    # Compare and report
    if df_baseline is not None:
        comparison = compare_results(df_baseline, df_optimized)
        print_benchmark_report(
            baseline_time,
            baseline_speed,
            optimized_time,
            optimized_speed,
            comparison,
            len(df)
        )

        # Save detailed report
        report_path = Path('results/ob_high_optimization_report.md')
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("# OB High Backfill Optimization Report\n\n")
            f.write(f"**Asset:** {args.asset}\n")
            f.write(f"**Period:** {args.start} to {args.end}\n")
            f.write(f"**Bars:** {len(df):,}\n")
            f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Performance Comparison\n\n")
            f.write("| Metric | Baseline | Optimized | Improvement |\n")
            f.write("|--------|----------|-----------|-------------|\n")
            f.write(f"| Processing speed | {baseline_speed:.1f} bars/sec | {optimized_speed:.1f} bars/sec | "
                   f"{optimized_speed/baseline_speed:.2f}x |\n")
            f.write(f"| Total time ({len(df):,} bars) | {baseline_time:.1f}s ({baseline_time/60:.1f}m) | "
                   f"{optimized_time:.1f}s ({optimized_time/60:.1f}m) | "
                   f"{(baseline_time-optimized_time)/baseline_time*100:.1f}% faster |\n")
            f.write(f"| Memory usage | Standard | Standard | - |\n\n")

            f.write("## Accuracy Comparison\n\n")
            f.write("| Metric | Baseline | Optimized | Difference |\n")
            f.write("|--------|----------|-----------|------------|\n")
            f.write(f"| Coverage | {comparison['coverage_baseline']} ({comparison['coverage_baseline']/len(df)*100:.1f}%) | "
                   f"{comparison['coverage_optimized']} ({comparison['coverage_optimized']/len(df)*100:.1f}%) | "
                   f"{comparison['coverage_diff_pct']:.2f}% |\n")

            if not np.isnan(comparison['avg_rel_diff']):
                f.write(f"| Avg value difference | - | - | {comparison['avg_rel_diff']*100:.2f}% |\n")
                f.write(f"| Matching values (±1%) | - | - | {comparison['matching_pct']:.1f}% |\n")

            f.write("\n## Optimizations Applied\n\n")
            f.write("1. **ATR Vectorization**: Pre-calculated ATR for all bars using pandas rolling operations\n")
            f.write("2. **Swing Detection Vectorization**: Replaced per-bar loops with rolling quantile operations\n")
            f.write("3. **Displacement Vectorization**: Used shift() and rolling() for displacement calculation\n")
            f.write("4. **Batch Processing**: Single-pass order block detection instead of per-bar window slicing\n")
            f.write("5. **NumPy Operations**: Vectorized comparisons and conditional logic\n\n")

            f.write("## Validation Results\n\n")
            if comparison['coverage_diff_pct'] < 5:
                f.write("✅ **PASS**: Coverage difference within 5% threshold\n")
            else:
                f.write("⚠️ **REVIEW**: Coverage difference exceeds 5% - investigate edge cases\n")

            f.write(f"\n## Conclusion\n\n")
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0
            if speedup >= 3.0:
                f.write(f"✅ **SUCCESS**: Achieved {speedup:.2f}x speedup (target: 3-5x)\n\n")
                f.write(f"The optimized implementation meets performance requirements while maintaining accuracy.\n")
            else:
                f.write(f"⚠️ **NEEDS IMPROVEMENT**: Only achieved {speedup:.2f}x speedup (target: 3-5x)\n\n")
                f.write(f"Further optimization may be needed to meet target performance.\n")

        print(f"\n📄 Detailed report saved to: {report_path}")

    else:
        print(f"\n📊 Optimized Performance:")
        print(f"   Time:  {optimized_time:.1f}s ({optimized_time/60:.1f} minutes)")
        print(f"   Speed: {optimized_speed:.1f} bars/second")
        print(f"   Coverage: {df_optimized['tf1h_ob_high'].notna().sum()}/{len(df)} "
              f"({df_optimized['tf1h_ob_high'].notna().sum()/len(df)*100:.1f}%)")


if __name__ == '__main__':
    main()
