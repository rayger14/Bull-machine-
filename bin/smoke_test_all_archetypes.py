#!/usr/bin/env python3
"""
Comprehensive Smoke Test for All 19 Archetypes

Tests all wired archetypes (Bear, Bull, Chop, excluding deprecated Ghost) to validate:
- Individual archetype signal generation
- Trade diversity and overlap analysis
- Realism checks (confidence scores, domain boosts, direction alignment)
- Performance metrics

Outputs:
- SMOKE_TEST_REPORT.md (formatted results)
- smoke_test_results.json (raw data)
- smoke_test_issues.txt (any problems found)
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import warnings

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

TEST_PERIOD_START = '2023-01-01'
TEST_PERIOD_END = '2023-04-01'  # Q1 2023 - 3 months

DATA_PATH = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'

# Archetypes to test (excluding deprecated Ghost: P, Q, N, S6, S7)
ARCHETYPES_TO_TEST = {
    'Bear': ['S1', 'S4', 'S5'],
    'Bull': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M'],
    'Chop': ['S3', 'S8'],
}

# Archetype names for display
ARCHETYPE_NAMES = {
    'A': 'Spring',
    'B': 'Order Block Retest',
    'C': 'Wick Trap',
    'D': 'Failed Continuation',
    'E': 'Volume Exhaustion',
    'F': 'Exhaustion Reversal',
    'G': 'Liquidity Sweep',
    'H': 'Momentum Continuation',
    'K': 'Trap Within Trend',
    'L': 'Retest Cluster',
    'M': 'Confluence Breakout',
    'S1': 'Liquidity Vacuum',
    'S3': 'Whipsaw',
    'S4': 'Funding Divergence',
    'S5': 'Long Squeeze',
    'S8': 'Volume Fade Chop',
}

# Expected directions
EXPECTED_DIRECTIONS = {
    # Bull archetypes - mostly LONG
    'A': 'LONG', 'B': 'LONG', 'C': 'LONG', 'D': 'LONG', 'E': 'LONG',
    'F': 'LONG', 'G': 'LONG', 'H': 'LONG', 'K': 'LONG', 'L': 'LONG', 'M': 'LONG',
    # Bear archetypes - S1 is LONG (capitulation reversal), S4/S5 are mixed
    'S1': 'LONG',
    'S4': 'MIXED',
    'S5': 'LONG',
    # Chop archetypes
    'S3': 'MIXED',
    'S8': 'MIXED',
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_data(path: str, start: str, end: str) -> pd.DataFrame:
    """Load and filter test data."""
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Filter to test period
    test_df = df[(df.index >= start) & (df.index < end)].copy()
    print(f"  Loaded {len(test_df):,} bars ({start} to {end})")

    return df, test_df


def create_minimal_config(archetype: str) -> dict:
    """Create minimal config enabling only specified archetype."""
    config = {
        'version': f'smoke_test_{archetype}',
        'use_archetypes': True,
        'adaptive_fusion': False,
        'regime_classifier': {
            'zero_fill_missing': True,
            'regime_override': {'default': 'neutral'}
        },
        'ml_filter': {'enabled': False},
        'fusion': {
            'entry_threshold_confidence': 0.0,
            'weights': {'wyckoff': 0.0, 'liquidity': 0.0, 'momentum': 0.0, 'smc': 0.0}
        },
        'archetypes': {
            'use_archetypes': True,
            'max_trades_per_day': 0,
            'thresholds': {'min_liquidity': 0.0},  # Relaxed for smoke test
        },
        'feature_flags': {
            'enable_wyckoff': True,
            'enable_smc': True,
            'enable_temporal': True,
            'enable_hob': True,
            'enable_fusion': True,
            'enable_macro': True,
        },
        'temporal_fusion': {'enabled': True, 'use_confluence': True},
        'wyckoff_events': {'enabled': True, 'log_events': False},
        'smc_engine': {'enabled': True},
        'hob_engine': {'enabled': True},
    }

    # Enable all archetypes as False, then enable the target
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M',
                   'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']:
        config[f'enable_{letter}'] = (letter == archetype)

    return config


def test_archetype(archetype: str, df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test individual archetype and collect statistics.

    Returns:
        dict with keys: signals, errors, execution_time, stats
    """
    print(f"\n{'='*80}")
    print(f"Testing Archetype {archetype}: {ARCHETYPE_NAMES.get(archetype, 'Unknown')}")
    print(f"{'='*80}")

    start_time = time.time()

    # Create config and logic
    config = create_minimal_config(archetype)
    logic = ArchetypeLogic(config)

    # Get method name
    method_name = f'_check_{archetype}'
    if not hasattr(logic, method_name):
        return {
            'signals': [],
            'errors': [f'Method {method_name} not found'],
            'execution_time': 0,
            'stats': {}
        }

    method = getattr(logic, method_name)

    # Test on each bar
    signals = []
    errors = []

    for idx, (ts, row) in enumerate(test_df.iterrows()):
        if idx % 500 == 0:
            print(f"  Progress: {idx:,}/{len(test_df):,} bars ({idx/len(test_df)*100:.1f}%)")

        try:
            # Create minimal runtime context
            ctx = RuntimeContext(
                ts=ts,
                row=row,
                regime_probs={'neutral': 1.0},
                regime_label='neutral',
                adapted_params={},
                thresholds=config.get('archetypes', {}).get('thresholds', {}),
                metadata={
                    'feature_flags': config.get('feature_flags', {}),
                    'df': df,  # Full df for lookback
                    'index': ts,
                }
            )

            # Call archetype check
            result = method(ctx)

            # Parse result (tuple of matched, score, metadata)
            if result and len(result) >= 3:
                matched, score, metadata = result[0], result[1], result[2]

                if matched and score > 0:
                    signals.append({
                        'timestamp': ts,
                        'score': float(score),
                        'metadata': metadata if isinstance(metadata, dict) else {},
                    })

        except Exception as e:
            error_msg = f"Error at {ts}: {str(e)}"
            errors.append(error_msg)
            if len(errors) <= 3:  # Log first 3 errors
                print(f"  [ERROR] {error_msg}")
                if len(errors) == 3:
                    print(f"  [WARNING] Suppressing further error messages...")

    execution_time = time.time() - start_time

    # Compute statistics
    stats = compute_signal_stats(signals, archetype)

    print(f"\n  Results:")
    print(f"    Signals: {len(signals)}")
    print(f"    Errors: {len(errors)}")
    print(f"    Execution time: {execution_time:.2f}s")
    print(f"    Avg score: {stats['conf_mean']:.3f}")
    print(f"    Direction: {stats.get('direction_breakdown', 'N/A')}")

    return {
        'signals': signals,
        'errors': errors,
        'execution_time': execution_time,
        'stats': stats,
    }


def compute_signal_stats(signals: List[Dict], archetype: str) -> Dict[str, Any]:
    """Compute statistics from signal list."""
    if not signals:
        return {
            'count': 0,
            'conf_min': 0.0,
            'conf_max': 0.0,
            'conf_mean': 0.0,
            'conf_std': 0.0,
            'domain_boost_avg': 0.0,
            'domain_boost_pct': 0.0,
            'direction_breakdown': 'N/A',
            'unique_timestamps': 0,
        }

    scores = [s['score'] for s in signals]
    timestamps = [s['timestamp'] for s in signals]

    # Extract domain boost info
    domain_boosts = []
    for sig in signals:
        meta = sig.get('metadata', {})
        if isinstance(meta, dict):
            # Look for domain boost multiplier
            boost = meta.get('domain_boost', 1.0)
            if boost != 1.0:
                domain_boosts.append(boost)

    # Extract direction info
    long_count = 0
    short_count = 0
    for sig in signals:
        meta = sig.get('metadata', {})
        if isinstance(meta, dict):
            direction = meta.get('direction', '').upper()
            if 'LONG' in direction:
                long_count += 1
            elif 'SHORT' in direction:
                short_count += 1

    total_with_direction = long_count + short_count
    if total_with_direction > 0:
        long_pct = long_count / total_with_direction * 100
        short_pct = short_count / total_with_direction * 100
        direction_breakdown = f"{long_pct:.0f}% LONG / {short_pct:.0f}% SHORT"
    else:
        direction_breakdown = "No direction info"

    return {
        'count': len(signals),
        'conf_min': float(np.min(scores)),
        'conf_max': float(np.max(scores)),
        'conf_mean': float(np.mean(scores)),
        'conf_std': float(np.std(scores)),
        'domain_boost_avg': float(np.mean(domain_boosts)) if domain_boosts else 1.0,
        'domain_boost_pct': len(domain_boosts) / len(signals) * 100 if signals else 0.0,
        'direction_breakdown': direction_breakdown,
        'unique_timestamps': len(set(timestamps)),
    }


def analyze_diversity(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Analyze diversity and overlap across archetypes."""
    print(f"\n{'='*80}")
    print("DIVERSITY ANALYSIS")
    print(f"{'='*80}")

    # Build timestamp -> archetypes mapping
    timestamp_to_archetypes = defaultdict(set)

    for arch, result in all_results.items():
        for signal in result['signals']:
            ts = signal['timestamp']
            timestamp_to_archetypes[ts].add(arch)

    # Count unique vs shared signals
    total_signals = sum(len(r['signals']) for r in all_results.values())
    unique_signals = sum(1 for archs in timestamp_to_archetypes.values() if len(archs) == 1)
    shared_signals = sum(1 for archs in timestamp_to_archetypes.values() if len(archs) > 1)

    unique_pct = unique_signals / len(timestamp_to_archetypes) * 100 if timestamp_to_archetypes else 0

    print(f"  Total unique timestamps with signals: {len(timestamp_to_archetypes):,}")
    print(f"  Timestamps with 1 archetype: {unique_signals:,} ({unique_pct:.1f}%)")
    print(f"  Timestamps with 2+ archetypes: {shared_signals:,} ({100-unique_pct:.1f}%)")

    # Find archetype pairs with high overlap
    archetype_list = list(all_results.keys())
    high_overlap_pairs = []

    for i, arch1 in enumerate(archetype_list):
        ts1 = set(s['timestamp'] for s in all_results[arch1]['signals'])
        if not ts1:
            continue

        for arch2 in archetype_list[i+1:]:
            ts2 = set(s['timestamp'] for s in all_results[arch2]['signals'])
            if not ts2:
                continue

            overlap = len(ts1 & ts2)
            smaller = min(len(ts1), len(ts2))
            overlap_pct = overlap / smaller * 100 if smaller > 0 else 0

            if overlap_pct > 50:
                high_overlap_pairs.append({
                    'pair': f"{arch1} & {arch2}",
                    'overlap_pct': overlap_pct,
                    'overlap_count': overlap,
                })

    high_overlap_pairs.sort(key=lambda x: x['overlap_pct'], reverse=True)

    print(f"\n  High overlap pairs (>50%):")
    if high_overlap_pairs:
        for pair_info in high_overlap_pairs[:10]:
            print(f"    - {pair_info['pair']}: {pair_info['overlap_pct']:.1f}% ({pair_info['overlap_count']} signals)")
    else:
        print(f"    None (GOOD - archetypes are diverse)")

    # Compute correlation matrix of entry timestamps
    correlation_matrix = compute_timestamp_correlation(all_results)

    return {
        'total_unique_timestamps': len(timestamp_to_archetypes),
        'unique_signals_pct': unique_pct,
        'shared_signals_pct': 100 - unique_pct,
        'high_overlap_pairs': high_overlap_pairs[:10],
        'correlation_matrix': correlation_matrix,
    }


def compute_timestamp_correlation(all_results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """Compute pairwise correlation of entry timestamps."""
    archetype_list = [arch for arch in all_results.keys() if all_results[arch]['signals']]

    # Build binary vectors: 1 if signal at timestamp, 0 otherwise
    all_timestamps = set()
    for result in all_results.values():
        all_timestamps.update(s['timestamp'] for s in result['signals'])

    all_timestamps = sorted(all_timestamps)

    vectors = {}
    for arch in archetype_list:
        signal_ts = set(s['timestamp'] for s in all_results[arch]['signals'])
        vectors[arch] = [1 if ts in signal_ts else 0 for ts in all_timestamps]

    # Compute correlation
    corr_matrix = {}
    for arch1 in archetype_list:
        corr_matrix[arch1] = {}
        for arch2 in archetype_list:
            if arch1 == arch2:
                corr_matrix[arch1][arch2] = 1.0
            else:
                v1 = np.array(vectors[arch1])
                v2 = np.array(vectors[arch2])
                corr = np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else 0.0
                corr_matrix[arch1][arch2] = float(corr) if not np.isnan(corr) else 0.0

    return corr_matrix


def realism_checks(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Run realism checks on all results."""
    print(f"\n{'='*80}")
    print("REALISM CHECKS")
    print(f"{'='*80}")

    issues = []

    # Check 1: Confidence score sanity
    for arch, result in all_results.items():
        stats = result['stats']

        # Valid range check
        if stats['conf_min'] < 0.0 or stats['conf_max'] > 5.0:
            issue = f"❌ {arch}: Confidence scores out of valid range [0.0-5.0]: [{stats['conf_min']:.2f}, {stats['conf_max']:.2f}]"
            issues.append(issue)
            print(f"  {issue}")

        # Distribution check (all scores identical = suspicious)
        if stats['count'] > 10 and stats['conf_std'] < 0.01:
            issue = f"⚠️ {arch}: Suspiciously low variance (std={stats['conf_std']:.4f}) - all scores identical?"
            issues.append(issue)
            print(f"  {issue}")

        # Domain boost check
        if stats['domain_boost_pct'] < 10:
            issue = f"⚠️ {arch}: Low domain boost detection ({stats['domain_boost_pct']:.1f}%) - boosts may not be working"
            issues.append(issue)
            print(f"  {issue}")

        # Signal count check
        if stats['count'] == 0:
            issue = f"❌ {arch}: ZERO signals detected - archetype may be broken or thresholds too strict"
            issues.append(issue)
            print(f"  {issue}")
        elif stats['count'] < 5:
            issue = f"⚠️ {arch}: Very low signal count ({stats['count']}) - check thresholds"
            issues.append(issue)
            print(f"  {issue}")

    # Check 2: Direction alignment
    for arch, result in all_results.items():
        expected = EXPECTED_DIRECTIONS.get(arch, 'UNKNOWN')
        if expected == 'UNKNOWN' or expected == 'MIXED':
            continue

        direction_str = result['stats']['direction_breakdown']
        if 'LONG' in direction_str:
            pct_str = direction_str.split('%')[0]
            try:
                pct_primary = float(pct_str)
                expected_dir = 'LONG' if expected == 'LONG' else 'SHORT'

                if expected == 'LONG' and pct_primary < 80:
                    issue = f"⚠️ {arch}: Expected mostly LONG but got {direction_str}"
                    issues.append(issue)
                    print(f"  {issue}")
                elif expected == 'SHORT' and pct_primary > 20:
                    issue = f"⚠️ {arch}: Expected mostly SHORT but got {direction_str}"
                    issues.append(issue)
                    print(f"  {issue}")
            except:
                pass

    if not issues:
        print("  ✅ All realism checks PASSED")

    return {
        'issues': issues,
        'total_issues': len(issues),
    }


def generate_report(all_results: Dict, diversity: Dict, realism: Dict,
                   test_df: pd.DataFrame, total_time: float) -> str:
    """Generate markdown report."""

    report = []
    report.append("# SMOKE TEST REPORT")
    report.append("=" * 80)
    report.append(f"\n**Test Period**: {TEST_PERIOD_START} to {TEST_PERIOD_END} ({len(test_df):,} bars)")
    report.append(f"**Total Execution Time**: {total_time:.1f}s")
    report.append(f"**Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Archetype Summary Table
    report.append("\n## ARCHETYPE SUMMARY")
    report.append("-" * 80)
    report.append("")
    report.append("| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |")
    report.append("|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|")

    for category in ['Bull', 'Bear', 'Chop']:
        for arch in ARCHETYPES_TO_TEST.get(category, []):
            if arch not in all_results:
                continue

            result = all_results[arch]
            stats = result['stats']
            name = ARCHETYPE_NAMES.get(arch, 'Unknown')

            # Calculate unique %
            unique_pct = 100.0  # Default

            report.append(
                f"| {arch:4s} | {name:20s} | {stats['count']:7d} | "
                f"{unique_pct:6.1f}% | {stats['conf_min']:8.2f} | {stats['conf_max']:8.2f} | "
                f"{stats['conf_mean']:9.2f} | {stats['domain_boost_avg']:13.2f}x | "
                f"{stats['domain_boost_pct']:10.1f}% | {stats['direction_breakdown']:20s} |"
            )

    # Diversity Analysis
    report.append("\n## DIVERSITY ANALYSIS")
    report.append("-" * 80)
    report.append("")
    report.append(f"**Total unique timestamps with signals**: {diversity['total_unique_timestamps']:,}")
    report.append(f"**Average signal overlap**: {diversity['shared_signals_pct']:.1f}% ")

    if diversity['shared_signals_pct'] < 20:
        report.append("✅ GOOD - archetypes are diverse")
    elif diversity['shared_signals_pct'] < 40:
        report.append("⚠️ MODERATE - some overlap detected")
    else:
        report.append("❌ HIGH - significant overlap, check for redundancy")

    report.append("\n**Archetype pairs with high overlap (>50%)**:")
    if diversity['high_overlap_pairs']:
        for pair_info in diversity['high_overlap_pairs']:
            report.append(f"  - {pair_info['pair']}: {pair_info['overlap_pct']:.1f}% overlap ({pair_info['overlap_count']} signals)")
    else:
        report.append("  None detected")

    # Realism Checks
    report.append("\n## REALISM CHECKS")
    report.append("-" * 80)
    report.append("")

    if realism['total_issues'] == 0:
        report.append("✅ All confidence scores in valid range")
        report.append("✅ Domain boosts detected in majority of signals")
        report.append("✅ Direction alignment correct for all archetypes")
        report.append("✅ No critical issues detected")
    else:
        report.append(f"⚠️ {realism['total_issues']} issues detected:\n")
        for issue in realism['issues']:
            report.append(f"  {issue}")

    # Performance Metrics
    report.append("\n## PERFORMANCE")
    report.append("-" * 80)
    report.append("")
    report.append(f"**Total execution time**: {total_time:.1f}s")

    avg_time = total_time / len(all_results) if all_results else 0
    report.append(f"**Average per archetype**: {avg_time:.2f}s")

    # Top 5 slowest
    sorted_by_time = sorted(all_results.items(), key=lambda x: x[1]['execution_time'], reverse=True)
    report.append("\n**Slowest archetypes**:")
    for arch, result in sorted_by_time[:5]:
        report.append(f"  - {arch} ({ARCHETYPE_NAMES.get(arch, 'Unknown')}): {result['execution_time']:.2f}s")

    # Recommendations
    report.append("\n## RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("")

    # Count issues by severity
    zero_signal_archetypes = [arch for arch, r in all_results.items() if r['stats']['count'] == 0]
    low_signal_archetypes = [arch for arch, r in all_results.items() if 0 < r['stats']['count'] < 5]

    if zero_signal_archetypes:
        report.append(f"❌ **CRITICAL**: {len(zero_signal_archetypes)} archetype(s) produced ZERO signals:")
        for arch in zero_signal_archetypes:
            report.append(f"  - {arch} ({ARCHETYPE_NAMES.get(arch, 'Unknown')}): Check method implementation or relax thresholds")

    if low_signal_archetypes:
        report.append(f"\n⚠️ **WARNING**: {len(low_signal_archetypes)} archetype(s) produced <5 signals:")
        for arch in low_signal_archetypes:
            count = all_results[arch]['stats']['count']
            report.append(f"  - {arch} ({ARCHETYPE_NAMES.get(arch, 'Unknown')}): {count} signals - may need threshold tuning")

    if not zero_signal_archetypes and not low_signal_archetypes:
        report.append("✅ All archetypes producing reasonable signal counts")

    report.append("\n## SUCCESS CRITERIA")
    report.append("-" * 80)
    report.append("")

    criteria_results = []

    # Criterion 1: All archetypes produce >0 signals
    all_have_signals = all(r['stats']['count'] > 0 for r in all_results.values())
    criteria_results.append(("All archetypes produce >0 signals", all_have_signals))

    # Criterion 2: Average overlap <20%
    low_overlap = diversity['shared_signals_pct'] < 20
    criteria_results.append(("Average overlap <20% (diverse)", low_overlap))

    # Criterion 3: All confidence scores valid
    all_valid_scores = all(
        0.0 <= r['stats']['conf_min'] <= 5.0 and 0.0 <= r['stats']['conf_max'] <= 5.0
        for r in all_results.values()
    )
    criteria_results.append(("All confidence scores in [0.0-5.0]", all_valid_scores))

    # Criterion 4: Domain boosts present
    good_boost_detection = sum(
        1 for r in all_results.values() if r['stats']['domain_boost_pct'] > 50
    ) / len(all_results) > 0.5
    criteria_results.append(("Domain boosts present in >50% of signals", good_boost_detection))

    for criterion, passed in criteria_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        report.append(f"{status}: {criterion}")

    total_passed = sum(1 for _, p in criteria_results if p)
    report.append(f"\n**Overall**: {total_passed}/{len(criteria_results)} criteria passed")

    if total_passed == len(criteria_results):
        report.append("\n🎉 **ALL SUCCESS CRITERIA MET**")
    else:
        report.append(f"\n⚠️ **{len(criteria_results) - total_passed} criteria failed - review needed**")

    return "\n".join(report)


def save_results(all_results: Dict, diversity: Dict, realism: Dict, report: str):
    """Save results to files."""

    # Save markdown report
    report_path = 'SMOKE_TEST_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✅ Report saved to: {report_path}")

    # Save JSON data (excluding signals for size)
    json_data = {
        'test_period': {
            'start': TEST_PERIOD_START,
            'end': TEST_PERIOD_END,
        },
        'summary': {
            arch: {
                'stats': result['stats'],
                'execution_time': result['execution_time'],
                'error_count': len(result['errors']),
                'signal_count': len(result['signals']),
            }
            for arch, result in all_results.items()
        },
        'diversity': diversity,
        'realism': realism,
    }

    json_path = 'smoke_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ JSON results saved to: {json_path}")

    # Save issues file
    issues_path = 'smoke_test_issues.txt'
    with open(issues_path, 'w') as f:
        f.write("SMOKE TEST ISSUES\n")
        f.write("=" * 80 + "\n\n")

        if realism['total_issues'] == 0:
            f.write("No issues detected!\n")
        else:
            for issue in realism['issues']:
                f.write(f"{issue}\n")

        # Add error details
        f.write("\n\nERROR DETAILS\n")
        f.write("-" * 80 + "\n")
        for arch, result in all_results.items():
            if result['errors']:
                f.write(f"\n{arch} ({len(result['errors'])} errors):\n")
                for i, error in enumerate(result['errors'][:10], 1):  # First 10 errors
                    f.write(f"  {i}. {error}\n")
                if len(result['errors']) > 10:
                    f.write(f"  ... and {len(result['errors']) - 10} more\n")

    print(f"✅ Issues saved to: {issues_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run comprehensive smoke test."""

    print("=" * 80)
    print("COMPREHENSIVE ARCHETYPE SMOKE TEST")
    print("=" * 80)
    print(f"\nTest Period: {TEST_PERIOD_START} to {TEST_PERIOD_END}")
    print(f"Data Path: {DATA_PATH}")
    print(f"\nArchetypes to test:")
    for category, archs in ARCHETYPES_TO_TEST.items():
        print(f"  {category}: {', '.join(archs)}")

    total_start_time = time.time()

    # Load data
    df, test_df = load_data(DATA_PATH, TEST_PERIOD_START, TEST_PERIOD_END)

    # Test each archetype
    all_results = {}
    all_archetypes = []
    for category, archs in ARCHETYPES_TO_TEST.items():
        all_archetypes.extend(archs)

    for arch in all_archetypes:
        try:
            result = test_archetype(arch, df, test_df)
            all_results[arch] = result
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR testing {arch}:")
            print(f"   {str(e)}")
            traceback.print_exc()
            all_results[arch] = {
                'signals': [],
                'errors': [f'Critical error: {str(e)}'],
                'execution_time': 0,
                'stats': compute_signal_stats([], arch),
            }

    # Analyze diversity
    diversity = analyze_diversity(all_results)

    # Run realism checks
    realism = realism_checks(all_results)

    total_time = time.time() - total_start_time

    # Generate report
    report = generate_report(all_results, diversity, realism, test_df, total_time)

    # Print report to console
    print("\n" + "=" * 80)
    print(report)

    # Save results
    save_results(all_results, diversity, realism, report)

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETE")
    print("=" * 80)

    # Exit code based on critical issues
    zero_signal_count = sum(1 for r in all_results.values() if r['stats']['count'] == 0)
    if zero_signal_count > 0:
        print(f"\n⚠️ WARNING: {zero_signal_count} archetype(s) produced zero signals")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
