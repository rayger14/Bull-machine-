#!/usr/bin/env python3
"""
Validate HMM Regime Classifier V2
==================================

Comprehensive validation framework for regime classifier:

1. Silhouette Score - Cluster quality (target: >0.5)
2. Transition Frequency - Regime stability (target: 10-20/year)
3. Event Accuracy - Known market events (target: >80%)
4. Archetype Alignment - Do archetypes fire in correct regimes?

Usage:
    python bin/validate_regime_hmm.py --labels data/regime_labels_v2.parquet

Outputs:
    - results/regime_hmm_validation_report.json
    - Console report with pass/fail for each metric
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime
from sklearn.metrics import silhouette_score, confusion_matrix
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.context.hmm_regime_model import REGIME_FEATURES_V2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Known crypto events (ground truth from research report)
KNOWN_EVENTS = {
    '2020-03-12': {'event': 'COVID crash', 'expected_regime': 'crisis'},
    '2022-05-09': {'event': 'LUNA collapse', 'expected_regime': 'crisis'},
    '2022-06-18': {'event': 'June 2022 bottom', 'expected_regime': 'crisis'},
    '2022-11-11': {'event': 'FTX collapse', 'expected_regime': 'crisis'},
    '2024-01-10': {'event': 'BTC ETF approval', 'expected_regime': 'risk_on'},
    '2024-08-05': {'event': 'Japan carry unwind', 'expected_regime': 'crisis'}
}


def validate_silhouette(df: pd.DataFrame) -> Dict:
    """
    Metric 1: Silhouette Score (cluster quality).

    Measures how well-separated regime clusters are.

    Target: >0.5 (reasonable clustering)

    Args:
        df: DataFrame with regime_label and features

    Returns:
        Dict with score and pass/fail
    """
    logger.info("\n" + "="*80)
    logger.info("METRIC 1: SILHOUETTE SCORE (Cluster Quality)")
    logger.info("="*80)

    try:
        # Extract features
        X = df[REGIME_FEATURES_V2].fillna(0).values

        # Map regime labels to integers
        label_map = {'risk_on': 0, 'neutral': 1, 'risk_off': 2, 'crisis': 3}
        labels = df['regime_label'].map(label_map).values

        # Compute silhouette score (sample for speed if large dataset)
        sample_size = min(10000, len(X))
        score = silhouette_score(X, labels, sample_size=sample_size)

        logger.info(f"\n   Silhouette Score: {score:.3f}")
        logger.info(f"   Target: >0.5")

        if score > 0.5:
            logger.info("   ✅ PASS - Good cluster separation")
            passed = True
        elif score > 0.3:
            logger.warning("   ⚠️  MARGINAL - Clusters overlap moderately")
            passed = False
        else:
            logger.error("   ❌ FAIL - Poor cluster separation")
            passed = False

        return {
            'score': float(score),
            'target': 0.5,
            'passed': passed,
            'interpretation': 'Good' if score > 0.5 else ('Marginal' if score > 0.3 else 'Poor')
        }

    except Exception as e:
        logger.error(f"   ❌ Error computing silhouette: {e}")
        return {'score': None, 'passed': False, 'error': str(e)}


def validate_transition_frequency(df: pd.DataFrame) -> Dict:
    """
    Metric 2: Transition Frequency (regime stability).

    Count regime changes per year.

    Target: 10-20 transitions/year
    - <10: Too coarse, missing regime shifts
    - 10-20: Optimal
    - >20: Thrashing, noisy

    Args:
        df: DataFrame with regime_label

    Returns:
        Dict with frequency and pass/fail
    """
    logger.info("\n" + "="*80)
    logger.info("METRIC 2: TRANSITION FREQUENCY (Regime Stability)")
    logger.info("="*80)

    # Count transitions
    transitions = (df['regime_label'] != df['regime_label'].shift(1)).sum()

    # Calculate years
    years = (df.index[-1] - df.index[0]).days / 365.25

    # Frequency
    freq = transitions / years

    logger.info(f"\n   Total transitions: {transitions}")
    logger.info(f"   Time span: {years:.2f} years")
    logger.info(f"   Transitions per year: {freq:.1f}")
    logger.info(f"   Target: 10-20 transitions/year")

    if 10 <= freq <= 20:
        logger.info("   ✅ PASS - Optimal regime stability")
        passed = True
        interpretation = 'Optimal'
    elif freq < 10:
        logger.warning("   ⚠️  Too stable - May miss intra-year regime shifts")
        passed = False
        interpretation = 'Too stable'
    else:
        logger.warning("   ⚠️  Too noisy - Regime thrashing detected")
        passed = False
        interpretation = 'Too noisy'

    # Transition matrix
    logger.info("\n   Regime transition matrix:")
    regimes = ['risk_on', 'neutral', 'risk_off', 'crisis']
    transition_matrix = pd.crosstab(
        df['regime_label'].shift(1),
        df['regime_label'],
        normalize='index'
    )

    logger.info(f"\n{transition_matrix.to_string()}")

    return {
        'transitions_per_year': float(freq),
        'total_transitions': int(transitions),
        'target_range': [10, 20],
        'passed': passed,
        'interpretation': interpretation,
        'transition_matrix': transition_matrix.to_dict()
    }


def validate_event_accuracy(df: pd.DataFrame) -> Dict:
    """
    Metric 3: Event Accuracy (known market events).

    Validate that classifier correctly identifies known market events.

    Target: >80% accuracy

    Args:
        df: DataFrame with regime_label

    Returns:
        Dict with accuracy and per-event results
    """
    logger.info("\n" + "="*80)
    logger.info("METRIC 3: EVENT ACCURACY (Known Market Events)")
    logger.info("="*80)

    results = []
    matched = 0
    total = 0

    for date_str, event_info in KNOWN_EVENTS.items():
        date = pd.to_datetime(date_str)

        # Check if date exists in dataset
        if date not in df.index:
            # Find nearest date
            nearest_idx = df.index.get_indexer([date], method='nearest')[0]
            actual_date = df.index[nearest_idx]
            days_diff = abs((actual_date - date).days)

            if days_diff > 7:
                logger.warning(f"   Skipping {event_info['event']}: Date {date_str} not found (nearest: {actual_date}, {days_diff} days away)")
                continue

            date = actual_date

        # Get detected regime
        detected_regime = df.loc[date, 'regime_label']
        confidence = df.loc[date, 'regime_confidence']
        expected_regime = event_info['expected_regime']

        # Check match
        match = (detected_regime == expected_regime)
        if match:
            matched += 1
        total += 1

        logger.info(f"\n   {date_str} - {event_info['event']}")
        logger.info(f"     Expected: {expected_regime}")
        logger.info(f"     Detected: {detected_regime}")
        logger.info(f"     Confidence: {confidence:.1%}")
        logger.info(f"     {'✅ MATCH' if match else '❌ MISS'}")

        results.append({
            'date': date_str,
            'event': event_info['event'],
            'expected': expected_regime,
            'detected': detected_regime,
            'confidence': float(confidence),
            'match': match
        })

    # Calculate accuracy
    accuracy = matched / total if total > 0 else 0.0

    logger.info(f"\n   Event Accuracy: {accuracy:.1%} ({matched}/{total})")
    logger.info(f"   Target: >80%")

    if accuracy >= 0.8:
        logger.info("   ✅ PASS - Accurately detects known events")
        passed = True
    elif accuracy >= 0.6:
        logger.warning("   ⚠️  MARGINAL - Some events misclassified")
        passed = False
    else:
        logger.error("   ❌ FAIL - Poor event detection")
        passed = False

    return {
        'accuracy': float(accuracy),
        'matched': matched,
        'total': total,
        'target': 0.8,
        'passed': passed,
        'events': results
    }


def validate_regime_duration(df: pd.DataFrame) -> Dict:
    """
    Metric 4: Regime Duration Statistics.

    Analyze how long each regime persists.

    Expected:
    - risk_on: 20-60 days (long bull trends)
    - neutral: 10-30 days (chop periods)
    - risk_off: 15-45 days (bear markets)
    - crisis: 3-14 days (short, sharp events)

    Args:
        df: DataFrame with regime_label

    Returns:
        Dict with duration stats per regime
    """
    logger.info("\n" + "="*80)
    logger.info("METRIC 4: REGIME DURATION STATISTICS")
    logger.info("="*80)

    # Calculate regime runs
    regime_runs = []
    current_regime = None
    run_start = None

    for idx, regime in zip(df.index, df['regime_label']):
        if regime != current_regime:
            if current_regime is not None:
                duration_hours = (idx - run_start).total_seconds() / 3600
                duration_days = duration_hours / 24
                regime_runs.append({
                    'regime': current_regime,
                    'start': run_start,
                    'end': idx,
                    'duration_hours': duration_hours,
                    'duration_days': duration_days
                })
            current_regime = regime
            run_start = idx

    # Add final run
    if current_regime is not None:
        duration_hours = (df.index[-1] - run_start).total_seconds() / 3600
        duration_days = duration_hours / 24
        regime_runs.append({
            'regime': current_regime,
            'start': run_start,
            'end': df.index[-1],
            'duration_hours': duration_hours,
            'duration_days': duration_days
        })

    runs_df = pd.DataFrame(regime_runs)

    # Aggregate stats
    duration_stats = runs_df.groupby('regime')['duration_days'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])

    logger.info("\n   Regime Duration Statistics (days):")
    logger.info(f"\n{duration_stats.to_string()}")

    # Check against expected ranges
    expectations = {
        'risk_on': (20, 60),
        'neutral': (10, 30),
        'risk_off': (15, 45),
        'crisis': (3, 14)
    }

    results = {}
    for regime, (min_exp, max_exp) in expectations.items():
        if regime in duration_stats.index:
            mean_duration = duration_stats.loc[regime, 'mean']
            in_range = min_exp <= mean_duration <= max_exp

            logger.info(f"\n   {regime}:")
            logger.info(f"     Mean duration: {mean_duration:.1f} days")
            logger.info(f"     Expected: {min_exp}-{max_exp} days")
            logger.info(f"     {'✅ In range' if in_range else '⚠️  Outside expected range'}")

            results[regime] = {
                'mean_duration': float(mean_duration),
                'expected_range': [min_exp, max_exp],
                'in_range': in_range,
                'count': int(duration_stats.loc[regime, 'count'])
            }

    return results


def validate_regime_distribution(df: pd.DataFrame) -> Dict:
    """
    Metric 5: Regime Distribution.

    Check if regime distribution matches expected proportions.

    Expected (from research report):
    - risk_on: 35-45%
    - neutral: 30-40%
    - risk_off: 20-25%
    - crisis: 5-10%

    Args:
        df: DataFrame with regime_label

    Returns:
        Dict with distribution stats
    """
    logger.info("\n" + "="*80)
    logger.info("METRIC 5: REGIME DISTRIBUTION")
    logger.info("="*80)

    # Calculate distribution
    regime_counts = df['regime_label'].value_counts()
    regime_pcts = regime_counts / len(df) * 100

    logger.info("\n   Regime Distribution:")

    expectations = {
        'risk_on': (35, 45),
        'neutral': (30, 40),
        'risk_off': (20, 25),
        'crisis': (5, 10)
    }

    results = {}
    all_in_range = True

    for regime, (min_exp, max_exp) in expectations.items():
        if regime in regime_pcts:
            pct = regime_pcts[regime]
            in_range = min_exp <= pct <= max_exp

            logger.info(f"\n   {regime:12s}: {pct:5.1f}% (expected: {min_exp}-{max_exp}%)")
            logger.info(f"                {'✅ In range' if in_range else '⚠️  Outside expected range'}")

            results[regime] = {
                'percentage': float(pct),
                'expected_range': [min_exp, max_exp],
                'in_range': in_range
            }

            if not in_range:
                all_in_range = False
        else:
            logger.warning(f"\n   {regime:12s}: 0.0% (MISSING!)")
            results[regime] = {'percentage': 0.0, 'in_range': False}
            all_in_range = False

    results['all_in_range'] = all_in_range

    if all_in_range:
        logger.info("\n   ✅ All regimes within expected distribution")
    else:
        logger.warning("\n   ⚠️  Some regimes outside expected distribution")

    return results


def generate_validation_report(metrics: Dict) -> None:
    """
    Generate comprehensive validation report.

    Args:
        metrics: Dict of all validation metrics
    """
    print("\n" + "="*80)
    print("VALIDATION REPORT SUMMARY")
    print("="*80)

    # Overall pass/fail
    passed_metrics = []
    failed_metrics = []

    if metrics['silhouette'].get('passed'):
        passed_metrics.append('Silhouette Score')
    else:
        failed_metrics.append('Silhouette Score')

    if metrics['transition_frequency'].get('passed'):
        passed_metrics.append('Transition Frequency')
    else:
        failed_metrics.append('Transition Frequency')

    if metrics['event_accuracy'].get('passed'):
        passed_metrics.append('Event Accuracy')
    else:
        failed_metrics.append('Event Accuracy')

    print(f"\n✅ PASSED: {len(passed_metrics)}/3 core metrics")
    for metric in passed_metrics:
        print(f"   - {metric}")

    if failed_metrics:
        print(f"\n❌ FAILED: {len(failed_metrics)}/3 core metrics")
        for metric in failed_metrics:
            print(f"   - {metric}")

    # Key numbers
    print("\n📊 Key Metrics:")
    print(f"   Silhouette Score:      {metrics['silhouette'].get('score', 'N/A'):.3f} (target: >0.5)")
    print(f"   Transitions/Year:      {metrics['transition_frequency'].get('transitions_per_year', 'N/A'):.1f} (target: 10-20)")
    print(f"   Event Accuracy:        {metrics['event_accuracy'].get('accuracy', 'N/A'):.1%} (target: >80%)")

    # Overall verdict
    if len(passed_metrics) == 3:
        print("\n" + "="*80)
        print("🎉 VALIDATION PASSED - HMM Classifier is production-ready!")
        print("="*80)
    elif len(passed_metrics) >= 2:
        print("\n" + "="*80)
        print("⚠️  VALIDATION MARGINAL - Consider re-training or tuning")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ VALIDATION FAILED - Major issues detected, re-training required")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Validate HMM Regime Classifier V2')
    parser.add_argument(
        '--labels',
        type=str,
        default='data/regime_labels_v2.parquet',
        help='Path to regime labels parquet file'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet',
        help='Path to feature store with raw features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/regime_hmm_validation_report.json',
        help='Path to save validation report'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("HMM REGIME CLASSIFIER V2 - VALIDATION")
    print("="*80)

    # Load data
    logger.info(f"\nLoading regime labels from {args.labels}...")
    labels_path = Path(args.labels)

    if not labels_path.exists():
        logger.error(f"❌ Labels file not found: {labels_path}")
        logger.error("   Run train_regime_hmm_v2.py first to generate labels")
        return 1

    df_labels = pd.read_parquet(labels_path)
    logger.info(f"   Loaded {len(df_labels):,} labels")

    # Load features
    logger.info(f"\nLoading features from {args.features}...")
    features_path = Path(args.features)

    if not features_path.exists():
        logger.error(f"❌ Features file not found: {features_path}")
        return 1

    df_features = pd.read_parquet(features_path)
    logger.info(f"   Loaded {len(df_features):,} feature rows")

    # Merge
    logger.info("\nMerging labels and features...")
    df = df_features.join(df_labels, how='inner')
    logger.info(f"   Merged dataset: {len(df):,} rows")

    # Check for regime_label column
    if 'regime_label' not in df.columns:
        logger.error("❌ No regime_label column found!")
        return 1

    # Run validation metrics
    metrics = {}

    metrics['silhouette'] = validate_silhouette(df)
    metrics['transition_frequency'] = validate_transition_frequency(df)
    metrics['event_accuracy'] = validate_event_accuracy(df)
    metrics['regime_duration'] = validate_regime_duration(df)
    metrics['regime_distribution'] = validate_regime_distribution(df)

    # Generate report
    generate_validation_report(metrics)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare JSON-serializable version
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'labels_path': str(labels_path),
            'features_path': str(features_path),
            'n_bars': len(df),
            'date_range': [str(df.index[0]), str(df.index[-1])]
        },
        'metrics': metrics
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n✅ Validation report saved: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
