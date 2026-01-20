#!/usr/bin/env python3
"""
Validate Regime Classifier Against Ground Truth

Compares regime classifier predictions against manually labeled ground truth
for 2020-2024 period to measure accuracy and identify systematic errors.

Usage:
    python bin/validate_regime_classifier.py
    python bin/validate_regime_classifier.py --model models/regime_classifier_gmm.pkl
    python bin/validate_regime_classifier.py --granularity monthly
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.context.regime_classifier import RegimeClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_ground_truth(gt_path: str) -> Dict:
    """Load ground truth regime labels from JSON"""
    with open(gt_path, 'r') as f:
        return json.load(f)


def load_macro_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Load macro features from cache for validation period"""
    # Try to load from features cache
    cache_dirs = [
        Path("data/features"),
        Path("data/cache"),
        Path("data/macro"),
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        # Look for macro feature files
        for file in cache_dir.glob("*macro*.parquet"):
            try:
                df = pd.read_parquet(file)
                logger.info(f"Loaded macro features from {file}")

                # Filter date range
                df.index = pd.to_datetime(df.index)
                df = df[(df.index >= start_date) & (df.index <= end_date)]

                return df
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")
                continue

    raise FileNotFoundError("Could not find macro features cache. Run feature extraction first.")


def align_predictions_to_ground_truth(
    predictions: pd.DataFrame,
    ground_truth: Dict,
    granularity: str = "monthly"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align regime classifier predictions with ground truth labels

    Args:
        predictions: DataFrame with regime predictions (hourly/daily)
        ground_truth: Ground truth dict with monthly/quarterly/yearly labels
        granularity: 'monthly', 'quarterly', or 'yearly'

    Returns:
        (aligned_predictions, comparison_df)
    """
    gt_labels = ground_truth[granularity]

    # Aggregate predictions to match granularity
    if granularity == "monthly":
        freq = "ME"
    elif granularity == "quarterly":
        freq = "QE"
    elif granularity == "yearly":
        freq = "YE"
    else:
        raise ValueError(f"Invalid granularity: {granularity}")

    # Get most common regime per period
    predictions['regime'] = predictions['regime'].fillna('neutral')
    aggregated = predictions.resample(freq)['regime'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'neutral')

    # Build comparison dataframe
    comparison = []
    for period_str, gt_regime in gt_labels.items():
        # Parse period string
        if granularity == "monthly":
            period_dt = pd.to_datetime(period_str + "-01")
        elif granularity == "quarterly":
            year, quarter = period_str.split("-Q")
            period_dt = pd.Period(f"{year}Q{quarter}").to_timestamp()
        elif granularity == "yearly":
            period_dt = pd.to_datetime(f"{period_str}-01-01")

        # Get prediction for this period
        pred_regime = aggregated.get(period_dt, 'neutral')

        comparison.append({
            'period': period_str,
            'ground_truth': gt_regime,
            'predicted': pred_regime,
            'match': gt_regime == pred_regime
        })

    comparison_df = pd.DataFrame(comparison)

    return aggregated, comparison_df


def compute_confusion_matrix(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Compute confusion matrix from comparison"""
    regimes = ['risk_on', 'neutral', 'risk_off', 'crisis']

    confusion = pd.DataFrame(0, index=regimes, columns=regimes)

    for _, row in comparison_df.iterrows():
        gt = row['ground_truth']
        pred = row['predicted']
        if gt in regimes and pred in regimes:
            confusion.loc[gt, pred] += 1

    return confusion


def print_validation_report(comparison_df: pd.DataFrame, confusion: pd.DataFrame):
    """Print detailed validation report"""
    print("\n" + "=" * 80)
    print("REGIME CLASSIFIER VALIDATION REPORT")
    print("=" * 80)

    # Overall accuracy
    accuracy = comparison_df['match'].mean()
    print(f"\nOverall Accuracy: {accuracy:.1%}")

    # Per-regime accuracy
    print("\n" + "-" * 80)
    print("PER-REGIME ACCURACY")
    print("-" * 80)

    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        regime_df = comparison_df[comparison_df['ground_truth'] == regime]
        if len(regime_df) > 0:
            regime_acc = regime_df['match'].mean()
            n_total = len(regime_df)
            n_correct = regime_df['match'].sum()
            print(f"{regime:12s}: {regime_acc:5.1%} ({n_correct}/{n_total})")

    # Confusion matrix
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print("\nRows = Ground Truth, Columns = Predicted\n")
    print(confusion.to_string())

    # Errors
    errors = comparison_df[~comparison_df['match']]
    if len(errors) > 0:
        print("\n" + "-" * 80)
        print(f"ERRORS ({len(errors)} periods)")
        print("-" * 80)
        for _, row in errors.iterrows():
            print(f"  {row['period']:15s}: {row['ground_truth']:12s} → {row['predicted']:12s}")

    # Key metrics
    print("\n" + "-" * 80)
    print("KEY METRICS")
    print("-" * 80)

    # Precision and recall for each regime
    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        tp = confusion.loc[regime, regime]
        fp = confusion[regime].sum() - tp
        fn = confusion.loc[regime].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{regime.upper():12s}")
        print(f"  Precision: {precision:5.1%}")
        print(f"  Recall:    {recall:5.1%}")
        print(f"  F1 Score:  {f1:5.1%}")


def main():
    parser = argparse.ArgumentParser(description="Validate regime classifier against ground truth")
    parser.add_argument("--model", default="models/regime_classifier_gmm.pkl", help="Path to trained regime classifier")
    parser.add_argument("--ground-truth", default="data/regime_ground_truth_2020_2024.json", help="Path to ground truth labels")
    parser.add_argument("--granularity", choices=["monthly", "quarterly", "yearly"], default="monthly", help="Validation granularity")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date for validation")
    parser.add_argument("--end-date", default="2024-12-31", help="End date for validation")

    args = parser.parse_args()

    # Load ground truth
    logger.info(f"Loading ground truth from {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)

    # Load macro features
    logger.info(f"Loading macro features from {args.start_date} to {args.end_date}")
    try:
        macro_df = load_macro_features(args.start_date, args.end_date)
        logger.info(f"Loaded {len(macro_df)} rows of macro features")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("\nTo generate macro features, run:")
        logger.info("  python bin/cache_features.py --asset BTC --start 2020-01-01 --end 2024-12-31")
        return 1

    # Load regime classifier
    logger.info(f"Loading regime classifier from {args.model}")
    feature_order = ["VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                     "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                     "funding", "oi", "rv_20d", "rv_60d"]

    try:
        classifier = RegimeClassifier.load(args.model, feature_order, zero_fill_missing=True)
    except FileNotFoundError:
        logger.error(f"Model not found: {args.model}")
        logger.info("\nTo train a regime classifier, run:")
        logger.info("  python bin/train_regime_gmm.py")
        return 1

    # Run predictions
    logger.info("Running regime classification on macro features...")
    predictions = classifier.classify_series(macro_df)

    # Align with ground truth
    logger.info(f"Aligning predictions to {args.granularity} ground truth...")
    aggregated, comparison = align_predictions_to_ground_truth(
        predictions,
        ground_truth,
        args.granularity
    )

    # Compute confusion matrix
    confusion = compute_confusion_matrix(comparison)

    # Print report
    print_validation_report(comparison, confusion)

    # Save results
    output_path = f"results/regime_validation_{args.granularity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    Path("results").mkdir(exist_ok=True)
    comparison.to_csv(output_path, index=False)
    logger.info(f"\n✓ Validation results saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
