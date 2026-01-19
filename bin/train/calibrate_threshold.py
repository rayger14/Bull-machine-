#!/usr/bin/env python3
"""
Threshold Calibration Tool

Recalibrates decision threshold for a trained ML model on new data.
Keeps model weights fixed (trained on historical regime with sufficient samples).
Only adjusts the probability threshold for optimal performance on target regime.

Usage:
    python3 bin/train/calibrate_threshold.py \
        --model models/btc_trade_quality_filter_v1.pkl \
        --data reports/ml/btc_trades_2024_full.csv \
        --split-date 2024-10-01 \
        --output models/btc_ml_threshold_2024.json \
        --target-metric f1
"""

import pandas as pd
import joblib
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)


def load_model(model_path: str):
    """Load trained ML model"""
    print(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)

    model = model_data['model']
    feature_names = model_data['feature_names']
    original_threshold = model_data.get('threshold', 0.707)

    print(f"  Features: {len(feature_names)}")
    print(f"  Original threshold: {original_threshold:.3f}")

    return model, feature_names, original_threshold


def load_calibration_data(data_path: str, feature_names: list, split_date: str = None):
    """Load calibration data and extract features"""
    print(f"\nLoading calibration data from {data_path}")
    df = pd.read_csv(data_path)

    # Parse entry time
    df['entry_time'] = pd.to_datetime(df['entry_time'])

    # Time-based split if provided
    if split_date:
        split_dt = pd.to_datetime(split_date)
        if df['entry_time'].dt.tz is not None:
            split_dt = split_dt.tz_localize(df['entry_time'].dt.tz)

        cal_df = df[df['entry_time'] < split_dt].copy()
        test_df = df[df['entry_time'] >= split_dt].copy()

        print(f"\nCalibration set: {len(cal_df)} trades ({cal_df['entry_time'].min().date()} to {cal_df['entry_time'].max().date()})")
        print(f"Test set: {len(test_df)} trades ({test_df['entry_time'].min().date()} to {test_df['entry_time'].max().date()})")
    else:
        cal_df = df.copy()
        test_df = None
        print(f"\nCalibration set: {len(cal_df)} trades")

    # Target
    cal_df['target'] = (cal_df['r_multiple'] > 0).astype(int)
    print(f"Calibration win rate: {cal_df['target'].mean()*100:.1f}%")

    # Extract features
    available_features = [col for col in feature_names if col in cal_df.columns]
    if len(available_features) < len(feature_names):
        print(f"WARNING: {len(feature_names) - len(available_features)} features missing from data")

    X_cal = cal_df[available_features].fillna(0)
    y_cal = cal_df['target']

    if test_df is not None:
        test_df['target'] = (test_df['r_multiple'] > 0).astype(int)
        print(f"Test win rate: {test_df['target'].mean()*100:.1f}%")
        X_test = test_df[available_features].fillna(0)
        y_test = test_df['target']
        return X_cal, y_cal, X_test, y_test

    return X_cal, y_cal, None, None


def calibrate_threshold(model, X_cal, y_cal, target_metric='f1', min_precision=None, min_recall=None):
    """Find optimal threshold on calibration set"""
    print(f"\n{'='*60}")
    print(f"THRESHOLD CALIBRATION (Target: {target_metric})")
    print(f"{'='*60}")

    # Get predictions
    y_proba = model.predict_proba(X_cal)[:, 1]
    auc = roc_auc_score(y_cal, y_proba)

    print(f"Calibration AUC: {auc:.3f}")

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_cal, y_proba)

    # Find optimal threshold based on target metric
    if target_metric == 'f1':
        f1_scores = 2 * recall * precision / (recall + precision + 1e-8)

        # Apply constraints if specified
        valid_mask = np.ones(len(f1_scores), dtype=bool)
        if min_precision is not None:
            valid_mask &= (precision >= min_precision)
        if min_recall is not None:
            valid_mask &= (recall >= min_recall)

        if not valid_mask.any():
            print(f"WARNING: No thresholds satisfy constraints (min_prec={min_precision}, min_rec={min_recall})")
            print("Falling back to unconstrained F1 optimization")
            valid_mask = np.ones(len(f1_scores), dtype=bool)

        masked_f1 = np.where(valid_mask, f1_scores, -np.inf)
        ix = np.argmax(masked_f1)
        optimal_threshold = thresholds[ix] if ix < len(thresholds) else 0.65
        optimal_f1 = f1_scores[ix]

        print(f"\nOptimal threshold: {optimal_threshold:.3f}")
        print(f"  F1: {optimal_f1:.3f}")
        print(f"  Precision: {precision[ix]:.3f}")
        print(f"  Recall: {recall[ix]:.3f}")

        return optimal_threshold, {
            'auc': float(auc),
            'threshold': float(optimal_threshold),
            'f1': float(optimal_f1),
            'precision': float(precision[ix]),
            'recall': float(recall[ix]),
            'n_samples': int(len(X_cal)),
            'win_rate': float(y_cal.mean())
        }

    else:
        raise ValueError(f"Unknown target metric: {target_metric}")


def evaluate_threshold(model, X, y, threshold, split_name='Test'):
    """Evaluate model at given threshold"""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y, y_proba)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred)

    print(f"\n{split_name} Set Performance (threshold={threshold:.3f}):")
    print(f"  AUC: {auc:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")

    return {
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'n_samples': int(len(X)),
        'win_rate': float(y.mean())
    }


def main():
    parser = argparse.ArgumentParser(description='Calibrate ML model threshold for new regime')
    parser.add_argument('--model', required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--data', required=True, help='Path to calibration CSV')
    parser.add_argument('--split-date', type=str, default=None,
                        help='Time-based split: calibrate on data < split-date, test on data >= split-date')
    parser.add_argument('--output', required=True, help='Path to save calibrated threshold JSON')
    parser.add_argument('--target-metric', type=str, default='f1', choices=['f1'],
                        help='Metric to optimize')
    parser.add_argument('--min-precision', type=float, default=None,
                        help='Minimum precision constraint (e.g., 0.75)')
    parser.add_argument('--min-recall', type=float, default=None,
                        help='Minimum recall constraint (e.g., 0.50)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model, feature_names, original_threshold = load_model(args.model)

    # Load calibration data
    if args.split_date:
        X_cal, y_cal, X_test, y_test = load_calibration_data(
            args.data, feature_names, args.split_date
        )
    else:
        X_cal, y_cal, _, _ = load_calibration_data(args.data, feature_names)
        X_test = y_test = None

    # Calibrate threshold
    optimal_threshold, cal_metrics = calibrate_threshold(
        model, X_cal, y_cal,
        target_metric=args.target_metric,
        min_precision=args.min_precision,
        min_recall=args.min_recall
    )

    # Evaluate on test set if available
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_metrics = evaluate_threshold(model, X_test, y_test, optimal_threshold, 'Test')

    # Save calibrated threshold
    output_data = {
        'calibrated_threshold': float(optimal_threshold),
        'original_threshold': float(original_threshold),
        'calibration_metrics': cal_metrics,
        'test_metrics': test_metrics,
        'calibration_config': {
            'model_path': args.model,
            'data_path': args.data,
            'split_date': args.split_date,
            'target_metric': args.target_metric,
            'min_precision': args.min_precision,
            'min_recall': args.min_recall
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original threshold (2022-2023): {original_threshold:.3f}")
    print(f"Calibrated threshold (2024): {optimal_threshold:.3f}")
    print(f"Threshold shift: {optimal_threshold - original_threshold:+.3f}")
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
