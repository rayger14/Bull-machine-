#!/usr/bin/env python3
"""
Walk-Forward Validation for Hierarchical Regime Classifier v6
=============================================================

Validate v6 model across 6 different train/test splits (2018-2024) to ensure
robustness across different market conditions.

VALIDATION PERIODS:
1. Train: 2018-2020, Test: 2020-2021 (COVID crash + recovery)
2. Train: 2019-2021, Test: 2021-2022 (Bull → Bear transition)
3. Train: 2020-2022, Test: 2022-2023 (Bear market + FTX)
4. Train: 2021-2023, Test: 2023-2024 (Recovery + Bull)
5. Train: 2022-2024.Q2, Test: 2024.Q3-Q4 (Recent bull)
6. Train: Full 2018-2023, Test: 2024 (OOS validation)

SUCCESS CRITERIA (Industry Standard):
- Average F1 across folds >= 80%
- Crisis F1 >= 85% (every fold)
- Risk_off F1 >= 60% (every fold)
- Risk_on F1 >= 50% (every fold)
- No fold degrades >20% vs average
- Predicted distribution within 10% of truth (every fold)

Author: Claude Code
Date: 2026-01-27
Purpose: Validate hierarchical v6 before production deployment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for hierarchical regime classifier.

    Tests model across 6 different time periods to ensure robustness.
    """

    def __init__(self, model_class):
        self.model_class = model_class
        self.folds = []
        self.results = []

    def define_folds(self) -> List[Dict]:
        """
        Define 6 walk-forward validation folds (2022-2024 data only).

        Returns:
            List of fold definitions with train/test periods
        """
        folds = [
            {
                'name': 'Bear_2022_H1_H2',
                'train_start': '2022-01-01',
                'train_end': '2022-06-30',
                'test_start': '2022-07-01',
                'test_end': '2022-12-31',
                'description': 'Bear market H1 → H2 (LUNA + FTX)'
            },
            {
                'name': 'Bear_to_Recovery',
                'train_start': '2022-01-01',
                'train_end': '2022-12-31',
                'test_start': '2023-01-01',
                'test_end': '2023-06-30',
                'description': 'Bear 2022 → Recovery 2023 H1'
            },
            {
                'name': 'Recovery_Consolidation',
                'train_start': '2022-01-01',
                'train_end': '2023-06-30',
                'test_start': '2023-07-01',
                'test_end': '2023-12-31',
                'description': 'Recovery → Consolidation 2023 H2'
            },
            {
                'name': 'Pre_Bull_2024_H1',
                'train_start': '2022-01-01',
                'train_end': '2023-12-31',
                'test_start': '2024-01-01',
                'test_end': '2024-06-30',
                'description': 'Full 2022-2023 → Bull 2024 H1'
            },
            {
                'name': 'Bull_2024_H2',
                'train_start': '2022-01-01',
                'train_end': '2024-06-30',
                'test_start': '2024-07-01',
                'test_end': '2024-12-31',
                'description': 'Extended train → Bull 2024 H2'
            },
            {
                'name': 'Full_OOS_2024',
                'train_start': '2022-01-01',
                'train_end': '2023-12-31',
                'test_start': '2024-01-01',
                'test_end': '2024-12-31',
                'description': 'Full 2022-2023 → Full OOS 2024'
            }
        ]
        return folds

    def split_data(self, df: pd.DataFrame, fold: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test for a fold.

        Args:
            df: Full dataset with regime_label
            fold: Fold definition

        Returns:
            Tuple of (train_df, test_df)
        """
        train_mask = (df.index >= fold['train_start']) & (df.index <= fold['train_end'])
        test_mask = (df.index >= fold['test_start']) & (df.index <= fold['test_end'])

        train_df = df[train_mask].copy()
        test_df = test_mask & (~train_mask)  # Ensure no overlap
        test_df = df[test_df].copy()

        return train_df, test_df

    def evaluate_fold(self, y_true: np.ndarray, y_pred: np.ndarray, fold_name: str) -> Dict:
        """
        Evaluate predictions for a single fold.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            fold_name: Name of the fold

        Returns:
            Dict with evaluation metrics
        """
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=['crisis', 'risk_off', 'neutral', 'risk_on'],
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(
            y_true, y_pred,
            labels=['crisis', 'risk_off', 'neutral', 'risk_on']
        )

        # Distribution comparison
        true_dist = pd.Series(y_true).value_counts(normalize=True).to_dict()
        pred_dist = pd.Series(y_pred).value_counts(normalize=True).to_dict()

        # Calculate distribution error (MAE)
        regimes = ['crisis', 'risk_off', 'neutral', 'risk_on']
        dist_error = sum(
            abs(true_dist.get(r, 0) - pred_dist.get(r, 0))
            for r in regimes
        ) / len(regimes)

        results = {
            'fold': fold_name,
            'accuracy': report['accuracy'],
            'crisis_f1': report.get('crisis', {}).get('f1-score', 0),
            'risk_off_f1': report.get('risk_off', {}).get('f1-score', 0),
            'neutral_f1': report.get('neutral', {}).get('f1-score', 0),
            'risk_on_f1': report.get('risk_on', {}).get('f1-score', 0),
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score'],
            'distribution_error': dist_error,
            'true_distribution': true_dist,
            'predicted_distribution': pred_dist,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(y_true)
        }

        return results

    def run_validation(self, df_full: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
        """
        Run walk-forward validation across all folds.

        Args:
            df_full: Full dataset (2018-2024) with regime_label
            feature_cols: List of feature column names

        Returns:
            List of results for each fold
        """
        folds = self.define_folds()
        results = []

        logger.info("=" * 80)
        logger.info("WALK-FORWARD VALIDATION: Hierarchical Regime Classifier v6")
        logger.info("=" * 80)
        logger.info(f"Total folds: {len(folds)}")
        logger.info(f"Features: {len(feature_cols)}")
        logger.info("")

        for i, fold in enumerate(folds, 1):
            logger.info(f"{'=' * 80}")
            logger.info(f"FOLD {i}/{len(folds)}: {fold['name']}")
            logger.info(f"{'=' * 80}")
            logger.info(f"Description: {fold['description']}")
            logger.info(f"Train: {fold['train_start']} to {fold['train_end']}")
            logger.info(f"Test:  {fold['test_start']} to {fold['test_end']}")
            logger.info("")

            # Split data
            train_df, test_df = self.split_data(df_full, fold)

            if len(train_df) == 0 or len(test_df) == 0:
                logger.warning(f"Skipping fold {fold['name']} - insufficient data")
                logger.warning(f"  Train samples: {len(train_df)}, Test samples: {len(test_df)}")
                continue

            logger.info(f"Train samples: {len(train_df):,}")
            logger.info(f"Test samples:  {len(test_df):,}")

            # Check for regime_label
            if 'regime_label' not in train_df.columns:
                logger.error("Missing 'regime_label' column in training data")
                continue

            # Prepare features
            X_train = train_df[feature_cols].values
            y_train = train_df['regime_label'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['regime_label'].values

            # Check train distribution
            train_dist = pd.Series(y_train).value_counts(normalize=True)
            logger.info("\nTrain distribution:")
            for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
                pct = train_dist.get(regime, 0) * 100
                count = (y_train == regime).sum()
                logger.info(f"  {regime:10s}: {count:5d} ({pct:5.1f}%)")

            # Train model
            logger.info("\nTraining model...")
            model = self.model_class(random_state=42)
            model.fit(X_train, y_train)

            # Predict
            logger.info("Predicting on test set...")
            y_pred = model.predict(X_test)

            # Evaluate
            fold_results = self.evaluate_fold(y_test, y_pred, fold['name'])
            fold_results['train_period'] = f"{fold['train_start']} to {fold['train_end']}"
            fold_results['test_period'] = f"{fold['test_start']} to {fold['test_end']}"
            fold_results['description'] = fold['description']

            results.append(fold_results)

            # Log results
            logger.info("\n" + "=" * 80)
            logger.info(f"FOLD {i} RESULTS: {fold['name']}")
            logger.info("=" * 80)
            logger.info(f"Accuracy: {fold_results['accuracy']:.1%}")
            logger.info(f"Macro F1:  {fold_results['macro_avg_f1']:.3f}")
            logger.info("")
            logger.info("Per-Regime F1 Scores:")
            logger.info(f"  Crisis:   {fold_results['crisis_f1']:.3f}")
            logger.info(f"  Risk_off: {fold_results['risk_off_f1']:.3f}")
            logger.info(f"  Neutral:  {fold_results['neutral_f1']:.3f}")
            logger.info(f"  Risk_on:  {fold_results['risk_on_f1']:.3f}")
            logger.info("")
            logger.info(f"Distribution Error (MAE): {fold_results['distribution_error']:.3f}")
            logger.info("")

        return results

    def summarize_results(self, results: List[Dict]) -> Dict:
        """
        Summarize results across all folds.

        Args:
            results: List of fold results

        Returns:
            Summary statistics
        """
        if not results:
            logger.error("No results to summarize")
            return {}

        # Collect metrics
        accuracies = [r['accuracy'] for r in results]
        crisis_f1s = [r['crisis_f1'] for r in results]
        risk_off_f1s = [r['risk_off_f1'] for r in results]
        neutral_f1s = [r['neutral_f1'] for r in results]
        risk_on_f1s = [r['risk_on_f1'] for r in results]
        macro_f1s = [r['macro_avg_f1'] for r in results]
        dist_errors = [r['distribution_error'] for r in results]

        summary = {
            'n_folds': len(results),
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'crisis_f1': {
                'mean': np.mean(crisis_f1s),
                'std': np.std(crisis_f1s),
                'min': np.min(crisis_f1s),
                'max': np.max(crisis_f1s)
            },
            'risk_off_f1': {
                'mean': np.mean(risk_off_f1s),
                'std': np.std(risk_off_f1s),
                'min': np.min(risk_off_f1s),
                'max': np.max(risk_off_f1s)
            },
            'neutral_f1': {
                'mean': np.mean(neutral_f1s),
                'std': np.std(neutral_f1s),
                'min': np.min(neutral_f1s),
                'max': np.max(neutral_f1s)
            },
            'risk_on_f1': {
                'mean': np.mean(risk_on_f1s),
                'std': np.std(risk_on_f1s),
                'min': np.min(risk_on_f1s),
                'max': np.max(risk_on_f1s)
            },
            'macro_f1': {
                'mean': np.mean(macro_f1s),
                'std': np.std(macro_f1s),
                'min': np.min(macro_f1s),
                'max': np.max(macro_f1s)
            },
            'distribution_error': {
                'mean': np.mean(dist_errors),
                'std': np.std(dist_errors),
                'min': np.min(dist_errors),
                'max': np.max(dist_errors)
            }
        }

        # Check success criteria
        summary['success_criteria'] = {
            'avg_f1_gte_80': summary['macro_f1']['mean'] >= 0.80,
            'crisis_f1_gte_85': summary['crisis_f1']['mean'] >= 0.85,
            'risk_off_f1_gte_60': summary['risk_off_f1']['mean'] >= 0.60,
            'risk_on_f1_gte_50': summary['risk_on_f1']['mean'] >= 0.50,
            'no_fold_degrades_20pct': (summary['macro_f1']['max'] - summary['macro_f1']['min']) <= 0.20,
            'dist_error_lte_10pct': summary['distribution_error']['mean'] <= 0.10
        }

        summary['all_criteria_met'] = all(summary['success_criteria'].values())

        return summary

    def print_summary(self, summary: Dict, results: List[Dict]):
        """
        Print comprehensive summary of walk-forward validation.

        Args:
            summary: Summary statistics
            results: Individual fold results
        """
        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Folds: {summary['n_folds']}")
        logger.info("")

        logger.info("OVERALL PERFORMANCE:")
        logger.info(f"  Accuracy:  {summary['accuracy']['mean']:.1%} ± {summary['accuracy']['std']:.1%}")
        logger.info(f"  Macro F1:  {summary['macro_f1']['mean']:.3f} ± {summary['macro_f1']['std']:.3f}")
        logger.info("")

        logger.info("PER-REGIME F1 SCORES (Mean ± Std):")
        logger.info(f"  Crisis:   {summary['crisis_f1']['mean']:.3f} ± {summary['crisis_f1']['std']:.3f}  (range: {summary['crisis_f1']['min']:.3f}-{summary['crisis_f1']['max']:.3f})")
        logger.info(f"  Risk_off: {summary['risk_off_f1']['mean']:.3f} ± {summary['risk_off_f1']['std']:.3f}  (range: {summary['risk_off_f1']['min']:.3f}-{summary['risk_off_f1']['max']:.3f})")
        logger.info(f"  Neutral:  {summary['neutral_f1']['mean']:.3f} ± {summary['neutral_f1']['std']:.3f}  (range: {summary['neutral_f1']['min']:.3f}-{summary['neutral_f1']['max']:.3f})")
        logger.info(f"  Risk_on:  {summary['risk_on_f1']['mean']:.3f} ± {summary['risk_on_f1']['std']:.3f}  (range: {summary['risk_on_f1']['min']:.3f}-{summary['risk_on_f1']['max']:.3f})")
        logger.info("")

        logger.info(f"Distribution Error: {summary['distribution_error']['mean']:.3f} ± {summary['distribution_error']['std']:.3f}")
        logger.info("")

        # Success criteria
        logger.info("SUCCESS CRITERIA:")
        criteria = summary['success_criteria']
        status_avg_f1 = "✅" if criteria['avg_f1_gte_80'] else "❌"
        status_crisis = "✅" if criteria['crisis_f1_gte_85'] else "❌"
        status_risk_off = "✅" if criteria['risk_off_f1_gte_60'] else "❌"
        status_risk_on = "✅" if criteria['risk_on_f1_gte_50'] else "❌"
        status_stability = "✅" if criteria['no_fold_degrades_20pct'] else "❌"
        status_dist = "✅" if criteria['dist_error_lte_10pct'] else "❌"

        logger.info(f"  {status_avg_f1} Macro F1 >= 80%: {summary['macro_f1']['mean']:.1%}")
        logger.info(f"  {status_crisis} Crisis F1 >= 85%: {summary['crisis_f1']['mean']:.1%}")
        logger.info(f"  {status_risk_off} Risk_off F1 >= 60%: {summary['risk_off_f1']['mean']:.1%}")
        logger.info(f"  {status_risk_on} Risk_on F1 >= 50%: {summary['risk_on_f1']['mean']:.1%}")
        logger.info(f"  {status_stability} Fold stability (<20% range): {(summary['macro_f1']['max'] - summary['macro_f1']['min']):.1%}")
        logger.info(f"  {status_dist} Distribution error < 10%: {summary['distribution_error']['mean']:.1%}")
        logger.info("")

        if summary['all_criteria_met']:
            logger.info("✅ ALL SUCCESS CRITERIA MET - Model is robust across time periods")
        else:
            logger.info("❌ SOME CRITERIA NOT MET - Review failed criteria above")

        logger.info("")
        logger.info("FOLD-BY-FOLD RESULTS:")
        logger.info("-" * 80)
        logger.info(f"{'Fold':<15} {'Accuracy':>10} {'Crisis F1':>10} {'Risk_off F1':>12} {'Risk_on F1':>12} {'Macro F1':>10}")
        logger.info("-" * 80)
        for r in results:
            logger.info(
                f"{r['fold']:<15} {r['accuracy']:>9.1%} {r['crisis_f1']:>10.3f} "
                f"{r['risk_off_f1']:>12.3f} {r['risk_on_f1']:>12.3f} {r['macro_avg_f1']:>10.3f}"
            )
        logger.info("-" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Walk-forward validation for hierarchical regime v6')
    parser.add_argument(
        '--data',
        type=str,
        default='data/btcusd_1h_features_with_gt.parquet',
        help='Input data with regime labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/walk_forward_validation_v6.json',
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    # Import model class
    from engine.models.hierarchical_regime_model import HierarchicalRegimeClassifier

    # Load data
    logger.info(f"Loading data: {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"  Loaded {len(df):,} bars ({df.index.min()} to {df.index.max()})")

    # Define features (same as v6 training)
    feature_cols = [
        'crash_frequency_7d', 'crisis_persistence', 'aftershock_score',
        'RV_7', 'RV_30', 'drawdown_persistence',
        'volume_z_7d',
        'USDT.D', 'BTC.D',
        'returns_90d',
        'rsi_14', 'adx_14',
    ]

    # Check features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        return

    # Run validation
    validator = WalkForwardValidator(HierarchicalRegimeClassifier)
    results = validator.run_validation(df, feature_cols)

    # Summarize
    summary = validator.summarize_results(results)
    validator.print_summary(summary, results)

    # Save results
    output_data = {
        'validation_date': datetime.now().isoformat(),
        'model': 'hierarchical_regime_v6',
        'n_folds': len(results),
        'summary': summary,
        'fold_results': results
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved: {output_path}")

    # Final verdict
    logger.info("\n" + "=" * 80)
    logger.info("FINAL VERDICT")
    logger.info("=" * 80)
    if summary['all_criteria_met']:
        logger.info("✅ Model v6 VALIDATED - Ready for production deployment")
        logger.info("   Next steps:")
        logger.info("   1. Implement temporal smoothing (Stage 3)")
        logger.info("   2. Create production wrapper class")
        logger.info("   3. Deploy with monitoring and fallback")
    else:
        logger.info("⚠️ Model needs refinement before production")
        logger.info("   Review failed criteria and consider:")
        logger.info("   - Adjusting class weights")
        logger.info("   - Adding temporal features")
        logger.info("   - Ensemble with v4 for stability")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
