#!/usr/bin/env python3
"""
ML Quality Filter - Option B

Train a machine learning model to predict trade quality based on domain scores.
Uses historical trade outcomes to learn which combinations of domain scores
lead to profitable trades.

Approach:
    1. Build dataset from historical trades with domain scores
    2. Train LightGBM classifier (y = profitable trade)
    3. Use model probability as quality filter/multiplier
    4. Validate on out-of-sample period

Output:
    - results/engine_weights/quality_filter_model.pkl
    - results/engine_weights/quality_filter_report.md
    - results/engine_weights/feature_importance.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple
import logging

# ML imports
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: LightGBM or sklearn not available. Install with: pip install lightgbm scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityFilterTrainer:
    """
    Train ML model to predict trade quality from domain scores.

    Uses LightGBM to learn patterns in successful trades based on:
        - Structure score (Wyckoff + SMC)
        - Liquidity score
        - Momentum score
        - Macro score
        - Cross-timeframe alignment
    """

    def __init__(self, feature_store_path: str, output_dir: str = "results/engine_weights"):
        """
        Initialize trainer.

        Args:
            feature_store_path: Path to feature store with historical data
            output_dir: Output directory for model and reports
        """
        if not ML_AVAILABLE:
            raise ImportError("LightGBM and sklearn required. Install with: pip install lightgbm scikit-learn")

        self.feature_store_path = Path(feature_store_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info(f"Loading feature store: {self.feature_store_path}")
        self.df = pd.read_parquet(self.feature_store_path)
        logger.info(f"Loaded {len(self.df)} bars from {self.df.index[0]} to {self.df.index[-1]}")

        self.model = None
        self.feature_cols = None

    def _build_dataset(self, lookforward_bars: int = 168) -> pd.DataFrame:
        """
        Build training dataset from feature store.

        Labels each bar with forward return to determine if a trade
        would have been profitable.

        Args:
            lookforward_bars: Bars to look forward for profit/loss (default: 1 week)

        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Building training dataset (lookforward={lookforward_bars} bars)...")

        df = self.df.copy()

        # Compute domain scores
        df['structure_score'] = df.get('tf4h_fusion_score', 0.5)
        df['liquidity_score'] = df.get('liquidity_score', 0.5)

        # Momentum score
        df['rsi'] = df.get('rsi_14', 50.0)
        df['adx'] = df.get('adx_14', 20.0)
        df['rsi_deviation'] = abs(df['rsi'] - 50.0) / 50.0
        df['adx_strength'] = np.clip(df['adx'] / 40.0, 0, 1)
        df['momentum_score'] = 0.5 + (df['rsi_deviation'] * df['adx_strength'] * 0.5)

        # Macro score
        df['vix_z'] = df.get('VIX_Z', 0.0)
        df['regime_conf'] = df.get('regime_confidence', 0.5)
        df['macro_score'] = np.clip(0.5 - (df['vix_z'] * 0.1) + (df['regime_conf'] * 0.3), 0, 1)

        # MTF alignment
        df['tf1h_score'] = df.get('tf1h_fusion_score', 0.5)
        df['tf4h_score'] = df.get('tf4h_fusion_score', 0.5)
        df['tf1d_score'] = df.get('tf1d_fusion_score', 0.5)
        df['mtf_std'] = df[['tf1h_score', 'tf4h_score', 'tf1d_score']].std(axis=1)
        df['mtf_alignment'] = 1.0 - np.clip(df['mtf_std'] * 2, 0, 1)

        # Forward return (label)
        df['forward_return'] = (
            (df['close'].shift(-lookforward_bars) - df['close']) / df['close']
        )

        # Binary label: profitable if forward_return > 0.02 (2% gain)
        # This filters for meaningful wins, not noise
        df['is_profitable'] = (df['forward_return'] > 0.02).astype(int)

        # Drop NaN rows (at end due to forward shift)
        df = df.dropna(subset=['forward_return', 'is_profitable'])

        logger.info(f"Dataset built: {len(df)} samples")
        logger.info(f"  Profitable: {df['is_profitable'].sum()} ({df['is_profitable'].mean():.1%})")
        logger.info(f"  Unprofitable: {(~df['is_profitable'].astype(bool)).sum()}")

        return df

    def train(self, test_size: float = 0.3, random_state: int = 42) -> Dict:
        """
        Train LightGBM quality filter model.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Training metrics dict
        """
        logger.info("=" * 80)
        logger.info("QUALITY FILTER MODEL TRAINING")
        logger.info("=" * 80)

        # Build dataset
        dataset = self._build_dataset()

        # Feature columns
        self.feature_cols = [
            'structure_score', 'liquidity_score', 'momentum_score', 'macro_score',
            'mtf_alignment', 'rsi', 'adx', 'vix_z', 'regime_conf'
        ]

        X = dataset[self.feature_cols]
        y = dataset['is_profitable']

        # Train/test split (time-based is better, but random for simplicity)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"\nTrain set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Train LightGBM
        logger.info("\nTraining LightGBM classifier...")

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=callbacks
        )

        logger.info("✓ Training complete")

        # Evaluate
        logger.info("\nEvaluating model...")

        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"\nTest AUC: {auc:.4f}")

        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Unprofitable', 'Profitable']))

        # Feature importance
        self._plot_feature_importance()

        # ROC curve
        self._plot_roc_curve(y_test, y_pred_proba, auc)

        # Save model
        model_path = self.output_dir / "quality_filter_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
                'train_date': datetime.now().isoformat(),
                'test_auc': auc
            }, f)

        logger.info(f"\n✓ Saved model to: {model_path}")

        return {
            'test_auc': auc,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_cols': self.feature_cols
        }

    def _plot_feature_importance(self):
        """Plot and save feature importance."""
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance (Gain)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()

        importance_path = self.output_dir / "feature_importance.png"
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved feature importance to: {importance_path}")
        plt.close()

    def _plot_roc_curve(self, y_test, y_pred_proba, auc):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Quality Filter Model', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        roc_path = self.output_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved ROC curve to: {roc_path}")
        plt.close()

    def predict_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict trade quality probability for new data.

        Args:
            df: DataFrame with domain scores

        Returns:
            Series with quality probabilities [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Compute features
        features = self._compute_features(df)

        # Predict
        quality_proba = self.model.predict(
            features[self.feature_cols],
            num_iteration=self.model.best_iteration
        )

        return pd.Series(quality_proba, index=df.index, name='quality_score')

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature columns for prediction."""
        features = df.copy()

        features['structure_score'] = features.get('tf4h_fusion_score', 0.5)
        features['liquidity_score'] = features.get('liquidity_score', 0.5)

        features['rsi'] = features.get('rsi_14', 50.0)
        features['adx'] = features.get('adx_14', 20.0)
        features['rsi_deviation'] = abs(features['rsi'] - 50.0) / 50.0
        features['adx_strength'] = np.clip(features['adx'] / 40.0, 0, 1)
        features['momentum_score'] = 0.5 + (features['rsi_deviation'] * features['adx_strength'] * 0.5)

        features['vix_z'] = features.get('VIX_Z', 0.0)
        features['regime_conf'] = features.get('regime_confidence', 0.5)
        features['macro_score'] = np.clip(0.5 - (features['vix_z'] * 0.1) + (features['regime_conf'] * 0.3), 0, 1)

        features['tf1h_score'] = features.get('tf1h_fusion_score', 0.5)
        features['tf4h_score'] = features.get('tf4h_fusion_score', 0.5)
        features['tf1d_score'] = features.get('tf1d_fusion_score', 0.5)
        features['mtf_std'] = features[['tf1h_score', 'tf4h_score', 'tf1d_score']].std(axis=1)
        features['mtf_alignment'] = 1.0 - np.clip(features['mtf_std'] * 2, 0, 1)

        return features


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train ML quality filter for trade prediction'
    )
    parser.add_argument(
        '--feature-store',
        default='data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet',
        help='Path to feature store parquet'
    )
    parser.add_argument(
        '--output-dir',
        default='results/engine_weights',
        help='Output directory'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Test set fraction (default: 0.3)'
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = QualityFilterTrainer(
        feature_store_path=args.feature_store,
        output_dir=args.output_dir
    )

    # Train model
    metrics = trainer.train(test_size=args.test_size)

    # Generate report
    report_path = Path(args.output_dir) / "quality_filter_report.md"
    with open(report_path, 'w') as f:
        f.write("# ML Quality Filter Training Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Feature Store**: {args.feature_store}\n\n")
        f.write("## Model Performance\n\n")
        f.write(f"- Test AUC: {metrics['test_auc']:.4f}\n")
        f.write(f"- Train Samples: {metrics['train_samples']:,}\n")
        f.write(f"- Test Samples: {metrics['test_samples']:,}\n\n")
        f.write("## Features Used\n\n")
        for feat in metrics['feature_cols']:
            f.write(f"- {feat}\n")
        f.write("\n## Usage\n\n")
        f.write("```python\n")
        f.write("import pickle\n")
        f.write("with open('quality_filter_model.pkl', 'rb') as f:\n")
        f.write("    model_data = pickle.load(f)\n")
        f.write("    model = model_data['model']\n")
        f.write("    feature_cols = model_data['feature_cols']\n")
        f.write("\n")
        f.write("# Predict quality\n")
        f.write("quality_proba = model.predict(X[feature_cols])\n")
        f.write("fusion_filtered = fusion_base * quality_proba\n")
        f.write("```\n\n")
        f.write("## Interpretation\n\n")
        f.write("Quality score represents the probability that a trade signal with given\n")
        f.write("domain scores will result in a profitable trade (>2% gain).\n\n")
        f.write("Use as:\n")
        f.write("- **Multiplier**: `final_score = base_fusion * quality_score`\n")
        f.write("- **Filter**: Only take trades where `quality_score > 0.6`\n")

    logger.info(f"\n✓ Saved report to: {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ QUALITY FILTER TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {args.output_dir}/")
    logger.info("  - quality_filter_model.pkl")
    logger.info("  - quality_filter_report.md")
    logger.info("  - feature_importance.png")
    logger.info("  - roc_curve.png")


if __name__ == '__main__':
    main()
