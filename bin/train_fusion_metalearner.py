#!/usr/bin/env python3
"""
Approach 1: ML Meta-Learner for Engine Weight Optimization

Fast approach (~2 hours runtime) using supervised learning to derive
optimal engine weights from historical trade outcomes.

Strategy:
- Load historical trades CSV (entry features + outcomes)
- Features: domain scores (structure, liquidity, momentum, wyckoff, macro)
- Label: trade_won (1/0)
- Model: Logistic Regression (interpretable) or Gradient Boosted Trees (nonlinear)
- Output: Learned weights + feature importances + validation metrics

Regime-aware:
- Train separate models for each regime (risk_on, risk_off, neutral, crisis)
- Compare performance across regimes

Version: 1.0
Author: Backend Architect
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/ml_metalearner_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetaLearnerTrainer:
    """
    ML-based meta-learner for engine weight optimization.

    Supports three approaches:
    1. Logistic Regression (interpretable coefficients)
    2. Gradient Boosted Trees (capture nonlinearities)
    3. Weighted Average Optimizer (scipy.optimize)
    """

    DOMAIN_FEATURES = [
        'structure_score',
        'liquidity_score',
        'momentum_score',
        'wyckoff_score',
        'macro_score'
    ]

    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize meta-learner trainer.

        Args:
            model_type: 'logistic', 'gbt', or 'weighted_avg'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.weights = None
        self.feature_importances = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        regime_filter: str = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels from trade history.

        Args:
            df: DataFrame with trade history
            regime_filter: Optional regime to filter (e.g., 'risk_off')

        Returns:
            (X: features DataFrame, y: labels Series)
        """
        # Filter by regime if specified
        if regime_filter:
            regime_cols = [c for c in df.columns if c.startswith('macro_regime_')]
            regime_col = f'macro_regime_{regime_filter}'
            if regime_col in regime_cols:
                df = df[df[regime_col] == 1].copy()
                logger.info(f"Filtered to regime '{regime_filter}': {len(df)} trades")

        # Extract domain scores
        # Derive missing scores if not present
        X = pd.DataFrame()

        # Structure score (proxy: boms_strength)
        X['structure_score'] = df.get('boms_strength', 0.0)

        # Liquidity score
        X['liquidity_score'] = df.get('entry_liquidity_score', 0.0)

        # Momentum score (derive from RSI/ADX)
        X['momentum_score'] = self._derive_momentum(df)

        # Wyckoff score
        X['wyckoff_score'] = df.get('wyckoff_phase_score', 0.0)

        # Macro score (derive from VIX_Z/DXY_Z)
        X['macro_score'] = self._derive_macro(df)

        # Label: trade won
        y = df['trade_won'].astype(int)

        # Drop rows with NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def _derive_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Derive momentum score from RSI/ADX/volume."""
        rsi = df.get('rsi_14', 50.0)
        adx = df.get('adx_14', 20.0)
        vol_z = df.get('volume_zscore', 0.0)

        # Normalize components
        rsi_comp = np.abs(rsi - 50.0) / 25.0
        adx_comp = (adx - 10.0) / 30.0
        vol_comp = vol_z / 2.0

        # Weighted combination
        momentum = 0.4 * rsi_comp + 0.3 * adx_comp + 0.3 * vol_comp
        return momentum.clip(0, 1)

    def _derive_macro(self, df: pd.DataFrame) -> pd.Series:
        """Derive macro score from VIX_Z/DXY_Z/funding."""
        vix_z = df.get('vix_z_score', 0.0)

        # Invert and normalize (lower VIX = better for longs)
        # Map -2 to +2 sigma to [1, 0]
        macro = 1.0 - ((vix_z + 2.0) / 4.0)
        return macro.clip(0, 1)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict:
        """
        Train meta-learner model.

        Args:
            X: Feature DataFrame
            y: Target labels
            test_size: Test set proportion
            cv_folds: Cross-validation folds

        Returns:
            Dict with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model based on type
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)

            # Extract weights from coefficients
            self.weights = dict(zip(X.columns, self.model.coef_[0]))
            self.feature_importances = {
                k: abs(v) for k, v in self.weights.items()
            }

        elif self.model_type == 'gbt':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)

            # Extract feature importances
            self.feature_importances = dict(zip(X.columns, self.model.feature_importances_))
            # Normalize to weights
            total_importance = sum(self.feature_importances.values())
            self.weights = {
                k: v / total_importance
                for k, v in self.feature_importances.items()
            }

        elif self.model_type == 'weighted_avg':
            # Optimize weights directly using scipy
            self.weights = self._optimize_weights(X_train, y_train)
            self.feature_importances = self.weights.copy()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Evaluate
        if self.model_type != 'weighted_avg':
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            # Weighted average prediction
            y_pred_proba = self._weighted_fusion(X_test, self.weights)
            y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Cross-validation score
        if self.model_type != 'weighted_avg':
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=cv_folds, scoring='roc_auc'
            )
        else:
            cv_scores = [auc]  # Single score for weighted avg

        metrics = {
            'test_auc': auc,
            'test_precision': report['1']['precision'],
            'test_recall': report['1']['recall'],
            'test_f1': report['1']['f1-score'],
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'weights': self.weights,
            'feature_importances': self.feature_importances
        }

        logger.info(f"Training complete: AUC={auc:.3f}, Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}")

        return metrics

    def _optimize_weights(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Optimize weights using scipy.optimize to maximize AUC.

        Args:
            X: Feature DataFrame
            y: Target labels

        Returns:
            Optimized weights dict
        """
        n_features = X.shape[1]

        def objective(weights):
            """Negative AUC (minimize)."""
            weights_dict = dict(zip(X.columns, weights))
            fusion = self._weighted_fusion(X, weights_dict)
            try:
                auc = roc_auc_score(y, fusion)
                return -auc  # Minimize negative AUC
            except:
                return 1.0  # Penalty for invalid predictions

        # Constraints: weights sum to 1, all >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0)] * n_features

        # Initial guess: equal weights
        x0 = np.ones(n_features) / n_features

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )

        weights = dict(zip(X.columns, result.x))
        logger.info(f"Optimized weights: {weights}")

        return weights

    def _weighted_fusion(self, X: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """
        Compute weighted fusion scores.

        Args:
            X: Feature DataFrame
            weights: Weights dict

        Returns:
            Fusion scores array
        """
        fusion = np.zeros(len(X))
        for col, weight in weights.items():
            if col in X.columns:
                fusion += weight * X[col].values
        return fusion

    def save_weights(self, output_path: str):
        """Save learned weights to JSON config."""
        config = {
            'engine_weights': self.weights,
            'feature_importances': self.feature_importances,
            'model_type': self.model_type,
            'version': '1.0'
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ML meta-learner for engine weights')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to trades CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for weights config JSON')
    parser.add_argument('--model', type=str, default='logistic',
                        choices=['logistic', 'gbt', 'weighted_avg'],
                        help='Model type')
    parser.add_argument('--regime', type=str, default=None,
                        help='Filter to specific regime (risk_on, risk_off, neutral, crisis)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion')

    args = parser.parse_args()

    logger.info(f"=== ML Meta-Learner Training ===")
    logger.info(f"Input: {args.input}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Regime filter: {args.regime or 'None (all regimes)'}")

    # Load data
    logger.info("Loading trade history...")
    df = pd.read_csv(args.input, parse_dates=['entry_time', 'exit_time'])
    logger.info(f"Loaded {len(df)} trades")

    # Train meta-learner
    trainer = MetaLearnerTrainer(model_type=args.model)
    X, y = trainer.prepare_data(df, regime_filter=args.regime)

    if len(X) < 50:
        logger.error(f"Insufficient data: {len(X)} samples (need >= 50)")
        sys.exit(1)

    logger.info("Training model...")
    metrics = trainer.train(X, y, test_size=args.test_size)

    # Save results
    trainer.save_weights(args.output)

    # Print summary
    print("\n=== Meta-Learner Results ===")
    print(f"Model: {args.model}")
    print(f"Test AUC: {metrics['test_auc']:.3f}")
    print(f"Test Precision: {metrics['test_precision']:.3f}")
    print(f"Test Recall: {metrics['test_recall']:.3f}")
    print(f"CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    print("\nLearned Weights:")
    for domain, weight in sorted(metrics['weights'].items(), key=lambda x: -x[1]):
        print(f"  {domain:20s}: {weight:.4f}")

    print(f"\nWeights saved to: {args.output}")


if __name__ == '__main__':
    main()
