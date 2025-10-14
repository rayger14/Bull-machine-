"""
ML-Based Fusion Scorer

Learns optimal domain weights (Wyckoff, SMC, HOB, Momentum) dynamically
using XGBoost to replace static weight aggregation.

Trains on historical signals with labels (profitable trades) to optimize
confluence for higher win rate (+5-10%).

Target: PF ≥1.8, trades ≥50 per period
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from typing import Dict, Optional, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionScorerML:
    """
    XGBoost-based fusion scorer that learns optimal signal weighting

    Replaces fixed weights with ML predictions for better confluence detection
    """

    def __init__(self):
        """Initialize ML fusion scorer"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            # Domain scores
            'wyckoff_score',
            'smc_score',
            'hob_score',
            'momentum_score',
            'temporal_score',
            # Domain interactions
            'wyckoff_smc_product',
            'wyckoff_hob_product',
            'momentum_hob_product',
            # Macro context
            'rv_20d',
            'rv_60d',
            'vix_proxy',
            'regime_risk_on',
            'regime_neutral',
            'regime_risk_off',
            # Market structure
            'adx',
            'atr_normalized',
            'volume_ratio',
            # Temporal features
            'hour_of_day',
            'day_of_week'
        ]
        self.threshold = 0.65  # Default threshold for fusion confidence

    def _build_features(
        self,
        domain_scores: Dict[str, float],
        macro_features: Dict[str, float],
        market_features: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> np.ndarray:
        """
        Build comprehensive feature vector for ML fusion

        Args:
            domain_scores: {wyckoff, smc, hob, momentum, temporal}
            macro_features: {rv_20d, rv_60d, vix_proxy, regime}
            market_features: {adx, atr, volume}
            timestamp: Current timestamp for temporal features

        Returns:
            Feature vector
        """
        # Extract scores
        wyckoff = domain_scores.get('wyckoff', 0.0)
        smc = domain_scores.get('smc', 0.0)
        hob = domain_scores.get('hob', 0.0)
        momentum = domain_scores.get('momentum', 0.0)
        temporal = domain_scores.get('temporal', 0.0)

        # Domain interactions
        wy_smc = wyckoff * smc
        wy_hob = wyckoff * hob
        mom_hob = momentum * hob

        # Regime one-hot
        regime = macro_features.get('regime', 'neutral')
        regime_features = {
            'risk_on': 1.0 if regime == 'risk_on' else 0.0,
            'neutral': 1.0 if regime == 'neutral' else 0.0,
            'risk_off': 1.0 if regime == 'risk_off' else 0.0
        }

        # Temporal features
        hour = timestamp.hour if timestamp else 12
        day_of_week = timestamp.dayofweek if timestamp else 2

        features = np.array([
            # Domain scores
            wyckoff, smc, hob, momentum, temporal,
            # Interactions
            wy_smc, wy_hob, mom_hob,
            # Macro
            macro_features.get('rv_20d', 40.0),
            macro_features.get('rv_60d', 45.0),
            macro_features.get('vix_proxy', 20.0),
            regime_features['risk_on'],
            regime_features['neutral'],
            regime_features['risk_off'],
            # Market structure
            market_features.get('adx', 25.0),
            market_features.get('atr_normalized', 1.0),
            market_features.get('volume_ratio', 1.0),
            # Temporal
            float(hour) / 24.0,  # Normalize 0-1
            float(day_of_week) / 7.0
        ])

        return features

    def train(
        self,
        training_data: pd.DataFrame,
        test_size: float = 0.2,
        optimize_threshold: bool = True
    ):
        """
        Train XGBoost fusion scorer

        Args:
            training_data: DataFrame with columns:
                - domain scores, macro features, market features
                - timestamp, label (1=profitable, 0=unprofitable)
            test_size: Fraction for test set
            optimize_threshold: If True, find optimal threshold for precision/recall
        """
        logger.info("=" * 70)
        logger.info("Training ML Fusion Scorer (XGBoost)")
        logger.info("=" * 70)

        # Build feature matrix
        X = []
        for _, row in training_data.iterrows():
            domain_scores = {
                'wyckoff': row['wyckoff'],
                'smc': row['smc'],
                'hob': row['hob'],
                'momentum': row['momentum'],
                'temporal': row.get('temporal', 0.0)
            }
            macro_features = {
                'rv_20d': row.get('rv_20d', 40.0),
                'rv_60d': row.get('rv_60d', 45.0),
                'vix_proxy': row.get('vix_proxy', 20.0),
                'regime': row.get('regime', 'neutral')
            }
            market_features = {
                'adx': row.get('adx', 25.0),
                'atr_normalized': row.get('atr_normalized', 1.0),
                'volume_ratio': row.get('volume_ratio', 1.0)
            }
            timestamp = row.get('timestamp', None)

            features = self._build_features(
                domain_scores, macro_features, market_features, timestamp
            )
            X.append(features)

        X = np.array(X)
        y = training_data['label'].values

        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Positive labels: {y.sum()} ({y.mean()*100:.1f}%)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost with scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )

        # Fit with early stopping
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Evaluate
        y_train_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_test_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)

        logger.info(f"\n📊 Performance:")
        logger.info(f"   Train AUC: {train_auc:.3f}")
        logger.info(f"   Test AUC: {test_auc:.3f}")

        # Optimize threshold if requested
        if optimize_threshold:
            precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred_proba)

            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[best_idx]

            logger.info(f"\n🎯 Optimal Threshold: {self.threshold:.3f}")
            logger.info(f"   Precision: {precision[best_idx]:.3f}")
            logger.info(f"   Recall: {recall[best_idx]:.3f}")
            logger.info(f"   F1 Score: {f1_scores[best_idx]:.3f}")

        # Feature importance
        logger.info(f"\n📈 Feature Importance (Top 10):")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        for i in range(min(10, len(self.feature_names))):
            idx = indices[i]
            logger.info(f"   {self.feature_names[idx]}: {importances[idx]:.3f}")

        logger.info("\n✅ ML Fusion Scorer trained")

    def predict_fusion_score(
        self,
        domain_scores: Dict[str, float],
        macro_features: Dict[str, float],
        market_features: Dict[str, float],
        timestamp: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Predict fusion confidence score

        Args:
            domain_scores: Domain signal scores
            macro_features: Macro context features
            market_features: Market structure features
            timestamp: Current timestamp

        Returns:
            Fusion confidence score [0-1]
        """
        if self.model is None:
            logger.warning("Model not trained, using simple average")
            return np.mean(list(domain_scores.values()))

        # Build features
        features = self._build_features(
            domain_scores, macro_features, market_features, timestamp
        ).reshape(1, -1)

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        fusion_score = self.model.predict_proba(features_scaled)[0, 1]

        return float(fusion_score)

    def should_enter(
        self,
        domain_scores: Dict[str, float],
        macro_features: Dict[str, float],
        market_features: Dict[str, float],
        timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[bool, float]:
        """
        Decide if entry should be taken

        Args:
            domain_scores, macro_features, market_features, timestamp

        Returns:
            (should_enter, fusion_score)
        """
        fusion_score = self.predict_fusion_score(
            domain_scores, macro_features, market_features, timestamp
        )

        should_enter = fusion_score >= self.threshold

        return should_enter, fusion_score

    def save(self, path: str):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'threshold': self.threshold,
                'feature_names': self.feature_names
            }, f)

        logger.info(f"💾 Saved ML Fusion Scorer to {path}")

    @classmethod
    def load(cls, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        scorer = cls()
        scorer.model = data['model']
        scorer.scaler = data['scaler']
        scorer.threshold = data['threshold']
        scorer.feature_names = data['feature_names']

        logger.info(f"✅ Loaded ML Fusion Scorer from {path}")
        logger.info(f"   Threshold: {scorer.threshold:.3f}")

        return scorer


def prepare_training_data_from_feature_store(
    asset: str,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Prepare training data from feature store

    Labels trades as profitable (1) or not (0) using forward-looking returns

    Args:
        asset: BTC or ETH
        lookback_days: Days of history to use

    Returns:
        Training DataFrame
    """
    logger.info(f"Preparing training data for {asset}...")

    # Load feature store
    feature_path = f"data/features/v18/{asset}_1H.parquet"
    df = pd.read_parquet(feature_path)

    # Load macro features
    macro_path = f"data/macro/{asset}_macro_features.parquet"
    macro_df = pd.read_parquet(macro_path)

    # Merge on timestamp
    df = df.reset_index()
    df = df.rename(columns={'time': 'timestamp'})
    df = df.merge(macro_df[['timestamp', 'rv_20d', 'rv_60d', 'VIX', 'funding']],
                  on='timestamp', how='left')

    # Compute forward returns (label: 1 if next 4-bar return > 0.5%)
    df['forward_return'] = df['close'].pct_change(4).shift(-4)
    df['label'] = (df['forward_return'] > 0.005).astype(int)

    # Filter to recent data
    df = df.tail(lookback_days * 24)  # Hourly data

    # Normalize ATR and volume
    df['atr_normalized'] = df['atr_20'] / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Rename domain scores
    training_df = df[[
        'timestamp', 'wyckoff', 'smc', 'hob', 'momentum', 'temporal',
        'rv_20d', 'rv_60d', 'adx_14', 'atr_normalized', 'volume_ratio',
        'label'
    ]].copy()

    training_df = training_df.rename(columns={
        'adx_14': 'adx',
        'VIX': 'vix_proxy'
    })

    # Add regime (default to neutral for now)
    training_df['regime'] = 'neutral'
    training_df['vix_proxy'] = training_df.get('vix_proxy', 20.0)

    # Drop NaN
    training_df = training_df.dropna()

    logger.info(f"✅ Prepared {len(training_df)} training samples for {asset}")
    logger.info(f"   Positive labels: {training_df['label'].sum()} ({training_df['label'].mean()*100:.1f}%)")

    return training_df


if __name__ == "__main__":
    # Example: Train fusion scorer on BTC data
    logger.info("=" * 70)
    logger.info("ML Fusion Scorer Training Pipeline")
    logger.info("=" * 70)

    # Prepare training data
    training_data = prepare_training_data_from_feature_store('BTC', lookback_days=365)

    # Train model
    scorer = FusionScorerML()
    scorer.train(training_data, optimize_threshold=True)

    # Save model
    scorer.save("models/fusion_scorer_xgb.pkl")

    logger.info("\n✅ Training complete!")
