"""
Kelly-Lite Dynamic Risk Sizing

ML-based position sizer that optimizes risk % per trade (0-2%) using:
- Fusion score
- Regime classification
- Realized volatility (RV 20d/60d)
- VIX proxy
- Recent drawdown
- Expected edge (R-multiple)

Uses gradient boosting to predict optimal risk % for maximum long-term growth
without blowups.

Guardrails:
- Hard clamp: 0-2%
- Decay after ≥2 consecutive losers
- Hard cap in risk_off/crisis regimes
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from typing import Dict, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KellyLiteSizer:
    """
    ML-based position sizer using gradient boosting

    Predicts optimal risk % based on market conditions and edge
    """

    def __init__(self, base_risk_pct: float = 0.0075):
        """
        Initialize Kelly-Lite sizer

        Args:
            base_risk_pct: Baseline risk percentage (default 0.75%)
        """
        self.base_risk_pct = base_risk_pct
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'fusion_score',
            'regime_risk_on',
            'regime_neutral',
            'regime_risk_off',
            'regime_crisis',
            'rv_20d',
            'rv_60d',
            'vix_proxy',
            'recent_dd',
            'expected_r',
            'consecutive_losses'
        ]

    def _build_features(
        self,
        fusion_score: float,
        regime: str,
        rv_20d: float,
        rv_60d: float,
        vix_proxy: float,
        recent_dd: float,
        expected_r: float,
        consecutive_losses: int
    ) -> np.ndarray:
        """
        Build feature vector for prediction

        Args:
            fusion_score: Fusion confidence [0-1]
            regime: Regime label (risk_on/neutral/risk_off/crisis)
            rv_20d: 20-day realized volatility
            rv_60d: 60-day realized volatility
            vix_proxy: VIX or volatility proxy
            recent_dd: Recent drawdown (negative %)
            expected_r: Expected R-multiple
            consecutive_losses: Number of consecutive losing trades

        Returns:
            Feature vector
        """
        # One-hot encode regime
        regime_features = {
            'risk_on': 1.0 if regime == 'risk_on' else 0.0,
            'neutral': 1.0 if regime == 'neutral' else 0.0,
            'risk_off': 1.0 if regime == 'risk_off' else 0.0,
            'crisis': 1.0 if regime == 'crisis' else 0.0
        }

        features = np.array([
            fusion_score,
            regime_features['risk_on'],
            regime_features['neutral'],
            regime_features['risk_off'],
            regime_features['crisis'],
            rv_20d,
            rv_60d,
            vix_proxy,
            recent_dd,
            expected_r,
            float(consecutive_losses)
        ])

        return features

    def train(self, training_data: pd.DataFrame):
        """
        Train Kelly-Lite model on historical trade data

        Args:
            training_data: DataFrame with columns:
                - fusion_score, regime, rv_20d, rv_60d, vix_proxy
                - recent_dd, expected_r, consecutive_losses
                - optimal_risk_pct (target - computed from Kelly criterion)
        """
        logger.info("Training Kelly-Lite sizer...")

        # Build feature matrix
        X = []
        for _, row in training_data.iterrows():
            features = self._build_features(
                fusion_score=row['fusion_score'],
                regime=row['regime'],
                rv_20d=row['rv_20d'],
                rv_60d=row['rv_60d'],
                vix_proxy=row['vix_proxy'],
                recent_dd=row['recent_dd'],
                expected_r=row['expected_r'],
                consecutive_losses=int(row['consecutive_losses'])
            )
            X.append(features)

        X = np.array(X)
        y = training_data['optimal_risk_pct'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train gradient boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        logger.info(f"  Train R²: {train_score:.3f}")
        logger.info(f"  Test R²: {test_score:.3f}")

        # Feature importance
        importances = self.model.feature_importances_
        for name, importance in zip(self.feature_names, importances):
            if importance > 0.05:
                logger.info(f"  {name}: {importance:.3f}")

        logger.info("✅ Kelly-Lite sizer trained")

    def predict_risk_pct(
        self,
        fusion_score: float,
        regime: str,
        rv_20d: float,
        rv_60d: float,
        vix_proxy: float,
        recent_dd: float = 0.0,
        expected_r: float = 1.0,
        consecutive_losses: int = 0
    ) -> float:
        """
        Predict optimal risk % for current trade

        Args:
            fusion_score: Fusion confidence [0-1]
            regime: Regime label
            rv_20d: 20-day RV
            rv_60d: 60-day RV
            vix_proxy: VIX or volatility proxy
            recent_dd: Recent drawdown (negative %)
            expected_r: Expected R-multiple
            consecutive_losses: Consecutive losing trades

        Returns:
            Risk percentage (0-2%)
        """
        if self.model is None:
            logger.warning("Model not trained, using base risk")
            return self.base_risk_pct

        # Build features
        features = self._build_features(
            fusion_score, regime, rv_20d, rv_60d, vix_proxy,
            recent_dd, expected_r, consecutive_losses
        ).reshape(1, -1)

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predicted_risk = self.model.predict(features_scaled)[0]

        # Apply guardrails
        predicted_risk = self._apply_guardrails(
            predicted_risk, regime, consecutive_losses, recent_dd
        )

        return predicted_risk

    def _apply_guardrails(
        self,
        predicted_risk: float,
        regime: str,
        consecutive_losses: int,
        recent_dd: float
    ) -> float:
        """
        Apply safety guardrails to predicted risk

        Args:
            predicted_risk: Raw prediction from model
            regime: Current regime
            consecutive_losses: Consecutive losses
            recent_dd: Recent drawdown

        Returns:
            Guarded risk percentage
        """
        # Hard clamp 0-2%
        risk = np.clip(predicted_risk, 0.0, 0.02)

        # Decay after ≥2 consecutive losers
        if consecutive_losses >= 2:
            decay_factor = 0.7 ** (consecutive_losses - 1)
            risk *= decay_factor
            logger.debug(f"Loss streak decay: {consecutive_losses} losses → {decay_factor:.2f}x")

        # Hard cap in risk_off/crisis
        if regime == 'risk_off':
            risk = min(risk, 0.01)  # Max 1% in risk-off
        elif regime == 'crisis':
            risk = min(risk, 0.005)  # Max 0.5% in crisis

        # Reduce risk if in drawdown
        if recent_dd < -0.10:  # >10% DD
            dd_factor = max(0.5, 1 + recent_dd)  # Scale down proportionally
            risk *= dd_factor
            logger.debug(f"Drawdown reduction: DD={recent_dd:.1%} → {dd_factor:.2f}x")

        return risk

    def save(self, path: str):
        """Save trained model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'base_risk_pct': self.base_risk_pct,
                'feature_names': self.feature_names
            }, f)

        logger.info(f"💾 Saved Kelly-Lite sizer to {path}")

    @classmethod
    def load(cls, path: str):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        sizer = cls(base_risk_pct=data['base_risk_pct'])
        sizer.model = data['model']
        sizer.scaler = data['scaler']
        sizer.feature_names = data['feature_names']

        logger.info(f"✅ Loaded Kelly-Lite sizer from {path}")
        return sizer


def generate_training_data(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate training data for Kelly-Lite from backtest results

    Computes optimal risk % retrospectively using Kelly criterion:
    f* = (p * b - q) / b
    where p = win rate, q = 1-p, b = avg_win/avg_loss

    Args:
        backtest_results: DataFrame with trade results

    Returns:
        Training DataFrame with optimal_risk_pct target
    """
    logger.info("Generating Kelly-Lite training data...")

    training_rows = []

    # Rolling window for Kelly calculation
    window = 50

    for i in range(window, len(backtest_results)):
        # Get recent trades
        recent = backtest_results.iloc[i-window:i]

        # Compute Kelly inputs
        wins = recent[recent['pnl'] > 0]
        losses = recent[recent['pnl'] < 0]

        if len(wins) == 0 or len(losses) == 0:
            continue

        p = len(wins) / len(recent)  # Win rate
        q = 1 - p
        avg_win = wins['pnl'].mean()
        avg_loss = abs(losses['pnl'].mean())
        b = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly criterion
        kelly_f = (p * b - q) / b if b > 0 else 0

        # Use fractional Kelly (0.25-0.50 of full Kelly for safety)
        optimal_risk = np.clip(kelly_f * 0.35, 0.0, 0.02)

        # Current trade features
        current = backtest_results.iloc[i]

        training_rows.append({
            'fusion_score': current.get('fusion_score', 0.7),
            'regime': current.get('regime', 'neutral'),
            'rv_20d': current.get('rv_20d', 40.0),
            'rv_60d': current.get('rv_60d', 45.0),
            'vix_proxy': current.get('vix_proxy', 20.0),
            'recent_dd': current.get('drawdown', 0.0),
            'expected_r': current.get('expected_r', 1.0),
            'consecutive_losses': current.get('consecutive_losses', 0),
            'optimal_risk_pct': optimal_risk
        })

    df = pd.DataFrame(training_rows)
    logger.info(f"✅ Generated {len(df)} training samples")

    return df


if __name__ == "__main__":
    # Example: Load optimization results and train Kelly-Lite
    logger.info("=" * 70)
    logger.info("Kelly-Lite Risk Sizer Training")
    logger.info("=" * 70)

    # Load ML dataset
    try:
        ml_data = pd.read_parquet("data/ml/optimization_results.parquet")
        logger.info(f"Loaded {len(ml_data)} optimization results")

        # For training, we need trade-level data, not config-level
        # This is a placeholder - in production, load from trade logs
        logger.info("⚠️  Kelly-Lite requires trade-level data")
        logger.info("   Use hybrid_runner trade logs for full training")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
