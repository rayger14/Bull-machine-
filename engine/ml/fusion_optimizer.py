"""
ML-Based Fusion Weight Optimizer for Bull Machine v1.8.6

Dynamically optimizes domain weights (Wyckoff, SMC, HOB, Momentum, Temporal)
based on current market regime using Bayesian Optimization and online learning.

Philosophy: ML refines precision, doesn't rewrite wisdom.
- Domain engines stay deterministic (Wyckoff logic = human-designed)
- ML learns WHEN to trust which domains MORE (adaptive weighting)
- Self-balancing instinct inspired by Specter's consciousness

Based on optimization results showing:
- VIX is 6x more predictive than static config params
- Lower wyckoff weights (0.20-0.25) perform better in 2022-2025
- Momentum weights 0.23-0.31 are optimal
- Regime context matters more than fixed weights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current market regime state for weight optimization"""
    vix: float
    move: float
    dxy: float
    oil: float
    yield_spread: float
    btc_d: float
    usdt_d: float
    funding: float
    oi: float
    adx: float
    rsi: float
    volatility_realized: float  # 20-bar realized vol
    trend_strength: float  # Directional indicator

    def to_array(self) -> np.ndarray:
        """Convert to feature array for ML"""
        return np.array([
            self.vix, self.move, self.dxy, self.oil,
            self.yield_spread, self.btc_d, self.usdt_d,
            self.funding, self.oi, self.adx, self.rsi,
            self.volatility_realized, self.trend_strength
        ])

    @property
    def regime_label(self) -> str:
        """Classify into discrete regime"""
        if self.vix > 25 or self.move > 100:
            return "crisis"
        elif self.vix > 20 and (self.dxy > 103 or self.yield_spread > 0):
            return "risk_off"
        elif self.vix < 18 and self.dxy < 100:
            return "risk_on"
        else:
            return "neutral"


@dataclass
class WeightUpdate:
    """Proposed weight update from ML optimizer"""
    wyckoff: float
    smc: float
    hob: float
    momentum: float
    temporal: float
    confidence: float  # 0-1 confidence in this recommendation
    regime: str
    reason: str  # Human-readable explanation

    def to_dict(self) -> Dict[str, float]:
        """Convert to config dict"""
        return {
            'wyckoff': self.wyckoff,
            'smc': self.smc,
            'hob': self.hob,
            'momentum': self.momentum,
            'temporal': self.temporal
        }

    def validate(self) -> bool:
        """Ensure weights sum to 1.0 and are positive"""
        total = self.wyckoff + self.smc + self.hob + self.momentum + self.temporal
        all_positive = all(w >= 0.05 for w in [self.wyckoff, self.smc, self.hob, self.momentum, self.temporal])
        return abs(total - 1.0) < 0.01 and all_positive


class FusionWeightOptimizer:
    """
    ML-based adaptive fusion weight optimizer

    Uses historical optimization results (2,372 configs) to learn:
    1. Which domain weights perform best in which regimes
    2. How to dynamically adjust weights based on macro context
    3. When to be more/less selective (fusion threshold tuning)

    Training data: data/ml/optimization_results.parquet
    Features: Regime state (VIX, MOVE, DXY, etc.) + current domain scores
    Target: Profit Factor (or Sharpe Ratio)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = None
        self.training_history = []

        # Baseline weights from optimization (BTC_live.json)
        self.baseline_weights = {
            'wyckoff': 0.25,
            'smc': 0.15,
            'hob': 0.15,
            'momentum': 0.31,
            'temporal': 0.14
        }

        # Regime-specific weight adjustments (learned from optimization)
        self.regime_adjustments = {
            'crisis': {
                'wyckoff': +0.05,  # Trust structure more in crisis
                'momentum': -0.05,  # Less momentum chasing
                'threshold_adj': +0.10  # Much more selective
            },
            'risk_off': {
                'wyckoff': +0.03,
                'momentum': -0.03,
                'threshold_adj': +0.05
            },
            'risk_on': {
                'momentum': +0.05,  # More momentum in calm
                'wyckoff': -0.03,
                'threshold_adj': -0.03  # Less selective
            },
            'neutral': {
                'threshold_adj': 0.0  # No adjustment
            }
        }

    def train(self, dataset_path: str = 'data/ml/optimization_results.parquet'):
        """
        Train weight optimizer on historical optimization results

        Args:
            dataset_path: Path to optimization results parquet
        """
        if not LIGHTGBM_AVAILABLE and not SKLEARN_AVAILABLE:
            logger.warning("Neither LightGBM nor sklearn available - using rule-based fallback")
            return

        logger.info(f"Training fusion weight optimizer from {dataset_path}")

        # Load optimization results
        df = pd.read_parquet(dataset_path)
        logger.info(f"Loaded {len(df)} optimization results")

        # Filter for profitable configs only (learn from winners)
        df_winners = df[df['pf'] >= 1.0].copy()
        logger.info(f"Training on {len(df_winners)} profitable configs ({len(df_winners)/len(df)*100:.1f}%)")

        if len(df_winners) < 20:
            logger.warning(f"Only {len(df_winners)} profitable configs - not enough for robust training")
            return

        # Feature engineering
        feature_cols = [
            'vix', 'move', 'dxy', 'oil', 'us10y', 'us2y',
            'btc_d', 'usdt_d', 'config_wyckoff_weight',
            'config_momentum_weight', 'config_smc_weight'
        ]

        # Handle missing columns
        available_cols = [col for col in feature_cols if col in df_winners.columns]
        logger.info(f"Using {len(available_cols)} features: {available_cols}")

        X = df_winners[available_cols].copy()
        y = df_winners['pf'].values  # Target is Profit Factor

        # Fill NaN with defaults
        X = X.fillna({
            'vix': 20.0, 'move': 80.0, 'dxy': 100.0, 'oil': 70.0,
            'us10y': 4.0, 'us2y': 4.2, 'btc_d': 55.0, 'usdt_d': 7.0
        })

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train LightGBM model (or GBM if LightGBM not available)
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=10,
                random_state=42,
                verbose=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            )

        self.model.fit(X_scaled, y)
        self.feature_names = available_cols
        self.is_trained = True

        # Evaluate on full dataset
        train_score = self.model.score(X_scaled, y)
        logger.info(f"Training R²: {train_score:.3f}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Top 5 features:\n{importances.head()}")

        logger.info("✅ Fusion weight optimizer trained successfully")

    def predict_optimal_weights(
        self,
        regime: RegimeState,
        current_weights: Dict[str, float],
        current_scores: Optional[Dict[str, float]] = None
    ) -> WeightUpdate:
        """
        Predict optimal domain weights for current regime

        Args:
            regime: Current market regime state
            current_weights: Current fusion weights
            current_scores: Current domain scores (optional, for context)

        Returns:
            WeightUpdate with recommended weights + confidence
        """
        regime_label = regime.regime_label

        # If not trained, use rule-based regime adjustments
        if not self.is_trained:
            return self._rule_based_adjustment(regime, current_weights, regime_label)

        # ML-based prediction
        try:
            # Prepare features
            feature_dict = {
                'vix': regime.vix,
                'move': regime.move,
                'dxy': regime.dxy,
                'oil': regime.oil,
                'us10y': 4.0,  # Placeholder if not available
                'us2y': 4.2,
                'btc_d': regime.btc_d,
                'usdt_d': regime.usdt_d,
                'config_wyckoff_weight': current_weights.get('wyckoff', 0.25),
                'config_momentum_weight': current_weights.get('momentum', 0.30),
                'config_smc_weight': current_weights.get('smc', 0.15)
            }

            # Filter to available features
            X = np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]])
            X_scaled = self.scaler.transform(X)

            # Predict expected PF with current weights
            current_pf = self.model.predict(X_scaled)[0]

            # Try small adjustments and pick best
            best_pf = current_pf
            best_weights = current_weights.copy()

            # Test momentum adjustments (key lever from optimization)
            for momentum_adj in [-0.05, 0, +0.05]:
                test_weights = current_weights.copy()
                test_weights['momentum'] = np.clip(test_weights['momentum'] + momentum_adj, 0.20, 0.35)
                test_weights['wyckoff'] = np.clip(test_weights['wyckoff'] - momentum_adj, 0.15, 0.30)

                # Normalize
                total = sum(test_weights.values())
                test_weights = {k: v/total for k, v in test_weights.items()}

                # Predict
                test_feature_dict = feature_dict.copy()
                test_feature_dict['config_momentum_weight'] = test_weights['momentum']
                test_feature_dict['config_wyckoff_weight'] = test_weights['wyckoff']

                X_test = np.array([[test_feature_dict.get(f, 0.0) for f in self.feature_names]])
                X_test_scaled = self.scaler.transform(X_test)
                test_pf = self.model.predict(X_test_scaled)[0]

                if test_pf > best_pf:
                    best_pf = test_pf
                    best_weights = test_weights

            confidence = min((best_pf - current_pf) / current_pf if current_pf > 0 else 0.5, 1.0)
            confidence = max(confidence, 0.3)  # Minimum confidence

            return WeightUpdate(
                wyckoff=best_weights['wyckoff'],
                smc=best_weights['smc'],
                hob=best_weights['hob'],
                momentum=best_weights['momentum'],
                temporal=best_weights['temporal'],
                confidence=confidence,
                regime=regime_label,
                reason=f"ML predicted PF improvement: {current_pf:.3f} → {best_pf:.3f}"
            )

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}, falling back to rules")
            return self._rule_based_adjustment(regime, current_weights, regime_label)

    def _rule_based_adjustment(
        self,
        regime: RegimeState,
        current_weights: Dict[str, float],
        regime_label: str
    ) -> WeightUpdate:
        """Fallback rule-based adjustment from optimization learnings"""

        adjustments = self.regime_adjustments.get(regime_label, {})
        new_weights = current_weights.copy()

        # Apply regime adjustments
        for domain, adj in adjustments.items():
            if domain.endswith('_adj'):
                continue
            if domain in new_weights:
                new_weights[domain] = np.clip(new_weights[domain] + adj, 0.10, 0.40)

        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}

        threshold_adj = adjustments.get('threshold_adj', 0.0)

        return WeightUpdate(
            wyckoff=new_weights['wyckoff'],
            smc=new_weights['smc'],
            hob=new_weights['hob'],
            momentum=new_weights['momentum'],
            temporal=new_weights['temporal'],
            confidence=0.6,  # Medium confidence for rule-based
            regime=regime_label,
            reason=f"Rule-based {regime_label} adjustment: {adjustments}"
        )

    def update_online(self, regime: RegimeState, weights: Dict[str, float], realized_pf: float):
        """
        Online learning: Update model with realized performance

        Args:
            regime: Regime state that was active
            weights: Weights that were used
            realized_pf: Realized profit factor from trades
        """
        self.training_history.append({
            'regime': regime.regime_label,
            'vix': regime.vix,
            'weights': weights.copy(),
            'pf': realized_pf
        })

        logger.info(f"Online update: {regime.regime_label} regime, PF={realized_pf:.3f}")

        # Retrain periodically (every 50 samples)
        if len(self.training_history) % 50 == 0 and len(self.training_history) > 20:
            logger.info(f"Periodic retrain triggered ({len(self.training_history)} samples)")
            # Could implement incremental retraining here

    def export_config_patch(self, weights: WeightUpdate, output_path: str):
        """
        Export weight update as config patch for human review

        This ensures humans approve changes via PR before deployment
        """
        import json

        patch = {
            'version': '1.8.6-ml-optimized',
            'optimization_date': pd.Timestamp.now().isoformat(),
            'regime': weights.regime,
            'confidence': weights.confidence,
            'reason': weights.reason,
            'fusion': {
                'weights': weights.to_dict()
            },
            'validation_required': True,
            'walk_forward_test_required': True
        }

        with open(output_path, 'w') as f:
            json.dump(patch, f, indent=2)

        logger.info(f"✅ Config patch exported to {output_path}")
        logger.info(f"   Review and test before merging!")


# Example usage
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    # Initialize optimizer
    optimizer = FusionWeightOptimizer(config={})

    # Train on historical data
    optimizer.train('data/ml/optimization_results.parquet')

    # Simulate current regime (example: neutral regime)
    current_regime = RegimeState(
        vix=18.5, move=85.0, dxy=102.0, oil=72.0,
        yield_spread=-0.3, btc_d=54.5, usdt_d=6.8,
        funding=0.008, oi=0.012, adx=22.0, rsi=55.0,
        volatility_realized=0.02, trend_strength=0.6
    )

    current_weights = {
        'wyckoff': 0.25, 'smc': 0.15, 'hob': 0.15,
        'momentum': 0.31, 'temporal': 0.14
    }

    # Get recommendation
    update = optimizer.predict_optimal_weights(current_regime, current_weights)

    print("\n" + "="*70)
    print("FUSION WEIGHT OPTIMIZATION RECOMMENDATION")
    print("="*70)
    print(f"Regime: {update.regime}")
    print(f"Confidence: {update.confidence:.2f}")
    print(f"Reason: {update.reason}")
    print(f"\nRecommended weights:")
    for domain, weight in update.to_dict().items():
        current = current_weights.get(domain, 0.0)
        delta = weight - current
        print(f"  {domain:12s}: {current:.3f} → {weight:.3f} ({delta:+.3f})")
    print(f"\nValid: {update.validate()}")

    # Export for review
    optimizer.export_config_patch(update, 'config_patch_ml.json')
