#!/usr/bin/env python3
"""
Meta-Ensemble Model: ML-Driven Archetype-Baseline Fusion

Treats archetypes as feature sources (not final signals) and learns when they
add value on top of baseline strategies using regime-specific ML models.

Architecture:
- Baselines provide "infantry" (consistent simple edge)
- Archetypes provide "special forces" (filters, multipliers, veto power)
- Meta-model learns optimal archetype-baseline combinations
- Regime routing ensures appropriate model selection

Version: 1.0
Author: System Architect
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import pickle

logger = logging.getLogger(__name__)


class MetaEnsemble:
    """
    ML-driven ensemble that fuses baseline signals with archetype scores.

    Responsibilities:
    - Route to regime-specific meta-models
    - Predict archetype multiplier for baseline signals
    - Compute ensemble signal and confidence
    - Log predictions for monitoring

    Usage:
        ensemble = MetaEnsemble.load(config_path="configs/meta_ensemble.json")
        signal, multiplier, metadata = ensemble.predict(features)
    """

    VERSION = "meta_ensemble@v1.0"

    def __init__(
        self,
        models: Dict[str, Any],
        feature_columns: List[str],
        regime_classifier: Optional[Any] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize meta-ensemble.

        Args:
            models: Dict of {regime: trained_model}
            feature_columns: Ordered list of feature names expected by models
            regime_classifier: Optional regime classifier (if not in features)
            config: Optional configuration dict
        """
        self.models = models
        self.feature_columns = feature_columns
        self.regime_classifier = regime_classifier
        self.config = config or {}

        # Multiplier bounds (prevent extreme values)
        self.min_multiplier = self.config.get('min_multiplier', 0.0)
        self.max_multiplier = self.config.get('max_multiplier', 2.0)

        # Fallback regime if classification fails
        self.fallback_regime = self.config.get('fallback_regime', 'neutral')

        logger.info(f"[MetaEnsemble] Initialized {self.VERSION}")
        logger.info(f"[MetaEnsemble] Regimes: {list(self.models.keys())}")
        logger.info(f"[MetaEnsemble] Features: {len(self.feature_columns)}")
        logger.info(f"[MetaEnsemble] Multiplier range: [{self.min_multiplier}, {self.max_multiplier}]")

    @classmethod
    def load(cls, config_path: str):
        """
        Load meta-ensemble from config file.

        Args:
            config_path: Path to JSON config with model paths

        Returns:
            MetaEnsemble instance
        """
        import json

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load regime-specific models
        models = {}
        for regime, model_path in config['model_paths'].items():
            model_path = Path(model_path)
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    models[regime] = pickle.load(f)
                logger.info(f"[MetaEnsemble] Loaded model for regime: {regime}")
            else:
                logger.warning(f"[MetaEnsemble] Model not found: {model_path}")

        # Load feature columns
        feature_columns = config.get('feature_columns', [])

        # Load regime classifier (optional)
        regime_classifier = None
        if 'regime_classifier_path' in config:
            from engine.context.regime_classifier import RegimeClassifier
            regime_classifier = RegimeClassifier.load(
                config['regime_classifier_path'],
                feature_order=config.get('regime_features', [])
            )

        return cls(
            models=models,
            feature_columns=feature_columns,
            regime_classifier=regime_classifier,
            config=config
        )

    def predict(
        self,
        features: Dict[str, Any],
        baseline_signal: Optional[int] = None
    ) -> Tuple[int, float, Dict]:
        """
        Predict ensemble signal using meta-model.

        Args:
            features: Dict with baseline, archetype, regime features
            baseline_signal: Optional baseline signal (extracted from features if not provided)

        Returns:
            (ensemble_signal, archetype_multiplier, metadata)
        """
        # Extract baseline signal if not provided
        if baseline_signal is None:
            baseline_signal = self._extract_baseline_signal(features)

        # Classify regime (or use provided regime_label)
        regime = features.get('regime_label')
        if regime is None and self.regime_classifier is not None:
            regime_result = self.regime_classifier.classify(features)
            regime = regime_result['regime']
            features['regime_label'] = regime

        # Get regime-specific model (fallback to neutral if not found)
        model = self.models.get(regime, self.models.get(self.fallback_regime))
        if model is None:
            logger.error(f"[MetaEnsemble] No model for regime: {regime}, no fallback available")
            return baseline_signal, 1.0, {'error': 'no_model', 'regime': regime}

        # Prepare feature vector
        X = self._prepare_features(features)

        # Predict archetype multiplier
        try:
            # Model outputs raw multiplier prediction
            multiplier = float(model.predict(X)[0])

            # Clip to bounds
            multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))

        except Exception as e:
            logger.error(f"[MetaEnsemble] Prediction failed: {e}")
            multiplier = 1.0  # Neutral fallback (no archetype adjustment)

        # Compute ensemble signal
        ensemble_signal = int(np.sign(baseline_signal * multiplier))

        # Build metadata
        metadata = {
            'regime': regime,
            'baseline_signal': baseline_signal,
            'archetype_multiplier': multiplier,
            'ensemble_signal': ensemble_signal,
            'model_version': self.VERSION,
            'feature_count': len(self.feature_columns)
        }

        return ensemble_signal, multiplier, metadata

    def _extract_baseline_signal(self, features: Dict[str, Any]) -> int:
        """
        Extract baseline signal from features.

        Priority order:
        1. sma_crossover_signal (most reliable)
        2. rsi_meanrev_signal (secondary)
        3. sma_trend_signal (fallback)

        Returns:
            Baseline signal {-1, 0, 1}
        """
        # Try each baseline in priority order
        for baseline_key in ['sma_crossover_signal', 'rsi_meanrev_signal', 'sma_trend_signal']:
            signal = features.get(baseline_key, 0)
            if signal != 0:
                return int(signal)

        # No active baseline signal
        return 0

    def _prepare_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare feature vector for model prediction.

        Args:
            features: Raw feature dict

        Returns:
            DataFrame with features in correct order
        """
        # Extract features in correct order
        feature_values = []
        for col in self.feature_columns:
            value = features.get(col, 0.0)  # Default to 0 if missing

            # Handle special cases
            if pd.isna(value):
                value = 0.0
            elif isinstance(value, bool):
                value = 1.0 if value else 0.0
            elif isinstance(value, str):
                # Encode categorical features (simple label encoding)
                value = self._encode_categorical(col, value)

            feature_values.append(float(value))

        # Create DataFrame
        X = pd.DataFrame([feature_values], columns=self.feature_columns)
        return X

    def _encode_categorical(self, feature_name: str, value: str) -> float:
        """
        Encode categorical features as numeric.

        Args:
            feature_name: Feature name
            value: Categorical value

        Returns:
            Numeric encoding
        """
        # Regime encoding
        if 'regime' in feature_name.lower():
            regime_map = {
                'risk_on': 1.0,
                'neutral': 0.0,
                'risk_off': -1.0,
                'crisis': -2.0
            }
            return regime_map.get(value.lower(), 0.0)

        # Volatility regime encoding
        if 'volatility' in feature_name.lower():
            vol_map = {
                'low': 0.0,
                'medium': 0.5,
                'high': 1.0
            }
            return vol_map.get(value.lower(), 0.5)

        # Wyckoff phase encoding
        if 'wyckoff_phase' in feature_name.lower():
            phase_map = {
                'A': -2.0,  # Accumulation start
                'B': -1.0,  # Building cause
                'C': 0.0,   # Testing
                'D': 1.0,   # Last point
                'E': 2.0,   # Markup
                'neutral': 0.0
            }
            return phase_map.get(value, 0.0)

        # Default: hash to [0, 1]
        return hash(value) % 100 / 100.0

    def evaluate(
        self,
        test_df: pd.DataFrame,
        target_col: str = 'target'
    ) -> Dict[str, float]:
        """
        Evaluate ensemble on test data.

        Args:
            test_df: Test DataFrame with features and target
            target_col: Name of target column

        Returns:
            Dict of evaluation metrics
        """
        predictions = []
        actuals = []

        for idx, row in test_df.iterrows():
            features = row.to_dict()
            _, multiplier, _ = self.predict(features)
            predictions.append(multiplier)
            actuals.append(row[target_col])

        # Compute metrics
        from sklearn.metrics import mean_squared_error, r2_score

        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        # Compute directional accuracy (if target is binary)
        if set(actuals).issubset({0, 1}):
            pred_binary = [1 if p > 0.5 else 0 for p in predictions]
            accuracy = sum(p == a for p, a in zip(pred_binary, actuals)) / len(actuals)
        else:
            accuracy = None

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'accuracy': accuracy
        }

        logger.info(f"[MetaEnsemble] Evaluation metrics: {metrics}")
        return metrics

    def get_feature_importance(self, regime: str) -> Optional[pd.DataFrame]:
        """
        Get feature importance for a specific regime model.

        Args:
            regime: Regime name

        Returns:
            DataFrame with feature importance or None if not available
        """
        model = self.models.get(regime)
        if model is None:
            logger.warning(f"[MetaEnsemble] No model for regime: {regime}")
            return None

        # Try to extract feature importance (LightGBM/XGBoost)
        try:
            if hasattr(model, 'feature_importance'):
                importance = model.feature_importance(importance_type='gain')
            elif hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                logger.warning("[MetaEnsemble] Model does not support feature importance")
                return None

            df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

            return df

        except Exception as e:
            logger.error(f"[MetaEnsemble] Failed to extract feature importance: {e}")
            return None


class EnsembleBacktester:
    """
    Backtest meta-ensemble against baseline-only and archetype-only strategies.

    Compares:
    - Baseline-only (no archetypes)
    - Archetype-only (no baselines)
    - Ensemble (baseline * archetype_multiplier)
    """

    def __init__(self, ensemble: MetaEnsemble):
        """
        Initialize backtester.

        Args:
            ensemble: Trained MetaEnsemble instance
        """
        self.ensemble = ensemble

    def backtest(
        self,
        test_df: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Run backtest comparing strategies.

        Args:
            test_df: Test data with features and price
            initial_capital: Starting capital

        Returns:
            DataFrame with PnL curves for each strategy
        """
        results = []

        # Initialize capital for each strategy
        capital = {
            'baseline_only': initial_capital,
            'ensemble': initial_capital,
            'buy_and_hold': initial_capital
        }

        position = {
            'baseline_only': 0.0,
            'ensemble': 0.0,
            'buy_and_hold': 1.0  # Always long
        }

        for idx, row in test_df.iterrows():
            features = row.to_dict()
            price = row['close']

            # Baseline-only strategy
            baseline_signal = self.ensemble._extract_baseline_signal(features)

            # Ensemble strategy
            ensemble_signal, multiplier, _ = self.ensemble.predict(features)

            # Update positions (simplified - assumes instant execution)
            position['baseline_only'] = baseline_signal
            position['ensemble'] = ensemble_signal

            # Compute returns
            if idx > 0:
                prev_price = test_df.iloc[idx - 1]['close']
                ret = (price - prev_price) / prev_price

                for strategy in capital.keys():
                    capital[strategy] *= (1 + ret * position[strategy])

            # Log results
            results.append({
                'timestamp': row['timestamp'],
                'price': price,
                'baseline_capital': capital['baseline_only'],
                'ensemble_capital': capital['ensemble'],
                'bnh_capital': capital['buy_and_hold'],
                'baseline_signal': baseline_signal,
                'ensemble_signal': ensemble_signal,
                'archetype_multiplier': multiplier
            })

        return pd.DataFrame(results)

    def compute_metrics(self, backtest_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute performance metrics for each strategy.

        Args:
            backtest_df: Backtest results DataFrame

        Returns:
            Dict of {strategy: metrics}
        """
        metrics = {}

        for strategy in ['baseline', 'ensemble', 'bnh']:
            capital_col = f'{strategy}_capital'
            returns = backtest_df[capital_col].pct_change().dropna()

            # Sharpe ratio (annualized, assuming hourly data)
            sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24)

            # Max drawdown
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Win rate
            win_rate = (returns > 0).sum() / len(returns)

            # Total return
            total_return = (backtest_df[capital_col].iloc[-1] / backtest_df[capital_col].iloc[0]) - 1

            metrics[strategy] = {
                'total_return': total_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(returns)
            }

        return metrics


# Example usage and testing
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    print("Meta-Ensemble Model - Example Usage")
    print("=" * 50)
    print("\nThis module requires:")
    print("1. Trained LightGBM/XGBoost models per regime")
    print("2. Feature columns specification")
    print("3. Test data with all required features")
    print("\nSee docs/ML_ENSEMBLE_FRAMEWORK_DESIGN.md for details")
