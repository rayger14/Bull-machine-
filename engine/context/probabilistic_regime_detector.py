"""
Probabilistic Regime Detection System
======================================

3-output probabilistic system replacing hard regime labels:

1. crisis_prob [0-1]: Probability of crisis regime
2. risk_temperature [0-1]: Continuous aggressiveness score (0=cold/defensive, 1=hot/aggressive)
3. instability_score [0-1]: Probability of regime change / choppy conditions

Based on HMM methodology from research (Medium: Regime-Based Portfolio Allocation).

Architecture:
- crisis_prob: From LogisticRegimeModel v4 (already trained, just expose probabilities)
- risk_temperature: Weighted composite of vol, trend, drawdown, momentum, liquidity
- instability_score: Composite of regime flip frequency, vol-of-vol, ADX collapse, wick rejection

Author: Claude Code
Date: 2026-01-27
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ProbabilisticRegimeDetector:
    """
    Probabilistic regime detection using continuous outputs.

    Replaces hard regime labels with 3 continuous scores:
    - crisis_prob: P(crisis) from LogisticRegimeModel
    - risk_temperature: Aggressiveness score [0-1]
    - instability_score: P(regime change) [0-1]
    """

    def __init__(
        self,
        crisis_model,  # LogisticRegimeModel instance
        crisis_threshold: float = 0.15,  # Minimum P(crisis) to trigger risk controls
        temperature_weights: Optional[Dict[str, float]] = None,
        instability_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize probabilistic detector.

        Args:
            crisis_model: Trained LogisticRegimeModel for crisis detection
            crisis_threshold: Minimum crisis_prob to trigger emergency controls (default 0.15)
            temperature_weights: Feature weights for risk_temperature calculation
            instability_weights: Feature weights for instability_score calculation
        """
        self.crisis_model = crisis_model
        self.crisis_threshold = crisis_threshold

        # Default weights for risk_temperature
        self.temperature_weights = temperature_weights or {
            'volatility': 0.40,      # Inverse: low vol = hotter
            'trend_strength': 0.25,  # Strong trend = hotter
            'drawdown': 0.15,        # Shallow drawdown = hotter
            'momentum': 0.10,        # Positive momentum = hotter
            'liquidity': 0.10        # Low stress = hotter
        }

        # Default weights for instability_score
        self.instability_weights = instability_weights or {
            'regime_flips': 0.30,    # Regime transitions in 48h
            'vol_of_vol': 0.25,      # RV_7 std deviation
            'adx_collapse': 0.20,    # Low ADX + high RV = chop
            'wick_rejection': 0.15,  # High wick ratio = indecision
            'volume_variance': 0.10  # Volume inconsistency
        }

        # State tracking for regime flip detection
        self.regime_history = []  # Track last 48 regime states
        self.max_history = 48  # 48 hours for flip detection

    def detect(self, features: pd.Series) -> Dict[str, float]:
        """
        Detect regime probabilities for a single bar.

        Args:
            features: Series with all required features

        Returns:
            Dict with:
                - crisis_prob: [0-1] probability of crisis
                - risk_temperature: [0-1] aggressiveness score
                - instability_score: [0-1] probability of regime change
                - metadata: Additional info for monitoring
        """
        # 1. Crisis probability (from trained model)
        crisis_prob = self._get_crisis_probability(features)

        # 2. Risk temperature (continuous aggressiveness)
        risk_temperature = self._calculate_risk_temperature(features)

        # 3. Instability score (regime change probability)
        instability_score = self._calculate_instability_score(features, crisis_prob)

        # Update regime history for flip detection
        self._update_regime_history(crisis_prob, risk_temperature)

        # Metadata for monitoring
        metadata = {
            'crisis_emergency': crisis_prob > self.crisis_threshold,
            'temperature_level': self._interpret_temperature(risk_temperature),
            'instability_level': self._interpret_instability(instability_score),
            'regime_flips_48h': self._count_recent_flips(),
        }

        return {
            'crisis_prob': float(crisis_prob),
            'risk_temperature': float(risk_temperature),
            'instability_score': float(instability_score),
            'metadata': metadata
        }

    def _get_crisis_probability(self, features: pd.Series) -> float:
        """
        Get crisis probability from LogisticRegimeModel.

        Args:
            features: Feature series

        Returns:
            P(crisis) in [0, 1]
        """
        try:
            # Extract model object and metadata from dict if needed
            if isinstance(self.crisis_model, dict):
                model = self.crisis_model.get('model')
                feature_order = self.crisis_model.get('feature_order', [])
                scaler = self.crisis_model.get('scaler')
            else:
                model = self.crisis_model
                feature_order = getattr(model, 'feature_names_in_', None)
                scaler = None

            # If model has feature_order, extract only those features in correct order
            if feature_order:
                # Extract required features, fill missing with 0
                feature_values = []
                for feat in feature_order:
                    if feat in features.index:
                        feature_values.append(features[feat])
                    else:
                        logger.debug(f"Missing feature '{feat}' for regime model, using 0.0")
                        feature_values.append(0.0)

                # Create DataFrame with correct features in correct order
                X = pd.DataFrame([feature_values], columns=feature_order)
            else:
                # Fallback: use all features from series
                X = features.to_frame().T

            # Apply scaling if available
            if scaler is not None:
                X_scaled = scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns)

            # Get probabilities
            proba = model.predict_proba(X)

            # Extract crisis probability (first class)
            crisis_prob = proba[0, 0]  # Assuming 'crisis' is index 0

            return np.clip(crisis_prob, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Crisis probability extraction failed: {e}")
            return 0.0

    def _calculate_risk_temperature(self, features: pd.Series) -> float:
        """
        Calculate risk temperature (continuous aggressiveness score).

        Formula:
        temperature = weighted_sum([
            volatility_score (inverse),
            trend_strength,
            drawdown_score (inverse),
            momentum_score,
            liquidity_score
        ])

        Args:
            features: Feature series

        Returns:
            Risk temperature in [0, 1]
        """
        # 1. Volatility score (inverse: low vol = hot)
        rv_7 = features.get('RV_7', 0.05)
        vol_score = 1.0 - min(rv_7 / 0.15, 1.0)  # Normalize: 0.15 = max vol

        # 2. Trend strength (high ADX = hot)
        adx = features.get('adx_14', 20)
        trend_score = min(adx / 40.0, 1.0)  # Normalize: 40 = strong trend

        # 3. Drawdown score (inverse: shallow = hot)
        drawdown = features.get('drawdown_persistence', 0.9)
        drawdown_score = 1.0 - drawdown  # 0 = no drawdown = hot

        # 4. Momentum score (positive returns = hot)
        returns_90d = features.get('returns_90d', 0.0)
        momentum_score = (returns_90d + 0.5) / 1.0  # Map [-0.5, 0.5] to [0, 1]
        momentum_score = np.clip(momentum_score, 0.0, 1.0)

        # 5. Liquidity score (low volume spike = hot)
        volume_z = features.get('volume_z_7d', 0.0)
        liquidity_score = 1.0 - min(abs(volume_z) / 5.0, 1.0)  # High spike = cold

        # Weighted composite
        temperature = (
            self.temperature_weights['volatility'] * vol_score +
            self.temperature_weights['trend_strength'] * trend_score +
            self.temperature_weights['drawdown'] * drawdown_score +
            self.temperature_weights['momentum'] * momentum_score +
            self.temperature_weights['liquidity'] * liquidity_score
        )

        return np.clip(temperature, 0.0, 1.0)

    def _calculate_instability_score(self, features: pd.Series, crisis_prob: float) -> float:
        """
        Calculate instability score (probability of regime change / chop).

        High instability means:
        - Frequent regime flips
        - Volatile volatility (vol-of-vol)
        - Low trend + high vol (chop)
        - High wick rejection (indecision)
        - Inconsistent volume

        Args:
            features: Feature series
            crisis_prob: Current crisis probability

        Returns:
            Instability score in [0, 1]
        """
        # 1. Regime flip frequency (from history)
        flip_count = self._count_recent_flips()
        flip_score = min(flip_count / 5.0, 1.0)  # 5+ flips in 48h = max

        # 2. Vol-of-vol (if available in features)
        vol_of_vol = features.get('vol_of_vol', None)
        if vol_of_vol is not None:
            vov_score = min(vol_of_vol / 0.05, 1.0)
        else:
            # Fallback: estimate from RV_7 variance
            vov_score = 0.5  # Neutral if not available

        # 3. ADX collapse (low trend + high vol = chop)
        adx = features.get('adx_14', 20)
        rv_7 = features.get('RV_7', 0.05)
        adx_collapse_score = 1.0 if (adx < 20 and rv_7 > 0.06) else 0.0

        # 4. Wick rejection (if available)
        wick_ratio = features.get('wick_ratio', None)
        if wick_ratio is not None:
            wick_score = min(wick_ratio / 0.5, 1.0)
        else:
            wick_score = 0.5  # Neutral if not available

        # 5. Volume variance (if available)
        vol_variance = features.get('volume_variance', None)
        if vol_variance is not None:
            vol_var_score = min(vol_variance / 2.0, 1.0)
        else:
            vol_var_score = 0.5  # Neutral if not available

        # Weighted composite
        instability = (
            self.instability_weights['regime_flips'] * flip_score +
            self.instability_weights['vol_of_vol'] * vov_score +
            self.instability_weights['adx_collapse'] * adx_collapse_score +
            self.instability_weights['wick_rejection'] * wick_score +
            self.instability_weights['volume_variance'] * vol_var_score
        )

        return np.clip(instability, 0.0, 1.0)

    def _update_regime_history(self, crisis_prob: float, risk_temperature: float):
        """
        Update regime history for flip detection.

        Args:
            crisis_prob: Current crisis probability
            risk_temperature: Current risk temperature
        """
        # Simplified regime state (for flip counting)
        if crisis_prob > 0.25:
            regime_state = 'crisis'
        elif risk_temperature > 0.65:
            regime_state = 'hot'
        elif risk_temperature < 0.35:
            regime_state = 'cold'
        else:
            regime_state = 'neutral'

        # Add to history
        self.regime_history.append(regime_state)

        # Keep only last max_history entries
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)

    def _count_recent_flips(self) -> int:
        """
        Count regime flips in recent history (48h).

        Returns:
            Number of regime state changes
        """
        if len(self.regime_history) < 2:
            return 0

        flips = 0
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i] != self.regime_history[i-1]:
                flips += 1

        return flips

    def _interpret_temperature(self, temperature: float) -> str:
        """Interpret temperature level."""
        if temperature < 0.3:
            return 'cold_defensive'
        elif temperature < 0.5:
            return 'cool_cautious'
        elif temperature < 0.7:
            return 'warm_normal'
        else:
            return 'hot_aggressive'

    def _interpret_instability(self, instability: float) -> str:
        """Interpret instability level."""
        if instability < 0.3:
            return 'stable'
        elif instability < 0.6:
            return 'moderate'
        else:
            return 'unstable_choppy'

    def get_soft_controls(self, regime_state: Dict[str, float]) -> Dict[str, float]:
        """
        Get soft controls for position sizing and trade frequency.

        Args:
            regime_state: Dict with crisis_prob, risk_temperature, instability_score

        Returns:
            Dict with position_size_multiplier and trade_frequency_multiplier
        """
        crisis_prob = regime_state['crisis_prob']
        risk_temp = regime_state['risk_temperature']
        instability = regime_state['instability_score']

        # Position size: Reduce for crisis, scale with temperature
        # Formula: size = (1 - crisis_prob * 0.85) * risk_temperature
        crisis_penalty = 1.0 - (crisis_prob * 0.85)  # Max 85% reduction
        position_multiplier = crisis_penalty * risk_temp

        # Trade frequency: Reduce for instability
        # Formula: frequency = 1 - instability * 0.60
        freq_multiplier = 1.0 - (instability * 0.60)  # Max 60% reduction

        return {
            'position_size_multiplier': float(np.clip(position_multiplier, 0.05, 1.0)),
            'trade_frequency_multiplier': float(np.clip(freq_multiplier, 0.20, 1.0)),
            'crisis_gate': crisis_prob > self.crisis_threshold  # Hard gate for emergencies
        }
