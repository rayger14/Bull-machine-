"""
Adaptive Fusion - Regime-Aware Parameter Morphing

Blends fusion weights, entry gates, exit policies, and sizing based on
regime probability distributions (no discrete switching).

Designed for smooth transitions and Specter-readiness.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def ema_smooth(prev_probs: Dict[str, float], curr_probs: Dict[str, float], alpha: float = 0.2) -> Dict[str, float]:
    """
    EMA-smooth regime probabilities for stability

    Args:
        prev_probs: Previous regime probabilities (or None if first bar)
        curr_probs: Current regime probabilities from classifier
        alpha: Smoothing factor (0.0 = all previous, 1.0 = all current)

    Returns:
        Smoothed regime probabilities
    """
    if prev_probs is None:
        return curr_probs.copy()

    smoothed = {}
    for regime in curr_probs:
        prev_val = prev_probs.get(regime, curr_probs[regime])
        smoothed[regime] = alpha * curr_probs[regime] + (1 - alpha) * prev_val

    # Renormalize to sum to 1.0
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {k: v / total for k, v in smoothed.items()}

    return smoothed


def adapt_weights(base_profiles: Dict[str, Dict[str, float]],
                  regime_probs: Dict[str, float],
                  min_weight: float = 0.05) -> Dict[str, float]:
    """
    Blend fusion weights across regime profiles using probabilities

    Args:
        base_profiles: Dict[regime_name, Dict[component, weight]]
                      e.g. {"risk_on": {"wyckoff": 0.46, "liquidity": 0.26, ...}, ...}
        regime_probs: Current regime probabilities (smoothed)
        min_weight: Floor for any component (prevents zeroing out)

    Returns:
        Dict[component, blended_weight] (sums to 1.0)
    """
    # Start with zero weights for all components
    component_names = next(iter(base_profiles.values())).keys()
    blended = {k: 0.0 for k in component_names}

    # Convex combination: sum(regime_prob * regime_weights)
    for regime, prob in regime_probs.items():
        if regime not in base_profiles:
            continue
        for component, weight in base_profiles[regime].items():
            blended[component] += prob * weight

    # Apply floor (guardrail against zero weights)
    blended = {k: max(min_weight, v) for k, v in blended.items()}

    # Renormalize to sum to 1.0
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended


def adapt_gates(base_profiles: Dict[str, Dict[str, float]],
                regime_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Blend entry gate thresholds across regime profiles

    Args:
        base_profiles: Dict[regime_name, Dict[gate_name, threshold]]
                      e.g. {"risk_on": {"min_liquidity": 0.16, "final_fusion_floor": 0.32}, ...}
        regime_probs: Current regime probabilities (smoothed)

    Returns:
        Dict[gate_name, blended_threshold]
    """
    # Start with zero for all gates
    gate_names = next(iter(base_profiles.values())).keys()
    blended = {k: 0.0 for k in gate_names}

    # Weighted average across regimes
    for regime, prob in regime_probs.items():
        if regime not in base_profiles:
            logger.warning(f"[ADAPT_GATES] Regime '{regime}' (prob={prob:.3f}) not in base_profiles (keys={list(base_profiles.keys())})")
            continue
        for gate, threshold in base_profiles[regime].items():
            blended[gate] += prob * threshold
            logger.debug(f"[ADAPT_GATES] regime={regime}, prob={prob:.3f}, gate={gate}, threshold={threshold:.3f}, blended[{gate}]={blended[gate]:.3f}")

    logger.info(f"[ADAPT_GATES] Final blended gates: {blended}")
    return blended


def adapt_exit_params(base_profiles: Dict[str, Dict[str, float]],
                     regime_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Blend exit policy parameters across regime profiles

    Args:
        base_profiles: Dict[regime_name, Dict[param_name, value]]
                      e.g. {"risk_on": {"trail_atr": 1.25, "max_bars": 92}, ...}
        regime_probs: Current regime probabilities (smoothed)

    Returns:
        Dict[param_name, blended_value]
    """
    # Start with zero for all params
    param_names = next(iter(base_profiles.values())).keys()
    blended = {k: 0.0 for k in param_names}

    # Weighted average across regimes
    for regime, prob in regime_probs.items():
        if regime not in base_profiles:
            continue
        for param, value in base_profiles[regime].items():
            blended[param] += prob * value

    return blended


def regime_size_mult(sizing_curve: Dict[str, float],
                     regime_probs: Dict[str, float],
                     limits: Dict[str, float]) -> float:
    """
    Compute regime-aware sizing multiplier

    Args:
        sizing_curve: Dict[regime_name, multiplier]
                     e.g. {"risk_on": 1.20, "neutral": 0.90, ...}
        regime_probs: Current regime probabilities (smoothed)
        limits: {"min": 0.6, "max": 1.35}

    Returns:
        Blended size multiplier (clipped to limits)
    """
    # Weighted average
    mult = sum(regime_probs.get(r, 0.0) * m for r, m in sizing_curve.items())

    # Clip to limits
    return max(limits["min"], min(limits["max"], mult))


def adapt_ml_threshold(thresholds_by_regime: Dict[str, float],
                      regime_probs: Dict[str, float]) -> float:
    """
    Blend ML filter threshold across regime probabilities

    Args:
        thresholds_by_regime: Dict[regime_name, threshold]
                             e.g. {"risk_on": 0.30, "neutral": 0.45, ...}
        regime_probs: Current regime probabilities (smoothed)

    Returns:
        Blended threshold value
    """
    # Weighted average
    threshold = sum(regime_probs.get(r, 0.0) * t for r, t in thresholds_by_regime.items())

    # Sanity bounds [0.0, 1.0]
    return max(0.0, min(1.0, threshold))


class AdaptiveFusion:
    """
    Stateful adaptive fusion coordinator

    Maintains EMA state for regime probabilities and provides
    all adaptive blending functions in one place.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adaptive fusion

        Args:
            config: Adaptive fusion config dict containing:
                   - fusion_regime_profiles
                   - gates_regime_profiles
                   - exit_regime_profiles
                   - sizing_regime_curve
                   - ml_thresholds_by_regime (optional)
                   - fusion_adapt: {"enable": bool, "ema_alpha": float, "min_weight": float}
        """
        self.cfg = config
        self.enabled = config.get("fusion_adapt", {}).get("enable", True)
        self.ema_alpha = config.get("fusion_adapt", {}).get("ema_alpha", 0.2)
        self.min_weight = config.get("fusion_adapt", {}).get("min_weight", 0.05)

        # State: smoothed regime probabilities
        self.regime_probs_ema = None

        logger.info(f"AdaptiveFusion initialized (enabled={self.enabled}, ema_alpha={self.ema_alpha})")

    def update(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update with new regime classification and return adapted parameters

        Args:
            regime_info: Output from RegimeClassifier.classify()
                        {"regime": str, "proba": Dict[str, float], ...}

        Returns:
            Dict with:
                - fusion_weights: Dict[component, weight]
                - gates: Dict[gate_name, threshold]
                - exit_params: Dict[param_name, value]
                - size_mult: float
                - ml_threshold: float (if configured)
                - regime_probs_ema: Dict[regime, probability] (for logging)
        """
        if not self.enabled:
            return self._neutral_output()

        # Smooth regime probabilities
        curr_probs = regime_info.get("proba", {})
        self.regime_probs_ema = ema_smooth(self.regime_probs_ema, curr_probs, self.ema_alpha)

        # DEBUG: Log probabilities to diagnose zero-value gates
        logger.info(f"[ADAPTIVE_UPDATE] curr_probs={curr_probs}, ema_probs={self.regime_probs_ema}")

        # Adapt all components
        result = {
            "regime_probs_ema": self.regime_probs_ema.copy(),
            "regime": max(self.regime_probs_ema, key=self.regime_probs_ema.get)
        }

        # Fusion weights
        if "fusion_regime_profiles" in self.cfg:
            result["fusion_weights"] = adapt_weights(
                self.cfg["fusion_regime_profiles"],
                self.regime_probs_ema,
                self.min_weight
            )

        # Entry gates
        if "gates_regime_profiles" in self.cfg:
            result["gates"] = adapt_gates(
                self.cfg["gates_regime_profiles"],
                self.regime_probs_ema
            )

        # Exit policy
        if "exit_regime_profiles" in self.cfg:
            result["exit_params"] = adapt_exit_params(
                self.cfg["exit_regime_profiles"],
                self.regime_probs_ema
            )

        # Sizing
        if "sizing_regime_curve" in self.cfg:
            result["size_mult"] = regime_size_mult(
                self.cfg["sizing_regime_curve"],
                self.regime_probs_ema,
                self.cfg.get("sizing_limits", {"min": 0.6, "max": 1.35})
            )

        # ML threshold
        if "ml_thresholds_by_regime" in self.cfg:
            result["ml_threshold"] = adapt_ml_threshold(
                self.cfg["ml_thresholds_by_regime"],
                self.regime_probs_ema
            )

        return result

    def _neutral_output(self) -> Dict[str, Any]:
        """Return neutral/disabled output (no adaptations)"""
        return {
            "regime": "neutral",
            "regime_probs_ema": {"neutral": 1.0},
            "fusion_weights": None,
            "gates": None,
            "exit_params": None,
            "size_mult": 1.0,
            "ml_threshold": None
        }
