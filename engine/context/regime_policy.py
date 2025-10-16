"""
Regime Policy for Bull Machine v1.9

Applies bounded adjustments to config based on classified regime.
Works with RegimeClassifier to adapt fusion thresholds, risk sizing, and weights.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RegimePolicy:
    """
    Applies regime-specific adjustments to trading config

    Takes regime classification output and returns bounded parameter deltas
    that preserve system stability while adapting to market conditions.
    """

    def __init__(self, policy_config: Dict[str, Any]):
        """
        Initialize regime policy

        Args:
            policy_config: Dict containing bounds and adjustment rules
        """
        self.cfg = policy_config
        self.enabled = policy_config.get("enabled", True)
        self.bounds = policy_config.get("bounds", {})

        logger.info(f"Regime policy initialized (enabled={self.enabled})")
        if self.enabled:
            logger.info(f"  Threshold deltas: {self.bounds.get('enter_threshold_delta', {})}")
            logger.info(f"  Risk multipliers: {self.bounds.get('risk_multiplier', {})}")

    @classmethod
    def load(cls, policy_path: str):
        """
        Load regime policy from JSON config

        Args:
            policy_path: Path to regime_policy.json

        Returns:
            RegimePolicy instance
        """
        if not Path(policy_path).exists():
            logger.warning(f"Policy not found: {policy_path}, using neutral defaults")
            return cls.neutral_policy()

        with open(policy_path) as f:
            policy_cfg = json.load(f)

        return cls(policy_cfg)

    @classmethod
    def neutral_policy(cls):
        """
        Create a neutral policy (no adjustments)

        Returns:
            RegimePolicy with zero adjustments
        """
        return cls({
            "enabled": False,
            "bounds": {
                "enter_threshold_delta": {"risk_on": 0.0, "neutral": 0.0, "risk_off": 0.0, "crisis": 0.0},
                "risk_multiplier": {"risk_on": 1.0, "neutral": 1.0, "risk_off": 1.0, "crisis": 1.0},
                "weight_nudges": {}
            }
        })

    def apply(self, base_cfg: Dict[str, Any], regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply regime-specific adjustments to base config

        Args:
            base_cfg: Base configuration dict
            regime_info: Output from RegimeClassifier.classify()

        Returns:
            Dict with adjustment deltas:
            {
                "enter_threshold_delta": float,
                "risk_multiplier": float,
                "weight_nudges": Dict[str, float],
                "regime": str,
                "confidence": float,
                "applied": bool
            }
        """
        if not self.enabled:
            return self._neutral_adjustment(regime_info)

        regime = regime_info.get("regime", "neutral")
        proba = regime_info.get("proba", {})
        confidence = proba.get(regime, 0.0)

        # Get bounded adjustments
        threshold_delta = self._get_threshold_delta(regime, confidence)
        risk_mult = self._get_risk_multiplier(regime, confidence)
        weight_nudges = self._get_weight_nudges(regime, confidence)

        adjustment = {
            "enter_threshold_delta": threshold_delta,
            "risk_multiplier": risk_mult,
            "weight_nudges": weight_nudges,
            "regime": regime,
            "confidence": confidence,
            "applied": True,
            "fallback": regime_info.get("fallback", False)
        }

        logger.info(f"Applied regime policy: {regime} (confidence={confidence:.2f})")
        logger.info(f"  Threshold delta: {threshold_delta:+.3f}")
        logger.info(f"  Risk multiplier: {risk_mult:.2f}x")
        if weight_nudges:
            logger.info(f"  Weight nudges: {weight_nudges}")

        return adjustment

    def _get_threshold_delta(self, regime: str, confidence: float) -> float:
        """
        Get fusion threshold adjustment for regime

        Args:
            regime: Regime label (risk_on/neutral/risk_off/crisis)
            confidence: Model confidence [0-1]

        Returns:
            Threshold delta in range [-0.10, +0.10]
        """
        bounds = self.bounds.get("enter_threshold_delta", {})
        base_delta = bounds.get(regime, 0.0)

        # Scale by confidence (require high confidence for large adjustments)
        min_confidence = self.cfg.get("min_confidence_for_adjustment", 0.60)

        if confidence < min_confidence:
            # Low confidence = use neutral adjustment
            return 0.0

        # Linear scaling: confidence 0.60 → 0%, confidence 1.0 → 100%
        confidence_scale = (confidence - min_confidence) / (1.0 - min_confidence)
        adjusted_delta = base_delta * confidence_scale

        # Enforce absolute bounds
        max_delta = self.cfg.get("max_threshold_delta", 0.10)
        return float(max(-max_delta, min(max_delta, adjusted_delta)))

    def _get_risk_multiplier(self, regime: str, confidence: float) -> float:
        """
        Get risk sizing multiplier for regime

        Args:
            regime: Regime label
            confidence: Model confidence [0-1]

        Returns:
            Risk multiplier in range [0.0, 1.5]
        """
        bounds = self.bounds.get("risk_multiplier", {})
        base_mult = bounds.get(regime, 1.0)

        min_confidence = self.cfg.get("min_confidence_for_adjustment", 0.60)

        if confidence < min_confidence:
            return 1.0  # Neutral

        # Interpolate between neutral (1.0) and target multiplier
        confidence_scale = (confidence - min_confidence) / (1.0 - min_confidence)
        adjusted_mult = 1.0 + (base_mult - 1.0) * confidence_scale

        # Enforce absolute bounds
        return float(max(0.0, min(1.5, adjusted_mult)))

    def _get_weight_nudges(self, regime: str, confidence: float) -> Dict[str, float]:
        """
        Get domain weight adjustments for regime

        Args:
            regime: Regime label
            confidence: Model confidence [0-1]

        Returns:
            Dict of weight deltas (e.g., {"wyckoff": +0.05, "momentum": -0.05})
        """
        bounds = self.bounds.get("weight_nudges", {})
        regime_nudges = bounds.get(regime, {})

        if not regime_nudges:
            return {}

        min_confidence = self.cfg.get("min_confidence_for_adjustment", 0.60)

        if confidence < min_confidence:
            return {}

        # Scale nudges by confidence
        confidence_scale = (confidence - min_confidence) / (1.0 - min_confidence)

        scaled_nudges = {}
        for domain, nudge in regime_nudges.items():
            scaled_nudges[domain] = float(nudge * confidence_scale)

        # Enforce max total weight shift
        max_total_shift = self.cfg.get("max_total_weight_shift", 0.15)
        total_shift = sum(abs(v) for v in scaled_nudges.values())

        if total_shift > max_total_shift:
            scale_factor = max_total_shift / total_shift
            scaled_nudges = {k: v * scale_factor for k, v in scaled_nudges.items()}

        return scaled_nudges

    def _neutral_adjustment(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return neutral adjustment (no changes)

        Args:
            regime_info: Regime classification output

        Returns:
            Dict with zero adjustments
        """
        return {
            "enter_threshold_delta": 0.0,
            "risk_multiplier": 1.0,
            "weight_nudges": {},
            "regime": regime_info.get("regime", "neutral"),
            "confidence": regime_info.get("proba", {}).get(regime_info.get("regime", "neutral"), 0.0),
            "applied": False,
            "fallback": regime_info.get("fallback", False)
        }


# Example usage and testing
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python regime_policy.py <policy_path>")
        print("\nTest with:")
        print("  python regime_policy.py configs/v19/regime_policy.json")
        sys.exit(1)

    policy_path = sys.argv[1]

    # Load policy
    try:
        policy = RegimePolicy.load(policy_path)
        print(f"\n✅ Loaded regime policy from {policy_path}")

        # Test adjustments for each regime
        test_cases = [
            {"regime": "risk_on", "proba": {"risk_on": 0.85, "neutral": 0.10, "risk_off": 0.03, "crisis": 0.02}},
            {"regime": "neutral", "proba": {"risk_on": 0.20, "neutral": 0.60, "risk_off": 0.15, "crisis": 0.05}},
            {"regime": "risk_off", "proba": {"risk_on": 0.05, "neutral": 0.15, "risk_off": 0.75, "crisis": 0.05}},
            {"regime": "crisis", "proba": {"risk_on": 0.01, "neutral": 0.05, "risk_off": 0.14, "crisis": 0.80}},
        ]

        base_cfg = {
            "fusion": {
                "entry_threshold_confidence": 0.65,
                "weights": {
                    "wyckoff": 0.25,
                    "smc": 0.15,
                    "liquidity": 0.15,
                    "momentum": 0.31,
                    "temporal": 0.14
                }
            }
        }

        print("\n📊 Test Regime Adjustments:")
        for test_case in test_cases:
            regime = test_case["regime"]
            print(f"\n{'='*60}")
            print(f"Regime: {regime.upper()}")
            print(f"Confidence: {test_case['proba'][regime]:.1%}")
            print(f"{'='*60}")

            adjustment = policy.apply(base_cfg, test_case)

            print(f"  Threshold delta: {adjustment['enter_threshold_delta']:+.3f}")
            print(f"    → New threshold: {0.65 + adjustment['enter_threshold_delta']:.3f}")
            print(f"  Risk multiplier: {adjustment['risk_multiplier']:.2f}x")
            if adjustment['weight_nudges']:
                print(f"  Weight nudges:")
                for domain, nudge in adjustment['weight_nudges'].items():
                    print(f"    {domain:12s}: {nudge:+.3f}")
            else:
                print(f"  Weight nudges: (none)")

        print("\n✅ Regime policy test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
