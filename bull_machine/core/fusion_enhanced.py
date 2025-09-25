"""
Bull Machine v1.5.0 - Enhanced Fusion Engine
Integrates v1.5.0 alphas with v1.4.2 baseline while preserving existing functionality
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from bull_machine.core.telemetry import log_telemetry
from bull_machine.core.config_loader import load_config, is_feature_enabled


class FusionEngineV141:
    """Base fusion engine v1.4.1 (minimal implementation for inheritance)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config.get("layer_weights", {
            'wyckoff': 0.30, 'liquidity': 0.25, 'structure': 0.15,
            'momentum': 0.15, 'volume': 0.15, 'context': 0.05, 'mtf': 0.10
        })

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            logging.info(f"Weights sum to {total_weight:.3f}, normalizing...")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}

        logging.info(f"Fusion engine v1.4.1 initialized with weights: {self.weights}")

    def regime_filter(self, df: pd.DataFrame, wyckoff_score: float, wyckoff_context: Dict) -> bool:
        """Simple regime filter - allows most trades through."""
        # Simple volume-based regime filter
        if len(df) < 20:
            return True

        vol_recent = df['volume'].iloc[-10:].mean()
        vol_longer = df['volume'].iloc[-50:].mean() if len(df) >= 50 else vol_recent
        vol_ratio = vol_recent / (vol_longer + 1e-9)

        # Allow if volume is reasonable and wyckoff score is decent
        regime_ok = vol_ratio >= 0.5 and wyckoff_score > 0.2

        if not regime_ok:
            phase = wyckoff_context.get("phase", "unknown")
            logging.info(f"Regime veto: low_vol_{phase}, vol_ratio={vol_ratio:.2f}, wyckoff={wyckoff_score:.2f}")

        return regime_ok


class FusionEngineV150(FusionEngineV141):
    """
    Enhanced fusion engine for v1.5.0 that extends v1.4.2 capabilities.

    Adds:
    - MTF DL2 filter integration
    - 6-candle leg rule validation
    - Orderflow LCA enhancement
    - Negative VIP reversal awareness
    - Asset profile-aware processing
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced fusion engine."""
        super().__init__(config)
        self.v150_features = config.get("features", {})
        self.quality_floors = config.get("quality_floors", {})
        self.asset_profile = config.get("profile_name", "base")

        logging.info(f"Enhanced Fusion Engine v1.5.0 initialized for profile: {self.asset_profile}")

    def apply_v150_alphas(self, df: pd.DataFrame, layer_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply v1.5.0 alpha enhancements to existing layer scores.

        This augments rather than replaces the v1.4.2 layer analysis.
        """
        enhanced_scores = layer_scores.copy()
        alpha_contributions = {}

        try:
            # MTF DL2 Filter + 6-Candle Leg Rule
            if is_feature_enabled(self.config, "mtf_dl2") or is_feature_enabled(self.config, "six_candle_leg"):
                mtf_enhancement = self._apply_mtf_enhancements(df)
                enhanced_scores["mtf"] += mtf_enhancement["adjustment"]
                alpha_contributions["mtf_enhancements"] = mtf_enhancement

            # Orderflow LCA Enhancement
            if is_feature_enabled(self.config, "orderflow_lca"):
                orderflow_enhancement = self._apply_orderflow_lca(df)
                enhanced_scores["structure"] += orderflow_enhancement["adjustment"]
                alpha_contributions["orderflow_lca"] = orderflow_enhancement

            # Negative VIP Reversal Awareness
            if is_feature_enabled(self.config, "negative_vip"):
                vip_enhancement = self._apply_negative_vip(df)
                enhanced_scores["volume"] += vip_enhancement["adjustment"]
                alpha_contributions["negative_vip"] = vip_enhancement

            # Log alpha contributions
            log_telemetry("alpha_contributions.json", {
                "profile": self.asset_profile,
                "alphas_applied": list(alpha_contributions.keys()),
                "original_scores": layer_scores,
                "enhanced_scores": enhanced_scores,
                "alpha_details": alpha_contributions
            })

        except Exception as e:
            logging.error(f"Error applying v1.5.0 alphas: {e}")
            # Return original scores if enhancement fails
            return layer_scores

        return enhanced_scores

    def _apply_mtf_enhancements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply MTF DL2 filter and 6-candle leg rule."""
        from bull_machine.modules.mtf.mtf_sync import six_candle_structure, mtf_dl2_filter

        enhancement = {
            "adjustment": 0.0,
            "components": {}
        }

        # 6-Candle Leg Rule (OPTIMIZED: Reduced impact)
        if is_feature_enabled(self.config, "six_candle_leg"):
            leg_valid = six_candle_structure(df)
            leg_adjustment = 0.05 if leg_valid else -0.05  # Reduced from 0.1/-0.15
            enhancement["adjustment"] += leg_adjustment
            enhancement["components"]["six_candle_leg"] = {
                "valid": leg_valid,
                "adjustment": leg_adjustment
            }

        # MTF DL2 Filter (OPTIMIZED: Pass timeframe for adaptive thresholds)
        if is_feature_enabled(self.config, "mtf_dl2"):
            timeframe = self.config.get("timeframe", "")
            dl2_ok = mtf_dl2_filter(df, timeframe)  # Pass timeframe instead of threshold
            dl2_adjustment = 0.05 if dl2_ok else -0.05  # Reduced harsh penalty
            enhancement["adjustment"] += dl2_adjustment
            enhancement["components"]["mtf_dl2"] = {
                "ok": dl2_ok,
                "timeframe": timeframe,
                "adjustment": dl2_adjustment
            }

        return enhancement

    def _apply_orderflow_lca(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply Orderflow LCA (Liquidity Capture Analysis)."""
        from bull_machine.modules.orderflow.lca import orderflow_lca, analyze_market_structure

        # Get orderflow score
        orderflow_score = orderflow_lca(df, self.config)

        # Convert to adjustment (0.4 = neutral baseline) - OPTIMIZED: Reduced impact
        baseline = 0.4
        raw_adjustment = (orderflow_score - baseline) * 0.15  # Reduced scale factor (was 0.3)

        # Apply asset-specific scaling (reduced)
        if self.asset_profile == "ETH":
            raw_adjustment *= 1.1  # Reduced from 1.2 to 1.1

        adjustment = max(-0.075, min(0.075, raw_adjustment)) * 0.5  # Reduced clamp and 50% impact

        # Get detailed market structure analysis
        market_structure = analyze_market_structure(df, self.config)

        return {
            "adjustment": adjustment,
            "orderflow_score": orderflow_score,
            "market_structure": market_structure,
            "asset_scaling": self.asset_profile
        }

    def _apply_negative_vip(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply Negative VIP (reversal awareness)."""
        from bull_machine.modules.sentiment.negative_vip import negative_vip_score, analyze_reversal_risk

        # Get VIP score
        vip_score = negative_vip_score(df, self.config)

        # High VIP score indicates reversal risk - reduce volume confidence (OPTIMIZED: 50% impact)
        baseline = 0.3
        reversal_risk = (vip_score - baseline) / 0.4  # Normalize to 0-1

        # Negative adjustment for high reversal risk (reduced impact)
        adjustment = -reversal_risk * 0.06  # Reduced from 0.12 to 0.06 (50% reduction)

        # Get detailed reversal analysis
        reversal_analysis = analyze_reversal_risk(df, self.config)

        return {
            "adjustment": adjustment,
            "vip_score": vip_score,
            "reversal_risk": reversal_risk,
            "reversal_analysis": reversal_analysis
        }

    def check_quality_floors(self, layer_scores: Dict[str, float]) -> bool:
        """
        Enhanced quality floor checking with v1.5.0 floors.

        Returns True if all quality floors are met, False otherwise.
        """
        if not self.quality_floors:
            return True

        violations = []

        for layer, min_score in self.quality_floors.items():
            if layer in layer_scores:
                actual_score = layer_scores[layer]
                if actual_score < min_score:
                    violations.append({
                        "layer": layer,
                        "required": min_score,
                        "actual": actual_score,
                        "deficit": min_score - actual_score
                    })

        quality_passed = len(violations) == 0

        # Log quality floor results
        log_telemetry("quality_floors.json", {
            "profile": self.asset_profile,
            "passed": quality_passed,
            "violations": violations,
            "layer_scores": layer_scores
        })

        if not quality_passed:
            logging.info(f"Quality floor violations: {violations}")

        return quality_passed

    def fuse(self, layer_scores: Dict[str, float], df: Optional[pd.DataFrame] = None,
             wyckoff_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced fusion that combines v1.4.2 baseline with v1.5.0 alphas.

        Process:
        1. Apply v1.5.0 alpha enhancements to layer scores
        2. Check quality floors (enhanced)
        3. Apply regime filter (preserved from v1.4.2)
        4. Calculate final weighted score
        5. Generate signal with confidence
        """

        # Step 1: Apply v1.5.0 alpha enhancements
        if df is not None and any(is_feature_enabled(self.config, f) for f in ["mtf_dl2", "six_candle_leg", "orderflow_lca", "negative_vip"]):
            enhanced_scores = self.apply_v150_alphas(df, layer_scores)
        else:
            enhanced_scores = layer_scores.copy()

        # Step 2: Check quality floors
        quality_passed = self.check_quality_floors(enhanced_scores)
        if not quality_passed:
            return {
                "score": 0.0,
                "signal": "neutral",
                "confidence": 0.0,
                "reason": "quality_floor_violation",
                "layer_scores": enhanced_scores,
                "quality_passed": False
            }

        # Step 3: Apply regime filter (preserve v1.4.2 logic)
        if df is not None and wyckoff_context is not None:
            regime_ok = self.regime_filter(df, enhanced_scores.get("wyckoff", 0), wyckoff_context)
            if not regime_ok:
                return {
                    "score": 0.0,
                    "signal": "neutral",
                    "confidence": 0.0,
                    "reason": "regime_filter_veto",
                    "layer_scores": enhanced_scores,
                    "regime_passed": False
                }

        # Step 4: Calculate weighted fusion score
        weighted_score = sum(enhanced_scores.get(layer, 0) * weight
                           for layer, weight in self.weights.items())

        # Step 5: Generate signal and confidence
        threshold = self.config.get("entry_threshold", 0.45)

        if weighted_score >= threshold:
            # Determine direction based on layer bias
            wyckoff_bias = wyckoff_context.get("bias", "neutral") if wyckoff_context else "neutral"
            signal = wyckoff_bias if wyckoff_bias != "neutral" else "long"
            confidence = min(1.0, weighted_score / threshold)
        else:
            signal = "neutral"
            confidence = weighted_score / threshold if threshold > 0 else 0

        result = {
            "score": weighted_score,
            "final_score": weighted_score,  # Alias for compatibility
            "signal": signal,
            "confidence": confidence,
            "threshold": threshold,
            "layer_scores": enhanced_scores,
            "original_scores": layer_scores,
            "quality_passed": True,
            "regime_passed": True,
            "profile": self.asset_profile,
            "alphas_enabled": [f for f in ["mtf_dl2", "six_candle_leg", "orderflow_lca", "negative_vip"]
                              if is_feature_enabled(self.config, f)]
        }

        # Log final fusion result
        log_telemetry("fusion_results.json", result)

        return result


def create_fusion_engine(config: Dict[str, Any]) -> FusionEngineV150:
    """Factory function to create enhanced fusion engine."""
    return FusionEngineV150(config)