"""
Bull Machine v1.5.0 - Enhanced Fusion Engine with Optimization Patches
Includes cooldown enforcement, alpha delta logging, and timeframe-aware features
"""

import pandas as pd
from typing import Dict, Any, Optional
from bull_machine.core.telemetry import log_telemetry
from bull_machine.core.config_loader import is_feature_enabled
from bull_machine.modules.fusion.enhanced import EnhancedFusionEngineV1_4
from bull_machine.modules.mtf.mtf_sync import six_candle_structure, mtf_dl2_filter
from bull_machine.modules.orderflow.lca import orderflow_lca
from bull_machine.modules.sentiment.negative_vip import negative_vip_score
from bull_machine.modules.regime_filter import regime_filter


class FusionEngineV150(EnhancedFusionEngineV1_4):
    """
    v1.5.0 Enhanced Fusion Engine with:
    - Cooldown bar enforcement
    - Alpha delta tracking
    - Timeframe-aware MTF DL2 and regime filters
    - Scaled v1.5.0 alpha impacts
    """

    def check_entry(self, df: pd.DataFrame, last_trade_bar: int, config: dict) -> bool:
        """
        Check if entry is allowed with cooldown enforcement.

        Args:
            df: Price data DataFrame
            last_trade_bar: Index of last trade entry
            config: Configuration with cooldown_bars

        Returns:
            bool: True if entry allowed, False if vetoed
        """
        # Cooldown enforcement
        cooldown_bars = int(config.get("cooldown_bars", 0) or 0)
        if cooldown_bars > 0:
            current_bar = len(df) - 1
            bars_since_trade = current_bar - last_trade_bar

            if bars_since_trade < cooldown_bars:
                log_telemetry("layer_masks.json", {
                    "cooldown_veto": True,
                    "cooldown_bars": cooldown_bars,
                    "bars_since_trade": bars_since_trade,
                    "current_bar": current_bar,
                    "last_trade_bar": last_trade_bar
                })
                return False

        # Compute base scores and check confluence
        layer_scores = self.compute_base_scores(df)
        veto = self.check_confluence_vetoes(df, layer_scores, config)

        return not veto

    def compute_base_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute base layer scores (placeholder - should integrate with existing layers).
        """
        # This would normally call the actual layer modules
        # For now, return placeholder scores
        return {
            'wyckoff': 0.45,
            'liquidity': 0.40,
            'structure': 0.42,
            'momentum': 0.48,
            'volume': 0.38,
            'context': 0.45,
            'mtf': 0.50
        }

    def check_confluence_vetoes(self, df: pd.DataFrame, layer_scores: Dict[str, float], config: dict) -> bool:
        """
        Check confluence vetoes with v1.5.0 alpha enhancements.

        Args:
            df: Price data DataFrame
            layer_scores: Base layer scores (will be modified in-place)
            config: Configuration dict

        Returns:
            bool: True if vetoed, False if passed
        """
        timeframe = config.get("timeframe", "")
        alpha_deltas = {}

        # Apply v1.5.0 alphas with scaled impacts
        if is_feature_enabled(config, "six_candle_leg"):
            # Six candle structure: ±0.05 impact
            structure_valid = six_candle_structure(df)
            delta = 0.05 if structure_valid else -0.05
            layer_scores["mtf"] = layer_scores.get("mtf", 0.5) + delta
            alpha_deltas["six_candle"] = delta

        if is_feature_enabled(config, "mtf_dl2"):
            # MTF DL2 filter with timeframe-aware threshold: ±0.05 impact
            dl2_ok = mtf_dl2_filter(df, timeframe)
            delta = 0.05 if dl2_ok else -0.05
            layer_scores["mtf"] = layer_scores.get("mtf", 0.5) + delta
            alpha_deltas["mtf_dl2"] = delta

        if is_feature_enabled(config, "orderflow_lca"):
            # Orderflow LCA: 50% scaled impact
            lca_score = orderflow_lca(df)
            delta = lca_score * 0.5  # Scale down impact
            layer_scores["structure"] = layer_scores.get("structure", 0.5) + delta
            alpha_deltas["orderflow_lca"] = float(delta)

        if is_feature_enabled(config, "negative_vip"):
            # Negative VIP: 50% scaled impact
            vip_score = negative_vip_score(df)
            delta = vip_score * 0.5  # Scale down impact
            layer_scores["volume"] = layer_scores.get("volume", 0.5) + delta
            alpha_deltas["negative_vip"] = float(delta)

        # Check regime filter (if applicable)
        if hasattr(self, 'check_regime') or config.get("use_regime_filter"):
            regime_ok = regime_filter(df, timeframe)
            if not regime_ok:
                alpha_deltas["regime_filter"] = -0.1
                layer_scores["volume"] = layer_scores.get("volume", 0.5) - 0.1

        # Check quality floors
        quality_floors = config.get("quality_floors", {})
        veto = False
        failed_layers = []

        for layer_name, floor_value in quality_floors.items():
            layer_score = layer_scores.get(layer_name, 0)
            if layer_score < floor_value:
                veto = True
                failed_layers.append(f"{layer_name}({layer_score:.3f}<{floor_value})")

        # Log telemetry
        log_telemetry("layer_masks.json", {
            "timeframe": timeframe,
            "veto": veto,
            "failed_layers": failed_layers,
            "layer_scores": layer_scores,
            "alpha_deltas": alpha_deltas,
            "profile_name": config.get("profile_name", "base"),
            "entry_threshold": config.get("entry_threshold", 0.45)
        })

        return veto

    def check_quality_floors(self, layer_scores: Dict[str, float]) -> bool:
        """
        Check if layer scores meet quality floors.

        Args:
            layer_scores: Dictionary of layer scores

        Returns:
            bool: True if all floors passed, False if any failed
        """
        quality_floors = self.config.get("quality_floors", {})

        for layer_name, floor_value in quality_floors.items():
            if layer_scores.get(layer_name, 0) < floor_value:
                return False

        return True

    def apply_v150_alphas(self, df: pd.DataFrame, layer_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply v1.5.0 alpha enhancements to layer scores.

        Args:
            df: Price data DataFrame
            layer_scores: Base layer scores

        Returns:
            Enhanced layer scores
        """
        enhanced_scores = layer_scores.copy()
        config = self.config

        # Track alpha contributions
        alpha_deltas = {}

        # Six candle structure
        if is_feature_enabled(config, "six_candle_leg"):
            if six_candle_structure(df):
                enhanced_scores["mtf"] += 0.03  # Reduced impact
                alpha_deltas["six_candle"] = 0.03
            else:
                enhanced_scores["mtf"] -= 0.03
                alpha_deltas["six_candle"] = -0.03

        # MTF DL2 filter
        if is_feature_enabled(config, "mtf_dl2"):
            timeframe = config.get("timeframe", "")
            if mtf_dl2_filter(df, timeframe):
                enhanced_scores["mtf"] += 0.02
                alpha_deltas["mtf_dl2"] = 0.02
            else:
                enhanced_scores["mtf"] -= 0.05
                alpha_deltas["mtf_dl2"] = -0.05

        # Orderflow LCA (if not disabled for timeframe)
        if is_feature_enabled(config, "orderflow_lca"):
            lca_impact = orderflow_lca(df) * 0.4  # Reduced from 0.8
            enhanced_scores["structure"] += lca_impact
            alpha_deltas["orderflow_lca"] = lca_impact

        # Negative VIP (if not disabled for timeframe)
        if is_feature_enabled(config, "negative_vip"):
            vip_impact = negative_vip_score(df) * 0.4  # Reduced from 0.8
            enhanced_scores["volume"] += vip_impact
            alpha_deltas["negative_vip"] = vip_impact

        # Log alpha contributions
        if alpha_deltas:
            log_telemetry("alpha_contributions.json", {
                "timeframe": config.get("timeframe", ""),
                "profile": config.get("profile_name", "base"),
                "alpha_deltas": alpha_deltas,
                "original_scores": layer_scores,
                "enhanced_scores": enhanced_scores
            })

        return enhanced_scores