#!/usr/bin/env python3
"""
Feature Registry - Single source of truth for canonical names, aliases, and types.

This registry eliminates confusion from scattered naming conventions across different
builders and data sources.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    canonical: str
    dtype: str
    tier: int  # 1=raw/technical, 2=MTF, 3=regime+macro
    required: bool
    aliases: List[str]
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    description: str = ""


class FeatureRegistry:
    """
    Central registry for all features across tiers.

    Responsibilities:
    - Define canonical names for all features
    - Map aliases to canonical names (e.g., tf1h_bos_flag → tf1h_bos_bullish)
    - Specify data types and valid ranges
    - Support validation and normalization
    """

    def __init__(self):
        self._features: Dict[str, FeatureSpec] = {}
        self._alias_map: Dict[str, str] = {}
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize all feature specifications."""

        # Tier 1: OHLCV + Technical Indicators
        self._register_tier1()

        # Tier 2: Multi-Timeframe Features
        self._register_tier2()

        # Tier 3: Regime + Macro
        self._register_tier3()

        # Build alias map
        self._build_alias_map()

    def _register_tier1(self):
        """Register Tier 1 features: OHLCV + technical indicators."""
        tier1_features = [
            FeatureSpec("open", "float64", 1, True, [], 0.0, None, "Opening price"),
            FeatureSpec("high", "float64", 1, True, [], 0.0, None, "Highest price"),
            FeatureSpec("low", "float64", 1, True, [], 0.0, None, "Lowest price"),
            FeatureSpec("close", "float64", 1, True, [], 0.0, None, "Closing price"),
            FeatureSpec("volume", "float64", 1, True, [], 0.0, None, "Trading volume"),

            # Technical indicators
            FeatureSpec("atr_20", "float64", 1, False, ["atr"], 0.0, None, "Average True Range (20)"),
            FeatureSpec("adx_14", "float64", 1, False, ["adx"], 0.0, 100.0, "Average Directional Index (14)"),
            FeatureSpec("rsi_14", "float64", 1, False, ["rsi"], 0.0, 100.0, "Relative Strength Index (14)"),
            FeatureSpec("ema_20", "float64", 1, False, [], 0.0, None, "EMA 20"),
            FeatureSpec("ema_50", "float64", 1, False, [], 0.0, None, "EMA 50"),
            FeatureSpec("ema_200", "float64", 1, False, [], 0.0, None, "EMA 200"),
            FeatureSpec("macd", "float64", 1, False, [], None, None, "MACD line"),
            FeatureSpec("macd_signal", "float64", 1, False, [], None, None, "MACD signal line"),
            FeatureSpec("macd_hist", "float64", 1, False, [], None, None, "MACD histogram"),
            FeatureSpec("bb_upper", "float64", 1, False, [], 0.0, None, "Bollinger Band upper"),
            FeatureSpec("bb_lower", "float64", 1, False, [], 0.0, None, "Bollinger Band lower"),
            FeatureSpec("bb_width", "float64", 1, False, [], 0.0, None, "Bollinger Band width"),
        ]

        for spec in tier1_features:
            self._features[spec.canonical] = spec

    def _register_tier2(self):
        """Register Tier 2 features: Multi-timeframe."""
        tier2_features = [
            # 4H timeframe
            FeatureSpec("tf4h_fusion_score", "float64", 2, False,
                       ["tf4h_fusion"], -1.0, 1.0,
                       "4H timeframe fusion score (trend + structure quality)"),
            FeatureSpec("tf4h_trend_strength", "float64", 2, False,
                       [], 0.0, 1.0,
                       "4H trend strength indicator"),
            FeatureSpec("tf4h_bos_bullish", "bool", 2, False,
                       ["tf4h_bos"], None, None,
                       "4H bullish break of structure detected"),
            FeatureSpec("tf4h_bos_bearish", "bool", 2, False,
                       [], None, None,
                       "4H bearish break of structure detected"),

            # Daily timeframe
            FeatureSpec("tf1d_trend_direction", "int8", 2, False,
                       ["tf1d_trend"], -1.0, 1.0,
                       "Daily trend direction: 1 (up), -1 (down), 0 (neutral)"),
            FeatureSpec("tf1d_volatility", "float64", 2, False,
                       [], 0.0, None,
                       "Daily volatility metric"),

            # 1H structure features
            FeatureSpec("tf1h_bos_bullish", "bool", 2, False,
                       ["tf1h_bos_flag", "tf1h_bos"], None, None,
                       "1H bullish break of structure detected"),
            FeatureSpec("tf1h_bos_bearish", "bool", 2, False,
                       [], None, None,
                       "1H bearish break of structure detected"),
            FeatureSpec("tf1h_fvg_bull", "bool", 2, False,
                       ["tf1h_fvg"], None, None,
                       "1H bullish fair value gap present"),
            FeatureSpec("tf1h_fvg_bear", "bool", 2, False,
                       [], None, None,
                       "1H bearish fair value gap present"),
            FeatureSpec("tf1h_ob_bull_top", "float64", 2, False,
                       [], None, None,
                       "Bullish order block upper boundary"),
            FeatureSpec("tf1h_ob_bull_bottom", "float64", 2, False,
                       [], None, None,
                       "Bullish order block lower boundary"),
            FeatureSpec("tf1h_ob_bear_top", "float64", 2, False,
                       [], None, None,
                       "Bearish order block upper boundary"),
            FeatureSpec("tf1h_ob_bear_bottom", "float64", 2, False,
                       [], None, None,
                       "Bearish order block lower boundary"),

            # Composite scores
            FeatureSpec("liquidity_score", "float64", 2, False,
                       ["liq_score"], 0.0, 1.0,
                       "Liquidity availability score"),
            FeatureSpec("smc_score", "float64", 2, False,
                       [], 0.0, 1.0,
                       "Smart money concepts composite score"),
            FeatureSpec("wyckoff_score", "float64", 2, False,
                       [], 0.0, 1.0,
                       "Wyckoff method composite score"),
            FeatureSpec("momentum_score", "float64", 2, False,
                       [], 0.0, 1.0,
                       "Momentum composite score"),

            # MTF alignment
            FeatureSpec("mtf_alignment_score", "float64", 2, False,
                       ["alignment_score"], -1.0, 1.0,
                       "Multi-timeframe alignment score"),
        ]

        for spec in tier2_features:
            self._features[spec.canonical] = spec

    def _register_tier3(self):
        """Register Tier 3 features: Regime + Macro."""
        tier3_features = [
            # Regime classification
            FeatureSpec("regime_label", "category", 3, True,
                       ["regime"], None, None,
                       "Market regime classification"),
            FeatureSpec("regime_confidence", "float64", 3, False,
                       [], 0.0, 1.0,
                       "Regime classification confidence"),
            FeatureSpec("regime_prob_risk_on", "float64", 3, False,
                       [], 0.0, 1.0,
                       "Probability of risk_on regime"),
            FeatureSpec("regime_prob_risk_off", "float64", 3, False,
                       [], 0.0, 1.0,
                       "Probability of risk_off regime"),
            FeatureSpec("regime_prob_neutral", "float64", 3, False,
                       [], 0.0, 1.0,
                       "Probability of neutral regime"),
            FeatureSpec("regime_prob_crisis", "float64", 3, False,
                       [], 0.0, 1.0,
                       "Probability of crisis regime"),

            # Macro context (optional)
            FeatureSpec("dxy_close", "float64", 3, False,
                       ["dxy"], None, None,
                       "Dollar Index close price"),
            FeatureSpec("vix_close", "float64", 3, False,
                       ["vix"], 0.0, None,
                       "VIX close price"),
            FeatureSpec("yields_10y", "float64", 3, False,
                       ["us10y"], 0.0, None,
                       "US 10Y Treasury yield"),
        ]

        for spec in tier3_features:
            self._features[spec.canonical] = spec

    def _build_alias_map(self):
        """Build alias → canonical name mapping."""
        for canonical, spec in self._features.items():
            # Map canonical to itself
            self._alias_map[canonical] = canonical

            # Map each alias to canonical
            for alias in spec.aliases:
                if alias in self._alias_map and self._alias_map[alias] != canonical:
                    raise ValueError(
                        f"Alias conflict: '{alias}' maps to both "
                        f"'{self._alias_map[alias]}' and '{canonical}'"
                    )
                self._alias_map[alias] = canonical

    def normalize_column_name(self, name: str) -> str:
        """
        Convert any column name (canonical or alias) to canonical form.

        Args:
            name: Column name (canonical or alias)

        Returns:
            Canonical name

        Raises:
            KeyError: If name is not recognized
        """
        if name in self._alias_map:
            return self._alias_map[name]

        # Not found - return as-is and let validation catch it
        return name

    def get_feature_spec(self, name: str) -> Optional[FeatureSpec]:
        """
        Get feature specification by canonical name or alias.

        Args:
            name: Column name (canonical or alias)

        Returns:
            FeatureSpec if found, None otherwise
        """
        canonical = self.normalize_column_name(name)
        return self._features.get(canonical)

    def get_required_features(self, tier: int) -> List[str]:
        """
        Get list of required features for a given tier (cumulative).

        Args:
            tier: Tier level (1, 2, or 3)

        Returns:
            List of canonical feature names that are required
        """
        return [
            name for name, spec in self._features.items()
            if spec.required and spec.tier <= tier
        ]

    def get_tier_features(self, tier: int) -> List[str]:
        """
        Get all features for a specific tier.

        Args:
            tier: Tier level (1, 2, or 3)

        Returns:
            List of canonical feature names in that tier
        """
        return [
            name for name, spec in self._features.items()
            if spec.tier == tier
        ]

    def get_all_features(self, max_tier: int = 3) -> List[str]:
        """
        Get all features up to and including max_tier.

        Args:
            max_tier: Maximum tier to include (1, 2, or 3)

        Returns:
            List of canonical feature names
        """
        return [
            name for name, spec in self._features.items()
            if spec.tier <= max_tier
        ]


# Global registry instance
_REGISTRY = None


def get_registry() -> FeatureRegistry:
    """Get the global feature registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = FeatureRegistry()
    return _REGISTRY
