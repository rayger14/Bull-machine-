#!/usr/bin/env python3
"""
Meta-Fusion Engine: Cross-Domain Scoring System

Provides reusable weighted fusion across domain engines:
- structure_score (Wyckoff + SMC)
- liquidity_score
- momentum_score
- wyckoff_event_score
- macro_score
- pti_score (when available)

Supports both static and optimized engine weights.

Version: 1.0
Author: Backend Architect
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class MetaFusionEngine:
    """
    Engine-level fusion scorer with configurable domain weights.

    Architecture:
    - Default weights: Equal weighting across all domains
    - Optimized weights: Loaded from config (ML or Optuna-derived)
    - Regime-aware: Optional regime-specific weight adjustments
    - Normalization: Ensures weights sum to 1.0
    """

    VERSION = "meta_fusion@v1.0"

    # Default weights (equal weighting baseline)
    DEFAULT_WEIGHTS = {
        'structure': 0.20,      # Wyckoff + SMC patterns
        'liquidity': 0.20,      # Liquidity score
        'momentum': 0.20,       # RSI/MACD/trend
        'wyckoff': 0.20,        # Wyckoff events
        'macro': 0.20,          # VIX_Z, DXY_Z, funding_Z
        'pti': 0.00             # PTI (optional, disabled by default)
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize meta-fusion engine.

        Args:
            config: Optional config dict with 'engine_weights' section
                   Example:
                   {
                       'engine_weights': {
                           'structure': 0.35,
                           'liquidity': 0.25,
                           'momentum': 0.15,
                           'wyckoff': 0.15,
                           'macro': 0.10
                       },
                       'regime_aware': True  # Optional regime-specific weights
                   }
        """
        self.config = config or {}

        # Load weights from config or use defaults
        raw_weights = self.config.get('engine_weights', self.DEFAULT_WEIGHTS.copy())

        # Normalize to ensure sum = 1.0
        self.weights = self._normalize_weights(raw_weights)

        # Regime-aware mode (optional)
        self.regime_aware = self.config.get('regime_aware', False)
        self.regime_weights = self.config.get('regime_engine_weights', {})

        logger.info(f"[MetaFusion] Initialized with weights: {self.weights}")
        if self.regime_aware:
            logger.info(f"[MetaFusion] Regime-aware mode enabled, regimes: {list(self.regime_weights.keys())}")

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Raw weights dict

        Returns:
            Normalized weights dict
        """
        total = sum(weights.values())
        if total == 0:
            logger.warning("[MetaFusion] Zero total weight, using defaults")
            return self.DEFAULT_WEIGHTS.copy()

        normalized = {k: v / total for k, v in weights.items()}
        return normalized

    def compute_fusion(
        self,
        domain_scores: Dict[str, float],
        regime_label: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Compute weighted meta-fusion score across domain engines.

        Args:
            domain_scores: Dict of domain -> score mappings
                          Example: {'structure': 0.75, 'liquidity': 0.60, ...}
            regime_label: Optional regime for regime-aware weighting

        Returns:
            (fusion_score: float, metadata: dict)
        """
        # Select appropriate weights (regime-aware or global)
        weights = self._get_weights_for_regime(regime_label)

        # Compute weighted sum
        fusion = 0.0
        used_domains = []

        for domain, weight in weights.items():
            score = domain_scores.get(domain, 0.0)

            # Skip domains with zero weight or missing scores
            if weight > 0 and score is not None:
                fusion += weight * score
                used_domains.append(domain)

        # Build metadata for transparency
        metadata = {
            'fusion_score': fusion,
            'weights_used': weights,
            'domains_used': used_domains,
            'regime': regime_label,
            'version': self.VERSION
        }

        return max(0.0, min(1.0, fusion)), metadata

    def _get_weights_for_regime(self, regime_label: Optional[str]) -> Dict[str, float]:
        """
        Get weights appropriate for the current regime.

        Args:
            regime_label: Regime label (risk_on, risk_off, neutral, crisis)

        Returns:
            Weights dict for the regime
        """
        if not self.regime_aware or regime_label is None:
            return self.weights

        # Check if regime-specific weights exist
        regime_weights = self.regime_weights.get(regime_label)
        if regime_weights:
            return self._normalize_weights(regime_weights)

        # Fallback to global weights
        return self.weights

    def extract_domain_scores(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract domain scores from a feature row.

        Handles multiple naming conventions and missing features gracefully.

        Args:
            row: DataFrame row with features

        Returns:
            Dict of domain -> score mappings
        """
        scores = {}

        # Structure score (order blocks, liquidity sweeps, springs)
        # Prefer tf1h_structure_score, fallback to boms_strength
        scores['structure'] = self._safe_get(row, [
            'structure_score',
            'tf1h_structure_score',
            'boms_strength'
        ], default=0.0)

        # Liquidity score
        scores['liquidity'] = self._safe_get(row, [
            'liquidity_score',
            'tf1h_liquidity_score'
        ], default=0.0)

        # Momentum score (RSI, MACD, trend strength)
        # Derive if not present
        scores['momentum'] = self._get_or_derive_momentum(row)

        # Wyckoff score
        scores['wyckoff'] = self._safe_get(row, [
            'wyckoff_score',
            'tf1d_wyckoff_score',
            'tf1h_wyckoff_score'
        ], default=0.0)

        # Macro score (VIX_Z, DXY_Z, funding_Z normalization)
        scores['macro'] = self._get_or_derive_macro(row)

        # PTI score (optional)
        scores['pti'] = self._safe_get(row, [
            'pti_score',
            'tf1h_pti_score'
        ], default=0.0)

        return scores

    def _safe_get(self, row: pd.Series, keys: list, default=0.0) -> float:
        """Safe getter with multiple key fallback."""
        for key in keys:
            if key in row.index and row[key] is not None and not pd.isna(row[key]):
                return float(row[key])
        return default

    def _get_or_derive_momentum(self, row: pd.Series) -> float:
        """
        Get or derive momentum score from RSI/MACD/ADX.

        Momentum scoring:
        - RSI distance from 50 (0.4 weight)
        - ADX strength (0.3 weight)
        - Volume z-score (0.3 weight)
        """
        # Prefer existing momentum_score
        if 'momentum_score' in row.index and not pd.isna(row['momentum_score']):
            return float(row['momentum_score'])

        # Derive from components
        rsi = self._safe_get(row, ['rsi_14', 'tf1h_rsi_14'], 50.0)
        adx = self._safe_get(row, ['adx_14', 'tf1h_adx_14'], 20.0)
        vol_z = self._safe_get(row, ['volume_zscore', 'tf1h_volume_zscore'], 0.0)

        # RSI component: 0 at neutral (50), 1 at extremes (25 or 75)
        rsi_comp = min(abs(rsi - 50.0) / 25.0, 1.0)

        # ADX component: 10→40 maps to 0→1
        adx_comp = max(0.0, min((adx - 10.0) / 30.0, 1.0))

        # Volume component: normalize z-score to [0, 1]
        vol_comp = max(0.0, min(vol_z / 2.0, 1.0))

        momentum = 0.4 * rsi_comp + 0.3 * adx_comp + 0.3 * vol_comp
        return max(0.0, min(1.0, momentum))

    def _get_or_derive_macro(self, row: pd.Series) -> float:
        """
        Get or derive macro score from VIX_Z, DXY_Z, funding_Z.

        Macro scoring:
        - VIX_Z (risk indicator, 0.35 weight)
        - DXY_Z (dollar strength, 0.35 weight)
        - funding_Z (crypto sentiment, 0.30 weight)
        """
        # Prefer existing macro_score
        if 'macro_score' in row.index and not pd.isna(row['macro_score']):
            return float(row['macro_score'])

        # Derive from macro features
        vix_z = self._safe_get(row, ['VIX_Z'], 0.0)
        dxy_z = self._safe_get(row, ['DXY_Z'], 0.0)
        funding_z = self._safe_get(row, ['funding_Z'], 0.0)

        # Normalize z-scores to [0, 1] range
        # High VIX = risk-off (negative for longs)
        # High DXY = dollar strength (negative for BTC)
        # High funding = longs overcrowded (negative for longs)

        # Invert and normalize (lower risk/stronger conditions = higher score)
        vix_comp = max(0.0, min(1.0 - (vix_z + 2.0) / 4.0, 1.0))  # -2 to +2 sigma
        dxy_comp = max(0.0, min(1.0 - (dxy_z + 2.0) / 4.0, 1.0))
        funding_comp = max(0.0, min(1.0 - (funding_z + 2.0) / 4.0, 1.0))

        macro = 0.35 * vix_comp + 0.35 * dxy_comp + 0.30 * funding_comp
        return max(0.0, min(1.0, macro))

    def apply_meta_fusion(self, row: pd.Series, regime_label: Optional[str] = None) -> Tuple[float, Dict]:
        """
        Convenience method: extract domain scores and compute meta-fusion.

        Args:
            row: DataFrame row with features
            regime_label: Optional regime for regime-aware weighting

        Returns:
            (fusion_score: float, metadata: dict)
        """
        domain_scores = self.extract_domain_scores(row)
        return self.compute_fusion(domain_scores, regime_label)


# ============================================================================
# Utility Functions
# ============================================================================

def load_optimized_weights(config_path: str) -> Dict[str, float]:
    """
    Load optimized engine weights from JSON config.

    Args:
        config_path: Path to config file with 'engine_weights' section

    Returns:
        Weights dict
    """
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    weights = config.get('engine_weights', MetaFusionEngine.DEFAULT_WEIGHTS.copy())
    logger.info(f"[MetaFusion] Loaded weights from {config_path}: {weights}")
    return weights


def create_meta_fusion_from_config(config: Dict) -> MetaFusionEngine:
    """
    Factory function to create MetaFusionEngine from config dict.

    Args:
        config: Config dict with optional 'engine_weights' and 'regime_aware'

    Returns:
        MetaFusionEngine instance
    """
    return MetaFusionEngine(config)
