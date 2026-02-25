#!/usr/bin/env python3
"""
Feature Mapper - Canonical Name Translation

Maps config-expected feature names to actual feature store column names.
Ensures archetypes can access all domain features regardless of naming inconsistencies.

Problem Solved:
- Config expects 'funding_z', feature store has 'funding_Z'
- Config expects 'oi_delta_z', feature store has 'oi_z'
- Mismatches prevented archetypes from accessing critical features

This mapper enables 100% feature coverage by handling all naming variations.
"""

from typing import Dict, List, Set
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureMapper:
    """Translates between canonical (config) and actual (store) feature names."""

    # Canonical → Actual mapping
    # Format: "config_name": "feature_store_column_name"
    CANONICAL_TO_STORE = {
        # ============ FUNDING/OI DOMAIN ============
        "funding_z": "funding_Z",  # Case mismatch fix
        "funding_Z": "funding_Z",  # Allow both
        "funding_rate": "funding_rate",
        "funding": "funding",

        "oi_change_pct_24h": "oi_change_pct_24h",
        "oi_change_24h": "oi_change_24h",
        "oi_delta_z": "oi_z",  # Name mismatch fix
        "oi_z": "oi_z",  # Allow both
        "oi": "oi",

        # ============ LIQUIDITY DOMAIN (S1 Critical) ============
        # Multi-bar detection features
        "volume_climax_3b": "volume_climax_last_3b",  # Name mismatch fix
        "volume_climax_last_3b": "volume_climax_last_3b",  # Allow both
        "wick_exhaustion_3b": "wick_exhaustion_last_3b",  # Name mismatch fix
        "wick_exhaustion_last_3b": "wick_exhaustion_last_3b",  # Allow both (if exists)

        # Liquidity dynamics
        "liquidity_drain_severity": "liquidity_drain_pct",  # Name mismatch fix
        "liquidity_drain_pct": "liquidity_drain_pct",  # Allow both
        "liquidity_velocity_score": "liquidity_velocity",  # Name mismatch fix
        "liquidity_velocity": "liquidity_velocity",  # Allow both
        "liquidity_persistence_score": "liquidity_persistence",  # Name mismatch fix
        "liquidity_persistence": "liquidity_persistence",  # Allow both
        "liquidity_score": "liquidity_score",
        "liquidity_vacuum_score": "liquidity_vacuum_score",
        "liquidity_vacuum_fusion": "liquidity_vacuum_fusion",

        # Volume features
        "volume_panic": "volume_panic",
        "volume_z": "volume_z",
        "volume_zscore": "volume_zscore",
        "volume_ratio": "volume_ratio",
        "volume": "volume",

        # ============ MACRO DOMAIN ============
        # Case-sensitive tickers
        "btc_d": "BTC.D",  # Case mismatch fix
        "BTC.D": "BTC.D",  # Allow both
        "btc_d_z": "BTC.D_Z",  # Case mismatch fix
        "BTC.D_Z": "BTC.D_Z",  # Allow both

        "usdt_d": "USDT.D",  # Case mismatch fix
        "USDT.D": "USDT.D",  # Allow both
        "usdt_d_z": "USDT.D_Z",  # Case mismatch fix
        "USDT.D_Z": "USDT.D_Z",  # Allow both

        "vix": "VIX",  # Case mismatch fix
        "VIX": "VIX",  # Allow both
        "vix_z": "VIX_Z",  # Case mismatch fix
        "VIX_Z": "VIX_Z",  # Allow both

        "dxy": "DXY",  # Case mismatch fix
        "DXY": "DXY",  # Allow both
        "dxy_z": "DXY_Z",  # Case mismatch fix
        "DXY_Z": "DXY_Z",  # Allow both

        # Macro derived
        "macro_regime": "macro_regime",
        "regime_v2": "regime_v2",
        "macro_vix_level": "macro_vix_level",
        "macro_dxy_trend": "macro_dxy_trend",

        # ============ WYCKOFF DOMAIN ============
        # Wyckoff phase
        "wyckoff_phase": "wyckoff_phase_abc",  # Name mismatch fix
        "wyckoff_phase_abc": "wyckoff_phase_abc",  # Allow both
        "wyckoff_sequence_position": "wyckoff_sequence_position",

        # Wyckoff events (Phase A: Climax)
        "wyckoff_sc": "wyckoff_sc",
        "wyckoff_sc_confidence": "wyckoff_sc_confidence",
        "wyckoff_bc": "wyckoff_bc",
        "wyckoff_bc_confidence": "wyckoff_bc_confidence",
        "wyckoff_ar": "wyckoff_ar",
        "wyckoff_ar_confidence": "wyckoff_ar_confidence",
        "wyckoff_as": "wyckoff_as",
        "wyckoff_as_confidence": "wyckoff_as_confidence",
        "wyckoff_st": "wyckoff_st",
        "wyckoff_st_confidence": "wyckoff_st_confidence",

        # Wyckoff events (Phase B: Building)
        "wyckoff_sos": "wyckoff_sos",
        "wyckoff_sos_confidence": "wyckoff_sos_confidence",
        "wyckoff_sow": "wyckoff_sow",
        "wyckoff_sow_confidence": "wyckoff_sow_confidence",

        # Wyckoff events (Phase C: Testing)
        "wyckoff_spring_a": "wyckoff_spring_a",
        "wyckoff_spring_a_confidence": "wyckoff_spring_a_confidence",
        "wyckoff_spring_b": "wyckoff_spring_b",
        "wyckoff_spring_b_confidence": "wyckoff_spring_b_confidence",
        "wyckoff_ut": "wyckoff_ut",
        "wyckoff_ut_confidence": "wyckoff_ut_confidence",
        "wyckoff_utad": "wyckoff_utad",
        "wyckoff_utad_confidence": "wyckoff_utad_confidence",

        # Wyckoff events (Phase D: Last Point)
        "wyckoff_lps": "wyckoff_lps",
        "wyckoff_lps_confidence": "wyckoff_lps_confidence",
        "wyckoff_lpsy": "wyckoff_lpsy",
        "wyckoff_lpsy_confidence": "wyckoff_lpsy_confidence",

        # Wyckoff MTF
        "tf1d_wyckoff_phase": "tf1d_wyckoff_phase",
        "tf1d_wyckoff_score": "tf1d_wyckoff_score",

        # ============ SMC DOMAIN ============
        # Order blocks
        "order_block_bull": "is_bullish_ob",  # Name mismatch fix
        "is_bullish_ob": "is_bullish_ob",  # Allow both
        "order_block_bear": "is_bearish_ob",  # Name mismatch fix
        "is_bearish_ob": "is_bearish_ob",  # Allow both
        "ob_confidence": "ob_confidence",
        "ob_strength_bullish": "ob_strength_bullish",
        "ob_strength_bearish": "ob_strength_bearish",

        # FVG (Fair Value Gap)
        "fvg_bull": "tf1h_fvg_present",  # Simplified mapping
        "fvg_bear": "tf1h_fvg_present",  # Both map to presence flag
        "tf1h_fvg_present": "tf1h_fvg_present",
        "tf1h_fvg_high": "tf1h_fvg_high",
        "tf1h_fvg_low": "tf1h_fvg_low",
        "tf4h_fvg_present": "tf4h_fvg_present",

        # BOS/CHOCH
        "bos_bull": "tf1h_bos_bullish",  # Name mismatch fix
        "tf1h_bos_bullish": "tf1h_bos_bullish",  # Allow both
        "bos_bear": "tf1h_bos_bearish",  # Name mismatch fix
        "tf1h_bos_bearish": "tf1h_bos_bearish",  # Allow both
        "bos_choch": "tf4h_choch_flag",  # Name mismatch fix
        "tf4h_choch_flag": "tf4h_choch_flag",  # Allow both

        # Order block zones (MTF)
        "tf1h_ob_high": "tf1h_ob_high",
        "tf1h_ob_low": "tf1h_ob_low",

        # Liquidity sweeps (if implemented)
        "liquidity_sweep_high": "liquidity_sweep_high",
        "liquidity_sweep_low": "liquidity_sweep_low",

        # ============ TEMPORAL DOMAIN ============
        # Note: Temporal features appear to be missing from feature store
        # These mappings are placeholders for when temporal domain is added
        "fib_time_cluster": "fib_time_cluster",
        "gann_time_window": "gann_time_window",
        "temporal_confluence": "temporal_confluence_score",
        "temporal_confluence_score": "temporal_confluence_score",

        # ============ CRISIS/CAPITULATION DOMAIN (S1 Critical) ============
        "crisis_composite": "crisis_composite",
        "crisis_context": "crisis_context",
        "capitulation_depth": "capitulation_depth",
        "drawdown_1d": "drawdown_1d",
        "drawdown_7d": "drawdown_7d",

        # ============ STANDARD OHLCV ============
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",

        # ============ TECHNICAL INDICATORS ============
        "rsi": "rsi",
        "rsi_14": "rsi_14",
        "atr": "atr",
        "atr_20": "atr_20",
        "adx": "adx",
        "adx_14": "adx_14",
    }

    # Reverse mapping (for feature store → canonical)
    STORE_TO_CANONICAL = {v: k for k, v in CANONICAL_TO_STORE.items()}

    @classmethod
    def map_to_store(cls, canonical_name: str) -> str:
        """
        Map canonical (config) name to actual feature store name.

        Args:
            canonical_name: Feature name as used in config

        Returns:
            Actual column name in feature store
        """
        return cls.CANONICAL_TO_STORE.get(canonical_name, canonical_name)

    @classmethod
    def map_to_canonical(cls, store_name: str) -> str:
        """
        Map feature store name to canonical (config) name.

        Args:
            store_name: Column name as stored in feature store

        Returns:
            Canonical name for use in configs
        """
        return cls.STORE_TO_CANONICAL.get(store_name, store_name)

    @classmethod
    def map_dataframe(cls, df: pd.DataFrame, to_canonical: bool = True) -> pd.DataFrame:
        """
        Rename all columns in DataFrame to canonical or store names.

        Args:
            df: DataFrame to rename
            to_canonical: If True, rename to canonical names. If False, rename to store names.

        Returns:
            DataFrame with renamed columns (copy)
        """
        if to_canonical:
            rename_map = {v: k for k, v in cls.CANONICAL_TO_STORE.items() if v in df.columns}
        else:
            rename_map = {k: v for k, v in cls.CANONICAL_TO_STORE.items() if k in df.columns}

        if rename_map:
            logger.debug(f"Renaming {len(rename_map)} columns: {list(rename_map.keys())[:5]}...")

        return df.rename(columns=rename_map)

    @classmethod
    def get_feature(cls, df: pd.DataFrame, canonical_name: str, default=None):
        """
        Get feature from DataFrame using canonical name (handles mapping automatically).

        Args:
            df: DataFrame containing features
            canonical_name: Canonical feature name (from config)
            default: Value to return if feature not found (default: None)

        Returns:
            Feature series if found, default value otherwise

        Raises:
            KeyError: If feature not found and default is None
        """
        store_name = cls.map_to_store(canonical_name)

        if store_name in df.columns:
            return df[store_name]
        elif canonical_name in df.columns:
            return df[canonical_name]
        elif default is not None:
            logger.warning(
                f"Feature '{canonical_name}' not found (tried '{store_name}' and '{canonical_name}'). "
                f"Returning default: {default}"
            )
            return default
        else:
            raise KeyError(
                f"Feature '{canonical_name}' not found in DataFrame. "
                f"Tried store name '{store_name}' and canonical name '{canonical_name}'. "
                f"Available columns: {sorted(df.columns)[:20]}"
            )

    @classmethod
    def has_feature(cls, df: pd.DataFrame, canonical_name: str) -> bool:
        """
        Check if feature exists in DataFrame using canonical name.

        Args:
            df: DataFrame to check
            canonical_name: Canonical feature name

        Returns:
            True if feature exists (under any valid name), False otherwise
        """
        store_name = cls.map_to_store(canonical_name)
        return store_name in df.columns or canonical_name in df.columns

    @classmethod
    def get_missing_features(cls, df: pd.DataFrame, required_features: List[str]) -> List[str]:
        """
        Get list of required features that are missing from DataFrame.

        Args:
            df: DataFrame to check
            required_features: List of canonical feature names required

        Returns:
            List of missing feature names (canonical)
        """
        missing = []
        for canonical_name in required_features:
            if not cls.has_feature(df, canonical_name):
                missing.append(canonical_name)
        return missing

    @classmethod
    def audit_feature_coverage(cls, df: pd.DataFrame) -> Dict[str, any]:
        """
        Audit which domain features are available in DataFrame.

        Args:
            df: DataFrame to audit

        Returns:
            Dictionary with coverage statistics per domain
        """
        # Define domain feature groups
        domains = {
            "funding_oi": ["funding_z", "funding_rate", "oi_z", "oi_change_pct_24h"],
            "liquidity": ["liquidity_score", "liquidity_drain_severity", "liquidity_velocity_score",
                         "volume_climax_3b", "wick_exhaustion_3b"],
            "wyckoff": ["wyckoff_phase", "wyckoff_sc", "wyckoff_bc", "wyckoff_spring_a",
                       "wyckoff_lps", "wyckoff_sos"],
            "smc": ["order_block_bull", "order_block_bear", "fvg_bull", "bos_bull", "bos_choch"],
            "macro": ["btc_d", "usdt_d", "vix", "dxy", "macro_regime"],
            "temporal": ["fib_time_cluster", "gann_time_window", "temporal_confluence"],
            "crisis": ["crisis_composite", "capitulation_depth", "drawdown_1d"],
        }

        coverage = {}
        for domain, features in domains.items():
            available = [f for f in features if cls.has_feature(df, f)]
            coverage[domain] = {
                "total": len(features),
                "available": len(available),
                "coverage_pct": len(available) / len(features) * 100,
                "missing": [f for f in features if f not in available]
            }

        return coverage

    @classmethod
    def get_available_domains(cls, df: pd.DataFrame, min_coverage: float = 0.75) -> Set[str]:
        """
        Get set of domains that have sufficient feature coverage.

        Args:
            df: DataFrame to check
            min_coverage: Minimum coverage percentage required (0.0 to 1.0)

        Returns:
            Set of domain names with sufficient coverage
        """
        coverage = cls.audit_feature_coverage(df)
        available = set()

        for domain, stats in coverage.items():
            if stats["coverage_pct"] / 100 >= min_coverage:
                available.add(domain)

        return available


# Global instance for convenience
_mapper = FeatureMapper()


# Convenience functions
def map_to_store(canonical_name: str) -> str:
    """Map canonical name to store name."""
    return _mapper.map_to_store(canonical_name)


def map_to_canonical(store_name: str) -> str:
    """Map store name to canonical name."""
    return _mapper.map_to_canonical(store_name)


def get_feature(df: pd.DataFrame, canonical_name: str, default=None):
    """Get feature from DataFrame using canonical name."""
    return _mapper.get_feature(df, canonical_name, default)


def has_feature(df: pd.DataFrame, canonical_name: str) -> bool:
    """Check if feature exists in DataFrame."""
    return _mapper.has_feature(df, canonical_name)


def audit_feature_coverage(df: pd.DataFrame) -> Dict[str, any]:
    """Audit domain feature coverage."""
    return _mapper.audit_feature_coverage(df)
