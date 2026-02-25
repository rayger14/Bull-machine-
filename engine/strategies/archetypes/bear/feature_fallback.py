#!/usr/bin/env python3
"""
Feature Fallback Module for Bear Archetypes

Provides graceful degradation for missing OI and funding features.
This module ensures bear archetypes (S1, S2, S4, S5) don't crash when
OI data is unavailable (33% coverage starting mid-2022).

DESIGN PRINCIPLES:
1. Never crash on missing features
2. Fallbacks must be economically sensible (correlated signals)
3. Log all degradation events for monitoring
4. Preserve archetype intent (don't change strategy)

FALLBACK HIERARCHY:
- oi_change_spike_24h → volume_panic (capitulation volume spike)
- oi_change_spike_12h → volume_climax_last_3b (multi-bar volume pattern)
- oi_change_spike_6h → volume_zscore (current volume spike)
- oi_change_spike_3h → volume_zscore (current volume spike)
- oi_z → volume_z (volume z-score as proxy)
- oi_change_24h → volume_pct_change_24h (volume change)
- oi_change_12h → volume_pct_change_12h (volume change)
- funding_Z → funding_rate (raw funding rate, not z-scored)

Author: Claude Code (Backend Architect)
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, Any

logger = logging.getLogger(__name__)


class FeatureFallbackManager:
    """
    Manages feature fallback logic for missing OI/funding data.

    Tracks degradation events and provides safe feature access with
    economically sensible fallbacks.
    """

    def __init__(self, log_fallbacks: bool = True):
        """
        Initialize fallback manager.

        Args:
            log_fallbacks: Whether to log fallback usage (default True)
        """
        self.log_fallbacks = log_fallbacks
        self._fallback_stats = {}  # Track usage of each fallback
        self._first_fallback_logged = set()  # Only log first occurrence per feature

    def safe_get(
        self,
        features: Union[pd.Series, dict],
        primary_key: str,
        fallback_keys: list,
        default: float = 0.0,
        transform: Optional[callable] = None
    ) -> float:
        """
        Safely get feature value with fallback chain.

        Args:
            features: DataFrame row (pd.Series) or dict of features
            primary_key: Primary feature name to try first
            fallback_keys: List of fallback feature names (in priority order)
            default: Default value if all keys missing
            transform: Optional transformation function for fallback values
                      (e.g., lambda x: x / 2 to scale volume to OI proxy)

        Returns:
            Feature value or fallback value or default

        Example:
            # Get OI spike with volume fallback
            oi_spike = safe_get(
                features,
                'oi_change_spike_24h',
                ['volume_panic', 'capitulation_depth'],
                default=0.0,
                transform=lambda x: x * 0.8  # Scale down volume signal
            )
        """
        # Try primary key first
        value = self._try_get(features, primary_key)
        if value is not None:
            return value

        # Try fallback chain
        for fallback_key in fallback_keys:
            value = self._try_get(features, fallback_key)
            if value is not None:
                # Log fallback usage (only once per feature)
                self._log_fallback(primary_key, fallback_key)

                # Apply transformation if provided
                if transform:
                    value = transform(value)

                return value

        # All keys missing, return default
        self._log_fallback(primary_key, "DEFAULT", default)
        return default

    def _try_get(self, features: Union[pd.Series, dict], key: str) -> Optional[float]:
        """
        Try to get value from features, handling both Series and dict.

        Returns:
            Value if found and not NaN/None, otherwise None
        """
        if isinstance(features, pd.Series):
            if key in features.index:
                val = features[key]
                if val is not None and not pd.isna(val):
                    return float(val)
        elif isinstance(features, dict):
            if key in features:
                val = features[key]
                if val is not None and not pd.isna(val):
                    return float(val)
        return None

    def _log_fallback(self, primary: str, fallback: str, value: Any = None) -> None:
        """Log fallback usage (only first occurrence)."""
        if not self.log_fallbacks:
            return

        key = f"{primary}->{fallback}"

        # Track statistics
        if key not in self._fallback_stats:
            self._fallback_stats[key] = 0
        self._fallback_stats[key] += 1

        # Log first occurrence
        if key not in self._first_fallback_logged:
            if fallback == "DEFAULT":
                logger.warning(
                    f"[DEGRADED] Feature '{primary}' missing, using default={value}"
                )
            else:
                logger.warning(
                    f"[DEGRADED] Feature '{primary}' missing, using fallback '{fallback}'"
                )
            self._first_fallback_logged.add(key)

    def get_stats(self) -> dict:
        """
        Get fallback usage statistics.

        Returns:
            Dict mapping fallback_key -> usage_count
        """
        return self._fallback_stats.copy()

    def log_summary(self) -> None:
        """Log summary of all fallback usage."""
        if not self._fallback_stats:
            logger.info("[FeatureFallback] No fallbacks used (all features available)")
            return

        logger.info("[FeatureFallback] Fallback usage summary:")
        for key, count in sorted(self._fallback_stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {key}: {count} occurrences")


# ============================================================================
# Pre-configured Fallback Maps for Bear Archetypes
# ============================================================================

# OI spike fallbacks (ordered by correlation strength)
OI_SPIKE_FALLBACKS = {
    'oi_change_spike_24h': [
        'volume_panic',           # Highest correlation (panic selling volume)
        'capitulation_depth',     # Deep drawdown proxy
        'volume_climax_last_3b',  # Multi-bar volume pattern
        'volume_zscore',          # Current volume spike
    ],
    'oi_change_spike_12h': [
        'volume_climax_last_3b',  # Multi-bar volume pattern (best match)
        'volume_panic',           # Single-bar panic
        'volume_zscore',          # Current volume spike
    ],
    'oi_change_spike_6h': [
        'volume_zscore',          # Best single-bar proxy
        'volume_panic',           # Panic selling
    ],
    'oi_change_spike_3h': [
        'volume_zscore',          # Best single-bar proxy
    ],
}

# OI change fallbacks (percentage change proxies)
OI_CHANGE_FALLBACKS = {
    'oi_change_24h': [
        'volume_pct_change_24h',  # Volume change as proxy
        'volume_panic',           # Volume spike indicator
    ],
    'oi_change_12h': [
        'volume_pct_change_12h',  # Volume change as proxy
        'volume_climax_last_3b',  # Multi-bar pattern
    ],
    'oi_z': [
        'volume_z',               # Volume z-score (best proxy)
        'volume_zscore',          # Alternative column name
    ],
}

# Funding fallbacks (for missing z-scored funding)
FUNDING_FALLBACKS = {
    'funding_Z': [
        'funding_rate',           # Raw funding rate (not z-scored)
        'funding',                # Alternative column name
    ],
    'funding_z': [
        'funding_rate',           # Raw funding rate
        'funding',                # Alternative column name
    ],
}


# ============================================================================
# Convenience Functions
# ============================================================================

def safe_get_oi_spike(
    features: Union[pd.Series, dict],
    timeframe: str = '24h',
    manager: Optional[FeatureFallbackManager] = None
) -> float:
    """
    Safely get OI spike feature with automatic fallback.

    Args:
        features: DataFrame row or dict
        timeframe: '24h', '12h', '6h', or '3h'
        manager: Optional fallback manager (creates default if None)

    Returns:
        OI spike value or fallback value
    """
    if manager is None:
        manager = FeatureFallbackManager()

    primary = f'oi_change_spike_{timeframe}'
    fallbacks = OI_SPIKE_FALLBACKS.get(primary, [])

    return manager.safe_get(features, primary, fallbacks, default=0.0)


def safe_get_oi_change(
    features: Union[pd.Series, dict],
    timeframe: str = '24h',
    manager: Optional[FeatureFallbackManager] = None
) -> float:
    """
    Safely get OI change percentage with automatic fallback.

    Args:
        features: DataFrame row or dict
        timeframe: '24h' or '12h'
        manager: Optional fallback manager

    Returns:
        OI change value or fallback value
    """
    if manager is None:
        manager = FeatureFallbackManager()

    primary = f'oi_change_{timeframe}'
    fallbacks = OI_CHANGE_FALLBACKS.get(primary, [])

    return manager.safe_get(features, primary, fallbacks, default=0.0)


def safe_get_oi_z(
    features: Union[pd.Series, dict],
    manager: Optional[FeatureFallbackManager] = None
) -> float:
    """
    Safely get OI z-score with automatic fallback to volume z-score.

    Args:
        features: DataFrame row or dict
        manager: Optional fallback manager

    Returns:
        OI z-score or volume z-score fallback
    """
    if manager is None:
        manager = FeatureFallbackManager()

    return manager.safe_get(
        features,
        'oi_z',
        OI_CHANGE_FALLBACKS['oi_z'],
        default=0.0
    )


def safe_get_funding_z(
    features: Union[pd.Series, dict],
    manager: Optional[FeatureFallbackManager] = None,
    compute_zscore: bool = True
) -> float:
    """
    Safely get funding z-score with automatic fallback to raw funding.

    Args:
        features: DataFrame row or dict
        manager: Optional fallback manager
        compute_zscore: If True and fallback to raw funding, compute rough z-score
                       using simple normalization (funding / 0.01)

    Returns:
        Funding z-score or fallback value
    """
    if manager is None:
        manager = FeatureFallbackManager()

    # Try funding_Z first (capital Z)
    value = manager.safe_get(
        features,
        'funding_Z',
        FUNDING_FALLBACKS['funding_Z'],
        default=None
    )

    # If got raw funding rate (not z-scored), optionally convert
    if value is not None and compute_zscore:
        # Check if this looks like raw funding (typically -0.1 to +0.1)
        if abs(value) < 1.0:  # Likely raw funding, not z-score
            # Simple normalization: divide by typical funding rate std dev (0.01)
            # This gives rough z-score approximation
            value = value / 0.01

    return value if value is not None else 0.0


# ============================================================================
# DataFrame Enrichment (Batch Mode)
# ============================================================================

def enrich_with_oi_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich dataframe with OI fallback features in batch mode.

    For any missing OI features, creates fallback columns using correlated signals.
    This ensures archetypes don't crash even with partial OI data coverage.

    Args:
        df: Feature dataframe

    Returns:
        Enriched dataframe with fallback features added (in-place modification)
    """
    logger.info("[FeatureFallback] Enriching dataframe with OI fallbacks...")

    # Check which OI features are missing
    missing_features = []

    # OI spike features
    for timeframe in ['24h', '12h', '6h', '3h']:
        col = f'oi_change_spike_{timeframe}'
        if col not in df.columns:
            missing_features.append(col)
            # Add fallback based on hierarchy
            if 'volume_panic' in df.columns:
                df[col] = df['volume_panic']
                logger.info(f"  Created fallback: {col} -> volume_panic")
            elif 'volume_zscore' in df.columns:
                df[col] = df['volume_zscore'].clip(lower=0) / 3.0  # Normalize
                logger.info(f"  Created fallback: {col} -> volume_zscore (normalized)")
            else:
                df[col] = 0.0
                logger.warning(f"  Created default: {col} = 0.0 (no volume data)")

    # OI change features
    for timeframe in ['24h', '12h']:
        col = f'oi_change_{timeframe}'
        if col not in df.columns:
            missing_features.append(col)
            vol_col = f'volume_pct_change_{timeframe}'
            if vol_col in df.columns:
                df[col] = df[vol_col] * 0.5  # Scale down (volume changes > OI changes)
                logger.info(f"  Created fallback: {col} -> {vol_col} (scaled)")
            else:
                df[col] = 0.0
                logger.warning(f"  Created default: {col} = 0.0")

    # OI z-score
    if 'oi_z' not in df.columns:
        missing_features.append('oi_z')
        if 'volume_z' in df.columns:
            df['oi_z'] = df['volume_z']
            logger.info("  Created fallback: oi_z -> volume_z")
        elif 'volume_zscore' in df.columns:
            df['oi_z'] = df['volume_zscore']
            logger.info("  Created fallback: oi_z -> volume_zscore")
        else:
            df['oi_z'] = 0.0
            logger.warning("  Created default: oi_z = 0.0")

    if missing_features:
        logger.info(f"[FeatureFallback] Created {len(missing_features)} fallback features")
    else:
        logger.info("[FeatureFallback] All OI features present, no fallbacks needed")

    return df


def enrich_with_funding_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich dataframe with funding fallback features in batch mode.

    Creates z-scored funding from raw funding rate if needed.

    Args:
        df: Feature dataframe

    Returns:
        Enriched dataframe with fallback features added (in-place modification)
    """
    logger.info("[FeatureFallback] Enriching dataframe with funding fallbacks...")

    # Check for funding_Z (capital Z - our preferred column)
    if 'funding_Z' not in df.columns:
        if 'funding_rate' in df.columns:
            # Compute rolling z-score from raw funding rate
            funding = df['funding_rate']
            rolling_mean = funding.rolling(window=24, min_periods=1).mean()
            rolling_std = funding.rolling(window=24, min_periods=1).std()

            df['funding_Z'] = np.where(
                rolling_std > 0,
                (funding - rolling_mean) / rolling_std,
                0.0
            )
            logger.info("  Created fallback: funding_Z -> funding_rate (z-scored)")
        elif 'funding' in df.columns:
            # Same for alternative column name
            funding = df['funding']
            rolling_mean = funding.rolling(window=24, min_periods=1).mean()
            rolling_std = funding.rolling(window=24, min_periods=1).std()

            df['funding_Z'] = np.where(
                rolling_std > 0,
                (funding - rolling_mean) / rolling_std,
                0.0
            )
            logger.info("  Created fallback: funding_Z -> funding (z-scored)")
        else:
            df['funding_Z'] = 0.0
            logger.warning("  Created default: funding_Z = 0.0 (no funding data)")
    else:
        logger.info("[FeatureFallback] funding_Z already present")

    return df


# ============================================================================
# Batch Enrichment (All Fallbacks)
# ============================================================================

def enrich_with_all_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all fallback enrichments in one call.

    Convenience function that applies both OI and funding fallbacks.

    Args:
        df: Feature dataframe

    Returns:
        Enriched dataframe (in-place modification)
    """
    df = enrich_with_oi_fallbacks(df)
    df = enrich_with_funding_fallbacks(df)
    return df


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == '__main__':
    """
    Test feature fallback on sample data.

    Usage:
        python3 engine/strategies/archetypes/bear/feature_fallback.py
    """
    print("="*80)
    print("FEATURE FALLBACK MODULE TEST")
    print("="*80)
    print("\nLoading 2022 feature data (partial OI coverage)...")

    try:
        df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
        df_2022 = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')].copy()

        print(f"Loaded {len(df_2022)} bars from 2022")

        # Test fallback manager
        print("\n" + "="*80)
        print("TEST 1: Fallback Manager (Single-Row Access)")
        print("="*80)

        manager = FeatureFallbackManager()
        sample_row = df_2022.iloc[1000]

        # Test OI spike fallback
        oi_spike = safe_get_oi_spike(sample_row, '24h', manager)
        print(f"\nOI spike (24h): {oi_spike:.4f}")

        # Test funding Z fallback
        funding_z = safe_get_funding_z(sample_row, manager)
        print(f"Funding Z: {funding_z:.4f}")

        # Show stats
        print("\n" + "="*80)
        print("Fallback usage statistics:")
        print("="*80)
        manager.log_summary()

        # Test batch enrichment
        print("\n" + "="*80)
        print("TEST 2: Batch Enrichment (DataFrame)")
        print("="*80)

        # Check what's missing
        oi_features = [
            'oi_change_spike_24h', 'oi_change_spike_12h',
            'oi_change_24h', 'oi_z', 'funding_Z'
        ]

        print("\nFeature availability BEFORE enrichment:")
        for feat in oi_features:
            present = "✓" if feat in df_2022.columns else "✗"
            print(f"  {present} {feat}")

        # Apply enrichment
        df_enriched = enrich_with_all_fallbacks(df_2022)

        print("\nFeature availability AFTER enrichment:")
        for feat in oi_features:
            present = "✓" if feat in df_enriched.columns else "✗"
            print(f"  {present} {feat}")

        # Show sample values
        print("\n" + "="*80)
        print("Sample enriched values (row 1000):")
        print("="*80)
        sample = df_enriched.iloc[1000]
        for feat in oi_features:
            if feat in sample.index:
                print(f"  {feat}: {sample[feat]:.4f}")

        print("\n" + "="*80)
        print("TEST COMPLETE!")
        print("="*80)

    except FileNotFoundError as e:
        print(f"ERROR: Feature file not found: {e}")
        print("Run this from project root")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
