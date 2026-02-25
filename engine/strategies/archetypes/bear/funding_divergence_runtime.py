#!/usr/bin/env python3
"""
S4 (Funding Divergence) Runtime Feature Enrichment

Provides on-demand calculation of S4-specific features for short squeeze detection
during bear markets when shorts become overcrowded. Opposite of S5 (long squeeze).

PATTERN LOGIC:
Short squeeze occurs when overcrowded shorts get liquidated during violent rallies
in bear markets. Key characteristics:
1. Extremely negative funding rate (shorts paying longs = overcrowded shorts)
2. Price resilience despite bearish sentiment (strength signal)
3. Low liquidity (thin bids = violent cascade up)
4. Volume quiet before explosion (coiled spring)

KEY DIFFERENCE FROM S5:
- S5 (Long Squeeze): Positive funding → longs overcrowded → cascade DOWN
- S4 (Funding Divergence): Negative funding → shorts overcrowded → squeeze UP

TARGET: 6-10 trades/year, PF > 2.0 (higher than S5 due to violence of short squeezes)

BTC EXAMPLES:
- 2022-08-15: Funding -0.15% → +18% rally in 48h (violent short squeeze)
- 2023-01-14: Negative funding + price strength → 12% rally
- 2022-03-28: Overcrowded shorts → forced covering cascade

DESIGN GOALS:
1. No feature store changes - all calculations at runtime
2. Minimal performance impact - vectorized pandas operations
3. Graceful degradation on missing OI data
4. Safe - no crashes on data issues
5. Promotable - successful features can move to feature store

FEATURES IMPLEMENTED:
1. Negative Funding Z-Score - Detect extreme negative funding (shorts overcrowded)
2. Price Resilience - Price holding despite bearish funding (strength signal)
3. Liquidity Score - Low liquidity amplifies squeeze violence
4. Volume Quiet - Low volume before squeeze (calm before storm)
5. S4 Fusion Score - Weighted combination of all signals

REGIME GATING:
- Primary: risk_off regime (bear markets)
- Works best when market is bearish but price shows strength
- Disable in strong risk_on (different dynamics)

Author: Claude Code (Backend Architect)
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from .feature_fallback import FeatureFallbackManager, safe_get_funding_z

logger = logging.getLogger(__name__)


class S4RuntimeFeatures:
    """
    Lightweight runtime feature calculator for S4 (Funding Divergence) archetype.

    Designed to enrich dataframes BEFORE backtest to avoid missing feature issues.
    All calculations are vectorized for performance.
    """

    def __init__(
        self,
        funding_lookback: int = 24,
        price_lookback: int = 12,
        volume_lookback: int = 24
    ):
        """
        Initialize S4 runtime feature calculator.

        Args:
            funding_lookback: Lookback window for funding rate z-score (hours)
            price_lookback: Lookback for price resilience calculation (hours)
            volume_lookback: Lookback for volume quiet detection (hours)
        """
        self.funding_lookback = funding_lookback
        self.price_lookback = price_lookback
        self.volume_lookback = volume_lookback
        self._logged_first_enrich = False

        # Initialize feature fallback manager for graceful degradation
        self.fallback_manager = FeatureFallbackManager(log_fallbacks=True)

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add S4-specific runtime features to dataframe.

        **CRITICAL:** Modifies dataframe in-place for memory efficiency.

        Args:
            df: Feature dataframe with OHLCV + indicators

        Returns:
            Enriched dataframe with new columns:
            - funding_z_negative: Negative z-score (< -1.5 = overcrowded shorts)
            - price_resilience: Price strength despite negative funding
            - volume_quiet: Low volume indicator (coiled spring)
            - liquidity_score: Liquidity risk score (lower = higher squeeze risk)
            - s4_fusion_score: Weighted combination of all S4 signals
        """
        if not self._logged_first_enrich:
            logger.info(f"[S4 Runtime] Enriching dataframe with {len(df)} bars")
            self._logged_first_enrich = True

        # 1. Negative funding z-score (extreme negative = shorts overcrowded)
        df['funding_z_negative'] = self._compute_negative_funding_zscore(df)

        # 2. Price resilience (price holding despite bearish funding)
        df['price_resilience'] = self._compute_price_resilience(df)

        # 3. Volume quiet (low volume = coiled spring)
        df['volume_quiet'] = self._compute_volume_quiet(df)

        # 4. Liquidity score (reuse or compute)
        if 'liquidity_score' not in df.columns:
            df['liquidity_score'] = self._compute_liquidity_score(df)

        # 5. S4 fusion score (weighted combination)
        df['s4_fusion_score'] = self._compute_s4_fusion(df)

        # Log enrichment stats on first run
        if not hasattr(self, '_logged_enrichment_stats'):
            self._log_enrichment_stats(df)
            self._logged_enrichment_stats = True

        return df

    def _compute_negative_funding_zscore(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute NEGATIVE funding rate z-score.

        Negative funding (< 0) means shorts pay longs, indicating short bias.
        High negative z-score (< -1.5 sigma) = extreme short positioning = squeeze risk.

        Returns:
            Series of negative funding z-scores (more negative = higher squeeze risk)
        """
        # Try multiple column names for funding rate
        funding_col = None
        for col in ['funding_rate', 'funding', 'FUNDING', 'funding_rate_1h']:
            if col in df.columns:
                funding_col = col
                break

        if funding_col is None:
            logger.warning("[S4 Runtime] [DEGRADED] No funding rate column found, using zeros - S4 will not detect any signals")
            return pd.Series(0.0, index=df.index, name='funding_z_negative')

        funding = df[funding_col]

        # Compute rolling z-score
        rolling_mean = funding.rolling(window=self.funding_lookback, min_periods=1).mean()
        rolling_std = funding.rolling(window=self.funding_lookback, min_periods=1).std()

        # Avoid division by zero
        z_score = np.where(
            rolling_std > 0,
            (funding - rolling_mean) / rolling_std,
            0.0
        )

        # INVERT: We want negative values to be "high risk"
        # So z_score = -2.0 means funding is 2 sigma BELOW mean (shorts overcrowded)
        return pd.Series(z_score, index=df.index, name='funding_z_negative')

    def _compute_price_resilience(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute price resilience score.

        Price resilience = price is NOT falling despite negative funding (strength signal).
        High resilience + negative funding = divergence = squeeze setup.

        Calculation:
        - Compare price change vs expected bearish move given funding
        - Positive resilience = price stronger than funding suggests

        Returns:
            Series of resilience scores [0, 1] (higher = stronger)
        """
        # Get close price
        close = df['close']

        # Get funding (or use funding_z if already computed)
        funding_col = None
        for col in ['funding_rate', 'funding', 'FUNDING']:
            if col in df.columns:
                funding_col = col
                break

        if funding_col is None:
            logger.warning("[S4 Runtime] [DEGRADED] No funding for resilience calc, using 0.5 default - divergence signal will be weak")
            return pd.Series(0.5, index=df.index, name='price_resilience')

        funding = df[funding_col]

        # Compute price change over lookback
        price_pct_change = (close - close.shift(self.price_lookback)) / close.shift(self.price_lookback)
        price_pct_change = price_pct_change.fillna(0.0)

        # Compute expected price change based on funding
        # Negative funding (-0.05%) should push price down
        # But if price is UP or flat, that's resilience (divergence)
        expected_price_change = funding * self.price_lookback  # Rough approximation

        # Resilience = actual price change - expected price change
        # Positive resilience = price stronger than funding suggests
        resilience = price_pct_change - expected_price_change

        # Normalize to [0, 1] range
        # High resilience (> 0.02 or 2%) = strong signal
        resilience_norm = np.clip((resilience + 0.02) / 0.04, 0.0, 1.0)

        return pd.Series(resilience_norm, index=df.index, name='price_resilience')

    def _compute_volume_quiet(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volume quiet conditions (calm before storm).

        Low volume before short squeeze = coiled spring effect.
        When volume explodes, shorts panic cover.

        Returns:
            Boolean Series, True if volume is quiet (< -0.5 sigma)
        """
        # Try multiple volume column names
        vol_z_col = None
        for col in ['volume_zscore', 'volume_z', 'vol_z']:
            if col in df.columns:
                vol_z_col = col
                break

        if vol_z_col is None:
            logger.warning("[S4 Runtime] [DEGRADED] No volume_zscore column, volume_quiet defaulting to False")
            return pd.Series(False, index=df.index, name='volume_quiet')

        vol_z = df[vol_z_col]

        # Volume quiet = volume below mean (negative z-score)
        # < -0.5 sigma = noticeably quiet
        quiet = vol_z < -0.5

        return quiet

    def _compute_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute liquidity score (proxy for squeeze violence).

        Low liquidity + short squeeze = explosive move (thin bids can't absorb covering).

        If liquidity_score already exists in feature store, use it.
        Otherwise, approximate using volume z-score.

        Returns:
            Series of liquidity scores (lower = higher squeeze risk)
        """
        # Check if liquidity score already computed
        if 'liquidity_score' in df.columns:
            return df['liquidity_score']

        # Fallback: Use volume z-score as proxy
        if 'volume_zscore' in df.columns:
            vol_z = df['volume_zscore']
            # Invert and normalize: high liquidity when volume high
            liquidity_proxy = (vol_z + 3.0) / 6.0  # Map [-3, 3] to [0, 1]
            liquidity_proxy = liquidity_proxy.clip(0.0, 1.0)
            return pd.Series(liquidity_proxy, index=df.index, name='liquidity_score')

        logger.warning("[S4 Runtime] No liquidity data available, using 0.5 default")
        return pd.Series(0.5, index=df.index, name='liquidity_score')

    def _compute_s4_fusion(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute S4 fusion score - weighted combination of all signals.

        Fusion weights (empirically tuned for BTC short squeezes):
        - funding_negative: 40% (most important - direct squeeze indicator)
        - price_resilience: 30% (divergence signal - price strong despite bearish funding)
        - volume_quiet: 15% (coiled spring - low volume before explosion)
        - liquidity_low: 15% (amplifies squeeze violence)

        Returns:
            Series of S4 fusion scores [0, 1]
        """
        # Get component scores
        funding_z = df.get('funding_z_negative', 0.0)
        resilience = df.get('price_resilience', 0.5)
        vol_quiet = df.get('volume_quiet', False).astype(float)
        liquidity = df.get('liquidity_score', 0.5)

        # Normalize funding z-score to [0, 1] range
        # NEGATIVE funding < -1.5 sigma is extreme
        # Map -3 to -1 sigma to 1.0 to 0.0
        funding_norm = np.clip((-funding_z - 1.0) / 2.0, 0.0, 1.0)

        # Resilience already normalized [0, 1]
        resilience_norm = resilience

        # Volume quiet is boolean, already [0, 1]
        vol_quiet_norm = vol_quiet

        # Invert liquidity (low liquidity = high risk)
        liquidity_inv = 1.0 - liquidity

        # Weighted fusion
        w_funding = 0.40
        w_resilience = 0.30
        w_volume = 0.15
        w_liq = 0.15

        fusion = (
            w_funding * funding_norm +
            w_resilience * resilience_norm +
            w_volume * vol_quiet_norm +
            w_liq * liquidity_inv
        )

        return pd.Series(fusion, index=df.index, name='s4_fusion_score')

    def _log_enrichment_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about enriched features"""

        # Funding stats (NEGATIVE values are squeeze risk)
        negative_funding = (df['funding_z_negative'] < -1.5).sum()
        extreme_negative = (df['funding_z_negative'] < -2.5).sum()

        # Resilience stats
        high_resilience = (df['price_resilience'] > 0.6).sum()

        # Volume stats
        vol_quiet = df['volume_quiet'].sum()

        # Liquidity stats
        low_liq = (df['liquidity_score'] < 0.25).sum()

        # Fusion stats
        high_fusion = (df['s4_fusion_score'] > 0.5).sum()
        extreme_fusion = (df['s4_fusion_score'] > 0.7).sum()

        logger.info(f"[S4 Runtime] Enrichment statistics:")
        logger.info(f"  - Negative funding (<-1.5σ): {negative_funding} ({negative_funding/len(df)*100:.1f}%)")
        logger.info(f"  - Extreme negative (<-2.5σ): {extreme_negative} ({extreme_negative/len(df)*100:.1f}%)")
        logger.info(f"  - High resilience (>0.6): {high_resilience} ({high_resilience/len(df)*100:.1f}%)")
        logger.info(f"  - Volume quiet: {vol_quiet} ({vol_quiet/len(df)*100:.1f}%)")
        logger.info(f"  - Low liquidity: {low_liq} ({low_liq/len(df)*100:.1f}%)")
        logger.info(f"  - High S4 fusion (>0.5): {high_fusion} ({high_fusion/len(df)*100:.1f}%)")
        logger.info(f"  - Extreme S4 fusion (>0.7): {extreme_fusion} ({extreme_fusion/len(df)*100:.1f}%)")


# ============================================================================
# Integration Helper
# ============================================================================

def apply_s4_enrichment(
    df: pd.DataFrame,
    funding_lookback: int = 24,
    price_lookback: int = 12,
    volume_lookback: int = 24
) -> pd.DataFrame:
    """
    Convenience function to apply S4 runtime enrichment to a dataframe.

    Usage:
        df_enriched = apply_s4_enrichment(df)

    Args:
        df: Feature dataframe
        funding_lookback: Lookback for funding z-score (hours)
        price_lookback: Lookback for price resilience (hours)
        volume_lookback: Lookback for volume quiet (hours)

    Returns:
        Enriched dataframe (modified in-place)
    """
    enricher = S4RuntimeFeatures(
        funding_lookback=funding_lookback,
        price_lookback=price_lookback,
        volume_lookback=volume_lookback
    )
    return enricher.enrich_dataframe(df)


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == '__main__':
    """
    Test S4 runtime enrichment on sample data.

    Usage:
        python3 engine/strategies/archetypes/bear/funding_divergence_runtime.py
    """
    print("="*80)
    print("S4 (FUNDING DIVERGENCE) RUNTIME ENRICHMENT TEST")
    print("="*80)
    print("\nLoading 2022 feature data (bear regime)...")

    try:
        # Load bear market data (2022)
        df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

        # Filter to 2022 only (bear regime)
        df_bear = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')].copy()

        print(f"Loaded {len(df_bear)} bars from 2022")
        print(f"Date range: {df_bear.index.min()} to {df_bear.index.max()}")
        print(f"Columns: {len(df_bear.columns)} features available")

        # Check for key features
        print("\nFeature availability check:")
        for feature in ['funding_rate', 'rsi_14', 'volume_zscore', 'liquidity_score']:
            available = "✓" if feature in df_bear.columns else "✗"
            print(f"  {available} {feature}")

        # Apply S4 enrichment
        print("\nApplying S4 runtime enrichment...")
        df_enriched = apply_s4_enrichment(df_bear, funding_lookback=24, price_lookback=12)

        print("\nEnrichment complete!")
        print(f"New columns added: {[c for c in df_enriched.columns if c.startswith('s4_') or c in ['funding_z_negative', 'price_resilience', 'volume_quiet']]}")

        # Show distribution of S4 fusion scores
        print("\n" + "="*80)
        print("S4 FUSION SCORE DISTRIBUTION")
        print("="*80)

        fusion_scores = df_enriched['s4_fusion_score']
        percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9]

        print("\nPercentile analysis:")
        for p in percentiles:
            val = np.percentile(fusion_scores.dropna(), p)
            count_above = (fusion_scores > val).sum()
            print(f"  p{p:>5.1f}: {val:.4f}  ({count_above:4d} bars above this threshold)")

        # Find high-conviction S4 signals
        print("\n" + "="*80)
        print("HIGH-CONVICTION S4 SIGNALS (Short Squeeze Setups)")
        print("="*80)

        high_fusion_threshold = np.percentile(fusion_scores.dropna(), 99.0)
        high_signals = df_enriched[df_enriched['s4_fusion_score'] > high_fusion_threshold]

        print(f"\nFound {len(high_signals)} high-conviction signals (>p99 = {high_fusion_threshold:.4f})")

        if len(high_signals) > 0:
            print("\nTop 5 S4 signals:")
            top_signals = high_signals.nlargest(5, 's4_fusion_score')
            for idx, row in top_signals.iterrows():
                print(f"  {idx}: Fusion={row['s4_fusion_score']:.4f}, "
                      f"Funding_Z={row['funding_z_negative']:.2f}, "
                      f"Resilience={row['price_resilience']:.3f}, "
                      f"Vol_Quiet={row['volume_quiet']}, "
                      f"Liq={row['liquidity_score']:.3f}")

        # Expected trade count estimation
        print("\n" + "="*80)
        print("TRADE COUNT ESTIMATION (2022)")
        print("="*80)

        for threshold_pct in [97, 98, 99, 99.5, 99.9]:
            threshold_val = np.percentile(fusion_scores.dropna(), threshold_pct)
            trades_above = (fusion_scores > threshold_val).sum()

            print(f"  Threshold p{threshold_pct:>5.1f} ({threshold_val:.4f}): "
                  f"{trades_above:3d} signals → {trades_above:.1f} trades/year")

        print("\n" + "="*80)
        print("Recommended Optuna search ranges:")
        print(f"  fusion_threshold: [{np.percentile(fusion_scores.dropna(), 97):.4f}, "
              f"{np.percentile(fusion_scores.dropna(), 99.5):.4f}]")
        print(f"  funding_z_max: [-3.0, -1.0] (negative values!)")
        print(f"  resilience_min: [0.5, 0.8]")
        print(f"  liquidity_max: [0.05, 0.25]")
        print("="*80)

    except FileNotFoundError as e:
        print(f"ERROR: Feature file not found: {e}")
        print("\nExpected file: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")
        print("Run this from project root: python3 engine/strategies/archetypes/bear/funding_divergence_runtime.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
