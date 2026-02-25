#!/usr/bin/env python3
"""
S5 (Long Squeeze) Runtime Feature Enrichment

Provides on-demand calculation of S5-specific features for long squeeze detection
in bull market capitulation/crisis phases. Follows the same architecture as S2 runtime.

PATTERN LOGIC:
Long squeeze occurs when overleveraged longs get liquidated during sharp pullbacks
in bull markets. Key characteristics:
1. Extremely high positive funding rate (longs paying shorts)
2. Rising open interest (many new longs entering)
3. Overbought conditions (RSI > 70)
4. Low liquidity (cascading liquidations)

TARGET: 7-12 trades/year, PF > 1.5

DESIGN GOALS:
1. No feature store changes - all calculations at runtime
2. Minimal performance impact - vectorized pandas operations
3. Graceful degradation on missing OI data (common issue)
4. Safe - no crashes on data issues
5. Promotable - successful features can move to feature store

FEATURES IMPLEMENTED:
1. Funding Z-Score - Standardized funding rate (detect extreme positive funding)
2. OI Change - Rising open interest indicator (with fallback if missing)
3. RSI Overbought - RSI > threshold detector
4. Liquidity Score - Low liquidity amplifies squeeze risk
5. S5 Fusion Score - Weighted combination of all signals

REGIME GATING:
- Primary: risk_on regime (bull markets)
- Crisis phase detection: When funding > +2 sigma (crisis/capitulation)
- Disable in risk_off (bear markets have different dynamics)

Author: Claude Code (Backend Architect)
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
import logging
from .feature_fallback import FeatureFallbackManager

logger = logging.getLogger(__name__)


class S5RuntimeFeatures:
    """
    Lightweight runtime feature calculator for S5 (Long Squeeze) archetype.

    Designed to enrich dataframes BEFORE backtest to avoid missing feature issues.
    All calculations are vectorized for performance.
    """

    def __init__(
        self,
        funding_lookback: int = 24,
        oi_lookback: int = 12,
        rsi_threshold: float = 70.0
    ):
        """
        Initialize S5 runtime feature calculator.

        Args:
            funding_lookback: Lookback window for funding rate z-score (hours)
            oi_lookback: Lookback window for OI change detection (hours)
            rsi_threshold: RSI overbought threshold (default 70)
        """
        self.funding_lookback = funding_lookback
        self.oi_lookback = oi_lookback
        self.rsi_threshold = rsi_threshold
        self._logged_first_enrich = False
        self._oi_available = True  # Track if OI data exists

        # Initialize feature fallback manager for graceful degradation
        self.fallback_manager = FeatureFallbackManager(log_fallbacks=True)

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add S5-specific runtime features to dataframe.

        **CRITICAL:** Modifies dataframe in-place for memory efficiency.

        Args:
            df: Feature dataframe with OHLCV + indicators

        Returns:
            Enriched dataframe with new columns:
            - funding_z_score: Z-score of funding rate (normalized)
            - oi_change: Change in open interest over lookback period
            - rsi_overbought: Boolean, True if RSI > threshold
            - liquidity_score: Liquidity risk score (lower = higher risk)
            - s5_fusion_score: Weighted combination of all S5 signals
        """
        if not self._logged_first_enrich:
            logger.info(f"[S5 Runtime] Enriching dataframe with {len(df)} bars")
            self._logged_first_enrich = True

        # 1. Funding rate z-score (extreme positive funding = long squeeze risk)
        df['funding_z_score'] = self._compute_funding_zscore(df)

        # 2. Open interest change (rising OI = more potential liquidations)
        df['oi_change'] = self._compute_oi_change(df)

        # 3. RSI overbought detection
        df['rsi_overbought'] = self._compute_rsi_overbought(df)

        # 4. Liquidity score (low liquidity = cascading liquidations)
        df['liquidity_score'] = self._compute_liquidity_score(df)

        # 5. S5 fusion score (weighted combination)
        df['s5_fusion_score'] = self._compute_s5_fusion(df)

        # Log enrichment stats on first run
        if not hasattr(self, '_logged_enrichment_stats'):
            self._log_enrichment_stats(df)
            self._logged_enrichment_stats = True

        return df

    def _compute_funding_zscore(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute funding rate z-score.

        Positive funding (> 0) means longs pay shorts, indicating long bias.
        High z-score (> 2 sigma) indicates extreme long positioning = squeeze risk.

        Returns:
            Series of funding rate z-scores
        """
        # Try multiple column names for funding rate
        funding_col = None
        for col in ['funding_rate', 'funding', 'FUNDING', 'funding_rate_1h']:
            if col in df.columns:
                funding_col = col
                break

        if funding_col is None:
            logger.warning("[S5 Runtime] [DEGRADED] No funding rate column found, using zeros - S5 will not detect any signals")
            return pd.Series(0.0, index=df.index, name='funding_z_score')

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

        return pd.Series(z_score, index=df.index, name='funding_z_score')

    def _compute_oi_change(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute open interest change over lookback period.

        Rising OI during high funding = many new longs entering (squeeze setup).

        **GRACEFUL DEGRADATION:** If OI data missing, returns zeros with warning.

        Returns:
            Series of OI percentage changes
        """
        # Try multiple column names for open interest
        oi_col = None
        for col in ['open_interest', 'oi', 'OI', 'oi_value', 'OI_CHANGE']:
            if col in df.columns:
                oi_col = col
                break

        if oi_col is None:
            if self._oi_available:  # Only log once
                logger.warning("[S5 Runtime] [DEGRADED] OI data not available - using 0.0 fallback for oi_change (expected for pre-2022 data)")
                logger.info("[S5 Runtime] S5 will use funding_extreme + rsi_overbought + liquidity signals (OI spike component disabled)")
                self._oi_available = False
            return pd.Series(0.0, index=df.index, name='oi_change')

        oi = df[oi_col]

        # Compute percentage change over lookback period
        oi_shifted = oi.shift(self.oi_lookback)
        oi_pct_change = np.where(
            oi_shifted > 0,
            (oi - oi_shifted) / oi_shifted,
            0.0
        )

        return pd.Series(oi_pct_change, index=df.index, name='oi_change')

    def _compute_rsi_overbought(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect RSI overbought conditions.

        RSI > threshold indicates overheated market, potential for mean reversion.

        Returns:
            Boolean Series, True if RSI > threshold
        """
        # Try multiple RSI column names
        rsi_col = None
        for col in ['rsi_14', 'RSI', 'rsi', 'RSI_14']:
            if col in df.columns:
                rsi_col = col
                break

        if rsi_col is None:
            logger.warning("[S5 Runtime] [DEGRADED] No RSI column found, rsi_overbought defaulting to False")
            return pd.Series(False, index=df.index, name='rsi_overbought')

        rsi = df[rsi_col]
        overbought = rsi > self.rsi_threshold

        return overbought

    def _compute_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute liquidity score (proxy for cascade risk).

        Low liquidity + high leverage = violent liquidation cascades.

        If liquidity_score already exists in feature store, use it.
        Otherwise, approximate using volume z-score.

        Returns:
            Series of liquidity scores (lower = higher risk)
        """
        # Check if liquidity score already computed
        if 'liquidity_score' in df.columns:
            return df['liquidity_score']

        # Fallback: Use volume z-score as proxy
        # Low volume (negative z-score) = low liquidity
        if 'volume_zscore' in df.columns:
            vol_z = df['volume_zscore']
            # Invert and normalize: high liquidity when volume high
            liquidity_proxy = (vol_z + 3.0) / 6.0  # Map [-3, 3] to [0, 1]
            liquidity_proxy = liquidity_proxy.clip(0.0, 1.0)
            return pd.Series(liquidity_proxy, index=df.index, name='liquidity_score')

        logger.warning("[S5 Runtime] [DEGRADED] No liquidity data available, using 0.5 default - cascade risk signal will be neutral")
        return pd.Series(0.5, index=df.index, name='liquidity_score')

    def _compute_s5_fusion(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute S5 fusion score - weighted combination of all signals.

        Fusion weights (empirically tuned):
        - funding_z_score: 35% (most important - direct squeeze indicator)
        - oi_change: 25% (amplifies risk if available)
        - rsi_overbought: 20% (confirms overheated conditions)
        - liquidity_low: 20% (amplifies cascade risk)

        Returns:
            Series of S5 fusion scores [0, 1]
        """
        # Get component scores
        funding_z = df.get('funding_z_score', 0.0)
        oi_change = df.get('oi_change', 0.0)
        rsi_ob = df.get('rsi_overbought', False).astype(float)
        liquidity = df.get('liquidity_score', 0.5)

        # Normalize funding z-score to [0, 1] range
        # Positive funding > 2 sigma is extreme (map 2-4 sigma to 0.5-1.0)
        funding_norm = np.clip(funding_z / 4.0, 0.0, 1.0)

        # Normalize OI change to [0, 1] range
        # Rising OI > 10% is significant (map 0-20% to 0-1)
        oi_norm = np.clip(oi_change / 0.20, 0.0, 1.0)

        # Invert liquidity (low liquidity = high risk)
        liquidity_inv = 1.0 - liquidity

        # Weighted fusion
        w_funding = 0.35
        w_oi = 0.25
        w_rsi = 0.20
        w_liq = 0.20

        fusion = (
            w_funding * funding_norm +
            w_oi * oi_norm +
            w_rsi * rsi_ob +
            w_liq * liquidity_inv
        )

        return pd.Series(fusion, index=df.index, name='s5_fusion_score')

    def _log_enrichment_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about enriched features"""

        # Funding stats
        high_funding = (df['funding_z_score'] > 2.0).sum()
        extreme_funding = (df['funding_z_score'] > 3.0).sum()

        # OI stats
        if self._oi_available:
            rising_oi = (df['oi_change'] > 0.10).sum()
        else:
            rising_oi = 0

        # RSI stats
        rsi_ob = df['rsi_overbought'].sum()

        # Liquidity stats
        low_liq = (df['liquidity_score'] < 0.25).sum()

        # Fusion stats
        high_fusion = (df['s5_fusion_score'] > 0.5).sum()
        extreme_fusion = (df['s5_fusion_score'] > 0.7).sum()

        logger.info("[S5 Runtime] Enrichment statistics:")
        logger.info(f"  - High funding (>2σ): {high_funding} ({high_funding/len(df)*100:.1f}%)")
        logger.info(f"  - Extreme funding (>3σ): {extreme_funding} ({extreme_funding/len(df)*100:.1f}%)")
        logger.info(f"  - Rising OI (>10%): {rising_oi} ({rising_oi/len(df)*100:.1f}%)")
        logger.info(f"  - RSI overbought: {rsi_ob} ({rsi_ob/len(df)*100:.1f}%)")
        logger.info(f"  - Low liquidity: {low_liq} ({low_liq/len(df)*100:.1f}%)")
        logger.info(f"  - High S5 fusion (>0.5): {high_fusion} ({high_fusion/len(df)*100:.1f}%)")
        logger.info(f"  - Extreme S5 fusion (>0.7): {extreme_fusion} ({extreme_fusion/len(df)*100:.1f}%)")


# ============================================================================
# Integration Helper
# ============================================================================

def apply_s5_enrichment(
    df: pd.DataFrame,
    funding_lookback: int = 24,
    oi_lookback: int = 12,
    rsi_threshold: float = 70.0
) -> pd.DataFrame:
    """
    Convenience function to apply S5 runtime enrichment to a dataframe.

    Usage:
        df_enriched = apply_s5_enrichment(df)

    Args:
        df: Feature dataframe
        funding_lookback: Lookback for funding z-score (hours)
        oi_lookback: Lookback for OI change (hours)
        rsi_threshold: RSI overbought threshold

    Returns:
        Enriched dataframe (modified in-place)
    """
    enricher = S5RuntimeFeatures(
        funding_lookback=funding_lookback,
        oi_lookback=oi_lookback,
        rsi_threshold=rsi_threshold
    )
    return enricher.enrich_dataframe(df)


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == '__main__':
    """
    Test S5 runtime enrichment on sample data.

    Usage:
        python3 engine/strategies/archetypes/bear/long_squeeze_runtime.py
    """
    print("="*80)
    print("S5 (LONG SQUEEZE) RUNTIME ENRICHMENT TEST")
    print("="*80)
    print("\nLoading 2023-2024 feature data (bull regime)...")

    try:
        # Load bull market data (2023-2024)
        df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

        # Filter to 2023-2024 only (bull regime)
        df_bull = df[(df.index >= '2023-01-01') & (df.index < '2025-01-01')].copy()

        print(f"Loaded {len(df_bull)} bars from 2023-2024")
        print(f"Date range: {df_bull.index.min()} to {df_bull.index.max()}")
        print(f"Columns: {len(df_bull.columns)} features available")

        # Check for key features
        print("\nFeature availability check:")
        for feature in ['funding_rate', 'open_interest', 'rsi_14', 'volume_zscore', 'liquidity_score']:
            available = "✓" if feature in df_bull.columns else "✗"
            print(f"  {available} {feature}")

        # Apply S5 enrichment
        print("\nApplying S5 runtime enrichment...")
        df_enriched = apply_s5_enrichment(df_bull, funding_lookback=24, oi_lookback=12, rsi_threshold=70.0)

        print("\nEnrichment complete!")
        print(f"New columns added: {[c for c in df_enriched.columns if c.startswith('s5_') or c in ['funding_z_score', 'oi_change', 'rsi_overbought']]}")

        # Show distribution of S5 fusion scores
        print("\n" + "="*80)
        print("S5 FUSION SCORE DISTRIBUTION")
        print("="*80)

        fusion_scores = df_enriched['s5_fusion_score']
        percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9]

        print("\nPercentile analysis:")
        for p in percentiles:
            val = np.percentile(fusion_scores.dropna(), p)
            count_above = (fusion_scores > val).sum()
            print(f"  p{p:>5.1f}: {val:.4f}  ({count_above:4d} bars above this threshold)")

        # Find high-conviction S5 signals
        print("\n" + "="*80)
        print("HIGH-CONVICTION S5 SIGNALS")
        print("="*80)

        high_fusion_threshold = np.percentile(fusion_scores.dropna(), 99.0)
        high_signals = df_enriched[df_enriched['s5_fusion_score'] > high_fusion_threshold]

        print(f"\nFound {len(high_signals)} high-conviction signals (>p99 = {high_fusion_threshold:.4f})")

        if len(high_signals) > 0:
            print("\nTop 5 S5 signals:")
            top_signals = high_signals.nlargest(5, 's5_fusion_score')
            for idx, row in top_signals.iterrows():
                print(f"  {idx}: Fusion={row['s5_fusion_score']:.4f}, "
                      f"Funding_Z={row['funding_z_score']:.2f}, "
                      f"OI_Change={row['oi_change']:.3f}, "
                      f"RSI_OB={row['rsi_overbought']}, "
                      f"Liq={row['liquidity_score']:.3f}")

        # Expected trade count estimation
        print("\n" + "="*80)
        print("TRADE COUNT ESTIMATION (2023-2024)")
        print("="*80)

        for threshold_pct in [97, 98, 99, 99.5, 99.9]:
            threshold_val = np.percentile(fusion_scores.dropna(), threshold_pct)
            trades_above = (fusion_scores > threshold_val).sum()
            trades_per_year = trades_above / 2.0  # 2 years of data

            print(f"  Threshold p{threshold_pct:>5.1f} ({threshold_val:.4f}): "
                  f"{trades_above:3d} signals → {trades_per_year:.1f} trades/year")

        print("\n" + "="*80)
        print("Recommended Optuna search ranges:")
        print(f"  fusion_threshold: [{np.percentile(fusion_scores.dropna(), 97):.4f}, "
              f"{np.percentile(fusion_scores.dropna(), 99.5):.4f}]")
        print("  funding_z_min: [1.0, 3.0]")
        print("  rsi_min: [70, 85]")
        print("  liquidity_max: [0.05, 0.25]")
        print("="*80)

    except FileNotFoundError as e:
        print(f"ERROR: Feature file not found: {e}")
        print("\nExpected file: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")
        print("Run this from project root: python3 engine/strategies/archetypes/bear/long_squeeze_runtime.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
