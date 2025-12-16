#!/usr/bin/env python3
"""
Liquidity Vacuum Reversal Runtime Feature Enrichment

Provides on-demand calculation of Liquidity Vacuum Reversal-specific features for
capitulation reversal detection during BTC sell-offs when liquidity drains and
panic selling exhausts itself.

PATTERN LOGIC:
Liquidity vacuum reversals occur when orderbook liquidity evaporates during sell-offs,
creating "air pockets" where sellers exhaust themselves. The resulting vacuum creates
explosive short-covering bounces as there's no resistance. Key characteristics:
1. Extreme liquidity drain (liquidity_score < 0.15)
2. Panic volume spike (volume_zscore > 2.0)
3. Deep lower wick (wick_lower_ratio > 0.30) - sellers exhausted, buyers stepping in
4. Often occurs during crisis/capitulation events

KEY DIFFERENCE FROM SIMILAR PATTERNS:
- Liquidity Vacuum: ANY liquidity vacuum context (general case)
- Capitulation Fade (future): EXTREME version requiring massive wick + volume climax + crisis
- Funding Divergence: Focuses on funding rates, NOT liquidity
- Long Squeeze: OPPOSITE direction (short bias)

TARGET: 10-15 trades/year, PF > 2.0 (high conviction reversals)

BTC EXAMPLES:
- 2022-06-18: Luna capitulation → -70% → violent 25% bounce in 24h
- 2022-11-09: FTX collapse → liquidity vacuum → explosive reversal
- 2022-05-12: LUNA death spiral → extreme capitulation → sharp bounce

DESIGN GOALS:
1. No feature store changes - all calculations at runtime
2. Minimal performance impact - vectorized pandas operations
3. Graceful degradation on missing data
4. Safe - no crashes on data issues
5. Promotable - successful features can move to feature store

FEATURES IMPLEMENTED (V1 - Single-bar):
1. Wick Lower Ratio - Deep lower wick detection (rejection signal)
2. Liquidity Vacuum Score - Extreme liquidity drain detection
3. Volume Panic - Capitulation selling detection
4. Crisis Context - Macro stress indicators (VIX, DXY)
5. Liquidity Vacuum Fusion Score - Weighted combination of all signals

FEATURES V2 (Multi-bar capitulation encoding - FIXES LIQUIDITY PARADOX):
6. Liquidity Drain Pct - RELATIVE drain vs 7d avg (fixes June 18 missed detection)
7. Liquidity Velocity - Rate of drain (separates quiet vs collapse)
8. Liquidity Persistence - Multi-bar sustained stress count
9. Capitulation Depth - Drawdown from recent high
10. Crisis Composite - Composite macro panic score
11. Volume Climax Last 3B - Max volume panic in last 3 bars
12. Wick Exhaustion Last 3B - Max wick rejection in last 3 bars

REGIME GATING:
- Primary: risk_off regime (bear markets)
- Works in any regime during capitulation events
- Higher weight during crisis periods

Author: Claude Code (Backend Architect)
Date: 2025-11-21
"""

import logging

import numpy as np
import pandas as pd
from .feature_fallback import FeatureFallbackManager, safe_get_funding_z

logger = logging.getLogger(__name__)


class LiquidityVacuumRuntimeFeatures:
    """
    Lightweight runtime feature calculator for Liquidity Vacuum Reversal archetype.

    Designed to enrich dataframes BEFORE backtest to avoid missing feature issues.
    All calculations are vectorized for performance.
    """

    def __init__(
        self,
        lookback: int = 24,
        volume_lookback: int = 24
    ):
        """
        Initialize Liquidity Vacuum runtime feature calculator.

        Args:
            lookback: General lookback window (hours)
            volume_lookback: Lookback for volume z-score calculation (hours)
        """
        self.lookback = lookback
        self.volume_lookback = volume_lookback
        self._logged_first_enrich = False

        # Initialize feature fallback manager for graceful degradation
        self.fallback_manager = FeatureFallbackManager(log_fallbacks=True)

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Liquidity Vacuum-specific runtime features to dataframe.

        **CRITICAL:** Modifies dataframe in-place for memory efficiency.

        Args:
            df: Feature dataframe with OHLCV + indicators

        Returns:
            Enriched dataframe with new columns:
            - wick_lower_ratio: Deep lower wick detection (NEW)
            - liquidity_vacuum_score: Extreme liquidity drain detection
            - volume_panic: Volume spike indicator (capitulation selling)
            - crisis_context: Macro stress indicator (VIX + DXY)
            - liquidity_vacuum_fusion: Weighted combination of all signals
        """
        if not self._logged_first_enrich:
            logger.info(f"[Liquidity Vacuum Runtime] Enriching dataframe with {len(df)} bars")
            self._logged_first_enrich = True

        # 1. Wick lower ratio (NEW - critical for pattern detection)
        df['wick_lower_ratio'] = self._compute_wick_lower_ratio(df)

        # 2. Liquidity vacuum score (extreme drain detection)
        df['liquidity_vacuum_score'] = self._compute_liquidity_vacuum_score(df)

        # 3. Volume panic (capitulation selling)
        df['volume_panic'] = self._compute_volume_panic(df)

        # 4. Crisis context (macro stress)
        df['crisis_context'] = self._compute_crisis_context(df)

        # 5. Liquidity Vacuum fusion score (weighted combination)
        df['liquidity_vacuum_fusion'] = self._compute_liquidity_vacuum_fusion(df)

        # ====================================================================
        # V2 FEATURES: Multi-bar capitulation encoding (FIXES LIQUIDITY PARADOX)
        # ====================================================================

        # 6. Liquidity drain percentage (RELATIVE vs 7d avg - fixes June 18!)
        df['liquidity_drain_pct'] = self._compute_liquidity_drain_pct(df)

        # 7. Liquidity velocity (rate of drain)
        df['liquidity_velocity'] = self._compute_liquidity_velocity(df)

        # 8. Liquidity persistence (multi-bar sustained stress)
        df['liquidity_persistence'] = self._compute_liquidity_persistence(df)

        # 9. Capitulation depth (drawdown from recent high)
        df['capitulation_depth'] = self._compute_capitulation_depth(df)

        # 10. Crisis composite (macro panic indicator)
        df['crisis_composite'] = self._compute_crisis_composite(df)

        # 11. Volume climax last 3 bars (multi-bar volume pattern)
        df['volume_climax_last_3b'] = self._compute_volume_climax_last_3b(df)

        # 12. Wick exhaustion last 3 bars (multi-bar wick pattern)
        df['wick_exhaustion_last_3b'] = self._compute_wick_exhaustion_last_3b(df)

        # Log enrichment stats on first run
        if not hasattr(self, '_logged_enrichment_stats'):
            self._log_enrichment_stats(df)
            self._logged_enrichment_stats = True

        return df

    def _compute_wick_lower_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate lower wick as percentage of candle range (vectorized).

        Lower wick = distance from body low to candle low
        Deep lower wick (>30% of range) indicates sellers exhausted, buyers stepping in.

        Returns:
            Series of wick_lower_ratio values [0, 1]
            - 0.0 = no lower wick
            - 0.3 = 30% of candle is lower wick (significant rejection)
            - 0.5 = 50% of candle is lower wick (extreme rejection)

        Example:
            High:  $20,000
            Close: $19,800 (body top if red candle)
            Open:  $19,500 (body bottom if red candle)
            Low:   $19,000

            Body low = min(open, close) = $19,500
            Wick lower = $19,500 - $19,000 = $500
            Candle range = $20,000 - $19,000 = $1,000
            Ratio = $500 / $1,000 = 0.50 (50% lower wick - EXTREME)
        """
        # Calculate candle range
        candle_range = df['high'] - df['low']

        # Calculate body low (min of open/close)
        body_low = pd.DataFrame({'open': df['open'], 'close': df['close']}).min(axis=1)

        # Calculate lower wick
        wick_lower = body_low - df['low']

        # Normalize to [0, 1] (avoid division by zero)
        wick_lower_ratio = np.where(
            candle_range > 0,
            (wick_lower / candle_range).clip(0.0, 1.0),
            0.0
        )

        return pd.Series(wick_lower_ratio, index=df.index, name='wick_lower_ratio')

    def _compute_liquidity_vacuum_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute liquidity vacuum score (extreme drain detection).

        Low liquidity score (< 0.15) indicates orderbook vacuum - creates explosive
        reversals as there's no resistance when shorts cover or buyers step in.

        Returns:
            Series of liquidity vacuum scores (inverted for clarity)
            - Higher score = deeper vacuum = stronger signal
        """
        # Check if liquidity score already computed
        if 'liquidity_score' not in df.columns:
            logger.warning("[Liquidity Vacuum Runtime] No liquidity_score column, using 0.5 default")
            return pd.Series(0.5, index=df.index, name='liquidity_vacuum_score')

        liquidity = df['liquidity_score']

        # Invert: low liquidity = high vacuum score
        # Normalize so liquidity < 0.15 maps to high scores
        vacuum_score = 1.0 - (liquidity / 0.15).clip(0.0, 1.0)

        return pd.Series(vacuum_score, index=df.index, name='liquidity_vacuum_score')

    def _compute_volume_panic(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volume panic conditions (capitulation selling).

        High volume spike (z-score > 2.0) indicates panic selling - when combined
        with liquidity vacuum, creates explosive reversal setup.

        Returns:
            Series of volume panic scores [0, 1]
            - Higher score = more panic = stronger signal
        """
        # Try multiple volume z-score column names
        vol_z_col = None
        for col in ['volume_zscore', 'volume_z', 'vol_z', 'volume_Z']:
            if col in df.columns:
                vol_z_col = col
                break

        if vol_z_col is None:
            logger.warning("[Liquidity Vacuum Runtime] No volume_zscore column, using 0.0 default")
            return pd.Series(0.0, index=df.index, name='volume_panic')

        vol_z = df[vol_z_col]

        # Normalize: z > 2.0 is panic, z > 3.0 is extreme panic
        # Map [2.0, 4.0] to [0.0, 1.0]
        panic_score = ((vol_z - 2.0) / 2.0).clip(0.0, 1.0)

        return pd.Series(panic_score, index=df.index, name='volume_panic')

    def _compute_crisis_context(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute crisis context score from macro indicators.

        High VIX + high DXY = risk-off environment where liquidity vacuums
        are more likely to occur and reversals are more violent.

        Returns:
            Series of crisis context scores [0, 1]
        """
        # Get VIX z-score (volatility/fear indicator)
        vix_z = df.get('VIX_Z', pd.Series(0.0, index=df.index))
        if isinstance(vix_z, (int, float)):
            vix_z = pd.Series(vix_z, index=df.index)

        # Get DXY z-score (dollar strength/risk-off indicator)
        dxy_z = df.get('DXY_Z', pd.Series(0.0, index=df.index))
        if isinstance(dxy_z, (int, float)):
            dxy_z = pd.Series(dxy_z, index=df.index)

        # Combined crisis score
        # VIX > 1.0 = elevated fear
        # DXY > 0.8 = dollar strength (risk-off)
        vix_component = (vix_z / 2.0).clip(0.0, 1.0)  # Map [0, 2] to [0, 1]
        dxy_component = (dxy_z / 2.0).clip(0.0, 1.0)  # Map [0, 2] to [0, 1]

        # Weighted average (VIX more important for crypto)
        crisis_score = 0.6 * vix_component + 0.4 * dxy_component

        return pd.Series(crisis_score, index=df.index, name='crisis_context')

    def _compute_liquidity_vacuum_fusion(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Liquidity Vacuum fusion score - weighted combination of all signals.

        Fusion weights (empirically tuned for BTC capitulation reversals):
        - liquidity_vacuum: 25% (orderbook drain - primary signal)
        - volume_panic: 20% (capitulation selling indicator)
        - wick_rejection: 20% (exhaustion signal)
        - funding_reversal: 15% (short squeeze fuel)
        - crisis_context: 10% (macro capitulation context)
        - oversold: 5% (RSI mean reversion)
        - volatility_spike: 3% (violent moves expected)
        - downtrend_confirm: 2% (context confirmation)

        Returns:
            Series of Liquidity Vacuum fusion scores [0, 1]
        """
        # Get REQUIRED component scores
        liquidity_vacuum = df.get('liquidity_vacuum_score', 0.5)
        volume_panic = df.get('volume_panic', 0.0)
        wick_lower = df.get('wick_lower_ratio', 0.0)

        # Normalize wick_lower (0.5 = extreme)
        wick_rejection = (wick_lower / 0.5).clip(0.0, 1.0)

        # Get OPTIONAL component scores (graceful degradation)
        # Use safe funding getter with fallback to raw funding rate
        funding_z = df.get('funding_Z', df.get('funding_rate', 0.0))
        if isinstance(funding_z, (int, float)):
            funding_z = pd.Series(funding_z, index=df.index)
        else:
            # If we got raw funding rate, normalize it (typical range: -0.1 to +0.1)
            if 'funding_Z' not in df.columns and 'funding_rate' in df.columns:
                logger.warning("[Liquidity Vacuum] funding_Z missing, using funding_rate (normalized)")
                funding_z = funding_z / 0.01  # Convert to rough z-score

        funding_reversal = pd.Series(
            np.where(funding_z < -0.5, 1.0, 0.5),
            index=df.index
        )

        crisis_context = df.get('crisis_context', 0.0)

        # RSI oversold
        rsi = df.get('rsi_14', 50)
        if isinstance(rsi, (int, float)):
            rsi = pd.Series(rsi, index=df.index)
        oversold = pd.Series(
            np.where(rsi < 30, 1.0, 1.0 - (rsi / 100)),
            index=df.index
        )

        # Volatility spike (ATR percentile)
        volatility_spike = df.get('atr_percentile', 0.5)
        if isinstance(volatility_spike, (int, float)):
            volatility_spike = pd.Series(volatility_spike, index=df.index)

        # Downtrend confirmation (4H external trend)
        tf4h_trend = df.get('tf4h_external_trend', 'neutral')
        if isinstance(tf4h_trend, str):
            downtrend_confirm = pd.Series(
                1.0 if tf4h_trend == 'down' else 0.3,
                index=df.index
            )
        else:
            downtrend_confirm = pd.Series(
                np.where(tf4h_trend == 'down', 1.0, 0.3),
                index=df.index
            )

        # Weighted fusion (sum of weights = 1.0)
        w_liquidity = 0.25
        w_volume = 0.20
        w_wick = 0.20
        w_funding = 0.15
        w_crisis = 0.10
        w_oversold = 0.05
        w_volatility = 0.03
        w_downtrend = 0.02

        fusion = (
            w_liquidity * liquidity_vacuum +
            w_volume * volume_panic +
            w_wick * wick_rejection +
            w_funding * funding_reversal +
            w_crisis * crisis_context +
            w_oversold * oversold +
            w_volatility * volatility_spike +
            w_downtrend * downtrend_confirm
        )

        return pd.Series(fusion, index=df.index, name='liquidity_vacuum_fusion')

    # ========================================================================
    # V2 FEATURES: Multi-bar capitulation encoding
    # ========================================================================

    def _compute_liquidity_drain_pct(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute liquidity drain percentage vs 7-day average (FIXES LIQUIDITY PARADOX).

        This is the KEY fix for S1 v2. Instead of absolute liquidity levels, we measure
        RELATIVE drain from recent average. This solves the problem where true capitulations
        (June 18, FTX) showed high absolute liquidity but were actually deep drains relative
        to their recent average.

        Example:
            - June 18 bottom: liquidity = 0.308 (looks normal)
            - But 7d average was 0.55 → drain = (0.308 - 0.55) / 0.55 = -44% DRAIN!
            - Now S1 can detect this as a true capitulation

        Returns:
            Series of liquidity drain percentages (negative = drain)
            - -0.30 = 30% below 7d average (moderate drain)
            - -0.50 = 50% below 7d average (severe drain)
            - +0.20 = 20% above 7d average (liquidity recovering)
        """
        if 'liquidity_score' not in df.columns:
            logger.warning("[S1 v2] No liquidity_score column, cannot compute drain_pct")
            return pd.Series(0.0, index=df.index, name='liquidity_drain_pct')

        liq = df['liquidity_score']

        # 7-day rolling average (168 hours)
        liq_7d_avg = liq.rolling(window=168, min_periods=24).mean()

        # Percentage change from average
        # Negative = drain (good for capitulation detection)
        # Positive = recovery (exit signal)
        drain_pct = (liq - liq_7d_avg) / liq_7d_avg.replace(0, np.nan)

        # Clip extreme values
        drain_pct = drain_pct.clip(-1.0, 1.0).fillna(0.0)

        return pd.Series(drain_pct, index=df.index, name='liquidity_drain_pct')

    def _compute_liquidity_velocity(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute liquidity velocity (rate of drain).

        Separates:
        - Quiet low liquidity (dead markets)
        - Active drain (capitulation in progress)

        Returns:
            Series of liquidity velocity (negative = draining)
            - -0.05 = rapidly draining
            - 0.00 = stable
            - +0.05 = rapidly recovering
        """
        if 'liquidity_score' not in df.columns:
            return pd.Series(0.0, index=df.index, name='liquidity_velocity')

        liq = df['liquidity_score']

        # Velocity = change over last 6 bars (6 hours)
        liq_6h_ago = liq.shift(6)
        velocity = (liq - liq_6h_ago) / liq_6h_ago.replace(0, np.nan)

        # Smooth with 3-bar rolling average to reduce noise
        velocity = velocity.rolling(window=3, min_periods=1).mean()

        velocity = velocity.clip(-1.0, 1.0).fillna(0.0)

        return pd.Series(velocity, index=df.index, name='liquidity_velocity')

    def _compute_liquidity_persistence(self, df: pd.DataFrame) -> pd.Series:
        """
        Count consecutive bars with liquidity drain (multi-bar confirmation).

        Single-bar outliers = noise
        Multi-bar sustained stress = true capitulation

        Returns:
            Series of persistence counts [0, N]
            - 0 = no drain
            - 3 = drained for 3 consecutive bars
            - 8 = drained for 8 bars (strong signal)
        """
        drain_threshold = -0.20  # 20% below 7d avg

        if 'liquidity_drain_pct' not in df.columns:
            # Need to compute it first
            drain_pct = self._compute_liquidity_drain_pct(df)
        else:
            drain_pct = df['liquidity_drain_pct']

        # Boolean series: True when draining
        is_draining = drain_pct < drain_threshold

        # Count consecutive True values
        # Reset counter on False
        persistence = is_draining.groupby((~is_draining).cumsum()).cumsum()
        persistence = persistence.where(is_draining, 0)

        return pd.Series(persistence, index=df.index, name='liquidity_persistence')

    def _compute_capitulation_depth(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute drawdown from recent high (filters micro-dips from macro capitulations).

        Not all liquidity drains are buys - some are mid-trend chop.
        Deep drawdowns (>15% from recent high) indicate true leg-downs worth buying.

        Returns:
            Series of drawdown percentages (negative = drawdown)
            - -0.10 = 10% below recent high (minor dip)
            - -0.25 = 25% below recent high (leg down)
            - -0.50 = 50% below recent high (capitulation)
        """
        if 'close' not in df.columns:
            return pd.Series(0.0, index=df.index, name='capitulation_depth')

        close = df['close']

        # Rolling 30-day high (720 hours)
        rolling_high = close.rolling(window=720, min_periods=168).max()

        # Drawdown from high
        drawdown = (close - rolling_high) / rolling_high

        # Clip (drawdowns are negative)
        drawdown = drawdown.clip(-1.0, 0.0).fillna(0.0)

        return pd.Series(drawdown, index=df.index, name='capitulation_depth')

    def _compute_crisis_composite(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute composite macro crisis score (weighted blend of stress indicators).

        Combines:
        - VIX z-score (fear/volatility)
        - Funding extreme (shorts trapped)
        - Realized volatility spike
        - Drawdown depth

        Returns:
            Series of crisis composite scores [0, 1]
            - 0.0 = calm conditions
            - 0.5 = moderate stress
            - 0.8+ = crisis environment
        """
        # Component 1: VIX (30% weight)
        vix_z = df.get('VIX_Z', 0.0)
        if isinstance(vix_z, (int, float)):
            vix_z = pd.Series(vix_z, index=df.index)
        vix_component = (vix_z / 2.0).clip(0.0, 1.0)

        # Component 2: Funding extreme (25% weight)
        # Safe fallback: funding_Z -> funding_rate (normalized)
        funding_z = df.get('funding_Z', df.get('funding_rate', 0.0))
        if isinstance(funding_z, (int, float)):
            funding_z = pd.Series(funding_z, index=df.index)
        else:
            # Normalize raw funding rate if needed
            if 'funding_Z' not in df.columns and 'funding_rate' in df.columns:
                funding_z = funding_z / 0.01  # Convert to rough z-score

        # Negative funding = bullish (shorts paying longs = potential squeeze)
        funding_component = (abs(funding_z.clip(-3.0, 0.0)) / 3.0)

        # Component 3: Realized volatility (20% weight)
        rv_20d = df.get('rv_20d', 0.0)
        if isinstance(rv_20d, (int, float)):
            rv_20d = pd.Series(rv_20d, index=df.index)
        # Convert to percentile (rough approximation)
        rv_percentile = rv_20d.rolling(window=720, min_periods=168).rank(pct=True)
        rv_component = rv_percentile.fillna(0.5)

        # Component 4: Drawdown depth (25% weight)
        if 'capitulation_depth' in df.columns:
            dd = df['capitulation_depth']
        else:
            dd = self._compute_capitulation_depth(df)
        # Map [-0.50, 0.0] to [1.0, 0.0]
        dd_component = (abs(dd.clip(-0.50, 0.0)) / 0.50)

        # Weighted composite
        crisis_composite = (
            0.30 * vix_component +
            0.25 * funding_component +
            0.20 * rv_component +
            0.25 * dd_component
        )

        crisis_composite = crisis_composite.clip(0.0, 1.0)

        return pd.Series(crisis_composite, index=df.index, name='crisis_composite')

    def _compute_volume_climax_last_3b(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute max volume panic in last 3 bars (multi-bar volume pattern).

        Allows pattern to see: "We had a massive volume spike 1-2 bars ago"
        instead of requiring volume spike on exact entry bar.

        Returns:
            Series of max volume panic scores in last 3 bars
        """
        if 'volume_panic' not in df.columns:
            # Compute it first
            vol_panic = self._compute_volume_panic(df)
        else:
            vol_panic = df['volume_panic']

        # Rolling max over last 3 bars
        climax_3b = vol_panic.rolling(window=3, min_periods=1).max()

        return pd.Series(climax_3b, index=df.index, name='volume_climax_last_3b')

    def _compute_wick_exhaustion_last_3b(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute max wick rejection in last 3 bars (multi-bar wick pattern).

        Allows pattern to see: "We had an extreme rejection wick 1-2 bars ago"
        instead of requiring wick on exact entry bar.

        Returns:
            Series of max wick lower ratios in last 3 bars
        """
        if 'wick_lower_ratio' not in df.columns:
            # Compute it first
            wick = self._compute_wick_lower_ratio(df)
        else:
            wick = df['wick_lower_ratio']

        # Rolling max over last 3 bars
        exhaustion_3b = wick.rolling(window=3, min_periods=1).max()

        return pd.Series(exhaustion_3b, index=df.index, name='wick_exhaustion_last_3b')

    def _log_enrichment_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about enriched features"""

        # Wick stats
        deep_wick = (df['wick_lower_ratio'] > 0.30).sum()
        extreme_wick = (df['wick_lower_ratio'] > 0.50).sum()

        # Liquidity stats
        low_liq = (df.get('liquidity_score', 1.0) < 0.15).sum()

        # Volume stats
        vol_panic = (df['volume_panic'] > 0.5).sum()

        # Crisis stats
        crisis = (df['crisis_context'] > 0.5).sum()

        # Fusion stats
        high_fusion = (df['liquidity_vacuum_fusion'] > 0.4).sum()
        extreme_fusion = (df['liquidity_vacuum_fusion'] > 0.6).sum()

        logger.info("[Liquidity Vacuum Runtime] Enrichment statistics:")
        logger.info(f"  - Deep lower wick (>0.30): {deep_wick} ({deep_wick/len(df)*100:.1f}%)")
        logger.info(f"  - Extreme lower wick (>0.50): {extreme_wick} ({extreme_wick/len(df)*100:.1f}%)")
        logger.info(f"  - Low liquidity (<0.15): {low_liq} ({low_liq/len(df)*100:.1f}%)")
        logger.info(f"  - Volume panic (>0.5): {vol_panic} ({vol_panic/len(df)*100:.1f}%)")
        logger.info(f"  - Crisis context (>0.5): {crisis} ({crisis/len(df)*100:.1f}%)")
        logger.info(f"  - High fusion (>0.4): {high_fusion} ({high_fusion/len(df)*100:.1f}%)")
        logger.info(f"  - Extreme fusion (>0.6): {extreme_fusion} ({extreme_fusion/len(df)*100:.1f}%)")


# ============================================================================
# Integration Helper
# ============================================================================

def apply_liquidity_vacuum_enrichment(
    df: pd.DataFrame,
    lookback: int = 24,
    volume_lookback: int = 24
) -> pd.DataFrame:
    """
    Convenience function to apply Liquidity Vacuum runtime enrichment to a dataframe.

    Usage:
        df_enriched = apply_liquidity_vacuum_enrichment(df)

    Args:
        df: Feature dataframe
        lookback: General lookback window (hours)
        volume_lookback: Lookback for volume calculations (hours)

    Returns:
        Enriched dataframe (modified in-place)
    """
    enricher = LiquidityVacuumRuntimeFeatures(
        lookback=lookback,
        volume_lookback=volume_lookback
    )
    return enricher.enrich_dataframe(df)


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == '__main__':
    """
    Test Liquidity Vacuum runtime enrichment on sample data.

    Usage:
        python3 engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py
    """
    print("="*80)
    print("LIQUIDITY VACUUM REVERSAL RUNTIME ENRICHMENT TEST")
    print("="*80)
    print("\nLoading 2022 feature data (bear regime with capitulation events)...")

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
        for feature in ['liquidity_score', 'volume_zscore', 'VIX_Z', 'DXY_Z', 'rsi_14']:
            available = "✓" if feature in df_bear.columns else "✗"
            print(f"  {available} {feature}")

        # Apply Liquidity Vacuum enrichment
        print("\nApplying Liquidity Vacuum runtime enrichment...")
        df_enriched = apply_liquidity_vacuum_enrichment(df_bear, lookback=24, volume_lookback=24)

        print("\nEnrichment complete!")
        print(f"New columns added: {[c for c in df_enriched.columns if 'vacuum' in c or 'wick_lower' in c or 'volume_panic' in c or 'crisis_context' in c]}")

        # Show distribution of Liquidity Vacuum fusion scores
        print("\n" + "="*80)
        print("LIQUIDITY VACUUM FUSION SCORE DISTRIBUTION")
        print("="*80)

        fusion_scores = df_enriched['liquidity_vacuum_fusion']
        percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9]

        print("\nPercentile analysis:")
        for p in percentiles:
            val = np.percentile(fusion_scores.dropna(), p)
            count_above = (fusion_scores > val).sum()
            print(f"  p{p:>5.1f}: {val:.4f}  ({count_above:4d} bars above this threshold)")

        # Find high-conviction Liquidity Vacuum signals
        print("\n" + "="*80)
        print("HIGH-CONVICTION LIQUIDITY VACUUM SIGNALS (Capitulation Reversals)")
        print("="*80)

        high_fusion_threshold = np.percentile(fusion_scores.dropna(), 99.0)
        high_signals = df_enriched[df_enriched['liquidity_vacuum_fusion'] > high_fusion_threshold]

        print(f"\nFound {len(high_signals)} high-conviction signals (>p99 = {high_fusion_threshold:.4f})")

        if len(high_signals) > 0:
            print("\nTop 10 Liquidity Vacuum signals:")
            top_signals = high_signals.nlargest(10, 'liquidity_vacuum_fusion')
            for idx, row in top_signals.iterrows():
                print(f"  {idx}: Fusion={row['liquidity_vacuum_fusion']:.4f}, "
                      f"Wick={row['wick_lower_ratio']:.3f}, "
                      f"Liq={row.get('liquidity_score', 0):.3f}, "
                      f"Vol_Panic={row['volume_panic']:.3f}, "
                      f"Crisis={row['crisis_context']:.3f}")

        # Check for historical capitulation events
        print("\n" + "="*80)
        print("HISTORICAL CAPITULATION EVENTS CHECK")
        print("="*80)

        # Known capitulation dates in 2022
        capitulation_dates = [
            ('2022-05-12', 'LUNA death spiral'),
            ('2022-06-18', 'LUNA capitulation bottom'),
            ('2022-11-09', 'FTX collapse')
        ]

        for date_str, event_name in capitulation_dates:
            try:
                event_data = df_enriched.loc[date_str]
                if isinstance(event_data, pd.DataFrame):
                    # Multiple rows for this date, take first
                    event_data = event_data.iloc[0]

                print(f"\n{event_name} ({date_str}):")
                print(f"  Fusion Score: {event_data['liquidity_vacuum_fusion']:.4f}")
                print(f"  Wick Lower: {event_data['wick_lower_ratio']:.3f}")
                print(f"  Liquidity: {event_data.get('liquidity_score', 0):.3f}")
                print(f"  Volume Panic: {event_data['volume_panic']:.3f}")
                print(f"  Crisis Context: {event_data['crisis_context']:.3f}")
            except KeyError:
                print(f"\n{event_name} ({date_str}): Data not available")

        # Expected trade count estimation
        print("\n" + "="*80)
        print("TRADE COUNT ESTIMATION (2022)")
        print("="*80)

        for threshold_pct in [95, 97, 98, 99, 99.5]:
            threshold_val = np.percentile(fusion_scores.dropna(), threshold_pct)
            trades_above = (fusion_scores > threshold_val).sum()

            print(f"  Threshold p{threshold_pct:>5.1f} ({threshold_val:.4f}): "
                  f"{trades_above:3d} signals → {trades_above:.1f} trades/year")

        print("\n" + "="*80)
        print("Recommended Optuna search ranges:")
        print(f"  fusion_threshold: [{np.percentile(fusion_scores.dropna(), 95):.4f}, "
              f"{np.percentile(fusion_scores.dropna(), 99):.4f}]")
        print("  liquidity_max: [0.10, 0.20]")
        print("  volume_z_min: [1.5, 2.5]")
        print("  wick_lower_min: [0.25, 0.40]")
        print("="*80)

    except FileNotFoundError as e:
        print(f"ERROR: Feature file not found: {e}")
        print("\nExpected file: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")
        print("Run this from project root: python3 engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
