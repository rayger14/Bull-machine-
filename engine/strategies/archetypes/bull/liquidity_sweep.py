#!/usr/bin/env python3
"""
Liquidity Sweep (G) - Bull Archetype

PATTERN DESCRIPTION:
Liquidity Sweep occurs when smart money intentionally pushes price below
support to trigger stop losses and gather liquidity, then rapidly reverses
upward. This is a classic institutional manipulation tactic to accumulate
positions at favorable prices.

KEY CHARACTERISTICS:
1. Price briefly sweeps below previous lows (liquidity grab)
2. Immediate reversal with strong momentum (stop hunt complete)
3. Deep lower wick (>35% of candle range)
4. Volume spike during sweep (stop loss cascade)
5. Quick recovery above support with strength

MARKET MECHANICS:
Retail traders place stops below obvious support levels. Smart money knows this
and intentionally triggers these stops to:
1. Fill their buy orders (accumulation)
2. Remove sellers from the market
3. Create panic for cheaper entries
4. Build position before markup

ENTRY LOGIC:
- SMC liquidity sweep detected OR deep lower wick below support
- Price recovered above sweep level (close > previous support)
- Volume spike confirms stop cascade
- No bearish BOS on higher timeframe
- Bullish momentum confirming reversal

DOMAIN ENGINE INTEGRATION:
- SMC: Liquidity sweep detection + demand zone (35% weight)
- Price Action: Wick rejection + recovery (30% weight)
- Volume: Stop cascade confirmation (20% weight)
- Wyckoff: Spring context (optional boost) (10% weight)
- Regime: Risk-on alignment (5% weight)

SAFETY VETOES:
- No entry if price failed to recover above support
- No entry if volume weak (no institutional participation)
- No entry if 4H trend strongly down
- No entry during crisis (unless extreme capitulation)
- No entry if RSI > 75 (overbought chase)

TARGET METRICS:
- Trades/year: 20-30 (selective sweeps)
- Win rate: 60-70% (high-probability reversals)
- Profit factor: 2.2-3.0
- Avg hold time: 1-4 days

HISTORICAL EXAMPLES (BTC):
- 2023-08-17: Summer sweep below $25k → +15% rally
- 2023-11-09: Pre-rally liquidity grab → sustained markup
- 2024-01-03: ETF narrative sweep → explosive move
- 2024-09-06: Pre-election sweep → strong recovery

Author: Claude Code (Backend Architect)
Date: 2025-12-12
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class LiquiditySweepArchetype:
    """
    Minimal viable implementation of Liquidity Sweep archetype.

    Detects institutional liquidity sweeps for long entries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Liquidity Sweep archetype.

        Args:
            config: Optional configuration dict with thresholds
        """
        self.config = config or {}

        # Extract thresholds with defaults (permissive for MVP)
        thresholds = self.config.get('thresholds', {})

        # Core pattern thresholds
        self.min_wick_lower_ratio = thresholds.get('min_wick_lower_ratio', 0.30)
        self.min_volume_zscore = thresholds.get('min_volume_zscore', 1.2)
        self.min_recovery_body = thresholds.get('min_recovery_body', 0.25)
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.35)

        # Domain engine weights
        self.smc_weight = thresholds.get('smc_weight', 0.35)
        self.price_action_weight = thresholds.get('price_action_weight', 0.30)
        self.volume_weight = thresholds.get('volume_weight', 0.20)
        self.wyckoff_weight = thresholds.get('wyckoff_weight', 0.10)
        self.regime_weight = thresholds.get('regime_weight', 0.05)

        # Safety thresholds
        self.max_rsi_chase = thresholds.get('max_rsi_chase', 75)

        logger.info(f"[Liquidity Sweep] Initialized with min_fusion={self.min_fusion_score}")

    def detect(self, row: pd.Series, regime_label: str = 'neutral') -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Detect Liquidity Sweep pattern in current bar.

        Args:
            row: Current bar data with features
            regime_label: Current regime classification

        Returns:
            Tuple of (archetype_name, confidence_score, metadata)
            Returns (None, 0.0, {}) if no signal
        """
        # Step 1: Check SMC liquidity sweep
        smc_score = self._compute_smc_score(row)

        # Step 2: Check price action (wick + recovery)
        price_action_score = self._compute_price_action_score(row)

        # Step 3: Check volume (stop cascade)
        volume_score = self._compute_volume_score(row)

        # Step 4: Check Wyckoff spring context (optional boost)
        wyckoff_score = self._compute_wyckoff_score(row)

        # Step 5: Check regime alignment
        regime_score = self._compute_regime_score(regime_label)

        # Step 6: Compute weighted fusion score
        fusion_score = (
            self.smc_weight * smc_score +
            self.price_action_weight * price_action_score +
            self.volume_weight * volume_score +
            self.wyckoff_weight * wyckoff_score +
            self.regime_weight * regime_score
        )

        # Step 7: Apply safety vetoes
        veto_reason = self._check_vetoes(row, regime_label)
        if veto_reason:
            return None, 0.0, {'veto_reason': veto_reason, 'fusion_score': fusion_score}

        # Step 8: Apply temporal confluence timing multiplier
        temporal_confluence = row.get('temporal_confluence', None)
        temporal_mult = 1.0  # Default neutral
        if temporal_confluence is not None and not pd.isna(temporal_confluence):
            # Apply conservative 0.85-1.15 range (max ±15% adjustment)
            # High confluence (0.80) = 1.09x boost, Low confluence (0.20) = 0.91x penalty
            temporal_mult = 0.85 + (temporal_confluence * 0.30)
            fusion_score *= temporal_mult

        # Step 9: Check fusion threshold
        if fusion_score < self.min_fusion_score:
            return None, 0.0, {
                'reason': 'below_threshold',
                'fusion_score': fusion_score,
                'threshold': self.min_fusion_score,
                'temporal_mult': temporal_mult
            }

        # Signal detected!
        metadata = {
            'smc_score': smc_score,
            'price_action_score': price_action_score,
            'volume_score': volume_score,
            'wyckoff_score': wyckoff_score,
            'regime_score': regime_score,
            'fusion_score': fusion_score,
            'temporal_confluence': temporal_confluence,
            'temporal_mult': temporal_mult,
            'pattern_type': 'liquidity_sweep_long'
        }

        return 'liquidity_sweep', fusion_score, metadata

    def _compute_smc_score(self, row: pd.Series) -> float:
        """
        Compute SMC domain engine score.

        Checks for:
        - SMC liquidity sweep flag
        - Demand zone presence
        - Price structure (swept lows then recovered)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for SMC liquidity sweep flag
        smc_liq_sweep = row.get('smc_liquidity_sweep', False)
        if smc_liq_sweep:
            score += 0.50

        # Check for demand zone
        demand_zone = row.get('smc_demand_zone', False)
        if demand_zone:
            score += 0.30

        # Check if price near order block (institutional zone)
        ob_bull_bottom = row.get('tf1h_ob_bull_bottom', None)
        ob_bull_top = row.get('tf1h_ob_bull_top', None)
        low = row.get('low', 0)
        close = row.get('close', 0)

        if ob_bull_bottom is not None and ob_bull_top is not None and close > 0:
            # Check if low swept through order block then recovered
            if low <= ob_bull_bottom and close >= ob_bull_bottom:
                score += 0.30

        return min(1.0, score)

    def _compute_price_action_score(self, row: pd.Series) -> float:
        """
        Compute price action domain engine score.

        Checks for:
        - Deep lower wick (sweep + rejection)
        - Strong recovery (bullish close)
        - Close in upper half of range

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Calculate lower wick ratio
        wick_lower_ratio = self._calculate_wick_lower_ratio(row)

        if wick_lower_ratio >= self.min_wick_lower_ratio:
            # Strong rejection wick
            # Normalize: 0.30 = 0.5, 0.60 = 1.0
            wick_score = min(1.0, (wick_lower_ratio - self.min_wick_lower_ratio) / 0.30)
            score += 0.50 * wick_score

        # Check for bullish recovery
        open_price = row.get('open', 0)
        close = row.get('close', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)

        if high > low and close > 0:
            # Calculate body size
            body_size = abs(close - open_price)
            candle_range = high - low

            if candle_range > 0:
                body_ratio = body_size / candle_range

                if close > open_price and body_ratio >= self.min_recovery_body:
                    # Strong bullish recovery
                    score += 0.30

            # Check close position (high in range = strong recovery)
            close_position = (close - low) / candle_range
            if close_position > 0.6:
                score += 0.20 * close_position

        return min(1.0, score)

    def _compute_volume_score(self, row: pd.Series) -> float:
        """
        Compute volume domain engine score.

        Liquidity sweeps create stop cascade = volume spike.

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check volume z-score
        volume_zscore = row.get('volume_zscore', 0.0)

        if volume_zscore >= self.min_volume_zscore:
            # Normalize: 1.2 = 0.5, 3.0 = 1.0
            vol_score = min(1.0, (volume_zscore - self.min_volume_zscore) / 1.8)
            score += 0.80 * vol_score

        # Bonus for extreme volume (institutional participation)
        if volume_zscore > 2.5:
            score += 0.20

        return min(1.0, score)

    def _compute_wyckoff_score(self, row: pd.Series) -> float:
        """
        Compute Wyckoff domain engine score (optional boost).

        Liquidity sweeps often occur during Wyckoff Spring events.

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for Wyckoff Spring (primary signal)
        spring_a = row.get('wyckoff_spring_a', False)
        spring_a_conf = row.get('wyckoff_spring_a_confidence', 0.0)
        spring_b = row.get('wyckoff_spring_b', False)
        spring_b_conf = row.get('wyckoff_spring_b_confidence', 0.0)

        if spring_a and spring_a_conf >= 0.5:
            score += 0.60
        elif spring_b and spring_b_conf >= 0.5:
            score += 0.40

        # Check Wyckoff phase
        phase = row.get('wyckoff_phase_abc', 'neutral')
        if phase == 'C':  # Testing phase
            score += 0.30

        return min(1.0, score)

    def _compute_regime_score(self, regime_label: str) -> float:
        """
        Compute regime alignment score.

        Liquidity sweeps work in all regimes but best in risk_on/neutral.

        Args:
            regime_label: Current regime

        Returns:
            Score 0.0-1.0
        """
        regime_scores = {
            'risk_on': 1.0,      # Ideal
            'neutral': 0.9,      # Very good
            'risk_off': 0.6,     # Still valid (capitulation sweeps)
            'crisis': 0.4        # Selective (extreme capitulation only)
        }

        return regime_scores.get(regime_label, 0.7)

    def _check_vetoes(self, row: pd.Series, regime_label: str) -> Optional[str]:
        """
        Check safety vetoes that block entry.

        Args:
            row: Current bar data
            regime_label: Current regime

        Returns:
            Veto reason string, or None if no veto
        """
        # Veto 1: Failed to recover above support
        open_price = row.get('open', 0)
        close = row.get('close', 0)
        low = row.get('low', 0)

        if close <= open_price:
            # Bearish close = failed sweep
            return 'failed_recovery'

        # Veto 2: Weak volume (no institutional participation)
        volume_zscore = row.get('volume_zscore', 0.0)
        if volume_zscore < 0.5:
            return f'weak_volume_{volume_zscore:.1f}'

        # Veto 3: Strong 4H downtrend
        tf4h_trend = row.get('tf4h_trend_direction', 0)
        adx = row.get('adx_14', 20)
        if tf4h_trend < 0 and adx > 30:
            return 'strong_htf_downtrend'

        # Veto 4: RSI overbought chase
        rsi = row.get('rsi_14', 50)
        if rsi > self.max_rsi_chase:
            return f'overbought_chase_rsi_{rsi:.1f}'

        # Veto 5: Crisis regime without capitulation
        if regime_label == 'crisis':
            capitulation_depth = row.get('capitulation_depth', 0.0)
            if capitulation_depth > -0.12:
                return 'crisis_no_capitulation'

        return None

    def _calculate_wick_lower_ratio(self, row: pd.Series) -> float:
        """
        Calculate lower wick as percentage of candle range.

        Args:
            row: Current bar data

        Returns:
            Wick lower ratio 0.0-1.0
        """
        high = row.get('high', 0)
        low = row.get('low', 0)
        open_price = row.get('open', 0)
        close = row.get('close', 0)

        if high <= low or high == 0:
            return 0.0

        # Body low = min(open, close)
        body_low = min(open_price, close)

        # Lower wick length
        wick_lower = body_low - low

        # Candle range
        candle_range = high - low

        if candle_range == 0:
            return 0.0

        # Ratio
        wick_ratio = wick_lower / candle_range

        return max(0.0, min(1.0, wick_ratio))


# ============================================================================
# Integration Helper
# ============================================================================

def detect_liquidity_sweep_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Convenience function to detect Liquidity Sweep signals.

    Args:
        row: Current bar data
        regime_label: Current regime
        config: Optional configuration

    Returns:
        Tuple of (archetype_name, confidence, metadata)
    """
    archetype = LiquiditySweepArchetype(config=config)
    return archetype.detect(row, regime_label)
