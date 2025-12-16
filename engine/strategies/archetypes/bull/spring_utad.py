#!/usr/bin/env python3
"""
Spring/UTAD (A) - Bull Archetype

PATTERN DESCRIPTION:
Spring: Wyckoff accumulation pattern where price briefly breaks below support
(shaking out weak hands), then rapidly reverses upward as smart money accumulates.
UTAD (Upthrust After Distribution): The bearish inverse - fake breakout above
resistance that traps longs before reversal down. For bull archetype, we detect
Springs (long setups).

KEY CHARACTERISTICS:
1. Brief violation of support/range low (stop hunt)
2. Rapid recovery with strong volume (smart money buying)
3. Wyckoff Phase C or D context (testing phase)
4. Price closes back inside range with momentum
5. Often accompanied by negative funding (shorts trapped)

ENTRY LOGIC:
- Detect Wyckoff Spring event (wyckoff_spring_a or wyckoff_spring_b)
- Price broke below support then recovered (lower wick > 30%)
- Volume spike during recovery (volume_zscore > 1.5)
- Wyckoff phase = C or D (testing/last point phases)
- Close back above support level

DOMAIN ENGINE INTEGRATION:
- Wyckoff: Spring event detection + phase confirmation (30% weight)
- SMC: Demand zone + liquidity sweep detection (25% weight)
- Price Action: Lower wick rejection + volume confirmation (25% weight)
- Momentum: RSI oversold bounce + ADX trend (15% weight)
- Regime: Prefer risk_on/neutral regimes (5% weight)

SAFETY VETOES:
- No entry if in strong downtrend (tf4h_trend = 'down' AND adx > 25)
- No entry if RSI already > 70 (overbought, missed the move)
- No entry if negative divergence present
- No entry during crisis regime (unless extreme capitulation)

TARGET METRICS:
- Trades/year: 15-25 (selective spring setups)
- Win rate: 55-65% (mean-reversion edge)
- Profit factor: 1.8-2.5
- Avg hold time: 2-5 days

HISTORICAL EXAMPLES (BTC):
- 2023-03-10: Banking crisis spring → +25% bounce
- 2023-06-15: Summer range spring → +15% markup
- 2023-09-11: Wyckoff spring → sustained markup phase
- 2024-01-23: Pre-ETF spring → explosive rally

Author: Claude Code (Backend Architect)
Date: 2025-12-12
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class SpringUTADArchetype:
    """
    Minimal viable implementation of Spring/UTAD archetype.

    Detects Wyckoff Spring patterns for long entries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Spring/UTAD archetype.

        Args:
            config: Optional configuration dict with thresholds
        """
        self.config = config or {}

        # Extract thresholds with defaults (permissive for MVP)
        thresholds = self.config.get('thresholds', {})

        # Core pattern thresholds
        self.min_wyckoff_confidence = thresholds.get('min_wyckoff_confidence', 0.50)
        self.min_wick_lower_ratio = thresholds.get('min_wick_lower_ratio', 0.25)
        self.min_volume_zscore = thresholds.get('min_volume_zscore', 1.0)
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.35)

        # Domain engine weights
        self.wyckoff_weight = thresholds.get('wyckoff_weight', 0.30)
        self.smc_weight = thresholds.get('smc_weight', 0.25)
        self.price_action_weight = thresholds.get('price_action_weight', 0.25)
        self.momentum_weight = thresholds.get('momentum_weight', 0.15)
        self.regime_weight = thresholds.get('regime_weight', 0.05)

        # Safety thresholds
        self.max_rsi_entry = thresholds.get('max_rsi_entry', 70)
        self.max_adx_downtrend = thresholds.get('max_adx_downtrend', 25)

        logger.info(f"[Spring/UTAD] Initialized with min_fusion={self.min_fusion_score}")

    def detect(self, row: pd.Series, regime_label: str = 'neutral') -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Detect Spring/UTAD pattern in current bar.

        Args:
            row: Current bar data with features
            regime_label: Current regime classification

        Returns:
            Tuple of (archetype_name, confidence_score, metadata)
            Returns (None, 0.0, {}) if no signal
        """
        # Step 1: Check Wyckoff Spring events
        wyckoff_score = self._compute_wyckoff_score(row)

        # Step 2: Check SMC demand zone and liquidity sweep
        smc_score = self._compute_smc_score(row)

        # Step 3: Check price action (wick rejection + volume)
        price_action_score = self._compute_price_action_score(row)

        # Step 4: Check momentum (RSI + ADX)
        momentum_score = self._compute_momentum_score(row)

        # Step 5: Check regime alignment
        regime_score = self._compute_regime_score(regime_label)

        # Step 6: Compute weighted fusion score
        fusion_score = (
            self.wyckoff_weight * wyckoff_score +
            self.smc_weight * smc_score +
            self.price_action_weight * price_action_score +
            self.momentum_weight * momentum_score +
            self.regime_weight * regime_score
        )

        # Step 7: Apply safety vetoes
        veto_reason = self._check_vetoes(row, regime_label)
        if veto_reason:
            return None, 0.0, {'veto_reason': veto_reason, 'fusion_score': fusion_score}

        # Step 8: Check fusion threshold
        if fusion_score < self.min_fusion_score:
            return None, 0.0, {
                'reason': 'below_threshold',
                'fusion_score': fusion_score,
                'threshold': self.min_fusion_score
            }

        # Signal detected!
        metadata = {
            'wyckoff_score': wyckoff_score,
            'smc_score': smc_score,
            'price_action_score': price_action_score,
            'momentum_score': momentum_score,
            'regime_score': regime_score,
            'fusion_score': fusion_score,
            'pattern_type': 'spring_long'
        }

        return 'spring_utad', fusion_score, metadata

    def _compute_wyckoff_score(self, row: pd.Series) -> float:
        """
        Compute Wyckoff domain engine score.

        Checks for:
        - Spring Type A or Type B events
        - Phase C or D context
        - LPS (Last Point of Support) events

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for Spring events (primary signal)
        spring_a = row.get('wyckoff_spring_a', False)
        spring_a_conf = row.get('wyckoff_spring_a_confidence', 0.0)
        spring_b = row.get('wyckoff_spring_b', False)
        spring_b_conf = row.get('wyckoff_spring_b_confidence', 0.0)

        if spring_a and spring_a_conf >= self.min_wyckoff_confidence:
            score += 0.50  # Strong spring signal
        elif spring_b and spring_b_conf >= self.min_wyckoff_confidence:
            score += 0.40  # Moderate spring signal

        # Check for LPS (Last Point of Support) - bullish
        lps = row.get('wyckoff_lps', False)
        lps_conf = row.get('wyckoff_lps_confidence', 0.0)
        if lps and lps_conf >= self.min_wyckoff_confidence:
            score += 0.30

        # Check Wyckoff phase context
        phase = row.get('wyckoff_phase_abc', 'neutral')
        if phase in ['C', 'D']:  # Testing/Last Point phases
            score += 0.20

        return min(1.0, score)

    def _compute_smc_score(self, row: pd.Series) -> float:
        """
        Compute SMC (Smart Money Concepts) domain engine score.

        Checks for:
        - Demand zone presence
        - Liquidity sweep detection (stop hunt)
        - Order block retest

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for demand zone
        demand_zone = row.get('smc_demand_zone', False)
        if demand_zone:
            score += 0.40

        # Check for liquidity sweep (stop hunt below lows)
        liq_sweep = row.get('smc_liquidity_sweep', False)
        if liq_sweep:
            score += 0.40

        # Check if price near bullish order block
        ob_bull_bottom = row.get('tf1h_ob_bull_bottom', None)
        ob_bull_top = row.get('tf1h_ob_bull_top', None)
        close = row.get('close', 0)

        if ob_bull_bottom is not None and ob_bull_top is not None and close > 0:
            # Check if price is within or just above order block
            if ob_bull_bottom <= close <= ob_bull_top * 1.02:  # 2% tolerance
                score += 0.20

        return min(1.0, score)

    def _compute_price_action_score(self, row: pd.Series) -> float:
        """
        Compute price action domain engine score.

        Checks for:
        - Deep lower wick (rejection)
        - Volume spike during recovery
        - Bullish close relative to range

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Calculate lower wick ratio
        wick_lower_ratio = self._calculate_wick_lower_ratio(row)

        if wick_lower_ratio >= self.min_wick_lower_ratio:
            # Normalize: 0.25 = 0.5, 0.50 = 1.0
            wick_score = min(1.0, (wick_lower_ratio - self.min_wick_lower_ratio) / 0.25)
            score += 0.50 * wick_score

        # Check volume spike
        volume_zscore = row.get('volume_zscore', 0.0)
        if volume_zscore >= self.min_volume_zscore:
            # Normalize: 1.0 = 0.5, 3.0 = 1.0
            vol_score = min(1.0, (volume_zscore - self.min_volume_zscore) / 2.0)
            score += 0.30 * vol_score

        # Check bullish close position
        open_price = row.get('open', 0)
        close = row.get('close', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)

        if high > low and close > 0:
            # Close in upper half of range = bullish
            range_position = (close - low) / (high - low)
            if range_position > 0.5:
                score += 0.20 * range_position

        return min(1.0, score)

    def _compute_momentum_score(self, row: pd.Series) -> float:
        """
        Compute momentum domain engine score.

        Checks for:
        - RSI oversold bounce (30-50 range ideal)
        - ADX trend strength
        - MACD turning positive

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # RSI check - prefer oversold bounce
        rsi = row.get('rsi_14', 50)
        if 25 <= rsi <= 50:  # Oversold to neutral = ideal spring setup
            # Map [25, 50] to [1.0, 0.5]
            rsi_score = 1.0 - ((rsi - 25) / 25) * 0.5
            score += 0.50 * rsi_score
        elif rsi > 50:
            # Already bouncing but not overbought
            score += 0.20

        # ADX check - prefer moderate trend strength
        adx = row.get('adx_14', 20)
        if 15 <= adx <= 30:  # Moderate trend forming
            score += 0.30
        elif adx > 30:  # Strong trend already
            score += 0.10

        # MACD check - prefer bullish cross
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        if macd > macd_signal:  # Bullish momentum
            score += 0.20

        return min(1.0, score)

    def _compute_regime_score(self, regime_label: str) -> float:
        """
        Compute regime alignment score.

        Spring patterns work best in risk_on and neutral regimes.
        During crisis, require extreme conditions.

        Args:
            regime_label: Current regime

        Returns:
            Score 0.0-1.0
        """
        regime_scores = {
            'risk_on': 1.0,      # Ideal for spring reversals
            'neutral': 0.8,      # Still good
            'risk_off': 0.4,     # Risky but possible
            'crisis': 0.2        # Very selective
        }

        return regime_scores.get(regime_label, 0.5)

    def _check_vetoes(self, row: pd.Series, regime_label: str) -> Optional[str]:
        """
        Check safety vetoes that block entry.

        Args:
            row: Current bar data
            regime_label: Current regime

        Returns:
            Veto reason string, or None if no veto
        """
        # Veto 1: Overbought RSI (missed the move)
        rsi = row.get('rsi_14', 50)
        if rsi > self.max_rsi_entry:
            return f'rsi_overbought_{rsi:.1f}'

        # Veto 2: Strong downtrend on higher timeframe
        tf4h_trend = row.get('tf4h_trend_direction', 0)
        adx = row.get('adx_14', 20)
        if tf4h_trend < 0 and adx > self.max_adx_downtrend:
            return f'strong_downtrend_adx_{adx:.1f}'

        # Veto 3: Bearish divergence present
        if row.get('bearish_divergence_detected', False):
            return 'bearish_divergence'

        # Veto 4: Crisis regime without extreme capitulation
        if regime_label == 'crisis':
            capitulation_depth = row.get('capitulation_depth', 0.0)
            if capitulation_depth > -0.15:  # Not deep enough drawdown
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

def detect_spring_utad_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Convenience function to detect Spring/UTAD signals.

    Args:
        row: Current bar data
        regime_label: Current regime
        config: Optional configuration

    Returns:
        Tuple of (archetype_name, confidence, metadata)
    """
    archetype = SpringUTADArchetype(config=config)
    return archetype.detect(row, regime_label)
