#!/usr/bin/env python3
"""
BOS/CHOCH Reversal (C) - Bull Archetype

PATTERN DESCRIPTION:
BOS (Break of Structure): When price breaks a significant high, confirming
trend continuation or reversal. CHOCH (Change of Character): When price behavior
shifts from one trend to another, signaling potential reversal.

This archetype detects bullish BOS/CHOCH patterns - moments when price breaks
above previous highs with momentum, confirming bullish trend shift.

KEY CHARACTERISTICS:
1. Bullish BOS flag triggered (tf1h_bos_bullish or tf4h_bos_bullish)
2. Strong momentum (ADX > 20, RSI 50-70)
3. Volume confirmation (volume_zscore > 1.0)
4. Higher timeframe alignment (4H trend bullish)
5. No immediate resistance overhead

SMC THEORY:
BOS represents smart money leaving footprints - institutional accumulation
followed by decisive breakout. CHOCH shows character shift from distribution
to accumulation. These are high-conviction continuation signals.

ENTRY LOGIC:
- BOS flag triggered on 1H or 4H timeframe
- Price closed above previous structure high
- Momentum indicators confirm strength
- Volume shows institutional participation
- No bearish divergence

DOMAIN ENGINE INTEGRATION:
- SMC: BOS/CHOCH detection + structure break (40% weight)
- Momentum: ADX + RSI + MACD confirmation (30% weight)
- Volume: Breakout volume validation (20% weight)
- Regime: Risk-on alignment (10% weight)

SAFETY VETOES:
- No entry if RSI > 80 (extreme overbought)
- No entry if bearish divergence present
- No entry if 4H trend is down (counter-trend)
- No entry during crisis regime
- No entry if volume declining on breakout (fake breakout)

TARGET METRICS:
- Trades/year: 25-40 (frequent in trending markets)
- Win rate: 65-75% (strong momentum edge)
- Profit factor: 2.5-3.5
- Avg hold time: 2-7 days

HISTORICAL EXAMPLES (BTC):
- 2023-10-24: BOS above $35k → sustained rally
- 2024-02-28: CHOCH at $60k → new ATH run
- 2024-10-14: BOS breakout → pre-election rally
- 2024-11-06: CHOCH confirmation → explosive move

Author: Claude Code (Backend Architect)
Date: 2025-12-12
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class BOSCHOCHReversalArchetype:
    """
    Minimal viable implementation of BOS/CHOCH Reversal archetype.

    Detects bullish break of structure and change of character patterns.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BOS/CHOCH Reversal archetype.

        Args:
            config: Optional configuration dict with thresholds
        """
        self.config = config or {}

        # Extract thresholds with defaults (permissive for MVP)
        thresholds = self.config.get('thresholds', {})

        # Core pattern thresholds
        self.min_adx = thresholds.get('min_adx', 18)
        self.min_rsi = thresholds.get('min_rsi', 45)
        self.max_rsi = thresholds.get('max_rsi', 80)
        self.min_volume_zscore = thresholds.get('min_volume_zscore', 0.8)
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.35)

        # Domain engine weights
        self.smc_weight = thresholds.get('smc_weight', 0.40)
        self.momentum_weight = thresholds.get('momentum_weight', 0.30)
        self.volume_weight = thresholds.get('volume_weight', 0.20)
        self.regime_weight = thresholds.get('regime_weight', 0.10)

        # Safety thresholds
        self.require_htf_alignment = thresholds.get('require_htf_alignment', True)

        logger.info(f"[BOS/CHOCH Reversal] Initialized with min_fusion={self.min_fusion_score}")

    def detect(self, row: pd.Series, regime_label: str = 'neutral') -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Detect BOS/CHOCH Reversal pattern in current bar.

        Args:
            row: Current bar data with features
            regime_label: Current regime classification

        Returns:
            Tuple of (archetype_name, confidence_score, metadata)
            Returns (None, 0.0, {}) if no signal
        """
        # Step 1: Check SMC BOS/CHOCH signals
        smc_score = self._compute_smc_score(row)

        # Step 2: Check momentum confirmation
        momentum_score = self._compute_momentum_score(row)

        # Step 3: Check volume confirmation
        volume_score = self._compute_volume_score(row)

        # Step 4: Check regime alignment
        regime_score = self._compute_regime_score(regime_label)

        # Step 5: Compute weighted fusion score
        fusion_score = (
            self.smc_weight * smc_score +
            self.momentum_weight * momentum_score +
            self.volume_weight * volume_score +
            self.regime_weight * regime_score
        )

        # Step 6: Apply safety vetoes
        veto_reason = self._check_vetoes(row, regime_label)
        if veto_reason:
            return None, 0.0, {'veto_reason': veto_reason, 'fusion_score': fusion_score}

        # Step 7: Check fusion threshold
        if fusion_score < self.min_fusion_score:
            return None, 0.0, {
                'reason': 'below_threshold',
                'fusion_score': fusion_score,
                'threshold': self.min_fusion_score
            }

        # Signal detected!
        metadata = {
            'smc_score': smc_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'regime_score': regime_score,
            'fusion_score': fusion_score,
            'pattern_type': 'bos_choch_long'
        }

        return 'bos_choch_reversal', fusion_score, metadata

    def _compute_smc_score(self, row: pd.Series) -> float:
        """
        Compute SMC domain engine score.

        Checks for:
        - Bullish BOS on 1H or 4H
        - CHOCH signal (change of character)
        - Structure break confirmation

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for bullish BOS on 1H
        tf1h_bos_bull = row.get('tf1h_bos_bullish', False)
        if tf1h_bos_bull:
            score += 0.40

        # Check for bullish BOS on 4H (stronger signal)
        tf4h_bos_bull = row.get('tf4h_bos_bullish', False)
        if tf4h_bos_bull:
            score += 0.50

        # Check for CHOCH (if available)
        smc_choch = row.get('smc_choch', False)
        if smc_choch:
            score += 0.30

        # Check 4H fusion score as structure quality indicator
        tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
        if tf4h_fusion > 0.5:
            score += 0.20 * (tf4h_fusion - 0.5) / 0.5

        return min(1.0, score)

    def _compute_momentum_score(self, row: pd.Series) -> float:
        """
        Compute momentum domain engine score.

        Checks for:
        - ADX trend strength
        - RSI in sweet spot (50-70)
        - MACD bullish

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # ADX check - trend strength
        adx = row.get('adx_14', 20)
        if adx >= self.min_adx:
            # Normalize: 18 = 0.5, 35+ = 1.0
            adx_score = min(1.0, (adx - self.min_adx) / 17)
            score += 0.40 * adx_score

        # RSI check - momentum sweet spot
        rsi = row.get('rsi_14', 50)
        if self.min_rsi <= rsi <= 70:
            # Ideal range for continuation
            # Peak score at RSI 60
            if rsi <= 60:
                rsi_score = (rsi - self.min_rsi) / 15
            else:
                rsi_score = 1.0 - ((rsi - 60) / 10) * 0.3
            score += 0.30 * rsi_score
        elif 70 < rsi < self.max_rsi:
            # Still acceptable but overbought
            score += 0.10

        # MACD check - bullish momentum
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        macd_hist = row.get('macd_hist', 0)

        if macd > macd_signal and macd_hist > 0:
            # Strong bullish momentum
            score += 0.30
        elif macd > macd_signal:
            # Moderate bullish momentum
            score += 0.15

        return min(1.0, score)

    def _compute_volume_score(self, row: pd.Series) -> float:
        """
        Compute volume domain engine score.

        Breakouts need volume confirmation - institutional participation.

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check volume z-score
        volume_zscore = row.get('volume_zscore', 0.0)

        if volume_zscore >= self.min_volume_zscore:
            # Normalize: 0.8 = 0.5, 2.5 = 1.0
            vol_score = min(1.0, (volume_zscore - self.min_volume_zscore) / 1.7)
            score += 0.70 * vol_score

        # Check if volume is increasing (trend)
        # For MVP, use snapshot volume_zscore
        # Future: implement rolling volume trend

        # Penalize if volume is declining
        if volume_zscore < -0.5:
            score *= 0.5  # Cut score in half if volume weak

        # Bonus for extreme volume
        if volume_zscore > 2.5:
            score += 0.30

        return min(1.0, score)

    def _compute_regime_score(self, regime_label: str) -> float:
        """
        Compute regime alignment score.

        BOS/CHOCH patterns work best in risk_on regimes.

        Args:
            regime_label: Current regime

        Returns:
            Score 0.0-1.0
        """
        regime_scores = {
            'risk_on': 1.0,      # Ideal for breakouts
            'neutral': 0.7,      # Still good
            'risk_off': 0.3,     # Risky
            'crisis': 0.0        # Avoid
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
        # Veto 1: Extreme overbought RSI
        rsi = row.get('rsi_14', 50)
        if rsi > self.max_rsi:
            return f'extreme_overbought_rsi_{rsi:.1f}'

        # Veto 2: Bearish divergence
        if row.get('bearish_divergence_detected', False):
            return 'bearish_divergence'

        # Veto 3: 4H trend is down (counter-trend breakout = risky)
        if self.require_htf_alignment:
            tf4h_trend = row.get('tf4h_trend_direction', 0)
            if tf4h_trend < 0:
                return 'htf_downtrend'

        # Veto 4: Crisis regime
        if regime_label == 'crisis':
            return 'crisis_regime'

        # Veto 5: Volume declining on breakout (fake breakout)
        volume_zscore = row.get('volume_zscore', 0.0)
        tf1h_bos_bull = row.get('tf1h_bos_bullish', False)
        tf4h_bos_bull = row.get('tf4h_bos_bullish', False)

        if (tf1h_bos_bull or tf4h_bos_bull) and volume_zscore < -0.5:
            return f'weak_volume_breakout_{volume_zscore:.1f}'

        return None


# ============================================================================
# Integration Helper
# ============================================================================

def detect_bos_choch_reversal_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Convenience function to detect BOS/CHOCH Reversal signals.

    Args:
        row: Current bar data
        regime_label: Current regime
        config: Optional configuration

    Returns:
        Tuple of (archetype_name, confidence, metadata)
    """
    archetype = BOSCHOCHReversalArchetype(config=config)
    return archetype.detect(row, regime_label)
