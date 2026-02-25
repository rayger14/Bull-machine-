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

        # Domain engine weights (rebalanced to include Wyckoff)
        self.smc_weight = thresholds.get('smc_weight', 0.35)
        self.momentum_weight = thresholds.get('momentum_weight', 0.25)
        self.volume_weight = thresholds.get('volume_weight', 0.15)
        self.wyckoff_weight = thresholds.get('wyckoff_weight', 0.15)
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

        # Step 4: Check Wyckoff events
        wyckoff_score = self._compute_wyckoff_score(row)

        # Step 5: Check regime alignment
        regime_score = self._compute_regime_score(regime_label)

        # Step 6: Compute weighted fusion score
        fusion_score = (
            self.smc_weight * smc_score +
            self.momentum_weight * momentum_score +
            self.volume_weight * volume_score +
            self.wyckoff_weight * wyckoff_score +
            self.regime_weight * regime_score
        )

        # THERMO-FLOOR BOOST: Extreme capitulation = strong buy signal (BTC only)
        symbol = row.get('symbol', 'BTCUSDT')
        if 'BTC' in symbol:
            # FIX: Use correct feature name from data
            thermo_distance = row.get('thermo_floor_distance', 0.0)
            # If price near/below mining cost = extreme capitulation
            if thermo_distance < -0.10:  # Price > 10% below mining cost
                # Miners selling at loss = bottom signal → boost by 2x
                fusion_score *= 2.00
                logger.debug(f"[C Thermo Boost] Extreme capitulation (distance={thermo_distance:.2f}), boosting by 2.0x")

        # Step 6: Apply safety vetoes
        veto_reason = self._check_vetoes(row, regime_label)
        if veto_reason:
            return None, 0.0, {'veto_reason': veto_reason, 'fusion_score': fusion_score}

        # Step 7: Apply temporal confluence timing multiplier
        temporal_confluence = row.get('temporal_confluence', None)
        temporal_mult = 1.0  # Default neutral
        if temporal_confluence is not None and not pd.isna(temporal_confluence):
            # Apply conservative 0.85-1.15 range (max ±15% adjustment)
            # High confluence (0.80) = 1.09x boost, Low confluence (0.20) = 0.91x penalty
            temporal_mult = 0.85 + (temporal_confluence * 0.30)
            fusion_score *= temporal_mult

        # Step 8: Check fusion threshold
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
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'wyckoff_score': wyckoff_score,
            'regime_score': regime_score,
            'fusion_score': fusion_score,
            'temporal_confluence': temporal_confluence,
            'temporal_mult': temporal_mult,
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
        - 4H CHOCH flag (additional confirmation)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for bullish BOS on 1H
        tf1h_bos_bull = row.get('tf1h_bos_bullish', False)
        if tf1h_bos_bull:
            score += 0.35

        # Check for bullish BOS on 4H (stronger signal)
        tf4h_bos_bull = row.get('tf4h_bos_bullish', False)
        if tf4h_bos_bull:
            score += 0.45

        # Check for CHOCH (if available)
        smc_choch = row.get('smc_choch', False)
        if smc_choch:
            score += 0.25

        # NEW: Check for 4H CHOCH flag (higher timeframe character change)
        tf4h_choch = row.get('tf4h_choch_flag', False)
        if tf4h_choch:
            score += 0.20  # Strong confirmation of trend reversal

        # Check 4H fusion score as structure quality indicator
        tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
        if tf4h_fusion > 0.5:
            score += 0.15 * (tf4h_fusion - 0.5) / 0.5

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

    def _compute_wyckoff_score(self, row: pd.Series) -> float:
        """
        Compute Wyckoff domain engine score.

        Checks for:
        - SOS (Sign of Strength) - breakout confirmation
        - LPS (Last Point of Support) - retest before continuation
        - Phase D/E - trend continuation context

        Returns:
            Score 0.0-1.0
        """
        score = 0.0
        confidence_threshold = 0.70  # High-confidence threshold

        # Check for SOS (Sign of Strength - perfect for BOS/CHOCH)
        sos = row.get('wyckoff_sos', False)
        sos_conf = row.get('wyckoff_sos_confidence', 0.0)
        if sos and sos_conf >= confidence_threshold:
            score += 0.50  # Highest weight - confirms breakout strength

        # Check for LPS (Last Point of Support - retest before markup)
        lps = row.get('wyckoff_lps', False)
        lps_conf = row.get('wyckoff_lps_confidence', 0.0)
        if lps and lps_conf >= confidence_threshold:
            score += 0.30

        # Check Wyckoff phase context
        phase = row.get('wyckoff_phase_abc', 'neutral')
        if phase == 'D':  # Trend beginning phase
            score += 0.25
        elif phase == 'E':  # Trend continuation phase
            score += 0.20

        # Check for AR (Automatic Rally) - initial bounce after capitulation
        ar = row.get('wyckoff_ar', False)
        ar_conf = row.get('wyckoff_ar_confidence', 0.0)
        if ar and ar_conf >= confidence_threshold:
            score += 0.20

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
        # LPPLS VETO: Don't buy parabolic tops (CRITICAL safety)
        # FIX: Use correct feature names from data
        lppls_veto = row.get('lppls_blowoff_detected', False)
        lppls_confidence = row.get('lppls_confidence', 0.0)
        if lppls_veto and lppls_confidence > 0.75:
            return f'lppls_blowoff_detected_conf_{lppls_confidence:.2f}'

        # PTI VETO: Don't go LONG when retail longs are trapped (they will be liquidated)
        # BOS/CHOCH Reversal is a LONG archetype - veto when bullish_trap detected
        # FIX: Use correct feature names from data
        pti_score = row.get('tf1h_pti_score', 0.0)
        pti_confidence = row.get('tf1h_pti_confidence', 0.0)
        # Derive trap type from tf1d_pti_reversal (1=bullish reversal, -1=bearish reversal)
        pti_reversal = row.get('tf1d_pti_reversal', 0)
        pti_trap_type = 'bullish_trap' if pti_reversal < 0 else ('bearish_trap' if pti_reversal > 0 else 'none')

        if (pti_trap_type == 'bullish_trap' and
            pti_score > 0.60 and
            pti_confidence > 0.70):
            # Smart money will push down to liquidate trapped longs
            return f'pti_bullish_trap_veto_score_{pti_score:.2f}_conf_{pti_confidence:.2f}'

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

        # Veto 6: UTAD (Upthrust After Distribution) - distribution top
        utad = row.get('wyckoff_utad', False)
        utad_conf = row.get('wyckoff_utad_confidence', 0.0)
        if utad and utad_conf >= 0.70:
            return 'wyckoff_utad_distribution_top'

        # Veto 7: SOW (Sign of Weakness) - bearish breakdown signal
        sow = row.get('wyckoff_sow', False)
        sow_conf = row.get('wyckoff_sow_confidence', 0.0)
        if sow and sow_conf >= 0.70:
            return 'wyckoff_sow_weakness_detected'

        # Veto 8: AS (Automatic Reaction) - relief drop after BC
        as_event = row.get('wyckoff_as', False)
        as_conf = row.get('wyckoff_as_confidence', 0.0)
        if as_event and as_conf >= 0.70:
            return 'wyckoff_as_selling_pressure'

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
