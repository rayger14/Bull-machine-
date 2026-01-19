#!/usr/bin/env python3
"""
Trap Within Trend (H) - Bull Archetype

PATTERN DESCRIPTION:
Trap Within Trend identifies false breakdowns within an established uptrend.
Price briefly breaks support to trap bears, then rapidly reverses to continue
the uptrend. This creates high-probability continuation setups as trapped shorts
cover and trend followers re-enter.

KEY CHARACTERISTICS:
1. Established uptrend on higher timeframe (4H/Daily)
2. Brief violation of short-term support (trap)
3. Rapid recovery back into trend structure
4. Deep lower wick showing rejection
5. Momentum remains bullish (ADX > 20, RSI > 45)

MARKET MECHANICS:
During healthy uptrends, price naturally pulls back to support. Smart money
uses these pullbacks to shake out weak hands and trap bears:
1. Price breaks below short-term support
2. Bears enter shorts (trapped)
3. Smart money accumulates at discount
4. Price reverses sharply (short squeeze)
5. Trend continues with renewed momentum

ENTRY LOGIC:
- 4H trend is UP (tf4h_trend_direction > 0)
- Price broke below recent support then recovered
- Deep lower wick (>30%) shows rejection
- Close back above support level
- Momentum indicators remain bullish
- No bearish divergence

DOMAIN ENGINE INTEGRATION:
- Momentum: Trend continuation confirmation (35% weight)
- Price Action: Trap + reversal pattern (30% weight)
- Wyckoff: Reaccumulation context (20% weight)
- Volume: Healthy pullback profile (10% weight)
- Regime: Risk-on alignment (5% weight)

SAFETY VETOES:
- No entry if 4H trend turned bearish
- No entry if momentum broken (ADX < 15)
- No entry if RSI < 40 (trend weakening)
- No entry if bearish BOS on 4H (trend reversal)
- No entry during crisis regime

TARGET METRICS:
- Trades/year: 25-40 (frequent in trending markets)
- Win rate: 65-75% (strong trend edge)
- Profit factor: 2.3-3.2
- Avg hold time: 2-5 days

HISTORICAL EXAMPLES (BTC):
- 2023-10-20: Pullback trap → ATH run continues
- 2024-02-08: Consolidation trap → breakout to $50k
- 2024-03-01: Support test → continuation to $70k
- 2024-10-18: Pre-election dip → explosive rally

Author: Claude Code (Backend Architect)
Date: 2025-12-12
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class TrapWithinTrendArchetype:
    """
    Minimal viable implementation of Trap Within Trend archetype.

    Detects false breakdown traps within uptrends for continuation entries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Trap Within Trend archetype.

        Args:
            config: Optional configuration dict with thresholds
        """
        self.config = config or {}

        # Extract thresholds with defaults (permissive for MVP)
        thresholds = self.config.get('thresholds', {})

        # Core pattern thresholds
        self.min_htf_trend = thresholds.get('min_htf_trend', 0)  # Must be uptrend
        self.min_wick_lower_ratio = thresholds.get('min_wick_lower_ratio', 0.25)
        self.min_adx = thresholds.get('min_adx', 15)
        self.min_rsi = thresholds.get('min_rsi', 40)
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.35)

        # Domain engine weights
        self.momentum_weight = thresholds.get('momentum_weight', 0.35)
        self.price_action_weight = thresholds.get('price_action_weight', 0.30)
        self.wyckoff_weight = thresholds.get('wyckoff_weight', 0.20)
        self.volume_weight = thresholds.get('volume_weight', 0.10)
        self.regime_weight = thresholds.get('regime_weight', 0.05)

        logger.info(f"[Trap Within Trend] Initialized with min_fusion={self.min_fusion_score}")

    def detect(self, row: pd.Series, regime_label: str = 'neutral') -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Detect Trap Within Trend pattern in current bar.

        Args:
            row: Current bar data with features
            regime_label: Current regime classification

        Returns:
            Tuple of (archetype_name, confidence_score, metadata)
            Returns (None, 0.0, {}) if no signal
        """
        # Step 1: Check momentum (trend continuation)
        momentum_score = self._compute_momentum_score(row)

        # Step 2: Check price action (trap + reversal)
        price_action_score = self._compute_price_action_score(row)

        # Step 3: Check Wyckoff reaccumulation
        wyckoff_score = self._compute_wyckoff_score(row)

        # Step 4: Check volume profile
        volume_score = self._compute_volume_score(row)

        # Step 5: Check regime alignment
        regime_score = self._compute_regime_score(regime_label)

        # Step 6: Compute weighted fusion score
        fusion_score = (
            self.momentum_weight * momentum_score +
            self.price_action_weight * price_action_score +
            self.wyckoff_weight * wyckoff_score +
            self.volume_weight * volume_score +
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
                logger.debug(f"[H Thermo Boost] Extreme capitulation (distance={thermo_distance:.2f}), boosting by 2.0x")

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
            'momentum_score': momentum_score,
            'price_action_score': price_action_score,
            'wyckoff_score': wyckoff_score,
            'volume_score': volume_score,
            'regime_score': regime_score,
            'fusion_score': fusion_score,
            'temporal_confluence': temporal_confluence,
            'temporal_mult': temporal_mult,
            'pattern_type': 'trap_within_trend_long'
        }

        return 'trap_within_trend', fusion_score, metadata

    def _compute_momentum_score(self, row: pd.Series) -> float:
        """
        Compute momentum domain engine score.

        Checks for:
        - 4H uptrend (primary requirement)
        - ADX trend strength maintained
        - RSI above 40 (trend intact)
        - Daily trend alignment

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check 4H trend (CRITICAL - must be uptrend)
        tf4h_trend = row.get('tf4h_trend_direction', 0)
        if tf4h_trend <= self.min_htf_trend:
            # No uptrend = no trap within trend
            return 0.0

        # Uptrend confirmed
        score += 0.40

        # Check 4H fusion score (trend quality)
        tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
        if tf4h_fusion > 0.5:
            score += 0.20

        # Check ADX (trend strength)
        adx = row.get('adx_14', 20)
        if adx >= self.min_adx:
            # Normalize: 15 = 0.5, 30+ = 1.0
            adx_score = min(1.0, (adx - self.min_adx) / 15)
            score += 0.20 * adx_score

        # Check RSI (trend momentum)
        rsi = row.get('rsi_14', 50)
        if rsi >= self.min_rsi:
            # Normalize: 40 = 0.5, 60+ = 1.0
            rsi_score = min(1.0, (rsi - self.min_rsi) / 20)
            score += 0.20 * rsi_score

        return min(1.0, score)

    def _compute_price_action_score(self, row: pd.Series) -> float:
        """
        Compute price action domain engine score.

        Checks for:
        - Deep lower wick (trap below support)
        - Bullish recovery (close > open)
        - Close in upper range (strength)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Calculate lower wick ratio
        wick_lower_ratio = self._calculate_wick_lower_ratio(row)

        if wick_lower_ratio >= self.min_wick_lower_ratio:
            # Trap detected
            # Normalize: 0.25 = 0.5, 0.50 = 1.0
            wick_score = min(1.0, (wick_lower_ratio - self.min_wick_lower_ratio) / 0.25)
            score += 0.50 * wick_score

        # Check for bullish recovery
        open_price = row.get('open', 0)
        close = row.get('close', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)

        if high > low and close > 0:
            candle_range = high - low

            # Bullish body
            if close > open_price:
                body_size = close - open_price
                if candle_range > 0:
                    body_ratio = body_size / candle_range
                    score += 0.30 * body_ratio

            # Close position (high in range = strength)
            if candle_range > 0:
                close_position = (close - low) / candle_range
                if close_position > 0.5:
                    score += 0.20 * close_position

        return min(1.0, score)

    def _compute_wyckoff_score(self, row: pd.Series) -> float:
        """
        Compute Wyckoff domain engine score.

        Checks for:
        - Reaccumulation phase (Phase B/C)
        - LPS (Last Point of Support)
        - SOS (Sign of Strength)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check Wyckoff phase
        phase = row.get('wyckoff_phase_abc', 'neutral')
        if phase == 'B':  # Reaccumulation
            score += 0.40
        elif phase == 'C':  # Testing
            score += 0.30

        # Check for LPS (Last Point of Support before continuation)
        lps = row.get('wyckoff_lps', False)
        lps_conf = row.get('wyckoff_lps_confidence', 0.0)
        if lps and lps_conf >= 0.5:
            score += 0.30

        # Check for SOS (Sign of Strength)
        sos = row.get('wyckoff_sos', False)
        sos_conf = row.get('wyckoff_sos_confidence', 0.0)
        if sos and sos_conf >= 0.5:
            score += 0.30

        return min(1.0, score)

    def _compute_volume_score(self, row: pd.Series) -> float:
        """
        Compute volume domain engine score.

        Healthy pullback in uptrend:
        - Volume decreases during pullback
        - Volume increases on reversal

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check volume z-score
        volume_zscore = row.get('volume_zscore', 0.0)

        # Moderate volume is healthy for pullback
        if -0.5 <= volume_zscore <= 1.5:
            score += 0.60

        # Higher volume on bounce is bullish
        if volume_zscore > 1.0:
            score += 0.40

        return min(1.0, score)

    def _compute_regime_score(self, regime_label: str) -> float:
        """
        Compute regime alignment score.

        Trap within trend works best in risk_on regimes.

        Args:
            regime_label: Current regime

        Returns:
            Score 0.0-1.0
        """
        regime_scores = {
            'risk_on': 1.0,      # Ideal for trend continuation
            'neutral': 0.8,      # Still good
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
        # PTI VETO: Don't go LONG when retail longs are trapped (they will be liquidated)
        # Trap Within Trend is a LONG archetype - veto when bullish_trap detected
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

        # Veto 0: LPPLS blowoff - Don't buy parabolic tops (CRITICAL safety)
        # FIX: Use correct feature names from data
        lppls_veto = row.get('lppls_blowoff_detected', False)
        lppls_confidence = row.get('lppls_confidence', 0.0)
        if lppls_veto and lppls_confidence > 0.75:
            return f'lppls_blowoff_detected_conf_{lppls_confidence:.2f}'

        # Veto 1: 4H trend turned bearish
        tf4h_trend = row.get('tf4h_trend_direction', 0)
        if tf4h_trend < 0:
            return 'htf_trend_bearish'

        # Veto 2: Momentum broken
        adx = row.get('adx_14', 20)
        if adx < self.min_adx:
            return f'weak_momentum_adx_{adx:.1f}'

        # Veto 3: RSI showing weakness
        rsi = row.get('rsi_14', 50)
        if rsi < self.min_rsi:
            return f'weak_rsi_{rsi:.1f}'

        # Veto 4: Bearish BOS on 4H (trend reversal)
        tf4h_bos_bearish = row.get('tf4h_bos_bearish', False)
        if tf4h_bos_bearish:
            return 'bearish_bos_4h'

        # Veto 5: Bearish divergence
        if row.get('bearish_divergence_detected', False):
            return 'bearish_divergence'

        # Veto 6: Crisis regime
        if regime_label == 'crisis':
            return 'crisis_regime'

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

def detect_trap_within_trend_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Convenience function to detect Trap Within Trend signals.

    Args:
        row: Current bar data
        regime_label: Current regime
        config: Optional configuration

    Returns:
        Tuple of (archetype_name, confidence, metadata)
    """
    archetype = TrapWithinTrendArchetype(config=config)
    return archetype.detect(row, regime_label)
