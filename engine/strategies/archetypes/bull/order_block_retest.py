#!/usr/bin/env python3
"""
Order Block Retest (B) - Bull Archetype

PATTERN DESCRIPTION:
Order Block Retest identifies institutional demand zones where smart money
previously accumulated large positions. When price returns to test these zones,
institutions defend their positions, creating high-probability reversal setups.

KEY CHARACTERISTICS:
1. Bullish order block identified (strong move up from a consolidation)
2. Price pulls back to retest the order block zone
3. Price holds above order block bottom (support holds)
4. Volume decreases during pullback (healthy retest)
5. Volume increases on bounce (institutional buying)

SMC THEORY:
Order blocks represent the last consolidation before a strong directional move.
Institutions leave "footprints" (unfilled orders) in these zones. When price
returns, unfilled orders get filled, causing price to reverse.

ENTRY LOGIC:
- Bullish order block exists (tf1h_ob_bull_bottom/top)
- Price is within order block zone or slightly above (0-5% above bottom)
- Price bounced from order block (close > open)
- Volume profile shows support (no volume dump)
- No bearish BOS/CHOCH on higher timeframe

DOMAIN ENGINE INTEGRATION:
- SMC: Order block validation + FVG confluence (35% weight)
- Price Action: Retest bounce confirmation (25% weight)
- Wyckoff: Reaccumulation context (20% weight)
- Volume: Healthy retest volume pattern (15% weight)
- Regime: Risk-on alignment (5% weight)

SAFETY VETOES:
- No entry if price closed below order block bottom (support broken)
- No entry if bearish BOS on 4H (trend reversal)
- No entry if volume spike DOWN during retest (distribution)
- No entry during crisis regime

TARGET METRICS:
- Trades/year: 20-35 (frequent setups in trending markets)
- Win rate: 60-70% (high-probability support zones)
- Profit factor: 2.0-3.0
- Avg hold time: 1-3 days

HISTORICAL EXAMPLES (BTC):
- 2023-10-16: 4H OB retest → +18% rally
- 2024-02-12: Daily OB retest → sustained markup
- 2024-03-05: 1H OB retest → quick +8% bounce
- 2024-10-14: Pre-election OB defense → explosive move

Author: Claude Code (Backend Architect)
Date: 2025-12-12
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class OrderBlockRetestArchetype:
    """
    Minimal viable implementation of Order Block Retest archetype.

    Detects price retests of bullish order blocks for long entries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Order Block Retest archetype.

        Args:
            config: Optional configuration dict with thresholds
        """
        self.config = config or {}

        # Extract thresholds with defaults (permissive for MVP)
        thresholds = self.config.get('thresholds', {})

        # Core pattern thresholds
        self.max_distance_from_ob = thresholds.get('max_distance_from_ob', 0.05)  # 5% above OB
        self.min_bounce_body = thresholds.get('min_bounce_body', 0.3)  # 30% of candle
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.35)

        # Domain engine weights
        self.smc_weight = thresholds.get('smc_weight', 0.35)
        self.price_action_weight = thresholds.get('price_action_weight', 0.25)
        self.wyckoff_weight = thresholds.get('wyckoff_weight', 0.20)
        self.volume_weight = thresholds.get('volume_weight', 0.15)
        self.regime_weight = thresholds.get('regime_weight', 0.05)

        # Safety thresholds
        self.max_volume_spike_down = thresholds.get('max_volume_spike_down', -1.5)

        logger.info(f"[Order Block Retest] Initialized with min_fusion={self.min_fusion_score}")

    def detect(self, row: pd.Series, regime_label: str = 'neutral') -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Detect Order Block Retest pattern in current bar.

        Args:
            row: Current bar data with features
            regime_label: Current regime classification

        Returns:
            Tuple of (archetype_name, confidence_score, metadata)
            Returns (None, 0.0, {}) if no signal
        """
        # Step 1: Check SMC order block and retest
        smc_score = self._compute_smc_score(row)

        # Step 2: Check price action bounce
        price_action_score = self._compute_price_action_score(row)

        # Step 3: Check Wyckoff reaccumulation context
        wyckoff_score = self._compute_wyckoff_score(row)

        # Step 4: Check volume pattern
        volume_score = self._compute_volume_score(row)

        # Step 5: Check regime alignment
        regime_score = self._compute_regime_score(regime_label)

        # Step 6: Compute weighted fusion score
        fusion_score = (
            self.smc_weight * smc_score +
            self.price_action_weight * price_action_score +
            self.wyckoff_weight * wyckoff_score +
            self.volume_weight * volume_score +
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
            'smc_score': smc_score,
            'price_action_score': price_action_score,
            'wyckoff_score': wyckoff_score,
            'volume_score': volume_score,
            'regime_score': regime_score,
            'fusion_score': fusion_score,
            'pattern_type': 'order_block_retest_long'
        }

        return 'order_block_retest', fusion_score, metadata

    def _compute_smc_score(self, row: pd.Series) -> float:
        """
        Compute SMC domain engine score.

        Checks for:
        - Bullish order block present
        - Price retesting order block zone
        - FVG confluence (optional boost)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for bullish order block
        ob_bull_bottom = row.get('tf1h_ob_bull_bottom', None)
        ob_bull_top = row.get('tf1h_ob_bull_top', None)
        close = row.get('close', 0)
        low = row.get('low', 0)

        if ob_bull_bottom is None or ob_bull_top is None or close == 0:
            return 0.0

        # Calculate distance from order block
        if close < ob_bull_bottom:
            # Price below OB = broken support
            return 0.0

        # Check if price is in or near order block
        ob_range = ob_bull_top - ob_bull_bottom
        max_distance = ob_bull_bottom + (ob_range * (1 + self.max_distance_from_ob))

        if ob_bull_bottom <= close <= max_distance:
            # Price is in retest zone
            # Score higher if price is closer to OB
            if close <= ob_bull_top:
                # Inside OB = perfect retest
                score += 0.60
            else:
                # Slightly above OB
                distance_ratio = (close - ob_bull_top) / (max_distance - ob_bull_top)
                score += 0.60 * (1.0 - distance_ratio)

        # Check if low touched OB (more precise retest)
        if ob_bull_bottom <= low <= ob_bull_top:
            score += 0.20

        # Check for FVG confluence
        fvg_bull = row.get('tf1h_fvg_bull', False)
        if fvg_bull:
            score += 0.20

        return min(1.0, score)

    def _compute_price_action_score(self, row: pd.Series) -> float:
        """
        Compute price action domain engine score.

        Checks for:
        - Bullish bounce (close > open)
        - Strong body relative to range
        - Price holding above support

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        open_price = row.get('open', 0)
        close = row.get('close', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)

        if high <= low or close == 0:
            return 0.0

        # Check for bullish candle
        if close > open_price:
            # Bullish body
            body_size = close - open_price
            candle_range = high - low

            if candle_range > 0:
                body_ratio = body_size / candle_range

                if body_ratio >= self.min_bounce_body:
                    # Strong bullish body
                    # Normalize: 0.30 = 0.5, 0.70 = 1.0
                    body_score = min(1.0, (body_ratio - self.min_bounce_body) / 0.40)
                    score += 0.60 * body_score

            # Check close position (high in range = bullish)
            close_position = (close - low) / candle_range
            if close_position > 0.6:
                score += 0.20

        # Check for lower wick (support test)
        body_low = min(open_price, close)
        wick_lower = body_low - low
        candle_range = high - low

        if candle_range > 0:
            wick_ratio = wick_lower / candle_range
            if 0.15 <= wick_ratio <= 0.40:  # Moderate wick = healthy retest
                score += 0.20

        return min(1.0, score)

    def _compute_wyckoff_score(self, row: pd.Series) -> float:
        """
        Compute Wyckoff domain engine score.

        Checks for:
        - Reaccumulation phase (Phase B/C)
        - SOS (Sign of Strength)
        - LPS (Last Point of Support)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check Wyckoff phase
        phase = row.get('wyckoff_phase_abc', 'neutral')
        if phase in ['B', 'C']:  # Reaccumulation phases
            score += 0.40

        # Check for SOS (Sign of Strength)
        sos = row.get('wyckoff_sos', False)
        sos_conf = row.get('wyckoff_sos_confidence', 0.0)
        if sos and sos_conf >= 0.5:
            score += 0.30

        # Check for LPS (Last Point of Support)
        lps = row.get('wyckoff_lps', False)
        lps_conf = row.get('wyckoff_lps_confidence', 0.0)
        if lps and lps_conf >= 0.5:
            score += 0.30

        return min(1.0, score)

    def _compute_volume_score(self, row: pd.Series) -> float:
        """
        Compute volume domain engine score.

        Healthy retest pattern:
        - Volume decreases during pullback (no panic)
        - Volume increases on bounce (buying pressure)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check volume z-score (prefer moderate, not extreme)
        volume_zscore = row.get('volume_zscore', 0.0)

        if -0.5 <= volume_zscore <= 1.5:
            # Healthy volume range for retest
            score += 0.50
        elif volume_zscore > 1.5:
            # High volume on bounce = good
            score += 0.30

        # Check if we have volume trend
        # (This would require rolling volume, simplified here)
        # For MVP, we use snapshot volume_zscore

        # Check for no volume dump
        if volume_zscore > -1.0:
            score += 0.50

        return min(1.0, score)

    def _compute_regime_score(self, regime_label: str) -> float:
        """
        Compute regime alignment score.

        Order block retests work best in risk_on and neutral regimes.

        Args:
            regime_label: Current regime

        Returns:
            Score 0.0-1.0
        """
        regime_scores = {
            'risk_on': 1.0,      # Ideal for OB retests
            'neutral': 0.8,      # Still good
            'risk_off': 0.3,     # Risky
            'crisis': 0.1        # Very selective
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
        # Veto 1: Price closed below order block (support broken)
        ob_bull_bottom = row.get('tf1h_ob_bull_bottom', None)
        close = row.get('close', 0)

        if ob_bull_bottom is not None and close > 0:
            if close < ob_bull_bottom:
                return 'ob_support_broken'

        # Veto 2: Bearish BOS on 4H (trend reversal)
        tf4h_bos_bearish = row.get('tf4h_bos_bearish', False)
        if tf4h_bos_bearish:
            return 'bearish_bos_4h'

        # Veto 3: Volume spike DOWN during retest (distribution)
        volume_zscore = row.get('volume_zscore', 0.0)
        close_price = row.get('close', 0)
        open_price = row.get('open', 0)

        if volume_zscore < self.max_volume_spike_down and close_price < open_price:
            return f'volume_dump_{volume_zscore:.1f}'

        # Veto 4: Crisis regime
        if regime_label == 'crisis':
            return 'crisis_regime'

        return None


# ============================================================================
# Integration Helper
# ============================================================================

def detect_order_block_retest_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Convenience function to detect Order Block Retest signals.

    Args:
        row: Current bar data
        regime_label: Current regime
        config: Optional configuration

    Returns:
        Tuple of (archetype_name, confidence, metadata)
    """
    archetype = OrderBlockRetestArchetype(config=config)
    return archetype.detect(row, regime_label)
