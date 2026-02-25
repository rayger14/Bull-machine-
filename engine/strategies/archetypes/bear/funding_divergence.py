#!/usr/bin/env python3
"""
Funding Divergence / Short Squeeze (S4) - Bear Market Archetype

Production implementation for short squeeze detection during extreme negative funding.

PATTERN LOGIC:
Short squeeze setups occur when funding rates go deeply negative (shorts paying longs),
while open interest is rising. This indicates overleveraged short positions that can be
forced to cover, creating explosive upward moves.

KEY CHARACTERISTICS:
1. Extreme negative funding (funding_rate < -0.01% OR funding_Z < -2.0)
2. Rising open interest (OI increasing while shorts building)
3. Liquidity recovering (orderbook healing)
4. Price holding support (not in freefall)

DIRECTION: LONG (counter-trend reversal in bear markets)

TARGET: 12-18 trades/year, PF > 1.8

BTC EXAMPLES:
- 2022-06-22: Post-Luna funding squeeze → +15% in 2 days
- 2022-12-15: FTX aftermath short squeeze → explosive rally
- 2023-01-10: Negative funding extreme → sustained bounce

Author: Claude Code (Backend Architect)
Date: 2026-01-08
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class FundingDivergenceArchetype:
    """
    Funding Divergence / Short Squeeze archetype.

    Detects short squeeze setups from extreme negative funding + rising OI.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Funding Divergence archetype."""
        self.config = config or {}
        thresholds = self.config.get('thresholds', {})

        # CRITICAL FIX (2026-01-09): Tightened thresholds to reduce overtrading
        # Previous settings caused 308 trades (-$515 PnL) by firing on normal bearish funding
        # New settings target only true short squeeze extremes (expected: 30-50 trades/year)
        #
        # Changes:
        # - min_funding_z: -2.0 → -3.0 (only extreme negative funding, true squeeze risk)
        # - min_fusion_score: 0.35 → 0.55 (require strong conviction, reduces noise)
        # - min_oi_change: 2.0 → 5.0 (real divergence required, not normal fluctuations)

        # Core thresholds
        self.max_funding_rate = thresholds.get('max_funding_rate', -0.0001)  # -0.01%
        self.min_funding_z = thresholds.get('min_funding_z', -3.0)  # Was -2.0
        self.min_oi_change = thresholds.get('min_oi_change', 5.0)  # Was 2.0
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.55)  # Was 0.35

        # Domain weights
        self.funding_weight = thresholds.get('funding_weight', 0.50)
        self.oi_weight = thresholds.get('oi_weight', 0.30)
        self.liquidity_weight = thresholds.get('liquidity_weight', 0.20)

        logger.info(f"[S4 Funding Divergence] Initialized with min_fusion={self.min_fusion_score}")

    def detect(
        self,
        row: pd.Series,
        regime_label: str = 'neutral'
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Detect Funding Divergence / Short Squeeze pattern."""
        # Step 1: Check funding extreme
        funding_score = self._compute_funding_score(row)

        # Step 2: Check OI divergence
        oi_score = self._compute_oi_score(row)

        # Step 3: Check liquidity recovery
        liquidity_score = self._compute_liquidity_score(row)

        # Step 4: Compute weighted fusion
        fusion_score = (
            self.funding_weight * funding_score +
            self.oi_weight * oi_score +
            self.liquidity_weight * liquidity_score
        )

        # Step 5: Apply vetoes
        veto_reason = self._check_vetoes(row, regime_label)
        if veto_reason:
            return None, 0.0, {'veto_reason': veto_reason, 'fusion_score': fusion_score}

        # Step 6: Apply temporal confluence timing multiplier
        temporal_confluence = row.get('temporal_confluence', None)
        temporal_mult = 1.0  # Default neutral
        if temporal_confluence is not None and not pd.isna(temporal_confluence):
            # Apply conservative 0.85-1.15 range (max ±15% adjustment)
            # High confluence (0.80) = 1.09x boost, Low confluence (0.20) = 0.91x penalty
            temporal_mult = 0.85 + (temporal_confluence * 0.30)
            fusion_score *= temporal_mult

        # Step 7: Check threshold
        if fusion_score < self.min_fusion_score:
            return None, 0.0, {
                'reason': 'below_threshold',
                'fusion_score': fusion_score,
                'threshold': self.min_fusion_score,
                'temporal_mult': temporal_mult
            }

        # Signal detected!
        metadata = {
            'funding_score': funding_score,
            'oi_score': oi_score,
            'liquidity_score': liquidity_score,
            'fusion_score': fusion_score,
            'temporal_confluence': temporal_confluence,
            'temporal_mult': temporal_mult,
            'pattern_type': 'funding_divergence_short_squeeze_long'
        }

        return 'funding_divergence', fusion_score, metadata

    def _compute_funding_score(self, row: pd.Series) -> float:
        """Compute funding extreme score."""
        score = 0.0

        # Check funding z-score
        funding_z = row.get('funding_Z', 0.0)
        if funding_z < self.min_funding_z:
            # More negative = higher score
            # z=-2.0 → 0.5, z=-3.0 → 0.75, z=-4.0 → 1.0
            score = min(1.0, (abs(funding_z) - 2.0) / 2.0 + 0.5)

        # Fallback: Check absolute funding rate
        funding_rate = row.get('funding_rate', 0.0)
        if funding_rate < self.max_funding_rate:
            score = max(score, 0.6)

        return score

    def _compute_oi_score(self, row: pd.Series) -> float:
        """Compute OI divergence score."""
        score = 0.0

        # Check OI change
        oi_change = row.get('oi_change_pct_24h', 0.0)
        if oi_change > self.min_oi_change:
            # Rising OI with negative funding = shorts building
            # +2% → 0.4, +5% → 0.7, +10% → 1.0
            score = min(1.0, (oi_change - 2.0) / 8.0 + 0.4)

        return score

    def _compute_liquidity_score(self, row: pd.Series) -> float:
        """Compute liquidity recovery score."""
        score = 0.0

        # Check liquidity score (recovering)
        liquidity = row.get('liquidity_score', 0.5)
        if liquidity > 0.3:  # Not in complete vacuum
            score = min(1.0, liquidity / 0.5) * 0.7  # Reduce weight for thermo-floor addition

        # NEW: Thermo-floor capitulation boost (BTC only)
        # Extreme capitulation = miners selling at loss = bottom signal
        symbol = row.get('symbol', 'BTCUSDT')
        if 'BTC' in symbol:
            # FIX: Use correct feature name from data
            thermo_distance = row.get('thermo_floor_distance', 0.0)
            # If price near/below mining cost floor = strong buy signal
            if thermo_distance < -0.05:  # Price > 5% below mining cost
                # Extreme capitulation boost: -0.05 → 0.15, -0.15 → 0.45, -0.25+ → 0.75
                capitulation_boost = min(0.75, abs(thermo_distance) * 3.0)
                score += capitulation_boost

        return min(1.0, score)

    def _check_vetoes(self, row: pd.Series, regime_label: str) -> Optional[str]:
        """Check safety vetoes."""
        # LPPLS VETO: Don't buy parabolic tops (CRITICAL safety)
        # FIX: Use correct feature names from data
        lppls_veto = row.get('lppls_blowoff_detected', False)
        lppls_confidence = row.get('lppls_confidence', 0.0)
        if lppls_veto and lppls_confidence > 0.75:
            return f'lppls_blowoff_detected_conf_{lppls_confidence:.2f}'

        # PTI VETO: Don't go LONG when retail longs are trapped (they will be liquidated)
        # S4 is a LONG archetype (short squeeze) - veto when bullish_trap detected
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

        # No other hard vetoes for S4
        return None


def detect_funding_divergence_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """Helper function for standalone detection."""
    archetype = FundingDivergenceArchetype(config)
    return archetype.detect(row, regime_label)
