#!/usr/bin/env python3
"""
Long Squeeze Cascade (S5) - Bear Market Archetype

Production implementation for long squeeze detection during bull exhaustion.

CRITICAL DIRECTION FIX:
S5 goes SHORT (contrarian short during bull exhaustion with overleveraged longs).

PATTERN LOGIC:
Long squeeze cascades occur when funding rates go deeply positive (longs paying shorts),
while price shows weakness. This indicates overleveraged long positions that can be
forced to liquidate, creating cascading sell-offs.

KEY CHARACTERISTICS:
1. Extreme positive funding (funding_rate > +0.01% OR funding_Z > +2.0)
2. BOS down detected (smart money selling)
3. Liquidity draining (orderbook thinning)
4. OI rising (more longs entering trap)

DIRECTION: SHORT (contrarian short in bull exhaustion)

TARGET: 8-12 trades/year, PF > 2.0

BTC EXAMPLES:
- 2021-04-18: Pre-May crash funding extreme → -20% cascade
- 2021-09-07: El Salvador overhype → funding peak → -18% dump
- 2022-04-04: False rally funding → -10% correction

Author: Claude Code (Backend Architect)
Date: 2026-01-08
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class LongSqueezeArchetype:
    """
    Long Squeeze Cascade archetype.

    Detects long squeeze setups from extreme positive funding + weakness.
    GOES SHORT (contrarian during bull exhaustion).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Long Squeeze archetype."""
        self.config = config or {}
        thresholds = self.config.get('thresholds', {})

        # Core thresholds - FIXED v2 (tightened)
        self.min_funding_rate = thresholds.get('min_funding_rate', 0.0001)  # +0.01%
        self.min_funding_z = thresholds.get('min_funding_z', 2.5)  # Was 2.0, now 2.5
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.55)  # Was 0.40, now 0.55
        self.min_oi_change = thresholds.get('min_oi_change_24h', 5.0)  # NEW: Rising OI required

        # Trend filters - NEW (FIX #2)
        self.adx_min = thresholds.get('adx_min', 25)
        self.adx_max = thresholds.get('adx_max', 50)
        self.require_trend_weakening = thresholds.get('require_trend_weakening', True)

        # Domain weights
        self.funding_weight = thresholds.get('funding_weight', 0.40)
        self.smc_weight = thresholds.get('smc_weight', 0.30)
        self.liquidity_weight = thresholds.get('liquidity_weight', 0.20)
        self.oi_weight = thresholds.get('oi_weight', 0.10)

        logger.info(f"[S5 Long Squeeze] Initialized FIXED v2 (2026-01-08)")
        logger.info(f"[S5 Long Squeeze] min_fusion={self.min_fusion_score}, funding_Z={self.min_funding_z}, OI_change={self.min_oi_change}%")
        logger.info(f"[S5 Long Squeeze] ADX range: {self.adx_min}-{self.adx_max}, weakening_filter={self.require_trend_weakening}")
        logger.info(f"[S5 Long Squeeze] DIRECTION=SHORT (contrarian in bull exhaustion)")

    def detect(
        self,
        row: pd.Series,
        regime_label: str = 'neutral'
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Detect Long Squeeze Cascade pattern."""
        # Step 1: Check funding extreme (positive = longs paying shorts)
        funding_score = self._compute_funding_score(row)

        # Step 2: Check SMC weakness (BOS down)
        smc_score = self._compute_smc_score(row)

        # Step 3: Check liquidity drain
        liquidity_score = self._compute_liquidity_score(row)

        # Step 4: Check OI divergence
        oi_score = self._compute_oi_score(row)

        # Step 5: Compute weighted fusion
        fusion_score = (
            self.funding_weight * funding_score +
            self.smc_weight * smc_score +
            self.liquidity_weight * liquidity_score +
            self.oi_weight * oi_score
        )

        # PTI BOOST: Boost SHORT signals when retail longs are trapped
        # S5 is a SHORT archetype - boost when bullish_trap detected (retail will be liquidated)
        # FIX: Use correct feature names from data
        pti_score = row.get('tf1h_pti_score', 0.0)
        # Derive trap type from tf1d_pti_reversal (1=bullish reversal, -1=bearish reversal)
        pti_reversal = row.get('tf1d_pti_reversal', 0)
        pti_trap_type = 'bullish_trap' if pti_reversal < 0 else ('bearish_trap' if pti_reversal > 0 else 'none')

        if pti_trap_type == 'bullish_trap' and pti_score > 0.60:
            # Smart money will liquidate trapped longs - boost short signal
            fusion_score *= 1.50
            logger.debug(f"[S5 PTI Boost] Bullish trap detected (score={pti_score:.2f}), boosting fusion_score by 1.5x")

        # LPPLS BOOST: Boost SHORT signals on parabolic blowoff tops
        # High probability reversal = excellent short entry
        # FIX: Use correct feature names from data
        lppls_veto = row.get('lppls_blowoff_detected', False)
        lppls_confidence = row.get('lppls_confidence', 0.0)

        if lppls_veto and lppls_confidence > 0.75:
            # Parabolic blowoff detected - boost short signal by 2x
            fusion_score *= 2.00
            logger.debug(f"[S5 LPPLS Boost] Blowoff top detected (conf={lppls_confidence:.2f}), boosting fusion_score by 2.0x")

        # Step 6: Apply vetoes
        veto_reason = self._check_vetoes(row, regime_label)
        if veto_reason:
            return None, 0.0, {
                'veto_reason': veto_reason,
                'fusion_score': fusion_score,
                'smc_score': smc_score,
                'funding_score': funding_score,
                'liquidity_score': liquidity_score,
                'oi_score': oi_score
            }

        # Step 7: Apply temporal confluence timing multiplier
        temporal_confluence = row.get('temporal_confluence', None)
        temporal_mult = 1.0  # Default neutral
        if temporal_confluence is not None and not pd.isna(temporal_confluence):
            # Apply conservative 0.85-1.15 range (max ±15% adjustment)
            # High confluence (0.80) = 1.09x boost, Low confluence (0.20) = 0.91x penalty
            temporal_mult = 0.85 + (temporal_confluence * 0.30)
            fusion_score *= temporal_mult

        # Step 8: Check threshold
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
            'smc_score': smc_score,
            'liquidity_score': liquidity_score,
            'oi_score': oi_score,
            'fusion_score': fusion_score,
            'temporal_confluence': temporal_confluence,
            'temporal_mult': temporal_mult,
            'pattern_type': 'long_squeeze_cascade_short',
            'direction': 'short'  # CRITICAL: S5 goes SHORT
        }

        return 'long_squeeze', fusion_score, metadata

    def _compute_funding_score(self, row: pd.Series) -> float:
        """Compute funding extreme score (positive = longs overleveraged)."""
        score = 0.0

        # Check funding z-score
        funding_z = row.get('funding_Z', 0.0)
        if funding_z > self.min_funding_z:
            # More positive = higher score
            # z=+2.0 → 0.5, z=+3.0 → 0.75, z=+4.0 → 1.0
            score = min(1.0, (funding_z - 2.0) / 2.0 + 0.5)

        # Fallback: Check absolute funding rate
        funding_rate = row.get('funding_rate', 0.0)
        if funding_rate > self.min_funding_rate:
            score = max(score, 0.6)

        return score

    def _compute_smc_score(self, row: pd.Series) -> float:
        """Compute SMC weakness score (BOS down detected)."""
        score = 0.0

        # Check BOS down detection
        bos_detected = row.get('bos_detected', False)
        if bos_detected:
            score += 0.50

        # Check CHOCH (change of character)
        choch_detected = row.get('choch_detected', False)
        if choch_detected:
            score += 0.30

        # NEW: Check for FVG low (Fair Value Gap downside target)
        # FVG low = price gap below that should be filled = downside target for shorts
        fvg_low = row.get('tf1h_fvg_low', False)
        if fvg_low:
            score += 0.20  # Confirms downside target exists

        return min(1.0, score)

    def _compute_liquidity_score(self, row: pd.Series) -> float:
        """Compute liquidity drain score."""
        score = 0.0

        # Check liquidity draining
        liquidity = row.get('liquidity_score', 0.5)
        if liquidity < 0.30:  # Draining = vulnerable to cascade
            score = 1.0 - (liquidity / 0.30)  # Lower = higher score

        # Check liquidity drain percentage
        drain_pct = row.get('liquidity_drain_pct', 0.0)
        if drain_pct < -0.20:  # Draining
            score = max(score, abs(drain_pct) / 0.50)

        return min(1.0, score)

    def _compute_oi_score(self, row: pd.Series) -> float:
        """Compute OI divergence score (rising OI = more longs entering)."""
        score = 0.0

        # Check OI change with HIGHER threshold (FIX #3)
        oi_change = row.get('oi_change_pct_24h', 0.0)
        if oi_change > self.min_oi_change:  # Rising OI with positive funding = longs building
            # More extreme = higher score
            # 5% → 0.5, 10% → 0.75, 15%+ → 1.0
            score = min(1.0, (oi_change - self.min_oi_change) / 10.0 + 0.5)

        return score

    def _check_vetoes(self, row: pd.Series, regime_label: str) -> Optional[str]:
        """Check safety vetoes - ENHANCED with trend filters (FIX #2)."""

        # VETO 0: Thermo-floor veto (BTC only) - Don't short into miner capitulation
        # Miners selling at loss = likely bottom, bounce imminent
        symbol = row.get('symbol', 'BTCUSDT')
        if 'BTC' in symbol:
            # FIX: Use correct feature name from data
            thermo_distance = row.get('thermo_floor_distance', 0.0)
            # If price near/below mining cost floor = veto shorts
            if thermo_distance < 0.10:  # Price within 10% above mining cost
                return f'thermo_floor_capitulation_veto_distance_{thermo_distance:.2f}'

        # VETO 1: Don't short parabolic uptrends (too risky)
        adx = row.get('adx_14', 0)
        trend_4h = row.get('tf4h_external_trend', 0)

        if trend_4h == 1 and adx > self.adx_max:
            # Parabolic uptrend (ADX >50) - too late or too risky to short
            return 'parabolic_uptrend_veto'

        # VETO 2: Don't short strong uptrends without extreme funding
        if trend_4h == 1 and adx > 35:
            # Strong uptrend - only allow if EXTREME funding (Z >3.0)
            funding_z = row.get('funding_Z', 0)
            if funding_z < 3.0:
                return 'strong_uptrend_insufficient_funding_veto'

        # VETO 3: Require trend weakening (if configured)
        if self.require_trend_weakening and adx > self.adx_min:
            # Check if ADX is declining (trend losing steam)
            # Note: Requires 'adx_14_1' or similar lagged ADX feature
            adx_prev = row.get('adx_14_1', row.get('adx_14_prev', adx))
            if adx >= adx_prev:
                # ADX still rising or flat = trend strengthening, not weakening
                return 'trend_strengthening_veto'

        # VETO 4: Risk_on regime requires extra confirmation
        if regime_label == 'risk_on':
            funding_z = row.get('funding_Z', 0)
            if funding_z < 2.5:
                # In bull market (risk_on), need extreme funding to short
                return 'risk_on_insufficient_funding_veto'

        # VETO 5: ADX range check
        if adx < self.adx_min:
            # ADX too low = no trend, choppy market
            return 'adx_too_low_choppy_market_veto'

        return None


def detect_long_squeeze_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """Helper function for standalone detection."""
    archetype = LongSqueezeArchetype(config)
    return archetype.detect(row, regime_label)
