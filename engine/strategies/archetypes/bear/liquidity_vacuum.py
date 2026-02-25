#!/usr/bin/env python3
"""
Liquidity Vacuum Reversal (S1) - Bear Market Archetype

Production implementation for capitulation reversal detection during liquidity drains.

PATTERN LOGIC:
Capitulation reversals occur when orderbook liquidity evaporates during sell-offs,
creating "air pockets" where sellers exhaust themselves. The resulting vacuum creates
explosive short-covering bounces as there's no resistance.

KEY CHARACTERISTICS:
1. Extreme liquidity drain (liquidity_score < 0.15 OR drain_pct < -30%)
2. Panic volume spike (volume_zscore > 2.0)
3. Deep lower wick (wick_lower_ratio > 0.30) - sellers exhausted
4. Often occurs during crisis/capitulation events

DIRECTION: LONG (counter-trend reversal in bear markets)

TARGET: 10-15 trades/year, PF > 2.0

BTC EXAMPLES:
- 2022-06-18: Luna capitulation → -70% → violent 25% bounce
- 2022-11-09: FTX collapse → liquidity vacuum → explosive reversal
- 2022-05-12: LUNA death spiral → extreme capitulation → sharp bounce

Author: Claude Code (Backend Architect)
Date: 2026-01-08
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class LiquidityVacuumArchetype:
    """
    Liquidity Vacuum Reversal archetype - Production implementation.

    Detects capitulation reversals during extreme liquidity drains.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Liquidity Vacuum archetype.

        Args:
            config: Optional configuration dict with thresholds
        """
        self.config = config or {}
        thresholds = self.config.get('thresholds', {})

        # Core thresholds
        self.min_liquidity_drain_pct = thresholds.get('min_liquidity_drain_pct', -0.30)  # -30%
        self.min_volume_zscore = thresholds.get('min_volume_zscore', 2.0)
        self.min_wick_lower_ratio = thresholds.get('min_wick_lower_ratio', 0.30)
        self.min_fusion_score = thresholds.get('min_fusion_score', 0.40)

        # Domain weights (rebalanced to include Wyckoff)
        self.liquidity_weight = thresholds.get('liquidity_weight', 0.30)
        self.volume_weight = thresholds.get('volume_weight', 0.25)
        self.wick_weight = thresholds.get('wick_weight', 0.15)
        self.wyckoff_weight = thresholds.get('wyckoff_weight', 0.15)  # NEW: Wyckoff events
        self.crisis_weight = thresholds.get('crisis_weight', 0.10)
        self.smc_weight = thresholds.get('smc_weight', 0.05)

        logger.info(f"[S1 Liquidity Vacuum] Initialized with min_fusion={self.min_fusion_score}")

    def detect(
        self,
        row: pd.Series,
        regime_label: str = 'neutral'
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Detect Liquidity Vacuum Reversal pattern.

        Args:
            row: Current bar data with OHLCV + features
            regime_label: Current regime classification

        Returns:
            Tuple of (archetype_name, confidence_score, metadata)
            Returns (None, 0.0, {}) if no signal
        """
        # Step 1: Check liquidity drain
        liquidity_score = self._compute_liquidity_score(row)

        # Step 2: Check volume panic
        volume_score = self._compute_volume_score(row)

        # Step 3: Check wick rejection
        wick_score = self._compute_wick_score(row)

        # Step 4: Check Wyckoff events (NEW)
        wyckoff_score = self._compute_wyckoff_score(row)

        # Step 5: Check crisis context
        crisis_score = self._compute_crisis_score(row)

        # Step 6: Check SMC features
        smc_score = self._compute_smc_score(row)

        # Step 7: Compute weighted fusion score
        fusion_score = (
            self.liquidity_weight * liquidity_score +
            self.volume_weight * volume_score +
            self.wick_weight * wick_score +
            self.wyckoff_weight * wyckoff_score +
            self.crisis_weight * crisis_score +
            self.smc_weight * smc_score
        )

        # Step 7: Apply vetoes
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
            'liquidity_score': liquidity_score,
            'volume_score': volume_score,
            'wick_score': wick_score,
            'wyckoff_score': wyckoff_score,
            'crisis_score': crisis_score,
            'smc_score': smc_score,
            'fusion_score': fusion_score,
            'temporal_confluence': temporal_confluence,
            'temporal_mult': temporal_mult,
            'pattern_type': 'liquidity_vacuum_reversal_long'
        }

        return 'liquidity_vacuum', fusion_score, metadata

    def _compute_liquidity_score(self, row: pd.Series) -> float:
        """Compute liquidity drain score."""
        score = 0.0

        # Check liquidity drain percentage (V2 feature - relative drain)
        drain_pct = row.get('liquidity_drain_pct', 0.0)
        if drain_pct < self.min_liquidity_drain_pct:
            # Score increases with drain severity
            # -0.30 → 0.5, -0.50 → 0.8, -0.70 → 1.0
            score += min(1.0, abs(drain_pct) / 0.70)

        # Fallback: Check absolute liquidity score
        liquidity_abs = row.get('liquidity_score', 1.0)
        if liquidity_abs < 0.15:  # Extreme low liquidity
            score = max(score, 0.6)

        return min(1.0, score)

    def _compute_volume_score(self, row: pd.Series) -> float:
        """Compute volume panic score."""
        score = 0.0

        # Check volume z-score (try multiple possible column names for compatibility)
        volume_z = row.get('volume_zscore', row.get('volume_z', 0.0))  # FIX: Support both naming conventions
        if volume_z > self.min_volume_zscore:
            # Score increases with volume spike
            # z=2.0 → 0.5, z=3.0 → 0.75, z=4.0 → 1.0
            score = min(1.0, (volume_z - 2.0) / 2.0 + 0.5)

        return score

    def _compute_wick_score(self, row: pd.Series) -> float:
        """Compute wick rejection score."""
        score = 0.0

        # Check lower wick ratio (calculate on-the-fly if not in dataset)
        wick_lower = row.get('wick_lower_ratio', None)
        if wick_lower is None:
            # FIX: Calculate wick ratio from OHLCV if not pre-computed
            open_price = row.get('open', 0)
            close_price = row.get('close', 0)
            low_price = row.get('low', 0)

            body_bottom = min(open_price, close_price)
            lower_wick_length = body_bottom - low_price
            body_range = abs(close_price - open_price)

            # Calculate wick ratio relative to body
            if body_range > 0:
                wick_lower = lower_wick_length / body_range
            else:
                # If body_range is zero (doji), use wick length relative to low
                wick_lower = lower_wick_length / low_price if low_price > 0 else 0.0

        if wick_lower > self.min_wick_lower_ratio:
            # Score increases with wick size
            # 0.30 → 0.5, 0.50 → 0.8, 0.70 → 1.0
            score = min(1.0, (wick_lower - 0.30) / 0.40 + 0.5)

        return score

    def _compute_crisis_score(self, row: pd.Series) -> float:
        """Compute crisis context score."""
        score = 0.0

        # Check VIX (fear gauge)
        vix_z = row.get('VIX_Z', 0.0)
        if vix_z > 1.5:
            score += 0.3

        # Check DXY (dollar strength - risk-off)
        dxy_z = row.get('DXY_Z', 0.0)
        if dxy_z > 1.0:
            score += 0.2

        # Check funding (extreme negative = stress)
        funding_z = row.get('funding_Z', 0.0)
        if funding_z < -1.0:
            score += 0.2

        # Thermo-floor capitulation boost (BTC only)
        # Extreme capitulation = miners selling at loss = bottom signal
        symbol = row.get('symbol', 'BTCUSDT')
        if 'BTC' in symbol:
            # FIX: Use correct feature name from data
            thermo_distance = row.get('thermo_floor_distance', 0.0)
            # thermo_floor_distance = (price - floor) / floor
            # If negative, price is below floor
            if thermo_distance < -0.10:  # Price > 10% below mining cost
                # Extreme capitulation: miners forced to sell at loss
                # -0.10 → 0.3, -0.20 → 0.6, -0.30+ → 1.0
                capitulation_boost = min(1.0, abs(thermo_distance + 0.10) / 0.20)
                score += 0.3 * capitulation_boost

        return min(1.0, score)

    def _compute_wyckoff_score(self, row: pd.Series) -> float:
        """
        Compute Wyckoff domain engine score.

        For S1 (Liquidity Vacuum - long reversal), we look for:
        - SC (Selling Climax) - capitulation bottom
        - AR (Automatic Rally) - relief bounce after SC
        - Spring A/B - fake breakdown then reversal
        - LPS - Last Point of Support before markup

        Returns:
            Score 0.0-1.0
        """
        score = 0.0
        confidence_threshold = 0.70

        # Check for SC (Selling Climax) - perfect for liquidity vacuum
        sc = row.get('wyckoff_sc', False)
        sc_conf = row.get('wyckoff_sc_confidence', 0.0)
        if sc and sc_conf >= confidence_threshold:
            score += 0.40  # Highest weight - capitulation bottom

        # Check for AR (Automatic Rally) - relief bounce after capitulation
        ar = row.get('wyckoff_ar', False)
        ar_conf = row.get('wyckoff_ar_confidence', 0.0)
        if ar and ar_conf >= confidence_threshold:
            score += 0.30

        # Check for Spring A - fake breakdown with recovery
        spring_a = row.get('wyckoff_spring_a', False)
        spring_a_conf = row.get('wyckoff_spring_a_confidence', 0.0)
        if spring_a and spring_a_conf >= confidence_threshold:
            score += 0.35  # Strong reversal signal

        # Check for LPS - Last Point of Support
        lps = row.get('wyckoff_lps', False)
        lps_conf = row.get('wyckoff_lps_confidence', 0.0)
        if lps and lps_conf >= confidence_threshold:
            score += 0.25

        # Check Wyckoff phase (Phase A = capitulation/relief)
        phase = row.get('wyckoff_phase_abc', 'neutral')
        if phase == 'A':  # Selling climax / Automatic rally phase
            score += 0.15

        return min(1.0, score)

    def _compute_smc_score(self, row: pd.Series) -> float:
        """
        Compute SMC (Smart Money Concepts) score.

        For S1 (Liquidity Vacuum), we look for:
        - smc_supply_zone: Indicates overhead supply that was swept (bearish)
        - smc_liquidity_sweep: Confirms liquidity grab before reversal (bullish for long entry)

        Returns:
            Score 0.0-1.0
        """
        score = 0.0

        # Check for liquidity sweep (stop hunt complete = reversal setup)
        liq_sweep = row.get('smc_liquidity_sweep', False)
        if liq_sweep:
            score += 0.60  # Strong signal - stops hunted, ready to reverse

        # Check for supply zone overhead (resistance cleared after vacuum)
        supply_zone = row.get('smc_supply_zone', False)
        if supply_zone:
            score += 0.40  # Supply absorbed during vacuum = bullish

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
        # S1 is a LONG archetype - veto when bullish_trap detected
        # FIX: Use correct feature names from data (tf1h_pti_score, tf1h_pti_confidence)
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

        # Wyckoff vetoes for distribution tops (should not enter long)
        # Veto 1: BC (Buying Climax) - euphoria top
        bc = row.get('wyckoff_bc', False)
        bc_conf = row.get('wyckoff_bc_confidence', 0.0)
        if bc and bc_conf >= 0.70:
            return 'wyckoff_bc_euphoria_top'

        # Veto 2: UTAD (Upthrust After Distribution) - distribution top
        utad = row.get('wyckoff_utad', False)
        utad_conf = row.get('wyckoff_utad_confidence', 0.0)
        if utad and utad_conf >= 0.70:
            return 'wyckoff_utad_distribution_top'

        # Veto 3: SOW (Sign of Weakness) - bearish breakdown
        sow = row.get('wyckoff_sow', False)
        sow_conf = row.get('wyckoff_sow_confidence', 0.0)
        if sow and sow_conf >= 0.70:
            return 'wyckoff_sow_weakness_detected'

        # No other hard vetoes for S1 - designed to work in all regimes during capitulation
        return None


def detect_liquidity_vacuum_signal(
    row: pd.Series,
    regime_label: str = 'neutral',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Helper function for standalone detection.

    Args:
        row: Current bar data
        regime_label: Current regime
        config: Optional config

    Returns:
        Tuple of (archetype_name, confidence, metadata)
    """
    archetype = LiquidityVacuumArchetype(config)
    return archetype.detect(row, regime_label)
