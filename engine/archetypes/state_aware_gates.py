#!/usr/bin/env python3
"""
State-Aware Gate Surfaces for Bull Machine v2

Computes dynamic entry thresholds based on market conditions:
- ADX (trend strength)
- ATR percentile (volatility regime)
- Funding rate z-score (late-long risk)
- 4H trend alignment
- Regime (from GMM)

This prevents low-quality entries in chop/overfunded/thin conditions
while preserving high-quality setups in trending markets.

Traders' Logic:
- Moneytaur: "No fuel = no trade" (DXY up + funding + chop = skip)
- Zeroika: "Clarity = signal + context" (gates adapt to state)
- Wyckoff: "Preserve capital in distribution" (high-cost bars = expensive)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


class StateAwareGates:
    """
    Computes dynamic archetype entry gates based on market state.

    Gates adjust UP (more restrictive) when:
    - ADX < 18 (weak trend, chop)
    - ATR_pctile < 25 (low volatility, tight ranges)
    - funding_z > 1.0 (late-long risk, crowded)
    - tf4h trend misaligned (lower-TF noise)

    Gates adjust DOWN (more permissive) when:
    - ADX > 30 (strong trend)
    - ATR_pctile > 60 (good vol, clear moves)
    - funding_z < 0 (shorts covering, upside fuel)
    - tf4h trend aligned (higher-TF confirmation)
    """

    VERSION = "state_aware_gates@v1"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with config parameters.

        Args:
            config: Gate surface config with state adjustment weights
        """
        self.config = config
        gate_cfg = config.get('state_aware_gates', {})

        # State adjustment weights (how much each factor shifts the gate)
        self.weights = gate_cfg.get('weights', {
            'adx_weak_penalty': 0.06,       # +6% when ADX < 18
            'atr_low_penalty': 0.05,        # +5% when ATR_pctile < 25
            'funding_high_penalty': 0.05,   # +5% when funding_z > 1.0
            'tf4h_misalign_penalty': 0.03,  # +3% when 4H trend != 1H trend
            'adx_strong_bonus': -0.03,      # -3% when ADX > 30
            'funding_low_bonus': -0.02      # -2% when funding_z < 0
        })

        # Thresholds for state detection
        self.thresholds = gate_cfg.get('thresholds', {
            'adx_weak': 18.0,
            'adx_strong': 30.0,
            'atr_pctile_low': 25.0,
            'atr_pctile_good': 60.0,
            'funding_z_high': 1.0,
            'funding_z_low': 0.0
        })

        # Clamps (prevent gates from moving too far)
        self.max_adjustment = gate_cfg.get('max_adjustment', 0.15)  # Max ±15%
        self.min_gate = gate_cfg.get('min_gate', 0.25)  # Never go below 25%
        self.max_gate = gate_cfg.get('max_gate', 0.75)  # Never go above 75%

        logger.info(f"[StateAwareGates] Initialized {self.VERSION}")
        logger.info(f"  Adjustment weights: {self.weights}")
        logger.info(f"  State thresholds: {self.thresholds}")

    def compute_gate(
        self,
        archetype: str,
        base_gate: float,
        ctx: RuntimeContext,
        log_components: bool = False
    ) -> tuple[float, Dict[str, Any]]:
        """
        Compute dynamic gate for archetype based on market state.

        Args:
            archetype: Archetype name (e.g., 'trap_within_trend')
            base_gate: Base fusion threshold from config/regime
            ctx: RuntimeContext with market data
            log_components: If True, log gate adjustment details

        Returns:
            (adjusted_gate, components_dict)
        """
        row = ctx.row
        regime = ctx.regime_label

        # Initialize adjustment tracking
        adjustments = {}
        total_adjustment = 0.0

        # === State Feature Extraction ===

        # ADX (trend strength)
        adx = self._get_feature(row, ['adx', 'ADX_14'], 0.0)

        # ATR percentile (volatility regime)
        atr_pctile = self._get_feature(row, ['atr_percentile', 'atr_pctile'], 0.5) * 100

        # Funding rate z-score (late-long risk)
        funding_z = self._get_feature(row, ['funding_Z', 'funding_z'], 0.0)

        # 4H trend (higher-TF context)
        tf4h_trend = self._get_feature(row, ['tf4h_trend', '4h_trend'], 0)  # 1=up, -1=down, 0=neutral

        # 1H trend (current TF)
        tf1h_trend = self._get_feature(row, ['trend', 'tf1h_trend'], 0)

        # === State-Based Adjustments ===

        # 1. ADX Weak (chop/range)
        if adx < self.thresholds['adx_weak']:
            penalty = self.weights['adx_weak_penalty']
            adjustments['adx_weak'] = penalty
            total_adjustment += penalty

        # 2. ADX Strong (clear trend)
        elif adx > self.thresholds['adx_strong']:
            bonus = self.weights['adx_strong_bonus']
            adjustments['adx_strong'] = bonus
            total_adjustment += bonus

        # 3. ATR Low (tight ranges, avoid micro-scalps)
        if atr_pctile < self.thresholds['atr_pctile_low']:
            penalty = self.weights['atr_low_penalty']
            adjustments['atr_low'] = penalty
            total_adjustment += penalty

        # 4. Funding High (late-long risk, crowded trade)
        if funding_z > self.thresholds['funding_z_high']:
            penalty = self.weights['funding_high_penalty'] * min(funding_z / 2.0, 1.0)  # Scale with z-score
            adjustments['funding_high'] = penalty
            total_adjustment += penalty

        # 5. Funding Low (shorts covering, upside fuel)
        elif funding_z < self.thresholds['funding_z_low']:
            bonus = self.weights['funding_low_bonus']
            adjustments['funding_low'] = bonus
            total_adjustment += bonus

        # 6. 4H Trend Misalignment (lower-TF noise)
        if tf4h_trend != 0 and tf1h_trend != 0 and tf4h_trend != tf1h_trend:
            penalty = self.weights['tf4h_misalign_penalty']
            adjustments['tf4h_misalign'] = penalty
            total_adjustment += penalty

        # === Archetype-Specific Logic ===

        # trap_within_trend: extra strict in chop (it's a trend archetype!)
        if archetype == 'trap_within_trend' and adx < self.thresholds['adx_weak']:
            extra_penalty = 0.03  # +3% on top of base ADX penalty
            adjustments['trap_chop_extra'] = extra_penalty
            total_adjustment += extra_penalty

        # order_block_retest: extra strict when OB strength is weak
        elif archetype == 'order_block_retest':
            boms_strength = self._get_feature(row, ['boms_strength', 'ob_strength'], 0.5)
            if boms_strength < 0.4:
                extra_penalty = 0.04
                adjustments['ob_weak'] = extra_penalty
                total_adjustment += extra_penalty

        # volume_exhaustion: extra strict when volume is low
        elif archetype == 'volume_exhaustion':
            volume_z = self._get_feature(row, ['volume_z', 'vol_z'], 0.0)
            if volume_z < 0.3:
                extra_penalty = 0.04
                adjustments['ve_low_vol'] = extra_penalty
                total_adjustment += extra_penalty

        # === Regime Modulation ===

        # In risk_on, be slightly more permissive (bull archetypes get bonus)
        if regime == 'risk_on' and archetype not in ['breakdown', 'rejection', 'distribution', 'whipsaw']:
            regime_bonus = -0.02
            adjustments['regime_risk_on'] = regime_bonus
            total_adjustment += regime_bonus

        # In crisis, be VERY strict for bull archetypes (should already be suppressed by routing)
        elif regime == 'crisis' and archetype not in ['breakdown', 'rejection', 'distribution', 'whipsaw']:
            regime_penalty = 0.08
            adjustments['regime_crisis'] = regime_penalty
            total_adjustment += regime_penalty

        # === Apply Adjustments with Clamps ===

        # Clamp total adjustment
        total_adjustment = np.clip(total_adjustment, -self.max_adjustment, self.max_adjustment)

        # Compute final gate
        adjusted_gate = base_gate + total_adjustment

        # Hard clamps
        adjusted_gate = np.clip(adjusted_gate, self.min_gate, self.max_gate)

        # === Logging (if requested) ===

        components = {
            'archetype': archetype,
            'base_gate': base_gate,
            'adjusted_gate': adjusted_gate,
            'total_adjustment': total_adjustment,
            'adjustments': adjustments,
            'state': {
                'adx': adx,
                'atr_pctile': atr_pctile,
                'funding_z': funding_z,
                'tf4h_trend': tf4h_trend,
                'tf1h_trend': tf1h_trend,
                'regime': regime
            }
        }

        if log_components or abs(total_adjustment) > 0.05:
            logger.info(f"[StateGate] {archetype}: {base_gate:.3f} → {adjusted_gate:.3f} "
                       f"(Δ={total_adjustment:+.3f}) | ADX={adx:.1f} ATR%={atr_pctile:.0f} "
                       f"Fund_z={funding_z:+.2f} | {adjustments}")

        return adjusted_gate, components

    def _get_feature(self, row, keys: list, default=0.0):
        """Get first available feature from list of possible column names."""
        for key in keys:
            if key in row.index and row[key] is not None and not np.isnan(row[key]):
                return row[key]
        return default


def apply_state_aware_gate(
    archetype: str,
    base_gate: float,
    ctx: RuntimeContext,
    gate_module: Optional[StateAwareGates] = None,
    log_components: bool = False
) -> float:
    """
    Convenience function to apply state-aware gating.

    If gate_module is None, returns base_gate unchanged (backward compatible).

    Args:
        archetype: Archetype name
        base_gate: Base threshold from config/regime
        ctx: RuntimeContext
        gate_module: StateAwareGates instance (or None to disable)
        log_components: Log gate adjustments

    Returns:
        Adjusted gate threshold
    """
    if gate_module is None:
        return base_gate

    adjusted_gate, _ = gate_module.compute_gate(archetype, base_gate, ctx, log_components)
    return adjusted_gate
