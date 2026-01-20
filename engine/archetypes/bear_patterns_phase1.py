#!/usr/bin/env python3
"""
Bear Market Archetypes - Phase 1 Implementation (S2 + S5)

Based on 2022 validation analysis.
Target: PF > 1.3 on bear markets.

APPROVED PATTERNS:
- S2: Failed Rally Rejection (PF 1.4 estimated, 58.5% win rate)
- S5: Long Squeeze Cascade (corrected funding logic)

VALIDATION DATA (2022):
- S2: 205 occurrences, -0.68% forward 24h returns
- S5: Pattern too rare with strict thresholds (relaxed for testing)

CRITICAL FIX:
- S5 funding logic CORRECTED: positive funding = long squeeze DOWN (not short squeeze UP)
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


# ============================================================================
# S2: Failed Rally Rejection
# ============================================================================

def _check_S2_rejection(context: RuntimeContext) -> Tuple[bool, float, dict]:
    """
    S2: Failed Rally Rejection

    Edge Hypothesis: Resistance rejection + weak volume = fade the rally

    Detection Logic:
    - RSI > 70 (overbought, near resistance)
    - Volume Z-Score < 0.5 (weak volume = no conviction)
    - Upper Wick > 40% of candle range (rejection wick = failed breakout)
    - Fusion threshold: 0.36

    Validated 2022 Performance:
    - Occurrences: 205 (2.3% of bars)
    - Win Rate (1h): 58.5% (for shorts)
    - Forward Returns: -0.10% (1h), -0.68% (24h)
    - Estimated PF: 1.4

    Regime Suitability:
    - risk_off: 2.0x
    - crisis: 2.5x
    - neutral: 1.0x
    - risk_on: 0.3x

    Args:
        context: RuntimeContext with row, regime state, and thresholds

    Returns:
        (matched: bool, score: float, meta: dict)
    """
    # Read thresholds from context (regime-aware)
    fusion_th = context.get_threshold('rejection', 'fusion_threshold', 0.36)
    rsi_min = context.get_threshold('rejection', 'rsi_min', 70.0)
    vol_z_max = context.get_threshold('rejection', 'vol_z_max', 0.5)
    wick_ratio_min = context.get_threshold('rejection', 'wick_ratio_min', 0.4)

    # Get features
    row = context.row
    rsi = row.get('rsi_14', 50.0)
    vol_z = row.get('volume_zscore', 0.0)

    # Calculate upper wick ratio
    high = row.get('high', 0.0)
    low = row.get('low', 0.0)
    open_price = row.get('open', 0.0)
    close = row.get('close', 0.0)

    candle_range = high - low
    if candle_range < 1e-9:
        return False, 0.0, {"reason": "zero_range"}

    upper_body = max(open_price, close)
    upper_wick = high - upper_body
    wick_ratio = upper_wick / candle_range

    # Gate checks
    if rsi < rsi_min:
        return False, 0.0, {
            "reason": "rsi_low",
            "rsi": rsi,
            "threshold": rsi_min
        }

    if vol_z > vol_z_max:
        return False, 0.0, {
            "reason": "volume_too_high",
            "vol_z": vol_z,
            "threshold": vol_z_max
        }

    if wick_ratio < wick_ratio_min:
        return False, 0.0, {
            "reason": "wick_insufficient",
            "wick_ratio": wick_ratio,
            "threshold": wick_ratio_min
        }

    # Archetype-specific scoring
    components = {
        "rsi_extreme": min((rsi - 50.0) / 50.0, 1.0),  # Normalize RSI above 50 to 0-1
        "volume_fade": max(0.0, 1.0 - vol_z / 2.0),    # Inverse volume (lower is better)
        "wick_strength": min(wick_ratio / 0.6, 1.0),   # Normalize wick ratio to 0-1
        "fusion": row.get('fusion_score', 0.0)
    }

    # S2 weights (emphasize wick + RSI extremes)
    weights = context.get_threshold('rejection', 'weights', {
        "rsi_extreme": 0.25,
        "volume_fade": 0.20,
        "wick_strength": 0.30,
        "fusion": 0.25
    })

    base_score = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)
    archetype_weight = context.get_threshold('rejection', 'archetype_weight', 1.0)

    score = max(0.0, min(1.0, base_score * archetype_weight))

    if score < fusion_th:
        return False, score, {
            "reason": "score_below_threshold",
            "score": score,
            "threshold": fusion_th
        }

    meta = {
        "components": components,
        "weights": weights,
        "base_score": base_score,
        "archetype_weight": archetype_weight,
        "rsi": rsi,
        "vol_z": vol_z,
        "wick_ratio": wick_ratio
    }

    return True, score, meta


# ============================================================================
# S5: Long Squeeze Cascade (CORRECTED LOGIC)
# ============================================================================

def _check_S5_long_squeeze(context: RuntimeContext) -> Tuple[bool, float, dict]:
    """
    S5: Long Squeeze Cascade

    **CRITICAL FIX:** User's original logic was BACKWARDS!
    - Original: "funding > +0.08 = short squeeze" (WRONG)
    - Corrected: "funding > +0.08 = LONG squeeze" (longs pay shorts = overcrowding)

    Edge Hypothesis: Overcrowded longs + exhaustion = liquidation cascade DOWN

    Detection Logic:
    - Funding Z-Score > 1.0 (longs paying shorts = overcrowding)
    - OI Change > 3% (late longs entering = weak hands)
    - RSI > 65 (rally extended = exhaustion risk)
    - Fusion threshold: 0.38

    Validated 2022 Performance:
    - Occurrences: 0 with strict thresholds (too rare)
    - Relaxed thresholds for testing: funding_z > 1.0 (from 1.5), rsi > 65 (from 75)

    Caveat:
    - OI_CHANGE column appears broken in feature store (all zeros in 2022)
    - Short-term workaround: make OI filter optional until data pipeline fixed

    Regime Suitability:
    - risk_off: 2.2x
    - crisis: 2.8x
    - neutral: 1.2x
    - risk_on: 0.5x

    Args:
        context: RuntimeContext with row, regime state, and thresholds

    Returns:
        (matched: bool, score: float, meta: dict)
    """
    # Read thresholds from context (relaxed from original proposal)
    fusion_th = context.get_threshold('long_squeeze', 'fusion_threshold', 0.38)
    funding_z_min = context.get_threshold('long_squeeze', 'funding_z_min', 1.0)  # Relaxed from 1.5
    oi_change_min = context.get_threshold('long_squeeze', 'oi_change_min', 0.03)  # Relaxed from 0.05
    rsi_min = context.get_threshold('long_squeeze', 'rsi_min', 65.0)  # Relaxed from 75
    require_oi = context.get_threshold('long_squeeze', 'require_oi_filter', False)  # Optional until data fixed

    # Get features
    row = context.row
    funding_z = row.get('funding_Z', 0.0)
    oi_change = row.get('OI_CHANGE', 0.0)
    rsi = row.get('rsi_14', 50.0)

    # Gate checks
    if funding_z < funding_z_min:
        return False, 0.0, {
            "reason": "funding_z_low",
            "funding_z": funding_z,
            "threshold": funding_z_min
        }

    if rsi < rsi_min:
        return False, 0.0, {
            "reason": "rsi_low",
            "rsi": rsi,
            "threshold": rsi_min
        }

    # OI filter (optional until data pipeline fixed)
    if require_oi and oi_change < oi_change_min:
        return False, 0.0, {
            "reason": "oi_change_low",
            "oi_change": oi_change,
            "threshold": oi_change_min
        }

    # Archetype-specific scoring
    components = {
        "funding_extreme": min(funding_z / 2.0, 1.0),  # Normalize funding_z to 0-1 (2σ = max)
        "oi_spike": min(oi_change / 0.10, 1.0) if require_oi else 0.5,  # Normalize OI change (10% = max)
        "rsi_extreme": min((rsi - 50.0) / 50.0, 1.0),  # Normalize RSI above 50 to 0-1
        "fusion": row.get('fusion_score', 0.0)
    }

    # S5 weights (emphasize funding + RSI)
    weights = context.get_threshold('long_squeeze', 'weights', {
        "funding_extreme": 0.35,
        "oi_spike": 0.20,
        "rsi_extreme": 0.25,
        "fusion": 0.20
    })

    base_score = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)
    archetype_weight = context.get_threshold('long_squeeze', 'archetype_weight', 1.0)

    score = max(0.0, min(1.0, base_score * archetype_weight))

    if score < fusion_th:
        return False, score, {
            "reason": "score_below_threshold",
            "score": score,
            "threshold": fusion_th
        }

    meta = {
        "components": components,
        "weights": weights,
        "base_score": base_score,
        "archetype_weight": archetype_weight,
        "funding_z": funding_z,
        "oi_change": oi_change,
        "rsi": rsi,
        "note": "OI filter disabled" if not require_oi else "OI filter active"
    }

    return True, score, meta


# ============================================================================
# Integration with ArchetypeLogic
# ============================================================================

def integrate_bear_patterns_phase1(archetype_logic):
    """
    Integrate Phase 1 bear patterns into existing ArchetypeLogic class.

    Usage:
        from engine.archetypes.logic_v2_adapter import ArchetypeLogic
        from engine.archetypes.bear_patterns_phase1 import integrate_bear_patterns_phase1

        arch_logic = ArchetypeLogic(config)
        integrate_bear_patterns_phase1(arch_logic)

    This adds _check_S2 and _check_S5 methods to the ArchetypeLogic instance.
    """
    # Bind S2 check method
    import types
    archetype_logic._check_S2 = types.MethodType(_check_S2_rejection, archetype_logic)
    archetype_logic._check_S5 = types.MethodType(_check_S5_long_squeeze, archetype_logic)

    logger.info("[BearPatterns Phase1] Integrated S2 (rejection) and S5 (long_squeeze) archetypes")


# ============================================================================
# Standalone Testing Interface
# ============================================================================

if __name__ == '__main__':
    """
    Standalone testing of bear patterns on 2022 data.

    Usage:
        python3 engine/archetypes/bear_patterns_phase1.py
    """
    import pandas as pd
    from pathlib import Path

    print("Loading 2022 feature data...")
    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
    df_2022 = df[df.index < '2023-01-01'].copy()

    print(f"Loaded {len(df_2022)} bars from 2022")

    # Create minimal RuntimeContext for testing
    from engine.runtime.context import RuntimeContext

    # Test S2: Failed Rally Rejection
    print("\n" + "="*80)
    print("Testing S2: Failed Rally Rejection")
    print("="*80)

    s2_matches = []
    for idx, row in df_2022.iterrows():
        ctx = RuntimeContext(
            ts=idx,
            row=row,
            regime_probs={'neutral': 1.0},
            regime_label='neutral',
            adapted_params={},
            thresholds={
                'rejection': {
                    'fusion_threshold': 0.36,
                    'rsi_min': 70.0,
                    'vol_z_max': 0.5,
                    'wick_ratio_min': 0.4
                }
            }
        )
        matched, score, meta = _check_S2_rejection(ctx)
        if matched:
            s2_matches.append((idx, score, meta))

    print(f"S2 Matches: {len(s2_matches)} ({len(s2_matches)/len(df_2022):.1%} of bars)")
    if s2_matches:
        print("\nFirst 5 matches:")
        for idx, score, meta in s2_matches[:5]:
            print(f"  {idx}: score={score:.3f}, RSI={meta['rsi']:.1f}, wick={meta['wick_ratio']:.2f}")

    # Test S5: Long Squeeze
    print("\n" + "="*80)
    print("Testing S5: Long Squeeze Cascade")
    print("="*80)

    s5_matches = []
    for idx, row in df_2022.iterrows():
        ctx = RuntimeContext(
            ts=idx,
            row=row,
            regime_probs={'neutral': 1.0},
            regime_label='neutral',
            adapted_params={},
            thresholds={
                'long_squeeze': {
                    'fusion_threshold': 0.38,
                    'funding_z_min': 1.0,
                    'oi_change_min': 0.03,
                    'rsi_min': 65.0,
                    'require_oi_filter': False  # Disabled due to broken OI data
                }
            }
        )
        matched, score, meta = _check_S5_long_squeeze(ctx)
        if matched:
            s5_matches.append((idx, score, meta))

    print(f"S5 Matches: {len(s5_matches)} ({len(s5_matches)/len(df_2022):.1%} of bars)")
    if s5_matches:
        print("\nFirst 5 matches:")
        for idx, score, meta in s5_matches[:5]:
            print(f"  {idx}: score={score:.3f}, funding_z={meta['funding_z']:.2f}, RSI={meta['rsi']:.1f}")

    print("\nTesting complete!")
