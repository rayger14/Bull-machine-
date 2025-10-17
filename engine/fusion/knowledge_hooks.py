"""
Knowledge Architecture Fusion Hooks (v2.0)

Integrates Week 1-4 knowledge modules into the live fusion pipeline:
- Week 1: Structure (Internal/External, BOMS, Squiggle, Range Outcomes)
- Week 2: Psychology & Volume (PTI, FRVP, Fakeout Intensity)
- Week 4: Macro Echo (DXY/Oil/Yields/VIX correlations)

Each hook:
1. Reads features from feature store v2.0
2. Returns bounded (threshold_delta, score_delta, risk_multiplier) tuple
3. Includes conflict guards to avoid double-counting
4. Is idempotent (same inputs → same outputs)

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FusionDelta:
    """
    Fusion adjustment from a knowledge hook.

    Attributes:
        threshold_delta: Adjustment to entry threshold (±0.10 max)
        score_delta: Adjustment to fusion score (±0.15 max)
        risk_multiplier: Risk multiplier (0.0-1.5 range)
        reasons: List of human-readable reasons
        fired: True if hook triggered
    """
    threshold_delta: float = 0.0
    score_delta: float = 0.0
    risk_multiplier: float = 1.0
    reasons: List[str] = None
    fired: bool = False

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def apply_internal_external_delta(feats: Dict, config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply Internal vs External structure conflict adjustment.

    Args:
        feats: Feature dictionary with internal_external columns
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - Conflict > 0.75: Threshold +0.05, Score -0.03 (strong reversal warning)
        - Conflict > 0.60: Threshold +0.03, Score -0.02 (moderate warning)
        - Alignment + high confidence: Score +0.02 (confirmation)
    """
    config = config or {}
    delta = FusionDelta()

    conflict_score = feats.get('conflict_score', 0.0)
    structure_alignment = feats.get('structure_alignment', False)
    internal_strength = feats.get('internal_strength', 0.0)
    external_strength = feats.get('external_strength', 0.0)

    # Strong conflict (early reversal signal)
    if conflict_score > 0.75:
        delta.threshold_delta = 0.05
        delta.score_delta = -0.03
        delta.reasons.append(f"STRUCT_CONFLICT: Strong divergence ({conflict_score:.2f})")
        delta.fired = True

    # Moderate conflict
    elif conflict_score > 0.60:
        delta.threshold_delta = 0.03
        delta.score_delta = -0.02
        delta.reasons.append(f"STRUCT_CONFLICT: Moderate divergence ({conflict_score:.2f})")
        delta.fired = True

    # Strong alignment (confirmation)
    elif structure_alignment and internal_strength > 0.7 and external_strength > 0.7:
        delta.score_delta = 0.02
        delta.reasons.append(f"STRUCT_ALIGN: Strong alignment (I:{internal_strength:.2f} E:{external_strength:.2f})")
        delta.fired = True

    return delta


def apply_boms_boost(feats: Dict, config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply BOMS (Break of Market Structure) boost.

    Args:
        feats: Feature dictionary with BOMS columns
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - BOMS detected + volume surge > 2.0x: Score +0.10, Risk x1.2
        - BOMS detected + FVG present: Score +0.08
        - BOMS detected (standard): Score +0.05

    Conflict Guard:
        - Max one BOMS boost per bar (don't stack with plain BOS)
    """
    config = config or {}
    delta = FusionDelta()

    boms_detected = feats.get('boms_detected', False)
    boms_volume_surge = feats.get('boms_volume_surge', 1.0)
    boms_fvg_present = feats.get('boms_fvg_present', False)
    boms_direction = feats.get('boms_direction', 'none')

    if not boms_detected or boms_direction == 'none':
        return delta

    delta.fired = True

    # High-conviction BOMS (volume surge)
    if boms_volume_surge > 2.0:
        delta.score_delta = 0.10
        delta.risk_multiplier = 1.2
        delta.reasons.append(f"BOMS: High-conviction break ({boms_direction}, vol {boms_volume_surge:.1f}x)")

    # BOMS with Fair Value Gap
    elif boms_fvg_present:
        delta.score_delta = 0.08
        delta.reasons.append(f"BOMS: Break with FVG ({boms_direction})")

    # Standard BOMS
    else:
        delta.score_delta = 0.05
        delta.reasons.append(f"BOMS: Standard break ({boms_direction})")

    return delta


def apply_squiggle_window(feats: Dict, config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply 1-2-3 Squiggle pattern entry window adjustment.

    Args:
        feats: Feature dictionary with squiggle columns
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - Stage 2 (retest) + high quality (>0.8): Threshold -0.05 (entry window)
        - Stage 2 (retest) + medium quality: Threshold -0.03

    Conflict Guard:
        - Never stack with Range Breakout boost on same bar
    """
    config = config or {}
    delta = FusionDelta()

    squiggle_stage = feats.get('squiggle_stage', 0)
    squiggle_entry_window = feats.get('squiggle_entry_window', False)
    squiggle_confidence = feats.get('squiggle_confidence', 0.0)
    squiggle_direction = feats.get('squiggle_direction', 'none')

    # Only fire during Stage 2 (retest)
    if squiggle_stage != 2 or not squiggle_entry_window:
        return delta

    delta.fired = True

    # High-quality retest
    if squiggle_confidence > 0.8:
        delta.threshold_delta = -0.05
        delta.reasons.append(f"SQUIGGLE: High-quality retest ({squiggle_direction}, conf {squiggle_confidence:.2f})")

    # Medium-quality retest
    else:
        delta.threshold_delta = -0.03
        delta.reasons.append(f"SQUIGGLE: Retest entry window ({squiggle_direction})")

    return delta


def apply_range_outcome(feats: Dict, config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply Range Outcome classification adjustment.

    Args:
        feats: Feature dictionary with range outcome columns
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - Confirmed breakout: Score +0.08, Risk x1.15
        - Fakeout detected: Score -0.10, Threshold +0.08 (penalty)
        - Rejection: Score -0.05 (caution)

    Conflict Guard:
        - Mutex: only one of {breakout, fakeout, rejection} applies per bar
    """
    config = config or {}
    delta = FusionDelta()

    range_outcome = feats.get('range_outcome', 'none')
    range_outcome_confidence = feats.get('range_outcome_confidence', 0.0)
    breakout_strength = feats.get('breakout_strength', 0.0)
    volume_confirmation = feats.get('volume_confirmation', False)

    if range_outcome == 'none':
        return delta

    delta.fired = True

    # Confirmed breakout
    if range_outcome == 'breakout' and range_outcome_confidence > 0.7:
        delta.score_delta = 0.08
        if volume_confirmation and breakout_strength > 0.8:
            delta.risk_multiplier = 1.15
            delta.reasons.append(f"RANGE: Strong breakout (conf {range_outcome_confidence:.2f}, str {breakout_strength:.2f})")
        else:
            delta.reasons.append(f"RANGE: Confirmed breakout (conf {range_outcome_confidence:.2f})")

    # Fakeout detected (penalty)
    elif range_outcome == 'fakeout':
        delta.score_delta = -0.10
        delta.threshold_delta = 0.08
        delta.reasons.append(f"RANGE: Fakeout detected (conf {range_outcome_confidence:.2f})")

    # Rejection (caution)
    elif range_outcome == 'rejection':
        delta.score_delta = -0.05
        delta.reasons.append(f"RANGE: Rejection at boundary (conf {range_outcome_confidence:.2f})")

    return delta


def apply_pti(feats: Dict, applied_fakeout: bool, config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply PTI (Psychology Trap Index) adjustment.

    Args:
        feats: Feature dictionary with PTI columns
        applied_fakeout: True if fakeout penalty already applied
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - Same-direction trap (high score): Score -0.15, Threshold +0.05
        - Opposite-direction trap: Score +0.05 (fade the herd)
        - Reversal likely: Additional threshold +0.03

    Conflict Guard:
        - If fakeout already penalized, halve PTI same-direction penalty
    """
    config = config or {}
    delta = FusionDelta()

    pti_score = feats.get('pti_score', 0.0)
    pti_trap_type = feats.get('pti_trap_type', 'none')
    pti_reversal_likely = feats.get('pti_reversal_likely', False)
    pti_confidence = feats.get('pti_confidence', 0.0)

    if pti_trap_type == 'none' or pti_score < 0.4:
        return delta

    delta.fired = True

    # High-intensity trap (same direction as potential entry)
    if pti_score > 0.6 and pti_confidence > 0.6:
        penalty = -0.15 if not applied_fakeout else -0.075
        delta.score_delta = penalty
        delta.threshold_delta = 0.05
        delta.reasons.append(f"PTI: {pti_trap_type} detected (score {pti_score:.2f})")

        if pti_reversal_likely:
            delta.threshold_delta += 0.03
            delta.reasons.append("PTI: Reversal likely")

    # Moderate trap
    elif pti_score > 0.4:
        penalty = -0.08 if not applied_fakeout else -0.04
        delta.score_delta = penalty
        delta.reasons.append(f"PTI: Moderate trap ({pti_trap_type}, score {pti_score:.2f})")

    return delta


def apply_fakeout_intensity(feats: Dict, config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply Fakeout Intensity adjustment.

    Args:
        feats: Feature dictionary with fakeout columns
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - High intensity (>0.7): Score -0.25, Threshold +0.10 (strong penalty)
        - Medium intensity (>0.5): Score -0.15, Threshold +0.05
        - Fast return (<3 bars): Additional -0.05 penalty
    """
    config = config or {}
    delta = FusionDelta()

    fakeout_detected = feats.get('fakeout_detected', False)
    fakeout_intensity = feats.get('fakeout_intensity', 0.0)
    fakeout_return_speed = feats.get('fakeout_return_speed', 999)
    fakeout_direction = feats.get('fakeout_direction', 'none')

    if not fakeout_detected or fakeout_intensity < 0.5:
        return delta

    delta.fired = True

    # High-intensity fakeout
    if fakeout_intensity > 0.7:
        delta.score_delta = -0.25
        delta.threshold_delta = 0.10
        delta.risk_multiplier = 0.7
        delta.reasons.append(f"FAKEOUT: High intensity ({fakeout_direction}, {fakeout_intensity:.2f})")

    # Medium-intensity fakeout
    else:
        delta.score_delta = -0.15
        delta.threshold_delta = 0.05
        delta.risk_multiplier = 0.85
        delta.reasons.append(f"FAKEOUT: Medium intensity ({fakeout_direction}, {fakeout_intensity:.2f})")

    # Fast return penalty
    if fakeout_return_speed < 3:
        delta.score_delta -= 0.05
        delta.reasons.append(f"FAKEOUT: Fast return ({fakeout_return_speed} bars)")

    return delta


def apply_frvp(feats: Dict, current_price: float, hob_bonus_applied: bool,
               config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply FRVP (Fixed Range Volume Profile) adjustment.

    Args:
        feats: Feature dictionary with FRVP columns
        current_price: Current price
        hob_bonus_applied: True if HOB already gave liquidity bonus
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - Long from below VA: Score +0.05 (buying into value)
        - Short from above VA: Score +0.05 (selling into value)
        - Near POC: Score +0.03 (fair value)
        - Near LVN: Score -0.05 (gap risk)

    Conflict Guard:
        - If HOB already gave liquidity bonus, halve FRVP's bonus
    """
    config = config or {}
    delta = FusionDelta()

    frvp_current_position = feats.get('frvp_current_position', 'in_va')
    frvp_distance_to_poc = feats.get('frvp_distance_to_poc', 0.1)
    frvp_distance_to_va = feats.get('frvp_distance_to_va', 0.1)
    frvp_lvn_count = feats.get('frvp_lvn_count', 0)

    bonus_multiplier = 0.5 if hob_bonus_applied else 1.0

    # Below Value Area (buying low)
    if frvp_current_position == 'below_va':
        delta.score_delta = 0.05 * bonus_multiplier
        delta.reasons.append(f"FRVP: Below value area (POC dist {frvp_distance_to_poc:.1%})")
        delta.fired = True

    # Above Value Area (selling high)
    elif frvp_current_position == 'above_va':
        delta.score_delta = 0.05 * bonus_multiplier
        delta.reasons.append(f"FRVP: Above value area (POC dist {frvp_distance_to_poc:.1%})")
        delta.fired = True

    # Near POC (fair value)
    elif abs(frvp_distance_to_poc) < 0.02:
        delta.score_delta = 0.03 * bonus_multiplier
        delta.reasons.append("FRVP: Near POC (fair value)")
        delta.fired = True

    # Near LVN (gap risk)
    if frvp_lvn_count > 0 and abs(frvp_distance_to_va) < 0.03:
        delta.score_delta -= 0.05
        delta.reasons.append(f"FRVP: Near LVN gap ({frvp_lvn_count} nodes)")
        delta.fired = True

    return delta


def apply_macro_echo(feats: Dict, config: Optional[Dict] = None) -> FusionDelta:
    """
    Apply Macro Echo correlation adjustment.

    Args:
        feats: Feature dictionary with macro echo columns
        config: Optional configuration

    Returns:
        FusionDelta with adjustments

    Logic:
        - Crisis regime: Score -0.20, Risk x0.5 (strong warning)
        - Risk-off regime: Score -0.10, Risk x0.7
        - Risk-on regime: Score +0.05, Risk x1.1
        - Correlation score: Additional ±0.10 scaled adjustment

    Note:
        - This is separate from hard macro veto (which blocks all entries)
        - Treat as soft tilt based on traditional finance correlations
    """
    config = config or {}
    delta = FusionDelta()

    macro_regime = feats.get('macro_regime', 'neutral')
    macro_correlation_score = feats.get('macro_correlation_score', 0.0)
    macro_vix_level = feats.get('macro_vix_level', 'medium')
    macro_exit_recommended = feats.get('macro_exit_recommended', False)

    # Crisis regime (VIX extreme or DXY+Yields spiking)
    if macro_regime == 'crisis':
        delta.score_delta = -0.20
        delta.risk_multiplier = 0.5
        delta.reasons.append(f"MACRO: Crisis regime (VIX {macro_vix_level})")
        delta.fired = True

    # Risk-off (dollar strength = crypto weakness)
    elif macro_regime == 'risk_off':
        delta.score_delta = -0.10
        delta.risk_multiplier = 0.7
        delta.reasons.append(f"MACRO: Risk-off regime (corr {macro_correlation_score:.2f})")
        delta.fired = True

    # Risk-on (dollar weakness = crypto strength)
    elif macro_regime == 'risk_on':
        delta.score_delta = 0.05
        delta.risk_multiplier = 1.1
        delta.reasons.append(f"MACRO: Risk-on regime (corr {macro_correlation_score:.2f})")
        delta.fired = True

    # Correlation score adjustment (scaled to ±0.10)
    if abs(macro_correlation_score) > 0.3:
        corr_adjustment = macro_correlation_score * 0.10
        delta.score_delta += corr_adjustment
        if not delta.fired:
            delta.reasons.append(f"MACRO: Correlation score {macro_correlation_score:.2f}")
        delta.fired = True

    return delta


def apply_knowledge_hooks(
    fusion_score: float,
    feats: Dict,
    current_price: float,
    config: Optional[Dict] = None
) -> Tuple[float, float, float, List[str]]:
    """
    Apply all knowledge architecture hooks to fusion score and threshold.

    Args:
        fusion_score: Current fusion score (0-1)
        feats: Feature dictionary from feature store v2.0
        current_price: Current price
        config: Optional configuration

    Returns:
        (adjusted_score, threshold_delta, risk_multiplier, reasons)

    Evaluation Order (to minimize overlap):
        1. Structure: Internal/External → BOMS → Range Outcomes → Squiggle
        2. Psychology/Volume: PTI → Fakeout Intensity → FRVP
        3. Macro Echo (soft tilt)

    Safety:
        - Total threshold_delta ∈ [-0.10, +0.10]
        - Total score_delta ∈ [-0.30, +0.30]
        - Total risk_multiplier ∈ [0.5, 1.5]
    """
    config = config or {}

    # Check if knowledge v2 is enabled
    if not config.get('knowledge_v2', {}).get('enabled', False):
        return fusion_score, 0.0, 1.0, []

    total_threshold_delta = 0.0
    total_score_delta = 0.0
    total_risk_multiplier = 1.0
    all_reasons = []

    # Track conflicts
    hob_bonus_applied = feats.get('hob_score', 0.0) > 0.6  # HOB gave liquidity bonus
    fakeout_applied = False

    # 1. Structure hooks

    # Internal vs External
    d = apply_internal_external_delta(feats, config)
    if d.fired:
        total_threshold_delta += d.threshold_delta
        total_score_delta += d.score_delta
        all_reasons.extend(d.reasons)

    # BOMS
    d = apply_boms_boost(feats, config)
    if d.fired:
        total_score_delta += d.score_delta
        total_risk_multiplier *= d.risk_multiplier
        all_reasons.extend(d.reasons)

    # Range Outcomes
    d = apply_range_outcome(feats, config)
    if d.fired:
        total_threshold_delta += d.threshold_delta
        total_score_delta += d.score_delta
        total_risk_multiplier *= d.risk_multiplier
        all_reasons.extend(d.reasons)

        # Track if fakeout was applied
        if feats.get('range_outcome') == 'fakeout':
            fakeout_applied = True

    # Squiggle (unless range breakout just fired)
    if feats.get('range_outcome') != 'breakout':
        d = apply_squiggle_window(feats, config)
        if d.fired:
            total_threshold_delta += d.threshold_delta
            all_reasons.extend(d.reasons)

    # 2. Psychology/Volume hooks

    # Fakeout Intensity
    d = apply_fakeout_intensity(feats, config)
    if d.fired:
        total_threshold_delta += d.threshold_delta
        total_score_delta += d.score_delta
        total_risk_multiplier *= d.risk_multiplier
        all_reasons.extend(d.reasons)
        fakeout_applied = True

    # PTI (with fakeout conflict guard)
    d = apply_pti(feats, fakeout_applied, config)
    if d.fired:
        total_threshold_delta += d.threshold_delta
        total_score_delta += d.score_delta
        all_reasons.extend(d.reasons)

    # FRVP (with HOB conflict guard)
    d = apply_frvp(feats, current_price, hob_bonus_applied, config)
    if d.fired:
        total_score_delta += d.score_delta
        all_reasons.extend(d.reasons)

    # 3. Macro Echo (soft tilt)
    d = apply_macro_echo(feats, config)
    if d.fired:
        total_score_delta += d.score_delta
        total_risk_multiplier *= d.risk_multiplier
        all_reasons.extend(d.reasons)

    # Apply safety bounds
    total_threshold_delta = np.clip(total_threshold_delta, -0.10, 0.10)
    total_score_delta = np.clip(total_score_delta, -0.30, 0.30)
    total_risk_multiplier = np.clip(total_risk_multiplier, 0.5, 1.5)

    # Adjust fusion score
    adjusted_score = np.clip(fusion_score + total_score_delta, 0.0, 1.0)

    return adjusted_score, total_threshold_delta, total_risk_multiplier, all_reasons


def assert_feature_contract(df: pd.DataFrame, schema_version: str = "2.0"):
    """
    Assert that feature store contains all required Week 1-4 columns.

    Args:
        df: Feature store DataFrame
        schema_version: Expected schema version

    Raises:
        AssertionError: If required columns are missing
    """
    required_columns = [
        # Week 1: Structure
        'internal_phase', 'external_trend', 'structure_alignment', 'conflict_score',
        'internal_strength', 'external_strength',
        'boms_detected', 'boms_direction', 'boms_volume_surge', 'boms_fvg_present',
        'boms_confirmation', 'boms_break_level', 'boms_displacement',
        'squiggle_stage', 'squiggle_pattern_id', 'squiggle_direction', 'squiggle_entry_window',
        'squiggle_confidence', 'squiggle_bos_level', 'squiggle_retest_quality', 'squiggle_bars_since_bos',
        'range_outcome', 'range_outcome_direction', 'range_outcome_confidence',
        'range_high', 'range_low', 'breakout_strength', 'volume_confirmation', 'bars_in_range',

        # Week 2: Psychology & Volume
        'pti_score', 'pti_trap_type', 'pti_confidence', 'pti_reversal_likely',
        'pti_rsi_divergence', 'pti_volume_exhaustion', 'pti_wick_trap', 'pti_failed_breakout',
        'frvp_poc', 'frvp_va_high', 'frvp_va_low', 'frvp_hvn_count', 'frvp_lvn_count',
        'frvp_current_position', 'frvp_distance_to_poc', 'frvp_distance_to_va',
        'fakeout_detected', 'fakeout_intensity', 'fakeout_direction', 'fakeout_breakout_level',
        'fakeout_return_speed', 'fakeout_volume_weakness', 'fakeout_wick_rejection', 'fakeout_no_followthrough',

        # Week 4: Macro Echo
        'macro_regime', 'macro_dxy_trend', 'macro_yields_trend', 'macro_oil_trend',
        'macro_vix_level', 'macro_correlation_score', 'macro_exit_recommended',
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise AssertionError(
            f"Feature store schema v{schema_version} missing required columns:\n" +
            "\n".join(f"  - {col}" for col in missing[:10]) +
            (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
        )

    print(f"✅ Feature contract validated: {len(required_columns)} columns present (schema v{schema_version})")
