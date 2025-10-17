"""
Internal vs External Structure Detection

Distinguishes micro reversals (internal) from macro trend continuation (external).
Critical for detecting early reversals within larger trends.

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from engine.wyckoff.wyckoff_engine import detect_wyckoff_phase


@dataclass
class StructureState:
    """
    Nested structure state combining internal (micro) and external (macro) analysis.

    Attributes:
        internal_phase: Local structure ('accumulation', 'distribution', 'transition', 'markup', 'markdown')
        external_trend: HTF trend ('bullish', 'bearish', 'range')
        alignment: True if internal matches external direction
        conflict_score: 0-1, higher = more conflict (early reversal signal)
        internal_strength: 0-1 confidence in internal structure
        external_strength: 0-1 confidence in external trend
    """
    internal_phase: str
    external_trend: str
    alignment: bool
    conflict_score: float
    internal_strength: float
    external_strength: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'internal_phase': self.internal_phase,
            'external_trend': self.external_trend,
            'structure_alignment': self.alignment,
            'conflict_score': self.conflict_score,
            'internal_strength': self.internal_strength,
            'external_strength': self.external_strength
        }


def get_trend(df: pd.DataFrame, period: int = 50) -> tuple:
    """
    Determine trend direction using SMA.

    Args:
        df: OHLCV DataFrame
        period: SMA period (default 50 for external, 20 for internal)

    Returns:
        (trend: str, strength: float)
    """
    if len(df) < period:
        return 'neutral', 0.0

    sma = df['close'].rolling(period).mean()
    current_price = df['close'].iloc[-1]
    current_sma = sma.iloc[-1]

    # Calculate deviation
    deviation = (current_price - current_sma) / current_sma

    # Determine trend
    if deviation > 0.02:  # 2% above SMA
        trend = 'bullish'
        strength = min(abs(deviation) / 0.10, 1.0)  # Normalize to 0-1
    elif deviation < -0.02:  # 2% below SMA
        trend = 'bearish'
        strength = min(abs(deviation) / 0.10, 1.0)
    else:
        trend = 'range'
        strength = 1.0 - min(abs(deviation) / 0.02, 1.0)  # Higher when closer to SMA

    return trend, strength


def detect_bos_direction(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    Detect recent Break of Structure direction (internal structure).

    Args:
        df: OHLCV DataFrame
        lookback: Bars to look back for swing points

    Returns:
        (direction: str, strength: float)
    """
    if len(df) < lookback + 5:
        return 'neutral', 0.0

    # Find recent swing high/low
    highs = df['high'].rolling(lookback).max()
    lows = df['low'].rolling(lookback).min()

    recent_high = highs.iloc[-lookback:-1].max()
    recent_low = lows.iloc[-lookback:-1].min()

    current_close = df['close'].iloc[-1]

    # Check for BOS
    if current_close > recent_high:
        # Bullish BOS
        displacement = (current_close - recent_high) / recent_high
        strength = min(displacement / 0.05, 1.0)  # 5% displacement = full strength
        return 'bullish', strength
    elif current_close < recent_low:
        # Bearish BOS
        displacement = (recent_low - current_close) / recent_low
        strength = min(displacement / 0.05, 1.0)
        return 'bearish', strength
    else:
        return 'neutral', 0.0


def map_wyckoff_to_phase(wyckoff_result: Dict) -> tuple:
    """
    Map Wyckoff phase to simplified internal phase.

    Returns:
        (phase: str, strength: float)
    """
    phase = wyckoff_result.get('phase', 'neutral')
    confidence = wyckoff_result.get('confidence', 0.0)

    # Bullish phases
    if phase in ['accumulation', 'spring', 'reaccumulation']:
        return 'accumulation', confidence
    elif phase == 'markup':
        return 'markup', confidence

    # Bearish phases
    elif phase in ['distribution', 'upthrust', 'redistribution']:
        return 'distribution', confidence
    elif phase == 'markdown':
        return 'markdown', confidence

    # Neutral
    else:
        return 'transition', confidence


def detect_structure_state(df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                           df_1d: pd.DataFrame, config: Optional[Dict] = None) -> StructureState:
    """
    Detect nested structure state (internal micro vs external macro).

    This is the main entry point for internal/external structure analysis.

    Args:
        df_1h: 1H OHLCV data
        df_4h: 4H OHLCV data
        df_1d: 1D OHLCV data
        config: Optional configuration dictionary

    Returns:
        StructureState with full nested analysis

    Logic:
        1. External (1D): Wyckoff phase + trend SMA50
        2. Internal (1H/4H): BOS direction + local Wyckoff
        3. Alignment: Check if internal matches external
        4. Conflict: Measure divergence (early reversal signal)

    Example:
        >>> # 1D in distribution, 1H shows accumulation = conflict (potential reversal)
        >>> state = detect_structure_state(df_1h, df_4h, df_1d)
        >>> state.conflict_score  # High = early reversal likely
        0.75
    """
    config = config or {}

    # 1. EXTERNAL STRUCTURE (1D - Macro trend)
    wyckoff_1d = detect_wyckoff_phase(df_1d, config, usdt_stag_strength=0.5)
    external_phase, wyck_strength = map_wyckoff_to_phase(wyckoff_1d)
    trend_1d, trend_strength = get_trend(df_1d, period=50)

    # Combine Wyckoff and trend for external strength
    external_strength = (wyck_strength + trend_strength) / 2.0

    # Determine external trend
    if external_phase in ['accumulation', 'markup'] or trend_1d == 'bullish':
        external_trend = 'bullish'
    elif external_phase in ['distribution', 'markdown'] or trend_1d == 'bearish':
        external_trend = 'bearish'
    else:
        external_trend = 'range'

    # 2. INTERNAL STRUCTURE (1H/4H - Micro patterns)
    # Use 4H for internal structure (balance between noise and responsiveness)
    bos_direction, bos_strength = detect_bos_direction(df_4h, lookback=20)
    wyckoff_4h = detect_wyckoff_phase(df_4h, config, usdt_stag_strength=0.5)
    internal_phase_raw, internal_wyck_strength = map_wyckoff_to_phase(wyckoff_4h)

    # Combine BOS and Wyckoff for internal
    internal_strength = (bos_strength + internal_wyck_strength) / 2.0

    # Determine internal phase (BOS takes precedence if strong)
    if bos_strength > 0.6:
        internal_phase = 'markup' if bos_direction == 'bullish' else 'markdown'
    else:
        internal_phase = internal_phase_raw

    # 3. ALIGNMENT CHECK
    # Bullish alignment: Both bullish
    # Bearish alignment: Both bearish
    # Conflict: Opposing directions

    bullish_external = external_trend == 'bullish'
    bearish_external = external_trend == 'bearish'
    bullish_internal = internal_phase in ['accumulation', 'markup'] or bos_direction == 'bullish'
    bearish_internal = internal_phase in ['distribution', 'markdown'] or bos_direction == 'bearish'

    alignment = (
        (bullish_external and bullish_internal) or
        (bearish_external and bearish_internal)
    )

    # 4. CONFLICT SCORE (0-1)
    # High conflict = strong internal structure opposite to external trend
    # This is an early reversal signal
    if not alignment:
        # Conflict magnitude = internal strength × (1 - external weakness)
        # Strong internal against weak external = high conflict
        conflict_score = internal_strength * (1.0 - (external_strength * 0.5))
    else:
        # Aligned structures have low conflict
        conflict_score = 0.0

    conflict_score = float(np.clip(conflict_score, 0, 1))

    return StructureState(
        internal_phase=internal_phase,
        external_trend=external_trend,
        alignment=alignment,
        conflict_score=conflict_score,
        internal_strength=float(internal_strength),
        external_strength=float(external_strength)
    )


def apply_structure_fusion_adjustment(fusion_score: float, structure_state: StructureState,
                                       config: Optional[Dict] = None) -> tuple:
    """
    Adjust fusion score based on internal/external structure alignment.

    Args:
        fusion_score: Current fusion score (0-1)
        structure_state: StructureState from detect_structure_state()
        config: Optional config with thresholds

    Returns:
        (adjusted_score: float, threshold_adjustment: float, reasons: list)

    Logic:
        - Alignment=True: No penalty
        - Conflict > 0.6: Threshold +0.05 (require higher confluence for entry)
        - Conflict > 0.75: Additional threshold +0.03 (very high bar)
    """
    config = config or {}
    conflict_threshold = config.get('structure_conflict_threshold', 0.6)
    high_conflict_threshold = config.get('structure_high_conflict_threshold', 0.75)

    threshold_adjustment = 0.0
    reasons = []

    if structure_state.alignment:
        reasons.append(f"Structure aligned: {structure_state.internal_phase}/{structure_state.external_trend}")
    else:
        # Conflict detected
        if structure_state.conflict_score > high_conflict_threshold:
            # Very high conflict - major divergence
            threshold_adjustment = +0.08
            reasons.append(f"HIGH conflict: internal={structure_state.internal_phase}, "
                          f"external={structure_state.external_trend}, score={structure_state.conflict_score:.2f}")
        elif structure_state.conflict_score > conflict_threshold:
            # Moderate conflict
            threshold_adjustment = +0.05
            reasons.append(f"Moderate conflict: internal={structure_state.internal_phase}, "
                          f"external={structure_state.external_trend}, score={structure_state.conflict_score:.2f}")
        else:
            # Low conflict - just note it
            reasons.append(f"Minor structure divergence: {structure_state.conflict_score:.2f}")

    return fusion_score, threshold_adjustment, reasons
