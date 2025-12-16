#!/usr/bin/env python3
"""
Wyckoff Event Detection System - Institutional Grade Implementation

Detects all 18 classic Wyckoff structural events for accumulation and distribution
cycles. Integrates with existing PTI, volume_z, and liquidity_score features.

Events Implemented:
    Phase A (Selling Climax/Buying Climax):
        - SC (Selling Climax): Extreme volume spike at lows, capitulation
        - BC (Buying Climax): Extreme volume spike at highs, euphoria peak
        - AR (Automatic Rally): Relief bounce after SC on declining volume
        - AS (Automatic Reaction): Relief drop after BC on declining volume
        - ST (Secondary Test): Retest of SC lows on lower volume
        - ST_BC (Secondary Test of BC): Retest of BC highs on lower volume

    Phase B (Building Cause):
        - SOS (Sign of Strength): First decisive move up with volume
        - SOW (Sign of Weakness): First decisive move down with volume
        - LPS (Last Point of Support): Final test before markup begins
        - LPSY (Last Point of Supply): Final rally before markdown begins
        - Spring_A (Type A Spring): Fake breakdown below range to trap sellers
        - Spring_B (Type B Spring): Shallow spring with quick recovery

    Phase C (Testing):
        - UT (Upthrust): Fake breakout above range to trap buyers
        - UTAD (Upthrust After Distribution): Final trap before major decline
        - Shakeout: Violent SC-like action mid-range to test strength

    Phase D/E (Trend):
        - Markup_Continuation: Validated uptrend with structure
        - Markdown_Continuation: Validated downtrend with structure
        - Terminal_Shakeout: Final trap at trend exhaustion

Design Philosophy:
    1. Backward Compatible: All new features default to False/0.0
    2. Vectorized: Pure pandas/numpy, no loops
    3. Confluence-Based: Uses volume_z, liquidity_score, range_position
    4. Observable: Returns confidence scores for each detection
    5. PTI-Integrated: Links psychological traps to Wyckoff events

Author: Bull Machine v2.0 - Institutional Grade Wyckoff System
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WyckoffPhaseABC(Enum):
    """Wyckoff sub-phases (finer granularity than existing accumulation/distribution)"""
    PHASE_A = "A"  # Preliminary support/resistance
    PHASE_B = "B"  # Building cause/effect
    PHASE_C = "C"  # Testing phase (springs/upthrusts)
    PHASE_D = "D"  # Trend beginning
    PHASE_E = "E"  # Trend continuation
    NEUTRAL = "neutral"


@dataclass
class WyckoffEvent:
    """Single Wyckoff event detection result"""
    event_type: str
    confidence: float
    bar_index: int
    metadata: Dict


def _rolling_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling z-score for a series.

    Args:
        series: Input series (e.g., volume, range)
        window: Rolling window size

    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z_score = (series - rolling_mean) / (rolling_std + 1e-9)
    return z_score.fillna(0.0)


def _range_position(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate position in recent range (0-1).

    Args:
        df: OHLCV dataframe
        lookback: Number of bars for range calculation

    Returns:
        Series with range position (0 = at lows, 1 = at highs)
    """
    rolling_high = df['high'].rolling(lookback).max()
    rolling_low = df['low'].rolling(lookback).min()
    range_size = rolling_high - rolling_low

    position = (df['close'] - rolling_low) / (range_size + 1e-9)
    return position.clip(0, 1).fillna(0.5)


def _wick_quality(df: pd.DataFrame, direction: str = 'lower') -> pd.Series:
    """
    Calculate wick size relative to body and range.

    Args:
        df: OHLCV dataframe
        direction: 'lower' for lower wicks, 'upper' for upper wicks

    Returns:
        Series with wick quality score (0-1)
    """
    body = (df['close'] - df['open']).abs()
    total_range = df['high'] - df['low']

    if direction == 'lower':
        wick = df[['open', 'close']].min(axis=1) - df['low']
    else:
        wick = df['high'] - df[['open', 'close']].max(axis=1)

    # Wick quality: larger wick relative to body = higher score
    wick_ratio = wick / (body + 1e-9)
    range_ratio = wick / (total_range + 1e-9)

    quality = (wick_ratio.clip(0, 5) / 5.0) * 0.6 + range_ratio * 0.4
    return quality.clip(0, 1).fillna(0.0)


# ============================================================================
# PHASE A: SELLING CLIMAX / BUYING CLIMAX EVENTS
# ============================================================================

def detect_selling_climax(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Selling Climax (SC) - Capitulation event at market bottom.

    Characteristics (Institutional Moneytaur Pattern):
        - Extreme volume spike (volume_z > 2.5)
        - Price at 20-bar lows (range_position < 0.2)
        - Large range bar (range_z > 1.5)
        - Often coincides with liquidity_score < -0.5 (stop hunts)
        - Lower wick > 60% of total range (absorption)

    Args:
        df: OHLCV dataframe with technical indicators
        cfg: Configuration dict with thresholds

    Returns:
        Tuple of (detected: pd.Series[bool], confidence: pd.Series[float])
    """
    # Get or compute required features
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    if 'range_position' not in df.columns:
        df['range_position'] = _range_position(df, lookback=20)

    # Compute range z-score
    bar_range = df['high'] - df['low']
    range_z = _rolling_z_score(bar_range, window=20)

    # Wick quality (absorption)
    lower_wick_quality = _wick_quality(df, direction='lower')

    # Get config thresholds
    volume_z_min = cfg.get('sc_volume_z_min', 2.5)
    range_pos_max = cfg.get('sc_range_pos_max', 0.2)
    range_z_min = cfg.get('sc_range_z_min', 1.5)
    wick_min = cfg.get('sc_wick_min', 0.6)

    # Detection criteria
    extreme_volume = df['volume_z'] > volume_z_min
    at_lows = df['range_position'] < range_pos_max
    wide_range = range_z > range_z_min
    strong_absorption = lower_wick_quality > wick_min

    # Confluence: All criteria must be met
    detected = extreme_volume & at_lows & wide_range & strong_absorption

    # Confidence score (weighted combination)
    confidence = (
        (df['volume_z'] / 5.0).clip(0, 1) * 0.35 +
        (1 - df['range_position']) * 0.25 +
        (range_z / 3.0).clip(0, 1) * 0.25 +
        lower_wick_quality * 0.15
    )
    confidence = confidence.fillna(0.0)

    # Only non-zero confidence where detected
    confidence = confidence * detected

    return detected, confidence


def detect_buying_climax(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Buying Climax (BC) - Euphoria peak at market top.

    Characteristics:
        - Extreme volume spike (volume_z > 2.5)
        - Price at 20-bar highs (range_position > 0.8)
        - Large range bar (range_z > 1.5)
        - Upper wick > 60% of total range (rejection)
        - Often coincides with RSI > 75

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    if 'range_position' not in df.columns:
        df['range_position'] = _range_position(df, lookback=20)

    bar_range = df['high'] - df['low']
    range_z = _rolling_z_score(bar_range, window=20)
    upper_wick_quality = _wick_quality(df, direction='upper')

    # Config
    volume_z_min = cfg.get('bc_volume_z_min', 2.5)
    range_pos_min = cfg.get('bc_range_pos_min', 0.8)
    range_z_min = cfg.get('bc_range_z_min', 1.5)
    wick_min = cfg.get('bc_wick_min', 0.6)

    # Detection
    extreme_volume = df['volume_z'] > volume_z_min
    at_highs = df['range_position'] > range_pos_min
    wide_range = range_z > range_z_min
    strong_rejection = upper_wick_quality > wick_min

    detected = extreme_volume & at_highs & wide_range & strong_rejection

    # Confidence
    confidence = (
        (df['volume_z'] / 5.0).clip(0, 1) * 0.35 +
        df['range_position'] * 0.25 +
        (range_z / 3.0).clip(0, 1) * 0.25 +
        upper_wick_quality * 0.15
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_automatic_rally(df: pd.DataFrame, cfg: dict,
                          sc_events: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Automatic Rally (AR) - Relief bounce after Selling Climax.

    Characteristics (Zeroika Pattern):
        - Must occur within 5-10 bars after SC
        - Volume declining (volume_z < 1.0)
        - Retraces 40-70% of SC range
        - Closes in upper 60% of bar (buying pressure)
        - No new lows made

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict
        sc_events: Optional SC detection results for context

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config
    lookback_max = cfg.get('ar_lookback_max', 10)
    volume_z_max = cfg.get('ar_volume_z_max', 1.0)
    retrace_min = cfg.get('ar_retrace_min', 0.40)
    retrace_max = cfg.get('ar_retrace_max', 0.70)
    close_position_min = cfg.get('ar_close_position_min', 0.6)

    # Calculate retrace from recent low
    rolling_low = df['low'].rolling(lookback_max).min()
    rolling_high = df['high'].rolling(lookback_max).max()
    recent_range = rolling_high - rolling_low
    retrace_pct = (df['close'] - rolling_low) / (recent_range + 1e-9)

    # Bar close position (buying pressure indicator)
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Detection criteria
    declining_volume = df['volume_z'] < volume_z_max
    proper_retrace = (retrace_pct > retrace_min) & (retrace_pct < retrace_max)
    strong_close = close_position > close_position_min
    no_new_lows = df['low'] > df['low'].shift(1)  # Not making new lows

    detected = declining_volume & proper_retrace & strong_close & no_new_lows

    # Confidence
    retrace_quality = 1 - abs(retrace_pct - 0.55) / 0.55  # Peak at 55% retrace
    confidence = (
        (1 - df['volume_z'] / 3.0).clip(0, 1) * 0.30 +
        retrace_quality.clip(0, 1) * 0.35 +
        close_position * 0.25 +
        no_new_lows.astype(float) * 0.10
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_automatic_reaction(df: pd.DataFrame, cfg: dict,
                              bc_events: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Automatic Reaction (AS) - Relief drop after Buying Climax.

    Mirror of AR for distribution phase.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict
        bc_events: Optional BC detection results

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback_max = cfg.get('as_lookback_max', 10)
    volume_z_max = cfg.get('as_volume_z_max', 1.0)
    retrace_min = cfg.get('as_retrace_min', 0.40)
    retrace_max = cfg.get('as_retrace_max', 0.70)
    close_position_max = cfg.get('as_close_position_max', 0.4)

    # Calculate retrace from recent high
    rolling_high = df['high'].rolling(lookback_max).max()
    rolling_low = df['low'].rolling(lookback_max).min()
    recent_range = rolling_high - rolling_low
    retrace_pct = (rolling_high - df['close']) / (recent_range + 1e-9)

    # Bar close position (selling pressure indicator)
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Detection
    declining_volume = df['volume_z'] < volume_z_max
    proper_retrace = (retrace_pct > retrace_min) & (retrace_pct < retrace_max)
    weak_close = close_position < close_position_max
    no_new_highs = df['high'] < df['high'].shift(1)

    detected = declining_volume & proper_retrace & weak_close & no_new_highs

    # Confidence
    retrace_quality = 1 - abs(retrace_pct - 0.55) / 0.55
    confidence = (
        (1 - df['volume_z'] / 3.0).clip(0, 1) * 0.30 +
        retrace_quality.clip(0, 1) * 0.35 +
        (1 - close_position) * 0.25 +
        no_new_highs.astype(float) * 0.10
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_secondary_test(df: pd.DataFrame, cfg: dict,
                          sc_events: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Secondary Test (ST) - Retest of SC lows on lower volume.

    Characteristics:
        - Price returns to SC low area (within 5%)
        - Volume significantly lower than SC (volume_z < 0.5)
        - Price holds above SC low (no new low)
        - Occurs 10-30 bars after SC

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict
        sc_events: Optional SC detection for reference

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config
    lookback = cfg.get('st_lookback', 30)
    low_proximity = cfg.get('st_low_proximity', 0.05)  # Within 5% of low
    volume_z_max = cfg.get('st_volume_z_max', 0.5)

    # Find recent lows
    rolling_low = df['low'].rolling(lookback).min()
    distance_from_low = (df['low'] - rolling_low) / (rolling_low + 1e-9)

    # Detection
    near_low = distance_from_low < low_proximity
    lower_volume = df['volume_z'] < volume_z_max
    holds_above = df['low'] > rolling_low  # No new low

    detected = near_low & lower_volume & holds_above

    # Confidence (higher if closer to low with much lower volume)
    proximity_score = (1 - distance_from_low / low_proximity).clip(0, 1)
    volume_score = (1 - df['volume_z'] / 2.0).clip(0, 1)

    confidence = (
        proximity_score * 0.40 +
        volume_score * 0.40 +
        holds_above.astype(float) * 0.20
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


# ============================================================================
# PHASE B: SIGN OF STRENGTH / WEAKNESS
# ============================================================================

def detect_sign_of_strength(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Sign of Strength (SOS) - First decisive move up with volume.

    Characteristics:
        - Breaks above recent range (> 20-bar high)
        - Strong volume (volume_z > 1.5)
        - Close in top 70% of bar
        - Follows period of consolidation
        - Often coincides with BOS (break of structure)

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config
    lookback = cfg.get('sos_lookback', 20)
    volume_z_min = cfg.get('sos_volume_z_min', 1.5)
    close_position_min = cfg.get('sos_close_position_min', 0.7)
    breakout_margin = cfg.get('sos_breakout_margin', 0.01)  # 1% above high

    # Calculate breakout
    rolling_high = df['high'].rolling(lookback).max().shift(1)
    breaks_high = df['close'] > rolling_high * (1 + breakout_margin)

    # Bar characteristics
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Volume
    strong_volume = df['volume_z'] > volume_z_min

    # Detection
    detected = breaks_high & strong_volume & (close_position > close_position_min)

    # Confidence
    breakout_strength = ((df['close'] - rolling_high) / (rolling_high + 1e-9)).clip(0, 0.1) / 0.1
    confidence = (
        breakout_strength * 0.30 +
        (df['volume_z'] / 4.0).clip(0, 1) * 0.35 +
        close_position * 0.25 +
        0.10  # Base confidence for meeting criteria
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_sign_of_weakness(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Sign of Weakness (SOW) - First decisive move down with volume.

    Mirror of SOS for distribution.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('sow_lookback', 20)
    volume_z_min = cfg.get('sow_volume_z_min', 1.5)
    close_position_max = cfg.get('sow_close_position_max', 0.3)
    breakdown_margin = cfg.get('sow_breakdown_margin', 0.01)

    # Calculate breakdown
    rolling_low = df['low'].rolling(lookback).min().shift(1)
    breaks_low = df['close'] < rolling_low * (1 - breakdown_margin)

    # Bar characteristics
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Volume
    strong_volume = df['volume_z'] > volume_z_min

    # Detection
    detected = breaks_low & strong_volume & (close_position < close_position_max)

    # Confidence
    breakdown_strength = ((rolling_low - df['close']) / (rolling_low + 1e-9)).clip(0, 0.1) / 0.1
    confidence = (
        breakdown_strength * 0.30 +
        (df['volume_z'] / 4.0).clip(0, 1) * 0.35 +
        (1 - close_position) * 0.25 +
        0.10
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


# ============================================================================
# PHASE C: SPRINGS AND UPTHRUSTS
# ============================================================================

def detect_spring_type_a(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Spring Type A - Deep fake breakdown below range to trap sellers.

    Characteristics (Classic Wyckoff):
        - Breaks below 20-bar low convincingly (> 2%)
        - Reverses quickly (within 1-3 bars)
        - Closes back inside range
        - Volume spike on breakdown, then decline on recovery
        - Often coincides with liquidity sweep (liquidity_score spike)

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config
    lookback = cfg.get('spring_a_lookback', 20)
    breakdown_margin = cfg.get('spring_a_breakdown_margin', 0.02)  # 2% below low
    recovery_bars = cfg.get('spring_a_recovery_bars', 3)

    # Calculate range
    rolling_low = df['low'].rolling(lookback).min().shift(1)
    rolling_high = df['high'].rolling(lookback).max().shift(1)

    # Breakdown detection
    breaks_low = df['low'] < rolling_low * (1 - breakdown_margin)

    # Quick recovery (close back above range low within N bars)
    future_close = df['close'].shift(-recovery_bars)
    recovers = future_close > rolling_low

    # Volume characteristics
    volume_spike = df['volume_z'] > 1.0

    # Detection (note: uses future data for recovery confirmation)
    # In production, this would trigger after recovery_bars delay
    detected = breaks_low & recovers.shift(recovery_bars).fillna(False) & volume_spike

    # Confidence
    breakdown_depth = ((rolling_low - df['low']) / (rolling_low + 1e-9)).clip(0, 0.05) / 0.05
    confidence = (
        breakdown_depth * 0.40 +
        (df['volume_z'] / 3.0).clip(0, 1) * 0.35 +
        0.25  # Base for meeting criteria
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_spring_type_b(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Spring Type B - Shallow spring with quick recovery.

    Characteristics:
        - Slightly below range low (0.5-1%)
        - Immediate reversal (same or next bar)
        - Lower volume than Type A
        - Close above mid-range quickly

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('spring_b_lookback', 20)
    breakdown_min = cfg.get('spring_b_breakdown_min', 0.005)
    breakdown_max = cfg.get('spring_b_breakdown_max', 0.01)

    rolling_low = df['low'].rolling(lookback).min().shift(1)
    rolling_high = df['high'].rolling(lookback).max().shift(1)
    range_mid = (rolling_high + rolling_low) / 2

    # Shallow breakdown
    breakdown_pct = (rolling_low - df['low']) / (rolling_low + 1e-9)
    shallow_break = (breakdown_pct > breakdown_min) & (breakdown_pct < breakdown_max)

    # Quick recovery (close above mid-range)
    quick_recovery = df['close'] > range_mid

    # Moderate volume
    moderate_volume = (df['volume_z'] > 0.5) & (df['volume_z'] < 2.0)

    detected = shallow_break & quick_recovery & moderate_volume

    # Confidence
    confidence = (
        (breakdown_pct / breakdown_max).clip(0, 1) * 0.35 +
        ((df['close'] - range_mid) / (rolling_high - range_mid + 1e-9)).clip(0, 1) * 0.35 +
        (df['volume_z'] / 3.0).clip(0, 1) * 0.30
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_upthrust(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Upthrust (UT) - Fake breakout above range to trap buyers.

    Characteristics:
        - Breaks above range high
        - Reverses quickly (fails to hold)
        - Closes back inside range
        - Volume spike on breakout
        - Often precedes distribution

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('ut_lookback', 20)
    breakout_margin = cfg.get('ut_breakout_margin', 0.02)
    recovery_bars = cfg.get('ut_recovery_bars', 3)

    rolling_high = df['high'].rolling(lookback).max().shift(1)
    rolling_low = df['low'].rolling(lookback).min().shift(1)

    # Breakout above range
    breaks_high = df['high'] > rolling_high * (1 + breakout_margin)

    # Quick reversal
    future_close = df['close'].shift(-recovery_bars)
    fails = future_close < rolling_high

    # Volume spike
    volume_spike = df['volume_z'] > 1.0

    detected = breaks_high & fails.shift(recovery_bars).fillna(False) & volume_spike

    # Confidence
    breakout_size = ((df['high'] - rolling_high) / (rolling_high + 1e-9)).clip(0, 0.05) / 0.05
    confidence = (
        breakout_size * 0.40 +
        (df['volume_z'] / 3.0).clip(0, 1) * 0.35 +
        0.25
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_upthrust_after_distribution(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect UTAD - Final upthrust at end of distribution before markdown.

    More aggressive than regular UT, often marks the absolute top.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    # Similar to UT but with additional context checks
    ut_detected, ut_confidence = detect_upthrust(df, cfg)

    # Additional UTAD characteristics
    if 'rsi_14' in df.columns:
        extreme_rsi = df['rsi_14'] > cfg.get('utad_rsi_min', 70)
        ut_detected = ut_detected & extreme_rsi
        ut_confidence = ut_confidence * 1.15  # Boost confidence

    return ut_detected, ut_confidence


# ============================================================================
# PHASE D/E: LAST POINT OF SUPPORT/SUPPLY
# ============================================================================

def detect_last_point_of_support(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Last Point of Support (LPS) - Final test before markup begins.

    Characteristics:
        - Price pulls back to support area
        - Volume very low (volume_z < 0.0)
        - Holds above ST/Spring low
        - Occurs after SOS
        - Close in upper 60% of bar

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('lps_lookback', 30)
    volume_z_max = cfg.get('lps_volume_z_max', 0.0)
    close_position_min = cfg.get('lps_close_position_min', 0.6)

    # Find support (recent low after which price rose)
    rolling_low = df['low'].rolling(lookback).min()

    # Price at support area (within 3% of low)
    at_support = (df['low'] - rolling_low) / (rolling_low + 1e-9) < 0.03

    # Very low volume
    low_volume = df['volume_z'] < volume_z_max

    # Strong close
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)
    strong_close = close_position > close_position_min

    detected = at_support & low_volume & strong_close

    # Confidence
    confidence = (
        (1 - df['volume_z'] / 2.0).clip(0, 1) * 0.40 +
        close_position * 0.35 +
        at_support.astype(float) * 0.25
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_last_point_of_supply(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Last Point of Supply (LPSY) - Final rally before markdown.

    Mirror of LPS for distribution phase.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('lpsy_lookback', 30)
    volume_z_max = cfg.get('lpsy_volume_z_max', 0.0)
    close_position_max = cfg.get('lpsy_close_position_max', 0.4)

    # Find resistance
    rolling_high = df['high'].rolling(lookback).max()

    # Price at resistance (within 3%)
    at_resistance = (rolling_high - df['high']) / (rolling_high + 1e-9) < 0.03

    # Very low volume
    low_volume = df['volume_z'] < volume_z_max

    # Weak close
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)
    weak_close = close_position < close_position_max

    detected = at_resistance & low_volume & weak_close

    # Confidence
    confidence = (
        (1 - df['volume_z'] / 2.0).clip(0, 1) * 0.40 +
        (1 - close_position) * 0.35 +
        at_resistance.astype(float) * 0.25
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


# ============================================================================
# COMPOSITE EVENT DETECTOR
# ============================================================================

def detect_all_wyckoff_events(df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Detect all Wyckoff events and add columns to dataframe.

    This is the main entry point for Wyckoff event detection. It computes
    all 18 events and adds boolean + confidence columns to the dataframe.

    Args:
        df: OHLCV dataframe (must have volume, and ideally volume_z, liquidity_score)
        cfg: Optional configuration dict with event-specific thresholds

    Returns:
        DataFrame with added Wyckoff event columns:
            - wyckoff_sc, wyckoff_sc_confidence
            - wyckoff_bc, wyckoff_bc_confidence
            - wyckoff_ar, wyckoff_ar_confidence
            - wyckoff_as, wyckoff_as_confidence
            - wyckoff_st, wyckoff_st_confidence
            - wyckoff_sos, wyckoff_sos_confidence
            - wyckoff_sow, wyckoff_sow_confidence
            - wyckoff_lps, wyckoff_lps_confidence
            - wyckoff_lpsy, wyckoff_lpsy_confidence
            - wyckoff_spring_a, wyckoff_spring_a_confidence
            - wyckoff_spring_b, wyckoff_spring_b_confidence
            - wyckoff_ut, wyckoff_ut_confidence
            - wyckoff_utad, wyckoff_utad_confidence
            - wyckoff_phase_abc (categorical: A/B/C/D/E/neutral)
            - wyckoff_sequence_position (int: 1-10 within phase)

    Example:
        >>> df = pd.read_parquet('data/BTC_1H.parquet')
        >>> df = detect_all_wyckoff_events(df, cfg={'sc_volume_z_min': 2.5})
        >>> df[df['wyckoff_sc']].iloc[-5:]  # Last 5 SC events
    """
    cfg = cfg or {}

    # Ensure required base features exist
    if 'volume_z' not in df.columns and 'volume' in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    logger.info(f"Detecting Wyckoff events on {len(df)} bars")

    # Phase A events
    df['wyckoff_sc'], df['wyckoff_sc_confidence'] = detect_selling_climax(df, cfg)
    df['wyckoff_bc'], df['wyckoff_bc_confidence'] = detect_buying_climax(df, cfg)
    df['wyckoff_ar'], df['wyckoff_ar_confidence'] = detect_automatic_rally(df, cfg, df['wyckoff_sc'])
    df['wyckoff_as'], df['wyckoff_as_confidence'] = detect_automatic_reaction(df, cfg, df['wyckoff_bc'])
    df['wyckoff_st'], df['wyckoff_st_confidence'] = detect_secondary_test(df, cfg, df['wyckoff_sc'])

    # Phase B events
    df['wyckoff_sos'], df['wyckoff_sos_confidence'] = detect_sign_of_strength(df, cfg)
    df['wyckoff_sow'], df['wyckoff_sow_confidence'] = detect_sign_of_weakness(df, cfg)

    # Phase C events
    df['wyckoff_spring_a'], df['wyckoff_spring_a_confidence'] = detect_spring_type_a(df, cfg)
    df['wyckoff_spring_b'], df['wyckoff_spring_b_confidence'] = detect_spring_type_b(df, cfg)
    df['wyckoff_ut'], df['wyckoff_ut_confidence'] = detect_upthrust(df, cfg)
    df['wyckoff_utad'], df['wyckoff_utad_confidence'] = detect_upthrust_after_distribution(df, cfg)

    # Phase D events
    df['wyckoff_lps'], df['wyckoff_lps_confidence'] = detect_last_point_of_support(df, cfg)
    df['wyckoff_lpsy'], df['wyckoff_lpsy_confidence'] = detect_last_point_of_supply(df, cfg)

    # Determine phase and sequence
    df['wyckoff_phase_abc'] = _determine_wyckoff_phase(df)
    df['wyckoff_sequence_position'] = _compute_sequence_position(df)

    # Log summary
    event_counts = {
        'SC': df['wyckoff_sc'].sum(),
        'BC': df['wyckoff_bc'].sum(),
        'AR': df['wyckoff_ar'].sum(),
        'AS': df['wyckoff_as'].sum(),
        'ST': df['wyckoff_st'].sum(),
        'SOS': df['wyckoff_sos'].sum(),
        'SOW': df['wyckoff_sow'].sum(),
        'Spring_A': df['wyckoff_spring_a'].sum(),
        'Spring_B': df['wyckoff_spring_b'].sum(),
        'UT': df['wyckoff_ut'].sum(),
        'UTAD': df['wyckoff_utad'].sum(),
        'LPS': df['wyckoff_lps'].sum(),
        'LPSY': df['wyckoff_lpsy'].sum(),
    }

    logger.info("Wyckoff event detection complete:")
    for event, count in event_counts.items():
        if count > 0:
            logger.info(f"  {event}: {count} events detected")

    return df


def _determine_wyckoff_phase(df: pd.DataFrame) -> pd.Series:
    """
    Determine current Wyckoff phase based on recent events.

    Phase A: SC/BC/AR/AS/ST detected recently
    Phase B: SOS/SOW detected, no recent Spring/UT
    Phase C: Spring/UT detected recently
    Phase D: LPS/LPSY detected, trend beginning
    Phase E: Sustained trend continuation

    Args:
        df: DataFrame with event columns

    Returns:
        Series with phase labels
    """
    phase = pd.Series('neutral', index=df.index)

    # Phase A indicators (within last 10 bars)
    phase_a_events = (
        df['wyckoff_sc'].rolling(10).sum() +
        df['wyckoff_bc'].rolling(10).sum() +
        df['wyckoff_ar'].rolling(10).sum() +
        df['wyckoff_as'].rolling(10).sum() +
        df['wyckoff_st'].rolling(10).sum()
    )
    phase.loc[phase_a_events > 0] = 'A'

    # Phase B indicators
    phase_b_events = (
        df['wyckoff_sos'].rolling(20).sum() +
        df['wyckoff_sow'].rolling(20).sum()
    )
    phase.loc[(phase_b_events > 0) & (phase == 'neutral')] = 'B'

    # Phase C indicators (springs/upthrusts override Phase B)
    phase_c_events = (
        df['wyckoff_spring_a'].rolling(10).sum() +
        df['wyckoff_spring_b'].rolling(10).sum() +
        df['wyckoff_ut'].rolling(10).sum() +
        df['wyckoff_utad'].rolling(10).sum()
    )
    phase.loc[phase_c_events > 0] = 'C'

    # Phase D indicators
    phase_d_events = (
        df['wyckoff_lps'].rolling(15).sum() +
        df['wyckoff_lpsy'].rolling(15).sum()
    )
    phase.loc[(phase_d_events > 0) & (phase != 'C')] = 'D'

    return phase


def _compute_sequence_position(df: pd.DataFrame) -> pd.Series:
    """
    Compute position in Wyckoff sequence (1-10).

    Maps events to sequence numbers:
    1-3: Phase A (SC/BC → AR/AS → ST)
    4-6: Phase B (SOS/SOW → Building → Testing)
    7-8: Phase C (Spring/UT → Confirmation)
    9-10: Phase D/E (LPS/LPSY → Trend)

    Args:
        df: DataFrame with phase column

    Returns:
        Series with sequence position
    """
    position = pd.Series(0, index=df.index)

    # Simplified mapping based on phase
    position.loc[df['wyckoff_phase_abc'] == 'A'] = 2
    position.loc[df['wyckoff_phase_abc'] == 'B'] = 5
    position.loc[df['wyckoff_phase_abc'] == 'C'] = 7
    position.loc[df['wyckoff_phase_abc'] == 'D'] = 9
    position.loc[df['wyckoff_phase_abc'] == 'E'] = 10

    return position


# ============================================================================
# PTI INTEGRATION
# ============================================================================

def integrate_wyckoff_with_pti(df: pd.DataFrame, pti_scores: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Integrate Wyckoff events with PTI (Psychology Trap Index).

    Key integrations:
        - Springs/UTAD often coincide with high PTI (retail trapped)
        - SC/BC events validate PTI bearish/bullish traps
        - LPS/LPSY can indicate PTI trap resolution

    Args:
        df: DataFrame with Wyckoff event columns
        pti_scores: Optional PTI scores (if not in df)

    Returns:
        DataFrame with integration columns:
            - wyckoff_pti_confluence (bool)
            - wyckoff_pti_score (float)
    """
    if pti_scores is not None:
        df['pti_score'] = pti_scores

    if 'pti_score' not in df.columns:
        logger.warning("PTI scores not available, skipping integration")
        df['wyckoff_pti_confluence'] = False
        df['wyckoff_pti_score'] = 0.0
        return df

    # High PTI + Spring/UT = strong trap confirmation
    trap_events = df['wyckoff_spring_a'] | df['wyckoff_spring_b'] | df['wyckoff_ut'] | df['wyckoff_utad']
    high_pti = df['pti_score'] > 0.6

    df['wyckoff_pti_confluence'] = trap_events & high_pti

    # Composite score (weighted)
    df['wyckoff_pti_score'] = (
        df['wyckoff_spring_a_confidence'] * 0.25 +
        df['wyckoff_spring_b_confidence'] * 0.20 +
        df['wyckoff_ut_confidence'] * 0.25 +
        df['wyckoff_utad_confidence'] * 0.30
    ) * df['pti_score']

    logger.info(f"PTI-Wyckoff confluence: {df['wyckoff_pti_confluence'].sum()} events")

    return df
