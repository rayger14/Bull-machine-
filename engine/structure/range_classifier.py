"""
Range Outcome Classification

Classify how ranges resolve: Breakout vs Fakeout vs Rejection.
Critical for avoiding false breakouts and catching true momentum.

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RangeOutcome:
    """
    Range breakout outcome classification.

    Attributes:
        outcome: 'breakout' | 'fakeout' | 'rejection' | 'range_bound' | 'none'
        direction: 'bullish' | 'bearish' | 'neutral'
        confidence: 0-1 classification confidence
        range_high: Upper boundary of range
        range_low: Lower boundary of range
        breakout_strength: 0-1 for breakouts
        volume_confirmation: True if volume supports outcome
        bars_in_range: Duration of range before outcome
    """
    outcome: str
    direction: str
    confidence: float
    range_high: float
    range_low: float
    breakout_strength: float
    volume_confirmation: bool
    bars_in_range: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'range_outcome': self.outcome,
            'range_outcome_direction': self.direction,
            'range_outcome_confidence': self.confidence,
            'range_high': self.range_high,
            'range_low': self.range_low,
            'breakout_strength': self.breakout_strength,
            'volume_confirmation': self.volume_confirmation,
            'bars_in_range': self.bars_in_range
        }


def detect_range_boundaries(df: pd.DataFrame, lookback: int = 20) -> Optional[Tuple[float, float, int]]:
    """
    Detect if price is in a range and find boundaries.

    Args:
        df: OHLCV DataFrame
        lookback: Bars to analyze for range

    Returns:
        (range_high, range_low, bars_in_range) or None
    """
    if len(df) < lookback + 5:
        return None

    # Analyze recent price action
    recent_highs = df['high'].iloc[-lookback:]
    recent_lows = df['low'].iloc[-lookback:]

    # Calculate resistance and support
    range_high = recent_highs.max()
    range_low = recent_lows.min()

    # Check if price is oscillating within range
    # Range criteria: ATR < 3% of price AND price touching both boundaries
    atr = (df['high'] - df['low']).iloc[-lookback:].mean()
    current_price = df['close'].iloc[-1]

    if atr / current_price > 0.03:
        # Too volatile to be a range
        return None

    # Count touches of upper and lower boundaries (±1.5%)
    tolerance = 0.015
    upper_touches = sum(1 for h in recent_highs if h >= range_high * (1 - tolerance))
    lower_touches = sum(1 for l in recent_lows if l <= range_low * (1 + tolerance))

    # Require at least 2 touches of each boundary
    if upper_touches >= 2 and lower_touches >= 2:
        # Count bars in range
        bars_in_range = 0
        for i in range(len(df) - lookback, len(df)):
            if range_low <= df['close'].iloc[i] <= range_high:
                bars_in_range += 1
            else:
                # Reset if price exits range
                bars_in_range = 0

        return (float(range_high), float(range_low), bars_in_range)

    return None


def classify_breakout(df: pd.DataFrame, range_high: float, range_low: float,
                      bars_in_range: int) -> Dict:
    """
    Classify breakout as real or fake.

    Args:
        df: OHLCV DataFrame
        range_high: Upper boundary
        range_low: Lower boundary
        bars_in_range: Duration in range

    Returns:
        {
            'outcome': 'breakout' | 'fakeout' | 'none',
            'direction': 'bullish' | 'bearish' | 'neutral',
            'confidence': float,
            'strength': float,
            'volume_confirmed': bool
        }
    """
    current_close = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    current_volume = df['volume'].iloc[-1]

    # Volume baseline
    vol_mean = df['volume'].rolling(20).mean().iloc[-1]
    volume_surge = current_volume / vol_mean if vol_mean > 0 else 1.0

    # Check for breakout
    range_size = range_high - range_low
    breakout_threshold = 0.005  # 0.5% beyond range

    # Bullish breakout
    if current_close > range_high * (1 + breakout_threshold):
        displacement = (current_close - range_high) / range_size
        strength = min(displacement / 0.05, 1.0)  # 5% displacement = full strength

        # Check for fakeout signals
        fakeout_signals = []

        # 1. Weak volume (< 1.2x mean)
        if volume_surge < 1.2:
            fakeout_signals.append('weak_volume')

        # 2. Long wick (wick > 50% of body)
        body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        upper_wick = current_high - max(df['open'].iloc[-1], df['close'].iloc[-1])
        if body > 0 and upper_wick / body > 0.5:
            fakeout_signals.append('long_wick')

        # 3. Immediate reversal (next bar closes back in range)
        # (Can't check future, so mark as tentative)

        # Classification
        if len(fakeout_signals) >= 2:
            # Likely fakeout
            return {
                'outcome': 'fakeout',
                'direction': 'bullish',
                'confidence': 0.7,
                'strength': strength,
                'volume_confirmed': False
            }
        elif volume_surge > 1.5:
            # Strong breakout
            return {
                'outcome': 'breakout',
                'direction': 'bullish',
                'confidence': 0.8,
                'strength': strength,
                'volume_confirmed': True
            }
        else:
            # Tentative breakout
            return {
                'outcome': 'breakout',
                'direction': 'bullish',
                'confidence': 0.5,
                'strength': strength,
                'volume_confirmed': False
            }

    # Bearish breakout
    elif current_close < range_low * (1 - breakout_threshold):
        displacement = (range_low - current_close) / range_size
        strength = min(displacement / 0.05, 1.0)

        # Check for fakeout signals
        fakeout_signals = []

        if volume_surge < 1.2:
            fakeout_signals.append('weak_volume')

        body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        lower_wick = min(df['open'].iloc[-1], df['close'].iloc[-1]) - current_low
        if body > 0 and lower_wick / body > 0.5:
            fakeout_signals.append('long_wick')

        # Classification
        if len(fakeout_signals) >= 2:
            return {
                'outcome': 'fakeout',
                'direction': 'bearish',
                'confidence': 0.7,
                'strength': strength,
                'volume_confirmed': False
            }
        elif volume_surge > 1.5:
            return {
                'outcome': 'breakout',
                'direction': 'bearish',
                'confidence': 0.8,
                'strength': strength,
                'volume_confirmed': True
            }
        else:
            return {
                'outcome': 'breakout',
                'direction': 'bearish',
                'confidence': 0.5,
                'strength': strength,
                'volume_confirmed': False
            }

    # No breakout
    return {
        'outcome': 'none',
        'direction': 'neutral',
        'confidence': 0.0,
        'strength': 0.0,
        'volume_confirmed': False
    }


def check_rejection(df: pd.DataFrame, range_high: float, range_low: float) -> Dict:
    """
    Check if price is rejecting from range boundaries.

    Args:
        df: OHLCV DataFrame
        range_high: Upper boundary
        range_low: Lower boundary

    Returns:
        {
            'outcome': 'rejection' | 'none',
            'direction': 'bullish' | 'bearish' | 'neutral',
            'confidence': float
        }
    """
    current_close = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]

    tolerance = 0.01  # 1% zone

    # Check for rejection from upper boundary
    if current_high >= range_high * (1 - tolerance) and current_close < range_high * 0.98:
        # Touched high but closed well below
        wick_size = current_high - current_close
        body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])

        if wick_size > body_size * 1.5:
            # Strong rejection (long upper wick)
            return {
                'outcome': 'rejection',
                'direction': 'bearish',  # Rejected resistance
                'confidence': 0.7
            }

    # Check for rejection from lower boundary
    elif current_low <= range_low * (1 + tolerance) and current_close > range_low * 1.02:
        # Touched low but closed well above
        wick_size = current_close - current_low
        body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])

        if wick_size > body_size * 1.5:
            # Strong rejection (long lower wick)
            return {
                'outcome': 'rejection',
                'direction': 'bullish',  # Rejected support
                'confidence': 0.7
            }

    return {
        'outcome': 'none',
        'direction': 'neutral',
        'confidence': 0.0
    }


def classify_range_outcome(df: pd.DataFrame, timeframe: str = '4H',
                           config: Optional[Dict] = None) -> RangeOutcome:
    """
    Classify how a range is resolving.

    Args:
        df: OHLCV DataFrame
        timeframe: '1H', '4H', or '1D'
        config: Optional configuration

    Returns:
        RangeOutcome with classification

    Outcomes:
        - 'breakout': Clean break with volume confirmation
        - 'fakeout': Breakout fails, price returns to range
        - 'rejection': Price rejects from boundary (wick)
        - 'range_bound': Still in range, no clear outcome
        - 'none': No range detected

    Example:
        >>> outcome = classify_range_outcome(df_4h, timeframe='4H')
        >>> if outcome.outcome == 'breakout' and outcome.volume_confirmation:
        >>>     # High-probability continuation
        >>>     fusion_score += 0.08
        >>> elif outcome.outcome == 'fakeout':
        >>>     # Avoid this breakout (likely trap)
        >>>     fusion_score -= 0.10
    """
    config = config or {}

    # Timeframe-specific settings
    lookback_map = {
        '1H': 15,
        '4H': 20,
        '1D': 30
    }
    lookback = lookback_map.get(timeframe, 20)
    lookback = config.get('range_lookback', lookback)

    if len(df) < lookback + 10:
        return RangeOutcome(
            outcome='none',
            direction='neutral',
            confidence=0.0,
            range_high=0.0,
            range_low=0.0,
            breakout_strength=0.0,
            volume_confirmation=False,
            bars_in_range=0
        )

    # Step 1: Detect range
    range_info = detect_range_boundaries(df, lookback=lookback)

    if not range_info:
        # No range detected
        return RangeOutcome(
            outcome='none',
            direction='neutral',
            confidence=0.0,
            range_high=0.0,
            range_low=0.0,
            breakout_strength=0.0,
            volume_confirmation=False,
            bars_in_range=0
        )

    range_high, range_low, bars_in_range = range_info

    # Step 2: Classify breakout
    breakout_info = classify_breakout(df, range_high, range_low, bars_in_range)

    if breakout_info['outcome'] in ['breakout', 'fakeout']:
        return RangeOutcome(
            outcome=breakout_info['outcome'],
            direction=breakout_info['direction'],
            confidence=breakout_info['confidence'],
            range_high=range_high,
            range_low=range_low,
            breakout_strength=breakout_info['strength'],
            volume_confirmation=breakout_info['volume_confirmed'],
            bars_in_range=bars_in_range
        )

    # Step 3: Check for rejection
    rejection_info = check_rejection(df, range_high, range_low)

    if rejection_info['outcome'] == 'rejection':
        return RangeOutcome(
            outcome='rejection',
            direction=rejection_info['direction'],
            confidence=rejection_info['confidence'],
            range_high=range_high,
            range_low=range_low,
            breakout_strength=0.0,
            volume_confirmation=False,
            bars_in_range=bars_in_range
        )

    # Step 4: Still in range
    return RangeOutcome(
        outcome='range_bound',
        direction='neutral',
        confidence=0.8,
        range_high=range_high,
        range_low=range_low,
        breakout_strength=0.0,
        volume_confirmation=False,
        bars_in_range=bars_in_range
    )


def apply_range_fusion_adjustment(fusion_score: float, range_outcome: RangeOutcome,
                                   config: Optional[Dict] = None) -> tuple:
    """
    Adjust fusion score based on range outcome.

    Args:
        fusion_score: Current fusion score
        range_outcome: RangeOutcome from classify_range_outcome()
        config: Optional config

    Returns:
        (adjusted_score: float, adjustment: float, reasons: list)

    Logic:
        - Confirmed breakout: +0.08 fusion boost
        - Fakeout detected: -0.10 penalty (avoid trap)
        - Rejection: -0.05 penalty (choppy range)
        - Range_bound: No adjustment
    """
    config = config or {}
    adjustment = 0.0
    reasons = []

    if range_outcome.outcome == 'breakout':
        if range_outcome.volume_confirmation:
            # Strong breakout
            adjustment = +0.08
            reasons.append(f"Confirmed {range_outcome.direction} breakout (vol={range_outcome.volume_confirmation})")
        else:
            # Tentative breakout
            adjustment = +0.04
            reasons.append(f"Tentative {range_outcome.direction} breakout (low vol)")

    elif range_outcome.outcome == 'fakeout':
        # Penalize fakeouts heavily
        adjustment = -0.10
        reasons.append(f"Fakeout detected ({range_outcome.direction}) - avoid")

    elif range_outcome.outcome == 'rejection':
        # Minor penalty for choppy range
        adjustment = -0.05
        reasons.append(f"Range rejection ({range_outcome.direction}) - choppy")

    elif range_outcome.outcome == 'range_bound':
        # No adjustment, but note it
        reasons.append(f"Range-bound ({range_outcome.bars_in_range} bars)")

    # Apply adjustment
    adjusted_score = max(0.0, min(fusion_score + adjustment, 1.0))

    return adjusted_score, adjustment, reasons
