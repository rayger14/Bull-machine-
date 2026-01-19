"""
Fake-out Intensity Detection

Quantifies the severity of fake breakouts to avoid trap entries.

Key Metrics:
- Volume profile during breakout (weak volume = likely fake)
- Return speed (how fast price returns to range)
- Wick-to-body ratio (rejection strength)
- Follow-through absence (no continuation bars)

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class FakeoutSignal:
    """
    Fake-out intensity signal.

    Attributes:
        fakeout_detected: True if fake-out confirmed
        intensity: 0-1, higher = more severe fake-out
        direction: 'bullish_fakeout' | 'bearish_fakeout' | 'none'
        components: Dict of component scores
        breakout_level: Price level that was faked
        return_speed: Bars taken to return to range
    """
    fakeout_detected: bool
    intensity: float
    direction: str
    components: Dict[str, float]
    breakout_level: float
    return_speed: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'fakeout_detected': self.fakeout_detected,
            'fakeout_intensity': self.intensity,
            'fakeout_direction': self.direction,
            'fakeout_breakout_level': self.breakout_level,
            'fakeout_return_speed': self.return_speed,
            # Components
            'fakeout_volume_weakness': self.components.get('volume_weakness', 0.0),
            'fakeout_wick_rejection': self.components.get('wick_rejection', 0.0),
            'fakeout_no_followthrough': self.components.get('no_followthrough', 0.0),
        }


def detect_weak_volume_breakout(df: pd.DataFrame, breakout_idx: int) -> float:
    """
    Detect if breakout had weak volume.

    Args:
        df: OHLCV DataFrame
        breakout_idx: Index of breakout bar

    Returns:
        Weakness score (0-1, higher = weaker volume)
    """
    if breakout_idx < 20 or breakout_idx >= len(df):
        return 0.0

    # Volume on breakout bar
    breakout_volume = df['volume'].iloc[breakout_idx]

    # Average volume before breakout
    pre_breakout_vol_mean = df['volume'].iloc[breakout_idx-20:breakout_idx].mean()

    if pre_breakout_vol_mean == 0:
        return 0.0

    # Volume ratio
    vol_ratio = breakout_volume / pre_breakout_vol_mean

    # Weak if < 1.2x mean (should be > 1.5x for real breakout)
    if vol_ratio < 1.2:
        weakness = (1.2 - vol_ratio) / 1.2
        return float(np.clip(weakness, 0, 1))

    return 0.0


def detect_wick_rejection(df: pd.DataFrame, breakout_idx: int, direction: str) -> float:
    """
    Detect wick rejection strength.

    Args:
        df: OHLCV DataFrame
        breakout_idx: Index of breakout bar
        direction: 'bullish' or 'bearish'

    Returns:
        Rejection score (0-1, higher = stronger rejection)
    """
    if breakout_idx >= len(df):
        return 0.0

    bar = df.iloc[breakout_idx]
    open_price = bar['open']
    high_price = bar['high']
    low_price = bar['low']
    close_price = bar['close']

    body = abs(close_price - open_price)
    if body == 0:
        return 0.0

    if direction == 'bullish':
        # Check upper wick
        upper_wick = high_price - max(open_price, close_price)
        wick_to_body = upper_wick / body

        # Strong rejection if wick > 2x body
        if wick_to_body > 2.0:
            rejection = min(wick_to_body / 4.0, 1.0)
            return float(rejection)

    elif direction == 'bearish':
        # Check lower wick
        lower_wick = min(open_price, close_price) - low_price
        wick_to_body = lower_wick / body

        if wick_to_body > 2.0:
            rejection = min(wick_to_body / 4.0, 1.0)
            return float(rejection)

    return 0.0


def detect_no_followthrough(df: pd.DataFrame, breakout_idx: int, direction: str) -> float:
    """
    Detect absence of follow-through after breakout.

    Args:
        df: OHLCV DataFrame
        breakout_idx: Index of breakout bar
        direction: 'bullish' or 'bearish'

    Returns:
        No-followthrough score (0-1, higher = less followthrough)
    """
    if breakout_idx + 3 >= len(df):
        return 0.0

    breakout_close = df['close'].iloc[breakout_idx]

    # Check next 3 bars
    followthrough_bars = 0

    for i in range(breakout_idx + 1, min(breakout_idx + 4, len(df))):
        bar_close = df['close'].iloc[i]

        if direction == 'bullish':
            # Expect higher closes
            if bar_close > breakout_close * 1.001:
                followthrough_bars += 1
        elif direction == 'bearish':
            # Expect lower closes
            if bar_close < breakout_close * 0.999:
                followthrough_bars += 1

    # No followthrough = 0 bars moving in breakout direction
    no_followthrough_score = 1.0 - (followthrough_bars / 3.0)
    return float(no_followthrough_score)


def calculate_return_speed(df: pd.DataFrame, breakout_idx: int,
                           range_level: float, direction: str) -> int:
    """
    Calculate how fast price returned to range after fake breakout.

    Args:
        df: OHLCV DataFrame
        breakout_idx: Index of breakout bar
        range_level: Range boundary level
        direction: 'bullish' or 'bearish'

    Returns:
        Number of bars to return to range (0 = immediate)
    """
    if breakout_idx + 1 >= len(df):
        return 999  # Not returned yet

    for i in range(breakout_idx + 1, len(df)):
        bar_close = df['close'].iloc[i]

        if direction == 'bullish':
            # Check if returned below range
            if bar_close < range_level * 0.99:
                return i - breakout_idx

        elif direction == 'bearish':
            # Check if returned above range
            if bar_close > range_level * 1.01:
                return i - breakout_idx

        # Max lookforward
        if i - breakout_idx > 10:
            return 999

    return 999  # Not returned


def detect_fakeout_intensity(df: pd.DataFrame, lookback: int = 30,
                              config: Optional[Dict] = None) -> FakeoutSignal:
    """
    Detect fake-out intensity.

    Args:
        df: OHLCV DataFrame
        lookback: Bars to analyze for range and breakout
        config: Optional configuration

    Returns:
        FakeoutSignal with intensity and components

    Logic:
        1. Identify recent range (lookback period)
        2. Detect breakout beyond range
        3. Check for fake-out indicators:
           - Weak volume (30% weight)
           - Wick rejection (35% weight)
           - No followthrough (35% weight)
        4. Confirm fake-out if price returns to range

    Example:
        >>> fakeout = detect_fakeout_intensity(df_4h, lookback=30)
        >>> if fakeout.intensity > 0.7:
        >>>     # Strong fake-out detected
        >>>     fusion_score -= 0.15
    """
    config = config or {}

    if len(df) < lookback + 10:
        return FakeoutSignal(
            fakeout_detected=False,
            intensity=0.0,
            direction='none',
            components={},
            breakout_level=0.0,
            return_speed=0
        )

    # Step 1: Find range boundaries
    range_period = df.iloc[-lookback:-5]
    range_high = range_period['high'].max()
    range_low = range_period['low'].min()

    # Step 2: Check recent bars for breakout
    recent_bars = df.iloc[-5:]

    for idx in range(len(recent_bars)):
        bar = recent_bars.iloc[idx]
        bar_idx_in_df = len(df) - 5 + idx

        # Bullish breakout
        if bar['high'] > range_high * 1.01:
            # Check if it's a fake-out
            volume_weakness = detect_weak_volume_breakout(df, bar_idx_in_df)
            wick_rejection = detect_wick_rejection(df, bar_idx_in_df, 'bullish')
            no_followthrough = detect_no_followthrough(df, bar_idx_in_df, 'bullish')
            return_speed = calculate_return_speed(df, bar_idx_in_df, range_high, 'bullish')

            # Calculate intensity (weighted average)
            components = {
                'volume_weakness': volume_weakness,
                'wick_rejection': wick_rejection,
                'no_followthrough': no_followthrough
            }

            intensity = (
                volume_weakness * 0.30 +
                wick_rejection * 0.35 +
                no_followthrough * 0.35
            )

            # Confirm fake-out if returned to range quickly
            fakeout_confirmed = return_speed < 5

            if fakeout_confirmed and intensity > 0.5:
                return FakeoutSignal(
                    fakeout_detected=True,
                    intensity=float(intensity),
                    direction='bullish_fakeout',
                    components=components,
                    breakout_level=float(range_high),
                    return_speed=return_speed
                )

        # Bearish breakout
        elif bar['low'] < range_low * 0.99:
            volume_weakness = detect_weak_volume_breakout(df, bar_idx_in_df)
            wick_rejection = detect_wick_rejection(df, bar_idx_in_df, 'bearish')
            no_followthrough = detect_no_followthrough(df, bar_idx_in_df, 'bearish')
            return_speed = calculate_return_speed(df, bar_idx_in_df, range_low, 'bearish')

            components = {
                'volume_weakness': volume_weakness,
                'wick_rejection': wick_rejection,
                'no_followthrough': no_followthrough
            }

            intensity = (
                volume_weakness * 0.30 +
                wick_rejection * 0.35 +
                no_followthrough * 0.35
            )

            fakeout_confirmed = return_speed < 5

            if fakeout_confirmed and intensity > 0.5:
                return FakeoutSignal(
                    fakeout_detected=True,
                    intensity=float(intensity),
                    direction='bearish_fakeout',
                    components=components,
                    breakout_level=float(range_low),
                    return_speed=return_speed
                )

    # No fake-out detected
    return FakeoutSignal(
        fakeout_detected=False,
        intensity=0.0,
        direction='none',
        components={},
        breakout_level=0.0,
        return_speed=0
    )


def apply_fakeout_fusion_penalty(fusion_score: float, fakeout: FakeoutSignal,
                                  direction: str, config: Optional[Dict] = None) -> tuple:
    """
    Apply fake-out fusion penalty.

    Args:
        fusion_score: Current fusion score
        fakeout: FakeoutSignal from detect_fakeout_intensity()
        direction: Intended trade direction
        config: Optional config

    Returns:
        (adjusted_score: float, adjustment: float, reasons: list)

    Logic:
        - Fake-out detected: -0.15 to -0.25 penalty (scaled by intensity)
        - Fast return (<3 bars): Additional -0.05 penalty
    """
    config = config or {}
    adjustment = 0.0
    reasons = []

    if not fakeout.fakeout_detected:
        return fusion_score, adjustment, reasons

    # Base penalty scaled by intensity
    base_penalty = -0.15 * fakeout.intensity

    # Additional penalty for very fast return
    if fakeout.return_speed < 3:
        base_penalty -= 0.05
        reasons.append(f"Fast fake-out return ({fakeout.return_speed} bars)")

    adjustment = base_penalty
    reasons.append(f"Fake-out detected ({fakeout.direction}, intensity={fakeout.intensity:.2f})")

    # Apply adjustment
    adjusted_score = max(0.0, min(fusion_score + adjustment, 1.0))

    return adjusted_score, adjustment, reasons
