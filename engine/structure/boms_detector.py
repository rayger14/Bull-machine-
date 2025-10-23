"""
BOMS (Break of Market Structure) Detection

More stringent than regular BOS - requires volume confirmation and FVG trail.
HTF BOMS signals are high-probability continuation setups.

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BOMSSignal:
    """
    Break of Market Structure signal.

    Attributes:
        boms_detected: True if BOMS confirmed
        direction: 'bullish' | 'bearish' | 'none'
        volume_surge: Volume ratio vs mean
        fvg_present: True if FVG left behind
        confirmation_bars: Bars since break
        break_level: Price level that was broken
        displacement: Size of move beyond break
    """
    boms_detected: bool
    direction: str
    volume_surge: float
    fvg_present: bool
    confirmation_bars: int
    break_level: float
    displacement: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'boms_detected': self.boms_detected,
            'boms_direction': self.direction,
            'boms_volume_surge': self.volume_surge,
            'boms_fvg_present': self.fvg_present,
            'boms_confirmation': self.confirmation_bars,
            'boms_break_level': self.break_level,
            'boms_displacement': self.displacement
        }


def find_swing_points(df: pd.DataFrame, window: int = 20) -> Dict:
    """
    Find swing highs and lows.

    Args:
        df: OHLCV DataFrame
        window: Lookback window for swings

    Returns:
        {
            'swing_high': float,
            'swing_low': float,
            'swing_high_idx': int,
            'swing_low_idx': int
        }
    """
    if len(df) < window:
        return {
            'swing_high': df['high'].max(),
            'swing_low': df['low'].min(),
            'swing_high_idx': df['high'].idxmax(),
            'swing_low_idx': df['low'].idxmin()
        }

    # Rolling max/min
    swing_high = df['high'].rolling(window).max().iloc[-window:-1].max()
    swing_low = df['low'].rolling(window).min().iloc[-window:-1].min()

    # Find indices - use iloc position instead of idxmax/idxmin
    high_window = df['high'].iloc[-window:-1]
    low_window = df['low'].iloc[-window:-1]

    swing_high_idx = high_window.idxmax()
    swing_low_idx = low_window.idxmin()

    # Convert to integer position
    swing_high_pos = df.index.get_loc(swing_high_idx) if swing_high_idx in df.index else -1
    swing_low_pos = df.index.get_loc(swing_low_idx) if swing_low_idx in df.index else -1

    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'swing_high_idx': int(swing_high_pos),
        'swing_low_idx': int(swing_low_pos)
    }


def detect_fvg_trail(df: pd.DataFrame, start_idx: int, direction: str) -> bool:
    """
    Detect if FVG (Fair Value Gap) was left behind in the move.

    Args:
        df: OHLCV DataFrame
        start_idx: Index where potential BOMS started
        direction: 'bullish' or 'bearish'

    Returns:
        True if FVG detected in last 3-5 bars
    """
    # Check last 5 bars for FVG
    lookback = min(5, len(df) - start_idx)

    for i in range(start_idx, start_idx + lookback):
        if i + 2 >= len(df):
            break

        # Bullish FVG: high[i] < low[i+2]
        if direction == 'bullish':
            if df['high'].iloc[i] < df['low'].iloc[i + 2]:
                gap_size = df['low'].iloc[i + 2] - df['high'].iloc[i]
                # Require meaningful gap (> 0.1% of price)
                if gap_size / df['close'].iloc[i] > 0.001:
                    return True

        # Bearish FVG: low[i] > high[i+2]
        elif direction == 'bearish':
            if df['low'].iloc[i] > df['high'].iloc[i + 2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i + 2]
                if gap_size / df['close'].iloc[i] > 0.001:
                    return True

    return False


def check_no_immediate_reversal(df: pd.DataFrame, break_idx: int,
                                direction: str, bars: int = 3) -> bool:
    """
    Confirm no immediate reversal after break.

    Args:
        df: OHLCV DataFrame
        break_idx: Index of break
        direction: 'bullish' or 'bearish'
        bars: Number of bars to check

    Returns:
        True if no reversal detected
    """
    if break_idx + bars >= len(df):
        return False

    if direction == 'bullish':
        # Check if any bar closed below break level
        break_level = df['close'].iloc[break_idx]
        for i in range(break_idx + 1, min(break_idx + bars + 1, len(df))):
            if df['close'].iloc[i] < break_level * 0.98:  # 2% tolerance
                return False
        return True

    elif direction == 'bearish':
        # Check if any bar closed above break level
        break_level = df['close'].iloc[break_idx]
        for i in range(break_idx + 1, min(break_idx + bars + 1, len(df))):
            if df['close'].iloc[i] > break_level * 1.02:  # 2% tolerance
                return False
        return True

    return False


def detect_boms(df: pd.DataFrame, timeframe: str = '4H',
               config: Optional[Dict] = None) -> BOMSSignal:
    """
    Detect Break of Market Structure (BOMS).

    BOMS is more stringent than regular BOS and requires:
    1. Close beyond prior swing high/low
    2. Volume > 1.5x mean (institutional participation)
    3. FVG left behind (imbalance trailing the move)
    4. No immediate reversal (confirmed follow-through)

    IMPORTANT: displacement field is ALWAYS calculated when price breaks structure,
    regardless of FVG/reversal confirmation. This allows archetype entry system
    to use displacement thresholds without requiring full BOMS confirmation.

    Args:
        df: OHLCV DataFrame
        timeframe: '1H', '4H', or '1D'
        config: Optional configuration

    Returns:
        BOMSSignal with detection results

    Example:
        >>> boms = detect_boms(df_4h, timeframe='4H')
        >>> if boms.boms_detected and boms.direction == 'bullish':
        >>>     # Strong HTF confirmation for longs
        >>>     fusion_score += 0.10
        >>> # Or use displacement without full confirmation:
        >>> if boms.displacement > 0.015:  # 1.5% displacement
        >>>     # Displacement-based entry (archetype system)
    """
    config = config or {}

    # Timeframe-specific settings
    volume_threshold = {
        '1H': 1.3,  # Less strict for 1H
        '4H': 1.5,  # Standard for 4H
        '1D': 1.8   # More strict for 1D
    }.get(timeframe, 1.5)

    swing_window = {
        '1H': 15,
        '4H': 20,
        '1D': 30
    }.get(timeframe, 20)

    # Override from config if present
    volume_threshold = config.get('boms_volume_threshold', volume_threshold)
    swing_window = config.get('boms_swing_window', swing_window)

    if len(df) < swing_window + 10:
        return BOMSSignal(
            boms_detected=False,
            direction='none',
            volume_surge=0.0,
            fvg_present=False,
            confirmation_bars=0,
            break_level=0.0,
            displacement=0.0
        )

    # Find swing points
    swings = find_swing_points(df, window=swing_window)
    swing_high = swings['swing_high']
    swing_low = swings['swing_low']

    # Track best displacement even if full BOMS not confirmed
    best_bullish_displacement = 0.0
    best_bearish_displacement = 0.0

    # Check recent bars for BOMS (last 5 bars)
    for i in range(len(df) - 5, len(df)):
        if i < 0:
            continue

        close = df['close'].iloc[i]
        volume = df['volume'].iloc[i]
        vol_mean = df['volume'].rolling(20).mean().iloc[i]

        # Skip if volume data invalid
        if vol_mean == 0 or np.isnan(vol_mean):
            continue

        volume_surge = volume / vol_mean

        # Check for bullish BOMS
        if close > swing_high and volume_surge > volume_threshold:
            # ALWAYS calculate displacement in ABSOLUTE price terms (for archetype system)
            # Archetype system compares displacement to ATR multiples, not percentages
            displacement = close - swing_high  # Absolute price difference
            best_bullish_displacement = max(best_bullish_displacement, displacement)

            # Check for FVG trail
            fvg_present = detect_fvg_trail(df, i - 3, 'bullish')

            if fvg_present:
                # Confirm no immediate reversal
                if check_no_immediate_reversal(df, i, 'bullish', bars=3):
                    confirmation_bars = len(df) - i - 1

                    return BOMSSignal(
                        boms_detected=True,
                        direction='bullish',
                        volume_surge=float(volume_surge),
                        fvg_present=True,
                        confirmation_bars=confirmation_bars,
                        break_level=float(swing_high),
                        displacement=float(displacement)
                    )

        # Check for bearish BOMS
        elif close < swing_low and volume_surge > volume_threshold:
            # ALWAYS calculate displacement in ABSOLUTE price terms (for archetype system)
            displacement = swing_low - close  # Absolute price difference
            best_bearish_displacement = max(best_bearish_displacement, displacement)

            # Check for FVG trail
            fvg_present = detect_fvg_trail(df, i - 3, 'bearish')

            if fvg_present:
                # Confirm no immediate reversal
                if check_no_immediate_reversal(df, i, 'bearish', bars=3):
                    confirmation_bars = len(df) - i - 1

                    return BOMSSignal(
                        boms_detected=True,
                        direction='bearish',
                        volume_surge=float(volume_surge),
                        fvg_present=True,
                        confirmation_bars=confirmation_bars,
                        break_level=float(swing_low),
                        displacement=float(displacement)
                    )

    # No full BOMS detected, but return best displacement if structure was broken
    # This allows archetype system to use displacement without requiring FVG/reversal confirmation
    if best_bullish_displacement > 0.0:
        return BOMSSignal(
            boms_detected=False,
            direction='none',
            volume_surge=0.0,
            fvg_present=False,
            confirmation_bars=0,
            break_level=float(swing_high),
            displacement=float(best_bullish_displacement)
        )
    elif best_bearish_displacement > 0.0:
        return BOMSSignal(
            boms_detected=False,
            direction='none',
            volume_surge=0.0,
            fvg_present=False,
            confirmation_bars=0,
            break_level=float(swing_low),
            displacement=float(best_bearish_displacement)
        )

    # No structure break at all
    return BOMSSignal(
        boms_detected=False,
        direction='none',
        volume_surge=0.0,
        fvg_present=False,
        confirmation_bars=0,
        break_level=0.0,
        displacement=0.0
    )


def apply_boms_fusion_boost(fusion_score: float, boms_signal: BOMSSignal,
                            timeframe: str, config: Optional[Dict] = None) -> tuple:
    """
    Apply BOMS fusion boost.

    Args:
        fusion_score: Current fusion score
        boms_signal: BOMS detection result
        timeframe: '1H', '4H', or '1D'
        config: Optional config

    Returns:
        (adjusted_score: float, boost_amount: float, reasons: list)

    Logic:
        - 4H/1D BOMS: +0.10 fusion boost (strong HTF confirmation)
        - 1H BOMS: +0.05 fusion boost (moderate LTF confirmation)
        - Required for entries > 2R risk
    """
    config = config or {}
    boost_amount = 0.0
    reasons = []

    if not boms_signal.boms_detected:
        return fusion_score, boost_amount, reasons

    # Timeframe-specific boosts
    boost_map = {
        '1H': 0.05,
        '4H': 0.10,
        '1D': 0.15
    }
    boost_amount = boost_map.get(timeframe, 0.10)

    # Additional boost for strong volume
    if boms_signal.volume_surge > 2.0:
        boost_amount += 0.02
        reasons.append(f"BOMS + exceptional volume ({boms_signal.volume_surge:.1f}x)")
    else:
        reasons.append(f"BOMS detected ({timeframe}, vol={boms_signal.volume_surge:.1f}x)")

    # Apply boost
    adjusted_score = min(fusion_score + boost_amount, 1.0)

    return adjusted_score, boost_amount, reasons
