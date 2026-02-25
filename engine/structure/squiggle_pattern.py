"""
1-2-3 Squiggle Pattern Detection

The classic BOS→Retest→Continuation setup.
Stage 2 (retest) is the golden entry window.

Author: Bull Machine v2.0
"""

import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SquigglePattern:
    """
    1-2-3 Squiggle Pattern state.

    Attributes:
        stage: Current stage (0=none, 1=BOS, 2=retest, 3=continuation)
        pattern_id: Unique identifier for this pattern instance
        direction: 'bullish' | 'bearish' | 'none'
        entry_window: True if at Stage 2 (retest phase)
        confidence: 0-1 pattern quality score
        bos_level: Price level where BOS occurred
        retest_quality: 0-1 how clean the retest is
        bars_since_bos: Time elapsed since BOS
    """
    stage: int
    pattern_id: str
    direction: str
    entry_window: bool
    confidence: float
    bos_level: float
    retest_quality: float
    bars_since_bos: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'squiggle_stage': self.stage,
            'squiggle_pattern_id': self.pattern_id,
            'squiggle_direction': self.direction,
            'squiggle_entry_window': self.entry_window,
            'squiggle_confidence': self.confidence,
            'squiggle_bos_level': self.bos_level,
            'squiggle_retest_quality': self.retest_quality,
            'squiggle_bars_since_bos': self.bars_since_bos
        }


def find_recent_bos(df: pd.DataFrame, lookback: int = 30) -> Optional[Dict]:
    """
    Find most recent Break of Structure.

    Args:
        df: OHLCV DataFrame
        lookback: Bars to search for BOS

    Returns:
        {
            'direction': 'bullish' | 'bearish',
            'level': float,
            'idx': int,
            'displacement': float
        } or None
    """
    if len(df) < lookback + 5:
        return None

    # Find swing high/low in lookback window
    swing_high = df['high'].iloc[-lookback:-5].max()
    swing_low = df['low'].iloc[-lookback:-5].min()

    # Check last 5 bars for BOS
    for i in range(len(df) - 5, len(df)):
        close = df['close'].iloc[i]

        # Bullish BOS: close above swing high
        if close > swing_high:
            displacement = (close - swing_high) / swing_high
            if displacement > 0.01:  # Minimum 1% break
                return {
                    'direction': 'bullish',
                    'level': float(swing_high),
                    'idx': i,
                    'displacement': float(displacement)
                }

        # Bearish BOS: close below swing low
        elif close < swing_low:
            displacement = (swing_low - close) / swing_low
            if displacement > 0.01:
                return {
                    'direction': 'bearish',
                    'level': float(swing_low),
                    'idx': i,
                    'displacement': float(displacement)
                }

    return None


def check_retest_zone(df: pd.DataFrame, bos_info: Dict) -> Optional[Dict]:
    """
    Check if price is retesting the BOS level.

    Args:
        df: OHLCV DataFrame
        bos_info: BOS information from find_recent_bos()

    Returns:
        {
            'in_retest': bool,
            'quality': float,  # 0-1
            'bars_in_zone': int
        } or None
    """
    if not bos_info:
        return None

    direction = bos_info['direction']
    bos_level = bos_info['level']
    bos_idx = bos_info['idx']

    # Define retest zone (±2% of BOS level)
    zone_size = 0.02
    upper_zone = bos_level * (1 + zone_size)
    lower_zone = bos_level * (1 - zone_size)

    # Check if current price is in retest zone
    current_price = df['close'].iloc[-1]
    current_close = current_price  # Alias for compatibility
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]

    in_retest = False
    quality = 0.0

    if direction == 'bullish':
        # For bullish BOS, retest should touch from above
        if current_low <= upper_zone and current_close >= bos_level * 0.98:
            in_retest = True
            # Quality: closer to exact BOS level = better
            proximity = 1.0 - abs(current_price - bos_level) / (bos_level * zone_size)
            quality = max(0.0, min(1.0, proximity))

    elif direction == 'bearish':
        # For bearish BOS, retest should touch from below
        if current_high >= lower_zone and current_close <= bos_level * 1.02:
            in_retest = True
            proximity = 1.0 - abs(current_price - bos_level) / (bos_level * zone_size)
            quality = max(0.0, min(1.0, proximity))

    # Count bars in retest zone
    bars_in_zone = 0
    for i in range(bos_idx + 1, len(df)):
        if lower_zone <= df['close'].iloc[i] <= upper_zone:
            bars_in_zone += 1

    return {
        'in_retest': in_retest,
        'quality': float(quality),
        'bars_in_zone': bars_in_zone
    }


def check_continuation(df: pd.DataFrame, bos_info: Dict,
                       retest_complete_idx: int) -> Optional[Dict]:
    """
    Check if continuation (Stage 3) has begun.

    Args:
        df: OHLCV DataFrame
        bos_info: BOS information
        retest_complete_idx: Bar where retest completed

    Returns:
        {
            'continuation_confirmed': bool,
            'strength': float  # 0-1
        }
    """
    if not bos_info or retest_complete_idx >= len(df) - 1:
        return {'continuation_confirmed': False, 'strength': 0.0}

    direction = bos_info['direction']
    bos_level = bos_info['level']

    # Check if price moved in BOS direction after retest
    current_price = df['close'].iloc[-1]
    retest_price = df['close'].iloc[retest_complete_idx]

    if direction == 'bullish':
        # Continuation = move above BOS level
        if current_price > bos_level * 1.01:
            displacement = (current_price - retest_price) / retest_price
            strength = min(displacement / 0.05, 1.0)  # 5% = full strength
            return {
                'continuation_confirmed': True,
                'strength': float(strength)
            }

    elif direction == 'bearish':
        # Continuation = move below BOS level
        if current_price < bos_level * 0.99:
            displacement = (retest_price - current_price) / retest_price
            strength = min(displacement / 0.05, 1.0)
            return {
                'continuation_confirmed': True,
                'strength': float(strength)
            }

    return {'continuation_confirmed': False, 'strength': 0.0}


def detect_squiggle_123(df: pd.DataFrame, timeframe: str = '4H',
                        config: Optional[Dict] = None) -> SquigglePattern:
    """
    Detect 1-2-3 Squiggle Pattern (BOS→Retest→Continuation).

    Args:
        df: OHLCV DataFrame
        timeframe: '1H', '4H', or '1D'
        config: Optional configuration

    Returns:
        SquigglePattern with current stage and entry window status

    Stages:
        1. BOS (Break of Structure) - Impulse move beyond prior swing
        2. Retest (OB/FVG mitigation) - Entry window ✅
        3. Continuation (Follow-through) - Confirmation of new trend

    Example:
        >>> pattern = detect_squiggle_123(df_4h, timeframe='4H')
        >>> if pattern.entry_window:  # Stage 2
        >>>     # Golden entry setup
        >>>     fusion_score += 0.05
    """
    config = config or {}

    # Timeframe-specific settings
    lookback_map = {
        '1H': 20,
        '4H': 30,
        '1D': 40
    }
    lookback = lookback_map.get(timeframe, 30)
    lookback = config.get('squiggle_lookback', lookback)

    if len(df) < lookback + 10:
        return SquigglePattern(
            stage=0,
            pattern_id='none',
            direction='none',
            entry_window=False,
            confidence=0.0,
            bos_level=0.0,
            retest_quality=0.0,
            bars_since_bos=0
        )

    # Stage 1: Find recent BOS
    bos_info = find_recent_bos(df, lookback=lookback)

    if not bos_info:
        # No BOS detected
        return SquigglePattern(
            stage=0,
            pattern_id='none',
            direction='none',
            entry_window=False,
            confidence=0.0,
            bos_level=0.0,
            retest_quality=0.0,
            bars_since_bos=0
        )

    # BOS detected - at least Stage 1
    direction = bos_info['direction']
    bos_level = bos_info['level']
    bos_idx = bos_info['idx']
    bars_since_bos = len(df) - bos_idx - 1

    # Check for timeout (BOS too old)
    max_bars_map = {
        '1H': 15,
        '4H': 10,
        '1D': 7
    }
    max_bars = max_bars_map.get(timeframe, 10)

    if bars_since_bos > max_bars:
        # BOS expired, pattern invalidated
        return SquigglePattern(
            stage=0,
            pattern_id='expired',
            direction='none',
            entry_window=False,
            confidence=0.0,
            bos_level=0.0,
            retest_quality=0.0,
            bars_since_bos=bars_since_bos
        )

    # Stage 2: Check for retest
    retest_info = check_retest_zone(df, bos_info)

    if retest_info and retest_info['in_retest']:
        # Stage 2: Retest in progress - ENTRY WINDOW
        pattern_id = f"{direction}_{timeframe}_{bos_idx}"
        confidence = bos_info['displacement'] * 0.5 + retest_info['quality'] * 0.5

        return SquigglePattern(
            stage=2,
            pattern_id=pattern_id,
            direction=direction,
            entry_window=True,  # ✅ Golden entry
            confidence=float(confidence),
            bos_level=bos_level,
            retest_quality=retest_info['quality'],
            bars_since_bos=bars_since_bos
        )

    # Stage 3: Check for continuation
    # (Simplified: if BOS exists but not in retest, check if continuation started)
    continuation_info = check_continuation(df, bos_info, bos_idx + 2)

    if continuation_info['continuation_confirmed']:
        # Stage 3: Continuation confirmed
        pattern_id = f"{direction}_{timeframe}_{bos_idx}"
        confidence = bos_info['displacement'] * 0.5 + continuation_info['strength'] * 0.5

        return SquigglePattern(
            stage=3,
            pattern_id=pattern_id,
            direction=direction,
            entry_window=False,  # Too late
            confidence=float(confidence),
            bos_level=bos_level,
            retest_quality=0.0,
            bars_since_bos=bars_since_bos
        )

    # Default: Stage 1 (BOS detected, waiting for retest)
    pattern_id = f"{direction}_{timeframe}_{bos_idx}"
    confidence = bos_info['displacement']

    return SquigglePattern(
        stage=1,
        pattern_id=pattern_id,
        direction=direction,
        entry_window=False,
        confidence=float(confidence),
        bos_level=bos_level,
        retest_quality=0.0,
        bars_since_bos=bars_since_bos
    )


def apply_squiggle_fusion_boost(fusion_score: float, squiggle: SquigglePattern,
                                 config: Optional[Dict] = None) -> tuple:
    """
    Apply Squiggle pattern fusion boost.

    Args:
        fusion_score: Current fusion score
        squiggle: SquigglePattern from detect_squiggle_123()
        config: Optional config

    Returns:
        (adjusted_score: float, boost_amount: float, reasons: list)

    Logic:
        - Stage 2 (entry_window): +0.05 fusion boost
        - High quality retest (>0.8): Additional +0.02
    """
    config = config or {}
    boost_amount = 0.0
    reasons = []

    if not squiggle.entry_window:
        return fusion_score, boost_amount, reasons

    # Stage 2 entry window
    boost_amount = 0.05
    reasons.append(f"Squiggle Stage 2 entry ({squiggle.direction}, quality={squiggle.retest_quality:.2f})")

    # Additional boost for high-quality retest
    if squiggle.retest_quality > 0.8:
        boost_amount += 0.02
        reasons.append(f"High-quality retest (>{0.8:.1f})")

    # Apply boost
    adjusted_score = min(fusion_score + boost_amount, 1.0)

    return adjusted_score, boost_amount, reasons
