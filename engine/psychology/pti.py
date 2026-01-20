"""
PTI (Psychology Trap Index)

Quantifies herd traps WITHOUT sentiment data.
Uses observable price action patterns that reliably indicate trapped retail positions.

Key Indicators:
- Extreme RSI reversals (overbought→crash, oversold→bounce)
- Failed breakouts with volume exhaustion
- Wick traps (long wicks that stop out breakout traders)
- Diminishing momentum (lower highs on RSI while price makes new highs)

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PTISignal:
    """
    Psychology Trap Index signal.

    Attributes:
        pti_score: 0-1, higher = more herd trapping detected
        trap_type: 'bullish_trap' | 'bearish_trap' | 'none'
        components: Dict of individual trap scores
        confidence: 0-1 confidence in trap detection
        reversal_likely: True if trap suggests imminent reversal
    """
    pti_score: float
    trap_type: str
    components: Dict[str, float]
    confidence: float
    reversal_likely: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'pti_score': self.pti_score,
            'pti_trap_type': self.trap_type,
            'pti_confidence': self.confidence,
            'pti_reversal_likely': self.reversal_likely,
            # Component scores
            'pti_rsi_divergence': self.components.get('rsi_divergence', 0.0),
            'pti_volume_exhaustion': self.components.get('volume_exhaustion', 0.0),
            'pti_wick_trap': self.components.get('wick_trap', 0.0),
            'pti_failed_breakout': self.components.get('failed_breakout', 0.0),
        }


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect RSI divergence (price makes new high/low but RSI doesn't).

    Bearish divergence: Price new high, RSI lower high → Bullish trap
    Bullish divergence: Price new low, RSI higher low → Bearish trap

    Returns:
        {
            'divergence_type': 'bullish' | 'bearish' | 'none',
            'strength': 0-1
        }
    """
    if len(df) < lookback + 5:
        return {'divergence_type': 'none', 'strength': 0.0}

    rsi = calculate_rsi(df, period=14)
    close = df['close']

    # Look for divergence in recent bars
    recent_close = close.iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]

    # Bearish divergence (bullish trap)
    price_new_high_idx = recent_close.idxmax()
    price_new_high = recent_close.max()

    # Find previous high before the new high
    prev_highs = recent_close.iloc[:-5]
    if len(prev_highs) > 0:
        prev_high_idx = prev_highs.idxmax()
        prev_high = prev_highs.max()

        # Check if RSI made lower high
        rsi_at_new_high = recent_rsi.loc[price_new_high_idx]
        rsi_at_prev_high = recent_rsi.loc[prev_high_idx]

        if price_new_high > prev_high * 1.01 and rsi_at_new_high < rsi_at_prev_high - 5:
            # Bearish divergence detected
            strength = min((rsi_at_prev_high - rsi_at_new_high) / 20, 1.0)
            return {'divergence_type': 'bearish', 'strength': float(strength)}

    # Bullish divergence (bearish trap)
    price_new_low_idx = recent_close.idxmin()
    price_new_low = recent_close.min()

    prev_lows = recent_close.iloc[:-5]
    if len(prev_lows) > 0:
        prev_low_idx = prev_lows.idxmin()
        prev_low = prev_lows.min()

        # Check if RSI made higher low
        rsi_at_new_low = recent_rsi.loc[price_new_low_idx]
        rsi_at_prev_low = recent_rsi.loc[prev_low_idx]

        if price_new_low < prev_low * 0.99 and rsi_at_new_low > rsi_at_prev_low + 5:
            # Bullish divergence detected
            strength = min((rsi_at_new_low - rsi_at_prev_low) / 20, 1.0)
            return {'divergence_type': 'bullish', 'strength': float(strength)}

    return {'divergence_type': 'none', 'strength': 0.0}


def detect_volume_exhaustion(df: pd.DataFrame, lookback: int = 10) -> Dict:
    """
    Detect volume exhaustion (declining volume on price moves).

    Bullish trap: Price rising but volume declining
    Bearish trap: Price falling but volume declining

    Returns:
        {
            'exhaustion_type': 'bullish' | 'bearish' | 'none',
            'strength': 0-1
        }
    """
    if len(df) < lookback + 5:
        return {'exhaustion_type': 'none', 'strength': 0.0}

    recent_close = df['close'].iloc[-lookback:]
    recent_volume = df['volume'].iloc[-lookback:]

    # Calculate price trend
    price_slope = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0]

    # Calculate volume trend
    vol_first_half = recent_volume.iloc[:lookback//2].mean()
    vol_second_half = recent_volume.iloc[lookback//2:].mean()
    vol_decline = (vol_first_half - vol_second_half) / (vol_first_half + 1e-10)

    # Bullish exhaustion (price up, volume down)
    if price_slope > 0.02 and vol_decline > 0.2:
        strength = min(vol_decline / 0.5, 1.0)
        return {'exhaustion_type': 'bullish', 'strength': float(strength)}

    # Bearish exhaustion (price down, volume down)
    elif price_slope < -0.02 and vol_decline > 0.2:
        strength = min(vol_decline / 0.5, 1.0)
        return {'exhaustion_type': 'bearish', 'strength': float(strength)}

    return {'exhaustion_type': 'none', 'strength': 0.0}


def detect_wick_trap(df: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Detect wick traps (long wicks that stop out breakout traders).

    Bullish trap: Long upper wick (rejection from high)
    Bearish trap: Long lower wick (rejection from low)

    Returns:
        {
            'trap_type': 'bullish' | 'bearish' | 'none',
            'strength': 0-1
        }
    """
    if len(df) < lookback:
        return {'trap_type': 'none', 'strength': 0.0}

    recent_bars = df.iloc[-lookback:]

    for i in range(len(recent_bars)):
        bar = recent_bars.iloc[i]
        open_price = bar['open']
        high_price = bar['high']
        low_price = bar['low']
        close_price = bar['close']

        body = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price

        # Bullish trap: Upper wick > 2x body
        if body > 0 and upper_wick / body > 2.0:
            strength = min(upper_wick / (body * 3), 1.0)
            return {'trap_type': 'bullish', 'strength': float(strength)}

        # Bearish trap: Lower wick > 2x body
        if body > 0 and lower_wick / body > 2.0:
            strength = min(lower_wick / (body * 3), 1.0)
            return {'trap_type': 'bearish', 'strength': float(strength)}

    return {'trap_type': 'none', 'strength': 0.0}


def detect_failed_breakout(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect failed breakouts (price breaks level then immediately reverses).

    Returns:
        {
            'failed_type': 'bullish' | 'bearish' | 'none',
            'strength': 0-1
        }
    """
    if len(df) < lookback + 10:
        return {'failed_type': 'none', 'strength': 0.0}

    # Find range high/low in lookback period
    range_high = df['high'].iloc[-lookback:-5].max()
    range_low = df['low'].iloc[-lookback:-5].min()

    # Check recent bars for breakout + failure
    recent_bars = df.iloc[-5:]

    # Bullish trap: Break above range then close back inside
    for i in range(len(recent_bars) - 1):
        if recent_bars['high'].iloc[i] > range_high * 1.01:
            # Breakout occurred
            if recent_bars['close'].iloc[-1] < range_high * 0.99:
                # Failed back into range
                displacement = (recent_bars['high'].iloc[i] - range_high) / range_high
                strength = min(displacement / 0.03, 1.0)
                return {'failed_type': 'bullish', 'strength': float(strength)}

    # Bearish trap: Break below range then close back inside
    for i in range(len(recent_bars) - 1):
        if recent_bars['low'].iloc[i] < range_low * 0.99:
            # Breakout occurred
            if recent_bars['close'].iloc[-1] > range_low * 1.01:
                # Failed back into range
                displacement = (range_low - recent_bars['low'].iloc[i]) / range_low
                strength = min(displacement / 0.03, 1.0)
                return {'failed_type': 'bearish', 'strength': float(strength)}

    return {'failed_type': 'none', 'strength': 0.0}


def calculate_pti(df: pd.DataFrame, timeframe: str = '4H',
                  config: Optional[Dict] = None) -> PTISignal:
    """
    Calculate Psychology Trap Index (PTI).

    Combines multiple trap detection methods to quantify herd positioning
    WITHOUT relying on sentiment data.

    Args:
        df: OHLCV DataFrame
        timeframe: '1H', '4H', or '1D'
        config: Optional configuration

    Returns:
        PTISignal with trap detection results

    Trap Types:
        - Bullish trap: Retail trapped long at top (likely reversal down)
        - Bearish trap: Retail trapped short at bottom (likely reversal up)

    Components:
        1. RSI Divergence: 30% weight
        2. Volume Exhaustion: 25% weight
        3. Wick Traps: 25% weight
        4. Failed Breakouts: 20% weight

    Example:
        >>> pti = calculate_pti(df_4h, timeframe='4H')
        >>> if pti.trap_type == 'bullish_trap' and pti.pti_score > 0.7:
        >>>     # Strong bullish trap → avoid longs, consider shorts
        >>>     fusion_score -= 0.15
    """
    config = config or {}

    # Timeframe-specific settings
    lookback_map = {
        '1H': 20,
        '4H': 30,
        '1D': 40
    }
    lookback = lookback_map.get(timeframe, 30)

    if len(df) < lookback + 20:
        return PTISignal(
            pti_score=0.0,
            trap_type='none',
            components={},
            confidence=0.0,
            reversal_likely=False
        )

    # Detect all trap components
    rsi_div = detect_rsi_divergence(df, lookback=lookback)
    vol_exh = detect_volume_exhaustion(df, lookback=lookback//2)
    wick_trap = detect_wick_trap(df, lookback=5)
    failed_bo = detect_failed_breakout(df, lookback=lookback)

    # Store component scores
    components = {
        'rsi_divergence': rsi_div['strength'],
        'volume_exhaustion': vol_exh['strength'],
        'wick_trap': wick_trap['strength'],
        'failed_breakout': failed_bo['strength']
    }

    # Determine trap type (majority vote)
    bullish_trap_signals = sum([
        1 if rsi_div['divergence_type'] == 'bearish' else 0,
        1 if vol_exh['exhaustion_type'] == 'bullish' else 0,
        1 if wick_trap['trap_type'] == 'bullish' else 0,
        1 if failed_bo['failed_type'] == 'bullish' else 0
    ])

    bearish_trap_signals = sum([
        1 if rsi_div['divergence_type'] == 'bullish' else 0,
        1 if vol_exh['exhaustion_type'] == 'bearish' else 0,
        1 if wick_trap['trap_type'] == 'bearish' else 0,
        1 if failed_bo['failed_type'] == 'bearish' else 0
    ])

    # Determine trap type
    if bullish_trap_signals >= 2:
        trap_type = 'bullish_trap'  # Retail trapped long
    elif bearish_trap_signals >= 2:
        trap_type = 'bearish_trap'  # Retail trapped short
    else:
        trap_type = 'none'

    # Calculate PTI score (weighted average)
    weights = {
        'rsi_divergence': 0.30,
        'volume_exhaustion': 0.25,
        'wick_trap': 0.25,
        'failed_breakout': 0.20
    }

    pti_score = sum(components[k] * weights[k] for k in components.keys())
    pti_score = float(np.clip(pti_score, 0, 1))

    # Confidence = alignment of signals
    total_signals = bullish_trap_signals + bearish_trap_signals
    confidence = max(bullish_trap_signals, bearish_trap_signals) / 4.0 if total_signals > 0 else 0.0

    # Reversal likely if PTI > 0.6 and confidence > 0.5
    reversal_likely = pti_score > 0.6 and confidence > 0.5

    return PTISignal(
        pti_score=pti_score,
        trap_type=trap_type,
        components=components,
        confidence=float(confidence),
        reversal_likely=reversal_likely
    )


def apply_pti_fusion_adjustment(fusion_score: float, pti_signal: PTISignal,
                                 direction: str, config: Optional[Dict] = None) -> tuple:
    """
    Apply PTI fusion adjustment.

    Args:
        fusion_score: Current fusion score
        pti_signal: PTI signal from calculate_pti()
        direction: Intended trade direction ('long' or 'short')
        config: Optional config

    Returns:
        (adjusted_score: float, adjustment: float, reasons: list)

    Logic:
        - Bullish trap + long entry: -0.15 penalty (retail trapped, reversal down likely)
        - Bearish trap + short entry: -0.15 penalty (retail trapped, reversal up likely)
        - Trap opposite to direction: +0.05 bonus (fade the herd)
    """
    config = config or {}
    adjustment = 0.0
    reasons = []

    if pti_signal.pti_score < 0.4:
        # Low PTI, no adjustment
        return fusion_score, adjustment, reasons

    # Check if entry aligns with trap (bad) or against trap (good)
    if pti_signal.trap_type == 'bullish_trap':
        if direction == 'long':
            # Trying to go long when retail trapped long = BAD
            adjustment = -0.15
            reasons.append(f"PTI bullish trap ({pti_signal.pti_score:.2f}) - avoid longs")
        elif direction == 'short':
            # Shorting when retail trapped long = GOOD (fade the herd)
            adjustment = +0.05
            reasons.append(f"PTI bullish trap ({pti_signal.pti_score:.2f}) - fade the herd")

    elif pti_signal.trap_type == 'bearish_trap':
        if direction == 'short':
            # Trying to go short when retail trapped short = BAD
            adjustment = -0.15
            reasons.append(f"PTI bearish trap ({pti_signal.pti_score:.2f}) - avoid shorts")
        elif direction == 'long':
            # Going long when retail trapped short = GOOD
            adjustment = +0.05
            reasons.append(f"PTI bearish trap ({pti_signal.pti_score:.2f}) - fade the herd")

    # Apply adjustment
    adjusted_score = max(0.0, min(fusion_score + adjustment, 1.0))

    return adjusted_score, adjustment, reasons
