"""
Bull Machine v1.6.0 - Hidden Fibonacci Zone Detection
Implementation of Moneytaur post:10 concepts for precise entry timing

Hidden Fibonacci zones derived from internal swing points within Wyckoff ranges,
not visible on standard charts but act as entry/exit magnets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from bull_machine.core.telemetry import log_telemetry

def _find_swing_points(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """
    Identify recent swing highs and lows for Fibonacci calculations.

    Uses rolling windows to find local extremes within Wyckoff ranges.
    """
    # Find swing high (highest high in lookback period)
    swing_high = df['high'].rolling(lookback, center=True).max().iloc[-lookback//2-1:-lookback//2]
    swing_high_idx = swing_high.idxmax() if len(swing_high) > 0 else df.index[-1]
    swing_high_price = df.loc[swing_high_idx, 'high']

    # Find swing low (lowest low in lookback period)
    swing_low = df['low'].rolling(lookback, center=True).min().iloc[-lookback//2-1:-lookback//2]
    swing_low_idx = swing_low.idxmin() if len(swing_low) > 0 else df.index[-1]
    swing_low_price = df.loc[swing_low_idx, 'low']

    # Calculate additional swing metrics
    swing_range = swing_high_price - swing_low_price
    swing_midpoint = (swing_high_price + swing_low_price) / 2

    return {
        'swing_high': swing_high_price,
        'swing_low': swing_low_price,
        'swing_range': swing_range,
        'swing_midpoint': swing_midpoint,
        'high_idx': swing_high_idx,
        'low_idx': swing_low_idx
    }

def _calculate_hidden_fib_levels(swing_info: Dict) -> Dict[str, float]:
    """
    Calculate hidden Fibonacci levels from swing points.

    Standard levels: 0.236, 0.382, 0.618, 0.786 (retracements)
    Extension levels: 1.272, 1.618, 2.618 (projections)
    """
    swing_high = swing_info['swing_high']
    swing_low = swing_info['swing_low']
    swing_range = swing_info['swing_range']

    # Retracement levels (from swing high)
    fib_levels = {
        'fib_236': swing_high - 0.236 * swing_range,
        'fib_382': swing_high - 0.382 * swing_range,
        'fib_500': swing_high - 0.500 * swing_range,  # 50% level
        'fib_618': swing_high - 0.618 * swing_range,
        'fib_786': swing_high - 0.786 * swing_range,

        # Extension levels (from swing low)
        'fib_1272': swing_low + 1.272 * swing_range,
        'fib_1618': swing_low + 1.618 * swing_range,
        'fib_2618': swing_low + 2.618 * swing_range
    }

    return fib_levels

def _score_fib_confluence(current_price: float, fib_levels: Dict, tolerance: float = 0.02) -> Dict[str, float]:
    """
    Score Fibonacci level confluence based on price proximity.

    Args:
        current_price: Current market price
        fib_levels: Dictionary of Fibonacci levels
        tolerance: Price tolerance as percentage of range

    Returns:
        Dictionary of scores for each Fibonacci level
    """
    scores = {}

    for level_name, fib_price in fib_levels.items():
        # Calculate distance as percentage
        price_distance = abs(current_price - fib_price) / current_price

        if price_distance <= tolerance:
            # Score inversely proportional to distance
            proximity_score = 1.0 - (price_distance / tolerance)

            # Weight different Fibonacci levels
            if 'fib_618' in level_name or 'fib_382' in level_name:
                base_score = 0.35  # Key retracement levels
            elif 'fib_1618' in level_name:
                base_score = 0.40  # Golden ratio extension
            elif 'fib_500' in level_name:
                base_score = 0.25  # 50% retracement
            elif 'fib_236' in level_name or 'fib_786' in level_name:
                base_score = 0.20  # Secondary levels
            else:
                base_score = 0.15  # Extension levels

            scores[level_name] = base_score * proximity_score
        else:
            scores[level_name] = 0.0

    return scores

def _volume_confluence_bonus(df: pd.DataFrame, fib_score: float) -> float:
    """
    Add volume confirmation bonus to Fibonacci scores.

    Higher volume near Fibonacci levels indicates institutional interest.
    """
    if fib_score == 0.0:
        return 0.0

    try:
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Volume bonus scaling
        if volume_ratio >= 2.0:
            bonus = 0.15  # High volume confirmation
        elif volume_ratio >= 1.5:
            bonus = 0.10  # Moderate volume confirmation
        elif volume_ratio >= 1.2:
            bonus = 0.05  # Light volume confirmation
        else:
            bonus = 0.0   # No volume confirmation

        return bonus

    except:
        return 0.0

def _timeframe_adjustment(base_score: float, tf: str) -> float:
    """
    Adjust Fibonacci scores based on timeframe significance.

    Higher timeframes carry more weight for Fibonacci levels.
    """
    if tf == '1D':
        multiplier = 1.2  # Daily Fibonacci levels more significant
    elif tf == '4H':
        multiplier = 1.0  # Base multiplier
    elif tf == '1H':
        multiplier = 0.9  # Hourly levels less significant
    else:
        multiplier = 1.0

    return base_score * multiplier

def compute_hidden_fib_scores(df: pd.DataFrame, tf: str) -> Dict[str, float]:
    """
    Compute scores for hidden Fibonacci retracement/extension zones.

    Based on Moneytaur post:10:
    - Hidden Fibonacci levels from internal swing points
    - Volume confirmation at key levels
    - Timeframe-weighted significance
    - Confluence scoring for entry precision

    Args:
        df: OHLCV DataFrame with sufficient history (min 25 bars)
        tf: Timeframe string ('1H', '4H', '1D')

    Returns:
        Dict with 'fib_retracement' and 'fib_extension' scores
    """
    if len(df) < 25:
        return {'fib_retracement': 0.0, 'fib_extension': 0.0}

    try:
        current_price = df['close'].iloc[-1]

        # Find swing points for Fibonacci calculation
        swing_info = _find_swing_points(df)

        # Skip if range too small
        if swing_info['swing_range'] < current_price * 0.015:  # Less than 1.5% range
            return {'fib_retracement': 0.0, 'fib_extension': 0.0}

        # Calculate hidden Fibonacci levels
        fib_levels = _calculate_hidden_fib_levels(swing_info)

        # Score confluence with current price
        tolerance = 0.025  # 2.5% tolerance for Fibonacci levels
        fib_scores = _score_fib_confluence(current_price, fib_levels, tolerance)

        # Aggregate retracement levels (0.236 to 0.786)
        retracement_levels = ['fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786']
        retracement_score = max([fib_scores.get(level, 0.0) for level in retracement_levels])

        # Aggregate extension levels (1.272+)
        extension_levels = ['fib_1272', 'fib_1618', 'fib_2618']
        extension_score = max([fib_scores.get(level, 0.0) for level in extension_levels])

        # Apply volume confirmation bonus
        retracement_score += _volume_confluence_bonus(df, retracement_score)
        extension_score += _volume_confluence_bonus(df, extension_score)

        # Apply timeframe adjustment
        retracement_score = _timeframe_adjustment(retracement_score, tf)
        extension_score = _timeframe_adjustment(extension_score, tf)

        # Cap scores
        retracement_score = min(retracement_score, 0.70)
        extension_score = min(extension_score, 0.70)

        # Log for analysis
        log_telemetry('layer_masks.json', {
            'module': 'hidden_fibs',
            'tf': tf,
            'fib_retracement': retracement_score,
            'fib_extension': extension_score,
            'swing_high': swing_info['swing_high'],
            'swing_low': swing_info['swing_low'],
            'swing_range': swing_info['swing_range'],
            'current_price': current_price,
            'key_levels': {
                'fib_382': fib_levels['fib_382'],
                'fib_618': fib_levels['fib_618'],
                'fib_1618': fib_levels['fib_1618']
            }
        })

        return {
            'fib_retracement': float(retracement_score),
            'fib_extension': float(extension_score)
        }

    except Exception as e:
        log_telemetry('layer_masks.json', {
            'module': 'hidden_fibs',
            'tf': tf,
            'error': str(e),
            'fib_retracement': 0.0,
            'fib_extension': 0.0
        })
        return {'fib_retracement': 0.0, 'fib_extension': 0.0}

def get_active_fib_levels(df: pd.DataFrame, tolerance: float = 0.03) -> List[Dict]:
    """
    Get currently active Fibonacci levels within tolerance range.

    Useful for debugging and visualization.

    Returns:
        List of dictionaries containing level info
    """
    if len(df) < 25:
        return []

    try:
        current_price = df['close'].iloc[-1]
        swing_info = _find_swing_points(df)
        fib_levels = _calculate_hidden_fib_levels(swing_info)

        active_levels = []
        for level_name, fib_price in fib_levels.items():
            distance = abs(current_price - fib_price) / current_price
            if distance <= tolerance:
                active_levels.append({
                    'level': level_name,
                    'price': fib_price,
                    'distance_pct': distance * 100,
                    'current_price': current_price
                })

        return sorted(active_levels, key=lambda x: x['distance_pct'])

    except:
        return []