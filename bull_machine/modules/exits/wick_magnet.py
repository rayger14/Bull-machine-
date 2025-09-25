"""
Bull Machine - Wick Magnet Detection
Crypto Chase inspired wick rejection analysis for exit timing
"""

import pandas as pd
import numpy as np
from bull_machine.core.telemetry import log_telemetry

def wick_magnet_distance(df: pd.DataFrame, threshold: float = 0.02) -> bool:
    """
    Detect wick magnet conditions for exit timing.

    A wick magnet occurs when price creates a significant rejection wick,
    indicating institutional resistance/support and potential reversal.

    Args:
        df: Price DataFrame with OHLCV data
        threshold: Minimum wick size as percentage of price

    Returns:
        bool: True if wick magnet detected
    """
    if len(df) < 3:
        return False

    current_candle = df.iloc[-1]
    open_price = current_candle['open']
    high_price = current_candle['high']
    low_price = current_candle['low']
    close_price = current_candle['close']
    volume = current_candle['volume']

    # Calculate wick sizes
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    body_size = abs(close_price - open_price)
    total_range = high_price - low_price

    # Avoid division by zero
    if close_price == 0 or total_range == 0:
        return False

    # Upper wick analysis (resistance rejection)
    upper_wick_pct = upper_wick / close_price
    upper_wick_dominance = upper_wick / total_range if total_range > 0 else 0

    # Lower wick analysis (support rejection)
    lower_wick_pct = lower_wick / close_price
    lower_wick_dominance = lower_wick / total_range if total_range > 0 else 0

    # Volume confirmation
    avg_volume = df['volume'].rolling(10).mean().iloc[-1]
    volume_spike = volume > avg_volume * 1.3 if avg_volume > 0 else False

    # Wick magnet conditions
    significant_upper_wick = (upper_wick_pct > threshold and upper_wick_dominance > 0.4)
    significant_lower_wick = (lower_wick_pct > threshold and lower_wick_dominance > 0.4)

    # Additional confirmation: wick should be at least 2x body size
    wick_to_body_ratio = max(upper_wick, lower_wick) / max(body_size, close_price * 0.001)

    magnet_detected = (significant_upper_wick or significant_lower_wick) and volume_spike and wick_to_body_ratio > 2.0

    # Determine magnet direction
    magnet_direction = "resistance" if significant_upper_wick else "support" if significant_lower_wick else "none"

    log_telemetry("layer_masks.json", {
        "wick_magnet": magnet_detected,
        "magnet_direction": magnet_direction,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "upper_wick_dominance": upper_wick_dominance,
        "lower_wick_dominance": lower_wick_dominance,
        "volume_spike": volume_spike,
        "wick_to_body_ratio": wick_to_body_ratio,
        "volume_ratio": volume / avg_volume if avg_volume > 0 else 0
    })

    return magnet_detected

def wick_strength_score(df: pd.DataFrame, lookback: int = 5) -> float:
    """
    Calculate wick strength over multiple candles.

    Args:
        df: Price DataFrame with OHLCV data
        lookback: Number of candles to analyze

    Returns:
        float: Wick strength score (0.0 to 1.0)
    """
    if len(df) < lookback:
        return 0.0

    recent_candles = df.iloc[-lookback:]
    wick_scores = []

    for _, candle in recent_candles.iterrows():
        open_p = candle['open']
        high_p = candle['high']
        low_p = candle['low']
        close_p = candle['close']

        if close_p == 0:
            continue

        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p
        total_range = high_p - low_p

        if total_range > 0:
            # Score based on wick size relative to range
            upper_score = (upper_wick / total_range) * (upper_wick / close_p)
            lower_score = (lower_wick / total_range) * (lower_wick / close_p)
            candle_score = max(upper_score, lower_score) * 10  # Scale up
            wick_scores.append(min(candle_score, 1.0))

    if not wick_scores:
        return 0.0

    # Weight recent candles more heavily
    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3][-len(wick_scores):])
    weighted_score = np.average(wick_scores, weights=weights)

    return min(weighted_score, 1.0)

def rejection_confluence(df: pd.DataFrame, level: float, tolerance: float = 0.01) -> bool:
    """
    Check if price shows rejection confluence at a specific level.

    Args:
        df: Price DataFrame
        level: Price level to check
        tolerance: Tolerance around level as percentage

    Returns:
        bool: True if rejection confluence detected
    """
    if len(df) < 10:
        return False

    recent_candles = df.iloc[-10:]
    rejections = 0

    for _, candle in recent_candles.iterrows():
        high_p = candle['high']
        low_p = candle['low']
        close_p = candle['close']

        # Check if price touched the level but closed away from it
        level_touched = (low_p <= level * (1 + tolerance) and high_p >= level * (1 - tolerance))

        if level_touched:
            # Check for rejection (close significantly away from level)
            close_distance = abs(close_p - level) / level
            if close_distance > tolerance * 2:
                rejections += 1

    # Need at least 2 rejections for confluence
    confluence_detected = rejections >= 2

    log_telemetry("layer_masks.json", {
        "rejection_confluence": confluence_detected,
        "rejections_count": rejections,
        "test_level": level,
        "tolerance": tolerance
    })

    return confluence_detected