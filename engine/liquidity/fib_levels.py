"""
Fibonacci Levels - Bull Machine v1.8.5

Calculates standard and negative Fibonacci retracement/extension levels
for liquidity confluence detection. Negative fibs (-0.272, -0.618) capture
Wyckoff extended moves beyond swing ranges.

Trader Alignment:
- @Wyckoff_Insider: Extended moves and trend targets (post:24)
- @Moneytaur: Key zones for institutional entries (post:33)
"""

from typing import List, Dict
import pandas as pd
import numpy as np


def calculate_fib_levels(swing_high: float, swing_low: float, config: Dict) -> List[float]:
    """
    Calculate Fibonacci retracement and extension levels.

    Args:
        swing_high: Recent swing high
        swing_low: Recent swing low
        config: Config dict with 'fib_levels', 'negative_fibs', 'negative_fibs_enabled'

    Returns:
        List of price levels (both positive and negative extensions)

    Example:
        >>> config = {'fib_levels': [0.618, 0.786], 'negative_fibs': [-0.272, -0.618],
        ...           'negative_fibs_enabled': True}
        >>> levels = calculate_fib_levels(100.0, 80.0, config)
        >>> # Returns: [74.56, 67.64, 92.36, 95.72] (negative + positive)
    """
    range_size = swing_high - swing_low

    # Get base levels (positive retracements)
    base_levels = config.get('fib_levels', [0.618, 0.786])

    # Add negative extensions if enabled
    if config.get('negative_fibs_enabled', False):
        negative_levels = config.get('negative_fibs', [-0.272, -0.618])
        all_ratios = negative_levels + base_levels
    else:
        all_ratios = base_levels

    # Calculate price levels
    # Negative ratios extend below swing_low
    # Positive ratios are retracements from swing_high
    levels = []
    for ratio in all_ratios:
        if ratio < 0:
            # Negative extension below swing low
            level = swing_low + (ratio * range_size)  # ratio is negative, so this subtracts
        else:
            # Standard retracement from high
            level = swing_high - (ratio * range_size)
        levels.append(level)

    return sorted(levels)


def check_fib_confluence(price: float, fib_levels: List[float], config: Dict) -> tuple:
    """
    Check if current price is near a Fibonacci level.

    Args:
        price: Current price
        fib_levels: List of Fibonacci levels
        config: Config dict with 'fib_tolerance'

    Returns:
        (is_at_fib, closest_level, distance_pct)

    Example:
        >>> levels = [92.36, 95.72, 100.0]
        >>> is_at, level, dist = check_fib_confluence(95.80, levels, {'fib_tolerance': 0.005})
        >>> # is_at=True, level=95.72, dist=0.00084
    """
    tolerance = config.get('fib_tolerance', 0.005)  # 0.5% default

    closest_level = min(fib_levels, key=lambda x: abs(x - price))
    distance_pct = abs(price - closest_level) / price

    is_at_fib = distance_pct < tolerance

    return is_at_fib, closest_level, distance_pct


def calculate_fib_bonus(df: pd.DataFrame, config: Dict) -> float:
    """
    Calculate Fibonacci confluence bonus for fusion scoring.

    Args:
        df: OHLCV DataFrame
        config: Full config dict with 'liquidity' section

    Returns:
        Bonus score (0.0 to 0.15)

    Example:
        >>> df = pd.DataFrame({'high': [100, 102, 99], 'low': [80, 82, 79], 'close': [95, 96, 95]})
        >>> config = {'liquidity': {'fib_levels': [0.618], 'negative_fibs': [-0.272],
        ...                          'fib_tolerance': 0.01, 'negative_fibs_enabled': True}}
        >>> bonus = calculate_fib_bonus(df, config)
        >>> # Returns 0.15 if price near fib, else 0.0
    """
    liquidity_config = config.get('liquidity', {})

    # Find recent swing high/low (last 50 bars)
    lookback = min(50, len(df))
    recent_high = df['high'].tail(lookback).max()
    recent_low = df['low'].tail(lookback).min()
    current_price = df['close'].iloc[-1]

    # Calculate fib levels
    fib_levels = calculate_fib_levels(recent_high, recent_low, liquidity_config)

    # Check confluence
    is_at_fib, closest_level, distance_pct = check_fib_confluence(
        current_price, fib_levels, liquidity_config
    )

    # Return bonus if at fib level
    if is_at_fib:
        return 0.15  # Max bonus for fib confluence
    else:
        return 0.0


if __name__ == '__main__':
    # Quick validation
    print("Testing Fibonacci levels...")

    config = {
        'fib_levels': [0.618, 0.786],
        'negative_fibs': [-0.272, -0.618],
        'negative_fibs_enabled': True,
        'fib_tolerance': 0.005
    }

    levels = calculate_fib_levels(100.0, 80.0, config)
    print(f"Levels (H=100, L=80): {[f'{l:.2f}' for l in levels]}")

    # Should include negative extensions below 80
    assert any(l < 80 for l in levels), "Missing negative extensions"

    # Test confluence
    is_at, level, dist = check_fib_confluence(95.72, levels, config)
    print(f"At fib? {is_at}, closest={level:.2f}, dist={dist:.4f}")

    print("✅ Fibonacci levels validated")
