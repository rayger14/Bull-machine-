"""Bull Machine v1.3 - Core Utility Functions"""

from typing import Dict, List
import pandas as pd


def extract_key_levels(liquidity_result) -> List[Dict]:
    """
    Extract key price levels from liquidity analysis.

    Returns:
        List of dictionaries with:
        - 'type': 'fvg'|'ob'|'phob'|'sweep'
        - 'price': float (mid-price for ranges, exact for points)
        - 'direction': 'bullish'|'bearish'|'neutral'
    """
    levels = []

    # Extract FVGs
    if hasattr(liquidity_result, "fvgs"):
        for fvg in liquidity_result.fvgs:
            levels.append(
                {
                    "type": "fvg",
                    "price": (fvg["high"] + fvg["low"]) / 2,  # Mid-price
                    "direction": fvg.get("direction", "neutral"),
                }
            )

    # Extract Order Blocks
    if hasattr(liquidity_result, "order_blocks"):
        for ob in liquidity_result.order_blocks:
            levels.append(
                {
                    "type": "ob",
                    "price": (ob["high"] + ob["low"]) / 2,  # Mid-price
                    "direction": ob.get("direction", "neutral"),
                }
            )

    # Extract pHOBs (premium/discount HOBs)
    if hasattr(liquidity_result, "phobs"):
        for phob in liquidity_result.phobs:
            levels.append(
                {
                    "type": "phob",
                    "price": phob["price"],
                    "direction": phob.get("direction", "neutral"),
                }
            )

    # Extract sweep levels
    if hasattr(liquidity_result, "sweeps"):
        for sweep in liquidity_result.sweeps:
            levels.append(
                {
                    "type": "sweep",
                    "price": sweep["level"],
                    "direction": "bullish" if sweep.get("reclaimed") else "bearish",
                }
            )

    # Sort by price for easier processing
    levels.sort(key=lambda x: x["price"])

    return levels


def calculate_range_position(price: float, low: float, high: float) -> float:
    """
    Calculate position within range (0=bottom, 0.5=middle, 1=top).

    Args:
        price: Current price
        low: Range low
        high: Range high

    Returns:
        Float between 0 and 1 representing position in range
    """
    if high <= low:
        return 0.5

    position = (price - low) / (high - low)
    return max(0.0, min(1.0, position))


def is_premium_zone(price: float, low: float, high: float, threshold: float = 0.618) -> bool:
    """Check if price is in premium zone (above threshold)"""
    position = calculate_range_position(price, low, high)
    return position >= threshold


def is_discount_zone(price: float, low: float, high: float, threshold: float = 0.382) -> bool:
    """Check if price is in discount zone (below threshold)"""
    position = calculate_range_position(price, low, high)
    return position <= threshold


def is_equilibrium_zone(
    price: float,
    low: float,
    high: float,
    lower_threshold: float = 0.45,
    upper_threshold: float = 0.55,
) -> bool:
    """Check if price is in equilibrium zone (middle)"""
    position = calculate_range_position(price, low, high)
    return lower_threshold <= position <= upper_threshold


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(df) < period + 1:
        return df["high"].tail(period).mean() - df["low"].tail(period).mean()

    high = df["high"].tail(period + 1)
    low = df["low"].tail(period + 1)
    close = df["close"].tail(period + 1)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    return tr.tail(period).mean()
