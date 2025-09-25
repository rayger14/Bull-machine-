"""
Bull Machine - Liquidity Sweep Detection
Moneytaur-inspired liquidity analysis and order block ranking
"""

import pandas as pd
import numpy as np
from typing import Dict
from bull_machine.core.telemetry import log_telemetry

def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Detect liquidity sweep events.

    A sweep occurs when price breaks below recent lows with high volume,
    typically indicating stop-loss hunting before reversal.

    Args:
        df: Price DataFrame with OHLCV data
        lookback: Period for finding recent lows

    Returns:
        bool: True if liquidity sweep detected
    """
    if len(df) < lookback + 5:
        return False

    current_low = df['low'].iloc[-1]
    current_volume = df['volume'].iloc[-1]

    # Find the lowest low in the lookback period (excluding current bar)
    recent_low = df['low'].iloc[-(lookback+1):-1].min()

    # Volume analysis
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    volume_spike = current_volume > avg_volume * 1.5

    # Sweep detection: current low breaks recent low with volume
    sweep_detected = current_low < recent_low and volume_spike

    # Additional confirmation: quick reversal (price back above recent low)
    if sweep_detected and len(df) > 1:
        next_close = df['close'].iloc[-1]
        quick_reversal = next_close > recent_low

        log_telemetry("layer_masks.json", {
            "liquidity_sweep": sweep_detected,
            "quick_reversal": quick_reversal,
            "current_low": current_low,
            "recent_low": recent_low,
            "volume_spike": volume_spike,
            "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 0
        })

        return sweep_detected

    log_telemetry("layer_masks.json", {
        "liquidity_sweep": sweep_detected,
        "current_low": current_low,
        "recent_low": recent_low,
        "volume_spike": volume_spike
    })

    return sweep_detected

def rank_orderblocks(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Rank order block strength based on volume and price action.

    Args:
        df: Price DataFrame with OHLCV data
        lookback: Period for analysis

    Returns:
        float: Order block strength score (0.0 to 1.0)
    """
    if len(df) < lookback + 10:
        return 0.3

    # Identify potential order block levels
    recent_high = df['high'].rolling(lookback).max().iloc[-1]
    recent_low = df['low'].rolling(lookback).min().iloc[-1]
    current_price = df['close'].iloc[-1]

    # Volume analysis
    v20 = df['volume'].rolling(20).mean().iloc[-1]
    v100 = df['volume'].rolling(100).mean().iloc[-1] if len(df) >= 100 else v20

    volume_strength = v20 / v100 if v100 > 0 else 1.0

    # Distance from key levels
    distance_to_high = abs(current_price - recent_high) / current_price
    distance_to_low = abs(current_price - recent_low) / current_price

    # Order block strength factors
    proximity_score = 1.0 - min(distance_to_high, distance_to_low) * 10  # Closer = stronger
    proximity_score = max(0.0, min(1.0, proximity_score))

    volume_score = min(volume_strength * 0.5, 0.7)  # Volume confirmation

    # Combine scores
    orderblock_score = (proximity_score * 0.6) + (volume_score * 0.4)
    orderblock_score = max(0.0, min(1.0, orderblock_score))

    log_telemetry("layer_masks.json", {
        "orderblock_score": orderblock_score,
        "proximity_score": proximity_score,
        "volume_score": volume_score,
        "volume_strength": volume_strength,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "current_price": current_price
    })

    return orderblock_score

def fvg_quality(df: pd.DataFrame) -> float:
    """
    Analyze Fair Value Gap (FVG) quality.

    Args:
        df: Price DataFrame with OHLCV data

    Returns:
        float: FVG quality score (0.0 to 1.0)
    """
    if len(df) < 5:
        return 0.3

    # Look for 3-candle FVG pattern
    # FVG exists when candle 1 high < candle 3 low (bullish) or candle 1 low > candle 3 high (bearish)

    c1_high = df['high'].iloc[-3]
    c1_low = df['low'].iloc[-3]
    c2_high = df['high'].iloc[-2]
    c2_low = df['low'].iloc[-2]
    c3_high = df['high'].iloc[-1]
    c3_low = df['low'].iloc[-1]

    # Bullish FVG
    bullish_fvg = c1_high < c3_low
    # Bearish FVG
    bearish_fvg = c1_low > c3_high

    fvg_detected = bullish_fvg or bearish_fvg

    if fvg_detected:
        # Gap size relative to price
        if bullish_fvg:
            gap_size = (c3_low - c1_high) / c1_high
        else:
            gap_size = (c1_low - c3_high) / c3_high

        # Volume confirmation
        c2_volume = df['volume'].iloc[-2]
        avg_volume = df['volume'].rolling(10).mean().iloc[-2]
        volume_confirmation = c2_volume > avg_volume * 1.2

        # Quality score based on gap size and volume
        base_score = min(gap_size * 20, 0.6)  # Larger gaps = higher quality
        volume_bonus = 0.2 if volume_confirmation else 0.0

        quality_score = base_score + volume_bonus
    else:
        quality_score = 0.3

    quality_score = max(0.0, min(1.0, quality_score))

    log_telemetry("layer_masks.json", {
        "fvg_quality": quality_score,
        "fvg_detected": fvg_detected,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "gap_size": gap_size if fvg_detected else 0
    })

    return quality_score