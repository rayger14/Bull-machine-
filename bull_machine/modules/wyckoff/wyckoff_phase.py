"""
Bull Machine - Wyckoff Phase Detection
M1/M2 phase analysis for spring/markup identification
"""

import pandas as pd
import numpy as np
from typing import Dict
from bull_machine.core.telemetry import log_telemetry

def wyckoff_phase_scores(df: pd.DataFrame, tf: str = "") -> float:
    """
    Detect Wyckoff M1/M2 phases (simplified implementation).

    Args:
        df: Price DataFrame with OHLCV data
        tf: Timeframe string for logging

    Returns:
        float: Phase score (0.0 to 1.0)
    """
    if len(df) < 25:
        return 0.3  # Neutral score for insufficient data

    current_price = df['close'].iloc[-1]

    # Spring detection (Phase C): Current low breaks previous 20-bar low
    low_20_lookback = df['low'].rolling(20).min().shift(1).iloc[-1]
    spring_detected = current_price < low_20_lookback

    # Markup detection (Phase E): Current high breaks previous 20-bar high
    high_20_lookback = df['high'].rolling(20).max().shift(1).iloc[-1]
    markup_detected = current_price > high_20_lookback

    # Volume confirmation
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    volume_spike = current_volume > avg_volume * 1.3

    # Calculate phase score
    if spring_detected and volume_spike:
        score = 0.8  # Strong spring with volume
    elif markup_detected and volume_spike:
        score = 0.75  # Strong markup with volume
    elif spring_detected or markup_detected:
        score = 0.6  # Phase detected without volume confirmation
    else:
        # Check for accumulation/distribution patterns
        price_range = df['high'].rolling(20).max().iloc[-1] - df['low'].rolling(20).min().iloc[-1]
        recent_range = df['high'].iloc[-5:].max() - df['low'].iloc[-5:].min()

        if recent_range < price_range * 0.3:
            score = 0.45  # Possible accumulation/distribution
        else:
            score = 0.3  # No clear phase

    # Log telemetry
    log_telemetry("layer_masks.json", {
        "wyckoff_phase_score": score,
        "timeframe": tf,
        "spring_detected": spring_detected,
        "markup_detected": markup_detected,
        "volume_spike": volume_spike,
        "current_price": current_price,
        "low_20_back": low_20_lookback,
        "high_20_back": high_20_lookback
    })

    return score

def wyckoff_trend_strength(df: pd.DataFrame, window: int = 50) -> float:
    """
    Calculate Wyckoff trend strength for context.

    Args:
        df: Price DataFrame
        window: Lookback window for trend analysis

    Returns:
        float: Trend strength (-1.0 to 1.0)
    """
    if len(df) < window:
        return 0.0

    # Price trend
    start_price = df['close'].iloc[-window]
    end_price = df['close'].iloc[-1]
    price_trend = (end_price - start_price) / start_price

    # Volume trend
    early_volume = df['volume'].iloc[-window:-window//2].mean()
    recent_volume = df['volume'].iloc[-window//2:].mean()
    volume_trend = (recent_volume - early_volume) / early_volume if early_volume > 0 else 0

    # Combine trends (Wyckoff: volume should confirm price)
    if price_trend > 0 and volume_trend > 0:
        strength = min(price_trend * 2, 1.0)  # Bullish with volume
    elif price_trend < 0 and volume_trend > 0:
        strength = max(price_trend * 2, -1.0)  # Bearish with volume
    else:
        strength = price_trend * 0.5  # Weak trend without volume confirmation

    return np.clip(strength, -1.0, 1.0)