"""
Bull Machine v1.5.0 - Negative VIPs (Reversal Awareness)
Volume-Intensive Price (VIP) patterns that signal potential reversals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from bull_machine.core.telemetry import log_telemetry


def detect_volume_spike(df: pd.DataFrame, spike_threshold: float = 1.5, window: int = 20) -> Dict[str, Any]:
    """
    Detect volume spikes that may indicate institutional activity.

    Args:
        df: DataFrame with volume data
        spike_threshold: Multiplier for volume spike detection
        window: Rolling window for average volume calculation

    Returns:
        Dict containing spike analysis
    """
    if len(df) < window + 1:
        return {"detected": False, "ratio": 0.0, "intensity": "none"}

    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].rolling(window=window).mean().iloc[-2]

    if avg_volume == 0:
        return {"detected": False, "ratio": 0.0, "intensity": "none"}

    volume_ratio = current_volume / avg_volume
    spike_detected = volume_ratio >= spike_threshold

    # Classify intensity
    if volume_ratio >= 3.0:
        intensity = "extreme"
    elif volume_ratio >= 2.0:
        intensity = "high"
    elif volume_ratio >= spike_threshold:
        intensity = "moderate"
    else:
        intensity = "none"

    result = {
        "detected": spike_detected,
        "ratio": float(volume_ratio),
        "intensity": intensity,
        "current_volume": float(current_volume),
        "avg_volume": float(avg_volume),
        "threshold": spike_threshold
    }

    return result


def detect_reversal_pattern(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect reversal patterns in price action.

    Common reversal patterns:
    - Price closes below previous bar's low (bearish reversal)
    - Price closes above previous bar's high (bullish reversal)
    - Extended wicks indicating rejection

    Args:
        df: DataFrame with OHLC data

    Returns:
        Dict containing reversal analysis
    """
    if len(df) < 3:
        return {"detected": False, "type": None, "strength": 0.0}

    current = df.iloc[-1]
    previous = df.iloc[-2]

    # Bearish reversal: close below previous low
    bearish_reversal = current["close"] < previous["low"]

    # Bullish reversal: close above previous high
    bullish_reversal = current["close"] > previous["high"]

    # Calculate reversal strength based on penetration depth
    if bearish_reversal:
        strength = (previous["low"] - current["close"]) / previous["low"] if previous["low"] > 0 else 0
        reversal_type = "bearish"
    elif bullish_reversal:
        strength = (current["close"] - previous["high"]) / previous["high"] if previous["high"] > 0 else 0
        reversal_type = "bullish"
    else:
        strength = 0.0
        reversal_type = None

    # Check for wick rejections
    current_range = current["high"] - current["low"]
    if current_range > 0:
        upper_wick = (current["high"] - max(current["open"], current["close"])) / current_range
        lower_wick = (min(current["open"], current["close"]) - current["low"]) / current_range

        # Significant wick indicates rejection
        if upper_wick > 0.3 and not bullish_reversal:
            reversal_type = "bearish_wick"
            strength = max(strength, upper_wick)
        elif lower_wick > 0.3 and not bearish_reversal:
            reversal_type = "bullish_wick"
            strength = max(strength, lower_wick)

    result = {
        "detected": bearish_reversal or bullish_reversal or (reversal_type in ["bearish_wick", "bullish_wick"]),
        "type": reversal_type,
        "strength": float(min(1.0, strength)),
        "bearish": bearish_reversal,
        "bullish": bullish_reversal,
        "wick_analysis": {
            "upper_wick_ratio": upper_wick if 'upper_wick' in locals() else 0.0,
            "lower_wick_ratio": lower_wick if 'lower_wick' in locals() else 0.0
        }
    }

    return result


def calculate_momentum_divergence(df: pd.DataFrame, rsi_period: int = 14) -> Dict[str, Any]:
    """
    Calculate momentum divergence using RSI.

    Divergence occurs when price makes new highs/lows but RSI does not confirm.

    Args:
        df: DataFrame with OHLC data
        rsi_period: Period for RSI calculation

    Returns:
        Dict containing divergence analysis
    """
    if len(df) < rsi_period + 10:
        return {"detected": False, "type": None, "strength": 0.0}

    # Calculate RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    if len(rsi) < 5:
        return {"detected": False, "type": None, "strength": 0.0}

    # Look for divergence in recent bars
    recent_highs = df["high"].iloc[-5:]
    recent_lows = df["low"].iloc[-5:]
    recent_rsi = rsi.iloc[-5:]

    # Bearish divergence: price higher high, RSI lower high
    price_higher_high = recent_highs.iloc[-1] > recent_highs.iloc[0]
    rsi_lower_high = recent_rsi.iloc[-1] < recent_rsi.iloc[0]
    bearish_divergence = price_higher_high and rsi_lower_high

    # Bullish divergence: price lower low, RSI higher low
    price_lower_low = recent_lows.iloc[-1] < recent_lows.iloc[0]
    rsi_higher_low = recent_rsi.iloc[-1] > recent_rsi.iloc[0]
    bullish_divergence = price_lower_low and rsi_higher_low

    if bearish_divergence:
        divergence_type = "bearish"
        strength = abs(recent_rsi.iloc[-1] - recent_rsi.iloc[0]) / 100
    elif bullish_divergence:
        divergence_type = "bullish"
        strength = abs(recent_rsi.iloc[-1] - recent_rsi.iloc[0]) / 100
    else:
        divergence_type = None
        strength = 0.0

    result = {
        "detected": bearish_divergence or bullish_divergence,
        "type": divergence_type,
        "strength": float(strength),
        "current_rsi": float(recent_rsi.iloc[-1]) if len(recent_rsi) > 0 else 0.0,
        "rsi_trend": "declining" if recent_rsi.iloc[-1] < recent_rsi.iloc[0] else "rising"
    }

    return result


def negative_vip_score(df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """
    Calculate Negative VIP (Volume-Intensive Price) score.

    Negative VIPs identify potential reversal points by combining:
    1. Volume spikes (institutional activity)
    2. Reversal price patterns
    3. Momentum divergence

    Higher scores indicate higher reversal probability.

    Args:
        df: DataFrame with OHLC and volume data
        config: Configuration parameters

    Returns:
        float: Negative VIP score (0.0 - 1.0)
    """
    if not config.get("features", {}).get("negative_vip", False):
        return 0.3  # Neutral score when disabled

    # Analyze volume spike
    volume_analysis = detect_volume_spike(df)

    # Analyze reversal patterns
    reversal_analysis = detect_reversal_pattern(df)

    # Analyze momentum divergence
    divergence_analysis = calculate_momentum_divergence(df)

    # Base score
    base_score = 0.3

    # Volume spike contribution
    volume_bonus = 0.0
    if volume_analysis["detected"]:
        intensity_multiplier = {
            "moderate": 0.1,
            "high": 0.2,
            "extreme": 0.3
        }
        volume_bonus = intensity_multiplier.get(volume_analysis["intensity"], 0.0)

    # Reversal pattern contribution
    reversal_bonus = 0.0
    if reversal_analysis["detected"]:
        reversal_bonus = 0.2 * reversal_analysis["strength"]

    # Divergence contribution
    divergence_bonus = 0.0
    if divergence_analysis["detected"]:
        divergence_bonus = 0.15 * divergence_analysis["strength"]

    # Calculate final score
    final_score = min(1.0, base_score + volume_bonus + reversal_bonus + divergence_bonus)

    # Confluence bonus for multiple signals
    signals_detected = sum([
        volume_analysis["detected"],
        reversal_analysis["detected"],
        divergence_analysis["detected"]
    ])

    if signals_detected >= 2:
        final_score = min(1.0, final_score + 0.1)

    # Log telemetry
    log_telemetry("layer_masks.json", {
        "negative_vip": float(final_score),
        "vip_components": {
            "base": float(base_score),
            "volume_bonus": float(volume_bonus),
            "reversal_bonus": float(reversal_bonus),
            "divergence_bonus": float(divergence_bonus)
        },
        "confluence_signals": int(signals_detected)
    })

    return final_score


def analyze_reversal_risk(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive reversal risk analysis.

    Args:
        df: DataFrame with OHLC and volume data
        config: Configuration parameters

    Returns:
        Dict containing comprehensive reversal analysis
    """
    vip_score = negative_vip_score(df, config)
    volume_analysis = detect_volume_spike(df)
    reversal_analysis = detect_reversal_pattern(df)
    divergence_analysis = calculate_momentum_divergence(df)

    # Determine risk level
    if vip_score >= 0.7:
        risk_level = "high"
    elif vip_score >= 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "vip_score": vip_score,
        "risk_level": risk_level,
        "volume_spike": volume_analysis,
        "reversal_pattern": reversal_analysis,
        "momentum_divergence": divergence_analysis,
        "recommendation": "reduce_position" if risk_level == "high" else "monitor"
    }