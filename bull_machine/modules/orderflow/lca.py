"""
Bull Machine v1.5.0 - Orderflow LCA (Liquidity Capture Analysis)
BOS (Break of Structure) detection with intent nudging.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from bull_machine.core.telemetry import log_telemetry


def detect_bos(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """
    Detect Break of Structure (BOS) events.

    A BOS occurs when price breaks above recent swing highs (bullish BOS)
    or below recent swing lows (bearish BOS).

    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to look back for swing detection

    Returns:
        Dict containing BOS detection results
    """
    if len(df) < lookback + 2:
        return {"detected": False, "direction": None, "strength": 0.0}

    # Calculate swing levels
    highs = df["high"].rolling(window=lookback, center=True).max()
    lows = df["low"].rolling(window=lookback, center=True).min()

    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]
    prev_swing_high = highs.iloc[-2] if not pd.isna(highs.iloc[-2]) else df["high"].iloc[-lookback-1]
    prev_swing_low = lows.iloc[-2] if not pd.isna(lows.iloc[-2]) else df["low"].iloc[-lookback-1]

    # Detect bullish BOS (break above previous swing high)
    bullish_bos = current_high > prev_swing_high
    bullish_strength = (current_high - prev_swing_high) / prev_swing_high if prev_swing_high > 0 else 0

    # Detect bearish BOS (break below previous swing low)
    bearish_bos = current_low < prev_swing_low
    bearish_strength = (prev_swing_low - current_low) / prev_swing_low if prev_swing_low > 0 else 0

    bos_result = {
        "detected": bullish_bos or bearish_bos,
        "direction": "bullish" if bullish_bos else ("bearish" if bearish_bos else None),
        "strength": max(bullish_strength, bearish_strength),
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "swing_levels": {
            "prev_high": float(prev_swing_high) if not pd.isna(prev_swing_high) else None,
            "prev_low": float(prev_swing_low) if not pd.isna(prev_swing_low) else None,
            "current_high": float(current_high),
            "current_low": float(current_low)
        }
    }

    log_telemetry("layer_masks.json", {
        "bos_detected": bos_result["detected"],
        "bos_direction": bos_result["direction"],
        "bos_strength": float(bos_result["strength"])
    })

    return bos_result


def detect_liquidity_capture(df: pd.DataFrame) -> bool:
    """
    Detect Liquidity Capture Analysis (LCA) patterns.

    LCA identifies when current close is above the previous bar's high,
    indicating potential liquidity capture above resistance.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if LCA pattern detected
    """
    if len(df) < 2:
        return False

    current_close = df["close"].iloc[-1]
    prev_high = df["high"].iloc[-2]

    # LCA: Current close above previous high
    lca_detected = current_close > prev_high

    log_telemetry("layer_masks.json", {
        "lca_detected": lca_detected,
        "current_close": float(current_close),
        "prev_high": float(prev_high),
        "lca_margin": float(current_close - prev_high) if lca_detected else 0.0
    })

    return lca_detected


def calculate_intent_nudge(df: pd.DataFrame, volume_threshold: float = 1.2) -> Dict[str, Any]:
    """
    Calculate intent nudge based on volume confirmation.

    Intent nudge measures the conviction behind price moves using volume.
    Higher volume during breakouts indicates stronger intent.

    Args:
        df: DataFrame with OHLC and volume data
        volume_threshold: Volume multiplier threshold for confirmation

    Returns:
        Dict containing intent analysis
    """
    if len(df) < 10:
        return {"nudge_score": 0.0, "conviction": "low"}

    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].rolling(window=10).mean().iloc[-2]

    if avg_volume == 0:
        return {"nudge_score": 0.0, "conviction": "low"}

    volume_ratio = current_volume / avg_volume
    nudge_score = min(1.0, volume_ratio / volume_threshold)

    # Determine conviction level
    if volume_ratio >= volume_threshold * 2:
        conviction = "very_high"
    elif volume_ratio >= volume_threshold * 1.5:
        conviction = "high"
    elif volume_ratio >= volume_threshold:
        conviction = "medium"
    else:
        conviction = "low"

    result = {
        "nudge_score": float(nudge_score),
        "volume_ratio": float(volume_ratio),
        "conviction": conviction,
        "current_volume": float(current_volume),
        "avg_volume": float(avg_volume)
    }

    log_telemetry("layer_masks.json", {
        "intent_nudge": result["nudge_score"],
        "volume_conviction": result["conviction"]
    })

    return result


def orderflow_lca(df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """
    Main Orderflow LCA function combining BOS detection and intent analysis.

    Combines:
    1. Liquidity Capture Analysis (LCA) - price above previous highs
    2. Break of Structure (BOS) detection
    3. Intent nudging via volume confirmation

    Args:
        df: Price/volume DataFrame
        config: Configuration parameters

    Returns:
        float: Combined orderflow score (0.0 - 1.0)
    """
    if not config.get("features", {}).get("orderflow_lca", False):
        return 0.5  # Neutral score when disabled

    # Detect LCA pattern
    lca_detected = detect_liquidity_capture(df)

    # Detect BOS
    bos_result = detect_bos(df)

    # Calculate intent nudge
    intent_result = calculate_intent_nudge(df)

    # Combine signals
    base_score = 0.4  # Baseline
    lca_bonus = 0.3 if lca_detected else 0.0
    bos_bonus = 0.2 * bos_result["strength"] if bos_result["detected"] else 0.0
    intent_bonus = 0.1 * intent_result["nudge_score"]

    final_score = min(1.0, base_score + lca_bonus + bos_bonus + intent_bonus)

    # Enhanced scoring for strong confluence
    if lca_detected and bos_result["detected"] and intent_result["conviction"] in ["high", "very_high"]:
        final_score = min(1.0, final_score + 0.1)  # Confluence bonus

    log_telemetry("layer_masks.json", {
        "ofl_lca": float(final_score),
        "ofl_bos": bos_result["detected"],
        "ofl_components": {
            "base": float(base_score),
            "lca_bonus": float(lca_bonus),
            "bos_bonus": float(bos_bonus),
            "intent_bonus": float(intent_bonus)
        }
    })

    return final_score


def analyze_market_structure(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive market structure analysis for orderflow.

    Args:
        df: Price/volume DataFrame
        config: Configuration parameters

    Returns:
        Dict containing comprehensive market structure analysis
    """
    return {
        "lca_score": orderflow_lca(df, config),
        "bos_analysis": detect_bos(df),
        "intent_analysis": calculate_intent_nudge(df),
        "structure_health": "strong" if orderflow_lca(df, config) > 0.7 else "weak"
    }