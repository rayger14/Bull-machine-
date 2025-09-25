"""
Bull Machine v1.5.0 - MTF (Multi-Timeframe) Sync Module
Enhanced MTF synchronization with DL2 filter and 6-candle leg rule
OPTIMIZED VERSION: Timeframe-aware thresholds and reduced impact
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from bull_machine.core.telemetry import log_telemetry
from bull_machine.core.config_loader import is_feature_enabled


def six_candle_structure(df: pd.DataFrame) -> bool:
    """
    6-Candle Leg Rule: Validates structural integrity of price action.

    Looks for alternating pattern in the last 6 candles to confirm
    healthy market structure vs erratic price action.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if valid 6-candle pattern detected
    """
    if len(df) < 6:
        log_telemetry("layer_masks.json", {"mtf_leg_valid": False, "reason": "insufficient_data"})
        return False

    # Get last 6 close prices
    closes = df['close'].iloc[-6:].values

    # Check for alternating pattern (simplified heuristic)
    # Valid if we see some alternation in direction changes
    direction_changes = []
    for i in range(1, len(closes)):
        direction_changes.append(closes[i] > closes[i-1])

    # Count direction changes (should have some alternation)
    changes = sum(1 for i in range(1, len(direction_changes))
                  if direction_changes[i] != direction_changes[i-1])

    # Valid if we have at least 2 direction changes in 6 candles
    valid = changes >= 2

    log_telemetry("layer_masks.json", {
        "mtf_leg_valid": valid,
        "direction_changes": changes,
        "closes_pattern": closes.tolist()
    })

    return valid


def mtf_dl2_filter(df: pd.DataFrame, timeframe: str = "") -> bool:
    """
    MTF DL2 (Deviation Level 2) Filter for extreme market conditions.

    Filters out signals during periods of extreme price deviation that could
    indicate market manipulation or data errors. Now supports timeframe-specific thresholds.

    Args:
        df: DataFrame with OHLC data
        timeframe: Timeframe string ("1D", "4H", "1H", etc.) for adaptive thresholds

    Returns:
        bool: True if market conditions are normal, False if extreme deviation detected
    """
    if len(df) < 20:
        return True  # Not enough data, allow trade

    # Calculate z-score of current price vs recent average
    recent_close = df['close'].iloc[-20:]
    current_price = df['close'].iloc[-1]

    mean_price = recent_close.mean()
    std_price = recent_close.std()

    if std_price == 0:
        return True  # No volatility, allow trade

    z_score = abs((current_price - mean_price) / std_price)

    # Adaptive threshold based on timeframe (OPTIMIZED: Relaxed for 4H)
    if "4H" in str(timeframe).upper():
        dl2_threshold = 2.5  # More lenient for 4H
    elif "1D" in str(timeframe).upper():
        dl2_threshold = 2.2  # Slightly relaxed for daily
    else:
        dl2_threshold = 2.0  # Default for other timeframes

    # Check if deviation exceeds threshold
    dl2_ok = z_score <= dl2_threshold

    # Log telemetry
    log_telemetry("layer_masks.json", {
        "mtf_dl2_deviation": float(z_score),
        "mtf_dl2_ok": dl2_ok,
        "dl2_threshold": dl2_threshold,
        "timeframe": timeframe
    })

    return dl2_ok


def enhanced_mtf_sync(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced MTF synchronization with v1.5.0 filters.

    Applies MTF DL2 filter and 6-candle leg rule based on configuration.

    Args:
        df: DataFrame with OHLC data
        config: Configuration dictionary with feature flags

    Returns:
        dict: MTF sync result with decision and filter details
    """
    result = {
        "mtf_decision": "ALLOW",
        "veto_reason": None,
        "filters_applied": [],
        "filter_results": {}
    }

    # Apply 6-Candle Leg Rule if enabled
    if is_feature_enabled(config, "six_candle_leg"):
        result["filters_applied"].append("six_candle_leg")
        leg_valid = six_candle_structure(df)
        result["filter_results"]["six_candle_leg"] = leg_valid

        if not leg_valid:
            result["mtf_decision"] = "VETO"
            result["veto_reason"] = "6-candle structure invalid"
            return result

    # Apply MTF DL2 Filter if enabled
    if is_feature_enabled(config, "mtf_dl2"):
        result["filters_applied"].append("mtf_dl2")
        timeframe = config.get("timeframe", "")
        dl2_ok = mtf_dl2_filter(df, timeframe)
        result["filter_results"]["mtf_dl2"] = dl2_ok

        if not dl2_ok:
            result["mtf_decision"] = "VETO"
            result["veto_reason"] = "MTF DL2 deviation too high"
            return result

    return result


def calculate_mtf_alignment_score(htf_bias: str, mtf_bias: str, ltf_bias: str) -> float:
    """
    Calculate MTF alignment score based on bias agreement across timeframes.

    Args:
        htf_bias: Higher timeframe bias ("long", "short", "neutral")
        mtf_bias: Middle timeframe bias
        ltf_bias: Lower timeframe bias

    Returns:
        float: Alignment score (0.0 to 1.0)
    """
    biases = [htf_bias, mtf_bias, ltf_bias]

    # Count non-neutral biases
    non_neutral = [b for b in biases if b != "neutral"]

    if len(non_neutral) == 0:
        return 0.2  # All neutral = poor alignment

    # Perfect alignment (all same direction)
    if len(set(non_neutral)) == 1 and len(non_neutral) == 3:
        return 1.0

    # Strong alignment (2/3 match)
    if len(non_neutral) >= 2:
        # Count how many match the most common bias
        from collections import Counter
        counts = Counter(non_neutral)
        most_common_count = counts.most_common(1)[0][1]

        if most_common_count >= 2:
            return 0.7  # Strong alignment

    # Mixed signals with neutrals
    return 0.2  # Poor alignment


# Legacy compatibility - kept for existing code
def mtf_dl2_filter_legacy(df: pd.DataFrame, dl2_threshold: float = 2.0) -> bool:
    """Legacy version with fixed threshold - kept for compatibility."""
    return mtf_dl2_filter(df, "")  # Use default threshold logic