"""
Bull Machine v1.5.0 - MTF Enhancements
- MTF DL2 Filter: Deviation limit filter to prevent extreme signals
- 6-Candle Leg Rule: Structure validation across recent price action
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from bull_machine.core.telemetry import log_telemetry


def six_candle_structure(df: pd.DataFrame) -> bool:
    """
    6-Candle Leg Rule: Validates alternating price structure
    across the last 6 candles to ensure proper market structure.

    Logic: c[0] < c[1] and c[2] > c[3] and c[4] < c[5]
    This creates a zigzag pattern indicating healthy price discovery.

    Args:
        df: DataFrame with 'close' column

    Returns:
        bool: True if 6-candle structure is valid
    """
    if len(df) < 6:
        return False

    closes = df["close"].iloc[-6:]
    c = closes.values

    # Alternating pattern: down, up, down, up, down, up
    valid = (c[0] < c[1]) and (c[2] > c[3]) and (c[4] < c[5])

    log_telemetry("layer_masks.json", {
        "mtf_leg_valid": valid,
        "six_candle_pattern": [float(x) for x in c],
        "leg_conditions": {
            "c0_lt_c1": c[0] < c[1],
            "c2_gt_c3": c[2] > c[3],
            "c4_lt_c5": c[4] < c[5]
        }
    })

    return valid


def mtf_dl2_filter(df: pd.DataFrame, timeframe: str = "") -> bool:
    """
    MTF DL2 Filter: Deviation Limit Level 2
    Prevents signals when price is too far from mean (in std deviations).

    This filter helps avoid entries during extreme market conditions
    where mean reversion is likely.

    Args:
        df: DataFrame with 'close' column
        timeframe: Timeframe string (e.g., "4H", "1D") for dynamic thresholds

    Returns:
        bool: True if price is within acceptable deviation range
    """
    if len(df) < 20:  # Need sufficient data for mean/std calculation
        return True

    # Dynamic threshold based on timeframe
    if str(timeframe).upper() == "4H":
        dl2_threshold = 2.5  # More lenient for 4H
    elif str(timeframe).upper() == "1D":
        dl2_threshold = 2.2  # Slightly relaxed for daily
    else:
        dl2_threshold = 2.0  # Default for other timeframes

    closes = df["close"]
    mean_price = closes.mean()
    std_price = closes.std()

    if std_price == 0:
        return True  # No volatility, allow signal

    current_price = closes.iloc[-1]
    z_score = (current_price - mean_price) / std_price

    deviation_ok = abs(z_score) <= dl2_threshold

    log_telemetry("layer_masks.json", {
        "mtf_dl2_deviation": float(z_score),
        "mtf_dl2_threshold": dl2_threshold,
        "mtf_dl2_ok": deviation_ok,
        "timeframe": timeframe,
        "price_stats": {
            "current": float(current_price),
            "mean": float(mean_price),
            "std": float(std_price)
        }
    })

    return deviation_ok


def enhanced_mtf_sync(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced MTF synchronization with v1.5.0 filters.

    Combines traditional MTF analysis with new filters:
    - 6-candle leg structure validation
    - DL2 deviation filtering

    Args:
        df: Price DataFrame
        config: Configuration with feature flags

    Returns:
        Dict containing MTF analysis results and filter status
    """
    result = {
        "mtf_decision": "ALLOW",
        "filters_applied": [],
        "filter_results": {}
    }

    # Apply 6-candle leg rule if enabled
    if config.get("features", {}).get("six_candle_leg", False):
        leg_valid = six_candle_structure(df)
        result["filters_applied"].append("six_candle_leg")
        result["filter_results"]["six_candle_leg"] = leg_valid

        if not leg_valid:
            result["mtf_decision"] = "VETO"
            result["veto_reason"] = "6-candle structure invalid"

    # Apply MTF DL2 filter if enabled
    if config.get("features", {}).get("mtf_dl2", False):
        dl2_threshold = config.get("mtf_dl2_threshold", 2.0)
        dl2_ok = mtf_dl2_filter(df, dl2_threshold)
        result["filters_applied"].append("mtf_dl2")
        result["filter_results"]["mtf_dl2"] = dl2_ok

        if not dl2_ok:
            result["mtf_decision"] = "VETO"
            result["veto_reason"] = "MTF DL2 deviation too high"

    # Log comprehensive MTF results
    log_telemetry("mtf_sync.json", {
        "mtf_enhanced": True,
        "decision": result["mtf_decision"],
        "filters": result["filters_applied"],
        "results": result["filter_results"]
    })

    return result


def calculate_mtf_alignment_score(htf_bias: str, mtf_bias: str, ltf_bias: str) -> float:
    """
    Calculate alignment score between timeframes.

    Args:
        htf_bias: Higher timeframe bias ('long'/'short'/'neutral')
        mtf_bias: Medium timeframe bias
        ltf_bias: Lower timeframe bias

    Returns:
        Alignment score between 0-1
    """
    biases = [htf_bias, mtf_bias, ltf_bias]

    # Count matching biases
    long_count = biases.count('long')
    short_count = biases.count('short')
    neutral_count = biases.count('neutral')

    # Perfect alignment
    if long_count == 3 or short_count == 3:
        return 1.0

    # Strong alignment (2/3 match)
    if long_count == 2 or short_count == 2:
        return 0.7

    # Weak alignment
    if neutral_count == 0:  # Mixed but no neutrals
        return 0.4

    # Poor alignment
    return 0.2