"""
Bull Machine v1.5.0 - Regime Filter
Volume regime detection with timeframe-aware thresholds
"""

import pandas as pd
from typing import Optional


def regime_filter(df: pd.DataFrame, tf: str = "") -> bool:
    """
    Check if current volume regime supports trading.

    Args:
        df: DataFrame with 'volume' column
        tf: Timeframe string (e.g., "4H", "1D")

    Returns:
        bool: True if volume regime is favorable for trading
    """
    if len(df) < 100:
        return True  # Not enough data, allow trading

    # Calculate volume moving averages
    v20 = df['volume'].rolling(20).mean().iloc[-1]
    v100 = df['volume'].rolling(100).mean().iloc[-1]

    # Avoid division by zero
    if v100 == 0 or pd.isna(v100):
        return True

    vol_ratio = v20 / v100

    # Timeframe-specific thresholds
    if str(tf).upper() == "4H":
        threshold = 1.0  # More lenient for 4H
    else:
        threshold = 1.2  # Standard threshold

    return vol_ratio >= threshold