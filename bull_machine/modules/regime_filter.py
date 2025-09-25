"""
Bull Machine v1.5.1 - Enhanced Regime Filter
Volume and volatility regime detection with configurable thresholds
"""

import pandas as pd
from typing import Dict


def regime_filter(df: pd.DataFrame, tf: str = "") -> bool:
    """
    Legacy regime filter for backward compatibility.
    """
    if len(df) < 100:
        return True

    v20 = df['volume'].rolling(20).mean().iloc[-1]
    v100 = df['volume'].rolling(100).mean().iloc[-1]

    if v100 == 0 or pd.isna(v100):
        return True

    vol_ratio = v20 / v100

    if str(tf).upper() == "4H":
        threshold = 1.0
    else:
        threshold = 1.2

    return vol_ratio >= threshold


def regime_ok(df: pd.DataFrame, tf: str, regime_cfg: Dict) -> bool:
    """
    Enhanced regime filter with volume and volatility gates.

    Args:
        df: DataFrame with OHLCV data
        tf: Timeframe string (e.g., "4H", "1D")
        regime_cfg: Configuration dict with volume/volatility thresholds

    Returns:
        bool: True if regime is favorable for trading
    """
    if len(df) < 100:
        return True  # Not enough data, allow trading

    # Volume regime check
    v20 = df['volume'].rolling(20).mean().iloc[-1]
    v100 = df['volume'].rolling(100).mean().iloc[-1]
    vol_ratio = v20 / (v100 + 1e-9)  # Avoid division by zero

    # Volatility regime check (ATR percentage)
    high_14 = df['high'].rolling(14).max().iloc[-1]
    low_14 = df['low'].rolling(14).min().iloc[-1]
    current_price = df['close'].iloc[-1]
    atr_pct = (high_14 - low_14) / current_price if current_price > 0 else 0

    # Configurable thresholds
    min_vol_ratio = float(regime_cfg.get("vol_ratio_min", 1.2 if tf == "1D" else 1.0))
    max_atr_pct = float(regime_cfg.get("atr_pct_max", 0.05))

    # Both conditions must be met
    volume_ok = vol_ratio >= min_vol_ratio
    volatility_ok = atr_pct <= max_atr_pct

    return volume_ok and volatility_ok