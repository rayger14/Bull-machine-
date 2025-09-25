"""
Bull Machine Core Numerics - Safe calculations for trading metrics
"""

import numpy as np
import pandas as pd

EPS = 1e-12

def safe_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range safely handling edge cases.

    Args:
        df: DataFrame with OHLC data

    Returns:
        Series with True Range values
    """
    pc = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - pc).abs(),
        (df['low'] - pc).abs()
    ], axis=1).max(axis=1)

    return tr.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def safe_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range with safe handling of edge cases.

    Args:
        df: DataFrame with OHLC data
        window: ATR calculation window

    Returns:
        Series with ATR values, never zero or NaN
    """
    tr = safe_true_range(df)
    atr = tr.rolling(window, min_periods=max(2, window // 3)).mean()

    # Replace zeros and NaNs with small epsilon to prevent division by zero
    return atr.replace(0.0, EPS).fillna(EPS)