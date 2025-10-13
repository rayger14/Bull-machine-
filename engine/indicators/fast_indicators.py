"""
Numba-accelerated technical indicators for Bull Machine v1.8.

Phase 2 Performance Optimization: Use JIT compilation for 10-100× speedup.
These implementations replace pandas rolling operations with pure NumPy + Numba.
"""

import numpy as np
from numba import njit
import pandas as pd
from typing import Tuple


@njit
def calc_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int = 14) -> np.ndarray:
    """
    Calculate ATR (Average True Range) using Numba JIT.

    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        period: ATR period (default 14)

    Returns:
        ATR values array (same length as input)
    """
    n = len(high)
    tr = np.zeros(n)
    atr = np.zeros(n)

    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR using SMA for initial, then EMA
    if n < period:
        return atr

    # Initial ATR = SMA of TR
    atr[period-1] = np.mean(tr[1:period])

    # Subsequent ATR = (prior ATR * (period-1) + current TR) / period
    # This is the Wilder's smoothing method
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr


@njit
def calc_rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate RSI (Relative Strength Index) using Numba JIT.

    Args:
        close: Close prices array
        period: RSI period (default 14)

    Returns:
        RSI values array (same length as input, 0-100 scale)
    """
    n = len(close)
    rsi = np.zeros(n)

    if n < period + 1:
        return rsi

    # Calculate price changes
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]

    # Separate gains and losses
    gain = np.zeros(n)
    loss = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            gain[i] = delta[i]
        else:
            loss[i] = -delta[i]

    # Initial average gain/loss using SMA
    avg_gain = np.mean(gain[1:period+1])
    avg_loss = np.mean(loss[1:period+1])

    # Calculate RSI for initial period
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate RSI using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit
def calc_adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ADX (Average Directional Index) using Numba JIT.

    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        period: ADX period (default 14)

    Returns:
        Tuple of (ADX, +DI, -DI) arrays
    """
    n = len(high)
    adx = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    if n < period * 2:
        return adx, plus_di, minus_di

    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        # True Range
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Calculate rolling means (SMA) to match pandas implementation
    # This matches the pandas .rolling(period).mean() behavior exactly
    # Pandas rolling(14).mean() at index i uses values [i-13:i+1] (inclusive on both ends)
    atr_sma = np.zeros(n)
    plus_dm_sma = np.zeros(n)
    minus_dm_sma = np.zeros(n)

    # Calculate SMA for each position (start at period-1 to match pandas)
    for i in range(period - 1, n):
        # For rolling(14): at i=13, use [0:14], at i=14 use [1:15], etc.
        start_idx = max(0, i - period + 1)
        atr_sma[i] = np.mean(tr[start_idx:i+1])
        plus_dm_sma[i] = np.mean(plus_dm[start_idx:i+1])
        minus_dm_sma[i] = np.mean(minus_dm[start_idx:i+1])

    # Calculate Directional Indicators
    for i in range(period, n):
        if atr_sma[i] > 0:
            plus_di[i] = 100.0 * plus_dm_sma[i] / atr_sma[i]
            minus_di[i] = 100.0 * minus_dm_sma[i] / atr_sma[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0

    # Calculate DX (Directional Index)
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0

    # Calculate ADX as SMA of DX (matches pandas .rolling().mean())
    for i in range(period * 2 - 1, n):
        adx[i] = np.mean(dx[i-period+1:i+1])

    return adx, plus_di, minus_di


# Pandas wrapper functions for easy integration

def calc_atr_fast(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR using Numba (pandas wrapper).

    Args:
        df: DataFrame with high/low/close columns (any case)
        period: ATR period (default 14)

    Returns:
        ATR values as pandas Series
    """
    # Standardize column names
    high = df['high'].values if 'high' in df.columns else df['High'].values
    low = df['low'].values if 'low' in df.columns else df['Low'].values
    close = df['close'].values if 'close' in df.columns else df['Close'].values

    atr_values = calc_atr_numba(high, low, close, period)
    return pd.Series(atr_values, index=df.index)


def calc_rsi_fast(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate RSI using Numba (pandas wrapper).

    Args:
        df: DataFrame with close column (any case)
        period: RSI period (default 14)

    Returns:
        RSI values as pandas Series (0-100 scale)
    """
    close = df['close'].values if 'close' in df.columns else df['Close'].values
    rsi_values = calc_rsi_numba(close, period)
    return pd.Series(rsi_values, index=df.index)


def calc_adx_fast(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX using Numba (pandas wrapper).

    Args:
        df: DataFrame with high/low/close columns (any case)
        period: ADX period (default 14)

    Returns:
        Tuple of (ADX, +DI, -DI) as pandas Series
    """
    high = df['high'].values if 'high' in df.columns else df['High'].values
    low = df['low'].values if 'low' in df.columns else df['Low'].values
    close = df['close'].values if 'close' in df.columns else df['Close'].values

    adx_values, plus_di_values, minus_di_values = calc_adx_numba(high, low, close, period)

    return (
        pd.Series(adx_values, index=df.index),
        pd.Series(plus_di_values, index=df.index),
        pd.Series(minus_di_values, index=df.index)
    )


def get_adx_scalar(df: pd.DataFrame, period: int = 14) -> float:
    """
    Get current ADX value as scalar (replacement for calc_adx in fast_signals.py).

    Args:
        df: DataFrame with high/low/close columns
        period: ADX period (default 14)

    Returns:
        Current ADX value (last bar)
    """
    adx, _, _ = calc_adx_fast(df, period)
    return adx.iloc[-1] if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else 0.0
