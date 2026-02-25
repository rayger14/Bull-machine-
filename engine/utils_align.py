#!/usr/bin/env python3
"""
HTF → 1H Alignment Utilities

Provides functions for:
1. Resampling 1H OHLCV data to higher timeframes (4H, 1D)
2. Aligning higher timeframe feature values back to 1H bars
3. Point-in-time correctness (forward-fill pattern)

Used by:
- bin/build_mtf_feature_store.py (feature store builder)
- bin/patch_feature_columns.py (column patching tool)

Architecture:
- resample_to_timeframe(): 1H → HTF OHLCV aggregation
- align_htf_to_1h(): HTF features → 1H bars (forward-fill)
- validate_alignment(): Check for lookahead bias

Example:
    # Resample 1H data to 4H
    df_4h = resample_to_timeframe(df_1h, '4H')

    # Calculate feature on 4H timeframe
    df_4h['boms_displacement'] = calculate_boms(df_4h)

    # Align back to 1H bars
    df_1h = align_htf_to_1h(
        df_1h=df_1h,
        df_htf=df_4h[['boms_displacement']],
        htf='4H',
        columns=['boms_displacement']
    )
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    base_timeframe: str = '1H'
) -> pd.DataFrame:
    """
    Resample OHLCV data from base timeframe to higher timeframe.

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        timeframe: Target timeframe ('4H', '1D', '1W', etc.)
        base_timeframe: Source timeframe (default '1H')

    Returns:
        Resampled DataFrame with OHLCV aggregation

    Notes:
        - Open: first value in period
        - High: maximum value in period
        - Low: minimum value in period
        - Close: last value in period
        - Volume: sum of volume in period
        - Uses closed='right', label='right' for point-in-time correctness

    Example:
        >>> df_1h = pd.DataFrame({
        ...     'open': [100, 101, 102, 103],
        ...     'close': [101, 102, 103, 104],
        ...     'volume': [1000, 1100, 1200, 1300]
        ... }, index=pd.date_range('2024-01-01', periods=4, freq='1H'))
        >>> df_4h = resample_to_timeframe(df_1h, '4H')
        >>> len(df_4h)
        1
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame index must be DatetimeIndex, got {type(df.index)}")

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

    # Resample with point-in-time correctness
    # closed='right': include right edge of interval (bar completes at timestamp)
    # label='right': label with right edge timestamp
    resampled = df.resample(timeframe, closed='right', label='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Drop any incomplete periods (NaN close means period not complete)
    resampled = resampled.dropna(subset=['close'])

    logger.debug(f"Resampled {base_timeframe} → {timeframe}: {len(df)} → {len(resampled)} bars")

    return resampled


def align_htf_to_1h(
    df_1h: pd.DataFrame,
    df_htf: pd.DataFrame,
    htf: str,
    columns: List[str],
    prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Align higher timeframe feature values to 1H bars using forward-fill.

    Args:
        df_1h: 1H DataFrame to receive aligned features
        df_htf: HTF DataFrame with features to align
        htf: Higher timeframe label ('4H', '1D', etc.)
        columns: List of column names to align from df_htf
        prefix: Optional prefix for aligned columns (e.g., 'tf4h_')

    Returns:
        df_1h with new aligned columns added

    Notes:
        - Uses reindex + ffill to propagate HTF values to 1H bars
        - HTF value applies to all 1H bars in that HTF period
        - No lookahead bias: HTF bar timestamp = completion time

    Example:
        >>> df_4h = pd.DataFrame({
        ...     'boms_displacement': [100.0, 200.0]
        ... }, index=pd.date_range('2024-01-01 04:00', periods=2, freq='4H'))
        >>> df_1h = pd.DataFrame({
        ...     'close': [50, 51, 52, 53, 54, 55, 56, 57]
        ... }, index=pd.date_range('2024-01-01', periods=8, freq='1H'))
        >>> df_1h = align_htf_to_1h(df_1h, df_4h, '4H', ['boms_displacement'], prefix='tf4h_')
        >>> df_1h['tf4h_boms_displacement'].iloc[4]
        100.0
    """
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        raise ValueError(f"df_1h index must be DatetimeIndex, got {type(df_1h.index)}")

    if not isinstance(df_htf.index, pd.DatetimeIndex):
        raise ValueError(f"df_htf index must be DatetimeIndex, got {type(df_htf.index)}")

    missing_cols = [col for col in columns if col not in df_htf.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_htf: {missing_cols}")

    # Extract only requested columns from HTF
    df_htf_subset = df_htf[columns].copy()

    # Reindex to 1H frequency and forward-fill
    # This propagates each HTF value to all 1H bars in that HTF period
    df_aligned = df_htf_subset.reindex(df_1h.index, method='ffill')

    # Add prefix if specified
    if prefix:
        df_aligned.columns = [f"{prefix}{col}" for col in df_aligned.columns]

    # Merge aligned columns back to df_1h
    for col in df_aligned.columns:
        df_1h[col] = df_aligned[col]

    logger.debug(f"Aligned {len(columns)} columns from {htf} to 1H: {columns}")

    return df_1h


def validate_alignment(
    df_1h: pd.DataFrame,
    df_htf: pd.DataFrame,
    htf: str,
    column: str
) -> dict:
    """
    Validate HTF alignment for lookahead bias and correctness.

    Args:
        df_1h: 1H DataFrame with aligned column
        df_htf: HTF DataFrame with source column
        htf: Higher timeframe label ('4H', '1D', etc.)
        column: Column name to validate

    Returns:
        Dict with validation results:
            - 'valid': bool
            - 'lookahead_bias': bool
            - 'missing_values': int
            - 'alignment_errors': int

    Example:
        >>> result = validate_alignment(df_1h, df_4h, '4H', 'boms_displacement')
        >>> assert result['valid']
        >>> assert not result['lookahead_bias']
    """
    if column not in df_1h.columns:
        return {
            'valid': False,
            'error': f"Column '{column}' not found in df_1h"
        }

    if column not in df_htf.columns:
        return {
            'valid': False,
            'error': f"Column '{column}' not found in df_htf"
        }

    # Check for lookahead bias: HTF value should not appear before its timestamp
    lookahead_bias = False
    alignment_errors = 0

    for htf_ts in df_htf.index:
        htf_value = df_htf.loc[htf_ts, column]

        # All 1H bars before HTF timestamp should NOT have this HTF value
        # (unless it came from a previous HTF bar)
        bars_before = df_1h[df_1h.index < htf_ts]
        if len(bars_before) > 0:
            # Check last bar before HTF timestamp
            last_bar_value = bars_before.iloc[-1][column]

            # If it matches current HTF value exactly and no previous HTF bar had this value,
            # that's lookahead bias
            prev_htf_values = df_htf[df_htf.index < htf_ts][column].values
            if last_bar_value == htf_value and htf_value not in prev_htf_values:
                lookahead_bias = True
                alignment_errors += 1

    # Check for missing values (should be filled by ffill)
    missing_values = df_1h[column].isna().sum()

    valid = not lookahead_bias and alignment_errors == 0

    return {
        'valid': valid,
        'lookahead_bias': lookahead_bias,
        'missing_values': int(missing_values),
        'alignment_errors': alignment_errors
    }


def get_htf_window(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    htf: str,
    lookback_bars: int = 100
) -> pd.DataFrame:
    """
    Get HTF window up to (and including) a specific 1H timestamp.

    Args:
        df: 1H DataFrame with OHLCV data
        timestamp: 1H timestamp to get HTF window for
        htf: Higher timeframe ('4H', '1D', etc.)
        lookback_bars: Number of HTF bars to include

    Returns:
        HTF DataFrame with up to lookback_bars before timestamp

    Notes:
        - Point-in-time: only includes HTF bars completed by timestamp
        - Used for calculating HTF features at specific 1H bar

    Example:
        >>> window_4h = get_htf_window(df_1h, pd.Timestamp('2024-01-01 12:00'), '4H', lookback_bars=50)
        >>> # Returns up to 50 completed 4H bars before 12:00
    """
    # Get all 1H bars up to timestamp
    df_up_to = df[df.index <= timestamp].copy()

    if len(df_up_to) == 0:
        return pd.DataFrame()

    # Resample to HTF
    df_htf = resample_to_timeframe(df_up_to, htf)

    # Return last N bars
    return df_htf.tail(lookback_bars)


# Example usage for testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Create sample 1H data
    dates_1h = pd.date_range('2024-01-01', periods=24, freq='1H')
    df_1h = pd.DataFrame({
        'open': np.random.randn(24).cumsum() + 100,
        'high': np.random.randn(24).cumsum() + 102,
        'low': np.random.randn(24).cumsum() + 98,
        'close': np.random.randn(24).cumsum() + 100,
        'volume': np.random.randint(1000, 5000, 24)
    }, index=dates_1h)

    # Resample to 4H
    df_4h = resample_to_timeframe(df_1h, '4H')
    print(f"\n1H bars: {len(df_1h)}, 4H bars: {len(df_4h)}")
    print(f"4H timestamps:\n{df_4h.index}")

    # Add a test feature to 4H
    df_4h['test_feature'] = np.arange(len(df_4h)) * 100

    # Align back to 1H
    df_1h = align_htf_to_1h(df_1h, df_4h, '4H', ['test_feature'], prefix='tf4h_')
    print(f"\nAligned 1H data:")
    print(df_1h[['close', 'tf4h_test_feature']].head(10))

    # Validate alignment
    result = validate_alignment(df_1h, df_4h, '4H', 'tf4h_test_feature')
    print(f"\nValidation result: {result}")
