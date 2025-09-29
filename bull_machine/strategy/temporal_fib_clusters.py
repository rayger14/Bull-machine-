"""
Fibonacci Time Clusters for Bull Machine v1.6.1

Time is pressure, not prediction. Fib clusters show when a move must resolve.
Based on Wyckoff Insider's rhythm patterns and temporal pressure zones.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def fib_time_clusters(pivots_df: pd.DataFrame, current_bar: int, tf: str = '1H',
                     tolerance_bars: int = 3) -> Optional[Dict[str, Any]]:
    """
    Calculate Fibonacci time clusters from pivot points.

    Time clusters occur when multiple Fibonacci bar counts from different pivots
    overlap within a tolerance window, creating "pressure zones" where price
    movement is likely to accelerate or reverse.

    Args:
        pivots_df: DataFrame with pivot data including 'bar_index' column
        current_bar: Current bar index to check clusters against
        tf: Timeframe (affects tolerance and strength calculation)
        tolerance_bars: Bar tolerance for cluster overlap detection

    Returns:
        Dict with cluster info if detected, None otherwise

    Example:
        >>> pivots = pd.DataFrame({'bar_index': [100, 150]})
        >>> cluster = fib_time_clusters(pivots, 155, '1H', 3)
        >>> # Returns cluster if 34/55 bars from pivots overlap near bar 155
    """
    if len(pivots_df) == 0:
        return None

    # Fibonacci bar counts - Wyckoff Insider's rhythm sequences
    fib_nums = [21, 34, 55, 89, 144]
    clusters = []

    # Calculate projections from each pivot
    for _, pivot in pivots_df.iterrows():
        pivot_bar = pivot['bar_index']

        for fib_num in fib_nums:
            proj_bar = pivot_bar + fib_num

            # Check if projection is near current bar
            if abs(proj_bar - current_bar) <= tolerance_bars:
                clusters.append({
                    'proj_bar': proj_bar,
                    'fib_num': fib_num,
                    'pivot_bar': pivot_bar,
                    'distance': abs(proj_bar - current_bar)
                })

    if len(clusters) < 2:  # Need at least 2 overlapping projections
        return None

    # Check for genuine overlaps (not same pivot + fib number)
    unique_clusters = []
    for cluster in clusters:
        is_unique = True
        for existing in unique_clusters:
            if (cluster['pivot_bar'] == existing['pivot_bar'] and
                cluster['fib_num'] == existing['fib_num']):
                is_unique = False
                break
        if is_unique:
            unique_clusters.append(cluster)

    if len(unique_clusters) < 2:
        return None

    # Calculate cluster strength based on overlaps and convergence
    overlap_count = len(unique_clusters)

    # Base strength from overlap count
    strength = min(0.80, overlap_count * 0.15)

    # Bonus for tight convergence
    all_proj_bars = [c['proj_bar'] for c in unique_clusters]
    convergence = max(all_proj_bars) - min(all_proj_bars)
    if convergence <= 1:  # Very tight cluster
        strength += 0.20
    elif convergence <= 2:  # Tight cluster
        strength += 0.10

    # Bonus for higher Fibonacci numbers (stronger timing)
    high_fib_count = sum(1 for c in unique_clusters if c['fib_num'] >= 55)
    if high_fib_count >= 2:
        strength += 0.15

    # Timeframe adjustment
    tf_multiplier = {'1H': 1.0, '4H': 1.1, '1D': 1.2}.get(tf, 1.0)
    strength *= tf_multiplier

    # Ensure maximum strength cap
    strength = min(0.80, strength)

    cluster_center = int(np.mean(all_proj_bars))

    return {
        'cluster_bar': cluster_center,
        'window': [min(all_proj_bars) - tolerance_bars, max(all_proj_bars) + tolerance_bars],
        'strength': round(strength, 3),
        'overlap_count': overlap_count,
        'convergence_bars': convergence,
        'fib_numbers': [c['fib_num'] for c in unique_clusters],
        'pivot_sources': [c['pivot_bar'] for c in unique_clusters]
    }


def detect_pivot_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Detect swing high/low pivot points for time cluster analysis.

    Args:
        df: OHLC DataFrame
        window: Lookback window for pivot detection

    Returns:
        DataFrame with pivot points and their bar indices
    """
    if len(df) < window * 2 + 1:
        return pd.DataFrame(columns=['bar_index', 'price', 'type'])

    pivots = []

    for i in range(window, len(df) - window):
        # Check for swing high
        is_high = all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1))
        is_high = is_high and all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1))

        # Check for swing low
        is_low = all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1))
        is_low = is_low and all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1))

        if is_high:
            pivots.append({
                'bar_index': i,
                'price': df['high'].iloc[i],
                'type': 'swing_high'
            })
        elif is_low:
            pivots.append({
                'bar_index': i,
                'price': df['low'].iloc[i],
                'type': 'swing_low'
            })

    return pd.DataFrame(pivots)


def get_time_cluster_for_current_bar(df: pd.DataFrame, current_idx: int,
                                   config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get time cluster analysis for the current bar position.

    Args:
        df: OHLC DataFrame
        current_idx: Current bar index in the DataFrame
        config: Configuration with temporal settings

    Returns:
        Time cluster info if detected, None otherwise
    """
    # Extract config parameters
    temporal_config = config.get('temporal', {})
    tolerance_bars = temporal_config.get('tolerance_bars', 3)
    pivot_window = temporal_config.get('pivot_window', 5)

    # Look back for pivots (use reasonable window to avoid excessive computation)
    lookback_bars = min(200, current_idx)
    if lookback_bars < 20:  # Need sufficient data
        return None

    window_df = df.iloc[current_idx - lookback_bars:current_idx + 1].copy()
    window_df.reset_index(drop=True, inplace=True)

    # Detect pivots in the window
    pivots_df = detect_pivot_points(window_df, pivot_window)

    if len(pivots_df) < 2:  # Need at least 2 pivots
        return None

    # Get time cluster for current position
    return fib_time_clusters(pivots_df, len(window_df) - 1,
                           config.get('timeframe', '1H'), tolerance_bars)