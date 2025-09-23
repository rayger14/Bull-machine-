"""
Liquidity & pHOB Scoring
Phase 1.2/1.3 Rules: Sweep mitigation, TTL decay, and imbalance detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging


def calculate_fib_level(high: float, low: float, current: float) -> float:
    """
    Calculate Fibonacci retracement level (0 = low, 1 = high).
    """
    if high == low:
        return 0.5
    return (current - low) / (high - low)


def score_sweep_mitigation(sweep_bar: Dict, reclaim_bar: Dict,
                          atr: float, fib_level: float) -> float:
    """
    Score liquidity sweep mitigation quality.

    Args:
        sweep_bar: Bar where liquidity was swept (dict with 'ts', 'low', 'high', 'close')
        reclaim_bar: Bar where level was reclaimed (dict with 'ts', 'close')
        atr: Average True Range for normalization
        fib_level: Fibonacci level of reclaim (0.618-0.786 is discount zone)

    Returns:
        Score 0-1 representing mitigation quality
    """

    # Calculate time to reclaim (in hours)
    speed_hrs = abs(reclaim_bar['ts'] - sweep_bar['ts']) / 3600

    # Calculate displacement (in ATR units)
    displacement = abs(reclaim_bar['close'] - sweep_bar['low']) / max(atr, 1e-9)

    # Check if in discount zone (golden pocket)
    is_discount = 0.618 <= fib_level <= 0.786

    # Base scoring components
    fib_score = 0.15 * fib_level  # Prefer deeper retracements
    displacement_score = 0.25 * min(displacement, 2.0)  # Cap at 2 ATR
    discount_bonus = 0.25 if is_discount else 0

    # Combine base scores
    base_score = fib_score + displacement_score + discount_bonus

    # Apply time decay and speed bonus
    if speed_hrs <= 5 and displacement >= 1.25 and is_discount:
        # Fast, strong reclaim in discount zone = high quality
        return min(base_score * 1.3, 0.85)
    else:
        # Apply exponential decay for slower reclaims
        decay_factor = 0.9 ** max(0, speed_hrs - 5)
        return base_score * decay_factor


def detect_liquidity_pools(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
    """
    Detect significant liquidity pools (equal highs/lows, order blocks).
    """
    pools = []

    if len(df) < lookback:
        return pools

    recent = df.tail(lookback)

    # Find equal highs (within 0.1% tolerance)
    highs = recent['high'].values
    high_counts = {}

    for h in highs:
        # Round to nearest 0.1%
        rounded = round(h / 10) * 10
        high_counts[rounded] = high_counts.get(rounded, 0) + 1

    # Equal highs with 3+ touches
    for level, count in high_counts.items():
        if count >= 3:
            pools.append({
                'type': 'equal_highs',
                'level': level,
                'touches': count,
                'strength': min(count / 5.0, 1.0)
            })

    # Find equal lows
    lows = recent['low'].values
    low_counts = {}

    for l in lows:
        rounded = round(l / 10) * 10
        low_counts[rounded] = low_counts.get(rounded, 0) + 1

    for level, count in low_counts.items():
        if count >= 3:
            pools.append({
                'type': 'equal_lows',
                'level': level,
                'touches': count,
                'strength': min(count / 5.0, 1.0)
            })

    # Detect order blocks (large volume candles with subsequent respect)
    vol_threshold = recent['volume'].quantile(0.8)

    for i in range(len(recent) - 10):
        if recent.iloc[i]['volume'] > vol_threshold:
            # Check if this level was respected
            ob_high = recent.iloc[i]['high']
            ob_low = recent.iloc[i]['low']

            # Count subsequent touches
            touches = 0
            for j in range(i + 1, min(i + 10, len(recent))):
                if ob_low <= recent.iloc[j]['low'] <= ob_high or \
                   ob_low <= recent.iloc[j]['high'] <= ob_high:
                    touches += 1

            if touches >= 2:
                pools.append({
                    'type': 'order_block',
                    'level': (ob_high + ob_low) / 2,
                    'range': (ob_high, ob_low),
                    'touches': touches,
                    'strength': min(touches / 3.0, 1.0) * 0.8
                })

    return pools


def calculate_liquidity_score(df: pd.DataFrame, current_idx: int = -1) -> Dict:
    """
    Calculate comprehensive liquidity score for current market state.
    """

    if len(df) < 20:
        return {
            'score': 0.5,
            'pools': [],
            'recent_sweep': None,
            'imbalance_filled': False
        }

    current = df.iloc[current_idx]

    # Detect liquidity pools
    pools = detect_liquidity_pools(df)

    # Check for recent sweep
    recent_sweep = None
    sweep_score = 0

    # Look for sweep in last 10 bars
    for i in range(max(-10, -len(df)), 0):
        bar = df.iloc[i]

        # Check if bar swept any pool
        for pool in pools:
            if pool['type'] == 'equal_lows' and bar['low'] < pool['level'] * 0.998:
                # Swept equal lows
                if bar['close'] > pool['level']:
                    # Reclaimed in same bar - strong signal
                    recent_sweep = {
                        'type': 'sweep_and_reclaim',
                        'level': pool['level'],
                        'bar_idx': i,
                        'strength': 0.8
                    }
                    sweep_score = 0.8
                    break
            elif pool['type'] == 'equal_highs' and bar['high'] > pool['level'] * 1.002:
                # Swept equal highs
                if bar['close'] < pool['level']:
                    # Reclaimed in same bar
                    recent_sweep = {
                        'type': 'sweep_and_reclaim',
                        'level': pool['level'],
                        'bar_idx': i,
                        'strength': 0.8
                    }
                    sweep_score = 0.8
                    break

    # Check for imbalance fill
    imbalance_filled = False

    if len(df) >= 3:
        # Simple FVG check
        prev2 = df.iloc[-3]
        prev1 = df.iloc[-2]

        # Bullish FVG
        if prev2['high'] < prev1['low']:
            gap_size = prev1['low'] - prev2['high']
            if current['low'] <= prev2['high'] + gap_size * 0.5:
                imbalance_filled = True

        # Bearish FVG
        elif prev2['low'] > prev1['high']:
            gap_size = prev2['low'] - prev1['high']
            if current['high'] >= prev2['low'] - gap_size * 0.5:
                imbalance_filled = True

    # Calculate final score
    pool_score = min(len(pools) / 5.0, 1.0) * 0.3  # More pools = more liquidity
    pool_strength = max([p['strength'] for p in pools], default=0) * 0.3
    imbalance_score = 0.2 if imbalance_filled else 0

    total_score = pool_score + pool_strength + sweep_score * 0.4 + imbalance_score

    return {
        'score': min(total_score, 1.0),
        'pools': pools,
        'recent_sweep': recent_sweep,
        'imbalance_filled': imbalance_filled,
        'components': {
            'pool_density': pool_score,
            'pool_strength': pool_strength,
            'sweep_quality': sweep_score,
            'imbalance': imbalance_score
        }
    }


def liquidity_ttl_decay(base_score: float, bars_since_event: int,
                        ttl_bars: int = 12) -> float:
    """
    Apply time-to-live decay to liquidity scores.

    Args:
        base_score: Initial liquidity score
        bars_since_event: Bars since liquidity event occurred
        ttl_bars: Time-to-live in bars before score starts decaying

    Returns:
        Decayed score
    """

    if bars_since_event <= ttl_bars:
        # Within TTL, no decay
        return base_score
    else:
        # Exponential decay after TTL
        decay_factor = 0.9 ** (bars_since_event - ttl_bars)
        return base_score * decay_factor