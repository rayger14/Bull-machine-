"""
Bull Machine v1.6.0 - Advanced M1/M2 Wyckoff Phase Detection
Implementation of Wyckoff_Insider post:31 concepts for enhanced signal quality

M1 Phase: Spring/shakeout at range lows with volume confirmation
M2 Phase: Markup/re-accumulation with breakout confirmation
"""

import pandas as pd
import numpy as np
from typing import Dict
from bull_machine.core.telemetry import log_telemetry

def _identify_range(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """Identify current trading range from recent price action."""
    range_high = df['high'].rolling(lookback).max().iloc[-1]
    range_low = df['low'].rolling(lookback).min().iloc[-1]
    range_midpoint = (range_high + range_low) / 2
    range_size = range_high - range_low

    return {
        'high': range_high,
        'low': range_low,
        'midpoint': range_midpoint,
        'size': range_size
    }

def _volume_confirmation(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """Calculate volume metrics for confirmation."""
    current_vol = df['volume'].iloc[-1]
    vol_ma = df['volume'].rolling(lookback).mean().iloc[-1]
    vol_std = df['volume'].rolling(lookback).std().iloc[-1]

    # Volume spike thresholds
    high_vol_threshold = vol_ma + 1.5 * vol_std
    med_vol_threshold = vol_ma + 0.5 * vol_std

    return {
        'current': current_vol,
        'ma': vol_ma,
        'high_threshold': high_vol_threshold,
        'med_threshold': med_vol_threshold,
        'spike_ratio': current_vol / vol_ma if vol_ma > 0 else 1.0
    }

def _detect_m1_spring(df: pd.DataFrame, range_info: Dict, vol_info: Dict, tf: str) -> float:
    """
    Detect M1 spring/shakeout patterns.

    M1 Criteria:
    - Price near range low (within 10% of range)
    - Volume spike (1.5x+ average)
    - Price rejection (close above low)
    - Timeframe-adjusted scoring
    """
    current_price = df['close'].iloc[-1]
    current_low = df['low'].iloc[-1]
    prev_close = df['close'].iloc[-2]

    score = 0.0

    # 1. Price position: Near range low
    range_low_zone = range_info['low'] + 0.1 * range_info['size']
    if current_price <= range_low_zone:
        # Base M1 score - higher for longer timeframes
        if tf == '1D':
            score += 0.40
        elif tf == '4H':
            score += 0.35
        else:  # 1H
            score += 0.30

        # 2. Volume confirmation: Spike during decline
        if vol_info['current'] >= vol_info['high_threshold']:
            score += 0.20
        elif vol_info['current'] >= vol_info['med_threshold']:
            score += 0.10

        # 3. Price rejection: Close above intrabar low
        if current_price > current_low:
            score += 0.15

        # 4. Momentum divergence: Previous bar was lower
        if len(df) >= 2 and prev_close > current_price:
            score += 0.10

        # 5. Range position bonus: Deeper in range = higher score
        range_position = (current_price - range_info['low']) / range_info['size']
        if range_position < 0.05:  # Very bottom of range
            score += 0.15
        elif range_position < 0.15:  # Bottom quarter
            score += 0.10

    return min(score, 0.80)  # Cap at 0.80

def _detect_m2_markup(df: pd.DataFrame, range_info: Dict, vol_info: Dict, tf: str) -> float:
    """
    Detect M2 markup/re-accumulation patterns.

    M2 Criteria:
    - Price breaks above range high or near resistance
    - Sustained volume (1.2x+ average)
    - Upward momentum confirmation
    - Strength relative to range
    """
    current_price = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    prev_close = df['close'].iloc[-2]

    score = 0.0

    # 1. Price position: Breaking range high or testing resistance
    range_high_zone = range_info['high'] - 0.05 * range_info['size']
    if current_price >= range_high_zone:
        # Base M2 score
        if tf == '1D':
            score += 0.35
        elif tf == '4H':
            score += 0.30
        else:  # 1H
            score += 0.25

        # 2. Volume confirmation: Sustained buying
        if vol_info['current'] >= vol_info['med_threshold']:
            score += 0.15
        if vol_info['spike_ratio'] >= 1.2:
            score += 0.10

        # 3. Breakout confirmation: Price above range high
        if current_price > range_info['high']:
            score += 0.20
        elif current_high > range_info['high']:  # Intrabar breakout
            score += 0.15

        # 4. Momentum confirmation: Upward price action
        if len(df) >= 2 and current_price > prev_close:
            score += 0.10

        # 5. Strength measurement: Distance above range
        if current_price > range_info['high']:
            breakout_strength = (current_price - range_info['high']) / range_info['size']
            if breakout_strength > 0.05:  # Strong breakout
                score += 0.15
            elif breakout_strength > 0.02:  # Moderate breakout
                score += 0.10

    return min(score, 0.75)  # Cap at 0.75

def compute_m1m2_scores(df: pd.DataFrame, tf: str) -> Dict[str, float]:
    """
    Compute Wyckoff M1 (spring/shakeout) and M2 (markup/re-accumulation) scores.

    Based on Wyckoff_Insider post:31:
    - M1: Spring patterns at range lows with volume spikes
    - M2: Markup patterns at range highs with sustained volume
    - Enhanced with price structure and momentum confirmation

    Args:
        df: OHLCV DataFrame with sufficient history (min 20 bars)
        tf: Timeframe string ('1H', '4H', '1D')

    Returns:
        Dict with 'm1' and 'm2' scores (0.0 to 0.8 range)
    """
    if len(df) < 20:
        return {'m1': 0.0, 'm2': 0.0}

    try:
        # Analyze current range and volume context
        range_info = _identify_range(df)
        vol_info = _volume_confirmation(df)

        # Skip if range too small (consolidation)
        if range_info['size'] < df['close'].iloc[-1] * 0.01:  # Less than 1% range
            return {'m1': 0.0, 'm2': 0.0}

        # Detect M1 and M2 patterns
        m1_score = _detect_m1_spring(df, range_info, vol_info, tf)
        m2_score = _detect_m2_markup(df, range_info, vol_info, tf)

        # Log for analysis
        log_telemetry('layer_masks.json', {
            'module': 'wyckoff_m1m2',
            'tf': tf,
            'm1_score': m1_score,
            'm2_score': m2_score,
            'range_high': range_info['high'],
            'range_low': range_info['low'],
            'range_size': range_info['size'],
            'volume_spike_ratio': vol_info['spike_ratio'],
            'current_price': df['close'].iloc[-1]
        })

        return {
            'm1': float(m1_score),
            'm2': float(m2_score)
        }

    except Exception as e:
        log_telemetry('layer_masks.json', {
            'module': 'wyckoff_m1m2',
            'tf': tf,
            'error': str(e),
            'm1_score': 0.0,
            'm2_score': 0.0
        })
        return {'m1': 0.0, 'm2': 0.0}

def validate_m1m2_signals(df: pd.DataFrame, m1_score: float, m2_score: float) -> Dict[str, bool]:
    """
    Additional validation for M1/M2 signals to prevent false positives.

    Returns:
        Dict with validation flags for m1_valid and m2_valid
    """
    validations = {'m1_valid': False, 'm2_valid': False}

    if len(df) < 5:
        return validations

    try:
        # M1 validation: Ensure we're not in strong uptrend
        if m1_score > 0.3:
            recent_trend = df['close'].iloc[-5:].pct_change().mean()
            validations['m1_valid'] = recent_trend < 0.02  # Not strong uptrend

        # M2 validation: Ensure we're not in strong downtrend
        if m2_score > 0.3:
            recent_trend = df['close'].iloc[-5:].pct_change().mean()
            validations['m2_valid'] = recent_trend > -0.02  # Not strong downtrend

        return validations

    except:
        return validations