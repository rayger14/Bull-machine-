"""
Bull Machine v1.6.0 - Advanced M1/M2 Wyckoff Phase Detection
Implementation of Wyckoff_Insider post:31 concepts for enhanced signal quality

M1 (Spring): Identifies false breakdowns at range lows indicating accumulation.
M2 (Markup): Identifies re-accumulation at range highs indicating continuation.

Returns:
    Dict[str, float]: {'m1': 0.0-1.0, 'm2': 0.0-1.0, 'side': 'long'|'short'|'neutral'}
"""

import pandas as pd
import numpy as np
from typing import Dict
from bull_machine.core.telemetry import log_telemetry
import numpy as np

def _sma(series, n:int)->float:
    s = series.rolling(n).mean()
    v = s.iloc[-1]
    return float(v) if np.isfinite(v) else float(series.iloc[-1])

def _rsi(series, n:int=14)->float:
    delta = series.diff()
    up = np.clip(delta, 0, None)
    down = -np.clip(delta, None, 0)
    roll_up = up.rolling(n).mean()
    roll_dn = down.rolling(n).mean()
    rs = np.where(roll_dn.iloc[-1]==0, np.inf, roll_up.iloc[-1]/roll_dn.iloc[-1])
    rsi = 100 - (100/(1+rs))
    if not np.isfinite(rsi): rsi = 50.0
    return float(rsi)

def _determine_trade_side_enhanced(df_ltf, df_htf, m1_score: float, m2_score: float, fib_r: float, fib_x: float) -> str:
    """
    Decide long/short bias using HTF trend + LTF confirmation.
    - LONG: M1 (spring) & HTF uptrend; or Fib retracement in uptrend; RSI>50 as tie-breaker
    - SHORT: M2 (markdown) & HTF downtrend; or Fib extension in downtrend; RSI<50 as tie-breaker
    """
    df_t = df_htf if df_htf is not None and len(df_htf)>=60 else df_ltf
    sma_20 = _sma(df_t['close'], 20)
    sma_50 = _sma(df_t['close'], 50)
    rsi_ltf = _rsi(df_ltf['close'], 14)
    uptrend = sma_20 > sma_50
    downtrend = sma_20 < sma_50
    # Primary cues
    if m1_score >= 0.60 and uptrend:
        return "long"
    if m2_score >= 0.50 and downtrend:
        return "short"
    # Secondary cues using fibs + RSI
    if fib_r >= 0.45 and uptrend and rsi_ltf >= 50:
        return "long"
    if fib_x >= 0.45 and downtrend and rsi_ltf <= 50:
        return "short"
    return "neutral"

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

def compute_m1m2_scores(df_ltf: pd.DataFrame, tf: str = None, df_htf: pd.DataFrame = None, fib_scores: dict = None) -> Dict[str, float]:
    """
    Return {'m1': float, 'm2': float, 'side': 'long'|'short'|'neutral'} with HTF-aware bias.
    df_ltf: lower timeframe frame used for entries / local momentum
    df_htf: higher timeframe for trend bias (1D recommended); if None, falls back to df_ltf
    fib_scores: optional {'fib_retracement':..., 'fib_extension':...} to aid bias decision
    """
    if len(df_ltf) < 20:
        return {'m1': 0.0, 'm2': 0.0, 'side': 'neutral'}

    try:
        # Diagnostic telemetry for M1/M2 prerequisites
        required = {'open', 'high', 'low', 'close', 'volume'}
        have = set(df_ltf.columns.str.lower())
        if not required.issubset(have):
            log_telemetry('layer_masks.json', {
                'm1m2_diag': 'missing_columns',
                'have_cols': list(have),
                'need_cols': list(required),
                'len': int(len(df_ltf))
            })
            return {'m1': 0.0, 'm2': 0.0, 'side': 'neutral'}

        # Analyze current range and volume context
        range_info = _identify_range(df_ltf)
        vol_info = _volume_confirmation(df_ltf)

        # Diagnostic: log range and volume context
        log_telemetry('layer_masks.json', {
            'm1m2_diag': 'range_context',
            'range_size_pct': range_info['size'] / df_ltf['close'].iloc[-1] * 100,
            'vol_spike_ratio': vol_info['spike_ratio'],
            'range_high': range_info['high'],
            'range_low': range_info['low']
        })

        # Skip if range too small (consolidation)
        if range_info['size'] < df_ltf['close'].iloc[-1] * 0.01:  # Less than 1% range
            log_telemetry('layer_masks.json', {
                'm1m2_diag': 'range_too_small',
                'range_size_pct': range_info['size'] / df_ltf['close'].iloc[-1] * 100
            })
            return {'m1': 0.0, 'm2': 0.0, 'side': 'neutral'}

        # Detect M1 and M2 patterns
        m1_score = _detect_m1_spring(df_ltf, range_info, vol_info, tf or '1H')
        m2_score = _detect_m2_markup(df_ltf, range_info, vol_info, tf or '1H')

        # Enhanced bias decision with HTF and fib context
        fib_r = (fib_scores or {}).get('fib_retracement', 0.0)
        fib_x = (fib_scores or {}).get('fib_extension', 0.0)
        side = _determine_trade_side_enhanced(df_ltf, df_htf, m1_score, m2_score, fib_r, fib_x)

        # Log for analysis with HTF trend info
        log_telemetry('layer_masks.json', {
            'module': 'wyckoff_m1m2',
            'tf': tf or '1H',
            'm1_score': m1_score,
            'm2_score': m2_score,
            'side': side,
            'htf_sma20_gt_sma50': bool(_sma((df_htf or df_ltf)['close'],20) > _sma((df_htf or df_ltf)['close'],50)),
            'range_high': range_info['high'],
            'range_low': range_info['low'],
            'range_size': range_info['size'],
            'volume_spike_ratio': vol_info['spike_ratio'],
            'current_price': df_ltf['close'].iloc[-1]
        })

        return {
            'm1': float(m1_score),
            'm2': float(m2_score),
            'side': side
        }

    except Exception as e:
        log_telemetry('layer_masks.json', {
            'module': 'wyckoff_m1m2',
            'tf': tf or '1H',
            'error': str(e),
            'm1_score': 0.0,
            'm2_score': 0.0
        })
        return {'m1': 0.0, 'm2': 0.0, 'side': 'neutral'}

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