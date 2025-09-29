"""
Bojan Microstructure Analysis - Bull Machine v1.6.2

Core Concepts:
- Wicks encode unfinished business (magnets for price pullback)
- Trap resets: sweep + flip bars marking reversals
- Wick dominance at HTF extremes (≥70% wick = magnet)
- pHOB (hidden order blocks) behind FVGs
- Fibonacci .705/.786 confluence zones

Integration with PO3:
- Bojan highs/lows in manipulation phases boost PO3 strength
- Trap resets align with PO3 sweeps for high-probability entries
- Wick magnets as profit targets at fib extensions

References:
- Wyckoff Insider's PO3 confluences
- Moneytaur's pHOB/FVG concepts
- Crypto Chase's BOS wick resets
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class BojanConfig:
    """Configuration for Bojan microstructure analysis"""
    lookback: int = 50
    wick_magnet_threshold: float = 0.70    # ≥70% wick = magnet
    trap_body_min: float = 1.25            # Trap reset body ≥1.25x ATR
    unfinished_threshold: float = 0.10     # No wick threshold (≤10%)
    phob_confidence_min: float = 0.30      # pHOB minimum confidence
    fib_prime_zones: List[float] = None    # [0.705, 0.786] confluence

    def __post_init__(self):
        if self.fib_prime_zones is None:
            self.fib_prime_zones = [0.705, 0.786]


def calculate_wick_body_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate wick and body dominance metrics for current bar"""
    if len(df) < 1:
        return {}

    bar = df.iloc[-1]
    open_price = bar['open'] if 'open' in bar else bar['close']
    high = bar['high']
    low = bar['low']
    close = bar['close']

    # True range calculation
    total_range = high - low
    if total_range == 0:
        return {
            'wick_dominance': 0.0,
            'body_dominance': 0.0,
            'upper_wick_ratio': 0.0,
            'lower_wick_ratio': 0.0,
            'body_ratio': 0.0
        }

    # Body and wick calculations
    body_high = max(open_price, close)
    body_low = min(open_price, close)

    upper_wick = high - body_high
    lower_wick = body_low - low
    body_size = abs(close - open_price)

    # Ratios
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    body_ratio = body_size / total_range
    total_wick_ratio = (upper_wick + lower_wick) / total_range

    return {
        'wick_dominance': total_wick_ratio,
        'body_dominance': body_ratio,
        'upper_wick_ratio': upper_wick_ratio,
        'lower_wick_ratio': lower_wick_ratio,
        'body_ratio': body_ratio,
        'total_range': total_range
    }


def detect_wick_magnet(df: pd.DataFrame, config: BojanConfig) -> Dict[str, any]:
    """Detect wick magnets (unfinished business zones)"""
    metrics = calculate_wick_body_metrics(df)

    if not metrics:
        return {'is_magnet': False, 'magnet_type': None, 'strength': 0.0}

    # Wick magnet detection (≥70% wick dominance)
    is_upper_magnet = metrics['upper_wick_ratio'] >= config.wick_magnet_threshold
    is_lower_magnet = metrics['lower_wick_ratio'] >= config.wick_magnet_threshold

    magnet_type = None
    strength = 0.0

    if is_upper_magnet:
        magnet_type = 'upper_wick_magnet'
        strength = metrics['upper_wick_ratio']
    elif is_lower_magnet:
        magnet_type = 'lower_wick_magnet'
        strength = metrics['lower_wick_ratio']

    return {
        'is_magnet': bool(is_upper_magnet or is_lower_magnet),
        'magnet_type': magnet_type,
        'strength': float(strength),
        'upper_wick_magnet': bool(is_upper_magnet),
        'lower_wick_magnet': bool(is_lower_magnet),
        'metrics': metrics
    }


def detect_trap_reset(df: pd.DataFrame, config: BojanConfig) -> Dict[str, any]:
    """Detect trap reset bars: sweep + flip with large body commitment"""
    if len(df) < 2:
        return {'is_trap_reset': False, 'direction': None, 'strength': 0.0}

    # Current and previous bars
    current = df.iloc[-1]
    previous = df.iloc[-2]

    # ATR calculation for body size validation
    if len(df) >= 14:
        atr = calculate_atr(df, period=14)
    else:
        # Fallback to recent range average
        ranges = df['high'] - df['low']
        atr = ranges.tail(min(len(df), 5)).mean()

    if atr == 0:
        return {'is_trap_reset': False, 'direction': None, 'strength': 0.0}

    # Current bar metrics
    current_open = current['open'] if 'open' in current else current['close']
    current_close = current['close']
    current_high = current['high']
    current_low = current['low']

    # Previous bar direction
    prev_open = previous['open'] if 'open' in previous else previous['close']
    prev_close = previous['close']
    prev_bullish = prev_close > prev_open

    # Current bar body size and direction
    current_body = abs(current_close - current_open)
    current_bullish = current_close > current_open
    body_atr_ratio = current_body / atr if atr > 0 else 0

    # Trap reset conditions:
    # 1. Direction flip from previous bar
    # 2. Large body commitment (≥1.25x ATR)
    # 3. Sweep behavior (wick in opposite direction)
    direction_flip = (prev_bullish and not current_bullish) or (not prev_bullish and current_bullish)
    large_body = body_atr_ratio >= config.trap_body_min

    # Sweep detection
    total_range = current_high - current_low
    if total_range == 0:
        return {'is_trap_reset': False, 'direction': None, 'strength': 0.0}

    # Check for sweep behavior (initial move opposite to final close)
    sweep_detected = False
    if current_bullish:
        # Bullish close but significant lower wick (swept lows first)
        lower_wick = min(current_open, current_close) - current_low
        sweep_detected = (lower_wick / total_range) > 0.3
    else:
        # Bearish close but significant upper wick (swept highs first)
        upper_wick = current_high - max(current_open, current_close)
        sweep_detected = (upper_wick / total_range) > 0.3

    is_trap_reset = direction_flip and large_body and sweep_detected

    return {
        'is_trap_reset': bool(is_trap_reset),
        'direction': 'bullish' if current_bullish else 'bearish',
        'strength': float(body_atr_ratio if is_trap_reset else 0.0),
        'body_atr_ratio': float(body_atr_ratio),
        'sweep_detected': bool(sweep_detected),
        'direction_flip': bool(direction_flip)
    }


def detect_unfinished_candles(df: pd.DataFrame, config: BojanConfig) -> Dict[str, any]:
    """Detect unfinished candles (no wick on one side = magnet for rebalance)"""
    metrics = calculate_wick_body_metrics(df)

    if not metrics:
        return {'has_unfinished': False, 'unfinished_type': None}

    # Unfinished detection: one side has minimal wick (≤10%)
    no_upper_wick = metrics['upper_wick_ratio'] <= config.unfinished_threshold
    no_lower_wick = metrics['lower_wick_ratio'] <= config.unfinished_threshold

    unfinished_type = None
    if no_upper_wick and not no_lower_wick:
        unfinished_type = 'no_upper_wick'  # Likely to revisit highs
    elif no_lower_wick and not no_upper_wick:
        unfinished_type = 'no_lower_wick'  # Likely to revisit lows

    return {
        'has_unfinished': unfinished_type is not None,
        'unfinished_type': unfinished_type,
        'upper_wick_ratio': metrics['upper_wick_ratio'],
        'lower_wick_ratio': metrics['lower_wick_ratio']
    }


def detect_phob_zones(df: pd.DataFrame, config: BojanConfig) -> Dict[str, any]:
    """Detect potential Hidden Order Block (pHOB) zones behind FVGs"""
    if len(df) < config.lookback:
        return {'phob_detected': False, 'confidence': 0.0, 'zones': []}

    recent_data = df.tail(config.lookback)

    # Simplified pHOB detection based on:
    # 1. Fair Value Gaps (FVG) - imbalances between candles
    # 2. Order blocks - strong rejection candles
    # 3. Confluence zones where they overlap

    phob_zones = []
    confidence_scores = []

    # FVG detection (simplified)
    for i in range(2, len(recent_data)):
        candle1 = recent_data.iloc[i-2]
        candle2 = recent_data.iloc[i-1]
        candle3 = recent_data.iloc[i]

        # Bullish FVG: candle1 low > candle3 high
        if candle1['low'] > candle3['high']:
            gap_size = candle1['low'] - candle3['high']
            mid_price = (candle1['low'] + candle3['high']) / 2

            # Check for order block characteristics in the area
            ob_strength = calculate_order_block_strength(recent_data, i-2, i)

            if ob_strength > config.phob_confidence_min:
                phob_zones.append({
                    'type': 'bullish_phob',
                    'high': candle1['low'],
                    'low': candle3['high'],
                    'mid': mid_price,
                    'strength': ob_strength
                })
                confidence_scores.append(ob_strength)

        # Bearish FVG: candle1 high < candle3 low
        elif candle1['high'] < candle3['low']:
            gap_size = candle3['low'] - candle1['high']
            mid_price = (candle3['low'] + candle1['high']) / 2

            ob_strength = calculate_order_block_strength(recent_data, i-2, i)

            if ob_strength > config.phob_confidence_min:
                phob_zones.append({
                    'type': 'bearish_phob',
                    'high': candle3['low'],
                    'low': candle1['high'],
                    'mid': mid_price,
                    'strength': ob_strength
                })
                confidence_scores.append(ob_strength)

    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

    return {
        'phob_detected': len(phob_zones) > 0,
        'confidence': avg_confidence,
        'zones': phob_zones,
        'zone_count': len(phob_zones)
    }


def calculate_order_block_strength(df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """Calculate order block strength based on volume and rejection patterns"""
    if end_idx >= len(df) or start_idx < 0:
        return 0.0

    ob_data = df.iloc[start_idx:end_idx+1]

    # Volume analysis
    if 'volume' in ob_data.columns:
        avg_volume = df['volume'].tail(20).mean()
        ob_volume = ob_data['volume'].mean()
        volume_strength = min(ob_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
    else:
        volume_strength = 1.0

    # Wick analysis for rejection
    total_wick_strength = 0.0
    for _, candle in ob_data.iterrows():
        metrics = calculate_wick_body_metrics(pd.DataFrame([candle]))
        if metrics:
            total_wick_strength += metrics['wick_dominance']

    avg_wick_strength = total_wick_strength / len(ob_data) if len(ob_data) > 0 else 0.0

    # Combined strength (0.0 to 1.0)
    strength = min((volume_strength * 0.6 + avg_wick_strength * 0.4), 1.0)

    return strength


def detect_fib_prime_zones(df: pd.DataFrame, config: BojanConfig) -> Dict[str, any]:
    """Detect Fibonacci .705/.786 confluence zones"""
    if len(df) < config.lookback:
        return {'in_prime_zone': False, 'zones': [], 'nearest_level': None}

    recent_data = df.tail(config.lookback)
    current_price = recent_data['close'].iloc[-1]

    # Calculate swing high/low for fib levels
    swing_high = recent_data['high'].max()
    swing_low = recent_data['low'].min()

    if swing_high <= swing_low:
        return {'in_prime_zone': False, 'zones': [], 'nearest_level': None}

    swing_range = swing_high - swing_low
    prime_zones = []

    # Calculate .705 and .786 levels
    for fib_level in config.fib_prime_zones:
        # Bullish retrace (from high)
        bull_level = swing_high - (swing_range * fib_level)

        # Bearish retrace (from low)
        bear_level = swing_low + (swing_range * fib_level)

        prime_zones.append({
            'level': fib_level,
            'bullish_zone': bull_level,
            'bearish_zone': bear_level
        })

    # Check if current price is near any prime zone (within 1%)
    tolerance = swing_range * 0.01  # 1% tolerance
    in_prime_zone = False
    nearest_level = None
    min_distance = float('inf')

    for zone in prime_zones:
        # Check bullish zone
        bull_distance = abs(current_price - zone['bullish_zone'])
        if bull_distance <= tolerance:
            in_prime_zone = True
            if bull_distance < min_distance:
                min_distance = bull_distance
                nearest_level = {'type': 'bullish', 'level': zone['level'], 'price': zone['bullish_zone']}

        # Check bearish zone
        bear_distance = abs(current_price - zone['bearish_zone'])
        if bear_distance <= tolerance:
            in_prime_zone = True
            if bear_distance < min_distance:
                min_distance = bear_distance
                nearest_level = {'type': 'bearish', 'level': zone['level'], 'price': zone['bearish_zone']}

    return {
        'in_prime_zone': in_prime_zone,
        'zones': prime_zones,
        'nearest_level': nearest_level,
        'current_price': current_price,
        'swing_high': swing_high,
        'swing_low': swing_low
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(df) < period:
        return 0.0

    # True Range calculation
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.tail(period).mean()

    return atr


def compute_bojan_score(df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, any]:
    """
    Comprehensive Bojan microstructure scoring

    Returns complete analysis including:
    - Wick magnets
    - Trap resets
    - Unfinished candles
    - pHOB zones
    - Fibonacci prime zones
    - Overall Bojan score
    """
    if len(df) < 1:
        return {
            'bojan_score': 0.0,
            'components': {},
            'signals': {},
            'direction_hint': 'neutral'
        }

    # Initialize config
    bojan_config = BojanConfig()
    if config:
        for key, value in config.items():
            if hasattr(bojan_config, key):
                setattr(bojan_config, key, value)

    # Component analysis
    wick_magnet = detect_wick_magnet(df, bojan_config)

    # Trap reset requires at least 2 bars
    if len(df) >= 2:
        trap_reset = detect_trap_reset(df, bojan_config)
    else:
        trap_reset = {'is_trap_reset': False, 'direction': None, 'strength': 0.0}

    unfinished = detect_unfinished_candles(df, bojan_config)
    phob_zones = detect_phob_zones(df, bojan_config)
    fib_prime = detect_fib_prime_zones(df, bojan_config)

    # Scoring weights
    components = {
        'wick_magnet': wick_magnet.get('strength', 0.0) * 0.25,
        'trap_reset': trap_reset.get('strength', 0.0) * 0.30,
        'unfinished': 0.10 if unfinished.get('has_unfinished', False) else 0.0,
        'phob': phob_zones.get('confidence', 0.0) * 0.20,
        'fib_prime': 0.15 if fib_prime.get('in_prime_zone', False) else 0.0
    }

    # Calculate total score
    base_score = sum(components.values())

    # Confluence bonuses
    confluence_bonus = 0.0

    # Wick magnet + trap reset confluence
    if wick_magnet.get('is_magnet', False) and trap_reset.get('is_trap_reset', False):
        confluence_bonus += 0.10

    # pHOB + fib prime confluence
    if phob_zones.get('phob_detected', False) and fib_prime.get('in_prime_zone', False):
        confluence_bonus += 0.15

    # Total Bojan score (capped at 1.0)
    total_score = min(base_score + confluence_bonus, 1.0)

    # Direction hint
    direction_hint = 'neutral'
    if trap_reset.get('is_trap_reset', False):
        direction_hint = trap_reset.get('direction', 'neutral')
    elif wick_magnet.get('is_magnet', False):
        if wick_magnet.get('magnet_type') == 'upper_wick_magnet':
            direction_hint = 'bearish'  # Upper wick suggests selling pressure
        elif wick_magnet.get('magnet_type') == 'lower_wick_magnet':
            direction_hint = 'bullish'  # Lower wick suggests buying pressure

    return {
        'bojan_score': float(total_score),
        'components': components,
        'confluence_bonus': float(confluence_bonus),
        'signals': {
            'wick_magnet': wick_magnet,
            'trap_reset': trap_reset,
            'unfinished': unfinished,
            'phob_zones': phob_zones,
            'fib_prime': fib_prime
        },
        'direction_hint': direction_hint,
        'config_used': bojan_config.__dict__
    }