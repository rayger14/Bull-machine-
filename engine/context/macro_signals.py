#!/usr/bin/env python3
"""
Macro Signal Helpers for Bull Machine v1.8.6
============================================

Provides EMA/ROC/zscore helpers and state detection functions for:
- DXY (Dollar strength/weakness)
- Oil (WTI regime)
- Gold (Flight-to-safety)
- Yields (Bond stress, curve steepening)
- TOTAL/TOTAL2/TOTAL3 (Crypto market breadth)
- USDT.D (Stablecoin dominance)

All functions are designed to be causal (no future leak) and work with
limited data availability.
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ema(values: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential moving average (causal)

    Args:
        values: Array of prices/values
        period: EMA period

    Returns:
        EMA array (same length as input, NaN for insufficient data)
    """
    if len(values) == 0:
        return np.array([])

    alpha = 2.0 / (period + 1)
    result = np.full_like(values, np.nan, dtype=float)

    # Initialize with first valid value
    for i in range(len(values)):
        if not np.isnan(values[i]):
            result[i] = values[i]
            break

    # Compute EMA causally
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            result[i] = result[i-1]
        elif np.isnan(result[i-1]):
            result[i] = values[i]
        else:
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]

    # Mask first (period - 1) values as NaN (warmup)
    result[:period-1] = np.nan

    return result


def roc(values: np.ndarray, period: int) -> np.ndarray:
    """
    Rate of change (% change over period)

    Args:
        values: Array of prices/values
        period: Lookback period

    Returns:
        ROC array (percentage change)
    """
    if len(values) <= period:
        return np.full_like(values, np.nan, dtype=float)

    result = np.full_like(values, np.nan, dtype=float)

    for i in range(period, len(values)):
        if values[i-period] != 0 and not np.isnan(values[i]) and not np.isnan(values[i-period]):
            result[i] = ((values[i] - values[i-period]) / values[i-period]) * 100.0

    return result


def zscore(values: np.ndarray, lookback: int) -> np.ndarray:
    """
    Rolling z-score (causal)

    Args:
        values: Array of prices/values
        lookback: Rolling window for mean/std

    Returns:
        Z-score array
    """
    if len(values) < lookback:
        return np.full_like(values, np.nan, dtype=float)

    result = np.full_like(values, np.nan, dtype=float)

    for i in range(lookback-1, len(values)):
        window = values[i-lookback+1:i+1]
        valid = window[~np.isnan(window)]

        if len(valid) >= lookback // 2:  # Need at least half the data
            mean = np.mean(valid)
            std = np.std(valid)

            if std > 0 and not np.isnan(values[i]):
                result[i] = (values[i] - mean) / std

    return result


# ============================================================================
# STATE DETECTION FUNCTIONS
# ============================================================================

def dxy_states(
    dxy: float,
    dxy_ema_fast: float,
    dxy_ema_slow: float,
    dxy_zscore: float,
    thresholds: Dict
) -> Tuple[str, float]:
    """
    DXY state detection: breakout (risk-off) vs breakdown (risk-on)

    Args:
        dxy: Current DXY value
        dxy_ema_fast: Fast EMA (e.g., 10-period)
        dxy_ema_slow: Slow EMA (e.g., 50-period)
        dxy_zscore: Z-score vs recent range
        thresholds: Config dict with 'dxy_breakout_z' and 'dxy_breakdown_z'

    Returns:
        (state, penalty):
            - ('breakout', -0.10): DXY surging, risk-off
            - ('breakdown', +0.05): DXY weakening, risk-on
            - ('neutral', 0.0): No clear signal
    """
    # Check for missing data
    if np.isnan(dxy) or np.isnan(dxy_ema_fast) or np.isnan(dxy_ema_slow) or np.isnan(dxy_zscore):
        return ('neutral', 0.0)

    breakout_z = thresholds.get('dxy_breakout_z', 1.5)
    breakdown_z = thresholds.get('dxy_breakdown_z', -1.5)

    # DXY breakout: strong dollar = risk-off for crypto
    if dxy_zscore > breakout_z and dxy_ema_fast > dxy_ema_slow:
        penalty = -0.10  # Reduce fusion score
        return ('breakout', penalty)

    # DXY breakdown: weak dollar = risk-on for crypto
    elif dxy_zscore < breakdown_z and dxy_ema_fast < dxy_ema_slow:
        boost = +0.05  # Increase fusion score
        return ('breakdown', boost)

    return ('neutral', 0.0)


def oil_state(
    oil_price: float,
    oil_roc_10: float,
    oil_zscore: float,
    thresholds: Dict
) -> Tuple[str, float]:
    """
    Oil state detection: hot (inflationary pressure) vs cool (easing)

    Args:
        oil_price: Current WTI price
        oil_roc_10: 10-period rate of change
        oil_zscore: Z-score vs recent range
        thresholds: Config dict with 'oil_hot_z' and 'oil_roc_threshold'

    Returns:
        (state, penalty):
            - ('hot', -0.05): Oil surging, inflation fears
            - ('cool', +0.03): Oil falling, easing pressure
            - ('neutral', 0.0): Normal regime
    """
    if np.isnan(oil_price) or np.isnan(oil_roc_10) or np.isnan(oil_zscore):
        return ('neutral', 0.0)

    hot_z = thresholds.get('oil_hot_z', 1.5)
    roc_threshold = thresholds.get('oil_roc_threshold', 15.0)

    # Oil surging: inflationary pressure, risk-off
    if oil_zscore > hot_z or oil_roc_10 > roc_threshold:
        penalty = -0.05
        return ('hot', penalty)

    # Oil falling: easing pressure, risk-on
    elif oil_roc_10 < -roc_threshold:
        boost = +0.03
        return ('cool', boost)

    return ('neutral', 0.0)


def gold_hedge(
    gold_price: float,
    gold_roc_10: float,
    gold_zscore: float,
    thresholds: Dict
) -> Tuple[str, float]:
    """
    Gold flight-to-safety detection

    Args:
        gold_price: Current XAUUSD price
        gold_roc_10: 10-period rate of change
        gold_zscore: Z-score vs recent range
        thresholds: Config dict with 'gold_flight_z' and 'gold_roc_threshold'

    Returns:
        (state, penalty):
            - ('flight', -0.05): Gold surging, flight-to-safety
            - ('neutral', 0.0): Normal regime
    """
    if np.isnan(gold_price) or np.isnan(gold_roc_10) or np.isnan(gold_zscore):
        return ('neutral', 0.0)

    flight_z = thresholds.get('gold_flight_z', 1.5)
    roc_threshold = thresholds.get('gold_roc_threshold', 10.0)

    # Gold surging: flight-to-safety, risk-off
    if gold_zscore > flight_z or gold_roc_10 > roc_threshold:
        penalty = -0.05
        return ('flight', penalty)

    return ('neutral', 0.0)


def yield_spike(
    us2y: float,
    us10y: float,
    us10y_roc_5: float,
    thresholds: Dict
) -> Tuple[str, float]:
    """
    Bond market stress detection (yield spike)

    Args:
        us2y: 2-year Treasury yield
        us10y: 10-year Treasury yield
        us10y_roc_5: 5-period rate of change on 10Y
        thresholds: Config dict with 'yield_spike_roc'

    Returns:
        (state, penalty):
            - ('spike', -0.08): Yields spiking, bond stress
            - ('neutral', 0.0): Normal regime
    """
    if np.isnan(us10y) or np.isnan(us10y_roc_5):
        return ('neutral', 0.0)

    spike_roc = thresholds.get('yield_spike_roc', 10.0)

    # 10Y yield spiking: bond stress, risk-off
    if us10y_roc_5 > spike_roc:
        penalty = -0.08
        return ('spike', penalty)

    return ('neutral', 0.0)


def curve_steepening(
    us2y: float,
    us10y: float,
    spread_ema: float,
    thresholds: Dict
) -> Tuple[str, float]:
    """
    Yield curve steepening detection (growth expectations)

    Args:
        us2y: 2-year Treasury yield
        us10y: 10-year Treasury yield
        spread_ema: EMA of 10Y-2Y spread
        thresholds: Config dict with 'curve_steep_threshold'

    Returns:
        (state, boost):
            - ('steepening', +0.03): Curve steepening, growth expectations
            - ('flattening', -0.03): Curve flattening, recession fears
            - ('neutral', 0.0): Normal regime
    """
    if np.isnan(us2y) or np.isnan(us10y) or np.isnan(spread_ema):
        return ('neutral', 0.0)

    current_spread = us10y - us2y
    steep_threshold = thresholds.get('curve_steep_threshold', 0.30)

    # Curve steepening vs EMA: growth expectations, risk-on
    if current_spread > spread_ema + steep_threshold:
        boost = +0.03
        return ('steepening', boost)

    # Curve flattening: recession fears, risk-off
    elif current_spread < spread_ema - steep_threshold:
        penalty = -0.03
        return ('flattening', penalty)

    return ('neutral', 0.0)


def total_breadth(
    total_mc: float,
    total2_mc: float,
    total3_mc: float,
    thresholds: Dict
) -> Tuple[str, float]:
    """
    Crypto market breadth (altcoin vs Bitcoin strength)

    Args:
        total_mc: Total crypto market cap
        total2_mc: Total excluding BTC
        total3_mc: Total excluding BTC+ETH
        thresholds: Config dict with 'breadth_threshold'

    Returns:
        (state, boost):
            - ('strong', +0.05): Altcoins outperforming, broad rally
            - ('weak', -0.05): Bitcoin dominance rising, alt weakness
            - ('neutral', 0.0): Normal regime
    """
    if np.isnan(total_mc) or np.isnan(total2_mc) or total_mc == 0:
        return ('neutral', 0.0)

    # Calculate BTC dominance
    btc_dominance = 1.0 - (total2_mc / total_mc)

    breadth_low = thresholds.get('breadth_dominance_low', 0.50)  # < 50% = alt season
    breadth_high = thresholds.get('breadth_dominance_high', 0.60)  # > 60% = BTC flight

    # Altcoin strength: BTC dominance falling
    if btc_dominance < breadth_low:
        boost = +0.05
        return ('strong', boost)

    # Bitcoin flight: Dominance rising
    elif btc_dominance > breadth_high:
        penalty = -0.05
        return ('weak', penalty)

    return ('neutral', 0.0)


def usdt_d_breakout(
    usdt_dominance: float,
    usdt_d_ema: float,
    usdt_d_zscore: float,
    thresholds: Dict
) -> Tuple[str, float]:
    """
    USDT.D breakout detection (stablecoin dominance = risk-off)

    Args:
        usdt_dominance: Current USDT dominance %
        usdt_d_ema: EMA of USDT.D
        usdt_d_zscore: Z-score vs recent range
        thresholds: Config dict with 'usdt_d_breakout_z'

    Returns:
        (state, penalty):
            - ('breakout', -0.05): USDT.D surging, flight to stables
            - ('breakdown', +0.03): USDT.D falling, risk-on
            - ('neutral', 0.0): Normal regime
    """
    if np.isnan(usdt_dominance) or np.isnan(usdt_d_ema) or np.isnan(usdt_d_zscore):
        return ('neutral', 0.0)

    breakout_z = thresholds.get('usdt_d_breakout_z', 1.5)

    # USDT.D breakout: flight to stables, risk-off
    if usdt_d_zscore > breakout_z and usdt_dominance > usdt_d_ema:
        penalty = -0.05
        return ('breakout', penalty)

    # USDT.D breakdown: risk-on
    elif usdt_d_zscore < -breakout_z and usdt_dominance < usdt_d_ema:
        boost = +0.03
        return ('breakdown', boost)

    return ('neutral', 0.0)


# ============================================================================
# COMPOSITE SCORE BUILDER
# ============================================================================

def build_macro_composite(
    snapshot: Dict,
    thresholds: Dict,
    weights: Dict
) -> Dict:
    """
    Build 8-factor macro composite score with state breakdown

    Args:
        snapshot: Current macro data dict with all indicators
        thresholds: Detection thresholds from config
        weights: Factor weights from config

    Returns:
        Dict with:
            - 'composite': Final composite score (-1.0 to +1.0)
            - 'states': Dict of individual state detections
            - 'adjustments': Dict of individual adjustments
    """
    # Initialize result
    result = {
        'composite': 0.0,
        'states': {},
        'adjustments': {}
    }

    # Extract indicators from snapshot
    vix = snapshot.get('vix', np.nan)
    move = snapshot.get('move', np.nan)
    dxy = snapshot.get('dxy', np.nan)
    oil = snapshot.get('oil', np.nan)
    gold = snapshot.get('gold', np.nan)
    us2y = snapshot.get('us2y', np.nan)
    us10y = snapshot.get('us10y', np.nan)
    total_mc = snapshot.get('total_mc', np.nan)
    total2_mc = snapshot.get('total2_mc', np.nan)
    usdt_d = snapshot.get('usdt_d', np.nan)

    # Get pre-computed indicators
    dxy_ema_fast = snapshot.get('dxy_ema_10', np.nan)
    dxy_ema_slow = snapshot.get('dxy_ema_50', np.nan)
    dxy_z = snapshot.get('dxy_zscore', np.nan)

    oil_roc = snapshot.get('oil_roc_10', np.nan)
    oil_z = snapshot.get('oil_zscore', np.nan)

    gold_roc = snapshot.get('gold_roc_10', np.nan)
    gold_z = snapshot.get('gold_zscore', np.nan)

    us10y_roc = snapshot.get('us10y_roc_5', np.nan)
    spread_ema = snapshot.get('spread_ema_20', np.nan)

    usdt_d_ema = snapshot.get('usdt_d_ema_20', np.nan)
    usdt_d_z = snapshot.get('usdt_d_zscore', np.nan)

    # Run state detections
    dxy_state, dxy_adj = dxy_states(dxy, dxy_ema_fast, dxy_ema_slow, dxy_z, thresholds)
    oil_st, oil_adj = oil_state(oil, oil_roc, oil_z, thresholds)
    gold_st, gold_adj = gold_hedge(gold, gold_roc, gold_z, thresholds)
    yield_st, yield_adj = yield_spike(us2y, us10y, us10y_roc, thresholds)
    curve_st, curve_adj = curve_steepening(us2y, us10y, spread_ema, thresholds)
    breadth_st, breadth_adj = total_breadth(total_mc, total2_mc, np.nan, thresholds)
    usdt_st, usdt_adj = usdt_d_breakout(usdt_d, usdt_d_ema, usdt_d_z, thresholds)

    # Store states
    result['states'] = {
        'dxy': dxy_state,
        'oil': oil_st,
        'gold': gold_st,
        'yield': yield_st,
        'curve': curve_st,
        'breadth': breadth_st,
        'usdt_d': usdt_st
    }

    # Store adjustments
    result['adjustments'] = {
        'dxy': dxy_adj,
        'oil': oil_adj,
        'gold': gold_adj,
        'yield': yield_adj,
        'curve': curve_adj,
        'breadth': breadth_adj,
        'usdt_d': usdt_adj
    }

    # Compute weighted composite (cap each adjustment at ±0.10)
    adjustments = [
        np.clip(dxy_adj, -0.10, 0.10),
        np.clip(oil_adj, -0.10, 0.10),
        np.clip(gold_adj, -0.10, 0.10),
        np.clip(yield_adj, -0.10, 0.10),
        np.clip(curve_adj, -0.10, 0.10),
        np.clip(breadth_adj, -0.10, 0.10),
        np.clip(usdt_adj, -0.10, 0.10)
    ]

    factor_weights = [
        weights.get('dxy_weight', 0.15),
        weights.get('oil_weight', 0.10),
        weights.get('gold_weight', 0.05),
        weights.get('yield_weight', 0.10),
        weights.get('curve_weight', 0.05),
        weights.get('breadth_weight', 0.05),
        weights.get('usdt_d_weight', 0.05)
    ]

    # Weighted sum (capped at ±0.10 total)
    composite = sum(a * w for a, w in zip(adjustments, factor_weights))
    composite = np.clip(composite, -0.10, 0.10)

    result['composite'] = composite

    return result
