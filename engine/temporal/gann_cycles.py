"""
Bull Machine v1.8.6 - Temporal/Gann Cycles Module

Implements:
1. ACF-based cycle detection (30/60/90 day vibrations)
2. Square of 9 proximity scoring
3. Gann angle adherence
4. Thermo-floor (mining cost floor)
5. Log premium (time-based difficulty multiplier)
6. Logistic bid (re-accumulation phase scoring)
7. LPPLS blowoff detection

Trader alignment:
- @Wyckoff_Insider: Accumulation/distribution cycles
- @Moneytaur: Smart money timing, institutional re-entry
- @ZeroIKA: Frequency-domain analysis, harmonic cycles

Design principles:
- Feature-flagged (temporal.enabled)
- Bounded bonus/veto (±0.15 default)
- Deterministic (same inputs → same outputs)
- No I/O - all data passed in
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import correlate
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


def _cycles_acf(df_1d: pd.DataFrame, config: Dict) -> Tuple[float, str, List[int]]:
    """
    ACF-based cycle detection for 30/60/90 day vibrations.

    Returns:
        (confluence_score, cycle_phase, detected_periods)

    Algorithm:
    1. Calculate autocorrelation on 1D closes (180 day lookback)
    2. Find peaks at 30±5, 60±5, 90±5 day lags
    3. Score = (sum of peak ACF values at cycle lags) / 3
    4. Phase = "accumulation" if near trough, "distribution" if near peak
    """
    lookback = config.get('acf_lookback_days', 180)
    target_cycles = config.get('target_cycles', [30, 60, 90])
    tolerance = config.get('cycle_tolerance_days', 5)

    if len(df_1d) < lookback:
        return 0.0, "insufficient_data", []

    prices = df_1d['close'].values[-lookback:]

    # Detrend using log returns
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)

    # Autocorrelation
    acf = correlate(log_returns, log_returns, mode='full')
    acf = acf[len(acf)//2:]  # Keep positive lags
    acf = acf / acf[0]  # Normalize

    # Find peaks near target cycles
    detected_peaks = []
    peak_values = []

    for cycle in target_cycles:
        start_lag = max(1, cycle - tolerance)
        end_lag = min(len(acf)-1, cycle + tolerance)

        if start_lag < end_lag:
            window = acf[start_lag:end_lag]
            if len(window) > 0:
                peak_idx = np.argmax(window) + start_lag
                peak_val = acf[peak_idx]

                # Only count if ACF > 0.15 (meaningful correlation)
                if peak_val > 0.15:
                    detected_peaks.append(peak_idx)
                    peak_values.append(peak_val)

    # Confluence score
    if len(peak_values) > 0:
        confluence = np.mean(peak_values)
    else:
        confluence = 0.0

    # Determine cycle phase (recent price action vs 90-day trend)
    recent_price = prices[-1]
    ma_90 = np.mean(prices[-min(90, len(prices)):])

    if recent_price < ma_90 * 0.95:
        phase = "accumulation"
    elif recent_price > ma_90 * 1.05:
        phase = "distribution"
    else:
        phase = "equilibrium"

    return confluence, phase, detected_peaks


def _square9(price: float, config: Dict) -> Tuple[float, Optional[float]]:
    """
    Gann Square of 9 proximity scoring.

    Returns:
        (proximity_score, nearest_sq9_level)

    Algorithm:
    1. Calculate nearest Square of 9 level: level = round(price / step) * step
    2. Distance = abs(price - level) / price
    3. Score = max(0, 1 - distance / tolerance)
    """
    step = config.get('square9_step', 9.0)
    tolerance = config.get('square9_tolerance', 2.0)  # % tolerance

    # Find nearest level
    nearest_level = round(price / step) * step

    # Calculate distance
    distance_pct = abs(price - nearest_level) / price * 100

    # Score (linear decay from 1.0 to 0.0 as distance increases)
    score = max(0.0, 1.0 - (distance_pct / tolerance))

    return score, nearest_level if score > 0 else None


def _angles_score(df_1h: pd.DataFrame, df_1d: pd.DataFrame, config: Dict) -> float:
    """
    Gann angle adherence scoring.

    Returns:
        angle_score (0-1)

    Algorithm:
    1. Calculate 1x1 angle (45 degrees): 1 unit price per 1 unit time
    2. Measure recent price trajectory vs 1x1 angle
    3. Score based on alignment (higher score when trajectory matches 1x1)
    """
    lookback_bars = config.get('gann_angle_lookback', 24)  # 24 hours

    if len(df_1h) < lookback_bars:
        return 0.0

    recent = df_1h.tail(lookback_bars)

    # Calculate slope (price change per bar)
    price_start = recent['close'].iloc[0]
    price_end = recent['close'].iloc[-1]
    price_change = (price_end - price_start) / price_start

    # Normalize by ATR to get units of risk
    atr = recent['high'].subtract(recent['low']).mean()
    normalized_slope = abs(price_change * price_start / atr) if atr > 0 else 0

    # 1x1 angle = 1 ATR per bar (idealized)
    # Score based on proximity to 1x1
    target_slope = 1.0
    deviation = abs(normalized_slope - target_slope)

    # Linear decay (perfect match = 1.0, >2 ATR deviation = 0.0)
    score = max(0.0, 1.0 - deviation / 2.0)

    return score


def _thermo_floor(df_1d: pd.DataFrame, macro_cache: Dict, config: Dict) -> Tuple[float, float]:
    """
    Calculate mining cost floor (thermo price).

    Returns:
        (floor_price, distance_from_floor)

    Algorithm:
    1. floor = hashrate × energy_cost × 600s × 144 blocks/day
    2. distance = (current_price - floor) / floor
    3. If price < floor * 1.1, bullish (miners capitulating)
    """
    # Default values if macro data not available
    hashrate = macro_cache.get('HASHRATE', {}).get('value', 600e18)  # H/s
    energy_cost_per_hash = config.get('energy_cost_per_hash', 0.05e-12)  # $/hash

    # Mining cost = hashrate × cost × block_time × blocks_per_day
    block_time = 600  # seconds
    blocks_per_day = 144
    daily_mining_cost = hashrate * energy_cost_per_hash * block_time * blocks_per_day

    # BTC per block (post-2024 halving = 3.125 BTC)
    btc_per_block = 3.125
    daily_btc_issuance = btc_per_block * blocks_per_day

    # Thermo floor = daily cost / daily issuance
    floor_price = daily_mining_cost / daily_btc_issuance if daily_btc_issuance > 0 else 0

    # Current price
    current_price = df_1d['close'].iloc[-1]

    # Distance from floor (% above/below)
    distance = (current_price - floor_price) / floor_price if floor_price > 0 else 0

    return floor_price, distance


def _log_premium(df_1d: pd.DataFrame, macro_cache: Dict, config: Dict) -> float:
    """
    Calculate time-based premium multiplier.

    Returns:
        premium_multiplier (≥1.0)

    Algorithm:
    premium = 1 + β × T × log(difficulty)

    Where:
    - β = sensitivity parameter (default 0.0001)
    - T = days since last halving / 1460 (normalize to 4-year cycle)
    - difficulty = network difficulty
    """
    beta = config.get('log_premium_beta', 0.0001)

    # Days since last halving (April 20, 2024)
    last_halving = pd.Timestamp('2024-04-20')
    current_ts = df_1d.index[-1]
    days_since_halving = (current_ts - last_halving).days

    # Normalize to 4-year cycle
    T = days_since_halving / 1460.0

    # Network difficulty
    difficulty = macro_cache.get('DIFFICULTY', {}).get('value', 1e14)

    # Premium
    premium = 1.0 + beta * T * np.log(difficulty)

    return max(1.0, premium)  # Floor at 1.0


def _logistic_bid(df_1h: pd.DataFrame, df_1d: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Score institutional re-accumulation phase via sigmoid.

    Returns:
        (score, phase)

    Algorithm:
    score = 1 / (1 + exp(-k × (volume_ratio - volume_threshold)))

    Where:
    - volume_ratio = recent_volume / historical_average
    - k = steepness (default 5.0)
    - volume_threshold = 1.2 (20% above average)

    Phase interpretation:
    - score > 0.7: "strong_bid" (institutional accumulation)
    - 0.3 < score < 0.7: "neutral"
    - score < 0.3: "weak_bid"
    """
    k = config.get('logistic_k', 5.0)
    threshold = config.get('volume_threshold', 1.2)

    if len(df_1d) < 30:
        return 0.5, "neutral"

    # Recent volume (7-day average)
    recent_vol = df_1d['volume'].tail(7).mean()

    # Historical average (30-day)
    hist_vol = df_1d['volume'].tail(30).mean()

    volume_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0

    # Logistic function
    score = 1.0 / (1.0 + np.exp(-k * (volume_ratio - threshold)))

    # Phase determination
    if score > 0.7:
        phase = "strong_bid"
    elif score < 0.3:
        phase = "weak_bid"
    else:
        phase = "neutral"

    return score, phase


def _lppls_simple(df_1d: pd.DataFrame, config: Dict) -> Tuple[bool, float, Optional[str]]:
    """
    Log-periodic power law singularity (LPPLS) blowoff detection.

    Returns:
        (veto, confidence, explanation)

    Simplified LPPLS:
    log(price) = A + B × (tc - t)^m × [1 + C × cos(ω × log(tc - t) + φ)]

    Where:
    - tc = critical time (singularity)
    - m = power law exponent (typically 0.2-0.9)
    - ω = log-periodic frequency
    - C = oscillation amplitude

    Detection heuristic (simplified for speed):
    1. Fit power law to recent price acceleration
    2. If m < 0.5 and price > 2× 90-day MA and volume declining → blowoff
    """
    veto_threshold = config.get('lppls_veto_confidence', 0.75)
    lookback = min(90, len(df_1d))

    if lookback < 30:
        return False, 0.0, None

    recent = df_1d.tail(lookback)
    log_prices = np.log(recent['close'].values)
    t = np.arange(len(log_prices))

    # Simple power law fit: log(P) = a + b × t^m
    try:
        def power_law(x, a, b, m):
            return a + b * np.power(x, m)

        popt, _ = curve_fit(power_law, t, log_prices, p0=[log_prices[0], 0.01, 0.5], maxfev=1000)
        a, b, m = popt

        # Check blowoff conditions
        price_current = recent['close'].iloc[-1]
        ma_90 = recent['close'].mean()
        price_extension = price_current / ma_90

        # Volume declining
        vol_recent = recent['volume'].tail(10).mean()
        vol_earlier = recent['volume'].head(20).mean()
        volume_declining = vol_recent < vol_earlier * 0.8

        # Blowoff heuristic
        blowoff_score = 0.0

        if m < 0.5:  # Decelerating power law
            blowoff_score += 0.3

        if price_extension > 2.0:  # Price 2× above MA
            blowoff_score += 0.4

        if volume_declining:  # Climax volume exhaustion
            blowoff_score += 0.3

        veto = blowoff_score >= veto_threshold

        explanation = None
        if veto:
            explanation = f"LPPLS blowoff detected: m={m:.2f}, extension={price_extension:.2f}×, vol_declining={volume_declining}"

        return veto, blowoff_score, explanation

    except Exception:
        # Fit failed, no veto
        return False, 0.0, None


def temporal_signal(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    config: Dict,
    macro_cache: Optional[Dict] = None
) -> Dict:
    """
    Main temporal/Gann signal calculator.

    Args:
        df_1h: 1-hour OHLCV dataframe
        df_4h: 4-hour OHLCV dataframe
        df_1d: 1-day OHLCV dataframe
        config: temporal configuration dict
        macro_cache: optional macro data (HASHRATE, DIFFICULTY, etc.)

    Returns:
        {
            'confluence_score': float (0-1),
            'cycle_phase': str,
            'veto': bool,
            'veto_reason': Optional[str],
            'features': {
                'acf_score': float,
                'acf_cycles': List[int],
                'square9_score': float,
                'square9_level': Optional[float],
                'gann_angle_score': float,
                'thermo_floor': float,
                'thermo_distance': float,
                'log_premium': float,
                'logistic_bid_score': float,
                'logistic_phase': str,
                'lppls_veto': bool,
                'lppls_confidence': float
            }
        }
    """
    if macro_cache is None:
        macro_cache = {}

    # Initialize result
    result = {
        'confluence_score': 0.0,
        'cycle_phase': 'unknown',
        'veto': False,
        'veto_reason': None,
        'features': {}
    }

    # 1. ACF Cycles
    acf_score, cycle_phase, detected_cycles = _cycles_acf(df_1d, config)
    result['features']['acf_score'] = acf_score
    result['features']['acf_cycles'] = detected_cycles
    result['cycle_phase'] = cycle_phase

    # 2. Square of 9
    current_price = df_1h['close'].iloc[-1]
    sq9_score, sq9_level = _square9(current_price, config)
    result['features']['square9_score'] = sq9_score
    result['features']['square9_level'] = sq9_level

    # 3. Gann Angles
    angle_score = _angles_score(df_1h, df_1d, config)
    result['features']['gann_angle_score'] = angle_score

    # 4. Thermo Floor
    floor_price, floor_distance = _thermo_floor(df_1d, macro_cache, config)
    result['features']['thermo_floor'] = floor_price
    result['features']['thermo_distance'] = floor_distance

    # 5. Log Premium
    premium = _log_premium(df_1d, macro_cache, config)
    result['features']['log_premium'] = premium

    # 6. Logistic Bid
    bid_score, bid_phase = _logistic_bid(df_1h, df_1d, config)
    result['features']['logistic_bid_score'] = bid_score
    result['features']['logistic_phase'] = bid_phase

    # 7. LPPLS Blowoff
    lppls_veto, lppls_conf, lppls_reason = _lppls_simple(df_1d, config)
    result['features']['lppls_veto'] = lppls_veto
    result['features']['lppls_confidence'] = lppls_conf

    if lppls_veto:
        result['veto'] = True
        result['veto_reason'] = lppls_reason
        return result

    # Calculate confluence score (weighted average)
    weights = config.get('feature_weights', {
        'acf': 0.25,
        'square9': 0.20,
        'angles': 0.15,
        'thermo': 0.15,
        'logistic': 0.25
    })

    # Normalize thermo distance to [0, 1]
    # Near floor (distance < 0.1) = 1.0, far from floor (distance > 2.0) = 0.0
    thermo_score = max(0.0, min(1.0, 1.0 - floor_distance / 2.0))

    confluence = (
        weights['acf'] * acf_score +
        weights['square9'] * sq9_score +
        weights['angles'] * angle_score +
        weights['thermo'] * thermo_score +
        weights['logistic'] * bid_score
    )

    # Apply log premium multiplier (bounded to avoid excessive boost)
    confluence = min(1.0, confluence * min(premium, 1.2))

    result['confluence_score'] = confluence

    return result
