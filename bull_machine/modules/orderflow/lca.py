"""
Bull Machine v1.5.0 - Orderflow LCA (Liquidity Capture Analysis)
BOS (Break of Structure) detection with intent nudging.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from bull_machine.core.telemetry import log_telemetry


def calculate_cvd(df: pd.DataFrame) -> float:
    """
    Calculate Cumulative Volume Delta (CVD).

    CVD measures the cumulative difference between buy and sell volume,
    helping identify true market intent behind price movements.

    Simplified implementation using price action to estimate buy/sell volume.
    For production, use actual orderbook data if available.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        float: Latest cumulative volume delta
    """
    if len(df) < 2:
        return 0.0

    # Estimate buy/sell volume from price action
    # Buy volume: portion of volume when close is in upper half of range
    # Sell volume: portion when close is in lower half
    price_range = df['high'] - df['low']

    # Avoid division by zero
    valid_range = price_range > 0

    buy_vol = pd.Series(0.0, index=df.index)
    sell_vol = pd.Series(0.0, index=df.index)

    # Calculate buy/sell volumes where range is valid
    buy_vol[valid_range] = df.loc[valid_range, 'volume'] * \
        (df.loc[valid_range, 'close'] - df.loc[valid_range, 'low']) / price_range[valid_range]
    sell_vol[valid_range] = df.loc[valid_range, 'volume'] - buy_vol[valid_range]

    # For bars with no range, split volume 50/50
    buy_vol[~valid_range] = df.loc[~valid_range, 'volume'] * 0.5
    sell_vol[~valid_range] = df.loc[~valid_range, 'volume'] * 0.5

    # Calculate cumulative delta
    cvd = (buy_vol - sell_vol).cumsum()

    return float(cvd.iloc[-1]) if len(cvd) > 0 else 0.0


def detect_bos(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """
    Detect Break of Structure (BOS) events.

    A BOS occurs when price breaks above recent swing highs (bullish BOS)
    or below recent swing lows (bearish BOS).

    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to look back for swing detection

    Returns:
        Dict containing BOS detection results
    """
    if len(df) < lookback + 2:
        return {"detected": False, "direction": None, "strength": 0.0}

    # Calculate swing levels - look at recent high/low before current bar
    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]

    # Get the highest high and lowest low from the lookback period (excluding current bar)
    lookback_data = df.iloc[-(lookback+1):-1]  # Last lookback bars before current
    prev_swing_high = lookback_data["high"].max()
    prev_swing_low = lookback_data["low"].min()

    # Enhanced BOS detection with body close validation (IamZeroIka's 1/3 rule)
    current_close = df["close"].iloc[-1]
    current_open = df["open"].iloc[-1]
    body_range = current_high - current_low

    # Check for liquidity sweep (fake break followed by reversal - Crypto Chase logic)
    liquidity_sweep = False
    if len(df) >= 3:
        # Check if previous bar broke structure but current reversed
        prev_high = df["high"].iloc[-2]
        prev_low = df["low"].iloc[-2]
        prev_close = df["close"].iloc[-2]

        # Bullish sweep: previous broke below low, current recovered above
        bullish_sweep = (prev_low < prev_swing_low and current_close > prev_swing_low)
        # Bearish sweep: previous broke above high, current recovered below
        bearish_sweep = (prev_high > prev_swing_high and current_close < prev_swing_high)
        liquidity_sweep = bullish_sweep or bearish_sweep

    # Detect bullish BOS with body close validation
    bullish_bos = False
    bullish_strength = 0.0
    if current_high > prev_swing_high:
        # Require close to be at least 1/3 beyond the swing level (body close rule)
        if body_range > 0:
            close_pct_above = (current_close - prev_swing_high) / body_range
            bullish_bos = close_pct_above >= (1/3) or liquidity_sweep
            bullish_strength = (current_high - prev_swing_high) / prev_swing_high if prev_swing_high > 0 else 0

    # Detect bearish BOS with body close validation
    bearish_bos = False
    bearish_strength = 0.0
    if current_low < prev_swing_low:
        # Require close to be at least 1/3 beyond the swing level (body close rule)
        if body_range > 0:
            close_pct_below = (prev_swing_low - current_close) / body_range
            bearish_bos = close_pct_below >= (1/3) or liquidity_sweep
            bearish_strength = (prev_swing_low - current_low) / prev_swing_low if prev_swing_low > 0 else 0

    bos_result = {
        "detected": bool(bullish_bos or bearish_bos),
        "direction": "bullish" if bullish_bos else ("bearish" if bearish_bos else None),
        "strength": float(max(bullish_strength, bearish_strength)),
        "bullish_bos": bool(bullish_bos),
        "bearish_bos": bool(bearish_bos),
        "swing_levels": {
            "prev_high": float(prev_swing_high) if not pd.isna(prev_swing_high) else None,
            "prev_low": float(prev_swing_low) if not pd.isna(prev_swing_low) else None,
            "current_high": float(current_high),
            "current_low": float(current_low)
        }
    }

    log_telemetry("layer_masks.json", {
        "bos_detected": bos_result["detected"],
        "bos_direction": bos_result["direction"],
        "bos_strength": float(bos_result["strength"])
    })

    return bos_result


def calculate_cvd(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Cumulative Volume Delta with slope for divergence detection.

    CVD measures buying vs selling pressure by estimating volume distribution
    within each bar based on where price closes relative to the range.
    Enhanced with IamZeroIka's slope analysis for trend divergence detection.

    Args:
        df: DataFrame with OHLC and volume data

    Returns:
        Dict with CVD delta and slope values
    """
    if len(df) < 2:
        return {"delta": 0.0, "slope": 0.0, "current_delta": 0.0, "delta_ma": 0.0}

    try:
        # Estimate buying vs selling volume based on bar position
        # Close near high = more buying, close near low = more selling
        ranges = df['high'] - df['low']

        # Avoid division by zero
        ranges = ranges.where(ranges != 0, df['close'] * 0.001)  # Use small fraction of price

        # Calculate volume distribution
        buy_pressure = (df['close'] - df['low']) / ranges
        buy_volume = df['volume'] * buy_pressure
        sell_volume = df['volume'] * (1 - buy_pressure)

        # Calculate cumulative delta
        volume_delta = buy_volume - sell_volume

        # Handle NaN values by filling with zeros
        volume_delta = volume_delta.fillna(0)
        cvd = volume_delta.cumsum()

        # IamZeroIka slope for divergence detection (post:39)
        cvd_slope = 0.0
        slope_window = min(10, len(cvd) - 1)
        if slope_window > 0:
            cvd_slope = (cvd.iloc[-1] - cvd.iloc[-slope_window-1]) / slope_window

        # Calculate delta moving average with available data
        ma_window = min(10, len(volume_delta))
        delta_ma = volume_delta.rolling(ma_window).mean().iloc[-1] if ma_window > 0 else 0.0

        return {
            "delta": float(cvd.iloc[-1]),
            "slope": float(cvd_slope),
            "current_delta": float(volume_delta.iloc[-1]),
            "delta_ma": float(delta_ma)
        }

    except Exception as e:
        log_telemetry("layer_masks.json", {
            "cvd_calculation_error": str(e),
            "fallback_values": True
        })
        return {"delta": 0.0, "slope": 0.0, "current_delta": 0.0, "delta_ma": 0.0}


def detect_liquidity_capture(df: pd.DataFrame) -> bool:
    """
    Detect Liquidity Capture Analysis (LCA) patterns.

    LCA identifies when current close is above the previous bar's high,
    indicating potential liquidity capture above resistance.

    Args:
        df: DataFrame with OHLC data

    Returns:
        bool: True if LCA pattern detected
    """
    if len(df) < 2:
        return False

    current_close = df["close"].iloc[-1]
    prev_high = df["high"].iloc[-2]

    # LCA: Current close above previous high
    lca_detected = current_close > prev_high

    log_telemetry("layer_masks.json", {
        "lca_detected": lca_detected,
        "current_close": float(current_close),
        "prev_high": float(prev_high),
        "lca_margin": float(current_close - prev_high) if lca_detected else 0.0
    })

    return lca_detected


def calculate_intent_nudge(df: pd.DataFrame, volume_threshold: float = 1.2) -> Dict[str, Any]:
    """
    Calculate intent nudge based on volume confirmation and CVD.

    Enhanced with CVD (Cumulative Volume Delta) to catch trap reversals
    and true market intent, especially in inverted/choppy flows.

    Args:
        df: DataFrame with OHLC and volume data
        volume_threshold: Volume multiplier threshold for confirmation

    Returns:
        Dict containing intent analysis with CVD-based conviction
    """
    if len(df) < 10:
        return {"nudge_score": 0.0, "conviction": "low"}

    current_volume = df["volume"].iloc[-1]
    # Use the most recent available rolling average (excluding current bar if possible)
    rolling_avg = df["volume"].rolling(window=10).mean()
    if len(df) > 10:
        avg_volume = rolling_avg.iloc[-2]  # Previous bar's average
    else:
        # For limited data, use average of all but current bar
        avg_volume = df["volume"].iloc[:-1].mean() if len(df) > 1 else 0

    if pd.isna(avg_volume) or avg_volume == 0:
        # Handle zero or NaN average volume edge case
        return {
            "nudge_score": 0.0,
            "volume_ratio": 0.0,
            "conviction": "low",
            "current_volume": float(current_volume),
            "avg_volume": 0.0,
            "cvd_delta": 0.0,
            "liquidity_pump": False
        }

    volume_ratio = current_volume / avg_volume

    # Calculate CVD for true intent detection
    cvd_delta = calculate_cvd(df)

    # Check for liquidity pump (trap reversal - Moneytaur logic)
    liquidity_pump = False
    if len(df) >= 3:
        # Volume spike with price reversal indicates pump
        recent_vol_spike = df["volume"].iloc[-3:].max() > avg_volume * 1.5
        price_reversal = (df["close"].iloc[-1] - df["close"].iloc[-3]) * \
                        (df["close"].iloc[-2] - df["close"].iloc[-3]) < 0
        liquidity_pump = recent_vol_spike and price_reversal

    # Enhanced conviction determination with CVD and trap detection
    if cvd_delta < 0:  # Bear flow
        if liquidity_pump and volume_ratio > 1.0:
            # Trap reversal detected - bullish intent despite bear CVD
            conviction = "high"  # Upgraded for trap
            nudge_score = min(1.0, volume_ratio / volume_threshold) * 1.2  # Boost for trap
        else:
            # True bear intent
            conviction = "low" if volume_ratio < volume_threshold else "medium"
            nudge_score = min(1.0, volume_ratio / volume_threshold) * 0.8  # Reduce for bear
    else:  # Bull flow or neutral
        if volume_ratio >= volume_threshold * 2:
            conviction = "very_high"
            nudge_score = 1.0
        elif volume_ratio >= volume_threshold * 1.5:
            conviction = "high"
            nudge_score = min(1.0, volume_ratio / volume_threshold)
        elif volume_ratio >= volume_threshold:
            conviction = "medium"
            nudge_score = min(1.0, volume_ratio / volume_threshold)
        else:
            conviction = "low"
            nudge_score = min(1.0, volume_ratio / volume_threshold)

    result = {
        "nudge_score": float(min(1.0, nudge_score)),
        "volume_ratio": float(volume_ratio),
        "conviction": conviction,
        "current_volume": float(current_volume),
        "avg_volume": float(avg_volume),
        "cvd_delta": float(cvd_delta),
        "liquidity_pump": bool(liquidity_pump)
    }

    log_telemetry("layer_masks.json", {
        "intent_nudge": result["nudge_score"],
        "volume_conviction": result["conviction"],
        "cvd_delta": result["cvd_delta"],
        "trap_detected": result["liquidity_pump"]
    })

    return result


def orderflow_lca(df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """
    Main Orderflow LCA function combining BOS detection and intent analysis.

    Enhanced v1.6.1 with CVD (Cumulative Volume Delta) analysis for true market intent.

    Combines:
    1. Liquidity Capture Analysis (LCA) - price above previous highs
    2. Break of Structure (BOS) detection
    3. Intent nudging via volume confirmation
    4. CVD slope analysis for divergence detection

    Args:
        df: Price/volume DataFrame
        config: Configuration parameters

    Returns:
        float: Combined orderflow score (0.0 - 1.0)
    """
    if not config.get("features", {}).get("orderflow_lca", False):
        return 0.5  # Neutral score when disabled

    # Detect LCA pattern
    lca_detected = detect_liquidity_capture(df)

    # Detect BOS
    bos_result = detect_bos(df)

    # Calculate intent nudge
    intent_result = calculate_intent_nudge(df)

    # Calculate CVD for true volume intent
    cvd_result = calculate_cvd(df)

    # Combine signals
    base_score = 0.4  # Baseline
    lca_bonus = 0.25 if lca_detected else 0.0
    bos_bonus = 0.2 * bos_result["strength"] if bos_result["detected"] else 0.0
    intent_bonus = 0.1 * intent_result["nudge_score"]

    # CVD enhancement - IamZeroIka's divergence insights
    cvd_bonus = 0.0
    if abs(cvd_result["slope"]) > 100:  # Significant slope change
        if cvd_result["delta"] > 0 and cvd_result["slope"] > 0:
            cvd_bonus = 0.15  # Strong bullish confluence
        elif cvd_result["delta"] < 0 and cvd_result["slope"] > 0:
            cvd_bonus = 0.20  # Hidden bullish divergence (bears exhausting)
        elif cvd_result["delta"] > 0 and cvd_result["slope"] < 0:
            cvd_bonus = -0.10  # Hidden bearish divergence (bulls weakening)

    final_score = min(1.0, max(0.0, base_score + lca_bonus + bos_bonus + intent_bonus + cvd_bonus))

    # Enhanced scoring for strong confluence
    confluence_count = sum([
        lca_detected,
        bos_result["detected"],
        intent_result["conviction"] in ["high", "very_high"],
        abs(cvd_result["slope"]) > 100
    ])

    if confluence_count >= 3:
        final_score = min(1.0, final_score + 0.1)  # Strong confluence bonus

    log_telemetry("layer_masks.json", {
        "ofl_lca": float(final_score),
        "ofl_bos": bos_result["detected"],
        "ofl_cvd_delta": float(cvd_result["delta"]),
        "ofl_cvd_slope": float(cvd_result["slope"]),
        "ofl_components": {
            "base": float(base_score),
            "lca_bonus": float(lca_bonus),
            "bos_bonus": float(bos_bonus),
            "intent_bonus": float(intent_bonus),
            "cvd_bonus": float(cvd_bonus),
            "confluence_count": confluence_count
        }
    })

    return final_score


def analyze_market_structure(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive market structure analysis for orderflow.

    Enhanced v1.6.1 with CVD analysis and slope divergence detection.

    Args:
        df: Price/volume DataFrame
        config: Configuration parameters

    Returns:
        Dict containing comprehensive market structure analysis
    """
    lca_score = orderflow_lca(df, config)
    bos_analysis = detect_bos(df)
    intent_analysis = calculate_intent_nudge(df)
    cvd_analysis = calculate_cvd(df)

    return {
        "lca_score": lca_score,
        "bos_analysis": bos_analysis,
        "intent_analysis": intent_analysis,
        "cvd_analysis": cvd_analysis,
        "structure_health": "strong" if lca_score > 0.7 else "weak",
        "orderflow_divergence": {
            "detected": abs(cvd_analysis["slope"]) > 100,
            "type": "bullish" if cvd_analysis["slope"] > 0 else "bearish",
            "strength": min(1.0, abs(cvd_analysis["slope"]) / 500)  # Normalize slope strength
        }
    }