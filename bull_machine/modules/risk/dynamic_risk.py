"""
Dynamic Risk Management
Scale base risk by sweep volume vs median & pool depth
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd


def wyckoff_state(df: pd.DataFrame) -> Dict:
    """Get Wyckoff phase for stop calculation."""
    if len(df) < 20:
        return {"phase": "unknown"}

    # Simple phase detection based on range position
    recent = df.tail(20)
    high_20 = recent["high"].max()
    low_20 = recent["low"].min()
    current_close = df.iloc[-1]["close"]

    if high_20 > low_20:
        range_pos = (current_close - low_20) / (high_20 - low_20)
        if range_pos > 0.7:
            return {"phase": "D"}  # Markup phase
        elif range_pos < 0.3:
            return {"phase": "E"}  # Markdown phase
        else:
            return {"phase": "C"}  # Consolidation

    return {"phase": "unknown"}


def calculate_stop_loss(df: pd.DataFrame, bias: str, entry_price: float, pool_depth_score: float, atr: float) -> float:
    """
    Calculate phase-aware volatility-adjusted stop loss distance.
    Wider stops in markup/markdown phases to let winners run.

    Args:
        df: OHLCV data
        bias: 'long' or 'short'
        entry_price: Entry price level
        pool_depth_score: Liquidity pool strength (0-1)
        atr: Current ATR value

    Returns:
        Stop loss price level
    """
    # Get Wyckoff phase
    phase_info = wyckoff_state(df)
    phase = phase_info["phase"]

    # Phase-based stop multiplier - wider in trending phases
    if phase in ("D", "E"):  # Markup/Markdown phases - let winners run
        base_multiplier = 2.0
    else:  # Consolidation phases - tighter stops
        base_multiplier = 1.5

    base_distance = base_multiplier * atr

    # Pool depth adjustment - stronger pools allow slightly wider stops for better R:R
    depth_multiplier = max(1.0, min(1.8, pool_depth_score * 1.8))

    # Volatility spike adjustment
    if len(df) >= 20:
        recent_atr = df["atr"].rolling(20).mean().iloc[-1] if "atr" in df.columns else atr
        vol_spike_ratio = atr / recent_atr if recent_atr > 0 else 1.0
        vol_adjustment = min(1.5, vol_spike_ratio)  # Cap at 1.5x for extreme spikes
    else:
        vol_adjustment = 1.0

    # Calculate final stop distance
    stop_distance = base_distance * depth_multiplier * vol_adjustment

    # Apply direction-specific stop
    if bias == "long":
        stop_price = entry_price - stop_distance
    else:
        stop_price = entry_price + stop_distance

    # Log for telemetry
    logging.info(
        f"Stop calculation: phase={phase}, base_mult={base_multiplier:.1f}, "
        f"depth_mult={depth_multiplier:.2f}, vol_adj={vol_adjustment:.2f}, "
        f"final_dist={stop_distance:.2f}"
    )

    return stop_price


def calculate_dynamic_position_size(
    base_risk_pct: float, df: pd.DataFrame, liquidity_data: Dict, volatility_context: Dict = None
) -> Dict:
    """
    Scale base risk by sweep volume vs median & pool depth.

    Args:
        base_risk_pct: Base risk percentage (e.g., 0.01 = 1%)
        df: OHLCV data
        liquidity_data: Output from calculate_liquidity_score()
        volatility_context: Optional volatility metrics

    Returns:
        Dict with adjusted position size and risk metrics
    """

    # Default values
    adjusted_risk = base_risk_pct
    risk_multiplier = 1.0
    risk_factors = {}

    if len(df) < 10:
        return {
            "adjusted_risk_pct": adjusted_risk,
            "risk_multiplier": risk_multiplier,
            "position_size_multiplier": 1.0,
            "risk_factors": {"insufficient_data": True},
        }

    # Volume analysis
    recent_volume = df.tail(10)["volume"]
    median_volume = df["volume"].median()
    current_volume = df.iloc[-1]["volume"]

    volume_ratio = current_volume / median_volume if median_volume > 0 else 1.0
    risk_factors["volume_ratio"] = volume_ratio

    # Volume-based adjustment
    if volume_ratio > 2.0:
        # Very high volume - increase risk (high conviction)
        volume_multiplier = min(1.3, 1.0 + (volume_ratio - 2.0) * 0.1)
    elif volume_ratio < 0.5:
        # Low volume - reduce risk (low conviction)
        volume_multiplier = max(0.7, 1.0 - (1.0 - volume_ratio) * 0.3)
    else:
        volume_multiplier = 1.0

    risk_factors["volume_multiplier"] = volume_multiplier

    # Liquidity pool depth adjustment
    pools = liquidity_data.get("pools", [])
    pool_depth_score = 0.0

    if pools:
        # Calculate average pool strength
        pool_strengths = [p.get("strength", 0) for p in pools]
        avg_pool_strength = np.mean(pool_strengths)

        # More/stronger pools = higher conviction = more risk
        pool_count_factor = min(len(pools) / 5.0, 1.0)  # Normalize to 0-1
        pool_depth_score = (avg_pool_strength + pool_count_factor) / 2

        # Pool depth multiplier
        if pool_depth_score > 0.7:
            pool_multiplier = 1.2  # Strong pools = increase risk
        elif pool_depth_score > 0.4:
            pool_multiplier = 1.0  # Moderate pools = neutral
        else:
            pool_multiplier = 0.8  # Weak pools = reduce risk
    else:
        pool_multiplier = 0.9  # No pools = slightly reduce risk

    risk_factors["pool_depth_score"] = pool_depth_score
    risk_factors["pool_multiplier"] = pool_multiplier

    # Recent sweep quality adjustment
    recent_sweep = liquidity_data.get("recent_sweep")
    if recent_sweep and recent_sweep.get("strength", 0) > 0.7:
        # High quality recent sweep = increase conviction
        sweep_multiplier = 1.15
    else:
        sweep_multiplier = 1.0

    risk_factors["sweep_multiplier"] = sweep_multiplier

    # Clustering bonus
    cluster_score = liquidity_data.get("cluster_score", 0)
    if cluster_score > 0.1:
        # Strong clustering = sweep magnet = increase risk
        cluster_multiplier = 1.0 + cluster_score
    else:
        cluster_multiplier = 1.0

    risk_factors["cluster_multiplier"] = cluster_multiplier

    # Volatility adjustment (if provided)
    volatility_multiplier = 1.0
    if volatility_context:
        vol_percentile = volatility_context.get("percentile", 50)
        if vol_percentile > 80:  # High volatility
            volatility_multiplier = 0.8  # Reduce risk
        elif vol_percentile < 20:  # Low volatility
            volatility_multiplier = 1.1  # Slightly increase risk

    risk_factors["volatility_multiplier"] = volatility_multiplier

    # Calculate final multiplier
    risk_multiplier = (
        volume_multiplier * pool_multiplier * sweep_multiplier * cluster_multiplier * volatility_multiplier
    )

    # Cap the multiplier to prevent extreme positions
    risk_multiplier = max(0.5, min(2.0, risk_multiplier))

    # Calculate adjusted risk
    adjusted_risk = base_risk_pct * risk_multiplier

    # Position size multiplier (inverse relationship with adjusted risk)
    # Higher risk = larger position for same $ risk
    position_size_multiplier = risk_multiplier

    logging.info(
        f"Dynamic risk: base={base_risk_pct:.3f}, multiplier={risk_multiplier:.2f}, adjusted={adjusted_risk:.3f}"
    )

    return {
        "adjusted_risk_pct": adjusted_risk,
        "risk_multiplier": risk_multiplier,
        "position_size_multiplier": position_size_multiplier,
        "risk_factors": risk_factors,
        "volume_context": {
            "current_vs_median": volume_ratio,
            "conviction_level": "high" if volume_ratio > 1.5 else "normal" if volume_ratio > 0.8 else "low",
        },
        "liquidity_context": {
            "pool_count": len(pools),
            "avg_pool_strength": pool_depth_score,
            "cluster_strength": cluster_score,
        },
    }


def calculate_volatility_context(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Calculate volatility context for risk adjustment.
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)

    # Calculate returns
    returns = recent["close"].pct_change().dropna()

    if len(returns) < 2:
        return {"current_volatility": 0.02, "percentile": 50, "regime": "normal"}

    # Current volatility (rolling 20-day)
    rolling_vol = returns.rolling(20, min_periods=10).std()
    current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else returns.std()

    # Historical volatility percentile
    historical_vols = rolling_vol.dropna()
    if len(historical_vols) > 5:
        percentile = (historical_vols <= current_vol).mean() * 100
    else:
        percentile = 50

    # Volatility regime
    if percentile > 80:
        regime = "high_vol"
    elif percentile < 20:
        regime = "low_vol"
    else:
        regime = "normal"

    return {
        "current_volatility": current_vol,
        "percentile": percentile,
        "regime": regime,
        "lookback_period": lookback,
    }


def validate_position_size(
    position_size_pct: float, max_position_pct: float = 0.05, account_balance: float = 10000
) -> Dict:
    """
    Validate and constrain position size within risk limits.
    """
    # Ensure position doesn't exceed maximum
    capped_size = min(position_size_pct, max_position_pct)

    # Calculate dollar amounts
    position_value = account_balance * capped_size
    max_loss = position_value  # Assuming 100% loss scenario

    # Risk-reward validation
    risk_too_high = capped_size > max_position_pct
    position_too_small = capped_size < 0.001  # 0.1% minimum

    return {
        "final_position_pct": capped_size,
        "position_value_usd": position_value,
        "max_potential_loss": max_loss,
        "risk_warnings": {
            "exceeds_max_risk": risk_too_high,
            "below_minimum": position_too_small,
            "within_limits": not risk_too_high and not position_too_small,
        },
        "risk_level": "high" if capped_size > 0.03 else "medium" if capped_size > 0.015 else "low",
    }
