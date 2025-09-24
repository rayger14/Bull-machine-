"""
Bojan Wick Magnets
Phase 2.1: Present but soft-gated for v1.4.1
"""

import logging
from typing import Dict, Optional

import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(df) < period + 1:
        return df["high"].tail(period).mean() - df["low"].tail(period).mean()

    high = df["high"].tail(period + 1)
    low = df["low"].tail(period + 1)
    close = df["close"].tail(period + 1)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

    return tr.tail(period).mean()


def wick_magnets(df: pd.DataFrame, ob_level: float, ttl_bars: int = 12) -> float:
    """
    Calculate Bojan wick magnet score based on wick size and OB proximity.

    Args:
        df: OHLCV DataFrame
        ob_level: Order block level to check proximity
        ttl_bars: Time-to-live for score decay

    Returns:
        Score 0-1 representing wick magnet strength (capped for v1.4.1)
    """

    if len(df) < 2:
        return 0.0

    last = df.iloc[-1]

    # Calculate ATR for normalization
    atr = calculate_atr(df)
    if atr <= 0:
        return 0.0

    # Calculate wick sizes (normalized by ATR)
    wick_up = last["high"] - max(last["open"], last["close"])
    wick_down = min(last["open"], last["close"]) - last["low"]
    max_wick = max(wick_up, wick_down)
    wick_ratio = max_wick / atr

    # Calculate proximity to order block (in ATR units)
    proximity = abs(last["close"] - ob_level) / atr

    # Find last time this level was hit
    try:
        # Check when price last touched the OB level
        touches = df[((df["high"] >= ob_level * 0.999) & (df["low"] <= ob_level * 1.001))]
        if len(touches) > 0:
            last_hit_idx = touches.index[-1]
            last_hit_pos = df.index.get_loc(last_hit_idx)
            current_pos = len(df) - 1
            bars_since = current_pos - last_hit_pos
        else:
            bars_since = 10**6  # Never touched
    except (ValueError, KeyError):
        bars_since = 10**6

    # Calculate base score
    # Weight: 60% wick size, 40% proximity
    wick_component = 0.6 * min(wick_ratio, 2.0) / 2.0  # Cap at 2 ATR
    proximity_component = 0.4 * (1 / (1 + proximity))  # Inverse distance

    # Minimum wick threshold (70% of ATR)
    if wick_ratio < 0.70:
        return 0.0

    base_score = wick_component + proximity_component

    # Apply TTL decay
    if bars_since <= ttl_bars:
        decay_factor = 1.0
    else:
        decay_factor = 0.9 ** max(0, bars_since - ttl_bars)

    final_score = base_score * decay_factor

    # **IMPORTANT: Cap contribution for v1.4.1 (soft gate)**
    # This limits Bojan influence until Phase 2.x
    capped_score = min(0.6, final_score)

    logging.debug(
        f"Bojan wick magnet: wick_ratio={wick_ratio:.2f}, "
        f"proximity={proximity:.2f}, bars_since={bars_since}, "
        f"score={final_score:.3f} -> capped={capped_score:.3f}"
    )

    return capped_score


def detect_bojan_patterns(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect Bojan high/low patterns for exit evaluation.
    Phase-gated: Returns patterns but with reduced confidence for v1.4.1.
    """

    if len(df) < lookback:
        return {
            "bojan_high": None,
            "bojan_low": None,
            "enabled": False,  # Phase-gated
        }

    recent = df.tail(lookback)
    atr = calculate_atr(df)

    patterns = {
        "bojan_high": None,
        "bojan_low": None,
        "enabled": False,  # Will be True in v2.x
    }

    # Look for Bojan Highs (large upper wicks with rejection)
    for i in range(-5, 0):
        bar = recent.iloc[i]

        # Upper wick calculation
        wick_up = bar["high"] - max(bar["open"], bar["close"])
        body = abs(bar["close"] - bar["open"])

        if atr > 0 and body > 0:
            wick_ratio = wick_up / atr
            wick_body_ratio = wick_up / body

            # Bojan High criteria
            if wick_ratio > 1.5 and wick_body_ratio > 2 and bar["close"] < bar["open"]:
                patterns["bojan_high"] = {
                    "bar_idx": i,
                    "level": bar["high"],
                    "wick_ratio": wick_ratio,
                    "confidence": min(0.6, wick_ratio / 3.0),  # Capped for v1.4.1
                }
                break

    # Look for Bojan Lows (large lower wicks with rejection)
    for i in range(-5, 0):
        bar = recent.iloc[i]

        # Lower wick calculation
        wick_down = min(bar["open"], bar["close"]) - bar["low"]
        body = abs(bar["close"] - bar["open"])

        if atr > 0 and body > 0:
            wick_ratio = wick_down / atr
            wick_body_ratio = wick_down / body

            # Bojan Low criteria
            if wick_ratio > 1.5 and wick_body_ratio > 2 and bar["close"] > bar["open"]:
                patterns["bojan_low"] = {
                    "bar_idx": i,
                    "level": bar["low"],
                    "wick_ratio": wick_ratio,
                    "confidence": min(0.6, wick_ratio / 3.0),  # Capped for v1.4.1
                }
                break

    return patterns


def bojan_exit_signal(df: pd.DataFrame, position_bias: str, htf_context: Dict = None) -> Optional[Dict]:
    """
    Generate Bojan-based exit signals.
    PHASE-GATED: Returns None unless explicitly enabled in config.
    """

    # Phase gate - return None for v1.4.1
    if not htf_context or not htf_context.get("bojan_exits_enabled", False):
        return None

    patterns = detect_bojan_patterns(df)

    if not patterns["enabled"]:
        return None

    # Long position + Bojan High = exit signal
    if position_bias == "long" and patterns["bojan_high"]:
        # Check HTF alignment if available
        if htf_context and htf_context.get("htf_resistance_near"):
            return {
                "action": "partial",
                "size_pct": 0.75,
                "reason": "Bojan High rejection at resistance",
                "confidence": patterns["bojan_high"]["confidence"],
            }

    # Short position + Bojan Low = exit signal
    elif position_bias == "short" and patterns["bojan_low"]:
        # Check HTF alignment if available
        if htf_context and htf_context.get("htf_support_near"):
            return {
                "action": "partial",
                "size_pct": 0.75,
                "reason": "Bojan Low rejection at support",
                "confidence": patterns["bojan_low"]["confidence"],
            }

    return None
