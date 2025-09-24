"""
MTF Sync Hardening for Wyckoff Layer
Phase 1.4 Rules: Multi-timeframe alignment with liquidity gating
"""

import logging
from typing import Dict, Tuple

import pandas as pd


def wyckoff_state(df: pd.DataFrame) -> Dict[str, any]:
    """
    Determine Wyckoff state from OHLCV data.
    Returns bias (long/short/neutral) and confidence score.
    """
    if len(df) < 20:
        return {"bias": "neutral", "confidence": 0.0, "phase": "unknown"}

    # Simplified Wyckoff analysis
    recent = df.tail(20)

    # Check for accumulation patterns
    range_high = recent["high"].max()
    range_low = recent["low"].min()
    current_position = (df.iloc[-1]["close"] - range_low) / (range_high - range_low) if range_high > range_low else 0.5

    # Volume analysis
    vol_sma = recent["volume"].mean()
    recent_vol = df.tail(5)["volume"].mean()
    vol_expansion = recent_vol > vol_sma * 1.2

    # Structure analysis
    higher_lows = df.tail(10)["low"].is_monotonic_increasing
    lower_highs = df.tail(10)["high"].is_monotonic_decreasing

    # Determine bias and confidence
    if current_position < 0.35 and vol_expansion:
        # Potential accumulation
        bias = "long"
        confidence = 0.65 + (0.15 if higher_lows else 0)
        phase = "accumulation_C" if current_position < 0.25 else "accumulation_B"
    elif current_position > 0.65 and vol_expansion:
        # Potential distribution
        bias = "short"
        confidence = 0.65 + (0.15 if lower_highs else 0)
        phase = "distribution_C" if current_position > 0.75 else "distribution_B"
    else:
        # Neutral/ranging
        bias = "neutral"
        confidence = 0.5
        phase = "ranging"

    return {
        "bias": bias,
        "confidence": min(confidence, 0.85),
        "phase": phase,
        "current_position": current_position,
        "volume_expansion": vol_expansion,
    }


def mtf_alignment(daily_df: pd.DataFrame, h4_df: pd.DataFrame, liquidity_score: float) -> Tuple[bool, Dict]:
    """
    Check multi-timeframe alignment between daily and 4H.
    Liquidity score acts as a gate for desync tolerance.

    Args:
        daily_df: Daily timeframe OHLCV
        h4_df: 4-hour timeframe OHLCV
        liquidity_score: Current liquidity layer score (0-1)

    Returns:
        (aligned, details) - aligned is True if timeframes agree or liquidity overrides
    """

    # Get Wyckoff states for each timeframe
    daily_state = wyckoff_state(daily_df)
    h4_state = wyckoff_state(h4_df)

    # Check basic alignment
    biases_match = daily_state["bias"] == h4_state["bias"]

    # High liquidity can override minor desyncs - softened from 0.75 to 0.70
    liquidity_override = liquidity_score >= 0.70

    # Calculate alignment score
    if biases_match:
        # Perfect alignment
        alignment_score = (daily_state["confidence"] + h4_state["confidence"]) / 2
        aligned = True
    elif liquidity_override and daily_state["bias"] != "neutral":
        # Liquidity override allows minor desync
        alignment_score = max(daily_state["confidence"], h4_state["confidence"]) * 0.8
        aligned = True
    else:
        # Desynced without override
        alignment_score = 0.3
        aligned = False

    # Additional quality checks
    quality_checks = {
        "htf_confidence": daily_state["confidence"] >= 0.70,
        "mtf_confidence": h4_state["confidence"] >= 0.70,
        "liquidity_support": liquidity_score >= 0.60,
    }

    # Final alignment requires quality thresholds
    if aligned:
        aligned = quality_checks["htf_confidence"] and quality_checks["mtf_confidence"]

    details = {
        "daily_state": daily_state,
        "h4_state": h4_state,
        "biases_match": biases_match,
        "liquidity_score": liquidity_score,
        "liquidity_override": liquidity_override,
        "alignment_score": alignment_score,
        "quality_checks": quality_checks,
        "final_aligned": aligned,
    }

    logging.info(
        f"MTF Alignment: aligned={aligned}, score={alignment_score:.2f}, "
        f"daily={daily_state['bias']}/{daily_state['confidence']:.2f}, "
        f"h4={h4_state['bias']}/{h4_state['confidence']:.2f}, "
        f"liq={liquidity_score:.2f}"
    )

    return aligned, details


def get_mtf_context(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> Dict:
    """
    Build complete MTF context for exit evaluation.
    """

    # Get key levels from each timeframe
    context = {
        "1h": {
            "recent_high": df_1h.tail(20)["high"].max() if len(df_1h) >= 20 else None,
            "recent_low": df_1h.tail(20)["low"].min() if len(df_1h) >= 20 else None,
            "current_close": df_1h.iloc[-1]["close"] if len(df_1h) > 0 else None,
        },
        "4h": {
            "recent_high": df_4h.tail(10)["high"].max() if len(df_4h) >= 10 else None,
            "recent_low": df_4h.tail(10)["low"].min() if len(df_4h) >= 10 else None,
            "current_close": df_4h.iloc[-1]["close"] if len(df_4h) > 0 else None,
            "close_4h": df_4h.iloc[-1]["close"] if len(df_4h) > 0 else None,  # For exit rules
        },
        "1d": {
            "recent_high": df_1d.tail(5)["high"].max() if len(df_1d) >= 5 else None,
            "recent_low": df_1d.tail(5)["low"].min() if len(df_1d) >= 5 else None,
            "current_close": df_1d.iloc[-1]["close"] if len(df_1d) > 0 else None,
        },
    }

    # Determine HTF bias
    if context["1d"]["current_close"] and context["1d"]["recent_low"]:
        d_range = context["1d"]["recent_high"] - context["1d"]["recent_low"]
        d_position = (context["1d"]["current_close"] - context["1d"]["recent_low"]) / d_range if d_range > 0 else 0.5

        if d_position > 0.65:
            context["htf_bias"] = "bullish"
            context["htf_resistance_near"] = True
            context["htf_support_near"] = False
        elif d_position < 0.35:
            context["htf_bias"] = "bearish"
            context["htf_resistance_near"] = False
            context["htf_support_near"] = True
        else:
            context["htf_bias"] = "neutral"
            context["htf_resistance_near"] = False
            context["htf_support_near"] = False
    else:
        context["htf_bias"] = "neutral"
        context["htf_resistance_near"] = False
        context["htf_support_near"] = False

    # Add HTF close for exit rules
    context["htf"] = {"close_4h": context["4h"]["close_4h"]}

    return context
