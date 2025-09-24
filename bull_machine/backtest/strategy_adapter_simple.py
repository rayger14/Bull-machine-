#!/usr/bin/env python3
"""
Simplified v1.4 Strategy Adapter
Works with available v1.3 components without missing dependencies
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bull_machine.core.sync import decide_mtf_entry
from bull_machine.core.types import BiasCtx


@dataclass
class BacktestSignal:
    """Backtest-compatible signal format"""

    action: str  # 'long'|'short'|'exit'|'flat'
    size: float
    confidence: float
    stop: Optional[float] = None
    tp: Optional[list[float]] = None
    risk_pct: float = 0.01
    reasons: list[str] = None


def analyze_market_bias(df: pd.DataFrame, tf: str) -> BiasCtx:
    """Analyze market bias from DataFrame using simple moving averages"""

    if len(df) < 50:
        return BiasCtx(
            tf=tf,
            bias="neutral",
            confirmed=False,
            strength=0.0,
            bars_confirmed=0,
            ma_distance=0.0,
            trend_quality=0.0,
            ma_slope=0.0,
        )

    # Calculate moving averages
    ma20 = df["close"].rolling(20, min_periods=1).mean()
    ma50 = df["close"].rolling(50, min_periods=1).mean()

    current_close = df["close"].iloc[-1]
    ma20_current = ma20.iloc[-1]
    ma50_current = ma50.iloc[-1]

    # Determine bias
    if ma20_current > ma50_current and current_close > ma20_current:
        bias = "long"
        strength = 0.75
    elif ma20_current < ma50_current and current_close < ma20_current:
        bias = "short"
        strength = 0.75
    else:
        bias = "neutral"
        strength = 0.35

    # Check for 2-bar confirmation
    confirmed = False
    bars_confirmed = 0
    if len(df) >= 2:
        prev_close = df["close"].iloc[-2]
        prev_ma20 = ma20.iloc[-2]

        if bias == "long" and prev_close > prev_ma20:
            confirmed = True
            bars_confirmed = 2
        elif bias == "short" and prev_close < prev_ma20:
            confirmed = True
            bars_confirmed = 2

    return BiasCtx(
        tf=tf,
        bias=bias,
        confirmed=confirmed,
        strength=strength,
        bars_confirmed=bars_confirmed,
        ma_distance=abs(current_close - ma20_current) / ma20_current if ma20_current > 0 else 0,
        trend_quality=0.6,
        ma_slope=0.01 if bias == "long" else -0.01 if bias == "short" else 0.0,
    )


def resample_to_mtf(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample DataFrame to target timeframe"""

    # Simple resampling rules
    if target_tf == "1D":
        freq = "1D"
    elif target_tf == "4H":
        freq = "4h"
    elif target_tf == "1H":
        freq = "1h"
    else:
        return df  # Return as-is if unknown timeframe

    # Try to resample (assumes proper datetime index)
    try:
        # If index is not datetime, create one
        if not isinstance(df.index, pd.DatetimeIndex):
            # Create synthetic datetime index
            df = df.copy()
            df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="1h")

        resampled = (
            df.resample(freq)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

        return resampled

    except Exception:
        # If resampling fails, return original data
        return df


def generate_simple_signal(
    symbol: str, tf: str, df_window: pd.DataFrame, balance: float = 10000
) -> Dict[str, Any]:
    """
    Generate trading signal using simplified MTF analysis

    This is a lightweight implementation that demonstrates the v1.4 framework
    without requiring the full v1.3 pipeline dependencies
    """

    if df_window is None or df_window.empty or len(df_window) < 100:
        return {"action": "flat", "reason": "insufficient_data"}

    try:
        # Analyze multiple timeframes
        htf_data = resample_to_mtf(df_window, "1D")
        mtf_data = resample_to_mtf(df_window, "4H")
        ltf_data = df_window  # Primary timeframe

        # Get bias for each timeframe
        htf_bias = analyze_market_bias(htf_data, "1D")
        mtf_bias = analyze_market_bias(mtf_data, "4H")
        ltf_bias = analyze_market_bias(ltf_data, tf)

        # MTF sync decision
        policy = {
            "desync_behavior": "raise",
            "desync_bump": 0.10,
            "eq_magnet_gate": True,
            "eq_bump": 0.05,
            "nested_bump": 0.03,
            "alignment_discount": 0.05,
        }

        # Simplified confluence checks
        nested_ok = True  # Assume good confluence for now
        eq_magnet = False  # Assume not in equilibrium

        sync_result = decide_mtf_entry(
            htf_bias, mtf_bias, ltf_bias.bias, nested_ok, eq_magnet, policy
        )

        # Generate signal based on MTF decision
        if sync_result.decision == "veto":
            return {"action": "flat", "reason": "mtf_veto"}

        # Check if we have a clear directional bias
        if htf_bias.bias == "neutral" and mtf_bias.bias == "neutral":
            return {"action": "flat", "reason": "no_clear_trend"}

        # Use HTF bias for direction (HTF dominance)
        direction = htf_bias.bias if htf_bias.bias != "neutral" else mtf_bias.bias

        if direction == "neutral":
            return {"action": "flat", "reason": "neutral_bias"}

        # Calculate position sizing
        current_price = df_window["close"].iloc[-1]

        # Simple stop loss (2% from entry)
        if direction == "long":
            stop_price = current_price * 0.98
        else:
            stop_price = current_price * 1.02

        risk_per_unit = abs(current_price - stop_price)
        risk_amount = balance * 0.01  # 1% risk per trade
        position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 1.0

        # Build TP ladder
        if direction == "long":
            tp_ladder = [
                current_price + risk_per_unit * 1.0,  # TP1: 1R
                current_price + risk_per_unit * 2.0,  # TP2: 2R
                current_price + risk_per_unit * 3.0,  # TP3: 3R
            ]
        else:
            tp_ladder = [
                current_price - risk_per_unit * 1.0,  # TP1: 1R
                current_price - risk_per_unit * 2.0,  # TP2: 2R
                current_price - risk_per_unit * 3.0,  # TP3: 3R
            ]

        # Calculate confidence based on alignment
        base_confidence = 0.60
        if sync_result.decision == "allow" and sync_result.threshold_bump < 0:
            confidence = base_confidence + 0.15  # Alignment bonus
        elif sync_result.decision == "allow":
            confidence = base_confidence
        else:  # raise
            confidence = base_confidence - 0.10

        return {
            "action": direction,
            "size": position_size,
            "confidence": confidence,
            "stop": stop_price,
            "tp": tp_ladder,
            "risk_pct": 0.01,
            "reasons": ["mtf_sync", f"htf_{htf_bias.bias}", f"mtf_{mtf_bias.bias}"],
            "mtf_decision": sync_result.decision,
            "alignment_score": sync_result.alignment_score,
        }

    except Exception as e:
        print(f"Error in simple signal generation: {e}")
        return {"action": "flat", "error": str(e)}


# Main strategy function for backtest framework
def strategy_from_df(
    symbol: str, tf: str, df_window: pd.DataFrame, balance: float = 10000, config_path: str = None
) -> Dict[str, Any]:
    """
    Main strategy adapter for v1.4 backtest framework

    This version uses the simplified signal generation that works
    with available v1.3 components
    """
    return generate_simple_signal(symbol, tf, df_window, balance)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")

    price = 50000
    data = []

    for date in dates:
        price += np.random.normal(0, 100)  # Random walk
        open_price = price + np.random.normal(0, 50)
        high_price = max(open_price, price) + abs(np.random.normal(0, 200))
        low_price = min(open_price, price) - abs(np.random.normal(0, 200))
        close_price = price

        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": np.random.uniform(100, 1000),
            }
        )

    df = pd.DataFrame(data, index=dates)

    # Test the strategy
    signal = strategy_from_df("BTCUSD", "1H", df, balance=100000)
    print(f"Test signal: {signal}")
