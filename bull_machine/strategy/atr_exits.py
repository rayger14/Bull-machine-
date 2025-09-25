"""
Bull Machine ATR-Based Exit Logic
Dynamic stops and targets based on market volatility
"""

from typing import Dict, Tuple
import pandas as pd
from bull_machine.core.numerics import safe_atr

def compute_exit_levels(df: pd.DataFrame, side: str, risk_cfg: Dict) -> Tuple[float, float]:
    """
    Calculate stop loss and take profit levels based on ATR.

    Args:
        df: Price DataFrame with OHLC data
        side: "long" or "short"
        risk_cfg: Risk configuration dict

    Returns:
        Tuple of (stop_loss, take_profit) prices
    """
    atr_window = risk_cfg.get("atr_window", 14)
    atr = float(safe_atr(df, atr_window).iloc[-1])

    sl_multiple = float(risk_cfg.get("sl_atr", 2.0))
    tp_multiple = float(risk_cfg.get("tp_atr", 3.0))
    current_price = float(df['close'].iloc[-1])

    if side == "long":
        stop_loss = current_price - sl_multiple * atr
        take_profit = current_price + tp_multiple * atr
    else:  # short
        stop_loss = current_price + sl_multiple * atr
        take_profit = current_price - tp_multiple * atr

    return stop_loss, take_profit

def maybe_trail_sl(df: pd.DataFrame, side: str, current_sl: float, risk_cfg: Dict) -> float:
    """
    Update trailing stop loss based on ATR.

    Args:
        df: Price DataFrame with OHLC data
        side: "long" or "short"
        current_sl: Current stop loss level
        risk_cfg: Risk configuration dict

    Returns:
        Updated stop loss level
    """
    trail_multiple = risk_cfg.get("trail_atr", 0.0)
    if not trail_multiple:
        return current_sl

    atr_window = risk_cfg.get("atr_window", 14)
    atr = float(safe_atr(df, atr_window).iloc[-1])
    current_price = float(df['close'].iloc[-1])
    trail_distance = trail_multiple * atr

    if side == "long":
        new_sl = current_price - trail_distance
        return max(current_sl, new_sl)  # Only move stop up
    else:  # short
        new_sl = current_price + trail_distance
        return min(current_sl, new_sl)  # Only move stop down