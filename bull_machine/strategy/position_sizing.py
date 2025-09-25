"""
Bull Machine ATR-Based Position Sizing
Consistent risk per trade across volatility regimes
"""

from typing import Dict
import pandas as pd
from bull_machine.core.numerics import safe_atr

def atr_risk_size(df: pd.DataFrame, equity: float, risk_cfg: Dict) -> float:
    """
    Calculate position size based on ATR and risk percentage.

    Args:
        df: Price DataFrame with OHLC data
        equity: Current portfolio equity
        risk_cfg: Risk configuration dict

    Returns:
        Position size in base currency
    """
    # Get current ATR
    atr_window = risk_cfg.get("atr_window", 14)
    atr = float(safe_atr(df, atr_window).iloc[-1])

    # Calculate risk amount (percentage of equity)
    risk_pct = risk_cfg.get("risk_pct", 0.005)  # Default 0.5%
    risk_amt = equity * float(risk_pct)

    # Calculate stop loss distance in price terms
    sl_atr_multiple = risk_cfg.get("sl_atr", 2.0)
    sl_dist = max(atr * float(sl_atr_multiple), 1e-8)

    # Position size = risk amount / stop distance
    position_size = max(risk_amt / sl_dist, 0.0)

    return position_size