#!/usr/bin/env python3
"""
Bull Machine v1.4 Strategy Adapter
Integrates v1.3 MTF engine with v1.4 backtest framework
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bull_machine.app.main_v13 import run_bull_machine_v1_3
from bull_machine.core.types import Bar, Series


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


def df_to_series(symbol: str, tf: str, df: pd.DataFrame) -> Series:
    """Convert DataFrame to v1.3 Series format"""
    bars = []

    for idx, row in df.iterrows():
        # Handle timestamp conversion
        if isinstance(idx, pd.Timestamp):
            ts = int(idx.value // 10**9)
        else:
            ts = int(idx)

        bar = Bar(
            ts=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
        )
        bars.append(bar)

    return Series(bars=bars, symbol=symbol, timeframe=tf)


def save_temp_csv(df: pd.DataFrame, symbol: str) -> str:
    """Save DataFrame to temporary CSV for v1.3 pipeline"""
    import tempfile

    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, prefix=f"{symbol}_backtest_")

    # Write data
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return temp_file.name


def strategy_from_df(
    symbol: str, tf: str, df_window: pd.DataFrame, balance: float = 10000, config_path: str = None
) -> Dict[str, Any]:
    """
    Main adapter: Call v1.3 pipeline from DataFrame and map to backtest format

    Args:
        symbol: Trading symbol (e.g., 'BTCUSD')
        tf: Primary timeframe (e.g., '1H')
        df_window: DataFrame with OHLCV data up to current bar
        balance: Account balance for risk sizing
        config_path: Path to v1.3 config file

    Returns:
        Dictionary with {'action': 'long|short|exit|flat', 'size': float, ...}
    """

    # Basic validation
    if df_window is None or df_window.empty or len(df_window) < 100:
        return {"action": "flat", "reason": "insufficient_data"}

    try:
        # Save to temp CSV (v1.3 expects CSV input)
        csv_path = save_temp_csv(df_window, symbol)

        # Call v1.3 pipeline
        result = run_bull_machine_v1_3(
            csv_file=csv_path,
            account_balance=balance,
            config_path=config_path,
            mtf_enabled=True,  # Always use MTF in v1.4
        )

        # Clean up temp file
        if os.path.exists(csv_path):
            os.remove(csv_path)

        # Map v1.3 result to backtest format
        if result.get("action") == "enter_trade":
            signal = result.get("signal", {})
            risk_plan = result.get("risk_plan", {})

            # Calculate position size based on risk
            risk_amount = risk_plan.get("risk_amount", balance * 0.01)
            position_size = risk_plan.get("size", 1.0)

            # Build TP ladder (1R, 2R, 3R)
            entry = risk_plan.get("entry", df_window["close"].iloc[-1])
            stop = risk_plan.get("stop", entry * 0.98)
            risk_per_unit = abs(entry - stop)

            tp_ladder = (
                [
                    entry + risk_per_unit * 1.0,  # TP1: 1R
                    entry + risk_per_unit * 2.0,  # TP2: 2R
                    entry + risk_per_unit * 3.0,  # TP3: 3R
                ]
                if signal.get("side") == "long"
                else [
                    entry - risk_per_unit * 1.0,  # TP1: 1R
                    entry - risk_per_unit * 2.0,  # TP2: 2R
                    entry - risk_per_unit * 3.0,  # TP3: 3R
                ]
            )

            # Build TP levels in broker-compatible format
            tp_levels = [
                {"price": tp_ladder[0], "r": 1.0, "pct": 33, "name": "tp1"},
                {"price": tp_ladder[1], "r": 2.0, "pct": 33, "name": "tp2"},
                {"price": tp_ladder[2], "r": 3.0, "pct": 34, "name": "tp3"},
            ]

            risk_plan = {
                "entry": entry,
                "stop": stop,
                "size": position_size,
                "tp_levels": tp_levels,
                "rules": {"be_at": "tp1", "trail_at": "tp2", "trail_mode": "swing"},
            }

            return {
                "action": signal.get("side", "flat"),
                "size": position_size,
                "confidence": signal.get("confidence", 0.5),
                "stop": stop,
                "tp": tp_ladder,  # Legacy format
                "risk_plan": risk_plan,  # New broker format
                "risk_pct": risk_amount / balance,
                "reasons": signal.get("reasons", []),
                "mtf_sync": result.get("mtf_sync"),
            }

        elif result.get("action") == "no_trade":
            # Check if we should exit existing position
            # (In a real implementation, we'd track position state)
            return {"action": "flat", "reason": result.get("reason", "no_signal")}

        else:
            return {"action": "flat"}

    except Exception as e:
        print(f"Error in v1.3 adapter: {e}")
        return {"action": "flat", "error": str(e)}


def strategy_from_series(series: Series, balance: float = 10000, config_path: str = None) -> Dict[str, Any]:
    """
    Alternative: Call v1.3 directly with Series object (no CSV)

    This is more efficient but requires modifying v1.3 to accept Series directly
    """

    # For now, convert to DataFrame and use the CSV approach
    data = []
    for bar in series.bars:
        data.append(
            {
                "time": bar.ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )

    df = pd.DataFrame(data)
    return strategy_from_df(series.symbol, series.timeframe, df, balance, config_path)


def calculate_position_size(balance: float, risk_pct: float, entry: float, stop: float) -> float:
    """
    Calculate position size based on risk management rules

    Args:
        balance: Account balance
        risk_pct: Risk percentage per trade (e.g., 0.01 for 1%)
        entry: Entry price
        stop: Stop loss price

    Returns:
        Position size in units
    """

    risk_amount = balance * risk_pct
    risk_per_unit = abs(entry - stop)

    if risk_per_unit == 0:
        return 0

    position_size = risk_amount / risk_per_unit

    # Apply maximum position size cap (e.g., 10% of balance in value)
    max_position_value = balance * 0.1
    max_position_size = max_position_value / entry

    return min(position_size, max_position_size)


def check_exposure_limits(
    current_positions: Dict[str, Any], new_signal: Dict[str, Any], max_exposure: float = 0.5
) -> bool:
    """
    Check if new position would violate exposure limits

    Args:
        current_positions: Dictionary of current positions
        new_signal: Proposed new position
        max_exposure: Maximum net exposure (0.5 = 50%)

    Returns:
        True if position is allowed, False if it would violate limits
    """

    # Calculate current net exposure
    long_exposure = sum(p["size"] * p["price"] for p in current_positions.values() if p.get("side") == "long")
    short_exposure = sum(p["size"] * p["price"] for p in current_positions.values() if p.get("side") == "short")

    net_exposure = abs(long_exposure - short_exposure)

    # Check if new position would exceed limits
    if new_signal["action"] == "long":
        new_net = net_exposure + (new_signal["size"] * new_signal.get("entry", 0))
    elif new_signal["action"] == "short":
        new_net = net_exposure + (new_signal["size"] * new_signal.get("entry", 0))
    else:
        return True  # Exit/flat always allowed

    total_capital = sum(p.get("balance", 0) for p in current_positions.values())

    if total_capital == 0:
        return True  # First position always allowed

    return (new_net / total_capital) <= max_exposure


# Example usage for testing
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        }
    )

    # Test the adapter
    signal = strategy_from_df("BTCUSD", "1H", sample_data, balance=10000)
    print(f"Signal: {signal}")
