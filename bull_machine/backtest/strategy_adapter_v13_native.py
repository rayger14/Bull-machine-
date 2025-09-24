"""
Native v1.3 Strategy Adapter for Bull Machine Backtest Engine

This adapter uses the actual main_v13.py pipeline without modifications,
providing the true v1.3 performance baseline.
"""

import pandas as pd
import tempfile
import os
from typing import Dict, Any, Optional
from bull_machine.app.main_v13 import run_bull_machine_v1_3

def save_temp_csv(df_window: pd.DataFrame, symbol: str) -> str:
    """Save DataFrame window as temporary CSV for v1.3 pipeline."""
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=f'_{symbol}.csv', prefix='v13_test_')
    os.close(fd)

    # Save with proper column names
    df_save = df_window.copy()
    if 'timestamp' not in df_save.columns and hasattr(df_save.index, 'to_series'):
        df_save['timestamp'] = df_save.index.astype(int) // 10**9  # Convert to Unix timestamp

    df_save.to_csv(temp_path, index=False)
    return temp_path

def strategy_from_df(symbol: str, tf: str, df_window: pd.DataFrame,
                     balance: float = 10000, config_path: str = None) -> Dict[str, Any]:
    """
    Native v1.3 strategy function that uses actual main_v13.py pipeline.

    Args:
        symbol: Asset symbol
        tf: Timeframe (used for logging only)
        df_window: Price data window
        balance: Account balance
        config_path: Optional config file path

    Returns:
        Dictionary with v1.3 pipeline result
    """

    # Minimum data requirements
    if len(df_window) < 50:
        return {
            "action": "no_trade",
            "reason": "insufficient_data",
            "version": "v1.3_native"
        }

    try:
        # Save DataFrame as temp CSV
        csv_path = save_temp_csv(df_window, symbol)

        # Run actual v1.3 pipeline
        result = run_bull_machine_v1_3(
            csv_file=csv_path,
            account_balance=balance,
            config_path=config_path,
            mtf_enabled=True
        )

        # Clean up temp file
        try:
            os.unlink(csv_path)
        except:
            pass

        # Add metadata
        result["version"] = "v1.3_native"
        result["symbol"] = symbol
        result["timeframe"] = tf

        return result

    except Exception as e:
        return {
            "action": "error",
            "message": str(e),
            "version": "v1.3_native",
            "symbol": symbol
        }

def get_strategy_adapter():
    """Return the native v1.3 strategy function."""
    return strategy_from_df