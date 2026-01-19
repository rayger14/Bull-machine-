"""
Baseline 5: Cash (Do Nothing)

The ultimate sanity check - never trade, always hold cash.

Purpose:
- Validates that backtesting engine doesn't generate phantom profits
- Tests if transaction costs are being applied correctly
- Confirms that "doing nothing" returns exactly 0% (not +0.1% or -0.1%)
- Debugging tool for engine issues

Strategy Logic:
1. Never enter any trades
2. Always return 'hold' signal
3. Final PnL should be exactly $0.00

Parameters: None

Expected Performance:
- Total PnL: $0.00
- Trades: 0
- Win rate: N/A
- Profit factor: N/A
- If this shows anything other than $0.00, your engine is broken
"""

import pandas as pd
from typing import Optional

from engine.models.base import BaseModel, Signal, Position


class Baseline5_Cash(BaseModel):
    """
    Cash baseline - never trade, always hold cash.

    This is a sanity check. If this strategy shows any PnL (positive or negative),
    your backtesting engine has a bug.
    """

    def __init__(self):
        """Initialize cash baseline."""
        super().__init__(name="Baseline5_Cash")

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Fit is a no-op for cash strategy.

        Args:
            train_data: Historical data (ignored)
            **kwargs: Additional parameters (ignored)
        """
        self._is_fitted = True

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Always return hold signal (never trade).

        Args:
            bar: Current bar data
            position: Current position (should always be None)

        Returns:
            Signal: Always hold (no action)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        close = bar['close']

        # Never trade - always hold
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close,
            metadata={'strategy': 'cash', 'action': 'do_nothing'}
        )

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """
        This should never be called (no entries).

        Args:
            bar: Current bar data (ignored)
            signal: Entry signal (should never be an entry signal)

        Returns:
            0.0 (never enter positions)
        """
        return 0.0

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'strategy': 'cash',
            'trades': 'never'
        }

    def get_state(self) -> dict:
        """Get internal state."""
        return {
            **super().get_state(),
            'strategy': 'cash',
        }
