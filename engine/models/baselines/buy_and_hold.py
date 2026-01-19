"""
Baseline 0: Buy and Hold

The simplest possible strategy - buy on first bar, hold until last bar.

Purpose:
- Sanity check: Does the market have positive expectancy?
- Baseline for all strategies: If you can't beat buy-and-hold, why trade?
- Tests if your costs/slippage assumptions are killing performance

Strategy Logic:
1. Enter long on first bar
2. Hold until end of test period
3. Never exit (except at end)

Parameters: None (it's just buy-and-hold)

Expected Performance:
- High profit factor IF market trends up
- Zero profit factor IF market trends down
- Very low trade count (1 trade for entire period)
- Zero drawdown control (rides through all dips)
"""

import pandas as pd
import numpy as np
from typing import Optional

from engine.models.base import BaseModel, Signal, Position


class Baseline0_BuyAndHold(BaseModel):
    """
    Buy and hold baseline - always long from first to last bar.

    This is the ultimate honesty check. If your fancy strategy can't beat
    simply holding the asset, you're just churning and burning capital.
    """

    def __init__(self, position_size: float = 1000.0):
        """
        Initialize buy-and-hold baseline.

        Args:
            position_size: Fixed position size in $ (default: $1000)
        """
        super().__init__(name="Baseline0_BuyAndHold")
        self.position_size = position_size
        self._has_entered = False  # Track if we've entered

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Fit is a no-op for buy-and-hold (no parameters to calibrate).

        Args:
            train_data: Historical data (ignored)
            **kwargs: Additional parameters (ignored)
        """
        self._is_fitted = True
        self._has_entered = False  # Reset entry flag

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal: Enter long on first bar, hold forever.

        Args:
            bar: Current bar data
            position: Current position (if any)

        Returns:
            Signal: Long on first bar, hold thereafter
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        close = bar['close']

        # Enter on first bar if not already in position
        if not self._has_entered and position is None:
            self._has_entered = True
            return Signal(
                direction='long',
                confidence=1.0,  # Maximum confidence (always buy)
                entry_price=close,
                stop_loss=0.0,  # No stop loss (hold through everything)
                take_profit=None,  # No take profit (hold forever)
                metadata={'strategy': 'buy_and_hold'}
            )

        # Once in position, always hold
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close,
            metadata={'strategy': 'buy_and_hold'}
        )

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """
        Return fixed position size.

        Args:
            bar: Current bar data (ignored)
            signal: Entry signal (ignored)

        Returns:
            Fixed position size in $
        """
        return self.position_size

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'position_size': self.position_size,
            'strategy': 'buy_and_hold'
        }

    def get_state(self) -> dict:
        """Get internal state."""
        return {
            **super().get_state(),
            'has_entered': self._has_entered,
        }
