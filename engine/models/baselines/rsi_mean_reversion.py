"""
Baseline 3: RSI Mean Reversion

Classic mean-reversion strategy using RSI overbought/oversold levels.

Purpose:
- Tests if "buy the dip" works better than trend-following
- Opposite philosophy to trend-following (bet on reversals)
- RSI(14) < 30 = oversold (buy signal)
- RSI(14) > 70 = overbought (sell signal)

Strategy Logic:
1. Calculate RSI(14)
2. Enter long when RSI < 30 (oversold, expect bounce)
3. Exit when RSI > 70 (overbought, take profit)
4. Use 5% stop loss for catastrophic drops

Parameters:
- rsi_period: 14 (Welles Wilder's original parameter)
- entry_threshold: 30 (oversold)
- exit_threshold: 70 (overbought)
- stop_loss_pct: 0.05 (5% stop loss)

Expected Performance:
- Works well in ranging markets (buy low, sell high)
- Catastrophic in strong trends (catches falling knives)
- Higher trade frequency than trend-following
- Win rate may be deceptive (small wins, big losses)
"""

import pandas as pd
import numpy as np
from typing import Optional

from engine.models.base import BaseModel, Signal, Position


class Baseline3_RSIMeanReversion(BaseModel):
    """
    Mean-reversion strategy using RSI oversold/overbought levels.

    Buy when RSI < 30, sell when RSI > 70.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        entry_threshold: float = 30.0,
        exit_threshold: float = 70.0,
        stop_loss_pct: float = 0.05,
        position_size: float = 1000.0
    ):
        """
        Initialize RSI mean-reversion baseline.

        Args:
            rsi_period: RSI lookback period (default: 14)
            entry_threshold: RSI level to enter (default: 30)
            exit_threshold: RSI level to exit (default: 70)
            stop_loss_pct: Stop loss as % of entry (default: 0.05 = 5%)
            position_size: Fixed position size in $ (default: $1000)
        """
        super().__init__(name=f"Baseline3_RSI{rsi_period}MR")
        self.rsi_period = rsi_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.position_size = position_size

        # Internal state
        self._last_rsi = None

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Fit is minimal - just validates data has RSI.

        Args:
            train_data: Historical data (must have 'close' column)
            **kwargs: Additional parameters (ignored)
        """
        if len(train_data) < self.rsi_period + 1:
            raise ValueError(
                f"Train data has {len(train_data)} bars, "
                f"but RSI requires {self.rsi_period + 1} bars minimum"
            )

        if 'close' not in train_data.columns:
            raise ValueError("Train data must have 'close' column")

        self._is_fitted = True
        self._last_rsi = None

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal based on RSI levels.

        Args:
            bar: Current bar data (must have RSI column)
            position: Current position (if any)

        Returns:
            Signal: Long if oversold, exit if overbought
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        close = bar['close']

        # Get RSI value (try multiple column name formats)
        rsi = self._get_rsi(bar)

        # Check if RSI available
        if rsi is None or pd.isna(rsi):
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'reason': 'insufficient_history'}
            )

        self._last_rsi = rsi

        # Entry logic: Enter long if oversold and not in position
        if rsi < self.entry_threshold and position is None:
            # Confidence scales with how oversold (more oversold = higher confidence)
            confidence = min(0.9, (self.entry_threshold - rsi) / self.entry_threshold)
            stop_loss = close * (1.0 - self.stop_loss_pct)

            return Signal(
                direction='long',
                confidence=confidence,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=None,
                metadata={
                    'strategy': 'rsi_mean_reversion',
                    'rsi': rsi,
                    'signal_type': 'oversold'
                }
            )

        # Exit logic: Exit if overbought and in position
        if rsi > self.exit_threshold and position is not None:
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={
                    'reason': 'signal',
                    'strategy': 'rsi_mean_reversion',
                    'rsi': rsi,
                    'signal_type': 'overbought'
                }
            )

        # Otherwise, hold current state
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close,
            metadata={'strategy': 'rsi_mean_reversion', 'rsi': rsi}
        )

    def _get_rsi(self, bar: pd.Series) -> Optional[float]:
        """
        Get RSI value from bar, trying multiple column name formats.

        Args:
            bar: Current bar data

        Returns:
            RSI value or None if not found
        """
        # Try standard column name format
        col_name = f'rsi_{self.rsi_period}'
        if col_name in bar:
            return bar[col_name]

        # Try alternative formats
        alt_names = [
            'rsi',
            'RSI',
            f'RSI_{self.rsi_period}',
            f'rsi{self.rsi_period}',
            f'RSI{self.rsi_period}',
            'rsi_14'  # Common default
        ]
        for alt_name in alt_names:
            if alt_name in bar:
                return bar[alt_name]

        return None

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
            'rsi_period': self.rsi_period,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'position_size': self.position_size,
            'strategy': 'rsi_mean_reversion'
        }

    def get_state(self) -> dict:
        """Get internal state."""
        return {
            **super().get_state(),
            'last_rsi': self._last_rsi,
        }
