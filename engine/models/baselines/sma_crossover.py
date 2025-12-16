"""
Baseline 2: SMA Crossover (Golden Cross / Death Cross)

Classic dual-SMA crossover strategy using 50/200 moving averages.

Purpose:
- Tests if simple momentum strategy beats trend-following
- Golden Cross: Fast SMA crosses above slow SMA (bullish signal)
- Death Cross: Fast SMA crosses below slow SMA (bearish signal)
- Extremely popular retail strategy (does it actually work?)

Strategy Logic:
1. Calculate SMA(50) and SMA(200)
2. Enter long when SMA(50) > SMA(200) (golden cross)
3. Exit to cash when SMA(50) < SMA(200) (death cross)
4. Use 5% stop loss for risk management

Parameters:
- fast_period: 50 (short-term trend)
- slow_period: 200 (long-term trend)
- stop_loss_pct: 0.05 (5% stop loss)

Expected Performance:
- Fewer trades than single SMA (waits for confirmation)
- Lags entries/exits more than SMA(200) alone
- May have better win rate but worse profit factor
- Classic "late to the party" problem
"""

import pandas as pd
import numpy as np
from typing import Optional

from engine.models.base import BaseModel, Signal, Position


class Baseline2_SMACrossover(BaseModel):
    """
    Dual SMA crossover strategy (Golden Cross / Death Cross).

    Long when fast SMA > slow SMA, cash when fast SMA < slow SMA.
    """

    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        stop_loss_pct: float = 0.05,
        position_size: float = 1000.0
    ):
        """
        Initialize SMA crossover baseline.

        Args:
            fast_period: Fast SMA period (default: 50)
            slow_period: Slow SMA period (default: 200)
            stop_loss_pct: Stop loss as % of entry (default: 0.05 = 5%)
            position_size: Fixed position size in $ (default: $1000)
        """
        super().__init__(name=f"Baseline2_SMA{fast_period}x{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pct = stop_loss_pct
        self.position_size = position_size

        # Internal state
        self._is_golden_cross = False
        self._prev_fast_sma = None
        self._prev_slow_sma = None

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Fit is minimal - just validates data has enough bars.

        Args:
            train_data: Historical data (must have 'close' column)
            **kwargs: Additional parameters (ignored)
        """
        if len(train_data) < self.slow_period:
            raise ValueError(
                f"Train data has {len(train_data)} bars, "
                f"but SMA requires {self.slow_period} bars minimum"
            )

        if 'close' not in train_data.columns:
            raise ValueError("Train data must have 'close' column")

        self._is_fitted = True
        self._is_golden_cross = False
        self._prev_fast_sma = None
        self._prev_slow_sma = None

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal based on SMA crossover.

        Args:
            bar: Current bar data (must have SMA columns)
            position: Current position (if any)

        Returns:
            Signal: Long if golden cross, exit if death cross
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        close = bar['close']

        # Get SMA values (try multiple column name formats)
        fast_sma = self._get_sma(bar, self.fast_period)
        slow_sma = self._get_sma(bar, self.slow_period)

        # Check if SMAs available
        if fast_sma is None or slow_sma is None or pd.isna(fast_sma) or pd.isna(slow_sma):
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'reason': 'insufficient_history'}
            )

        # Determine if in golden cross state
        is_golden_cross = fast_sma > slow_sma
        self._is_golden_cross = is_golden_cross

        # Detect crossover event (for metadata)
        crossover_event = None
        if self._prev_fast_sma is not None and self._prev_slow_sma is not None:
            # Bullish crossover (golden cross)
            if fast_sma > slow_sma and self._prev_fast_sma <= self._prev_slow_sma:
                crossover_event = 'golden_cross'
            # Bearish crossover (death cross)
            elif fast_sma < slow_sma and self._prev_fast_sma >= self._prev_slow_sma:
                crossover_event = 'death_cross'

        # Store for next iteration
        self._prev_fast_sma = fast_sma
        self._prev_slow_sma = slow_sma

        # Entry logic: Enter long if golden cross and not in position
        if is_golden_cross and position is None:
            stop_loss = close * (1.0 - self.stop_loss_pct)
            confidence = 0.9 if crossover_event == 'golden_cross' else 0.7
            return Signal(
                direction='long',
                confidence=confidence,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=None,
                metadata={
                    'strategy': 'sma_crossover',
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma,
                    'crossover_event': crossover_event
                }
            )

        # Exit logic: Exit if death cross and in position
        if not is_golden_cross and position is not None:
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={
                    'reason': 'signal',
                    'strategy': 'sma_crossover',
                    'crossover_event': crossover_event
                }
            )

        # Otherwise, hold current state
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close,
            metadata={'strategy': 'sma_crossover'}
        )

    def _get_sma(self, bar: pd.Series, period: int) -> Optional[float]:
        """
        Get SMA value from bar, trying multiple column name formats.

        Args:
            bar: Current bar data
            period: SMA period

        Returns:
            SMA value or None if not found
        """
        # Try standard column name format
        col_name = f'sma_{period}'
        if col_name in bar:
            return bar[col_name]

        # Try alternative formats
        alt_names = [
            f'SMA_{period}',
            f'sma{period}',
            f'SMA{period}'
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
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'stop_loss_pct': self.stop_loss_pct,
            'position_size': self.position_size,
            'strategy': 'sma_crossover'
        }

    def get_state(self) -> dict:
        """Get internal state."""
        return {
            **super().get_state(),
            'is_golden_cross': self._is_golden_cross,
        }
