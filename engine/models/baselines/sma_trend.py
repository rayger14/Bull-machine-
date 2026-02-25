"""
Baseline 1: SMA 200 Trend Following

Classic trend-following strategy using 200-period Simple Moving Average.

Purpose:
- Tests if simple trend-following beats buy-and-hold
- Avoids major drawdowns by going to cash in downtrends
- Standard institutional approach (many funds use SMA200)

Strategy Logic:
1. Calculate SMA(200) of close prices
2. Enter long when close > SMA(200) (uptrend)
3. Exit to cash when close < SMA(200) (downtrend)
4. Use 5% stop loss for risk management

Parameters:
- sma_period: 200 (classic institutional standard)
- stop_loss_pct: 0.05 (5% stop loss)

Expected Performance:
- Outperforms buy-and-hold in ranging/bear markets
- Underperforms buy-and-hold in strong bull markets (whipsaws)
- Lower drawdowns than buy-and-hold
- Trade frequency: Low (only on trend changes)
"""

import pandas as pd
from typing import Optional

from engine.models.base import BaseModel, Signal, Position


class Baseline1_SMA200Trend(BaseModel):
    """
    Trend-following strategy using 200-period SMA.

    Long when price > SMA(200), cash when price < SMA(200).
    """

    def __init__(
        self,
        sma_period: int = 200,
        stop_loss_pct: float = 0.05,
        position_size: float = 1000.0
    ):
        """
        Initialize SMA trend-following baseline.

        Args:
            sma_period: SMA lookback period (default: 200)
            stop_loss_pct: Stop loss as % of entry (default: 0.05 = 5%)
            position_size: Fixed position size in $ (default: $1000)
        """
        super().__init__(name=f"Baseline1_SMA{sma_period}Trend")
        self.sma_period = sma_period
        self.stop_loss_pct = stop_loss_pct
        self.position_size = position_size

        # Internal state
        self._sma_values = None
        self._is_in_uptrend = False

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Fit is minimal for SMA - just validates data has enough bars.

        Args:
            train_data: Historical data (must have 'close' column)
            **kwargs: Additional parameters (ignored)
        """
        if len(train_data) < self.sma_period:
            raise ValueError(
                f"Train data has {len(train_data)} bars, "
                f"but SMA requires {self.sma_period} bars minimum"
            )

        if 'close' not in train_data.columns:
            raise ValueError("Train data must have 'close' column")

        self._is_fitted = True
        self._is_in_uptrend = False

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal based on price vs SMA(200).

        Args:
            bar: Current bar data (must have 'close' and 'sma_200' columns)
            position: Current position (if any)

        Returns:
            Signal: Long if uptrend, hold (exit) if downtrend
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        close = bar['close']

        # Calculate SMA if not in bar (fallback to column name)
        sma_col = f'sma_{self.sma_period}'
        if sma_col in bar:
            sma = bar[sma_col]
        elif 'sma_200' in bar:
            sma = bar['sma_200']
        else:
            # SMA not available - cannot trade
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'error': 'SMA not available'}
            )

        # Check if NaN (not enough history yet)
        if pd.isna(sma):
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'reason': 'insufficient_history'}
            )

        # Determine trend
        in_uptrend = close > sma
        self._is_in_uptrend = in_uptrend

        # Entry logic: Enter long if uptrend and not in position
        if in_uptrend and position is None:
            stop_loss = close * (1.0 - self.stop_loss_pct)
            return Signal(
                direction='long',
                confidence=0.8,  # High confidence for clear trend
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=None,
                metadata={'strategy': 'sma_trend', 'sma': sma}
            )

        # Exit logic: Exit if downtrend and in position
        if not in_uptrend and position is not None:
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'reason': 'signal', 'strategy': 'sma_trend'}
            )

        # Otherwise, hold current state
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close,
            metadata={'strategy': 'sma_trend'}
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
            'sma_period': self.sma_period,
            'stop_loss_pct': self.stop_loss_pct,
            'position_size': self.position_size,
            'strategy': 'sma_trend'
        }

    def get_state(self) -> dict:
        """Get internal state."""
        return {
            **super().get_state(),
            'is_in_uptrend': self._is_in_uptrend,
        }
