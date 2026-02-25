"""
Baseline 4: Volatility-Targeted Trend Following

SMA trend-following with volatility-adjusted position sizing.

Purpose:
- Tests if risk-adjusted position sizing improves trend-following
- Professional approach: Scale position size by volatility
- When volatility spikes (ATR high), reduce position size
- When volatility drops (ATR low), increase position size
- Targets constant volatility exposure (like CTAs/hedge funds)

Strategy Logic:
1. Follow SMA(200) trend (same as Baseline1)
2. Calculate ATR(14) for volatility measurement
3. Scale position size to target fixed % volatility
4. Position size = (target_vol * capital) / ATR
5. Use ATR-based stop loss (2.5 x ATR)

Parameters:
- sma_period: 200 (trend determination)
- atr_period: 14 (volatility measurement)
- target_vol: 0.02 (2% daily volatility target)
- stop_atr_mult: 2.5 (stop loss = 2.5 x ATR)

Expected Performance:
- Smoother equity curve than fixed sizing
- Better risk-adjusted returns (Sharpe ratio)
- Automatically de-risks in volatile markets
- May underperform in low-vol trending markets
"""

import pandas as pd
from typing import Optional

from engine.models.base import BaseModel, Signal, Position


class Baseline4_VolTargetTrend(BaseModel):
    """
    Volatility-targeted trend-following strategy.

    Same as SMA trend but with volatility-adjusted position sizing.
    """

    def __init__(
        self,
        sma_period: int = 200,
        atr_period: int = 14,
        target_vol: float = 0.02,  # 2% volatility target
        stop_atr_mult: float = 2.5,
        base_capital: float = 10000.0
    ):
        """
        Initialize volatility-targeted trend baseline.

        Args:
            sma_period: SMA lookback period (default: 200)
            atr_period: ATR lookback period (default: 14)
            target_vol: Target daily volatility (default: 0.02 = 2%)
            stop_atr_mult: Stop loss as multiple of ATR (default: 2.5)
            base_capital: Base capital for sizing (default: $10,000)
        """
        super().__init__(name=f"Baseline4_VolTarget{int(target_vol*100)}pct")
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.target_vol = target_vol
        self.stop_atr_mult = stop_atr_mult
        self.base_capital = base_capital

        # Internal state
        self._is_in_uptrend = False
        self._current_atr = None

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Fit is minimal - validates data has required indicators.

        Args:
            train_data: Historical data (must have 'close' column)
            **kwargs: Additional parameters (ignored)
        """
        if len(train_data) < max(self.sma_period, self.atr_period):
            raise ValueError(
                f"Train data has {len(train_data)} bars, "
                f"but requires {max(self.sma_period, self.atr_period)} bars minimum"
            )

        if 'close' not in train_data.columns:
            raise ValueError("Train data must have 'close' column")

        self._is_fitted = True
        self._is_in_uptrend = False
        self._current_atr = None

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal based on price vs SMA, with vol-adjusted sizing.

        Args:
            bar: Current bar data (must have SMA and ATR columns)
            position: Current position (if any)

        Returns:
            Signal: Long if uptrend with vol-adjusted stop, exit if downtrend
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        close = bar['close']

        # Get SMA value
        sma = self._get_indicator(bar, 'sma', self.sma_period)
        if sma is None or pd.isna(sma):
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'reason': 'insufficient_sma_history'}
            )

        # Get ATR value
        atr = self._get_indicator(bar, 'atr', self.atr_period)
        if atr is None or pd.isna(atr):
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'reason': 'insufficient_atr_history'}
            )

        self._current_atr = atr

        # Determine trend
        in_uptrend = close > sma
        self._is_in_uptrend = in_uptrend

        # Entry logic: Enter long if uptrend and not in position
        if in_uptrend and position is None:
            # ATR-based stop loss
            stop_loss = close - (self.stop_atr_mult * atr)

            return Signal(
                direction='long',
                confidence=0.8,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=None,
                metadata={
                    'strategy': 'vol_target_trend',
                    'sma': sma,
                    'atr': atr,
                    'stop_atr_mult': self.stop_atr_mult
                }
            )

        # Exit logic: Exit if downtrend and in position
        if not in_uptrend and position is not None:
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                metadata={'reason': 'signal', 'strategy': 'vol_target_trend'}
            )

        # Otherwise, hold current state
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close,
            metadata={'strategy': 'vol_target_trend'}
        )

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """
        Calculate volatility-adjusted position size.

        Position size = (target_vol * capital) / (ATR / price)

        Args:
            bar: Current bar data
            signal: Entry signal (contains ATR in metadata)

        Returns:
            Volatility-adjusted position size in $
        """
        if 'atr' not in signal.metadata:
            # Fallback to base capital if ATR not available
            return self.base_capital * 0.1  # 10% of capital

        atr = signal.metadata['atr']
        close = signal.entry_price

        # Avoid division by zero
        if atr <= 0:
            return self.base_capital * 0.1

        # Calculate volatility-normalized size
        # target_vol * capital / (atr/price) = target_vol * capital * (price/atr)
        vol_normalized_size = self.target_vol * self.base_capital * (close / atr)

        # Cap at 50% of capital (safety limit)
        max_size = self.base_capital * 0.5
        position_size = min(vol_normalized_size, max_size)

        return position_size

    def _get_indicator(self, bar: pd.Series, indicator: str, period: int) -> Optional[float]:
        """
        Get indicator value from bar, trying multiple column name formats.

        Args:
            bar: Current bar data
            indicator: Indicator name ('sma' or 'atr')
            period: Indicator period

        Returns:
            Indicator value or None if not found
        """
        # Try standard column name format
        col_name = f'{indicator}_{period}'
        if col_name in bar:
            return bar[col_name]

        # Try alternative formats
        alt_names = [
            f'{indicator.upper()}_{period}',
            f'{indicator}{period}',
            f'{indicator.upper()}{period}'
        ]

        # For common defaults
        if indicator == 'sma' and period == 200:
            alt_names.append('sma_200')
        if indicator == 'atr' and period == 14:
            alt_names.extend(['atr', 'ATR', 'atr_14'])

        for alt_name in alt_names:
            if alt_name in bar:
                return bar[alt_name]

        return None

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'sma_period': self.sma_period,
            'atr_period': self.atr_period,
            'target_vol': self.target_vol,
            'stop_atr_mult': self.stop_atr_mult,
            'base_capital': self.base_capital,
            'strategy': 'vol_target_trend'
        }

    def get_state(self) -> dict:
        """Get internal state."""
        return {
            **super().get_state(),
            'is_in_uptrend': self._is_in_uptrend,
            'current_atr': self._current_atr,
        }
