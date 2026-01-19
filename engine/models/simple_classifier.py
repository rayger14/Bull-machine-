"""
Simple buy-hold-sell classifier baseline.

This is a minimal baseline model that:
- Buys on drawdowns below threshold (e.g., -10%)
- Sells/exits when price recovers above threshold (e.g., +5%)
- Uses fixed ATR-based stops

Useful for:
1. Baseline comparison (how much value do archetypes add?)
2. Testing backtesting infrastructure
3. Quick prototyping
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base import BaseModel, Signal, Position


class BuyHoldSellClassifier(BaseModel):
    """
    Simple drawdown-based buy-hold-sell classifier.

    Entry Logic:
        - Buy when 30-day drawdown < buy_threshold (e.g., -10%)
        - Optionally require volume spike (volume_z > 2.0)

    Exit Logic:
        - Sell when profit > profit_target (e.g., +5%)
        - OR stop loss hit (entry - 2.5 * ATR)

    Example:
        >>> model = BuyHoldSellClassifier(buy_threshold=-0.10, profit_target=0.05)
        >>> model.fit(train_data)  # Optionally optimize thresholds
        >>> signal = model.predict(bar)
    """

    def __init__(
        self,
        buy_threshold: float = -0.10,
        profit_target: float = 0.05,
        stop_atr_mult: float = 2.5,
        require_volume_spike: bool = False,
        volume_z_min: float = 2.0,
        name: str = "BuyHoldSell"
    ):
        """
        Initialize classifier.

        Args:
            buy_threshold: Drawdown threshold for entry (e.g., -0.10 = -10%)
            profit_target: Profit target for exit (e.g., 0.05 = +5%)
            stop_atr_mult: Stop loss multiplier (e.g., 2.5 * ATR)
            require_volume_spike: Require volume confirmation
            volume_z_min: Minimum volume z-score if required
            name: Model name
        """
        super().__init__(name=name)
        self.buy_threshold = buy_threshold
        self.profit_target = profit_target
        self.stop_atr_mult = stop_atr_mult
        self.require_volume_spike = require_volume_spike
        self.volume_z_min = volume_z_min

    def fit(self, train_data: pd.DataFrame, optimize: bool = True) -> None:
        """
        Optionally optimize thresholds on training data.

        Args:
            train_data: Historical data for calibration
            optimize: If True, run grid search for best thresholds

        Note: For now, this is a placeholder. Can add Optuna optimization later.
        """
        if optimize:
            # Placeholder for future optimization
            # Could use Optuna to find best buy_threshold, profit_target
            pass

        self._is_fitted = True

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal based on drawdown and volume.

        Logic:
            IF no position:
                IF drawdown < buy_threshold AND (volume_spike OR not required):
                    ENTER LONG
            IF position:
                IF profit > profit_target:
                    EXIT (take profit)

        Args:
            bar: Current bar data
            position: Current open position (if any)

        Returns:
            Signal with direction, confidence, stop loss
        """
        close = bar['close']

        # Exit logic (if in position)
        if position is not None:
            profit_pct = (close - position.entry_price) / position.entry_price
            if profit_pct >= self.profit_target:
                return Signal(
                    direction='hold',  # Exit signal (backtester interprets as close)
                    confidence=1.0,
                    entry_price=close,
                    metadata={'reason': 'profit_target', 'profit_pct': profit_pct}
                )

        # Entry logic (if not in position)
        if position is None:
            # Compute drawdown from 30-day high
            if 'capitulation_depth' in bar:
                drawdown = bar['capitulation_depth']
            else:
                # Fallback: compute on the fly (less accurate without full history)
                drawdown = -0.05  # Placeholder

            # Check drawdown threshold
            if drawdown < self.buy_threshold:
                # Check volume if required
                volume_ok = True
                if self.require_volume_spike:
                    if 'volume_z' in bar:
                        volume_ok = bar['volume_z'] > self.volume_z_min
                    else:
                        volume_ok = False

                if volume_ok:
                    # Compute stop loss
                    atr = bar.get('atr_14', bar.get('atr_20', close * 0.02))
                    stop_loss = close - (atr * self.stop_atr_mult)

                    return Signal(
                        direction='long',
                        confidence=abs(drawdown) / abs(self.buy_threshold),  # 0.0-1.0
                        entry_price=close,
                        stop_loss=stop_loss,
                        metadata={
                            'drawdown': drawdown,
                            'volume_z': bar.get('volume_z', None)
                        }
                    )

        # Default: hold
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close
        )

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """
        Fixed 2% risk position sizing.

        Args:
            bar: Current bar data
            signal: Entry signal

        Returns:
            Position size in quote currency ($)
        """
        # Assume $10,000 portfolio (could be parameterized)
        portfolio_value = 10000
        risk_pct = 0.02

        if signal.stop_loss is None:
            # Fallback: 2% of portfolio
            return portfolio_value * risk_pct

        # Risk = (entry - stop) * position_size
        # Position_size = (portfolio * risk_pct) / (entry - stop)
        risk_per_unit = signal.entry_price - signal.stop_loss
        if risk_per_unit <= 0:
            return 0.0

        position_size = (portfolio_value * risk_pct) / risk_per_unit
        return position_size

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'buy_threshold': self.buy_threshold,
            'profit_target': self.profit_target,
            'stop_atr_mult': self.stop_atr_mult,
            'require_volume_spike': self.require_volume_spike,
            'volume_z_min': self.volume_z_min
        }
