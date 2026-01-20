"""
Base model interface for all trading strategies.

Provides clear separation between:
1. Model logic (what to trade)
2. Backtesting logic (how to evaluate)
3. Risk management (position sizing, stops)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import pandas as pd


@dataclass
class Signal:
    """Trading signal output from a model."""

    direction: Literal['long', 'short', 'hold']
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    regime_label: Optional[str] = None  # Regime when signal was generated (crisis/risk_off/neutral/risk_on)
    metadata: Dict[str, Any] = None  # Model-specific context

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal (not hold)."""
        return self.direction in ['long', 'short']


@dataclass
class Position:
    """Open position state."""

    direction: Literal['long', 'short']
    entry_price: float
    entry_time: pd.Timestamp
    size: float  # Position size in quote currency ($)
    stop_loss: float
    take_profit: Optional[float] = None
    regime_label: Optional[str] = None  # Regime when position was opened
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModel(ABC):
    """
    Abstract base class for all trading models.

    This interface ensures clean separation between:
    - Model calibration (fit)
    - Signal generation (predict)
    - Position management (get_stop_loss, get_position_size)
    - Model introspection (get_params, get_state)

    Usage:
        model = ArchetypeModel(config)
        model.fit(train_data)

        for bar in test_data.iterrows():
            signal = model.predict(bar)
            if signal.is_entry:
                # Execute trade
    """

    def __init__(self, name: str = None):
        """
        Initialize model.

        Args:
            name: Human-readable model name (e.g., "S1-Crisis-Optimized")
        """
        self.name = name or self.__class__.__name__
        self._is_fitted = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Calibrate/train the model on training data.

        For archetype models: Run Optuna optimization
        For ML models: Train classifier/regressor
        For simple baselines: Optimize thresholds

        Args:
            train_data: Historical data for calibration
            **kwargs: Model-specific training parameters

        Side effects:
            - Sets internal model parameters
            - Sets self._is_fitted = True
        """
        pass

    @abstractmethod
    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate trading signal for current bar.

        Args:
            bar: Current bar data (row from DataFrame)
            position: Current open position (if any)

        Returns:
            Signal object with direction, confidence, stop loss, etc.

        Examples:
            # Entry signal
            Signal(direction='long', confidence=0.85, entry_price=50000, stop_loss=48500)

            # Hold (no action)
            Signal(direction='hold', confidence=0.0, entry_price=bar['close'])
        """
        pass

    @abstractmethod
    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """
        Calculate position size for a signal.

        Args:
            bar: Current bar data
            signal: Entry signal

        Returns:
            Position size in quote currency ($)

        Example:
            # Risk 2% of $10,000 portfolio with $1,500 stop distance
            # Position size = ($10,000 * 0.02) / $1,500 = $133
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters (for logging, comparison).

        Returns:
            Dictionary of model parameters

        Example:
            {
                'fusion_threshold': 0.65,
                'crisis_composite_min': 0.35,
                'stop_atr_mult': 2.5
            }
        """
        return {}

    def get_state(self) -> Dict[str, Any]:
        """
        Get internal model state (for debugging, analysis).

        Returns:
            Dictionary of internal state variables

        Example:
            {
                'last_signal_time': '2024-11-27 10:00',
                'cooldown_remaining': 3,
                'regime': 'crisis'
            }
        """
        return {
            'name': self.name,
            'is_fitted': self._is_fitted
        }

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ', '.join(f"{k}={v}" for k, v in list(params.items())[:3])
        return f"{self.name}({param_str}{'...' if len(params) > 3 else ''})"
