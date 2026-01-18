"""
Core backtesting engine that works with any BaseModel.

This is model-agnostic - it doesn't know about archetypes, fusion, or any
specific strategy logic. It just calls model.predict() and executes trades.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import logging

from engine.models.base import BaseModel, Signal, Position

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Completed trade record."""

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    size: float  # Position size ($)
    direction: str  # 'long' or 'short'
    pnl: float  # Profit/loss in $
    pnl_pct: float  # Profit/loss as %
    stop_loss: float
    exit_reason: str  # 'profit_target', 'stop_loss', 'signal', 'end_of_data'
    regime_label: str = 'unknown'  # Regime when trade was entered (crisis/risk_off/neutral/risk_on)
    metadata: dict = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        """Trade duration in hours."""
        return (self.exit_time - self.entry_time).total_seconds() / 3600

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0


@dataclass
class BacktestResults:
    """Backtest results container with comprehensive performance metrics."""

    model_name: str
    trades: List[Trade]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    equity_curve: pd.Series
    initial_capital: float = 10000.0
    metrics: dict = field(default_factory=dict)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.is_winner)

    @property
    def losing_trades(self) -> int:
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profits to gross losses."""
        wins = sum(t.pnl for t in self.trades if t.is_winner)
        losses = abs(sum(t.pnl for t in self.trades if not t.is_winner))
        return wins / losses if losses > 0 else 0.0

    @property
    def avg_win(self) -> float:
        """Average winning trade PnL."""
        if self.winning_trades == 0:
            return 0.0
        wins = [t.pnl for t in self.trades if t.is_winner]
        return sum(wins) / len(wins)

    @property
    def avg_loss(self) -> float:
        """Average losing trade PnL (negative number)."""
        if self.losing_trades == 0:
            return 0.0
        losses = [t.pnl for t in self.trades if not t.is_winner]
        return sum(losses) / len(losses)

    @property
    def avg_r_per_trade(self) -> float:
        """Average R-multiple per trade (PnL / risk)."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as % of peak equity."""
        if len(self.equity_curve) == 0:
            return 0.0

        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        return abs(drawdown.min()) * 100  # Return as percentage

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (annualized, assuming 252 trading days)."""
        if len(self.equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize (assuming hourly data -> 252 days * 24 hours)
        periods_per_year = 252 * 24
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        return sharpe

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage of initial capital."""
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def avg_trade_duration_hours(self) -> float:
        """Average trade duration in hours."""
        if self.total_trades == 0:
            return 0.0
        durations = [t.duration_hours for t in self.trades]
        return sum(durations) / len(durations)

    def to_dict(self, period: str = 'test') -> dict:
        """
        Convert results to dictionary for CSV export.

        Args:
            period: Period label (train/test/oos)

        Returns:
            Dictionary with all metrics
        """
        return {
            'model_name': self.model_name,
            'period': period,
            'profit_factor': round(self.profit_factor, 3),
            'win_rate': round(self.win_rate, 2),
            'num_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'avg_r_per_trade': round(self.avg_r_per_trade, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'avg_trade_duration_hours': round(self.avg_trade_duration_hours, 2),
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
        }

    def summary(self) -> str:
        """Return formatted summary string."""
        return f"""
{'='*60}
BACKTEST RESULTS: {self.model_name}
{'='*60}
Period: {self.start_date.date()} to {self.end_date.date()}

TRADES:
  Total: {self.total_trades}
  Winners: {self.winning_trades} ({self.win_rate:.1f}%)
  Losers: {self.losing_trades}

PERFORMANCE:
  Total PnL: ${self.total_pnl:.2f}
  Profit Factor: {self.profit_factor:.2f}

ADDITIONAL METRICS:
{self._format_metrics()}
{'='*60}
        """.strip()

    def _format_metrics(self) -> str:
        """Format additional metrics."""
        if not self.metrics:
            return "  (none)"
        lines = []
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.3f}")
            else:
                lines.append(f"  {key}: {value}")
        return '\n'.join(lines)


class BacktestEngine:
    """
    Model-agnostic backtesting engine.

    Works with any model that implements BaseModel interface.

    Usage:
        engine = BacktestEngine(model, data)
        results = engine.run(start='2023-01-01', end='2023-12-31')
        print(results.summary())
    """

    def __init__(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001  # 0.1% per trade
    ):
        """
        Initialize backtest engine.

        Args:
            model: Trading model (must implement BaseModel)
            data: Historical OHLCV + features data
            initial_capital: Starting portfolio value ($)
            commission_pct: Commission per trade (0.001 = 0.1%)
        """
        self.model = model
        self.data = data
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

        # State
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity: List[float] = []
        self.timestamps: List[pd.Timestamp] = []

    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        verbose: bool = True
    ) -> BacktestResults:
        """
        Run backtest on specified period.

        Args:
            start: Start date (YYYY-MM-DD) or None for data start
            end: End date (YYYY-MM-DD) or None for data end
            verbose: Log trade execution

        Returns:
            BacktestResults with trades, equity curve, metrics
        """
        # Filter data to period
        test_data = self.data.copy()
        if start:
            start_ts = pd.Timestamp(start)
            if test_data.index.tz is not None:
                start_ts = start_ts.tz_localize('UTC')
            test_data = test_data[test_data.index >= start_ts]
        if end:
            end_ts = pd.Timestamp(end)
            if test_data.index.tz is not None:
                end_ts = end_ts.tz_localize('UTC')
            test_data = test_data[test_data.index <= end_ts]

        if len(test_data) == 0:
            raise ValueError(f"No data in period {start} to {end}")

        # Reset state
        self.position = None
        self.trades = []
        self.equity = [self.initial_capital]
        self.timestamps = [test_data.index[0]]

        if verbose:
            logger.info(f"Running backtest: {self.model.name}")
            logger.info(f"Period: {test_data.index[0]} to {test_data.index[-1]}")
            logger.info(f"Bars: {len(test_data):,}")

        # Iterate through bars
        for idx, (timestamp, bar) in enumerate(test_data.iterrows()):
            # Get signal from model
            signal = self.model.predict(bar, self.position)

            # Execute trade logic
            self._execute_bar(timestamp, bar, signal, verbose=verbose)

            # Track equity
            current_equity = self._compute_equity(bar)
            self.equity.append(current_equity)
            self.timestamps.append(timestamp)

        # Close any open position at end
        if self.position is not None:
            final_bar = test_data.iloc[-1]
            self._close_position(
                test_data.index[-1],
                final_bar['close'],
                reason='end_of_data'
            )

        # Build results
        equity_series = pd.Series(self.equity, index=self.timestamps)
        results = BacktestResults(
            model_name=self.model.name,
            trades=self.trades,
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            equity_curve=equity_series,
            initial_capital=self.initial_capital
        )

        if verbose:
            logger.info(f"\nCompleted: {len(self.trades)} trades, PnL=${results.total_pnl:.2f}")

        return results

    def _execute_bar(
        self,
        timestamp: pd.Timestamp,
        bar: pd.Series,
        signal: Signal,
        verbose: bool = False
    ) -> None:
        """Execute trading logic for one bar."""

        close = bar['close']

        # Check stop loss first (if in position)
        if self.position is not None:
            if close <= self.position.stop_loss:
                stop_price = self.position.stop_loss
                self._close_position(timestamp, stop_price, reason='stop_loss')
                if verbose:
                    logger.info(f"  STOP LOSS @ {stop_price:.2f}")
                return

        # Entry logic
        if signal.is_entry and self.position is None:
            position_size = self.model.get_position_size(bar, signal)
            commission = position_size * self.commission_pct

            self.position = Position(
                direction=signal.direction,
                entry_price=signal.entry_price,
                entry_time=timestamp,
                size=position_size - commission,
                stop_loss=signal.stop_loss or (signal.entry_price * 0.95),  # 5% default
                take_profit=signal.take_profit,
                regime_label=signal.regime_label,
                metadata=signal.metadata
            )

            if verbose:
                logger.info(
                    f"ENTRY @ {timestamp}: {signal.direction.upper()} "
                    f"${signal.entry_price:.2f}, size=${position_size:.2f}, "
                    f"SL=${self.position.stop_loss:.2f}, regime={signal.regime_label}"
                )

        # Exit logic (signal says hold = close position)
        elif signal.direction == 'hold' and self.position is not None:
            # Check if this is an exit signal (has metadata reason)
            if signal.metadata and signal.metadata.get('reason') in ['profit_target', 'signal']:
                self._close_position(timestamp, close, reason=signal.metadata['reason'])
                if verbose:
                    logger.info(f"  EXIT @ {close:.2f} ({signal.metadata['reason']})")

    def _close_position(self, timestamp: pd.Timestamp, exit_price: float, reason: str) -> None:
        """Close current position and record trade."""
        if self.position is None:
            return

        # Calculate PnL
        if self.position.direction == 'long':
            pnl = (exit_price - self.position.entry_price) * (self.position.size / self.position.entry_price)
        else:  # short
            pnl = (self.position.entry_price - exit_price) * (self.position.size / self.position.entry_price)

        # Commission on exit
        commission = self.position.size * self.commission_pct
        pnl -= commission

        pnl_pct = pnl / self.position.size * 100

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            entry_price=self.position.entry_price,
            exit_time=timestamp,
            exit_price=exit_price,
            size=self.position.size,
            direction=self.position.direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            stop_loss=self.position.stop_loss,
            exit_reason=reason,
            regime_label=self.position.regime_label or 'unknown',
            metadata=self.position.metadata
        )

        self.trades.append(trade)
        self.position = None

    def _compute_equity(self, bar: pd.Series) -> float:
        """Compute current equity (realized + unrealized)."""
        realized_pnl = sum(t.pnl for t in self.trades)
        equity = self.initial_capital + realized_pnl

        # Add unrealized PnL if in position
        if self.position is not None:
            close = bar['close']
            if self.position.direction == 'long':
                unrealized = (close - self.position.entry_price) * (self.position.size / self.position.entry_price)
            else:
                unrealized = (self.position.entry_price - close) * (self.position.size / self.position.entry_price)
            equity += unrealized

        return equity
