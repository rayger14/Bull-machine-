"""
Production-Grade Event-Driven Backtesting Engine

Nautilus-inspired architecture without external dependencies.
Provides production-grade features:
- Event loop (on_bar, on_order_filled, on_position_closed)
- Fill simulation (slippage, fees, realistic execution)
- Portfolio management (positions, cash, margin)
- Order management system (OMS)

Compatible with Python 3.9+ (no Nautilus dependency required).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionSide(Enum):
    """Position side."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Bar:
    """OHLCV bar."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __repr__(self):
        return f"Bar({self.timestamp}, O:{self.open:.2f}, H:{self.high:.2f}, L:{self.low:.2f}, C:{self.close:.2f})"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    timestamp: pd.Timestamp
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_timestamp: Optional[pd.Timestamp] = None
    position_id: Optional[str] = None  # Position ID for tracking
    is_exit: bool = False  # Flag to indicate this order is closing an existing position

    def __repr__(self):
        return f"Order({self.order_id}, {self.side.value}, qty={self.quantity:.4f}, status={self.status.value})"


@dataclass
class Position:
    """Trading position."""
    position_id: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: dict = None  # For storing original_quantity and executed_scale_outs

    def __post_init__(self):
        """Initialize metadata dict if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized PnL based on current price."""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

    def __repr__(self):
        return f"Position({self.side.value}, qty={self.quantity:.4f}, entry={self.entry_price:.2f}, upnl={self.unrealized_pnl:.2f})"


@dataclass
class Trade:
    """Completed trade."""
    trade_id: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    side: PositionSide
    quantity: float
    entry_price: float
    exit_price: float
    realized_pnl: float
    commission: float
    bars_held: int
    archetype: str = 'unknown'  # Added for archetype tracking

    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.side == PositionSide.LONG:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100

    def __repr__(self):
        return f"Trade({self.side.value}, archetype={self.archetype}, pnl={self.realized_pnl:.2f}, ret={self.return_pct:.2f}%, bars={self.bars_held})"


class Portfolio:
    """Portfolio manager."""

    def __init__(self, initial_cash: float = 100000.0, commission_rate: float = 0.001, slippage_bps: float = 2.0):
        """
        Initialize portfolio.

        Args:
            initial_cash: Starting cash balance
            commission_rate: Commission rate (0.001 = 0.1%)
            slippage_bps: Slippage in basis points (2.0 = 2bps)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps

        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.trades: List[Trade] = []

        self.total_commission = 0.0
        self.total_slippage = 0.0

        # Performance tracking
        self.equity_curve: List[float] = [initial_cash]
        self.timestamps: List[pd.Timestamp] = []

    def get_equity(self, current_price: float) -> float:
        """Calculate current equity (cash + current market value of all positions)."""
        equity = self.cash

        # Add current market value of all open positions
        for position in self.positions.values():
            # Position value at current market price
            equity += position.quantity * current_price

        return equity

    def get_position_value(self) -> float:
        """Get total position value."""
        total_value = 0.0
        for position in self.positions.values():
            total_value += position.quantity * position.entry_price
        return total_value

    def can_open_position(self, size_usd: float, current_price: float) -> bool:
        """Check if we have enough cash to open a position."""
        # Calculate required cash (including commission and slippage)
        quantity = size_usd / current_price
        cost = size_usd
        commission = cost * self.commission_rate
        slippage = cost * (self.slippage_bps / 10000.0)
        total_required = cost + commission + slippage

        return self.cash >= total_required

    def submit_market_order(self, side: OrderSide, quantity: float, timestamp: pd.Timestamp, position_id: str = None, is_exit: bool = False) -> Order:
        """Submit market order."""
        order_id = f"order_{len(self.filled_orders) + len(self.pending_orders)}"
        order = Order(
            order_id=order_id,
            timestamp=timestamp,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            status=OrderStatus.PENDING,
            position_id=position_id,
            is_exit=is_exit
        )
        self.pending_orders[order_id] = order
        return order

    def process_pending_orders(self, bar: Bar) -> List[Order]:
        """Process pending orders with realistic fill simulation."""
        filled = []

        for order_id, order in list(self.pending_orders.items()):
            if order.order_type == OrderType.MARKET:
                # Market order - fill at next bar open with slippage
                fill_price = self._apply_slippage(bar.open, order.side)
                order.fill_price = fill_price
                order.fill_timestamp = bar.timestamp
                order.status = OrderStatus.FILLED

                # Remove from pending and add to filled
                del self.pending_orders[order_id]
                self.filled_orders.append(order)
                filled.append(order)

                # Update portfolio
                self._execute_fill(order, bar)

        return filled

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to execution price."""
        slippage_pct = self.slippage_bps / 10000.0
        if side == OrderSide.BUY:
            # Buy higher
            return price * (1 + slippage_pct)
        else:
            # Sell lower
            return price * (1 - slippage_pct)

    def _execute_fill(self, order: Order, bar: Bar):
        """Execute order fill and update portfolio."""
        fill_price = order.fill_price
        quantity = order.quantity

        # Calculate costs
        notional = fill_price * quantity
        commission = notional * self.commission_rate
        slippage = notional * (self.slippage_bps / 10000.0)

        self.total_commission += commission
        self.total_slippage += slippage

        if order.side == OrderSide.BUY:
            # Check if we're closing an existing SHORT position or opening a new LONG
            if order.position_id and order.position_id in self.positions:
                # Closing a SHORT position
                logger.info(f"[FILL] BUY closing SHORT [{order.position_id}]: {quantity:.8f} @ ${fill_price:.2f}")
                self._close_position(order.position_id, quantity, fill_price, bar.timestamp)
                # Deduct cash to cover the short (buy back)
                self.cash -= (notional + commission)
            elif not order.is_exit:
                # Opening a NEW long position (only if not an exit order)
                position_id = order.position_id or self._generate_position_id(PositionSide.LONG, timestamp=bar.timestamp)
                logger.info(f"[FILL] BUY opening LONG [{position_id}]: {quantity:.8f} @ ${fill_price:.2f}")
                self._open_position(PositionSide.LONG, quantity, fill_price, bar.timestamp, position_id)
                # Deduct cash
                self.cash -= (notional + commission)
            # else: Exit order but position already gone - ignore (already closed)

        else:  # SELL
            # Check if we're closing an existing LONG position or opening a new SHORT
            if order.position_id and order.position_id in self.positions:
                # Closing a LONG position
                existing_pos = self.positions[order.position_id]
                logger.info(f"[FILL] SELL closing {existing_pos.side.value} [{order.position_id}]: {quantity:.8f} of {existing_pos.quantity:.8f} @ ${fill_price:.2f}")
                self._close_position(order.position_id, quantity, fill_price, bar.timestamp)
                # Add cash from selling
                self.cash += (notional - commission)
            elif not order.is_exit:
                # Opening a NEW short position (only if not an exit order)
                position_id = order.position_id or self._generate_position_id(PositionSide.SHORT, timestamp=bar.timestamp)
                logger.info(f"[FILL] SELL opening SHORT [{position_id}]: {quantity:.8f} @ ${fill_price:.2f}")
                self._open_position(PositionSide.SHORT, quantity, fill_price, bar.timestamp, position_id)
                # Add cash from short sale
                self.cash += (notional - commission)
            # else: Exit order but position already gone - ignore (already closed)

    def _generate_position_id(self, side: PositionSide, timestamp: pd.Timestamp) -> str:
        """Generate unique position ID."""
        side_str = 'long' if side == PositionSide.LONG else 'short'
        # Use timestamp to ensure uniqueness
        return f"{side_str}_{int(timestamp.timestamp())}"

    def _open_position(self, side: PositionSide, quantity: float, entry_price: float,
                      timestamp: pd.Timestamp, position_id: str):
        """Open new position with specified ID."""
        self.positions[position_id] = Position(
            position_id=position_id,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            metadata={
                'original_quantity': quantity,  # Store for scale-out calculations
                'executed_scale_outs': 0.0      # Track total scaled out
            }
        )

        logger.info(f"Opened {side.value} position [{position_id}]: qty={quantity:.4f} @ ${entry_price:.2f}")

    def _close_position(self, position_id: str, quantity: float, exit_price: float, exit_timestamp: pd.Timestamp):
        """Close position and record trade."""
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found - cannot close")
            return

        position = self.positions[position_id]

        # CRITICAL: Ensure we don't try to close more than we have
        # This prevents compounding errors from price changes between signal and fill
        close_quantity = min(quantity, position.quantity)
        if close_quantity < quantity * 0.99:  # Allow 1% tolerance
            logger.warning(f"Attempted to close {quantity:.8f} but only {position.quantity:.8f} available - capping to {close_quantity:.8f}")

        # Calculate PnL
        if position.side == PositionSide.LONG:
            realized_pnl = (exit_price - position.entry_price) * close_quantity
        else:
            realized_pnl = (position.entry_price - exit_price) * close_quantity

        # Subtract commission
        commission = (exit_price * close_quantity) * self.commission_rate
        realized_pnl -= commission

        # Calculate bars held
        bars_held = 0  # Will be calculated by backtest engine

        # Extract archetype from position_id
        # Format: {direction}_{archetype}_{timestamp}
        archetype = 'unknown'
        try:
            parts = position_id.split('_')
            if len(parts) >= 3:
                # Archetype is everything between direction and timestamp
                # Handle archetypes with underscores like "trap_within_trend"
                archetype = '_'.join(parts[1:-1])
        except Exception:
            pass  # Keep default 'unknown'

        # Record trade
        trade = Trade(
            trade_id=f"trade_{len(self.trades)}",
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=exit_timestamp,
            side=position.side,
            quantity=close_quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            commission=commission,
            bars_held=bars_held,
            archetype=archetype
        )
        self.trades.append(trade)

        # Update position quantity
        position.quantity -= close_quantity

        # Track executed scale-outs in metadata
        if 'executed_scale_outs' in position.metadata:
            position.metadata['executed_scale_outs'] += close_quantity

        if position.quantity <= 1e-10:  # Close if essentially zero
            del self.positions[position_id]

        logger.info(f"Closed {position.side.value} position: pnl=${realized_pnl:.2f}, ret={trade.return_pct:.2f}%")

    def update_equity_curve(self, timestamp: pd.Timestamp, current_price: float):
        """Update equity curve."""
        equity = self.get_equity(current_price)
        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
            }

        # Trade statistics
        winning_trades = [t for t in self.trades if t.realized_pnl > 0]
        losing_trades = [t for t in self.trades if t.realized_pnl <= 0]

        total_wins = sum(t.realized_pnl for t in winning_trades)
        total_losses = abs(sum(t.realized_pnl for t in losing_trades))

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        total_pnl = sum(t.realized_pnl for t in self.trades)
        total_return = (total_pnl / self.initial_cash) * 100

        # Drawdown calculation
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified - assumes daily returns)
        if len(self.equity_curve) > 1:
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': total_wins / len(winning_trades) if winning_trades else 0.0,
            'avg_loss': total_losses / len(losing_trades) if losing_trades else 0.0,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
        }


class EventEngine:
    """
    Production-grade event-driven backtesting engine.

    Nautilus-inspired architecture:
    - Event loop (on_bar, on_order_filled, on_position_closed)
    - Portfolio management
    - Realistic fill simulation
    """

    def __init__(
        self,
        strategy: 'BaseStrategy',
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 2.0
    ):
        """
        Initialize event engine.

        Args:
            strategy: Trading strategy (must implement BaseStrategy interface)
            initial_cash: Starting cash balance
            commission_rate: Commission rate (0.001 = 0.1%)
            slippage_bps: Slippage in basis points
        """
        self.strategy = strategy
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps
        )

        self.bars: List[Bar] = []
        self.current_bar: Optional[Bar] = None
        self.bar_index = 0

        logger.info(f"Initialized EventEngine with {strategy.__class__.__name__}")
        logger.info(f"Initial cash: ${initial_cash:,.2f}, Commission: {commission_rate*100:.2f}%, Slippage: {slippage_bps}bps")

    def run(self, bars: List[Bar]):
        """
        Run backtest on historical bars.

        Args:
            bars: List of OHLCV bars
        """
        self.bars = bars
        logger.info(f"Starting backtest with {len(bars)} bars")

        # Call strategy on_start
        self.strategy.on_start(self)

        # Main event loop
        for i, bar in enumerate(bars):
            self.current_bar = bar
            self.bar_index = i

            # Process pending orders from previous bar
            filled_orders = self.portfolio.process_pending_orders(bar)

            # Notify strategy of fills
            for order in filled_orders:
                self.strategy.on_order_filled(order, self)

            # Call strategy on_bar
            self.strategy.on_bar(bar, self)

            # Update equity curve
            self.portfolio.update_equity_curve(bar.timestamp, bar.close)

        # Call strategy on_stop
        self.strategy.on_stop(self)

        logger.info(f"Backtest completed - {len(self.portfolio.trades)} trades executed")

    def submit_order(self, side: OrderSide, size_usd: float, position_id: str = None, is_exit: bool = False) -> Optional[Order]:
        """
        Submit market order.

        Args:
            side: Order side (BUY/SELL)
            size_usd: Position size in USD
            position_id: Optional position ID to use (for matching with strategy tracking)
            is_exit: Flag indicating this order is closing an existing position

        Returns:
            Order object if submitted, None if rejected
        """
        if not self.current_bar:
            logger.error("No current bar - cannot submit order")
            return None

        # Calculate quantity
        quantity = size_usd / self.current_bar.close
        logger.info(f"[ORDER SUBMIT] {side.value} ${size_usd:.2f} / ${self.current_bar.close:.2f} = {quantity:.8f} BTC | position_id={position_id} | is_exit={is_exit}")

        # Check if we can open position (only for BUY orders - SELL orders close positions, no cash needed)
        if side == OrderSide.BUY and not self.portfolio.can_open_position(size_usd, self.current_bar.close):
            logger.warning(f"Insufficient cash to open ${size_usd:.2f} position")
            return None

        # Submit order (with optional position_id and is_exit flag)
        order = self.portfolio.submit_market_order(side, quantity, self.current_bar.timestamp, position_id=position_id, is_exit=is_exit)
        logger.debug(f"Submitted {side.value} order: ${size_usd:.2f} ({quantity:.4f} units) [ID: {position_id}]")

        return order

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.portfolio.get_performance_stats()

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        # Ensure arrays are same length (equity_curve has initial value)
        timestamps = self.portfolio.timestamps
        equity = self.portfolio.equity_curve[1:]  # Skip initial value

        # Handle case where no trades occurred
        if len(timestamps) == 0:
            return pd.DataFrame({'timestamp': [], 'equity': []})

        return pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity
        })


class BaseStrategy:
    """Base class for trading strategies."""

    def __init__(self, name: str = "Strategy"):
        """Initialize strategy."""
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def on_start(self, engine: EventEngine):
        """Called when backtest starts."""
        self.logger.info(f"{self.name} started")

    def on_bar(self, bar: Bar, engine: EventEngine):
        """Called on each new bar."""
        raise NotImplementedError("Subclasses must implement on_bar()")

    def on_order_filled(self, order: Order, engine: EventEngine):
        """Called when an order is filled."""
        self.logger.info(f"Order filled: {order}")

    def on_stop(self, engine: EventEngine):
        """Called when backtest ends."""
        stats = engine.get_performance_stats()
        self.logger.info(f"{self.name} stopped - PnL: ${stats['total_pnl']:,.2f}, Trades: {stats['total_trades']}")
