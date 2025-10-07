#!/usr/bin/env python3
"""
Execution Simulator for Bull Machine v1.7.3 Paper Trading
Simulates fills, slippage, fees, and position/PnL tracking
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class OrderSide(Enum):
    LONG = "long"
    SHORT = "short"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Order representation for simulation."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    size: float
    price: float
    order_type: str = "market"
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None


@dataclass
class Position:
    """Position tracking for simulation."""
    symbol: str
    size: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @property
    def is_flat(self) -> bool:
        return abs(self.size) < 1e-8


@dataclass
class Fill:
    """Fill record for trade tracking."""
    timestamp: datetime
    order_id: str
    symbol: str
    side: OrderSide
    size: float
    price: float
    fees: float
    pnl: float = 0.0


class ExecutionSimulator:
    """Paper trading execution simulator with realistic costs."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.fills: List[Fill] = []

        # Cost model (institutional-grade)
        self.fees_bps = 10  # 10 basis points (0.1%)
        self.slippage_bps = 5  # 5 basis points slippage
        self.spread_bps = 2  # 2 basis points bid-ask spread

        # Risk parameters
        self.max_leverage = 3.0
        self.max_position_size = 0.5  # 50% of balance per position

        self.order_counter = 0

    def create_order(self, symbol: str, side: OrderSide, size_usd: float,
                     current_price: float, timestamp: datetime) -> Order:
        """Create a new order."""
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"

        # Convert USD size to coin size
        coin_size = size_usd / current_price

        order = Order(
            id=order_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            size=coin_size,
            price=current_price
        )

        self.orders.append(order)
        return order

    def simulate_fill(self, order: Order, market_price: float, timestamp: datetime) -> Optional[Fill]:
        """
        Simulate order fill with realistic costs.

        Args:
            order: Order to fill
            market_price: Current market price
            timestamp: Fill timestamp

        Returns:
            Fill object if order can be filled
        """
        if order.status != OrderStatus.PENDING:
            return None

        # Calculate fill price with slippage and spread
        if order.side == OrderSide.LONG:
            # Buy at ask (market price + spread/2 + slippage)
            fill_price = market_price * (1 + (self.spread_bps + self.slippage_bps) / 10000)
        else:
            # Sell at bid (market price - spread/2 - slippage)
            fill_price = market_price * (1 - (self.spread_bps + self.slippage_bps) / 10000)

        # Calculate notional value and fees
        notional = order.size * fill_price
        fees = notional * (self.fees_bps / 10000)

        # Risk checks
        if not self._risk_check(order.symbol, order.side, notional + fees):
            print(f"⚠️  Risk check failed for {order.id}")
            order.status = OrderStatus.CANCELLED
            return None

        # Update order
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_time = timestamp

        # Calculate PnL if closing position
        pnl = self._calculate_pnl(order.symbol, order.side, order.size, fill_price)

        # Update position
        self._update_position(order.symbol, order.side, order.size, fill_price)

        # Update balance
        if order.side == OrderSide.LONG:
            self.balance -= (notional + fees)
        else:
            self.balance += (notional - fees)

        # Create fill record
        fill = Fill(
            timestamp=timestamp,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            price=fill_price,
            fees=fees,
            pnl=pnl
        )

        self.fills.append(fill)
        return fill

    def _risk_check(self, symbol: str, side: OrderSide, notional: float) -> bool:
        """Check risk limits before execution."""
        # Check available balance
        if side == OrderSide.LONG and notional > self.balance:
            return False

        # Check position size limits
        current_position = self.positions.get(symbol, Position(symbol))
        new_notional = abs(current_position.size * current_position.avg_price) + notional

        if new_notional > self.initial_balance * self.max_position_size:
            return False

        # Check leverage
        total_exposure = sum(abs(pos.size * pos.avg_price) for pos in self.positions.values())
        total_exposure += notional

        if total_exposure > self.balance * self.max_leverage:
            return False

        return True

    def _calculate_pnl(self, symbol: str, side: OrderSide, size: float, price: float) -> float:
        """Calculate PnL for position changes."""
        position = self.positions.get(symbol, Position(symbol))

        if position.is_flat:
            return 0.0

        # Check if this is a closing trade
        if (position.is_long and side == OrderSide.SHORT) or (position.is_short and side == OrderSide.LONG):
            close_size = min(abs(size), abs(position.size))
            pnl = close_size * (price - position.avg_price)

            if position.is_short:
                pnl = -pnl

            return pnl

        return 0.0

    def _update_position(self, symbol: str, side: OrderSide, size: float, price: float):
        """Update position after fill."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        position = self.positions[symbol]

        if side == OrderSide.LONG:
            new_size = position.size + size
        else:
            new_size = position.size - size

        # Calculate new average price
        if new_size == 0:
            position.size = 0
            position.avg_price = 0
        elif (position.size >= 0 and new_size >= 0) or (position.size <= 0 and new_size <= 0):
            # Adding to position
            total_cost = (position.size * position.avg_price) + (size * price * (1 if side == OrderSide.LONG else -1))
            position.avg_price = abs(total_cost / new_size) if new_size != 0 else 0
            position.size = new_size
        else:
            # Flipping position or partial close
            if abs(new_size) < abs(position.size):
                # Partial close - keep average price
                position.size = new_size
            else:
                # Flip - new average price
                position.size = new_size
                position.avg_price = price

    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """Update unrealized PnL for open positions."""
        if symbol in self.positions:
            position = self.positions[symbol]
            if not position.is_flat:
                position.unrealized_pnl = position.size * (current_price - position.avg_price)

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(fill.pnl for fill in self.fills)

        return {
            'balance': self.balance,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_realized,
            'total_pnl': total_realized + total_unrealized,
            'total_equity': self.initial_balance + total_realized + total_unrealized,
            'return_pct': ((total_realized + total_unrealized) / self.initial_balance) * 100,
            'positions': {symbol: {
                'size': pos.size,
                'avg_price': pos.avg_price,
                'unrealized_pnl': pos.unrealized_pnl
            } for symbol, pos in self.positions.items() if not pos.is_flat},
            'total_trades': len(self.fills),
            'total_fees': sum(fill.fees for fill in self.fills)
        }

    def get_trade_summary(self) -> Dict:
        """Get trading performance summary."""
        if not self.fills:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': float('inf'),
                'total_fees': 0
            }

        winning_trades = [f for f in self.fills if f.pnl > 0]
        losing_trades = [f for f in self.fills if f.pnl < 0]

        return {
            'total_trades': len(self.fills),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.fills) * 100 if self.fills else 0,
            'avg_win': np.mean([f.pnl for f in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([f.pnl for f in losing_trades]) if losing_trades else 0,
            'profit_factor': (sum(f.pnl for f in winning_trades) / abs(sum(f.pnl for f in losing_trades))) if losing_trades else float('inf'),
            'total_fees': sum(f.fees for f in self.fills)
        }
