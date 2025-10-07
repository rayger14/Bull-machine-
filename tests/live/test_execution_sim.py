#!/usr/bin/env python3
"""
Test execution simulator PnL math with costs
"""

import sys
import os
from pathlib import Path
import pytest
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bin.live.execution_sim import ExecutionSimulator, OrderSide, OrderStatus


class TestExecutionSimulator:
    """Test suite for execution simulator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = ExecutionSimulator(initial_balance=10000.0)

    def test_long_order_creation_and_fill(self):
        """Test creating and filling a long order."""
        timestamp = datetime(2025, 6, 1, 10, 0)
        market_price = 2000.0
        size_usd = 1000.0

        # Create order
        order = self.simulator.create_order("ETH", OrderSide.LONG, size_usd, market_price, timestamp)

        assert order.symbol == "ETH"
        assert order.side == OrderSide.LONG
        assert order.status == OrderStatus.PENDING
        assert abs(order.size - (size_usd / market_price)) < 1e-8

        # Fill order
        fill = self.simulator.simulate_fill(order, market_price, timestamp)

        assert fill is not None
        assert order.status == OrderStatus.FILLED
        assert fill.price > market_price  # Should include slippage and spread
        assert fill.fees > 0  # Should have fees

    def test_short_order_creation_and_fill(self):
        """Test creating and filling a short order."""
        timestamp = datetime(2025, 6, 1, 10, 0)
        market_price = 2000.0
        size_usd = 1000.0

        # Create and fill short order
        order = self.simulator.create_order("ETH", OrderSide.SHORT, size_usd, market_price, timestamp)
        fill = self.simulator.simulate_fill(order, market_price, timestamp)

        assert fill is not None
        assert order.side == OrderSide.SHORT
        assert fill.price < market_price  # Should include negative slippage and spread

    def test_position_tracking_long(self):
        """Test position tracking for long trades."""
        timestamp = datetime(2025, 6, 1, 10, 0)
        market_price = 2000.0

        # Open long position
        order1 = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, market_price, timestamp)
        fill1 = self.simulator.simulate_fill(order1, market_price, timestamp)

        # Check position
        position = self.simulator.positions["ETH"]
        assert position.is_long
        assert position.size > 0
        assert position.avg_price > 0

        # Add to position
        order2 = self.simulator.create_order("ETH", OrderSide.LONG, 500.0, market_price * 1.1, timestamp)
        fill2 = self.simulator.simulate_fill(order2, market_price * 1.1, timestamp)

        # Position should be larger
        assert position.size > fill1.size

    def test_position_tracking_short(self):
        """Test position tracking for short trades."""
        timestamp = datetime(2025, 6, 1, 10, 0)
        market_price = 2000.0

        # Open short position
        order = self.simulator.create_order("ETH", OrderSide.SHORT, 1000.0, market_price, timestamp)
        fill = self.simulator.simulate_fill(order, market_price, timestamp)

        # Check position
        position = self.simulator.positions["ETH"]
        assert position.is_short
        assert position.size < 0

    def test_pnl_calculation_profitable_long(self):
        """Test PnL calculation for profitable long trade."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Open long at 2000
        order1 = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2000.0, timestamp)
        fill1 = self.simulator.simulate_fill(order1, 2000.0, timestamp)

        # Close long at 2100 (profit)
        order2 = self.simulator.create_order("ETH", OrderSide.SHORT, fill1.size * 2100.0, 2100.0, timestamp)
        fill2 = self.simulator.simulate_fill(order2, 2100.0, timestamp)

        # Should have positive PnL
        assert fill2.pnl > 0

    def test_pnl_calculation_losing_long(self):
        """Test PnL calculation for losing long trade."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Open long at 2000
        order1 = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2000.0, timestamp)
        fill1 = self.simulator.simulate_fill(order1, 2000.0, timestamp)

        # Close long at 1900 (loss)
        order2 = self.simulator.create_order("ETH", OrderSide.SHORT, fill1.size * 1900.0, 1900.0, timestamp)
        fill2 = self.simulator.simulate_fill(order2, 1900.0, timestamp)

        # Should have negative PnL
        assert fill2.pnl < 0

    def test_fees_calculation(self):
        """Test that fees are calculated correctly."""
        timestamp = datetime(2025, 6, 1, 10, 0)
        market_price = 2000.0
        size_usd = 1000.0

        order = self.simulator.create_order("ETH", OrderSide.LONG, size_usd, market_price, timestamp)
        fill = self.simulator.simulate_fill(order, market_price, timestamp)

        # Calculate expected fees (10 bps of notional)
        notional = fill.size * fill.price
        expected_fees = notional * 0.001  # 10 bps = 0.1% = 0.001

        assert abs(fill.fees - expected_fees) < 0.01

    def test_slippage_and_spread_impact(self):
        """Test slippage and spread impact on fill prices."""
        timestamp = datetime(2025, 6, 1, 10, 0)
        market_price = 2000.0

        # Long order should fill higher than market (slippage + spread)
        order_long = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, market_price, timestamp)
        fill_long = self.simulator.simulate_fill(order_long, market_price, timestamp)

        # Short order should fill lower than market (slippage + spread)
        order_short = self.simulator.create_order("ETH", OrderSide.SHORT, 1000.0, market_price, timestamp)
        fill_short = self.simulator.simulate_fill(order_short, market_price, timestamp)

        assert fill_long.price > market_price
        assert fill_short.price < market_price

    def test_balance_tracking(self):
        """Test balance tracking after trades."""
        initial_balance = self.simulator.balance
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Long trade should reduce balance
        order = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2000.0, timestamp)
        fill = self.simulator.simulate_fill(order, 2000.0, timestamp)

        notional_cost = fill.size * fill.price + fill.fees
        expected_balance = initial_balance - notional_cost

        assert abs(self.simulator.balance - expected_balance) < 0.01

    def test_unrealized_pnl_update(self):
        """Test unrealized PnL calculation."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Open long position
        order = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2000.0, timestamp)
        fill = self.simulator.simulate_fill(order, 2000.0, timestamp)

        # Update with higher price
        self.simulator.update_unrealized_pnl("ETH", 2100.0)
        position = self.simulator.positions["ETH"]

        # Should have positive unrealized PnL
        assert position.unrealized_pnl > 0

        # Update with lower price
        self.simulator.update_unrealized_pnl("ETH", 1900.0)

        # Should have negative unrealized PnL
        assert position.unrealized_pnl < 0

    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Make some trades
        order1 = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2000.0, timestamp)
        fill1 = self.simulator.simulate_fill(order1, 2000.0, timestamp)

        order2 = self.simulator.create_order("ETH", OrderSide.SHORT, fill1.size * 2100.0, 2100.0, timestamp)
        fill2 = self.simulator.simulate_fill(order2, 2100.0, timestamp)

        # Update unrealized PnL
        self.simulator.update_unrealized_pnl("ETH", 2050.0)

        # Get summary
        summary = self.simulator.get_portfolio_summary()

        assert 'balance' in summary
        assert 'total_equity' in summary
        assert 'return_pct' in summary
        assert 'total_trades' in summary

    def test_trade_summary(self):
        """Test trade summary generation."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Make profitable trade
        order1 = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2000.0, timestamp)
        fill1 = self.simulator.simulate_fill(order1, 2000.0, timestamp)
        order2 = self.simulator.create_order("ETH", OrderSide.SHORT, fill1.size * 2100.0, 2100.0, timestamp)
        fill2 = self.simulator.simulate_fill(order2, 2100.0, timestamp)

        # Make losing trade
        order3 = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2100.0, timestamp)
        fill3 = self.simulator.simulate_fill(order3, 2100.0, timestamp)
        order4 = self.simulator.create_order("ETH", OrderSide.SHORT, fill3.size * 2000.0, 2000.0, timestamp)
        fill4 = self.simulator.simulate_fill(order4, 2000.0, timestamp)

        # Get trade summary
        summary = self.simulator.get_trade_summary()

        assert summary['total_trades'] == 4
        assert summary['winning_trades'] >= 1
        assert summary['losing_trades'] >= 1
        assert 0 <= summary['win_rate'] <= 100

    def test_risk_checks(self):
        """Test risk management checks."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Try to place order larger than balance
        huge_order = self.simulator.create_order("ETH", OrderSide.LONG, 50000.0, 2000.0, timestamp)
        fill = self.simulator.simulate_fill(huge_order, 2000.0, timestamp)

        # Should be rejected due to insufficient balance
        assert fill is None
        assert huge_order.status == OrderStatus.CANCELLED

    def test_partial_position_close(self):
        """Test partial position closing."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Open position
        order1 = self.simulator.create_order("ETH", OrderSide.LONG, 2000.0, 2000.0, timestamp)
        fill1 = self.simulator.simulate_fill(order1, 2000.0, timestamp)

        # Partially close position
        partial_size = fill1.size * 0.5
        order2 = self.simulator.create_order("ETH", OrderSide.SHORT, partial_size * 2100.0, 2100.0, timestamp)
        fill2 = self.simulator.simulate_fill(order2, 2100.0, timestamp)

        # Should still have remaining position
        position = self.simulator.positions["ETH"]
        assert position.size > 0
        assert position.size < fill1.size

    def test_position_flip(self):
        """Test position flipping from long to short."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Open long position
        order1 = self.simulator.create_order("ETH", OrderSide.LONG, 1000.0, 2000.0, timestamp)
        fill1 = self.simulator.simulate_fill(order1, 2000.0, timestamp)

        # Close and flip to short
        flip_size = fill1.size * 1.5  # 1.5x the long size
        order2 = self.simulator.create_order("ETH", OrderSide.SHORT, flip_size * 2100.0, 2100.0, timestamp)
        fill2 = self.simulator.simulate_fill(order2, 2100.0, timestamp)

        # Should now be short
        position = self.simulator.positions["ETH"]
        assert position.is_short


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])