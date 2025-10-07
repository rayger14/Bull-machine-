#!/usr/bin/env python3
"""
Paper Trading Mode for Bull Machine v1.7.3
WebSocket/Mock → simulate fills, PnL, health bands
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bin.live.execution_sim import ExecutionSimulator, OrderSide
from bin.live.live_mock_feed import LiveMockFeedRunner
from bull_machine_config import RESULTS_DIR, get_config_path


class PaperTradingRunner(LiveMockFeedRunner):
    """Paper trading runner with execution simulation."""

    def __init__(self, asset: str, config_path: str, initial_balance: float = 10000.0,
                 start_date: str = None, end_date: str = None, speed: float = 1.0):
        # Initialize parent
        super().__init__(asset, config_path, start_date, end_date, speed)

        # Initialize execution simulator
        self.executor = ExecutionSimulator(initial_balance)
        self.initial_balance = initial_balance

        # Risk configuration
        self.risk_config = self.config.get('risk', {})
        self.base_risk_pct = self.risk_config.get('position_size_pct', 0.075)  # 7.5% default
        self.momentum_sizing = self.risk_config.get('momentum_sizing', 0.5)  # 0.5x for momentum-only

        # Trade tracking
        self.trades = []
        self.active_signals = []

        print(f"💰 Bull Machine v1.7.3 Paper Trading - {self.asset}")
        print(f"💵 Initial Balance: ${initial_balance:,.2f}")
        print(f"⚖️  Base Risk: {self.base_risk_pct*100:.1f}%")

    def run(self):
        """Execute paper trading with realistic execution simulation."""
        print("\n💰 Starting Paper Trading...")

        try:
            # Stream data and execute trades
            stream = self.adapter.stream_csv(
                self.asset, "1H", self.start_date, self.end_date, self.speed
            )

            # Load higher timeframes
            stream_4h = self.adapter.stream_csv(
                self.asset, "4H", self.start_date, self.end_date, self.speed
            )
            stream_1d = self.adapter.stream_csv(
                self.asset, "1D", self.start_date, self.end_date, self.speed
            )

            # Convert to lists for processing
            all_1h = list(stream)
            all_4h = list(stream_4h)
            all_1d = list(stream_1d)

            print("📊 Paper Trading Data Loaded:")
            print(f"   1H bars: {len(all_1h)}")
            print(f"   4H bars: {len(all_4h)}")
            print(f"   1D bars: {len(all_1d)}")

            # Process each tick for paper trading
            for i, tick_1h in enumerate(all_1h):
                current_time = tick_1h['timestamp']
                current_price = tick_1h['Close']

                # Update data structures
                self.df_1h = self.adapter.update_ohlcv(self.df_1h, tick_1h, max_bars=500)
                self._update_higher_timeframes(current_time, all_4h, all_1d)

                # Update unrealized PnL
                self.executor.update_unrealized_pnl(self.asset, current_price)

                # Process any pending orders
                self._process_pending_orders(current_price, current_time)

                # Generate new signals if sufficient data
                if len(self.df_1h) >= 50:
                    self._process_paper_signal(current_time, current_price)

                # Print progress
                if i % 100 == 0:
                    portfolio = self.executor.get_portfolio_summary()
                    print(f"   Progress: {i+1}/{len(all_1h)} | Balance: ${portfolio['total_equity']:,.2f} | P&L: {portfolio['return_pct']:+.1f}%")

            print("\n✅ Paper Trading Complete")
            self._save_trading_results()
            self._print_trading_summary()

        except Exception as e:
            print(f"❌ Paper Trading Error: {e}")
            raise

    def _process_paper_signal(self, timestamp: datetime, current_price: float):
        """Process signal and execute paper trades."""
        # Right-edge align timeframes
        df_1h_aligned, df_4h_aligned, df_1d_aligned = self.adapter.align_mtf(
            self.df_1h, self.df_4h, self.df_1d
        )

        # Generate signal
        signal_result = self._generate_signal(
            df_1h_aligned, df_4h_aligned, df_1d_aligned, timestamp
        )

        if not signal_result or signal_result.get('action') != 'signal':
            return

        # Extract signal details
        side = signal_result.get('side', 'neutral')
        confidence = signal_result.get('confidence', 0)

        if side in ['long', 'short'] and confidence > 0.3:  # Minimum confidence threshold
            # Calculate position size based on signal type
            position_size_usd = self._calculate_position_size(signal_result, current_price)

            if position_size_usd > 0:
                # Create and execute order
                order_side = OrderSide.LONG if side == 'long' else OrderSide.SHORT
                order = self.executor.create_order(
                    symbol=self.asset,
                    side=order_side,
                    size_usd=position_size_usd,
                    current_price=current_price,
                    timestamp=timestamp
                )

                # Simulate fill
                fill = self.executor.simulate_fill(order, current_price, timestamp)

                if fill:
                    # Record trade
                    trade_record = {
                        'timestamp': timestamp.isoformat(),
                        'asset': self.asset,
                        'side': side,
                        'size': fill.size,
                        'price': fill.price,
                        'size_usd': fill.size * fill.price,
                        'fees': fill.fees,
                        'confidence': confidence,
                        'signal_reasons': signal_result.get('reasons', []),
                        'order_id': fill.order_id
                    }

                    self.trades.append(trade_record)
                    print(f"   💰 Paper Trade: {side} {fill.size:.4f} {self.asset} @ ${fill.price:.2f} (fees: ${fill.fees:.2f})")

        # Record signal for analysis
        self.signals.append(signal_result)
        self.signal_count += 1

        # Update health monitoring
        self._update_health_monitoring(signal_result, timestamp)

    def _calculate_position_size(self, signal_result: dict, current_price: float) -> float:
        """Calculate position size based on signal and risk parameters."""
        portfolio = self.executor.get_portfolio_summary()
        available_equity = portfolio['total_equity']

        # Base risk percentage
        base_risk = self.base_risk_pct

        # Adjust based on signal type and modules
        modules = signal_result.get('modules', {})
        confidence = signal_result.get('confidence', 0)

        # Check if momentum-only signal (reduce size)
        if len(modules) == 1 and 'momentum' in modules:
            risk_multiplier = self.momentum_sizing
        else:
            risk_multiplier = 1.0

        # Confidence-based adjustment
        confidence_multiplier = min(1.0, confidence / 0.6)  # Scale from 0.6 to 1.0

        # Calculate final position size
        final_risk_pct = base_risk * risk_multiplier * confidence_multiplier
        position_size_usd = available_equity * final_risk_pct

        # Minimum position size check
        min_size_usd = 100.0  # $100 minimum
        if position_size_usd < min_size_usd:
            return 0.0

        return position_size_usd

    def _process_pending_orders(self, current_price: float, timestamp: datetime):
        """Process any pending orders (for future stop-loss/take-profit implementation)."""
        # Currently simple market orders only
        # Future: implement stop-loss and take-profit orders
        pass

    def _save_trading_results(self):
        """Save paper trading results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save trades
        trades_file = RESULTS_DIR / f"live_trades_{self.asset}_{timestamp}.jsonl"
        with open(trades_file, 'w') as f:
            for trade in self.trades:
                f.write(json.dumps(trade) + '\n')

        # Save signals
        signals_file = RESULTS_DIR / f"live_signals_{self.asset}_{timestamp}.jsonl"
        with open(signals_file, 'w') as f:
            for signal in self.signals:
                f.write(json.dumps(signal) + '\n')

        # Save portfolio summary
        portfolio_file = RESULTS_DIR / f"portfolio_summary_{self.asset}_{timestamp}.json"
        portfolio_summary = self.executor.get_portfolio_summary()
        trade_summary = self.executor.get_trade_summary()

        combined_summary = {
            'paper_trading_summary': {
                'asset': self.asset,
                'initial_balance': self.initial_balance,
                'period': f"{self.start_date} to {self.end_date}",
                'portfolio': portfolio_summary,
                'trading': trade_summary,
                'health': self.health_monitor.get_health_summary()
            }
        }

        with open(portfolio_file, 'w') as f:
            json.dump(combined_summary, f, indent=2, default=str)

        print("📁 Paper Trading Results Saved:")
        print(f"   Trades: {trades_file}")
        print(f"   Signals: {signals_file}")
        print(f"   Summary: {portfolio_file}")

    def _print_trading_summary(self):
        """Print comprehensive paper trading summary."""
        portfolio = self.executor.get_portfolio_summary()
        trading = self.executor.get_trade_summary()

        print("\n📊 Paper Trading Summary:")
        print(f"   Asset: {self.asset}")
        print(f"   Period: {self.start_date} → {self.end_date}")

        print("\n💰 Portfolio Performance:")
        print(f"   Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   Final Equity: ${portfolio['total_equity']:,.2f}")
        print(f"   Total Return: {portfolio['return_pct']:+.2f}%")
        print(f"   Realized P&L: ${portfolio['realized_pnl']:+,.2f}")
        print(f"   Unrealized P&L: ${portfolio['unrealized_pnl']:+,.2f}")

        print("\n📈 Trading Statistics:")
        print(f"   Total Trades: {trading['total_trades']}")
        print(f"   Win Rate: {trading['win_rate']:.1f}%")
        print(f"   Average Win: ${trading['avg_win']:+,.2f}")
        print(f"   Average Loss: ${trading['avg_loss']:+,.2f}")
        print(f"   Profit Factor: {trading['profit_factor']:.2f}")
        print(f"   Total Fees: ${trading['total_fees']:,.2f}")

        # Open positions
        if portfolio['positions']:
            print("\n🔄 Open Positions:")
            for symbol, pos in portfolio['positions'].items():
                print(f"   {symbol}: {pos['size']:+.4f} @ ${pos['avg_price']:.2f} (P&L: ${pos['unrealized_pnl']:+,.2f})")

        # Health summary
        health = self.health_monitor.get_health_summary()
        print(f"\n🏥 Health Status: {health.get('status', 'unknown')}")
        if 'current_metrics' in health:
            metrics = health['current_metrics']
            print(f"   Macro Veto Rate: {metrics.get('macro_veto_rate', 0):.1f}%")
            print(f"   SMC 2+ Hit Rate: {metrics.get('smc_2hit_rate', 0):.1f}%")


def main():
    """Main entry point for paper trading runner."""
    parser = argparse.ArgumentParser(description="Bull Machine v1.7.3 Paper Trading Runner")
    parser.add_argument("--asset", choices=["ETH", "BTC", "SOL"], required=True,
                       help="Asset to paper trade")
    parser.add_argument("--start", required=True,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Initial balance (default: $10,000)")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Replay speed multiplier")
    parser.add_argument("--config",
                       help="Custom config path (default: auto-select)")

    args = parser.parse_args()

    # Use provided config or auto-select
    if args.config:
        config_path = args.config
    else:
        config_path = get_config_path(args.asset)

    print(f"📋 Using config: {config_path}")

    # Run paper trading
    runner = PaperTradingRunner(
        asset=args.asset,
        config_path=config_path,
        initial_balance=args.balance,
        start_date=args.start,
        end_date=args.end,
        speed=args.speed
    )

    runner.run()


if __name__ == "__main__":
    main()
