#!/usr/bin/env python3
"""
Shadow Live Mode for Bull Machine v1.7.3
WebSocket placeholder adapter → logs signals (no orders)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bin.live.adapters import WebSocketAdapter
from bin.live.live_mock_feed import LiveMockFeedRunner
from bull_machine_config import get_config_path


class ShadowLiveRunner(LiveMockFeedRunner):
    """Shadow live runner with WebSocket placeholder (no real network)."""

    def __init__(self, asset: str, config_path: str, duration_hours: int = 24):
        # Initialize parent with mock data for demonstration
        super().__init__(
            asset=asset,
            config_path=config_path,
            start_date="2025-06-01",
            end_date="2025-06-02",
            speed=1.0
        )

        self.duration_hours = duration_hours
        self.ws_adapter = WebSocketAdapter()

        print(f"📡 Bull Machine v1.7.3 Shadow Live - {self.asset}")
        print(f"⏱️  Duration: {duration_hours} hours")
        print("🔍 Mode: Signal logging only (NO ORDERS)")

    def run_shadow(self):
        """Run shadow mode with WebSocket placeholder."""
        print("\n📡 Starting Shadow Live Mode...")

        try:
            # Connect to placeholder WebSocket
            if not self.ws_adapter.connect(f"{self.asset}USD"):
                print("❌ Failed to connect to WebSocket adapter")
                return

            print(f"✅ Connected to {self.asset} WebSocket (placeholder)")

            # Shadow mode simulation
            start_time = datetime.now()
            tick_count = 0

            while True:
                current_time = datetime.now()
                elapsed_hours = (current_time - start_time).total_seconds() / 3600

                # Check duration limit
                if elapsed_hours >= self.duration_hours:
                    print(f"\n⏰ Duration limit reached ({self.duration_hours}h)")
                    break

                # Simulate WebSocket tick (in real implementation, this would be actual WS data)
                tick = self._simulate_websocket_tick(current_time)

                if tick:
                    # Process tick using same pipeline as mock feed
                    self._process_shadow_tick(tick, current_time)
                    tick_count += 1

                    if tick_count % 100 == 0:
                        print(f"   Processed {tick_count} ticks ({elapsed_hours:.1f}h elapsed)")

                # Shadow mode runs in real-time (or accelerated for demo)
                time.sleep(0.1)  # 100ms intervals for demo

            print(f"\n✅ Shadow Live Complete - {tick_count} ticks processed")
            self._save_results()
            self._print_summary()

        except KeyboardInterrupt:
            print("\n⚠️  Shadow Live interrupted by user")
            self._save_results()

        except Exception as e:
            print(f"❌ Shadow Live Error: {e}")
            raise

        finally:
            self.ws_adapter.disconnect()
            print("📡 WebSocket disconnected")

    def _simulate_websocket_tick(self, timestamp: datetime) -> dict:
        """
        Simulate WebSocket tick for demonstration.
        In production, this would parse real WebSocket events.
        """
        # For demo, use mock data with slight price variation
        base_price = 2000.0 if self.asset == "ETH" else 30000.0  # Rough price levels

        # Simulate price movement
        price_noise = (hash(str(timestamp)) % 1000) / 10000  # -0.05 to +0.05
        current_price = base_price * (1 + price_noise)

        return {
            'timestamp': timestamp,
            'Open': current_price * 0.999,
            'High': current_price * 1.001,
            'Low': current_price * 0.998,
            'Close': current_price,
            'Volume': 1000.0
        }

    def _process_shadow_tick(self, tick: dict, timestamp: datetime):
        """Process WebSocket tick using existing signal pipeline."""
        # Update internal data structures
        self.df_1h = self.adapter.update_ohlcv(self.df_1h, tick, max_bars=500)

        # Generate signal every Nth tick (simulating 1H intervals)
        if len(self.df_1h) % 60 == 0 and len(self.df_1h) >= 50:  # Every 60 ticks ~ 1H
            # Use same signal generation as mock feed
            signal_result = self._generate_signal(
                self.df_1h, self.df_4h, self.df_1d, timestamp
            )

            if signal_result:
                # Log signal (NO ORDER PLACEMENT)
                self._log_shadow_signal(signal_result)

                if signal_result.get('action') == 'signal':
                    side = signal_result.get('side', 'neutral')
                    confidence = signal_result.get('confidence', 0)
                    price = signal_result.get('price', 0)
                    print(f"   🔍 Shadow Signal: {side} @ {price:.2f} (conf: {confidence:.2f}) [LOGGED ONLY]")

    def _log_shadow_signal(self, signal_result: dict):
        """Log shadow signal for analysis."""
        # Add shadow mode metadata
        signal_result['mode'] = 'shadow'
        signal_result['orders_placed'] = False
        signal_result['logged_only'] = True

        self.signals.append(signal_result)
        self.signal_count += 1

        # Update health monitoring
        self._update_health_monitoring(signal_result, datetime.fromisoformat(signal_result['timestamp']))

    def run(self):
        """Override parent run method for shadow mode."""
        self.run_shadow()


def main():
    """Main entry point for shadow live runner."""
    parser = argparse.ArgumentParser(description="Bull Machine v1.7.3 Shadow Live Runner")
    parser.add_argument("--asset", choices=["ETH", "BTC", "SOL"], required=True,
                       help="Asset to run shadow live for")
    parser.add_argument("--duration", type=int, default=1,
                       help="Duration in hours (default: 1)")
    parser.add_argument("--config",
                       help="Custom config path (default: auto-select)")

    args = parser.parse_args()

    # Use provided config or auto-select
    if args.config:
        config_path = args.config
    else:
        config_path = get_config_path(args.asset)

    print(f"📋 Using config: {config_path}")

    # Run shadow live
    runner = ShadowLiveRunner(
        asset=args.asset,
        config_path=config_path,
        duration_hours=args.duration
    )

    runner.run()


if __name__ == "__main__":
    main()
