#!/usr/bin/env python3
"""
Live Mock Feed Runner for Bull Machine v1.7.3
CSV replay using existing MTF + Advanced Fusion system
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bin.live.adapters import LiveDataAdapter
from bin.live.health_monitor import HealthMonitor, VIXHysteresis
from bull_machine.modules.fusion.enhanced import EnhancedFusionEngineV1_4
from bull_machine_config import RESULTS_DIR, get_config_path

# Import existing engines
from engine.timeframes.mtf_alignment import MTFAlignmentEngine


class LiveMockFeedRunner:
    """Production-grade mock feed runner using existing Bull Machine engines."""

    def __init__(self, asset: str, config_path: str, start_date: str = None,
                 end_date: str = None, speed: float = 1.0):
        self.asset = asset.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.speed = speed

        print(f"🚀 Bull Machine v1.7.3 Mock Feed - {self.asset}")
        print(f"📅 Period: {start_date} → {end_date}")
        print(f"⚡ Speed: {speed}x")

        # Initialize components
        self.adapter = LiveDataAdapter()
        self.health_monitor = HealthMonitor()
        self.vix_hysteresis = VIXHysteresis(on_threshold=22.0, off_threshold=18.0)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize engines using existing production code
        self.mtf_engine = MTFAlignmentEngine(self.config.get('mtf', {}))
        self.fusion_engine = EnhancedFusionEngineV1_4(self.config.get('fusion', {}))

        # Initialize data containers (right-edge aligned)
        self.df_1h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_4h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_1d = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Results tracking
        self.signals = []
        self.signal_count = 0

        # Ensure results directory exists
        RESULTS_DIR.mkdir(exist_ok=True)

    def run(self):
        """Execute mock feed replay with existing engines."""
        print("\n🔄 Starting Mock Feed Replay...")

        try:
            # Stream 1H data as primary timeframe
            stream = self.adapter.stream_csv(
                self.asset, "1H", self.start_date, self.end_date, self.speed
            )

            # Also load 4H and 1D data for MTF
            stream_4h = self.adapter.stream_csv(
                self.asset, "4H", self.start_date, self.end_date, self.speed
            )
            stream_1d = self.adapter.stream_csv(
                self.asset, "1D", self.start_date, self.end_date, self.speed
            )

            # Convert streams to dataframes for batch processing
            print("📊 Loading multi-timeframe data...")

            # Load all data first (in production, this would be real-time)
            all_1h = list(stream)
            all_4h = list(stream_4h)
            all_1d = list(stream_1d)

            print(f"   1H bars: {len(all_1h)}")
            print(f"   4H bars: {len(all_4h)}")
            print(f"   1D bars: {len(all_1d)}")

            # Process each 1H tick
            for i, tick_1h in enumerate(all_1h):
                if i % 50 == 0:
                    print(f"   Processing: {tick_1h['timestamp']} ({i+1}/{len(all_1h)})")

                # Update 1H data
                self.df_1h = self.adapter.update_ohlcv(self.df_1h, tick_1h, max_bars=500)

                # Update 4H and 1D when appropriate
                current_time = tick_1h['timestamp']
                self._update_higher_timeframes(current_time, all_4h, all_1d)

                # Ensure minimum data for analysis
                if len(self.df_1h) < 50:
                    continue

                # Right-edge align timeframes
                df_1h_aligned, df_4h_aligned, df_1d_aligned = self.adapter.align_mtf(
                    self.df_1h, self.df_4h, self.df_1d
                )

                # Generate signal using existing engines
                signal_result = self._generate_signal(
                    df_1h_aligned, df_4h_aligned, df_1d_aligned, current_time
                )

                if signal_result:
                    self.signals.append(signal_result)
                    self.signal_count += 1

                    # Health monitoring
                    self._update_health_monitoring(signal_result, current_time)

                    # Log signal
                    if signal_result.get('action') != 'hold':
                        side = signal_result.get('side', 'neutral')
                        confidence = signal_result.get('confidence', 0)
                        print(f"   📈 Signal: {side} (confidence: {confidence:.2f})")

            print(f"\n✅ Mock Feed Complete - {self.signal_count} signals generated")
            self._save_results()
            self._print_summary()

        except Exception as e:
            print(f"❌ Mock Feed Error: {e}")
            raise

    def _update_higher_timeframes(self, current_time: datetime, all_4h: list, all_1d: list):
        """Update 4H and 1D data when new bars are available."""
        # Find matching 4H bars
        for tick_4h in all_4h:
            if (tick_4h['timestamp'] <= current_time and
                (len(self.df_4h) == 0 or tick_4h['timestamp'] > self.df_4h.index[-1])):
                self.df_4h = self.adapter.update_ohlcv(self.df_4h, tick_4h, max_bars=200)

        # Find matching 1D bars
        for tick_1d in all_1d:
            if (tick_1d['timestamp'] <= current_time and
                (len(self.df_1d) == 0 or tick_1d['timestamp'] > self.df_1d.index[-1])):
                self.df_1d = self.adapter.update_ohlcv(self.df_1d, tick_1d, max_bars=100)

    def _generate_signal(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                        df_1d: pd.DataFrame, timestamp: datetime) -> dict:
        """Generate signal using existing MTF + Fusion engines."""
        try:
            # Prepare modules dict (simplified for mock)
            modules = {
                'wyckoff': self._analyze_wyckoff(df_1h, df_4h, df_1d),
                'liquidity': self._analyze_liquidity(df_1h),
                'structure': self._analyze_structure(df_1h),
                'momentum': self._analyze_momentum(df_1h, df_4h),
                'context': self._analyze_context(df_1d)
            }

            # MTF confluence analysis
            sync_report = self.mtf_engine.mtf_confluence(df_1h, df_4h, df_1d)

            # Apply VIX hysteresis (mock VIX with volatility proxy)
            current_price = df_1h['Close'].iloc[-1]
            volatility_proxy = self._calculate_volatility_proxy(df_1h)
            vix_active = self.vix_hysteresis.update(volatility_proxy)

            # Generate signal via existing fusion engine
            signal = self.fusion_engine.fuse_with_mtf(modules, sync_report)

            # Build comprehensive result
            signal_result = {
                'timestamp': timestamp.isoformat(),
                'asset': self.asset,
                'price': current_price,
                'action': 'hold',
                'side': 'neutral',
                'confidence': 0.0,
                'vix_active': vix_active,
                'modules': {k: bool(v) for k, v in modules.items() if v},
                'sync_report': {
                    'nested_ok': getattr(sync_report, 'nested_ok', False),
                    'eq_magnet': getattr(sync_report, 'eq_magnet', False),
                    'decision': getattr(sync_report, 'decision', 'hold')
                }
            }

            # Update with signal if generated
            if signal:
                signal_result.update({
                    'action': 'signal',
                    'side': getattr(signal, 'side', 'neutral'),
                    'confidence': getattr(signal, 'confidence', 0.0),
                    'reasons': getattr(signal, 'reasons', []),
                    'ttl_bars': getattr(signal, 'ttl_bars', 0)
                })

            return signal_result

        except Exception as e:
            print(f"⚠️  Signal generation error: {e}")
            return None

    def _analyze_wyckoff(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame):
        """Simplified Wyckoff analysis for mock."""
        if len(df_1h) < 20:
            return None

        # Simple trend + pullback detection
        close = df_1h['Close'].iloc[-1]
        ma20 = df_1h['Close'].rolling(20).mean().iloc[-1]
        ma50 = df_1h['Close'].rolling(50).mean().iloc[-1] if len(df_1h) >= 50 else ma20

        if close > ma20 > ma50:
            return {'bias': 'long', 'phase': 'D', 'confidence': 0.7}
        elif close < ma20 < ma50:
            return {'bias': 'short', 'phase': 'D', 'confidence': 0.7}

        return None

    def _analyze_liquidity(self, df_1h: pd.DataFrame):
        """Simplified liquidity analysis for mock."""
        if len(df_1h) < 10:
            return None

        # Simple wick analysis
        recent = df_1h.tail(10)
        avg_wick = (recent['High'] - recent['Close']).mean()
        current_wick = df_1h['High'].iloc[-1] - df_1h['Close'].iloc[-1]

        if current_wick > avg_wick * 1.5:
            return {'score': 0.6, 'pressure': 'bearish'}

        return None

    def _analyze_structure(self, df_1h: pd.DataFrame):
        """Simplified structure analysis for mock."""
        if len(df_1h) < 20:
            return None

        # Simple support/resistance
        high_20 = df_1h['High'].rolling(20).max().iloc[-1]
        low_20 = df_1h['Low'].rolling(20).min().iloc[-1]
        close = df_1h['Close'].iloc[-1]

        if close >= high_20 * 0.98:  # Near resistance
            return {'level': 'resistance', 'strength': 0.6}
        elif close <= low_20 * 1.02:  # Near support
            return {'level': 'support', 'strength': 0.6}

        return None

    def _analyze_momentum(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame):
        """Simplified momentum analysis for mock."""
        if len(df_1h) < 14:
            return None

        # Simple RSI
        close = df_1h['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]
        if current_rsi > 70:
            return {'rsi': current_rsi, 'signal': 'overbought'}
        elif current_rsi < 30:
            return {'rsi': current_rsi, 'signal': 'oversold'}

        return None

    def _analyze_context(self, df_1d: pd.DataFrame):
        """Simplified context analysis for mock."""
        if len(df_1d) < 5:
            return None

        # Simple trend context
        close = df_1d['Close'].iloc[-1]
        ma5 = df_1d['Close'].rolling(5).mean().iloc[-1]

        return {'trend': 'up' if close > ma5 else 'down', 'strength': 0.5}

    def _calculate_volatility_proxy(self, df_1h: pd.DataFrame) -> float:
        """Calculate volatility proxy for VIX simulation."""
        if len(df_1h) < 20:
            return 15.0

        # Use 20-period volatility as VIX proxy
        returns = df_1h['Close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(24 * 365) * 100

        return volatility if not pd.isna(volatility) else 15.0

    def _update_health_monitoring(self, signal_result: dict, timestamp: datetime):
        """Update health monitoring with signal result."""
        # Mock domain activity for health tracking
        domains_active = {
            'wyckoff': 'wyckoff' in signal_result.get('modules', {}),
            'liquidity': 'liquidity' in signal_result.get('modules', {}),
            'structure': 'structure' in signal_result.get('modules', {}),
            'momentum': 'momentum' in signal_result.get('modules', {}),
            'hob': False,  # Not implemented in mock
            'orderflow': False  # Not implemented in mock
        }

        # Record signal for health tracking
        self.health_monitor.record_signal(signal_result, domains_active, timestamp)

        # Log health metrics periodically
        if self.signal_count % 25 == 0:
            metrics = self.health_monitor.get_current_metrics(timestamp)
            self.health_monitor.log_health(metrics)

    def _save_results(self):
        """Save signals and health data to results directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save signals
        signals_file = RESULTS_DIR / f"live_signals_{self.asset}_{timestamp}.jsonl"
        with open(signals_file, 'w') as f:
            for signal in self.signals:
                f.write(json.dumps(signal) + '\n')

        # Save health summary
        health_file = RESULTS_DIR / f"health_summary_{self.asset}_{timestamp}.json"
        health_summary = self.health_monitor.get_health_summary()
        with open(health_file, 'w') as f:
            json.dump(health_summary, f, indent=2)

        print("📁 Results saved:")
        print(f"   Signals: {signals_file}")
        print(f"   Health:  {health_file}")

    def _print_summary(self):
        """Print execution summary."""
        print("\n📊 Mock Feed Summary:")
        print(f"   Asset: {self.asset}")
        print(f"   Total Signals: {len(self.signals)}")

        # Count signal types
        signal_signals = [s for s in self.signals if s.get('action') == 'signal']
        long_signals = [s for s in signal_signals if s.get('side') == 'long']
        short_signals = [s for s in signal_signals if s.get('side') == 'short']

        print("   Signal Breakdown:")
        print(f"     Long: {len(long_signals)}")
        print(f"     Short: {len(short_signals)}")
        print(f"     Hold: {len(self.signals) - len(signal_signals)}")

        # Health summary
        health_summary = self.health_monitor.get_health_summary()
        print(f"   Health Status: {health_summary.get('status', 'unknown')}")

        if 'current_metrics' in health_summary:
            metrics = health_summary['current_metrics']
            print("   Health Metrics:")
            print(f"     Macro Veto Rate: {metrics.get('macro_veto_rate', 0):.1f}%")
            print(f"     SMC 2+ Hit Rate: {metrics.get('smc_2hit_rate', 0):.1f}%")


def main():
    """Main entry point for mock feed runner."""
    parser = argparse.ArgumentParser(description="Bull Machine v1.7.3 Mock Feed Runner")
    parser.add_argument("--asset", choices=["ETH", "BTC", "SOL"], required=True,
                       help="Asset to run mock feed for")
    parser.add_argument("--start", required=True,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True,
                       help="End date (YYYY-MM-DD)")
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

    # Run mock feed
    runner = LiveMockFeedRunner(
        asset=args.asset,
        config_path=config_path,
        start_date=args.start,
        end_date=args.end,
        speed=args.speed
    )

    runner.run()


if __name__ == "__main__":
    main()
