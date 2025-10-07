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

# Import new macro context system
from engine.context.loader import load_macro_data, fetch_macro_snapshot, get_macro_health_status
from engine.context.macro_engine import analyze_macro, create_default_macro_config

# Import real domain engines (v1.7.3+)
from engine.wyckoff.wyckoff_engine import detect_wyckoff_phase
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import calculate_rsi, calculate_macd_norm, momentum_delta
from engine.smc.smc_engine import SMCEngine

# Import fast signal generation for live trading
from bin.live.fast_signals import generate_fast_signal


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

        # Initialize macro context system
        self.macro_config = create_default_macro_config()
        # Override with config if provided
        if 'context' in self.config:
            self.macro_config.update(self.config['context'])

        # Load macro data
        print("📊 Loading macro context data...")
        self.macro_data = load_macro_data()
        macro_health = get_macro_health_status(
            {k: {'value': 1.0, 'stale': len(v) == 0}
             for k, v in self.macro_data.items()}
        )
        print(f"   Macro health: {macro_health['fresh_series']}/{macro_health['total_series']} series available")

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
            # Macro context analysis (must be done first for VIX)
            current_price = df_1h['Close'].iloc[-1]
            macro_snapshot = fetch_macro_snapshot(self.macro_data, pd.Timestamp(timestamp))
            macro_result = analyze_macro(macro_snapshot, self.macro_config)

            # Get VIX values (use real VIX if available, otherwise proxy)
            if macro_snapshot.get('VIX', {}).get('value') is not None:
                vix_now = macro_snapshot['VIX']['value']
            else:
                vix_now = self._calculate_volatility_proxy(df_1h)

            # Get previous VIX (use hysteresis state if available)
            vix_prev = self.vix_hysteresis.previous_value if hasattr(self.vix_hysteresis, 'previous_value') else vix_now
            vix_active = self.vix_hysteresis.update(vix_now)

            # Generate fast signal (price action: ADX + SMA + RSI)
            fast_signal = generate_fast_signal(df_1h, df_4h, df_1d, self.config)

            # Debug: Log signal generation
            debug_trigger = (len(df_1h) == 50 or len(df_1h) % 100 == 0)
            if debug_trigger or fast_signal:
                print(f"\n🔍 Signal check at {timestamp} (bar {len(df_1h)}):")
                if fast_signal:
                    print(f"   ✓ SIGNAL: {fast_signal['side'].upper()} @ {fast_signal['confidence']:.2f} confidence")
                    print(f"   Reasons: {', '.join(fast_signal['reasons'])}")
                else:
                    print(f"   ✗ No signal (ADX < 20 or no clear setup)")

            # Check macro veto BEFORE returning signal
            if macro_result['veto_strength'] >= self.macro_config['macro_veto_threshold']:
                # Macro veto - no signal generated
                return {
                    'timestamp': timestamp.isoformat(),
                    'asset': self.asset,
                    'price': current_price,
                    'action': 'hold',
                    'side': 'neutral',
                    'confidence': 0.0,
                    'vix_active': vix_active,
                    'macro_vetoed': True,
                    'veto_reason': macro_result['notes'],
                    'macro_delta': macro_result.get('macro_delta', 0.0),
                    'macro_regime': macro_result['regime'],
                    'macro_signals': macro_result['signals']
                }

            # Convert fast signal to full signal result
            signal = fast_signal

            # Build result
            if signal:
                # Signal generated
                return {
                    'timestamp': timestamp.isoformat(),
                    'asset': self.asset,
                    'price': current_price,
                    'action': 'signal',
                    'side': signal['side'],
                    'confidence': signal['confidence'],
                    'reasons': signal.get('reasons', []),
                    'vix_active': vix_active,
                    'macro_vetoed': False,
                    'macro_delta': macro_result.get('macro_delta', 0.0),
                    'macro_regime': macro_result['regime'],
                    'macro_signals': macro_result['signals'],
                    'adx': signal.get('adx', 0.0),
                    'rsi': signal.get('rsi', 0.0),
                    'price_vs_ma20': signal.get('price_vs_ma20', 0.0)
                }
            else:
                # No signal
                return {
                    'timestamp': timestamp.isoformat(),
                    'asset': self.asset,
                    'price': current_price,
                    'action': 'hold',
                    'side': 'neutral',
                    'confidence': 0.0,
                    'vix_active': vix_active,
                    'macro_vetoed': False,
                    'macro_delta': macro_result.get('macro_delta', 0.0),
                    'macro_regime': macro_result['regime'],
                    'macro_signals': macro_result['signals']
                }

        except Exception as e:
            import traceback
            print(f"⚠️  Signal generation error: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return None

    def _analyze_wyckoff(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame):
        """Real Wyckoff phase detection using production engine."""
        if len(df_1d) < 50:
            return None

        try:
            # Standardize column names (engines expect lowercase)
            df_1d_std = df_1d.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

            # Get USDT stagnation strength from macro context (0.0 if no macro data)
            usdt_stag_strength = 0.0
            if hasattr(self, 'macro_snapshot') and self.macro_snapshot:
                usdt_stag_strength = self.macro_snapshot.get('usdt_stag_strength', 0.0)

            # Call real Wyckoff engine on daily timeframe
            wyckoff_cfg = self.config.get('wyckoff', {})
            result = detect_wyckoff_phase(df_1d_std, wyckoff_cfg, usdt_stag_strength)

            if result and result.get('confidence', 0) > 0.0:
                phase = result.get('phase', '')
                confidence = result.get('confidence', 0.0)

                # Map phases to bias
                bullish_phases = ['accumulation', 'spring', 'sos', 'ar']
                bearish_phases = ['distribution', 'utad', 'markdown', 'bc']

                if any(p in phase.lower() for p in bullish_phases):
                    return {'bias': 'long', 'phase': phase, 'confidence': confidence}
                elif any(p in phase.lower() for p in bearish_phases):
                    return {'bias': 'short', 'phase': phase, 'confidence': confidence}

            return None

        except Exception as e:
            print(f"⚠️  Wyckoff engine error: {e}")
            return None

    def _analyze_liquidity(self, df_1h: pd.DataFrame):
        """Real HOB/pHOB detection using production engine."""
        if len(df_1h) < 50:
            return None

        try:
            # Standardize column names (engines expect lowercase)
            df_1h_std = df_1h.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

            # Initialize HOB detector with config
            hob_cfg = self.config.get('liquidity', {})
            detector = HOBDetector(hob_cfg)

            # Detect HOB pattern
            hob_signal = detector.detect_hob(df_1h_std)

            if hob_signal and hob_signal.quality_score > 0.0:
                return {
                    'score': hob_signal.quality_score,
                    'pressure': hob_signal.direction,  # 'bullish' or 'bearish'
                    'confidence': hob_signal.confidence
                }

            return None

        except Exception as e:
            print(f"⚠️  Liquidity engine error: {e}")
            return None

    def _analyze_structure(self, df_1h: pd.DataFrame):
        """Real SMC analysis using production engine."""
        if len(df_1h) < 50:
            return None

        try:
            # Standardize column names (engines expect lowercase)
            df_1h_std = df_1h.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

            # Initialize SMC engine
            smc_cfg = self.config.get('smc', {})
            smc_engine = SMCEngine(smc_cfg)

            # Analyze market structure
            smc_signal = smc_engine.analyze(df_1h_std)

            if smc_signal and smc_signal.strength > 0.0:
                return {
                    'level': smc_signal.direction,  # 'long', 'short', 'neutral'
                    'strength': smc_signal.strength,
                    'confidence': smc_signal.confidence,
                    'institutional_bias': smc_signal.institutional_bias
                }

            return None

        except Exception as e:
            print(f"⚠️  Structure engine error: {e}")
            return None

    def _analyze_momentum(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame):
        """Real momentum analysis using production engine."""
        if len(df_1h) < 26:
            return None

        try:
            # Standardize column names (engines expect lowercase)
            df_1h_std = df_1h.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

            # Calculate real momentum metrics
            rsi = calculate_rsi(df_1h_std, period=14)
            macd_norm = calculate_macd_norm(df_1h_std, fast=12, slow=26, signal=9)

            # Get momentum delta
            momentum_cfg = self.config.get('momentum', {})
            delta = momentum_delta(df_1h_std, momentum_cfg)

            # Combine metrics into signal
            if rsi > 70 and macd_norm > 0:
                return {
                    'rsi': rsi,
                    'macd': macd_norm,
                    'delta': delta,
                    'signal': 'overbought'
                }
            elif rsi < 30 and macd_norm < 0:
                return {
                    'rsi': rsi,
                    'macd': macd_norm,
                    'delta': delta,
                    'signal': 'oversold'
                }
            elif abs(delta) > 0.02:  # Significant momentum shift
                return {
                    'rsi': rsi,
                    'macd': macd_norm,
                    'delta': delta,
                    'signal': 'momentum_shift'
                }

            return None

        except Exception as e:
            print(f"⚠️  Momentum engine error: {e}")
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
