"""
v1.8 Hybrid Runner: Fast Signals + Periodic Fusion Validation

Combines fast signal generation (bin/live/fast_signals.py) with periodic
full domain engine validation for optimal speed + precision balance.

Architecture:
- Every 1H bar: Generate fast signal (ADX + SMA + RSI)
- Every 4H bar: Run full fusion validation (Wyckoff + HOB + Momentum + SMC)
- Macro veto: Applied first, before any signal generation
- Safety guards: Loss streak cool-off, ATR throttle

Modes:
- advisory: Log fusion, execute fast (speed test)
- prefilter: Fusion must pass, fast executes (balanced)
- execute_only_if_fusion_confirms: Both must agree (conservative)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib

# Bull Machine imports
from engine.io.tradingview_loader import load_tv
from engine.timeframes.mtf_alignment import MTFAlignmentEngine
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.fusion.enhanced_v1_4 import EnhancedFusionEngineV1_4
from bin.live.fast_signals import generate_fast_signal
from bin.live.live_data_adapter import LiveDataAdapter


class HybridRunner:
    """v1.8 Hybrid runner with fast signals + periodic fusion validation."""

    def __init__(self, asset: str, config_path: str, start_date: str = None, end_date: str = None):
        self.asset = asset.upper()
        self.start_date = start_date
        self.end_date = end_date

        print(f"🚀 Bull Machine v1.8 Hybrid - {self.asset}")
        print(f"📅 Period: {start_date} → {end_date}")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Validate v1.8 config
        self._validate_config()

        # Initialize components
        self.adapter = LiveDataAdapter()
        self.mtf_engine = MTFAlignmentEngine(self.config.get('mtf', {}))
        self.fusion_engine = EnhancedFusionEngineV1_4(self.config.get('fusion', {}))

        # Load macro data
        print("📊 Loading macro context data...")
        self.macro_data = load_macro_data()
        self.macro_config = create_default_macro_config()
        if 'context' in self.config:
            self.macro_config.update(self.config['context'])

        # Initialize data containers
        self.df_1h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_4h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_1d = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # State tracking
        self.signals = []
        self.last_4h_bar = None
        self.recent_trades = []  # For loss streak tracking
        self.config_hash = self._compute_config_hash()

    def _validate_config(self):
        """Validate v1.8 hybrid configuration."""
        required = {
            'fast_signals': ['enabled', 'mode', 'min_confidence'],
            'fusion': ['entry_threshold_confidence', 'full_engine_validation_interval_bars'],
            'safety': ['loss_streak_threshold', 'atr_floor_percentile', 'atr_cap_percentile']
        }

        for section, keys in required.items():
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")
            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing config key: {section}.{key}")

        # Validate mode
        valid_modes = ['advisory', 'prefilter', 'execute_only_if_fusion_confirms']
        if self.config['fast_signals']['mode'] not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")

    def _compute_config_hash(self) -> str:
        """Compute hash of config for determinism validation."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    def run(self):
        """Execute hybrid paper trading with fast signals + periodic fusion."""
        print("\n💰 Starting Hybrid Paper Trading...")

        # Load data streams
        stream_1h = self.adapter.stream_csv(self.asset, "1H", self.start_date, self.end_date)
        stream_4h = self.adapter.stream_csv(self.asset, "4H", self.start_date, self.end_date)
        stream_1d = self.adapter.stream_csv(self.asset, "1D", self.start_date, self.end_date)

        all_1h = list(stream_1h)
        all_4h = list(stream_4h)
        all_1d = list(stream_1d)

        print(f"📊 Data loaded: {len(all_1h)} 1H bars, {len(all_4h)} 4H bars, {len(all_1d)} 1D bars")

        # Process each 1H bar
        for i, tick_1h in enumerate(all_1h):
            current_time = tick_1h['timestamp']
            current_price = tick_1h['Close']

            # Update data structures
            self.df_1h = self.adapter.update_ohlcv(self.df_1h, tick_1h, max_bars=500)
            self._update_higher_timeframes(current_time, all_4h, all_1d)

            # Need minimum data
            if len(self.df_1h) < 50 or len(self.df_4h) < 14 or len(self.df_1d) < 50:
                continue

            # Align timeframes
            df_1h_aligned, df_4h_aligned, df_1d_aligned = self.adapter.align_mtf(
                self.df_1h, self.df_4h, self.df_1d
            )

            # Generate signal
            signal_result = self._generate_hybrid_signal(
                df_1h_aligned, df_4h_aligned, df_1d_aligned, current_time, current_price
            )

            if signal_result:
                self.signals.append(signal_result)

                # Log signal
                self._log_signal(signal_result)

            # Progress
            if i % 100 == 0:
                print(f"   Progress: {i+1}/{len(all_1h)} | Signals: {len(self.signals)}")

        print(f"\n✅ Hybrid run complete: {len(self.signals)} signals generated")
        return self.signals

    def _generate_hybrid_signal(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                                 df_1d: pd.DataFrame, timestamp: datetime,
                                 current_price: float) -> Optional[Dict]:
        """
        Generate hybrid signal with fast signals + periodic fusion validation.

        Flow:
        1. Check macro veto (first, always)
        2. Apply safety guards (loss streak, ATR throttle)
        3. Generate fast signal
        4. If on 4H boundary, run full fusion validation
        5. Apply execution mode logic
        6. Return signal or None
        """

        # 1. MACRO VETO (first check)
        macro_snapshot = fetch_macro_snapshot(self.macro_data, timestamp)
        macro_result = analyze_macro(macro_snapshot, self.macro_config)

        if macro_result['veto_strength'] >= self.macro_config['macro_veto_threshold']:
            return {
                'timestamp': timestamp.isoformat(),
                'asset': self.asset,
                'price': current_price,
                'action': 'hold',
                'macro_vetoed': True,
                'veto_reason': macro_result['notes'],
                'config_hash': self.config_hash
            }

        # 2. SAFETY GUARDS
        # Loss streak cool-off
        loss_streak = self._count_consecutive_losses()
        require_fusion = loss_streak >= self.config['safety']['loss_streak_threshold']

        # ATR throttle
        atr_ok = self._check_atr_throttle(df_1h)
        if not atr_ok:
            return None  # Too quiet or too volatile

        # 3. GENERATE FAST SIGNAL
        fast_signal = None
        if self.config['fast_signals']['enabled']:
            fast_signal = generate_fast_signal(df_1h, df_4h, df_1d, self.config)

            # Check minimum confidence
            if fast_signal and fast_signal['confidence'] < self.config['fast_signals']['min_confidence']:
                fast_signal = None

        # 4. PERIODIC FUSION VALIDATION
        fusion_signal = None
        current_4h_bar = self._get_4h_bar_id(timestamp)

        # Run fusion if:
        # - On 4H bar boundary OR
        # - Loss streak requires fusion confirm OR
        # - Mode requires fusion
        run_fusion = (
            current_4h_bar != self.last_4h_bar or
            require_fusion or
            self.config['fast_signals']['mode'] in ['prefilter', 'execute_only_if_fusion_confirms']
        )

        if run_fusion:
            fusion_signal = self._run_full_fusion(df_1h, df_4h, df_1d, timestamp, current_price)
            self.last_4h_bar = current_4h_bar

        # 5. APPLY EXECUTION MODE
        execute_signal = self._apply_execution_mode(fast_signal, fusion_signal, require_fusion)

        if not execute_signal:
            return None

        # 6. RETURN SIGNAL
        return {
            'timestamp': timestamp.isoformat(),
            'asset': self.asset,
            'price': current_price,
            'action': 'signal',
            'side': execute_signal['side'],
            'confidence': execute_signal['confidence'],
            'reasons': execute_signal.get('reasons', []),
            'fast_signal': fast_signal is not None,
            'fusion_signal': fusion_signal is not None,
            'fusion_validated': fusion_signal is not None and current_4h_bar != self.last_4h_bar,
            'mode': self.config['fast_signals']['mode'],
            'loss_streak_override': require_fusion,
            'macro_vetoed': False,
            'config_hash': self.config_hash
        }

    def _run_full_fusion(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                         df_1d: pd.DataFrame, timestamp: datetime,
                         current_price: float) -> Optional[Dict]:
        """Run full fusion engine with all domain modules."""
        # This would call the real domain engines
        # For now, return None (will implement when real engines are optimized)
        # TODO: Implement full fusion validation
        return None

    def _apply_execution_mode(self, fast_signal: Optional[Dict],
                              fusion_signal: Optional[Dict],
                              require_fusion: bool) -> Optional[Dict]:
        """Apply execution mode logic to determine which signal to execute."""

        # Loss streak override: require fusion confirm
        if require_fusion:
            if not fusion_signal:
                return None
            return fusion_signal

        mode = self.config['fast_signals']['mode']

        if mode == 'advisory':
            # Log fusion for analysis, execute fast
            return fast_signal

        elif mode == 'prefilter':
            # Fusion must pass, fast executes
            if fusion_signal and fast_signal:
                return fast_signal
            return None

        elif mode == 'execute_only_if_fusion_confirms':
            # Both must agree
            if not (fast_signal and fusion_signal):
                return None

            if fast_signal['side'] == fusion_signal['side']:
                return fusion_signal  # Use fusion confidence
            return None

    def _count_consecutive_losses(self) -> int:
        """Count consecutive losses in last 24 hours."""
        if not self.recent_trades:
            return 0

        count = 0
        for trade in reversed(self.recent_trades[-10:]):  # Check last 10 trades
            if trade.get('pnl', 0) < 0:
                count += 1
            else:
                break
        return count

    def _check_atr_throttle(self, df_1h: pd.DataFrame) -> bool:
        """Check if ATR is within acceptable range."""
        if len(df_1h) < 100:
            return True  # Not enough data, allow

        # Calculate ATR
        high = df_1h['High']
        low = df_1h['Low']
        close = df_1h['Close']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        current_atr = atr.iloc[-1]

        # Calculate percentile
        atr_window = atr.tail(100)
        percentile = (atr_window < current_atr).sum() / len(atr_window)

        floor = self.config['safety']['atr_floor_percentile']
        cap = self.config['safety']['atr_cap_percentile']

        return floor <= percentile <= cap

    def _get_4h_bar_id(self, timestamp: datetime) -> int:
        """Get 4H bar ID for tracking fusion validation intervals."""
        return timestamp.hour // 4

    def _update_higher_timeframes(self, current_time: datetime, all_4h: List, all_1d: List):
        """Update 4H and 1D dataframes when appropriate."""
        # Find matching 4H bar
        for tick_4h in all_4h:
            if tick_4h['timestamp'] <= current_time:
                if self.df_4h.empty or tick_4h['timestamp'] != self.df_4h.index[-1]:
                    self.df_4h = self.adapter.update_ohlcv(self.df_4h, tick_4h, max_bars=200)

        # Find matching 1D bar
        for tick_1d in all_1d:
            if tick_1d['timestamp'] <= current_time:
                if self.df_1d.empty or tick_1d['timestamp'] != self.df_1d.index[-1]:
                    self.df_1d = self.adapter.update_ohlcv(self.df_1d, tick_1d, max_bars=100)

    def _log_signal(self, signal: Dict):
        """Log signal to JSONL file."""
        import os
        os.makedirs('results', exist_ok=True)

        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'results/hybrid_signals_{self.asset}_{date_str}.jsonl'

        with open(log_file, 'a') as f:
            json.dump(signal, f)
            f.write('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bull Machine v1.8 Hybrid Runner')
    parser.add_argument('--asset', required=True, help='Asset symbol (BTC, ETH, SOL)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', required=True, help='Path to v1.8 config file')

    args = parser.parse_args()

    runner = HybridRunner(args.asset, args.config, args.start, args.end)
    runner.run()
