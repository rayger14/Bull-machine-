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

import sys
import os
from pathlib import Path

# Add project root to path (needed for imports)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib

# Bull Machine imports
from engine.io.tradingview_loader import load_tv
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.fusion.domain_fusion import analyze_fusion
from bin.live.fast_signals import generate_fast_signal
from bin.live.pnl_tracker_v2 import Portfolio


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

        # No adapter needed - using load_tv directly

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

        # Initialize portfolio for P&L tracking
        self.portfolio = Portfolio(initial_balance=10000, config=self.config)

        # Load data using TradingView loader (same as btc_simple_backtest.py)
        print("📊 Loading data...")
        df_1h_full = load_tv(f'{self.asset}_1H')
        df_4h_full = load_tv(f'{self.asset}_4H')
        df_1d_full = load_tv(f'{self.asset}_1D')

        # Standardize column names (load_tv returns lowercase)
        for df in [df_1h_full, df_4h_full, df_1d_full]:
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                              'close': 'Close', 'volume': 'Volume'}, inplace=True)

        # Filter to date range
        if self.start_date:
            df_1h_full = df_1h_full[df_1h_full.index >= self.start_date]
            df_4h_full = df_4h_full[df_4h_full.index >= self.start_date]
            df_1d_full = df_1d_full[df_1d_full.index >= self.start_date]

        if self.end_date:
            df_1h_full = df_1h_full[df_1h_full.index <= self.end_date]
            df_4h_full = df_4h_full[df_4h_full.index <= self.end_date]
            df_1d_full = df_1d_full[df_1d_full.index <= self.end_date]

        print(f"📊 Data loaded: {len(df_1h_full)} 1H bars, {len(df_4h_full)} 4H bars, {len(df_1d_full)} 1D bars")

        # Reset data containers
        self.df_1h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_4h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_1d = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Process each 1H bar incrementally (simulate live streaming)
        for i in range(len(df_1h_full)):
            current_time = df_1h_full.index[i]
            current_price = df_1h_full['Close'].iloc[i]
            high = df_1h_full['High'].iloc[i]
            low = df_1h_full['Low'].iloc[i]

            # Build incremental dataframes (growing window)
            self.df_1h = df_1h_full.iloc[:i+1].copy()

            # Update higher timeframes to match current time
            self.df_4h = df_4h_full[df_4h_full.index <= current_time].copy()
            self.df_1d = df_1d_full[df_1d_full.index <= current_time].copy()

            # Update open positions (check stops/targets)
            self.portfolio.update_positions(self.asset, current_time, high, low, current_price)

            # Need minimum data
            if len(self.df_1h) < 50 or len(self.df_4h) < 14 or len(self.df_1d) < 50:
                continue

            # Generate signal (dataframes already aligned by time)
            signal_result = self._generate_hybrid_signal(
                self.df_1h, self.df_4h, self.df_1d, current_time, current_price
            )

            if signal_result:
                self.signals.append(signal_result)
                # Log signal
                self._log_signal(signal_result)

                # Open position if tradeable signal
                if signal_result.get('action') == 'signal' and signal_result.get('side') in ['long', 'short']:
                    # Check if we already have a position for this asset
                    if self.asset in self.portfolio.positions:
                        # Close opposite position if signal flipped
                        if self.portfolio.positions[self.asset].side != signal_result.get('side'):
                            self.portfolio.force_close_position(self.asset, current_price, current_time)

                    # Try to open new position
                    self.portfolio.open_position(
                        asset=self.asset,
                        side=signal_result.get('side'),
                        entry_price=current_price,
                        df_1h=self.df_1h,
                        timestamp=current_time
                    )

            # Progress
            if i % 100 == 0:
                print(f"   Progress: {i+1}/{len(df_1h_full)} | Signals: {len(self.signals)}")

        print(f"\n✅ Hybrid run complete: {len(self.signals)} signals generated")

        # Close any remaining open positions at final price
        final_time = df_1h_full.index[-1]
        final_price = df_1h_full['Close'].iloc[-1]
        if self.asset in self.portfolio.positions:
            self.portfolio.force_close_position(self.asset, final_price, final_time)

        # Print P&L summary
        self.portfolio.print_summary()

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
        # Convert timestamp to timezone-naive for macro data comparison
        timestamp_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') else timestamp
        macro_snapshot = fetch_macro_snapshot(self.macro_data, timestamp_naive)
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
        """
        Run REAL fusion validation using all production domain engines.

        Uses:
        - Wyckoff phase detection (accumulation/distribution)
        - SMC (BOS/CHOCH/FVG/OB)
        - HOB/Liquidity (order blocks, volume profile)
        - Momentum (RSI/MACD divergence)
        - MTF confluence validation

        Returns:
            Signal dict with side, confidence, and domain scores, or None
        """
        try:
            # Run real fusion analysis with all domain engines + MTF validation
            fusion_signal = analyze_fusion(df_1h, df_4h, df_1d, self.config)

            # Log fusion validation
            self._log_fusion_validation(timestamp, fusion_signal.score, {
                'wyckoff': fusion_signal.wyckoff_score,
                'hob': fusion_signal.hob_score,
                'momentum': fusion_signal.momentum_score,
                'smc': fusion_signal.smc_score,
                'mtf_aligned': fusion_signal.mtf_aligned,
                'mtf_confidence': fusion_signal.mtf_confidence
            })

            # Check threshold
            if fusion_signal.score < self.config['fusion']['entry_threshold_confidence']:
                return None

            # Require MTF alignment if configured
            if self.config.get('mtf', {}).get('require_alignment', False):
                if not fusion_signal.mtf_aligned:
                    return None

            return {
                'side': fusion_signal.direction,
                'confidence': fusion_signal.score,
                'reasons': fusion_signal.reasons,
                'wyckoff': fusion_signal.wyckoff_score,
                'hob': fusion_signal.hob_score,
                'momentum': fusion_signal.momentum_score,
                'smc': fusion_signal.smc_score,
                'mtf_aligned': fusion_signal.mtf_aligned,
                'features': fusion_signal.features
            }

        except Exception as e:
            print(f"⚠️  Fusion validation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_fusion_validation(self, timestamp: datetime, fusion_score: float,
                                components: Dict):
        """Log fusion validation for analysis."""
        import os
        os.makedirs('results', exist_ok=True)

        log_entry = {
            'timestamp': timestamp.isoformat(),
            'fusion_score': fusion_score,
            **components
        }

        with open('results/fusion_validation.jsonl', 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

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
    parser.add_argument('--start', help='Start date (YYYY-MM-DD), optional for full range')
    parser.add_argument('--end', help='End date (YYYY-MM-DD), optional for full range')
    parser.add_argument('--config', required=True, help='Path to v1.8 config file')

    args = parser.parse_args()

    runner = HybridRunner(args.asset, args.config, args.start, args.end)
    runner.run()
