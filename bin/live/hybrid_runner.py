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
import gc
import logging

# Bull Machine imports
from engine.io.tradingview_loader import load_tv
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.fusion.domain_fusion import analyze_fusion
from bin.live.fast_signals import generate_fast_signal
from bin.live.smart_exits import SmartExitPortfolio
from bull_machine.utils.merge_windows import (
    merge_windows, calculate_coverage, calculate_density, should_fallback_to_full
)
from utils.config_compat import normalize_config_for_hybrid
from utils.datetime_utils import to_timezone_naive
from bin.live.constants import (
    MIN_BARS_1H, MIN_BARS_4H, MIN_BARS_1D,
    GC_INTERVAL_BARS,
    LOG_FLUSH_INTERVAL, PROGRESS_REPORT_INTERVAL,
    ATR_MIN_BARS, ATR_PERIOD, ATR_PERCENTILE_WINDOW,
    MAX_TRADES_FOR_LOSS_STREAK,
    DEFAULT_OUTPUT_DIR
)

# Initialize logger
logger = logging.getLogger(__name__)


def load_candidates(candidates_path: str, window_bars: int = 48,
                    total_bars: int = None, coverage_threshold: float = 0.65,
                    density_threshold: float = 0.15) -> tuple:
    """
    Load candidates from JSONL and create merged windows to prevent overlap explosion.

    Args:
        candidates_path: Path to candidates.jsonl file
        window_bars: Number of bars before/after each candidate to include (default: 48 for ±2 days)
        total_bars: Total number of bars in dataset (for auto-fallback check)
        coverage_threshold: Max coverage before fallback to full replay (default: 0.65)
        density_threshold: Max density before fallback to full replay (default: 0.15)

    Returns:
        Tuple of (candidates_dict, candidate_bars_set, should_fallback, fallback_reason):
        - candidates_dict: {timestamp: {'side': str, 'score': float, 'reason': str}}
        - candidate_bars_set: Set of all timestamps to process (merged windows)
        - should_fallback: Whether to fallback to full replay
        - fallback_reason: Reason for fallback (empty string if not falling back)
    """
    candidates = {}

    with open(candidates_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ts = pd.to_datetime(entry['timestamp'])
            candidates[ts] = {
                'side': entry['side'],
                'score': entry['score'],
                'reason': entry['reason']
            }

    print(f"📋 Loaded {len(candidates)} candidates")

    # Convert timestamps to bar indices for window merging
    all_timestamps = sorted(candidates.keys())

    # Create intervals: (start_idx, end_idx) for each candidate
    # We'll use hour offset from first timestamp as "index"
    if not all_timestamps:
        return candidates, set(), False, ""

    base_time = all_timestamps[0]
    intervals = []
    ts_to_index = {}
    index_to_ts = {}

    for ts in all_timestamps:
        hours_offset = int((ts - base_time).total_seconds() / 3600)
        ts_to_index[ts] = hours_offset
        index_to_ts[hours_offset] = ts
        intervals.append((hours_offset - window_bars, hours_offset + window_bars))

    # Merge overlapping windows
    merged_intervals = merge_windows(intervals, min_gap=0, expand=0)

    # Auto-fallback check
    should_fallback = False
    fallback_reason = ""
    if total_bars:
        should_fallback, fallback_reason = should_fallback_to_full(
            merged_intervals, len(candidates), total_bars,
            coverage_threshold, density_threshold
        )

    # Convert merged intervals back to timestamps
    candidate_bars = set()
    for start_idx, end_idx in merged_intervals:
        for idx in range(start_idx, end_idx + 1):
            # Map index back to timestamp (approximate for gaps)
            ts = base_time + pd.Timedelta(hours=idx)
            candidate_bars.add(ts)

    # Calculate efficiency metrics
    unmerged_count = len(intervals) * (2 * window_bars + 1)
    merged_count = len(candidate_bars)
    reduction_pct = (1 - merged_count / unmerged_count) * 100 if unmerged_count > 0 else 0

    print(f"📊 Window merge stats:")
    print(f"   Before merge: {unmerged_count} bar-visits ({len(intervals)} windows × {2*window_bars+1} bars)")
    print(f"   After merge:  {merged_count} unique bars")
    print(f"   Reduction:    {reduction_pct:.1f}%")

    if total_bars:
        coverage = calculate_coverage(merged_intervals, total_bars)
        density = calculate_density(len(candidates), total_bars)
        print(f"   Coverage:     {coverage*100:.1f}% of {total_bars} total bars")
        print(f"   Density:      {density*100:.1f}% candidates per bar")

    if should_fallback:
        print(f"⚠️  AUTO-FALLBACK: {fallback_reason}")
        print(f"   Batch mode not beneficial - will use full replay instead")

    return candidates, candidate_bars, should_fallback, fallback_reason


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

        # Apply config compatibility layer (hob↔liquidity aliases)
        self.config = normalize_config_for_hybrid(self.config)

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

        # PHASE 1 PERFORMANCE: Macro caching (safe - data changes daily)
        self.cache_macro_snapshot = None
        self.cache_macro_timestamp = None

        # PHASE 1 PERFORMANCE: Buffered logging
        self.log_buffer_signal_blocks = []
        self.log_buffer_fusion = []
        self.log_buffer_decision = []
        self.log_flush_interval = LOG_FLUSH_INTERVAL

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
        valid_modes = ['advisory', 'prefilter', 'execute_only_if_fusion_confirms', 'fusion_only']
        if self.config['fast_signals']['mode'] not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")

    def _compute_config_hash(self) -> str:
        """Compute hash of config for determinism validation."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    def run(self, candidates_path: str = None):
        """Execute hybrid paper trading with fast signals + periodic fusion.

        Args:
            candidates_path: Optional path to candidates.jsonl for batch mode.
                           If provided, only processes bars in candidate windows.
        """
        print("\n💰 Starting Hybrid Paper Trading...")

        # Initialize portfolio for P&L tracking (Smart Exits)
        self.portfolio = SmartExitPortfolio(initial_balance=10000, config=self.config, macro_data=self.macro_data)

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

        # Load candidates if in batch mode (now with auto-fallback)
        batch_mode = candidates_path is not None
        if batch_mode:
            print("\n🎯 BATCH MODE: Candidate-driven replay with window merging")
            batch_config = self.config.get('batch_mode', {})
            window_bars = batch_config.get('candidate_window_bars', 48)
            coverage_threshold = batch_config.get('coverage_threshold', 0.65)
            density_threshold = batch_config.get('density_threshold', 0.15)

            candidates, candidate_bars, should_fallback, fallback_reason = load_candidates(
                candidates_path, window_bars,
                total_bars=len(df_1h_full),
                coverage_threshold=coverage_threshold,
                density_threshold=density_threshold
            )

            # Auto-fallback to full mode if batch isn't beneficial
            if should_fallback:
                print(f"\n⚠️  AUTO-FALLBACK triggered: {fallback_reason}")
                print("   Switching to FULL MODE for this run\n")
                batch_mode = False
                candidates = {}
                candidate_bars = None
        else:
            print("\n💰 FULL MODE: Processing all bars")
            candidates = {}
            candidate_bars = None

        # Reset data containers
        self.df_1h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_4h = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_1d = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Process each 1H bar incrementally (simulate live streaming)
        bars_processed = 0
        bars_skipped = 0
        is_backtest = self.start_date is not None  # Detect backtest mode

        for i in range(len(df_1h_full)):
            current_time = df_1h_full.index[i]
            current_price = df_1h_full['Close'].iloc[i]
            high = df_1h_full['High'].iloc[i]
            low = df_1h_full['Low'].iloc[i]

            # BATCH MODE: Skip bars not in candidate windows
            if batch_mode and current_time not in candidate_bars:
                bars_skipped += 1
                continue

            bars_processed += 1

            # Periodic GC for backtest mode (avoid memory creep in long runs)
            if is_backtest and bars_processed % GC_INTERVAL_BARS == 0:
                gc.collect()

            # Build incremental dataframes (growing window)
            self.df_1h = df_1h_full.iloc[:i+1].copy()

            # Update higher timeframes to match current time
            self.df_4h = df_4h_full[df_4h_full.index <= current_time].copy()
            self.df_1d = df_1d_full[df_1d_full.index <= current_time].copy()

            # Fetch macro snapshot for smart exits
            timestamp_naive = to_timezone_naive(current_time)
            macro_snapshot = fetch_macro_snapshot(self.macro_data, timestamp_naive)

            # Update open positions (check stops/targets with smart exits)
            self.portfolio.update_positions(
                self.asset, current_time, high, low, current_price,
                self.df_1h, self.df_4h, macro_snapshot, self.config_hash
            )

            # Need minimum data for indicator calculations
            if len(self.df_1h) < MIN_BARS_1H or len(self.df_4h) < MIN_BARS_4H or len(self.df_1d) < MIN_BARS_1D:
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
                    # Log decision before attempting open
                    decision_log = {
                        'timestamp': str(current_time),
                        'asset': self.asset,
                        'side': signal_result.get('side'),
                        'fusion_score': signal_result.get('fusion_score', 0.0),
                        'threshold': self.config.get('fusion', {}).get('entry_threshold_confidence', 0.70),
                        'mtf_aligned': signal_result.get('mtf_aligned', False),
                        'macro_veto': signal_result.get('macro_veto', 0.0),
                        'reasons': signal_result.get('reasons', [])
                    }
                    with open('results/decision_log.jsonl', 'a') as f:
                        f.write(json.dumps(decision_log) + '\n')

                    # Check if we already have a position for this asset
                    if self.asset in self.portfolio.positions:
                        # Close opposite position if signal flipped
                        if self.portfolio.positions[self.asset].side != signal_result.get('side'):
                            self.portfolio.force_close_position(self.asset, current_price, current_time)

                    # Try to open new position with dynamic risk sizing
                    self.portfolio.open_position(
                        asset=self.asset,
                        side=signal_result.get('side'),
                        entry_price=current_price,
                        df_1h=self.df_1h,
                        timestamp=current_time,
                        config_hash=self.config_hash,
                        fusion_score=signal_result.get('fusion_score', 0.5),
                        df_4h=self.df_4h
                    )

            # PHASE 1 PERFORMANCE: Flush buffered logs periodically
            if i % PROGRESS_REPORT_INTERVAL == 0:
                if batch_mode:
                    print(f"   Progress: {i+1}/{len(df_1h_full)} | Processed: {bars_processed} | Skipped: {bars_skipped} | Signals: {len(self.signals)}")
                else:
                    print(f"   Progress: {i+1}/{len(df_1h_full)} | Signals: {len(self.signals)}")
                self._flush_log_buffers()

        # Final statistics
        if batch_mode:
            print(f"\n✅ Batch mode complete:")
            print(f"   Total bars:      {len(df_1h_full)}")
            print(f"   Bars processed:  {bars_processed} ({bars_processed/len(df_1h_full)*100:.1f}%)")
            print(f"   Bars skipped:    {bars_skipped} ({bars_skipped/len(df_1h_full)*100:.1f}%)")
            print(f"   Signals:         {len(self.signals)}")
            print(f"   Speedup factor:  ~{len(df_1h_full)/max(bars_processed, 1):.1f}×")
        else:
            print(f"\n✅ Hybrid run complete: {len(self.signals)} signals generated")

        # PHASE 1 PERFORMANCE: Final flush of any remaining buffered logs
        self._flush_log_buffers()

        # Close any remaining open positions at final price
        final_time = df_1h_full.index[-1]
        final_price = df_1h_full['Close'].iloc[-1]
        if self.asset in self.portfolio.positions:
            self.portfolio.force_close_position(self.asset, final_price, final_time, self.config_hash)

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
        timestamp_naive = to_timezone_naive(timestamp)

        # PHASE 1 PERFORMANCE: Cache macro snapshot (changes once per day max)
        macro_date = timestamp_naive.date()
        if self.cache_macro_timestamp != macro_date:
            self.cache_macro_snapshot = fetch_macro_snapshot(self.macro_data, timestamp_naive)
            self.cache_macro_timestamp = macro_date

        macro_snapshot = self.cache_macro_snapshot
        macro_result = analyze_macro(macro_snapshot, self.macro_config)

        if macro_result['veto_strength'] >= self.macro_config['macro_veto_threshold']:
            # Log macro veto for analysis
            logger.debug(
                f"Macro veto triggered at {timestamp.isoformat()}: "
                f"strength={macro_result['veto_strength']:.3f}, "
                f"threshold={self.macro_config['macro_veto_threshold']:.3f}"
            )
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
            logger.debug(f"ATR throttle blocked signal at {timestamp.isoformat()}")
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
            logger.debug(
                f"Execution mode blocked at {timestamp.isoformat()}: "
                f"fast={fast_signal is not None}, fusion={fusion_signal is not None}"
            )
            return None

        # 6. RETURN SIGNAL
        return {
            'timestamp': timestamp.isoformat(),
            'asset': self.asset,
            'price': current_price,
            'action': 'signal',
            'side': execute_signal['side'],
            'confidence': execute_signal['confidence'],
            'fusion_score': execute_signal['confidence'],  # FIX: Add fusion_score key for position sizing
            'reasons': execute_signal.get('reasons', []),
            'fast_signal': fast_signal is not None,
            'fusion_signal': fusion_signal is not None,
            'fusion_validated': fusion_signal is not None and current_4h_bar != self.last_4h_bar,
            'mtf_aligned': execute_signal.get('mtf_aligned', False),  # FIX: Add MTF alignment for decision logging
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

    def _flush_log_buffers(self):
        """PHASE 1 PERFORMANCE: Flush buffered logs to disk."""
        import os
        os.makedirs('results', exist_ok=True)

        if self.log_buffer_signal_blocks:
            with open('results/signal_blocks.jsonl', 'a') as f:
                for entry in self.log_buffer_signal_blocks:
                    f.write(json.dumps(entry) + '\n')
            self.log_buffer_signal_blocks = []

        if self.log_buffer_fusion:
            with open('results/fusion_validation.jsonl', 'a') as f:
                for entry in self.log_buffer_fusion:
                    json.dump(entry, f)
                    f.write('\n')
            self.log_buffer_fusion = []

    def _log_fusion_validation(self, timestamp: datetime, fusion_score: float,
                                components: Dict):
        """Log fusion validation for analysis (buffered)."""
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'fusion_score': fusion_score,
            **components
        }
        # PHASE 1 PERFORMANCE: Buffer instead of immediate write
        self.log_buffer_fusion.append(log_entry)

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

        elif mode == 'fusion_only':
            # Execute fusion signals only (ignore fast signals)
            return fusion_signal

        # Default: return None if mode not recognized
        return None

    def _count_consecutive_losses(self) -> int:
        """Count consecutive losses in recent trades."""
        if not self.recent_trades:
            return 0

        count = 0
        for trade in reversed(self.recent_trades[-MAX_TRADES_FOR_LOSS_STREAK:]):
            if trade.get('pnl', 0) < 0:
                count += 1
            else:
                break
        return count

    def _check_atr_throttle(self, df_1h: pd.DataFrame) -> bool:
        """Check if ATR is within acceptable range."""
        if len(df_1h) < ATR_MIN_BARS:
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

        atr = tr.rolling(ATR_PERIOD).mean()
        current_atr = atr.iloc[-1]

        # Calculate percentile
        atr_window = atr.tail(ATR_PERCENTILE_WINDOW)
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
    parser.add_argument('--candidates', type=str, default=None,
                        help='Path to candidates JSONL (batch mode - process only candidate windows)')

    args = parser.parse_args()

    runner = HybridRunner(args.asset, args.config, args.start, args.end)
    runner.run(candidates_path=args.candidates)
