#!/usr/bin/env python3
"""
Standalone Backtest for Bull Machine v11 Isolated Architecture

Runs a complete backtest using the v11 IsolatedArchetypeEngine:
- Per-archetype fusion calculation (ArchetypeInstance)
- Portfolio allocation with correlation constraints (PortfolioAllocator)
- ATR-based position sizing with regime scaling
- Per-archetype exit logic (scale-outs, time exits, trailing stops)
- Comprehensive PnL tracking (Sharpe, PF, drawdown)

Compatible with both v10 and v11 configs via --config flag.

Usage:
    # v11 isolated architecture (default)
    python bin/backtest_v11_standalone.py --config configs/bull_machine_isolated_v11_fixed.json

    # Custom date range
    python bin/backtest_v11_standalone.py --config configs/bull_machine_isolated_v11_fixed.json \
        --start-date 2023-01-01 --end-date 2023-03-31

    # Verbose mode
    python bin/backtest_v11_standalone.py --config configs/bull_machine_isolated_v11_fixed.json --verbose

Author: Claude Code (System Architect)
Date: 2026-02-05
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.analysis.counterfactual import CounterfactualEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for internal position & trade tracking
# ---------------------------------------------------------------------------

@dataclass
class TrackedPosition:
    """Internal position state for the standalone backtest."""
    position_id: str
    archetype: str
    direction: str          # 'long' or 'short'
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: float
    original_quantity: float  # CRITICAL: store original for exit calcs
    current_quantity: float
    fusion_score: float
    regime_at_entry: str
    atr_at_entry: float
    bars_held: int = 0
    executed_scale_outs: List[float] = field(default_factory=list)
    total_exits_pct: float = 0.0
    trailing_stop: Optional[float] = None
    # Context at entry (for trade attribution)
    threshold_at_entry: float = 0.0
    threshold_margin: float = 0.0     # fusion_score - threshold
    risk_temp_at_entry: float = 0.0
    instability_at_entry: float = 0.0
    crisis_prob_at_entry: float = 0.0
    # CMI sub-components (for weight optimization)
    trend_align_at_entry: float = 0.0
    trend_strength_at_entry: float = 0.0
    sentiment_at_entry: float = 0.0
    dd_score_at_entry: float = 0.0
    chop_at_entry: float = 0.0
    adx_weakness_at_entry: float = 0.0
    wick_sc_at_entry: float = 0.0
    vol_instab_at_entry: float = 0.0
    base_crisis_at_entry: float = 0.0
    vol_shock_at_entry: float = 0.0
    sentiment_crisis_at_entry: float = 0.0
    leverage_applied: float = 1.0
    position_size_usd: float = 0.0
    margin_used: float = 0.0  # margin locked at exchange (notional / leverage)
    # Domain scores (for fusion predictiveness analysis)
    wyckoff_score_at_entry: float = 0.0
    liquidity_score_at_entry: float = 0.0
    momentum_score_at_entry: float = 0.0
    smc_score_at_entry: float = 0.0
    gate_penalty_at_entry: float = 1.0
    # Structural health monitoring (for --health-mode)
    entry_health_score: float = 0.0
    entry_wyckoff_active: bool = False
    entry_liquidity_active: bool = False
    entry_momentum_active: bool = False
    entry_smc_active: bool = False
    health_trailing_tightened: bool = False
    # Entry metadata for invalidation exits
    entry_metadata: Dict[str, Any] = field(default_factory=dict)
    # 2-bar confirmation for structural invalidation
    structural_breach_count: int = 0
    # Runner state
    runner_trailing_stop: Optional[float] = None


@dataclass
class CompletedTrade:
    """Completed trade record."""
    timestamp_entry: pd.Timestamp
    timestamp_exit: pd.Timestamp
    archetype: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_regime: str
    duration_hours: float
    fusion_score: float
    exit_reason: str
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr_at_entry: float = 0.0
    threshold_at_entry: float = 0.0
    threshold_margin: float = 0.0
    risk_temp_at_entry: float = 0.0
    instability_at_entry: float = 0.0
    crisis_prob_at_entry: float = 0.0
    # CMI sub-components
    trend_align_at_entry: float = 0.0
    trend_strength_at_entry: float = 0.0
    sentiment_at_entry: float = 0.0
    dd_score_at_entry: float = 0.0
    chop_at_entry: float = 0.0
    adx_weakness_at_entry: float = 0.0
    wick_sc_at_entry: float = 0.0
    vol_instab_at_entry: float = 0.0
    base_crisis_at_entry: float = 0.0
    vol_shock_at_entry: float = 0.0
    sentiment_crisis_at_entry: float = 0.0
    leverage_applied: float = 1.0
    position_size_usd: float = 0.0
    # Domain scores (for fusion predictiveness analysis)
    wyckoff_score_at_entry: float = 0.0
    liquidity_score_at_entry: float = 0.0
    momentum_score_at_entry: float = 0.0
    smc_score_at_entry: float = 0.0
    gate_penalty_at_entry: float = 1.0
    position_id: str = ""


class _PositionAdapter:
    """Lightweight adapter wrapping TrackedPosition for ExitLogic's Position interface.

    ExitLogic expects a Position with .metadata, .entry_price, .entry_time,
    .stop_loss, .direction, .runner_trailing_stop. This adapter bridges
    TrackedPosition fields to that interface without importing Position.
    """

    def __init__(self, tracked_pos: 'TrackedPosition'):
        self._pos = tracked_pos
        self.entry_price = tracked_pos.entry_price
        self.entry_time = tracked_pos.entry_time
        self.stop_loss = tracked_pos.stop_loss
        self.direction = tracked_pos.direction
        self.runner_trailing_stop = tracked_pos.runner_trailing_stop
        # ExitLogic reads/writes metadata for scale-out tracking & invalidation
        self.metadata = dict(tracked_pos.entry_metadata)
        # Sync executed_scale_outs into metadata (ExitLogic reads from here)
        self.metadata['executed_scale_outs'] = list(tracked_pos.executed_scale_outs)


# ---------------------------------------------------------------------------
# Standalone Backtest Engine
# ---------------------------------------------------------------------------

class StandaloneBacktestEngine:
    """
    Complete standalone backtest engine for v11 isolated architecture.

    This engine:
    1. Loads the feature store (parquet)
    2. Initializes IsolatedArchetypeEngine from YAML configs
    3. Iterates through each bar
    4. Generates signals via per-archetype fusion
    5. Manages positions with per-archetype exit logic
    6. Tracks equity, PnL, drawdown
    """

    def __init__(
        self,
        config: Dict[str, Any],
        feature_store_path: str = '',
        initial_cash: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 2.0,
        features_df: 'pd.DataFrame | None' = None,
        signal_mode: str = 'fusion',
        sizing_mode: str = 'fixed',
        health_mode: str = 'off',
        invalidation_mode: bool = False,
    ):
        self.config = config
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.signal_mode = signal_mode
        self.sizing_mode = sizing_mode
        self.health_mode = health_mode
        self.invalidation_mode = invalidation_mode

        # Position tracking
        self.positions: Dict[str, TrackedPosition] = {}
        self.trades: List[CompletedTrade] = []
        self.equity_curve: List[float] = [initial_cash]
        self.equity_timestamps: List[pd.Timestamp] = []

        # Statistics
        self.total_signals = 0
        self.signals_allocated = 0
        self.signals_rejected = 0

        # Same-direction entry spacing (prevent correlated clusters)
        self._last_long_entry_bar = -999
        self._last_short_entry_bar = -999
        self._entry_spacing_bars = 2  # min bars between same-direction entries

        # Load feature store (or use pre-loaded DataFrame)
        if features_df is not None:
            self.features_df = features_df.copy()
        else:
            logger.info(f"Loading feature store: {feature_store_path}")
            self.features_df = pd.read_parquet(feature_store_path)
        if not isinstance(self.features_df.index, pd.DatetimeIndex):
            self.features_df.index = pd.to_datetime(self.features_df.index)
        self.features_df = self.features_df.sort_index()
        logger.info(
            f"Feature store loaded: {len(self.features_df):,} bars, "
            f"{len(self.features_df.columns)} columns, "
            f"{self.features_df.index.min()} to {self.features_df.index.max()}"
        )

        # Inject derived features needed for archetype gates
        if 'prior_12h_return' not in self.features_df.columns:
            self.features_df['prior_12h_return'] = (
                self.features_df['close'].pct_change(12).fillna(0.0)
            )
            logger.info("Injected prior_12h_return into feature store.")

        if 'range_position_20' not in self.features_df.columns:
            high_20 = self.features_df['high'].rolling(20).max()
            low_20 = self.features_df['low'].rolling(20).min()
            self.features_df['range_position_20'] = (
                (self.features_df['close'] - low_20) / (high_20 - low_20 + 1e-10)
            ).fillna(0.5)
            logger.info("Injected range_position_20 into feature store.")

        # Derive proper 4-regime labels from probability columns
        self._derive_regime_labels()

        # Initialize components
        self._init_archetype_engine()
        self._init_exit_logic()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _derive_regime_labels(self):
        """
        Derive 4-regime labels from price-action features.

        The ML regime model is miscalibrated (assigns 63% crisis probability
        on average). Instead, we use a price-based approach with SMAs
        computed from close prices if not available in the feature store:
          - risk_on:  close > SMA200 AND close > SMA50 (strong uptrend)
          - neutral:  close > SMA200 AND close <= SMA50 (uptrend pullback)
          - risk_off: close <= SMA200 AND close > SMA50 OR below both w/ low vol
          - crisis:   close <= SMA200 AND close <= SMA50 AND high volatility
        """
        if 'close' not in self.features_df.columns:
            logger.warning("No 'close' column found. Using raw regime_label.")
            return

        # Check if regime_label already has 4+ values
        unique_regimes = self.features_df['regime_label'].nunique()
        if unique_regimes >= 4:
            logger.info(f"regime_label has {unique_regimes} values - using as-is.")
            return

        logger.info(
            f"regime_label is degenerate ({unique_regimes} values). "
            f"Deriving 4-regime labels from price-action features."
        )

        df = self.features_df
        close = df['close']

        # Compute SMAs from close if store values are mostly NaN
        sma200 = df.get('sma_200')
        sma50 = df.get('sma_50')
        if sma200 is None or sma200.isna().mean() > 0.3:
            sma200 = close.rolling(200, min_periods=50).mean()
            logger.info("Computed SMA200 from close prices (store values missing).")
        if sma50 is None or sma50.isna().mean() > 0.3:
            sma50 = close.rolling(50, min_periods=20).mean()
            logger.info("Computed SMA50 from close prices (store values missing).")

        # Use atr_percentile for volatility if available, else compute from returns
        if 'atr_percentile' in df.columns and df['atr_percentile'].notna().mean() > 0.5:
            high_vol = df['atr_percentile'] > 0.75
        else:
            ret_vol = close.pct_change().rolling(168).std()  # 7-day vol
            high_vol = ret_vol > ret_vol.quantile(0.75)
            logger.info("Computed volatility from returns (atr_percentile missing).")

        above_sma200 = close > sma200
        above_sma50 = close > sma50

        regimes = pd.Series('neutral', index=df.index)
        # Strong uptrend: above both MAs
        regimes[above_sma200 & above_sma50] = 'risk_on'
        # Uptrend pullback: above 200 but below 50
        regimes[above_sma200 & ~above_sma50] = 'neutral'
        # Mixed / transitional: below 200 but above 50
        regimes[~above_sma200 & above_sma50] = 'risk_off'
        # Downtrend: below both MAs, low vol
        regimes[~above_sma200 & ~above_sma50] = 'risk_off'
        # Crisis: below both MAs AND high volatility
        regimes[~above_sma200 & ~above_sma50 & high_vol] = 'crisis'

        # Handle initial NaN period from rolling window warmup
        regimes[sma200.isna()] = 'neutral'

        self.features_df['regime_label'] = regimes

        dist = self.features_df['regime_label'].value_counts()
        logger.info(f"Derived 4-regime distribution: {dist.to_dict()}")

    def _init_archetype_engine(self):
        """Initialize the IsolatedArchetypeEngine from config."""
        from engine.integrations.isolated_archetype_engine import IsolatedArchetypeEngine

        archetype_config_dir = self.config.get('archetype_config_dir', 'configs/archetypes/')
        portfolio_config = self.config.get('portfolio_allocation', {})

        # Regime classifier settings
        regime_cfg = self.config.get('regime_classifier', {})
        enable_regime = regime_cfg.get('enabled', False)
        regime_model_path = regime_cfg.get('model_path', None)

        # Check if model file actually exists
        if regime_model_path and not Path(regime_model_path).exists():
            logger.warning(
                f"Regime model not found: {regime_model_path}. "
                f"Falling back to static regime from feature store."
            )
            enable_regime = False
            regime_model_path = None

        # Ensure structural checks run in backtest mode (frozen feature bypass)
        if 'structural_checks' not in self.config:
            self.config['structural_checks'] = {}
        self.config['structural_checks'].setdefault('mode_context', 'backtest')
        self.config['structural_checks'].setdefault('enabled', True)

        self.engine = IsolatedArchetypeEngine(
            archetype_config_dir=archetype_config_dir,
            portfolio_config=portfolio_config,
            enable_regime=enable_regime,
            regime_model_path=regime_model_path,
            config=self.config  # Pass full config for ML options (use_ml_fusion, use_kelly_sizing)
        )

        # Adaptive fusion config (replaces fusion_thresholds_by_regime)
        self.adaptive_fusion = self.config.get('adaptive_fusion', {})
        # Legacy: fusion_thresholds_by_regime (used as fallback if adaptive_fusion not enabled)
        self.fusion_thresholds_by_regime = self.config.get('fusion_thresholds_by_regime', {})
        # Regime-dependent max positions (reduce exposure in crisis)
        self.max_positions_by_regime = self.config.get('max_positions_by_regime', {})

        # Probabilistic regime detector for adaptive fusion
        self._prob_detector = None
        if self.adaptive_fusion.get('enabled', False):
            try:
                from engine.context.probabilistic_regime_detector import ProbabilisticRegimeDetector
                # Load ML model
                crisis_model = None
                model_path = Path('models/logistic_regime_v4_no_funding_stratified.pkl')
                if model_path.exists():
                    try:
                        import joblib
                        crisis_model = joblib.load(model_path)
                        logger.info(f"Regime ML model loaded for adaptive fusion: {model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load regime ML model: {e}")

                if crisis_model is None:
                    class _MockCrisisModel:
                        def predict_proba(self, X):
                            return np.array([[0.05, 0.25, 0.50, 0.20]])
                    crisis_model = _MockCrisisModel()

                self._prob_detector = ProbabilisticRegimeDetector(
                    crisis_model=crisis_model,
                    crisis_threshold=0.15
                )
                logger.info(
                    f"Adaptive fusion enabled: "
                    f"crisis_coeff={self.adaptive_fusion.get('crisis_coefficient', 0.4)}, "
                    f"temp_coeff={self.adaptive_fusion.get('temperature_coefficient', 0.3)}, "
                    f"instab_coeff={self.adaptive_fusion.get('instability_coefficient', 0.6)}, "
                    f"flat_threshold={self.adaptive_fusion.get('flat_threshold', 0.18)}"
                )
            except ImportError:
                logger.warning("ProbabilisticRegimeDetector not available. Using legacy thresholds.")
        # Disabled archetypes (e.g., long_squeeze loses in all regimes on bull data)
        self.disabled_archetypes = set(self.config.get('disabled_archetypes', []))
        # Per-archetype regime restrictions — NOW SOFT (via regime_preferences multiplier)
        # archetype_allowed_regimes no longer contains hard blocks, only notes
        # Kept for backwards compatibility with configs that still have entries
        raw_allowed = self.config.get('archetype_allowed_regimes', {})
        self.archetype_allowed_regimes = {k: v for k, v in raw_allowed.items() if k != 'notes' and isinstance(v, list) and len(v) > 0}
        if self.disabled_archetypes:
            logger.info(f"Disabled archetypes: {self.disabled_archetypes}")

        # Position sizing config
        sizing_cfg = self.config.get('position_sizing', {})
        self.risk_per_trade = sizing_cfg.get('risk_per_trade_pct', 0.02)
        self.max_position_pct = sizing_cfg.get('max_position_size_pct', 0.12)  # legacy, unused
        self.max_margin_pct = sizing_cfg.get('max_margin_per_position_pct', 0.35)
        self.leverage = self.config.get('leverage', 1.0)

        logger.info(
            f"IsolatedArchetypeEngine initialized: "
            f"{len(self.engine.archetypes)} archetypes, "
            f"risk_per_trade={self.risk_per_trade:.1%}, "
            f"max_position_pct={self.max_position_pct:.1%}, "
            f"leverage={self.leverage:.1f}x"
        )

    def _init_exit_logic(self):
        """Initialize per-archetype exit logic."""
        from engine.archetypes.exit_logic import ExitLogic, create_default_exit_config

        exit_config = create_default_exit_config()
        if 'exit_logic' in self.config:
            exit_config.update(self.config['exit_logic'])

        self.exit_logic = ExitLogic(exit_config)
        logger.info("ExitLogic initialized (archetype-specific exit rules)")

    # ------------------------------------------------------------------
    # Reset (enables engine reuse across multiple run() calls)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all stateful fields so the engine can be re-run on a different date range.

        Reuses the already-initialized archetype engine and feature store —
        only clears position/trade/equity state. This avoids the costly
        archetype re-initialization (YAML loading, Wyckoff setup, etc.)
        between sequential WFO/CPCV splits.
        """
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_cash]
        self.equity_timestamps = []
        self.total_signals = 0
        self.signals_allocated = 0
        self.signals_rejected = 0
        self._last_long_entry_bar = -999
        self._last_short_entry_bar = -999
        # Reset archetype engine state (cooling, structural memory)
        if hasattr(self, 'engine') and hasattr(self.engine, 'reset'):
            self.engine.reset()

    # ------------------------------------------------------------------
    # Core backtest loop
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Run the backtest over the feature store.

        Args:
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)
        """
        df = self.features_df.copy()

        # Apply date filters
        if start_date:
            df = df[df.index >= start_date]
            logger.info(f"Filtered to start_date >= {start_date}")
        if end_date:
            df = df[df.index <= end_date]
            logger.info(f"Filtered to end_date <= {end_date}")

        if len(df) == 0:
            logger.error("No data after date filtering!")
            return

        logger.info(f"Running backtest: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

        t0 = time.time()

        # CMI weights — config-driven, defaults match config 2026-02-21
        cmi_weights = self.adaptive_fusion.get('cmi_weights', {})
        rt_w = cmi_weights.get('risk_temp', {})
        w_trend = rt_w.get('trend_align', 0.30)
        w_strength = rt_w.get('trend_strength', 0.05)
        w_sentiment = rt_w.get('sentiment', 0.15)
        w_dd = rt_w.get('dd_score', 0.50)
        w_deriv = rt_w.get('derivatives_heat', 0.0)

        inst_w = cmi_weights.get('instability', {})
        w_chop = inst_w.get('chop', 0.40)
        w_adx_weak = inst_w.get('adx_weakness', 0.10)
        w_wick = inst_w.get('wick_score', 0.25)
        w_vol = inst_w.get('vol_instab', 0.25)

        crisis_w = cmi_weights.get('crisis', {})
        w_base_crisis = crisis_w.get('base_crisis', 0.45)
        w_vol_shock = crisis_w.get('vol_shock', 0.10)
        w_sent_crisis = crisis_w.get('sentiment_crisis', 0.45)

        # Tracking accumulators for periodic stats logging
        _regime_counts: Dict[str, int] = {}
        _threshold_values: List[float] = []
        _risk_temp_values: List[float] = []
        _instability_values: List[float] = []
        _signals_per_period: int = 0
        _entries_per_period: int = 0

        # Cache high/low arrays for prev_high/prev_low lookback in _open_position
        self._highs = df['high'].values if 'high' in df.columns else None
        self._lows = df['low'].values if 'low' in df.columns else None

        for bar_idx, (ts, row) in enumerate(df.iterrows()):
            # Step 1: Update bars held for all positions
            for pos in self.positions.values():
                pos.bars_held += 1

            # Step 2: Check exits BEFORE entries
            self._check_all_exits(row, ts, bar_idx)

            # Step 3: Generate signals from IsolatedArchetypeEngine
            # Pass prev_row and lookback_df for structural pattern checks (logic.py)
            prev_row = df.iloc[bar_idx - 1] if bar_idx > 0 else None
            lookback_start = max(0, bar_idx - 500)
            lookback_df = df.iloc[lookback_start:bar_idx + 1]

            signals = self.engine.get_signals(
                bar=row,
                bar_index=bar_idx,
                prev_row=prev_row,
                lookback_df=lookback_df,
                signal_mode=self.signal_mode,
            )

            if signals:
                self.total_signals += len(signals)

                # Step 3a: Filter disabled archetypes
                if self.disabled_archetypes:
                    pre_filter = len(signals)
                    signals = [s for s in signals if s.archetype_id not in self.disabled_archetypes]
                    disabled_count = pre_filter - len(signals)
                    if disabled_count > 0:
                        self.signals_rejected += disabled_count

                if not signals:
                    equity = self._compute_equity(row['close'])
                    self.equity_curve.append(equity)
                    self.equity_timestamps.append(ts)
                    continue

                # Use pre-computed SMA-based regime label (from _derive_regime_labels at init)
                # SMA has fewer false transitions than EMA (4-8/year vs 8-15 for EMA on 1H BTC)
                # This matches the live system's SMA-based regime detection for parity
                current_regime = row.get('regime_label', 'neutral') if hasattr(row, 'get') else 'neutral'

                # Track regime distribution for periodic logging
                _regime_counts[current_regime] = _regime_counts.get(current_regime, 0) + 1
                raw_signal_count = len(signals)
                max_pos = 3  # Default; overridden by adaptive or legacy path below
                # Defaults for variables computed by adaptive fusion (needed if block is skipped)
                dd_score = 0.5
                risk_temp = 0.5
                trend_align = 0.5
                trend_strength = 0.0
                sentiment_score = 0.5
                instability = 0.3
                crisis_prob = 0.0
                crisis_penalty = 1.0
                flat_threshold = 0.18
                adx_weakness = 0.0
                wick_sc = 0.0
                vol_instab = 0.0
                base_crisis = 0.0
                vol_shock = 0.0
                chop = 0.5
                sentiment_crisis = 0.0
                base_threshold = self.adaptive_fusion.get('base_threshold', 0.18) if self.adaptive_fusion else 0.18
                temp_range = self.adaptive_fusion.get('temp_range', 0.35) if self.adaptive_fusion else 0.35
                instab_range = self.adaptive_fusion.get('instab_range', 0.15) if self.adaptive_fusion else 0.15
                per_arch_thresholds = self.adaptive_fusion.get('per_archetype_base_threshold', {}) if self.adaptive_fusion else {}

                # Step 3b: Adaptive fusion — continuous regime modulation
                # In structural mode, skip threshold filtering (hard gates are sufficient)
                if self.adaptive_fusion.get('enabled', False) and self.signal_mode != 'structural':
                    # Direct scoring from feature store (bypasses ProbabilisticRegimeDetector
                    # which has hardcoded normalizations that don't match our feature store)
                    def _get(col, default=0.0):
                        if hasattr(row, 'get'):
                            val = row.get(col, default)
                            try:
                                val = float(val)
                                if val != val:  # NaN check
                                    return float(default)
                                return val
                            except (TypeError, ValueError):
                                return float(default)
                        return float(default)

                    # --- CMI v0: risk_temperature [0-1]: 0=cold/bear, 1=hot/bull ---
                    # Orthogonal to archetype fusion — no Wyckoff/temporal/SMC here
                    # Trend bias (45%) — EMA features have 100% coverage
                    p_above_50 = _get('price_above_ema_50', 0)
                    ema_50_200 = _get('ema_50_above_200', 0)
                    if p_above_50 and ema_50_200:
                        trend_align = 1.0   # Bull: price > ema50 > ema200
                    elif p_above_50:
                        trend_align = 0.6   # Early recovery
                    elif ema_50_200:
                        trend_align = 0.4   # Distribution / topping
                    else:
                        trend_align = 0.0   # Bear: price < ema50, ema50 < ema200

                    # Momentum health (25%)
                    adx = _get('adx', _get('adx_14', 20.0))
                    trend_strength = min(adx / 40.0, 1.0)

                    # Sentiment contrarian bump (15%) — Fear & Greed
                    fear_greed = _get('fear_greed_norm', 0.5)
                    sentiment_score = fear_greed  # 0=extreme fear, 1=extreme greed

                    # Drawdown context: shallow drawdown = hot
                    # Default 0.5 = neutral (old 0.9 biased bearish, suppressed risk_temp)
                    dd_persist = _get('drawdown_persistence', 0.5)
                    dd_score = max(1.0 - dd_persist, 0.0)

                    # --- Derivatives heat: institutional conviction from OI/funding/taker ---
                    # NaN-safe: defaults to 0.5 (neutral) when data unavailable (pre-2022)
                    if w_deriv > 0:
                        oi_4h = _get('oi_change_4h', 0.0)
                        oi_4h_raw = row.get('oi_change_4h') if hasattr(row, 'get') else None
                        has_oi_data = oi_4h_raw is not None and not (isinstance(oi_4h_raw, float) and oi_4h_raw != oi_4h_raw)

                        if has_oi_data:
                            # OI momentum: rising OI = hot (institutional conviction)
                            oi_momentum = min(max(oi_4h + 0.5, 0.0), 1.0)  # center around 0, scale to [0,1]
                            # Funding health: moderate = hot, extreme = cold (overcrowded)
                            fund_rate = _get('binance_funding_rate', 0.0)
                            funding_health = max(1.0 - abs(fund_rate) * 5000.0, 0.0)  # extreme = unhealthy
                            # Taker conviction: net buying = hot
                            taker = _get('taker_imbalance', 0.0)
                            taker_conviction = min(max(taker + 0.5, 0.0), 1.0)  # center, scale to [0,1]

                            derivatives_heat = 0.40 * oi_momentum + 0.30 * funding_health + 0.30 * taker_conviction
                        else:
                            derivatives_heat = 0.5  # Neutral when no data
                    else:
                        derivatives_heat = 0.5

                    risk_temp = w_trend * trend_align + w_strength * trend_strength + w_sentiment * sentiment_score + w_dd * dd_score + w_deriv * derivatives_heat

                    # --- instability_score [0-1]: 0=stable/trending, 1=choppy ---
                    chop = _get('chop_score', 0.5)
                    adx_weakness = max(1.0 - adx / 40.0, 0.0)
                    wick = _get('wick_ratio', 1.0)
                    wick_sc = min(wick / 5.0, 1.0)
                    vol_z = _get('volume_z_7d', 0.0)
                    vol_instab = min(abs(vol_z) / 2.5, 1.0)

                    instability = w_chop * chop + w_adx_weak * adx_weakness + w_wick * wick_sc + w_vol * vol_instab

                    # --- CMI v0: crisis_prob [0-1]: pure stress measurement ---
                    # NO Wyckoff/accumulation offset — crisis stays as safety, never weakened
                    # Core stress signals (60%)
                    dd = _get('drawdown_persistence', 0.0)
                    crash_freq = _get('crash_frequency_7d', 0.0)
                    crisis_persist = _get('crisis_persistence', 0.0)
                    crisis_signals = int(dd > 0.96) + int(crash_freq >= 2) + int(crisis_persist > 0.55)
                    if crisis_signals >= 2:
                        base_crisis = min(0.7 + 0.1 * crisis_signals, 1.0)
                    elif crisis_signals == 1:
                        base_crisis = 0.10
                    else:
                        base_crisis = 0.02

                    # Volatility shock (20%)
                    rv = _get('rv_20d', 0.6)
                    vol_shock = min(max(rv - 0.8, 0.0) / 0.4, 1.0)  # rv > 0.8 → shock

                    # Sentiment extreme (20%) — extreme fear amplifies crisis
                    sentiment_crisis = max(0.0, (0.20 - fear_greed) / 0.20)  # F&G < 20 → crisis

                    crisis_prob = w_base_crisis * base_crisis + w_vol_shock * vol_shock + w_sent_crisis * sentiment_crisis

                    # Dynamic threshold approach: threshold varies with market conditions
                    # This replicates the old system's behavior (low threshold in bull, high in bear)
                    # but with continuous values instead of hard regime labels
                    c_coeff = self.adaptive_fusion.get('crisis_coefficient', 0.4)
                    base_threshold = self.adaptive_fusion.get('base_threshold', 0.18)
                    temp_range = self.adaptive_fusion.get('temp_range', 0.35)
                    instab_range = self.adaptive_fusion.get('instab_range', 0.15)
                    per_arch_thresholds = self.adaptive_fusion.get('per_archetype_base_threshold', {})

                    # Crisis penalty on fusion score
                    crisis_penalty = 1.0 - crisis_prob * c_coeff

                    # Dynamic threshold: rises in bear/choppy, falls in bull/stable
                    # Global threshold used for logging and as default
                    flat_threshold = base_threshold + (1.0 - risk_temp) * temp_range + instability * instab_range

                    # Apply crisis penalty and filter against per-archetype dynamic threshold
                    pre_filter = len(signals)
                    adjusted_signals = []
                    # Load archetype configs to check for bypass_fusion_threshold flag
                    _bypass_archetypes = set()
                    if hasattr(self.engine, 'archetype_configs'):
                        for aid, acfg in self.engine.archetype_configs.items():
                            if acfg.get('bypass_fusion_threshold', False):
                                _bypass_archetypes.add(aid)

                    for s in signals:
                        # Archetypes with bypass_fusion_threshold skip the dynamic threshold
                        # (their edge comes from hard gates, not fusion scoring)
                        if s.archetype_id in _bypass_archetypes:
                            adjusted_signals.append(s)
                            continue

                        # Per-archetype base threshold (falls back to global)
                        arch_base = per_arch_thresholds.get(s.archetype_id, base_threshold)
                        arch_threshold = arch_base + (1.0 - risk_temp) * temp_range + instability * instab_range
                        adjusted_fusion = s.fusion_score * crisis_penalty
                        if adjusted_fusion >= arch_threshold:
                            s.fusion_score = adjusted_fusion
                            adjusted_signals.append(s)
                        else:
                            logger.debug(
                                f"[FILTER] {s.archetype_id} rejected: "
                                f"fusion={s.fusion_score:.3f}*penalty={crisis_penalty:.3f}"
                                f"={adjusted_fusion:.3f} < threshold={arch_threshold:.3f}"
                            )
                    signals = adjusted_signals
                    filtered_by_regime = pre_filter - len(signals)
                    if filtered_by_regime > 0:
                        self.signals_rejected += filtered_by_regime

                    # Track threshold stats for periodic logging
                    _threshold_values.append(flat_threshold)
                    _risk_temp_values.append(risk_temp)
                    _instability_values.append(instability)
                    _signals_per_period += raw_signal_count

                    # Emergency exposure cap (Moneytaur: crisis_prob > 0.7 → half sizing)
                    emergency_threshold = self.adaptive_fusion.get('emergency_crisis_threshold', 0.7)
                    emergency_mult = self.adaptive_fusion.get('emergency_size_multiplier', 0.50)
                    if crisis_prob > emergency_threshold and signals:
                        for s in signals:
                            s.fusion_score *= emergency_mult
                        logger.info(
                            f"[EMERGENCY] crisis_prob={crisis_prob:.3f} > {emergency_threshold} — "
                            f"applied {emergency_mult}x sizing cap to {len(signals)} signals"
                        )

                    # Stress-scaled position limit (replaces hard max_positions_by_regime)
                    base_max_pos = self.adaptive_fusion.get('base_max_positions', 3)
                    stress_level = max(crisis_prob, instability * 0.5)
                    max_pos = max(1, round(base_max_pos * (1.0 - 0.5 * stress_level)))

                    # Per-candle signal summary (DEBUG: every candle with signals)
                    passed_names = [s.archetype_id for s in signals]
                    logger.debug(
                        f"[THRESHOLD] bar={bar_idx} | dynamic_threshold={flat_threshold:.3f} | "
                        f"risk_temp={risk_temp:.3f} | instability={instability:.3f} | "
                        f"crisis_prob={crisis_prob:.3f} | regime={current_regime} | "
                        f"stress={stress_level:.3f} | max_pos={max_pos}"
                    )
                    logger.debug(
                        f"[SIGNALS] bar={bar_idx} | raw={raw_signal_count} | "
                        f"post_filter={pre_filter} | post_threshold={len(signals)} | "
                        f"passed: {passed_names}"
                    )

                elif self.fusion_thresholds_by_regime:
                    # Legacy fallback: hard regime thresholds
                    regime_threshold = self.fusion_thresholds_by_regime.get(current_regime, 0.18)
                    pre_filter = len(signals)
                    signals = [s for s in signals if s.fusion_score >= regime_threshold]
                    filtered_by_regime = pre_filter - len(signals)
                    if filtered_by_regime > 0:
                        self.signals_rejected += filtered_by_regime
                    _signals_per_period += raw_signal_count
                    logger.debug(
                        f"[THRESHOLD] bar={bar_idx} | legacy regime_threshold={regime_threshold:.3f} | "
                        f"regime={current_regime} | raw={raw_signal_count} | "
                        f"post_threshold={len(signals)}"
                    )

                # Step 3c: Apply stress-scaled position limits (CMI v0)
                if signals:
                    if not self.adaptive_fusion.get('enabled', False):
                        # Legacy fallback: use hard regime limits if adaptive disabled
                        max_pos = self.max_positions_by_regime.get(current_regime, 3) if self.max_positions_by_regime else 3
                    # else: max_pos already computed above in adaptive fusion block
                    if len(self.positions) >= max_pos:
                        self.signals_rejected += len(signals)
                        signals = []

                # Step 3d: Same-direction entry spacing (prevent correlated clusters)
                if signals:
                    spaced_signals = []
                    for s in signals:
                        if s.direction == 'long' and (bar_idx - self._last_long_entry_bar) < self._entry_spacing_bars:
                            self.signals_rejected += 1
                            continue
                        if s.direction == 'short' and (bar_idx - self._last_short_entry_bar) < self._entry_spacing_bars:
                            self.signals_rejected += 1
                            continue
                        spaced_signals.append(s)
                    signals = spaced_signals

                if not signals:
                    equity = self._compute_equity(row['close'])
                    self.equity_curve.append(equity)
                    self.equity_timestamps.append(ts)
                    continue

                # Step 3e: Inject CMI confidence values into signal metadata for dynamic sizing
                # dd_score (r=+0.167), risk_temp (r=+0.126), trend_align (r=+0.105) are the
                # actual positive predictors — allocator uses these instead of fusion_score (r=-0.102)
                for s in signals:
                    s.metadata['dd_score'] = dd_score
                    s.metadata['risk_temp'] = risk_temp
                    s.metadata['trend_align'] = trend_align

                # Step 4: Allocate via PortfolioAllocator
                current_position_archetypes = [
                    pos.archetype for pos in self.positions.values()
                ]
                intents, rejections = self.engine.allocate(
                    signals,
                    current_positions=current_position_archetypes
                )

                self.signals_rejected += len(rejections)

                # Step 5: Execute allocations
                for intent in intents:
                    sig = intent.signal
                    # Per-trade entry logging with full context
                    _current_threshold = flat_threshold if self.adaptive_fusion.get('enabled', False) else self.fusion_thresholds_by_regime.get(current_regime, 0.18)
                    logger.info(
                        f"[TRADE_ENTRY] bar={bar_idx} | {sig.archetype_id} | "
                        f"fusion={sig.fusion_score:.3f} | threshold={_current_threshold:.3f} | "
                        f"regime={current_regime} | price=${sig.entry_price:,.2f} | "
                        f"SL=${sig.stop_loss:,.2f} | TP=${sig.take_profit:,.2f}"
                    )
                    _entries_per_period += 1
                    # Compute per-archetype threshold for attribution
                    _arch_base = per_arch_thresholds.get(sig.archetype_id, base_threshold) if self.adaptive_fusion.get('enabled', False) else 0.0
                    _arch_threshold = _arch_base + (1.0 - risk_temp) * temp_range + instability * instab_range if self.adaptive_fusion.get('enabled', False) else _current_threshold
                    self._open_position(
                        timestamp=ts,
                        archetype=sig.archetype_id,
                        direction=sig.direction,
                        entry_price=sig.entry_price,
                        stop_loss=sig.stop_loss,
                        take_profit=sig.take_profit,
                        fusion_score=sig.fusion_score,
                        regime_label=current_regime,
                        features=row,
                        allocated_size_pct=intent.allocated_size_pct,
                        bar_idx=bar_idx,
                        threshold_at_entry=_arch_threshold,
                        risk_temp=risk_temp if self.adaptive_fusion.get('enabled', False) else 0.0,
                        instability=instability if self.adaptive_fusion.get('enabled', False) else 0.0,
                        crisis_prob=crisis_prob if self.adaptive_fusion.get('enabled', False) else 0.0,
                        # CMI sub-components
                        trend_align=trend_align if self.adaptive_fusion.get('enabled', False) else 0.0,
                        trend_strength=trend_strength if self.adaptive_fusion.get('enabled', False) else 0.0,
                        sentiment_score=sentiment_score if self.adaptive_fusion.get('enabled', False) else 0.0,
                        dd_score=dd_score if self.adaptive_fusion.get('enabled', False) else 0.0,
                        chop=chop if self.adaptive_fusion.get('enabled', False) else 0.0,
                        adx_weakness=adx_weakness if self.adaptive_fusion.get('enabled', False) else 0.0,
                        wick_sc=wick_sc if self.adaptive_fusion.get('enabled', False) else 0.0,
                        vol_instab=vol_instab if self.adaptive_fusion.get('enabled', False) else 0.0,
                        base_crisis=base_crisis if self.adaptive_fusion.get('enabled', False) else 0.0,
                        vol_shock=vol_shock if self.adaptive_fusion.get('enabled', False) else 0.0,
                        sentiment_crisis=sentiment_crisis if self.adaptive_fusion.get('enabled', False) else 0.0,
                        domain_scores={
                            'wyckoff': sig.metadata.get('wyckoff_score', 0.0),
                            'liquidity': sig.metadata.get('liquidity_score', 0.0),
                            'momentum': sig.metadata.get('momentum_score', 0.0),
                            'smc': sig.metadata.get('smc_score', 0.0),
                        } if sig.metadata else None,
                    )
                    # Capture domain scores from signal metadata
                    sig_meta = sig.metadata or {}
                    pos_id = f"{sig.direction}_{sig.archetype_id}_{int(ts.timestamp())}"
                    if pos_id in self.positions:
                        self.positions[pos_id].wyckoff_score_at_entry = sig_meta.get('wyckoff_score', 0.0)
                        self.positions[pos_id].liquidity_score_at_entry = sig_meta.get('liquidity_score', 0.0)
                        self.positions[pos_id].momentum_score_at_entry = sig_meta.get('momentum_score', 0.0)
                        self.positions[pos_id].smc_score_at_entry = sig_meta.get('smc_score', 0.0)
                        self.positions[pos_id].gate_penalty_at_entry = sig_meta.get('gate_penalty', 1.0)
                        # Health monitor baseline (only when health mode is active)
                        if self.health_mode != 'off':
                            _pos = self.positions[pos_id]
                            _inst = self.engine.archetypes.get(sig.archetype_id)
                            if _inst is not None:
                                _fw = _inst.config.fusion_weights
                                _ws = sig_meta.get('wyckoff_score', 0.0)
                                _ls = sig_meta.get('liquidity_score', 0.0)
                                _ms = sig_meta.get('momentum_score', 0.0)
                                _ss = sig_meta.get('smc_score', 0.0)
                                _pos.entry_wyckoff_active = _ws > 0.0
                                _pos.entry_liquidity_active = _ls > 0.0
                                _pos.entry_momentum_active = _ms > 0.0
                                _pos.entry_smc_active = _ss > 0.0
                                # Weighted entry health score (active domains only)
                                _active_w = (
                                    (_fw.get('wyckoff', 0.25) if _pos.entry_wyckoff_active else 0.0) +
                                    (_fw.get('liquidity', 0.25) if _pos.entry_liquidity_active else 0.0) +
                                    (_fw.get('momentum', 0.25) if _pos.entry_momentum_active else 0.0) +
                                    (_fw.get('smc', 0.25) if _pos.entry_smc_active else 0.0)
                                )
                                if _active_w > 0:
                                    _pos.entry_health_score = (
                                        (_fw.get('wyckoff', 0.25) * _ws if _pos.entry_wyckoff_active else 0.0) +
                                        (_fw.get('liquidity', 0.25) * _ls if _pos.entry_liquidity_active else 0.0) +
                                        (_fw.get('momentum', 0.25) * _ms if _pos.entry_momentum_active else 0.0) +
                                        (_fw.get('smc', 0.25) * _ss if _pos.entry_smc_active else 0.0)
                                    ) / _active_w
                    # Track last entry bar per direction for spacing
                    if sig.direction == 'long':
                        self._last_long_entry_bar = bar_idx
                    elif sig.direction == 'short':
                        self._last_short_entry_bar = bar_idx

            # Step 6: Update equity curve
            equity = self._compute_equity(row['close'])
            self.equity_curve.append(equity)
            self.equity_timestamps.append(ts)

            # Periodic logging: summary every 500 bars, detailed every 5000
            if bar_idx % 500 == 0 and bar_idx > 0:
                elapsed = time.time() - t0

                # Compute threshold stats for the period
                if _threshold_values:
                    avg_thr = np.mean(_threshold_values)
                    min_thr = np.min(_threshold_values)
                    max_thr = np.max(_threshold_values)
                    avg_rt = np.mean(_risk_temp_values)
                    avg_inst = np.mean(_instability_values)
                else:
                    avg_thr = min_thr = max_thr = avg_rt = avg_inst = 0.0

                logger.info(
                    f"[PERIODIC] Bar {bar_idx:,}/{len(df):,} | "
                    f"{ts} | Equity: ${equity:,.0f} | "
                    f"Positions: {len(self.positions)} | "
                    f"Trades: {len(self.trades)} | "
                    f"Signals(period): {_signals_per_period} | "
                    f"Entries(period): {_entries_per_period} | "
                    f"{bar_idx/elapsed:.0f} bars/sec"
                )
                logger.info(
                    f"[PERIODIC_REGIME] distribution={_regime_counts} | "
                    f"threshold: avg={avg_thr:.3f} min={min_thr:.3f} max={max_thr:.3f} | "
                    f"risk_temp_avg={avg_rt:.3f} | instability_avg={avg_inst:.3f}"
                )

                # Reset period accumulators every 5000 bars
                if bar_idx % 5000 == 0:
                    _regime_counts = {}
                    _threshold_values = []
                    _risk_temp_values = []
                    _instability_values = []
                    _signals_per_period = 0
                    _entries_per_period = 0

        # Step 7: Close remaining positions at end
        if self.positions:
            last_row = df.iloc[-1]
            last_ts = df.index[-1]
            logger.info(f"Closing {len(self.positions)} open positions at backtest end")
            for pos_id in list(self.positions.keys()):
                self._close_position(
                    pos_id, last_row['close'], last_ts,
                    exit_reason="backtest_end", exit_pct=1.0
                )

        elapsed = time.time() - t0
        bars_per_sec = len(df) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Backtest completed in {elapsed:.1f}s "
            f"({bars_per_sec:,.0f} bars/sec)"
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_position(
        self,
        timestamp: pd.Timestamp,
        archetype: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        fusion_score: float,
        regime_label: str,
        features: pd.Series,
        allocated_size_pct: float,
        bar_idx: int = 0,
        threshold_at_entry: float = 0.0,
        domain_scores: dict = None,
        risk_temp: float = 0.0,
        instability: float = 0.0,
        crisis_prob: float = 0.0,
        # CMI sub-components
        trend_align: float = 0.0,
        trend_strength: float = 0.0,
        sentiment_score: float = 0.0,
        dd_score: float = 0.0,
        chop: float = 0.0,
        adx_weakness: float = 0.0,
        wick_sc: float = 0.0,
        vol_instab: float = 0.0,
        base_crisis: float = 0.0,
        vol_shock: float = 0.0,
        sentiment_crisis: float = 0.0,
    ):
        """Open a new position."""
        # Calculate position size
        portfolio_value = self.initial_cash  # Use initial capital, not depleted cash
        atr = features.get('atr_14', entry_price * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        # Guard against NaN stop_loss / take_profit from archetype signals
        if pd.isna(stop_loss) or stop_loss <= 0:
            stop_loss = entry_price - (atr * 2.0) if direction == 'long' else entry_price + (atr * 2.0)
        if pd.isna(take_profit) or take_profit <= 0:
            take_profit = entry_price + (atr * 4.0) if direction == 'long' else entry_price - (atr * 4.0)

        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        if pd.isna(stop_distance_pct) or stop_distance_pct <= 0:
            stop_distance_pct = 0.025  # fallback 2.5%

        # Use Kelly-adjusted risk if Kelly sizer is active on the engine
        risk_per_trade = self.risk_per_trade

        # Domain-count sizing: scale risk by how many domains are active (score > 0.25)
        # Keeps the same trades as fixed mode — only the size changes
        # 1 active → 0.75x | 2 → 1.0x (baseline) | 3 → 1.25x | 4 → 1.5x
        if self.sizing_mode == 'domain' and domain_scores:
            DOMAIN_THRESHOLD = 0.25
            DOMAIN_MULTIPLIERS = {0: 0.75, 1: 0.75, 2: 1.0, 3: 1.25, 4: 1.5}
            n_active = sum(1 for s in domain_scores.values() if s > DOMAIN_THRESHOLD)
            size_mult = DOMAIN_MULTIPLIERS.get(n_active, 1.0)
            risk_per_trade *= size_mult
            logger.debug(
                "[DOMAIN_SIZING] %s: %d/4 domains active → %.2fx size (risk=%.3f%%)",
                archetype, n_active, size_mult, risk_per_trade * 100
            )

        # CMI-based sizing: scale risk by dd_score quartile (real predictive signal)
        # dd_score Q1 (<0.25): 0.75x | Q2-Q3 (0.25-0.75): 1.0x | Q4 (>0.75): 1.25x
        # Source: quartile analysis showed WR gradient 63.8% → 87.3%, avg_pnl -$5 → $300
        # dd_score is a CMI parameter passed directly (not from parquet)
        elif self.sizing_mode == 'cmi':
            _dd = dd_score if not pd.isna(dd_score) else 0.5
            if _dd < 0.25:
                size_mult = 0.75
            elif _dd > 0.75:
                size_mult = 1.25
            else:
                size_mult = 1.0
            risk_per_trade *= size_mult
            logger.debug(
                "[CMI_SIZING] %s: dd_score=%.3f → %.2fx size (risk=%.3f%%)",
                archetype, _dd, size_mult, risk_per_trade * 100
            )

        if self.engine.kelly_sizer is not None:
            features_dict = features.to_dict() if hasattr(features, 'to_dict') else dict(features)
            kelly_risk = self.engine.get_kelly_risk_pct(
                fusion_score=fusion_score,
                regime=regime_label,
                features=features_dict,
                base_risk_pct=self.risk_per_trade
            )
            risk_per_trade = kelly_risk

        risk_dollars = portfolio_value * risk_per_trade
        notional = risk_dollars / stop_distance_pct

        # Scale by allocated percentage from PortfolioAllocator (conviction multiplier)
        notional *= (allocated_size_pct / 0.02)  # normalize: 0.02 is baseline

        # Margin = notional / leverage (what the exchange actually locks)
        margin = notional / self.leverage

        # Cap margin at max_margin_per_position_pct of initial capital
        max_margin = self.initial_cash * self.max_margin_pct
        if margin > max_margin:
            margin = max_margin
            notional = margin * self.leverage  # scale notional down proportionally

        position_size_usd = notional  # for trade log / PnL calculation

        # Check margin availability (not notional)
        commission = notional * self.commission_rate
        slippage = notional * (self.slippage_bps / 10000.0)
        margin_cost = margin + commission + slippage

        if margin_cost > self.cash:
            logger.debug(f"[SKIP] Insufficient margin for {archetype} position (margin ${margin:,.0f} > cash ${self.cash:,.0f})")
            self.signals_rejected += 1
            return

        # Apply slippage to entry price
        if direction == 'long':
            fill_price = entry_price * (1 + self.slippage_bps / 10000.0)
        else:
            fill_price = entry_price * (1 - self.slippage_bps / 10000.0)

        quantity = notional / fill_price

        # Deduct margin from cash (not full notional — this is a leveraged perp)
        self.cash -= margin_cost

        # Create position
        pos_id = f"{direction}_{archetype}_{int(timestamp.timestamp())}"
        self.positions[pos_id] = TrackedPosition(
            position_id=pos_id,
            archetype=archetype,
            direction=direction,
            entry_price=fill_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            original_quantity=quantity,
            current_quantity=quantity,
            fusion_score=fusion_score,
            regime_at_entry=regime_label,
            atr_at_entry=atr,
        )

        # Compute 20-bar lookback swing high/low for ExitLogic S1 target
        _prev_high = fill_price
        _prev_low = fill_price
        if self._highs is not None and bar_idx > 0:
            lb_start = max(0, bar_idx - 20)
            _prev_high = float(np.nanmax(self._highs[lb_start:bar_idx]))
            _prev_low = float(np.nanmin(self._lows[lb_start:bar_idx]))

        # Capture entry metadata for ExitLogic invalidation checks
        self.positions[pos_id].entry_metadata = {
            'entry_prev_low': _prev_low,
            'entry_prev_high': _prev_high,
            'entry_wick_low': features.get('low', fill_price) if hasattr(features, 'get') else fill_price,
            'entry_spring_low': features.get('low', fill_price) if hasattr(features, 'get') else fill_price,
            'entry_ob_low': features.get('order_block_low', features.get('low', fill_price)) if hasattr(features, 'get') else fill_price,
            'entry_support_level': stop_loss,
            'entry_funding_z': features.get('funding_Z', 0.0) if hasattr(features, 'get') else 0.0,
            'entry_oi_delta': features.get('oi_change_4h', 0.0) if hasattr(features, 'get') else 0.0,
            'entry_volume': features.get('volume', 0.0) if hasattr(features, 'get') else 0.0,
            'entry_adx': features.get('adx_14', 0.0) if hasattr(features, 'get') else 0.0,
            'archetype': archetype,
            'executed_scale_outs': [],
        }

        # Store context for trade attribution
        self.positions[pos_id].threshold_at_entry = threshold_at_entry
        self.positions[pos_id].threshold_margin = fusion_score - threshold_at_entry
        self.positions[pos_id].risk_temp_at_entry = risk_temp
        self.positions[pos_id].instability_at_entry = instability
        self.positions[pos_id].crisis_prob_at_entry = crisis_prob
        self.positions[pos_id].leverage_applied = self.leverage
        self.positions[pos_id].position_size_usd = position_size_usd
        self.positions[pos_id].margin_used = margin
        # CMI sub-components
        self.positions[pos_id].trend_align_at_entry = trend_align
        self.positions[pos_id].trend_strength_at_entry = trend_strength
        self.positions[pos_id].sentiment_at_entry = sentiment_score
        self.positions[pos_id].dd_score_at_entry = dd_score
        self.positions[pos_id].chop_at_entry = chop
        self.positions[pos_id].adx_weakness_at_entry = adx_weakness
        self.positions[pos_id].wick_sc_at_entry = wick_sc
        self.positions[pos_id].vol_instab_at_entry = vol_instab
        self.positions[pos_id].base_crisis_at_entry = base_crisis
        self.positions[pos_id].vol_shock_at_entry = vol_shock
        self.positions[pos_id].sentiment_crisis_at_entry = sentiment_crisis

        self.signals_allocated += 1
        logger.info(
            f"[ENTRY] {direction.upper()} {archetype} @ ${fill_price:,.2f} | "
            f"Size: ${position_size_usd:,.0f} | Qty: {quantity:.6f} | "
            f"Stop: ${stop_loss:,.2f} | Regime: {regime_label} | "
            f"Fusion: {fusion_score:.3f} | ID: {pos_id}"
        )

    def _close_position(
        self,
        pos_id: str,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
        exit_reason: str = "unknown",
        exit_pct: float = 1.0
    ):
        """Close a position (full or partial)."""
        if pos_id not in self.positions:
            return

        pos = self.positions[pos_id]

        # CRITICAL: Calculate exit from ORIGINAL quantity (not current) for partial exits
        # This avoids the zombie position compounding bug documented in CLAUDE.md
        exit_quantity = pos.original_quantity * exit_pct

        # Cap to current quantity
        exit_quantity = min(exit_quantity, pos.current_quantity)

        if exit_quantity <= 1e-10:
            return

        # Apply slippage to exit
        if pos.direction == 'long':
            fill_exit = exit_price * (1 - self.slippage_bps / 10000.0)
        else:
            fill_exit = exit_price * (1 + self.slippage_bps / 10000.0)

        # Calculate PnL
        if pos.direction == 'long':
            pnl = (fill_exit - pos.entry_price) * exit_quantity
        else:
            pnl = (pos.entry_price - fill_exit) * exit_quantity

        # Commission on exit notional
        exit_value = fill_exit * exit_quantity
        commission = exit_value * self.commission_rate
        pnl -= commission

        # Return margin + realized PnL to cash (not full exit notional)
        exit_fraction = exit_quantity / pos.original_quantity
        margin_returned = pos.margin_used * exit_fraction
        self.cash += margin_returned + pnl

        # PnL percentage — based on margin deployed (return on capital), not notional
        exit_fraction = exit_quantity / pos.original_quantity
        margin_for_exit = pos.margin_used * exit_fraction
        pnl_pct = (pnl / margin_for_exit * 100) if margin_for_exit > 0 else 0.0

        # Duration
        duration_hours = (exit_timestamp - pos.entry_time).total_seconds() / 3600.0

        # Record trade
        self.trades.append(CompletedTrade(
            timestamp_entry=pos.entry_time,
            timestamp_exit=exit_timestamp,
            archetype=pos.archetype,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=fill_exit,
            quantity=exit_quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_regime=pos.regime_at_entry,
            duration_hours=duration_hours,
            fusion_score=pos.fusion_score,
            exit_reason=exit_reason,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            atr_at_entry=pos.atr_at_entry,
            threshold_at_entry=pos.threshold_at_entry,
            threshold_margin=pos.threshold_margin,
            risk_temp_at_entry=pos.risk_temp_at_entry,
            instability_at_entry=pos.instability_at_entry,
            crisis_prob_at_entry=pos.crisis_prob_at_entry,
            # CMI sub-components
            trend_align_at_entry=pos.trend_align_at_entry,
            trend_strength_at_entry=pos.trend_strength_at_entry,
            sentiment_at_entry=pos.sentiment_at_entry,
            dd_score_at_entry=pos.dd_score_at_entry,
            chop_at_entry=pos.chop_at_entry,
            adx_weakness_at_entry=pos.adx_weakness_at_entry,
            wick_sc_at_entry=pos.wick_sc_at_entry,
            vol_instab_at_entry=pos.vol_instab_at_entry,
            base_crisis_at_entry=pos.base_crisis_at_entry,
            vol_shock_at_entry=pos.vol_shock_at_entry,
            sentiment_crisis_at_entry=pos.sentiment_crisis_at_entry,
            leverage_applied=pos.leverage_applied,
            position_size_usd=pos.position_size_usd,
            # Domain scores
            wyckoff_score_at_entry=pos.wyckoff_score_at_entry,
            liquidity_score_at_entry=pos.liquidity_score_at_entry,
            momentum_score_at_entry=pos.momentum_score_at_entry,
            smc_score_at_entry=pos.smc_score_at_entry,
            gate_penalty_at_entry=pos.gate_penalty_at_entry,
            position_id=pos.position_id,
        ))

        # Update position
        pos.current_quantity -= exit_quantity
        pos.total_exits_pct += exit_pct

        # Update Kelly sizer state (track consecutive losses & drawdown)
        if self.engine.kelly_sizer is not None:
            equity = self._compute_equity(fill_exit)
            running_max = max(self.equity_curve) if self.equity_curve else self.initial_cash
            current_dd = (equity - running_max) / running_max if running_max > 0 else 0.0
            self.engine.update_kelly_state(
                trade_pnl=pnl,
                current_drawdown=current_dd
            )

        logger.info(
            f"[EXIT] {exit_reason} | {pos.archetype} @ ${fill_exit:,.2f} | "
            f"PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%) | "
            f"Duration: {duration_hours:.1f}h | Remaining: {pos.current_quantity:.6f}"
        )

        # Remove if fully closed
        if pos.current_quantity < 1e-10 or pos.total_exits_pct >= 0.99:
            # Clean up any dust — return remaining margin + dust PnL
            if pos.current_quantity > 1e-10:
                dust_frac = pos.current_quantity / pos.original_quantity
                dust_margin = pos.margin_used * dust_frac
                if pos.direction == 'long':
                    dust_pnl = (fill_exit - pos.entry_price) * pos.current_quantity
                else:
                    dust_pnl = (pos.entry_price - fill_exit) * pos.current_quantity
                self.cash += dust_margin + dust_pnl
            del self.positions[pos_id]

    # ------------------------------------------------------------------
    # Exit checking
    # ------------------------------------------------------------------

    def _check_domain_health(
        self, pos: 'TrackedPosition', pos_id: str, row: pd.Series, ts: pd.Timestamp
    ) -> Optional[str]:
        """
        Check structural domain health — returns exit reason string or None.

        Recomputes domain scores for the active domains at entry, computes weighted
        delta normalized by active domain weight total, then:
          - Mild degradation (delta < -0.30): tighten trailing to 2.0x ATR (one-shot)
          - Severe degradation (delta < -0.50):
              tighten mode: tighten trailing to 1.5x ATR
              exit mode: return 'health_severe' to trigger close

        Skips if bars_held < 12 (allow position to mature before monitoring).
        """
        if pos.bars_held < 12:
            return None

        inst = self.engine.archetypes.get(pos.archetype)
        if inst is None:
            return None

        features = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        fw = inst.config.fusion_weights

        # Recompute only active domains
        scores: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        if pos.entry_wyckoff_active:
            scores['wyckoff'] = inst._get_wyckoff_score(features)
            weights['wyckoff'] = fw.get('wyckoff', 0.25)
        if pos.entry_liquidity_active:
            scores['liquidity'] = inst._get_liquidity_score(features)
            weights['liquidity'] = fw.get('liquidity', 0.25)
        if pos.entry_momentum_active:
            scores['momentum'] = inst._get_momentum_score(features)
            weights['momentum'] = fw.get('momentum', 0.25)
        if pos.entry_smc_active:
            scores['smc'] = inst._get_smc_score(features)
            weights['smc'] = fw.get('smc', 0.25)

        total_weight = sum(weights.values())
        if total_weight <= 0 or not scores:
            return None

        current_health = sum(weights[d] * scores[d] for d in scores) / total_weight
        health_delta = current_health - pos.entry_health_score

        atr = float(row.get('atr_14', pos.atr_at_entry))
        if pd.isna(atr) or atr <= 0:
            atr = pos.atr_at_entry

        # Severe degradation
        if health_delta < -0.50:
            if self.health_mode == 'exit':
                return 'health_severe'
            elif self.health_mode == 'tighten':
                # Tighten trailing to 1.5x ATR from current price
                if pos.direction == 'long':
                    new_stop = float(row['close']) - 1.5 * atr
                    current_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                    if new_stop > current_stop:
                        pos.trailing_stop = new_stop
                else:
                    new_stop = float(row['close']) + 1.5 * atr
                    current_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                    if new_stop < current_stop:
                        pos.trailing_stop = new_stop
            return None

        # Mild degradation (one-shot flag)
        if health_delta < -0.30 and not pos.health_trailing_tightened:
            pos.health_trailing_tightened = True
            if pos.direction == 'long':
                new_stop = float(row['close']) - 2.0 * atr
                current_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                if new_stop > current_stop:
                    pos.trailing_stop = new_stop
            else:
                new_stop = float(row['close']) + 2.0 * atr
                current_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                if new_stop < current_stop:
                    pos.trailing_stop = new_stop

        return None

    def _check_structural_invalidation(
        self, pos: 'TrackedPosition', row: pd.Series
    ) -> Optional[str]:
        """
        Price-level structural invalidation — per-archetype thesis checks.

        Unlike domain score monitoring (too noisy), this checks whether a specific
        price level that DEFINED the setup has been breached. These are hard
        structural facts: spring low violated = accumulation failed.

        Only fires after bars_held >= 2 (skip entry bar noise).
        Returns exit reason string or None.

        Archetypes monitored:
          spring         — close < spring_low * 0.998 (0.2% buffer)
          wick_trap      — close < wick_low * 0.999 + volume confirmation
          retest_cluster — close < cluster_low - 0.5 * ATR
          order_block_retest — close < ob_low * 0.999
        """
        if pos.bars_held < 4:
            pos.structural_breach_count = 0
            return None
        if pos.direction != 'long':
            return None  # short-side structural checks not yet calibrated

        close = float(row['close'])
        meta = pos.entry_metadata
        archetype = pos.archetype

        breached = False

        if archetype == 'spring':
            spring_low = meta.get('entry_spring_low', 0.0)
            # 0.4% buffer: normal 1H BTC noise can breach 0.2% without invalidating structure
            if spring_low > 0 and close < spring_low * 0.996:
                breached = True

        elif archetype == 'wick_trap':
            wick_low = meta.get('entry_wick_low', 0.0)
            entry_vol = meta.get('entry_volume', 0.0)
            curr_vol = float(row.get('volume', 0.0))
            if wick_low > 0 and close < wick_low * 0.997:
                # Require significant breakdown volume (≥1.2x entry bar volume)
                # Low-volume breaches are often stop-hunts that recover
                if entry_vol <= 0 or curr_vol >= entry_vol * 1.2:
                    breached = True

        elif archetype == 'retest_cluster':
            cluster_low = meta.get('entry_support_level', 0.0)
            atr = float(row.get('atr_14', pos.atr_at_entry))
            if pd.isna(atr) or atr <= 0:
                atr = pos.atr_at_entry
            if cluster_low > 0 and close < cluster_low - 0.5 * atr:
                breached = True

        elif archetype == 'order_block_retest':
            ob_low = meta.get('entry_ob_low', 0.0)
            # 0.3% buffer for order block retest
            if ob_low > 0 and close < ob_low * 0.997:
                breached = True

        if breached:
            pos.structural_breach_count += 1
            # Require 2 consecutive closes below the level — filters whipsaws that recover
            if pos.structural_breach_count >= 2:
                return 'thesis_invalidated'
        else:
            # Price recovered above level — reset counter
            pos.structural_breach_count = 0

        return None

    def _check_all_exits(self, row: pd.Series, ts: pd.Timestamp, bar_idx: int):
        """Check exits for all open positions using ExitLogic."""
        from engine.runtime.context import RuntimeContext

        # Build RuntimeContext once per bar (shared across all position checks)
        regime_label = row.get('regime_label', 'neutral') if hasattr(row, 'get') else 'neutral'
        bar_context = RuntimeContext(
            ts=ts,
            row=row,
            regime_probs={regime_label: 1.0},
            regime_label=regime_label,
            adapted_params={},
            thresholds={},
        )

        for pos_id in list(self.positions.keys()):
            pos = self.positions[pos_id]
            close_price = row['close']
            atr = row.get('atr_14', pos.atr_at_entry)
            if pd.isna(atr) or atr <= 0:
                atr = pos.atr_at_entry

            # --- 1. Check stop loss (hard stop, MUST stay inline for fill-at-stop-level) ---
            stop_hit = False
            if pos.direction == 'long':
                effective_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                if row['low'] <= effective_stop:
                    stop_hit = True
                    exit_price = effective_stop
            else:
                effective_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                if row['high'] >= effective_stop:
                    stop_hit = True
                    exit_price = effective_stop

            if stop_hit:
                self._close_position(
                    pos_id, exit_price, ts,
                    exit_reason="stop_loss", exit_pct=1.0
                )
                continue

            # --- 2. Delegate to ExitLogic for all other exits ---
            # Build Position-like adapter for ExitLogic
            pos_adapter = _PositionAdapter(pos)

            exit_signal = self.exit_logic.check_exit(
                bar=row,
                position=pos_adapter,
                archetype=pos.archetype,
                context=bar_context,
            )

            if exit_signal is not None:
                # Sync scale-out tracking back from adapter
                pos.executed_scale_outs = pos_adapter.metadata.get('executed_scale_outs', pos.executed_scale_outs)
                # Persist all ExitLogic flags (scaled_at_prev_high, moon_bag_taken, etc.)
                for flag_key in ('scaled_at_prev_high', 'moon_bag_taken'):
                    if pos_adapter.metadata.get(flag_key):
                        pos.entry_metadata[flag_key] = True

                # Handle trailing stop update (no exit, just stop movement)
                if exit_signal.stop_update is not None:
                    pos.trailing_stop = exit_signal.stop_update

                # Handle exit signal
                if exit_signal.exit_pct > 0:
                    exit_reason = exit_signal.reason or exit_signal.exit_type
                    self._close_position(
                        pos_id, close_price, ts,
                        exit_reason=exit_reason,
                        exit_pct=exit_signal.exit_pct,
                    )
            else:
                # Sync trailing stop updates from ExitLogic (trailing updates return None)
                if pos_adapter.stop_loss != pos.stop_loss:
                    pos.trailing_stop = pos_adapter.stop_loss

            # --- 3. Domain health monitor (lowest priority, after ExitLogic) ---
            if self.health_mode != 'off' and pos_id in self.positions:
                health_reason = self._check_domain_health(pos, pos_id, row, ts)
                if health_reason and pos_id in self.positions:
                    self._close_position(
                        pos_id, float(row['close']), ts,
                        exit_reason=health_reason, exit_pct=1.0
                    )

            # --- 4. Structural invalidation (price-level thesis checks) ---
            if self.invalidation_mode and pos_id in self.positions:
                inv_reason = self._check_structural_invalidation(pos, row)
                if inv_reason and pos_id in self.positions:
                    self._close_position(
                        pos_id, float(row['close']), ts,
                        exit_reason=inv_reason, exit_pct=1.0
                    )

    # ------------------------------------------------------------------
    # Equity computation
    # ------------------------------------------------------------------

    def _compute_equity(self, current_price: float) -> float:
        """Compute current equity (cash + locked margin + unrealized PnL)."""
        equity = self.cash
        for pos in self.positions.values():
            remaining_frac = pos.current_quantity / pos.original_quantity if pos.original_quantity > 0 else 0
            margin_locked = pos.margin_used * remaining_frac
            if pos.direction == 'long':
                unrealized = (current_price - pos.entry_price) * pos.current_quantity
            else:
                unrealized = (pos.entry_price - current_price) * pos.current_quantity
            equity += margin_locked + unrealized
        return equity

    # ------------------------------------------------------------------
    # Performance statistics
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> Dict[str, Any]:
        """Compute comprehensive performance statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_usd': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'avg_hold_hours': 0.0,
                'archetype_diversity': 0,
            }

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in winners) if winners else 0
        total_losses = abs(sum(t.pnl for t in losers)) if losers else 0

        # Sharpe ratio from equity curve
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() > 0 else 0.0

        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        sortino = (returns.mean() / downside_std * np.sqrt(252 * 24)) if downside_std > 0 else 0.0

        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown_pct = (equity_series - running_max) / running_max * 100
        drawdown_usd = equity_series - running_max

        total_pnl = sum(t.pnl for t in self.trades)
        unique_archetypes = set(t.archetype for t in self.trades)

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(self.trades) * 100 if self.trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'total_pnl': total_pnl,
            'total_return': total_pnl / self.initial_cash * 100,
            'max_drawdown': drawdown_pct.min(),
            'max_drawdown_usd': drawdown_usd.min(),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'avg_trade_pnl': total_pnl / len(self.trades),
            'avg_win': total_wins / len(winners) if winners else 0,
            'avg_loss': total_losses / len(losers) if losers else 0,
            'max_win': max(t.pnl for t in self.trades),
            'max_loss': min(t.pnl for t in self.trades),
            'avg_hold_hours': np.mean([t.duration_hours for t in self.trades]),
            'archetype_diversity': len(unique_archetypes),
            'unique_archetypes': sorted(unique_archetypes),
        }

    def get_archetype_breakdown(self) -> pd.DataFrame:
        """Per-archetype performance breakdown."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        df_trades = pd.DataFrame([{
            'archetype': t.archetype,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'duration_hours': t.duration_hours,
            'fusion_score': t.fusion_score,
            'direction': t.direction,
        } for t in self.trades])

        for arch in df_trades['archetype'].unique():
            arch_df = df_trades[df_trades['archetype'] == arch]
            wins = arch_df[arch_df['pnl'] > 0]
            losses = arch_df[arch_df['pnl'] <= 0]
            total_win_pnl = wins['pnl'].sum() if len(wins) > 0 else 0
            total_loss_pnl = abs(losses['pnl'].sum()) if len(losses) > 0 else 0

            records.append({
                'archetype': arch,
                'trades': len(arch_df),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(arch_df) * 100 if len(arch_df) > 0 else 0,
                'total_pnl': arch_df['pnl'].sum(),
                'avg_pnl': arch_df['pnl'].mean(),
                'profit_factor': total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf'),
                'avg_hold_hours': arch_df['duration_hours'].mean(),
                'avg_fusion': arch_df['fusion_score'].mean(),
                'direction': arch_df['direction'].iloc[0],
            })

        breakdown = pd.DataFrame(records)
        return breakdown.sort_values('total_pnl', ascending=False)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_trade_log(self, output_path: str):
        """Save trade log as CSV."""
        if not self.trades:
            logger.warning("No trades to save")
            return

        records = [{
            'timestamp': t.timestamp_entry,
            'archetype': t.archetype,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'entry_regime': t.entry_regime,
            'duration_hours': t.duration_hours,
            'fusion_score': t.fusion_score,
            'exit_reason': t.exit_reason,
            'exit_timestamp': t.timestamp_exit,
            'quantity': t.quantity,
            'stop_loss': t.stop_loss,
            'take_profit': t.take_profit,
            'atr_at_entry': t.atr_at_entry,
            'threshold_at_entry': t.threshold_at_entry,
            'threshold_margin': t.threshold_margin,
            'risk_temp': t.risk_temp_at_entry,
            'instability': t.instability_at_entry,
            'crisis_prob': t.crisis_prob_at_entry,
            'trend_align': t.trend_align_at_entry,
            'trend_strength': t.trend_strength_at_entry,
            'sentiment_score': t.sentiment_at_entry,
            'dd_score': t.dd_score_at_entry,
            'chop': t.chop_at_entry,
            'adx_weakness': t.adx_weakness_at_entry,
            'wick_sc': t.wick_sc_at_entry,
            'vol_instab': t.vol_instab_at_entry,
            'base_crisis': t.base_crisis_at_entry,
            'vol_shock': t.vol_shock_at_entry,
            'sentiment_crisis': t.sentiment_crisis_at_entry,
            'leverage': t.leverage_applied,
            'position_size_usd': t.position_size_usd,
            # Domain scores (for fusion predictiveness analysis)
            'wyckoff_score': t.wyckoff_score_at_entry,
            'liquidity_score': t.liquidity_score_at_entry,
            'momentum_score': t.momentum_score_at_entry,
            'smc_score': t.smc_score_at_entry,
            'gate_penalty': t.gate_penalty_at_entry,
            'position_id': t.position_id,
        } for t in self.trades]

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logger.info(f"Trade log saved: {output_path} ({len(df)} trades)")

    def save_equity_curve(self, output_path: str):
        """Save equity curve as CSV."""
        if not self.equity_timestamps:
            return

        df = pd.DataFrame({
            'timestamp': self.equity_timestamps,
            'equity': self.equity_curve[1:]  # Skip initial value
        })
        df.to_csv(output_path, index=False)
        logger.info(f"Equity curve saved: {output_path}")

    def print_summary(self):
        """Print formatted performance summary."""
        stats = self.get_performance_stats()

        print("\n" + "=" * 80)
        print("BULL MACHINE v11 STANDALONE BACKTEST - PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Architecture:        v11 Isolated Archetype Engine")
        print(f"Total Trades:        {stats['total_trades']}")
        print(f"Winning Trades:      {stats['winning_trades']}")
        print(f"Losing Trades:       {stats['losing_trades']}")
        print(f"Win Rate:            {stats['win_rate']:.2f}%")
        print(f"Profit Factor:       {stats['profit_factor']:.2f}")
        print("-" * 80)
        print(f"Total PnL:           ${stats['total_pnl']:,.2f}")
        print(f"Total Return:        {stats['total_return']:.2f}%")
        print(f"Avg Trade PnL:       ${stats['avg_trade_pnl']:,.2f}")
        print(f"Avg Win:             ${stats['avg_win']:,.2f}")
        print(f"Avg Loss:            ${stats['avg_loss']:,.2f}")
        print(f"Max Win:             ${stats['max_win']:,.2f}")
        print(f"Max Loss:            ${stats['max_loss']:,.2f}")
        print("-" * 80)
        print(f"Max Drawdown:        {stats['max_drawdown']:.2f}%")
        print(f"Max Drawdown $:      ${stats['max_drawdown_usd']:,.2f}")
        print(f"Sharpe Ratio:        {stats['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:       {stats['sortino_ratio']:.2f}")
        print("-" * 80)
        print(f"Avg Hold Duration:   {stats['avg_hold_hours']:.1f} hours")
        print(f"Archetype Diversity: {stats['archetype_diversity']} unique archetypes")
        if 'unique_archetypes' in stats:
            for arch in stats['unique_archetypes']:
                print(f"  - {arch}")
        print("-" * 80)
        print(f"Total Signals:       {self.total_signals}")
        print(f"Signals Allocated:   {self.signals_allocated}")
        print(f"Signals Rejected:    {self.signals_rejected}")

        # Print engine-level stats
        engine_stats = self.engine.get_stats()
        print(f"Signals by Archetype:")
        for arch, count in sorted(
            engine_stats['signals_by_archetype'].items(),
            key=lambda x: x[1], reverse=True
        ):
            if count > 0:
                print(f"  {arch:25s}: {count:5d} signals")

        print(f"Signals Blocked by Cooling: {engine_stats.get('signals_blocked_by_cooling', 0)}")
        print(f"Signals Filtered by Score:  {engine_stats.get('signals_filtered_by_score', 0)}")

        # ML Enhancement stats
        if engine_stats.get('ml_fusion_active') or engine_stats.get('kelly_sizing_active'):
            print("-" * 80)
            print("ML ENHANCEMENTS:")
            if engine_stats.get('ml_fusion_active'):
                print(f"  ML Fusion Scoring:  ACTIVE ({engine_stats.get('ml_fusion_applied', 0):,} signals scored)")
            else:
                print(f"  ML Fusion Scoring:  disabled")
            if engine_stats.get('kelly_sizing_active'):
                print(f"  Kelly Sizing:       ACTIVE ({engine_stats.get('kelly_sizing_applied', 0):,} trades sized)")
            else:
                print(f"  Kelly Sizing:       disabled")

        # Archetype breakdown
        breakdown = self.get_archetype_breakdown()
        if len(breakdown) > 0:
            print("\n" + "=" * 80)
            print("ARCHETYPE PERFORMANCE BREAKDOWN")
            print("=" * 80)
            for _, row in breakdown.iterrows():
                pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "inf"
                print(
                    f"  {row['archetype']:25s} | "
                    f"{row['direction']:5s} | "
                    f"Trades: {row['trades']:4d} | "
                    f"WR: {row['win_rate']:5.1f}% | "
                    f"PF: {pf_str:>6s} | "
                    f"PnL: ${row['total_pnl']:>10,.2f} | "
                    f"Avg Hold: {row['avg_hold_hours']:.0f}h"
                )

        print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run v11 standalone backtest for Bull Machine'
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/bull_machine_isolated_v11_fixed.json',
        help='Path to config JSON'
    )
    parser.add_argument(
        '--feature-store', type=str,
        default='data/features_mtf/BTC_1H_LATEST.parquet',
        help='Path to feature store parquet file'
    )
    parser.add_argument(
        '--start-date', type=str, default=None,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--initial-cash', type=float, default=100_000.0,
        help='Initial cash ($)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/v11_standalone',
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Verbose logging'
    )
    parser.add_argument(
        '--commission-rate', type=float, default=0.0002,
        help='Commission rate per side (default: 0.0002 = 2 bps, matches Coinbase live)'
    )
    parser.add_argument(
        '--slippage-bps', type=float, default=3.0,
        help='Slippage in basis points per side (default: 3.0, matches live engine)'
    )
    parser.add_argument('--counterfactual', action='store_true', default=False,
                        help='Run counterfactual analysis on completed trades')
    parser.add_argument(
        '--signal-mode', type=str, default='fusion',
        choices=['fusion', 'structural', 'composite'],
        help='Signal selection mode: fusion (default, weighted score vs threshold), '
             'structural (skip fusion threshold, hard gates only), '
             'composite (N-of-M: >=3 of 4 domains must score >0.25)'
    )
    parser.add_argument(
        '--sizing-mode', type=str, default='fixed',
        choices=['fixed', 'domain', 'cmi'],
        help='Position sizing mode: fixed (default, flat 2%% risk), '
             'domain (scale by active domain count: 1=0.75x, 2=1.0x, 3=1.25x, 4=1.5x), '
             'cmi (scale by dd_score quartile: <0.25=0.75x, 0.25-0.75=1.0x, >0.75=1.25x)'
    )
    parser.add_argument(
        '--invalidation-mode', action='store_true', default=False,
        help='Enable price-level structural invalidation exits for spring, wick_trap, '
             'retest_cluster, order_block_retest (close below thesis level → exit 100%%)'
    )
    parser.add_argument(
        '--health-mode', type=str, default='off',
        choices=['off', 'tighten', 'exit'],
        help='Structural domain health monitor: off (default, disabled), '
             'tighten (tighten trailing stop on domain score degradation), '
             'exit (force close on severe domain score degradation: delta < -0.30)'
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress noisy sub-loggers in non-verbose mode
    if not args.verbose:
        logging.getLogger('engine.archetypes.archetype_instance').setLevel(logging.WARNING)
        logging.getLogger('engine.portfolio.archetype_allocator').setLevel(logging.WARNING)
        logging.getLogger('engine.config.archetype_config_loader').setLevel(logging.WARNING)
        logging.getLogger('engine.context.regime_service').setLevel(logging.WARNING)
        logging.getLogger('engine.archetypes.exit_logic').setLevel(logging.WARNING)
        logging.getLogger('engine.portfolio.regime_allocator').setLevel(logging.WARNING)

    # Print header
    print("=" * 80)
    print("BULL MACHINE v11 STANDALONE BACKTEST")
    print("=" * 80)
    print(f"Config:        {args.config}")
    print(f"Feature Store: {args.feature_store}")
    print(f"Date Range:    {args.start_date or 'start'} to {args.end_date or 'end'}")
    print(f"Initial Cash:  ${args.initial_cash:,.0f}")
    print(f"Commission:    {args.commission_rate*10000:.1f} bps per side")
    print(f"Slippage:      {args.slippage_bps:.1f} bps per side")
    print(f"Signal Mode:   {args.signal_mode}")
    print(f"Sizing Mode:   {args.sizing_mode}")
    print(f"Health Mode:   {args.health_mode}")
    print(f"Invalidation:  {'on' if args.invalidation_mode else 'off'}")
    print(f"Output:        {args.output_dir}")
    print("=" * 80)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded config: {config.get('version', 'unknown')}")

    # Validate feature store exists
    feature_store_path = Path(args.feature_store)
    if not feature_store_path.exists():
        logger.error(f"Feature store not found: {feature_store_path}")
        logger.error("Run: python3 bin/build_complete_feature_store_with_regime.py")
        sys.exit(1)

    # Initialize and run backtest
    engine = StandaloneBacktestEngine(
        config=config,
        feature_store_path=str(feature_store_path),
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        signal_mode=args.signal_mode,
        sizing_mode=args.sizing_mode,
        health_mode=args.health_mode,
        invalidation_mode=args.invalidation_mode,
    )

    engine.run(
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Print summary
    engine.print_summary()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine.save_trade_log(str(output_dir / 'trade_log.csv'))
    engine.save_equity_curve(str(output_dir / 'equity_curve.csv'))

    # Counterfactual analysis
    if args.counterfactual and engine.trades:
        logger.info(f"Running counterfactual analysis on {len(engine.trades)} trades...")
        cf_engine = CounterfactualEngine(
            engine.features_df,
            commission_rate=engine.commission_rate,
            slippage_bps=engine.slippage_bps,
        )
        cf_results = cf_engine.analyze_all(engine.trades)
        cf_summary = CounterfactualEngine.summarize(cf_results)

        # Save counterfactual results
        cf_path = str(output_dir / 'counterfactual_analysis.json')
        with open(cf_path, 'w') as f:
            json.dump({
                'summary': cf_summary,
                'trades': [r.to_dict() for r in cf_results],
            }, f, indent=2, default=str)
        logger.info(f"Counterfactual analysis saved to {cf_path}")

        # Print insights
        if cf_summary.get('insights'):
            print("\n=== COUNTERFACTUAL INSIGHTS ===")
            for insight in cf_summary['insights']:
                print(f"  - {insight}")
            print(f"\n  Optimal trades: {cf_summary.get('optimal_trades_pct', 0):.0f}%")
            print(f"  Best scenario: {cf_summary.get('best_overall_scenario', 'N/A')} "
                  f"(avg +${cf_summary.get('best_overall_avg_improvement', 0):.0f}/trade)")
            print()

    # Save performance stats
    stats = engine.get_performance_stats()
    stats_path = output_dir / 'performance_stats.json'
    # Convert non-serializable types
    stats_clean = {}
    for k, v in stats.items():
        if isinstance(v, (list, set)):
            stats_clean[k] = list(v)
        elif isinstance(v, (np.floating, np.integer)):
            stats_clean[k] = float(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            stats_clean[k] = str(v)
        else:
            stats_clean[k] = v
    with open(stats_path, 'w') as f:
        json.dump(stats_clean, f, indent=2, default=str)
    logger.info(f"Performance stats saved: {stats_path}")

    # Save archetype breakdown
    breakdown = engine.get_archetype_breakdown()
    if len(breakdown) > 0:
        breakdown_path = output_dir / 'archetype_breakdown.csv'
        breakdown.to_csv(str(breakdown_path), index=False)
        logger.info(f"Archetype breakdown saved: {breakdown_path}")

    # Save config for reproducibility
    config_copy_path = output_dir / 'config_used.json'
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")
    print(f"  - trade_log.csv")
    print(f"  - equity_curve.csv")
    print(f"  - performance_stats.json")
    print(f"  - archetype_breakdown.csv")
    print(f"  - config_used.json")


if __name__ == '__main__':
    main()
