#!/usr/bin/env python3
"""
Composite Confirmation Backtester — EXPERIMENT

Replaces the broken fusion scoring system (r=-0.082 negative predictive power)
with N-of-M weighted binary confirmation counting. Each archetype has a set of
domain-specific confirmations; each confirmation awards integer points. A signal
must accumulate >= min_score to trade.

Three modes:
  structural   — raw structural quality (no fusion, no confirmation)
  composite    — structural + N-of-M domain confirmation (THIS IS THE EXPERIMENT)
  production   — full production backtester (fusion + CMI + thresholds)

The composite mode does NOT modify the main engine. It reads features directly
and applies confirmation logic post-detect().

Usage:
    python3 bin/backtest_composite.py --mode structural --start-date 2020-01-01
    python3 bin/backtest_composite.py --mode composite --start-date 2020-01-01
    python3 bin/backtest_composite.py --mode composite --start-date 2020-01-01 --min-score-override 3
    python3 bin/backtest_composite.py --mode all --start-date 2020-01-01

Author: Claude Code
Date: 2026-03-16
Branch: feat/composite-confirmation-experiment
"""

import sys
import json
import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE CONFIRMATION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Confirmation:
    """A single binary confirmation check."""
    name: str
    feature: str            # Feature key in feature store
    op: str                 # 'min', 'max', 'bool_true', 'eq', 'range'
    value: Any = None       # Threshold value (or [lo, hi] for 'range')
    points: int = 1         # Points awarded if check passes
    nan_action: str = 'deny'  # 'award' = NaN passes, 'deny' = NaN fails
    description: str = ''


@dataclass
class ArchetypeConfirmation:
    """Full confirmation definition for one archetype."""
    archetype: str
    min_score: int          # Minimum points to trade
    max_possible: int       # Sum of all points (for logging)
    confirmations: List[Confirmation]


def _check_confirmation(conf: Confirmation, features: Dict) -> bool:
    """Evaluate a single confirmation check against feature values."""
    val = features.get(conf.feature, None)

    # NaN / missing handling
    if val is None:
        return conf.nan_action == 'award'
    if isinstance(val, float) and (val != val):  # NaN check
        return conf.nan_action == 'award'

    try:
        val = float(val)
    except (TypeError, ValueError):
        if conf.op == 'bool_true':
            return bool(val)
        return conf.nan_action == 'award'

    if conf.op == 'min':
        return val >= conf.value
    elif conf.op == 'max':
        return val <= conf.value
    elif conf.op == 'bool_true':
        return bool(val) and val > 0
    elif conf.op == 'eq':
        return abs(val - conf.value) < 1e-6
    elif conf.op == 'range':
        lo, hi = conf.value
        return lo <= val <= hi
    else:
        return False


def evaluate_composite(arch_conf: ArchetypeConfirmation, features: Dict,
                       min_score_override: Optional[int] = None) -> Tuple[bool, int, int, List[str]]:
    """
    Evaluate N-of-M composite confirmation for an archetype signal.

    Returns:
        (passed, score, max_possible, passing_checks)
    """
    score = 0
    passing = []

    for conf in arch_conf.confirmations:
        if _check_confirmation(conf, features):
            score += conf.points
            passing.append(conf.name)

    min_required = min_score_override if min_score_override is not None else arch_conf.min_score
    passed = score >= min_required

    return passed, score, arch_conf.max_possible, passing


# ═══════════════════════════════════════════════════════════════════════
# PER-ARCHETYPE CONFIRMATION DEFINITIONS
#
# Design principles:
#   - Core structural checks (the _check_* functions) remain untouched
#   - Confirmations replace the 4-domain weighted fusion score
#   - Each check is BINARY (pass/fail), not 0-1 continuous
#   - Points weight importance: 2pt = critical, 1pt = supporting
#   - nan_action='award' for context checks (Wyckoff phase, etc.)
#   - nan_action='deny' for core feature checks (volume, RSI, etc.)
#   - min_score is calibrated to allow partial confirmation
#     (typically 50-60% of max_possible)
# ═══════════════════════════════════════════════════════════════════════

COMPOSITE_DEFINITIONS: Dict[str, ArchetypeConfirmation] = {}


def _define_wick_trap():
    """K — Wick Trap: Wick rejection at liquidity level with volume."""
    return ArchetypeConfirmation(
        archetype='wick_trap',
        min_score=5,  # Need 5 of 8 possible points (was 4 — too lenient)
        max_possible=8,
        confirmations=[
            # --- LIQUIDITY DOMAIN (core for this archetype) ---
            Confirmation(
                name='liquidity_present',
                feature='liquidity_score', op='min', value=0.3,
                points=2, nan_action='deny',
                description='Liquidity level present (core edge)',
            ),
            Confirmation(
                name='volume_above_median',
                feature='volume_zscore', op='min', value=0.0,
                points=1, nan_action='deny',
                description='Real wick = real volume behind it',
            ),
            # --- MOMENTUM DOMAIN ---
            Confirmation(
                name='rsi_oversold_zone',
                feature='rsi_14', op='max', value=45.0,
                points=1, nan_action='deny',
                description='RSI in oversold territory (wick = reversal)',
            ),
            Confirmation(
                name='momentum_positive',
                feature='momentum_score', op='min', value=0.3,
                points=1, nan_action='award',
                description='Momentum context supports reversal',
            ),
            # --- WYCKOFF DOMAIN ---
            Confirmation(
                name='wyckoff_bullish',
                feature='wyckoff_bullish_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Wyckoff accumulation context',
            ),
            # --- SMC DOMAIN ---
            Confirmation(
                name='bos_context',
                feature='smc_bos_bullish', op='bool_true',
                points=1, nan_action='award',
                description='Break of structure context',
            ),
            Confirmation(
                name='temporal_confluence',
                feature='temporal_confluence_score', op='min', value=0.3,
                points=1, nan_action='award',
                description='Gann/temporal timing confluence',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['wick_trap'] = _define_wick_trap()


def _define_trap_within_trend():
    """H — Trap Within Trend: False breakdown in uptrend."""
    return ArchetypeConfirmation(
        archetype='trap_within_trend',
        min_score=5,  # Need 5 of 9 possible (was 4 — too lenient)
        max_possible=9,
        confirmations=[
            # --- TREND CONTEXT (core for this archetype) ---
            Confirmation(
                name='adx_trending',
                feature='adx_14', op='min', value=15.0,
                points=2, nan_action='deny',
                description='ADX confirms trend exists (core: no trend = no trap)',
            ),
            Confirmation(
                name='trend_aligned',
                feature='ema_slope_50', op='min', value=0.0,
                points=2, nan_action='deny',
                description='EMA slope confirms uptrend context',
            ),
            # --- LIQUIDITY DOMAIN ---
            Confirmation(
                name='liquidity_present',
                feature='liquidity_score', op='min', value=0.3,
                points=1, nan_action='deny',
                description='Liquidity level was swept',
            ),
            # --- WICK QUALITY ---
            Confirmation(
                name='wick_lower_dominant',
                feature='wick_lower_ratio', op='min', value=0.5,
                points=1, nan_action='deny',
                description='Lower wick confirms trap direction',
            ),
            # --- WYCKOFF ---
            Confirmation(
                name='wyckoff_bullish',
                feature='wyckoff_bullish_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Wyckoff accumulation context',
            ),
            # --- MOMENTUM ---
            Confirmation(
                name='rsi_not_extreme_high',
                feature='rsi_14', op='max', value=65.0,
                points=1, nan_action='deny',
                description='Not already overbought',
            ),
            # --- SMC ---
            Confirmation(
                name='smc_context',
                feature='smc_score', op='min', value=0.2,
                points=1, nan_action='award',
                description='Smart money structure context',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['trap_within_trend'] = _define_trap_within_trend()


def _define_liquidity_sweep():
    """G — Liquidity Sweep: Sweep below support with reclaim."""
    return ArchetypeConfirmation(
        archetype='liquidity_sweep',
        min_score=5,  # Need 5 of 9 possible (was 4 — too lenient)
        max_possible=9,
        confirmations=[
            # --- LIQUIDITY DOMAIN (core) ---
            Confirmation(
                name='liquidity_score',
                feature='liquidity_score', op='min', value=0.2,
                points=2, nan_action='deny',
                description='Minimum liquidity level (core)',
            ),
            Confirmation(
                name='wick_lower_dominant',
                feature='wick_lower_ratio', op='min', value=1.3,
                points=2, nan_action='deny',
                description='Lower wick >> body confirms sweep direction',
            ),
            # --- VOLUME ---
            Confirmation(
                name='volume_present',
                feature='volume_zscore', op='min', value=-0.5,
                points=1, nan_action='deny',
                description='Some volume activity',
            ),
            # --- WYCKOFF ---
            Confirmation(
                name='wyckoff_bullish',
                feature='wyckoff_bullish_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Wyckoff context supports reversal',
            ),
            # --- MOMENTUM ---
            Confirmation(
                name='rsi_oversold',
                feature='rsi_14', op='max', value=45.0,
                points=1, nan_action='deny',
                description='RSI in oversold zone (sweep at lows)',
            ),
            # --- SMC ---
            Confirmation(
                name='absorption_confirm',
                feature='absorption_flag', op='eq', value=1,
                points=1, nan_action='award',
                description='Panic absorption confirms sweep',
            ),
            Confirmation(
                name='bos_context',
                feature='smc_bos_bullish', op='bool_true',
                points=1, nan_action='award',
                description='BOS supports reclaim',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['liquidity_sweep'] = _define_liquidity_sweep()


def _define_failed_continuation():
    """D — Failed Continuation: FVG + weak momentum = failed breakdown."""
    return ArchetypeConfirmation(
        archetype='failed_continuation',
        min_score=4,  # Need 4 of 7 possible (was 3 — too lenient)
        max_possible=7,
        confirmations=[
            # --- MOMENTUM (core for this archetype) ---
            Confirmation(
                name='rsi_weak',
                feature='rsi_14', op='max', value=55.0,
                points=2, nan_action='deny',
                description='RSI confirms weak continuation (core)',
            ),
            Confirmation(
                name='volume_fading',
                feature='volume_zscore', op='max', value=1.5,
                points=1, nan_action='deny',
                description='Volume declining (failed move has no fuel)',
            ),
            # --- SMC ---
            Confirmation(
                name='fvg_present',
                feature='tf1h_fvg_bullish', op='bool_true',
                points=1, nan_action='award',
                description='Fair value gap present',
            ),
            Confirmation(
                name='effort_fading',
                feature='effort_result_ratio', op='max', value=1.0,
                points=1, nan_action='award',
                description='Low effort-to-result = failing move',
            ),
            # --- WYCKOFF ---
            Confirmation(
                name='wyckoff_bullish',
                feature='wyckoff_bullish_score', op='min', value=0.05,
                points=1, nan_action='award',
                description='Wyckoff context supports reversal',
            ),
            # --- DIVERGENCE ---
            Confirmation(
                name='rsi_divergence',
                feature='rsi_divergence', op='min', value=0.05,
                points=1, nan_action='award',
                description='RSI divergence confirms failed bearish',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['failed_continuation'] = _define_failed_continuation()


def _define_retest_cluster():
    """L — Retest Cluster: Multi-level retest with fakeout."""
    return ArchetypeConfirmation(
        archetype='retest_cluster',
        min_score=5,  # Need 5 of 8 possible (was 4 — too lenient)
        max_possible=8,
        confirmations=[
            # --- VOLUME (core for retest) ---
            Confirmation(
                name='volume_spike',
                feature='volume_zscore', op='min', value=0.5,
                points=2, nan_action='deny',
                description='Volume confirms real move (core)',
            ),
            # --- MOMENTUM ---
            Confirmation(
                name='rsi_extreme',
                feature='rsi_14', op='max', value=65.0,
                points=1, nan_action='deny',
                description='RSI at extreme for reversal',
            ),
            Confirmation(
                name='rsi_divergence',
                feature='rsi_divergence', op='min', value=0.05,
                points=1, nan_action='award',
                description='RSI divergence at retest',
            ),
            # --- TEMPORAL ---
            Confirmation(
                name='temporal_confluence',
                feature='temporal_confluence_score', op='min', value=0.3,
                points=1, nan_action='award',
                description='Temporal confluence validates timing',
            ),
            # --- WYCKOFF ---
            Confirmation(
                name='wyckoff_context',
                feature='wyckoff_bullish_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Wyckoff supports retest',
            ),
            # --- LIQUIDITY ---
            Confirmation(
                name='liquidity_level',
                feature='liquidity_score', op='min', value=0.2,
                points=1, nan_action='award',
                description='Retest at liquidity level',
            ),
            # --- SMC ---
            Confirmation(
                name='smc_context',
                feature='smc_score', op='min', value=0.2,
                points=1, nan_action='award',
                description='Smart money structure',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['retest_cluster'] = _define_retest_cluster()


def _define_funding_divergence():
    """S4 — Funding Divergence: Short squeeze setup."""
    return ArchetypeConfirmation(
        archetype='funding_divergence',
        min_score=4,  # Need 4 of 7 possible (was 3 — too lenient)
        max_possible=7,
        confirmations=[
            # --- DERIVATIVES (core for this archetype) ---
            Confirmation(
                name='negative_funding',
                feature='funding_Z', op='max', value=0.0,
                points=2, nan_action='award',
                description='Negative funding rate (shorts paying, core)',
            ),
            Confirmation(
                name='funding_oi_divergence',
                feature='funding_oi_divergence', op='eq', value=1,
                points=2, nan_action='award',
                description='Bullish divergence: shorts building',
            ),
            # --- MOMENTUM ---
            Confirmation(
                name='price_resilient',
                feature='rsi_14', op='min', value=35.0,
                points=1, nan_action='deny',
                description='Price holding despite negative funding',
            ),
            # --- VOLUME ---
            Confirmation(
                name='volume_present',
                feature='volume_zscore', op='min', value=-0.5,
                points=1, nan_action='deny',
                description='Some activity present',
            ),
            # --- LIQUIDITY ---
            Confirmation(
                name='liquidity_context',
                feature='liquidity_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Liquidity context for squeeze target',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['funding_divergence'] = _define_funding_divergence()


def _define_spring():
    """A — Spring: Wyckoff spring at accumulation lows."""
    return ArchetypeConfirmation(
        archetype='spring',
        min_score=4,  # Need 4 of 7 possible (was 3 — too lenient)
        max_possible=7,
        confirmations=[
            # --- WYCKOFF (core for spring) ---
            Confirmation(
                name='wyckoff_bullish',
                feature='wyckoff_bullish_score', op='min', value=0.15,
                points=2, nan_action='deny',
                description='Wyckoff accumulation context (core)',
            ),
            Confirmation(
                name='accumulation_phase',
                feature='derived:wyckoff_in_accumulation', op='bool_true',
                points=1, nan_action='award',
                description='In accumulation phase',
            ),
            # --- PTI ---
            Confirmation(
                name='pti_context',
                feature='tf1h_pti_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='PTI confirms spring',
            ),
            # --- VOLUME ---
            Confirmation(
                name='volume_present',
                feature='volume_zscore', op='min', value=-0.5,
                points=1, nan_action='deny',
                description='Spring has volume behind it',
            ),
            # --- LIQUIDITY ---
            Confirmation(
                name='liquidity_context',
                feature='liquidity_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Liquidity level at spring',
            ),
            # --- MOMENTUM ---
            Confirmation(
                name='rsi_oversold',
                feature='rsi_14', op='max', value=45.0,
                points=1, nan_action='deny',
                description='Spring at oversold levels',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['spring'] = _define_spring()


def _define_liquidity_vacuum():
    """S1 — Liquidity Vacuum: Capitulation reversal."""
    return ArchetypeConfirmation(
        archetype='liquidity_vacuum',
        min_score=5,  # Need 5 of 8 possible (was 4 — too lenient)
        max_possible=8,
        confirmations=[
            # --- LIQUIDITY (core) ---
            Confirmation(
                name='low_liquidity',
                feature='liquidity_score', op='max', value=0.45,
                points=2, nan_action='deny',
                description='Low liquidity (capitulation, core)',
            ),
            # --- VOLUME ---
            Confirmation(
                name='volume_climax',
                feature='volume_zscore', op='min', value=0.5,
                points=1, nan_action='deny',
                description='Volume climax at capitulation',
            ),
            Confirmation(
                name='wick_exhaustion',
                feature='wick_exhaustion_last_3b', op='min', value=1.0,
                points=1, nan_action='award',
                description='Wick exhaustion validates capitulation',
            ),
            # --- WYCKOFF ---
            Confirmation(
                name='wyckoff_context',
                feature='wyckoff_bullish_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Wyckoff supports reversal',
            ),
            # --- MOMENTUM ---
            Confirmation(
                name='rsi_oversold',
                feature='rsi_14', op='max', value=35.0,
                points=1, nan_action='deny',
                description='Deep oversold at capitulation',
            ),
            # --- ABSORPTION ---
            Confirmation(
                name='absorption',
                feature='absorption_flag', op='eq', value=1,
                points=1, nan_action='award',
                description='Panic absorption at lows',
            ),
            # --- SMC ---
            Confirmation(
                name='bos_bearish',
                feature='smc_bos_bearish', op='bool_true',
                points=1, nan_action='award',
                description='BOS bearish confirms breakdown happened',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['liquidity_vacuum'] = _define_liquidity_vacuum()


def _define_order_block_retest():
    """B — Order Block Retest."""
    return ArchetypeConfirmation(
        archetype='order_block_retest',
        min_score=3,  # Need 3 of 6 possible
        max_possible=6,
        confirmations=[
            Confirmation(
                name='order_block_present',
                feature='order_block_bullish', op='bool_true',
                points=2, nan_action='award',
                description='Order block present (core)',
            ),
            Confirmation(
                name='volume_confirm',
                feature='volume_zscore', op='min', value=0.0,
                points=1, nan_action='deny',
                description='Volume at retest',
            ),
            Confirmation(
                name='wyckoff_context',
                feature='wyckoff_bullish_score', op='min', value=0.1,
                points=1, nan_action='award',
                description='Wyckoff supports retest',
            ),
            Confirmation(
                name='smc_context',
                feature='smc_score', op='min', value=0.2,
                points=1, nan_action='award',
                description='SMC context',
            ),
            Confirmation(
                name='momentum_ok',
                feature='rsi_14', op='range', value=[30, 65],
                points=1, nan_action='deny',
                description='RSI not extreme',
            ),
        ]
    )

COMPOSITE_DEFINITIONS['order_block_retest'] = _define_order_block_retest()


# Default fallback for archetypes without specific definitions
def _define_default(archetype_name: str) -> ArchetypeConfirmation:
    """Generic confirmation for archetypes without specific definitions.
    Tightened: nan_action='deny' on most checks to prevent auto-passing."""
    return ArchetypeConfirmation(
        archetype=archetype_name,
        min_score=4,  # Need 4 of 7 (57%) — tighter than before
        max_possible=7,
        confirmations=[
            Confirmation(
                name='volume_present',
                feature='volume_zscore', op='min', value=0.0,
                points=1, nan_action='deny',
                description='Above-median volume activity',
            ),
            Confirmation(
                name='wyckoff_context',
                feature='wyckoff_bullish_score', op='min', value=0.1,
                points=1, nan_action='deny',
                description='Wyckoff context',
            ),
            Confirmation(
                name='liquidity_context',
                feature='liquidity_score', op='min', value=0.2,
                points=1, nan_action='deny',
                description='Liquidity context',
            ),
            Confirmation(
                name='momentum_ok',
                feature='rsi_14', op='range', value=[30, 65],
                points=1, nan_action='deny',
                description='RSI not at extreme',
            ),
            Confirmation(
                name='smc_context',
                feature='smc_score', op='min', value=0.2,
                points=1, nan_action='deny',
                description='SMC context',
            ),
            Confirmation(
                name='trend_context',
                feature='adx_14', op='min', value=15.0,
                points=1, nan_action='deny',
                description='ADX confirms directional move',
            ),
            Confirmation(
                name='temporal_confluence',
                feature='temporal_confluence_score', op='min', value=0.3,
                points=1, nan_action='deny',
                description='Temporal confluence',
            ),
        ]
    )


def get_confirmation(archetype_name: str) -> ArchetypeConfirmation:
    """Get confirmation definition for an archetype."""
    if archetype_name in COMPOSITE_DEFINITIONS:
        return COMPOSITE_DEFINITIONS[archetype_name]
    return _define_default(archetype_name)


# ═══════════════════════════════════════════════════════════════════════
# POSITION & TRADE DATACLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Tracked position."""
    position_id: str
    archetype: str
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: float
    original_quantity: float
    current_quantity: float
    fusion_score: float
    composite_score: int
    composite_max: int
    atr_at_entry: float
    bars_held: int = 0
    executed_scale_outs: List[float] = field(default_factory=list)
    total_exits_pct: float = 0.0
    trailing_stop: Optional[float] = None
    margin_used: float = 0.0
    position_size_usd: float = 0.0
    entry_metadata: Dict[str, Any] = field(default_factory=dict)
    runner_trailing_stop: Optional[float] = None


@dataclass
class Trade:
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
    duration_hours: float
    fusion_score: float
    composite_score: int
    composite_max: int
    exit_reason: str
    stop_loss: float = 0.0
    take_profit: float = 0.0


class _PosAdapter:
    """Bridge Position to ExitLogic's expected interface."""
    def __init__(self, pos: Position):
        self._pos = pos
        self.entry_price = pos.entry_price
        self.entry_time = pos.entry_time
        self.stop_loss = pos.stop_loss
        self.direction = pos.direction
        self.runner_trailing_stop = pos.runner_trailing_stop
        self.metadata = dict(pos.entry_metadata)
        self.metadata['executed_scale_outs'] = list(pos.executed_scale_outs)


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

class CompositeBacktester:
    """Backtest with N-of-M composite confirmation replacing fusion scoring.

    Modes:
        structural — every structural signal trades (no filtering)
        composite  — structural + composite confirmation (N-of-M binary checks)
    """

    def __init__(self, config: Dict, features_df: pd.DataFrame,
                 initial_cash: float = 100_000.0,
                 commission_rate: float = 0.0002,
                 slippage_bps: float = 3.0,
                 mode: str = 'composite',
                 archetype_filter: Optional[List[str]] = None,
                 min_score_override: Optional[int] = None,
                 max_concurrent: int = 0):

        self.config = config
        self.features_df = features_df
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.mode = mode  # 'structural' or 'composite'
        self.archetype_filter = archetype_filter
        self.min_score_override = min_score_override
        self.max_concurrent = max_concurrent

        # Position sizing
        sizing_cfg = config.get('position_sizing', {})
        self.risk_per_trade = sizing_cfg.get('risk_per_trade_pct', 0.02)
        self.leverage = config.get('leverage', 1.5)
        self.max_margin_pct = sizing_cfg.get('max_margin_per_position_pct', 0.35)

        # State
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_cash]
        self.equity_timestamps: List[pd.Timestamp] = []

        # Per-archetype tracking
        self.arch_trades: Dict[str, List[Trade]] = {}

        # Stats
        self.raw_signals = 0
        self.composite_rejected = 0
        self.signals_traded = 0
        self.composite_score_dist: Dict[str, List[int]] = {}  # archetype -> [scores]

        # Initialize engine
        self._init_engine()
        self._init_exit_logic()

        # Precompute numpy arrays
        self._highs = features_df['high'].values if 'high' in features_df.columns else None
        self._lows = features_df['low'].values if 'low' in features_df.columns else None

    def _init_engine(self):
        """Initialize the archetype engine for signal generation."""
        from engine.integrations.isolated_archetype_engine import IsolatedArchetypeEngine

        archetype_config_dir = self.config.get('archetype_config_dir', 'configs/archetypes/')

        if 'structural_checks' not in self.config:
            self.config['structural_checks'] = {}
        self.config['structural_checks'].setdefault('mode_context', 'backtest')
        self.config['structural_checks'].setdefault('enabled', True)

        self.engine = IsolatedArchetypeEngine(
            archetype_config_dir=archetype_config_dir,
            portfolio_config={},
            enable_regime=False,
            config=self.config,
        )

        # Disable cooling periods
        for name, arch in self.engine.archetypes.items():
            arch._cooling_bar = -9999

        # Apply archetype filter
        if self.archetype_filter:
            filtered = {k: v for k, v in self.engine.archetypes.items()
                        if k in self.archetype_filter}
            if not filtered:
                print(f"WARNING: No archetypes match filter {self.archetype_filter}")
                print(f"Available: {list(self.engine.archetypes.keys())}")
            self.engine.archetypes = filtered

        print(f"Archetypes loaded: {list(self.engine.archetypes.keys())} ({len(self.engine.archetypes)})")

    def _init_exit_logic(self):
        """Initialize exit logic (identical to production)."""
        from engine.archetypes.exit_logic import ExitLogic, create_default_exit_config
        exit_config = create_default_exit_config()
        if 'exit_logic' in self.config:
            exit_config.update(self.config['exit_logic'])
        self.exit_logic = ExitLogic(exit_config)

    # ── Main Loop ──────────────────────────────────────────────────────

    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Run the backtest."""
        df = self.features_df
        if start_date:
            ts_start = pd.Timestamp(start_date)
            if df.index.tz is not None:
                ts_start = ts_start.tz_localize(df.index.tz)
            df = df[df.index >= ts_start]
        if end_date:
            ts_end = pd.Timestamp(end_date)
            if df.index.tz is not None:
                ts_end = ts_end.tz_localize(df.index.tz)
            df = df[df.index <= ts_end]

        if len(df) < 100:
            print(f"ERROR: Only {len(df)} bars after date filter. Need 100+.")
            return

        print(f"\nRunning {self.mode.upper()} backtest: {len(df):,} bars, "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        if self.mode == 'composite':
            print(f"Min score override: {self.min_score_override or 'per-archetype default'}")
        print()

        t0 = time.time()
        lookback_size = 100

        for bar_idx in range(1, len(df)):
            ts = df.index[bar_idx]
            row = df.iloc[bar_idx]
            prev_row = df.iloc[bar_idx - 1]

            lb_start = max(0, bar_idx - lookback_size)
            lookback_df = df.iloc[lb_start:bar_idx]

            # Update bars_held
            for pos in self.positions.values():
                pos.bars_held += 1

            # Check exits
            self._check_all_exits(row, ts, bar_idx)

            # Generate signals + composite filter
            self._generate_and_trade_signals(row, prev_row, lookback_df, ts, bar_idx)

            # Equity
            equity = self._compute_equity(row['close'])
            self.equity_curve.append(equity)
            self.equity_timestamps.append(ts)

            if bar_idx % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  bar {bar_idx:,}/{len(df):,} | "
                      f"positions={len(self.positions)} | "
                      f"trades={len(self.trades)} | "
                      f"signals={self.raw_signals} | "
                      f"rejected={self.composite_rejected} | "
                      f"{elapsed:.0f}s")

        # Close remaining positions
        last_row = df.iloc[-1]
        last_ts = df.index[-1]
        for pos_id in list(self.positions.keys()):
            self._close_position(pos_id, last_row['close'], last_ts, "end_of_data", 1.0)

        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    def _generate_and_trade_signals(self, row, prev_row, lookback_df, ts, bar_idx):
        """Generate structural signals and apply composite confirmation."""
        features = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

        for name, archetype in self.engine.archetypes.items():
            archetype._cooling_bar = -9999

            signal = archetype.detect(
                features, 'neutral',
                current_bar_idx=bar_idx,
                prev_row=prev_row,
                lookback_df=lookback_df,
                structural_checker=self.engine.structural_checker,
            )

            if signal is None:
                continue

            self.raw_signals += 1

            # Composite confirmation filter (only in composite mode)
            composite_score = 0
            composite_max = 0
            if self.mode == 'composite':
                arch_conf = get_confirmation(name)
                passed, score, max_possible, passing_checks = evaluate_composite(
                    arch_conf, features, self.min_score_override
                )
                composite_score = score
                composite_max = max_possible

                # Track score distribution
                if name not in self.composite_score_dist:
                    self.composite_score_dist[name] = []
                self.composite_score_dist[name].append(score)

                if not passed:
                    self.composite_rejected += 1
                    continue

            # Max concurrent check
            if self.max_concurrent > 0:
                arch_positions = sum(1 for p in self.positions.values() if p.archetype == name)
                if arch_positions >= self.max_concurrent:
                    continue

            fusion = signal.metadata.get('fusion_score', 0.5)
            self._open_position(
                timestamp=ts,
                archetype=name,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                fusion_score=fusion,
                composite_score=composite_score,
                composite_max=composite_max,
                features=row,
                bar_idx=bar_idx,
            )

    # ── Position Management ────────────────────────────────────────────

    def _open_position(self, timestamp, archetype, direction, entry_price,
                       stop_loss, take_profit, fusion_score, composite_score,
                       composite_max, features, bar_idx):
        """Open a position with fixed risk sizing."""
        atr = features.get('atr_14', entry_price * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        if pd.isna(stop_loss) or stop_loss <= 0:
            stop_loss = entry_price - (atr * 2.0) if direction == 'long' else entry_price + (atr * 2.0)
        if pd.isna(take_profit) or take_profit <= 0:
            take_profit = entry_price + (atr * 4.0) if direction == 'long' else entry_price - (atr * 4.0)

        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        if pd.isna(stop_distance_pct) or stop_distance_pct <= 0:
            stop_distance_pct = 0.025

        risk_dollars = self.initial_cash * self.risk_per_trade
        notional = risk_dollars / stop_distance_pct
        margin = notional / self.leverage

        max_margin = self.initial_cash * self.max_margin_pct
        if margin > max_margin:
            margin = max_margin
            notional = margin * self.leverage

        commission = notional * self.commission_rate
        slippage = notional * (self.slippage_bps / 10000.0)
        margin_cost = margin + commission + slippage

        if margin_cost > self.cash:
            return

        if direction == 'long':
            fill_price = entry_price * (1 + self.slippage_bps / 10000.0)
        else:
            fill_price = entry_price * (1 - self.slippage_bps / 10000.0)

        quantity = notional / fill_price
        self.cash -= margin_cost

        pos_id = f"{direction}_{archetype}_{int(timestamp.timestamp())}_{bar_idx}"

        _prev_high = fill_price
        _prev_low = fill_price
        if self._highs is not None and bar_idx > 0:
            lb_start = max(0, bar_idx - 20)
            _prev_high = float(np.nanmax(self._highs[lb_start:bar_idx]))
            _prev_low = float(np.nanmin(self._lows[lb_start:bar_idx]))

        self.positions[pos_id] = Position(
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
            composite_score=composite_score,
            composite_max=composite_max,
            atr_at_entry=atr,
            margin_used=margin,
            position_size_usd=notional,
            entry_metadata={
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
            },
        )
        self.signals_traded += 1

    def _close_position(self, pos_id, exit_price, exit_timestamp, exit_reason, exit_pct=1.0):
        """Close a position (full or partial)."""
        if pos_id not in self.positions:
            return

        pos = self.positions[pos_id]
        exit_quantity = pos.original_quantity * exit_pct
        exit_quantity = min(exit_quantity, pos.current_quantity)
        if exit_quantity <= 1e-10:
            return

        if pos.direction == 'long':
            fill_exit = exit_price * (1 - self.slippage_bps / 10000.0)
        else:
            fill_exit = exit_price * (1 + self.slippage_bps / 10000.0)

        if pos.direction == 'long':
            pnl = (fill_exit - pos.entry_price) * exit_quantity
        else:
            pnl = (pos.entry_price - fill_exit) * exit_quantity

        commission = fill_exit * exit_quantity * self.commission_rate
        pnl -= commission

        exit_fraction = exit_quantity / pos.original_quantity
        margin_returned = pos.margin_used * exit_fraction
        self.cash += margin_returned + pnl

        entry_value = pos.entry_price * exit_quantity
        pnl_pct = (pnl / entry_value * 100) if entry_value > 0 else 0.0
        duration_hours = (exit_timestamp - pos.entry_time).total_seconds() / 3600.0

        trade = Trade(
            timestamp_entry=pos.entry_time,
            timestamp_exit=exit_timestamp,
            archetype=pos.archetype,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=fill_exit,
            quantity=exit_quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_hours=duration_hours,
            fusion_score=pos.fusion_score,
            composite_score=pos.composite_score,
            composite_max=pos.composite_max,
            exit_reason=exit_reason,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
        )
        self.trades.append(trade)

        if pos.archetype not in self.arch_trades:
            self.arch_trades[pos.archetype] = []
        self.arch_trades[pos.archetype].append(trade)

        pos.current_quantity -= exit_quantity
        pos.total_exits_pct += exit_pct

        if pos.current_quantity < 1e-10 or pos.total_exits_pct >= 0.99:
            if pos.current_quantity > 1e-10:
                dust_frac = pos.current_quantity / pos.original_quantity
                dust_margin = pos.margin_used * dust_frac
                if pos.direction == 'long':
                    dust_pnl = (fill_exit - pos.entry_price) * pos.current_quantity
                else:
                    dust_pnl = (pos.entry_price - fill_exit) * pos.current_quantity
                self.cash += dust_margin + dust_pnl
            del self.positions[pos_id]

    def _check_all_exits(self, row, ts, bar_idx):
        """Check exits for all open positions."""
        from engine.runtime.context import RuntimeContext

        regime_label = row.get('regime_label', 'neutral') if hasattr(row, 'get') else 'neutral'
        bar_context = RuntimeContext(
            ts=ts, row=row,
            regime_probs={regime_label: 1.0},
            regime_label=regime_label,
            adapted_params={}, thresholds={},
        )

        for pos_id in list(self.positions.keys()):
            pos = self.positions[pos_id]

            # Hard stop loss
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
                self._close_position(pos_id, exit_price, ts, "stop_loss", 1.0)
                continue

            # ExitLogic
            pos_adapter = _PosAdapter(pos)
            exit_signal = self.exit_logic.check_exit(
                bar=row, position=pos_adapter,
                archetype=pos.archetype, context=bar_context,
            )

            if exit_signal is not None:
                pos.executed_scale_outs = pos_adapter.metadata.get('executed_scale_outs', pos.executed_scale_outs)
                for flag_key in ('scaled_at_prev_high', 'moon_bag_taken'):
                    if pos_adapter.metadata.get(flag_key):
                        pos.entry_metadata[flag_key] = True

                if exit_signal.stop_update is not None:
                    pos.trailing_stop = exit_signal.stop_update

                if exit_signal.exit_pct > 0:
                    exit_reason = exit_signal.reason or exit_signal.exit_type
                    self._close_position(pos_id, row['close'], ts, exit_reason, exit_signal.exit_pct)
            else:
                if pos_adapter.stop_loss != pos.stop_loss:
                    pos.trailing_stop = pos_adapter.stop_loss

    def _compute_equity(self, current_price):
        """Compute current equity."""
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

    # ── Reporting ──────────────────────────────────────────────────────

    def get_per_archetype_stats(self) -> Dict[str, Dict]:
        """Compute stats per archetype."""
        results = {}
        for arch_name, trades in self.arch_trades.items():
            if not trades:
                continue

            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            total_wins = sum(t.pnl for t in winners)
            total_losses = abs(sum(t.pnl for t in losers))

            pf = total_wins / total_losses if total_losses > 0 else float('inf')
            wr = len(winners) / len(trades) * 100 if trades else 0

            equity = [self.initial_cash]
            for t in sorted(trades, key=lambda x: x.timestamp_exit):
                equity.append(equity[-1] + t.pnl)
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (e - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd

            # Composite score stats
            avg_composite = 0
            if arch_name in self.composite_score_dist:
                scores = self.composite_score_dist[arch_name]
                avg_composite = np.mean(scores) if scores else 0

            results[arch_name] = {
                'trades': len(trades),
                'wins': len(winners),
                'losses': len(losers),
                'win_rate': wr,
                'profit_factor': pf,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': np.mean([t.pnl for t in trades]),
                'max_dd': max_dd,
                'avg_hold_hours': np.mean([t.duration_hours for t in trades]),
                'direction': trades[0].direction if trades else 'unknown',
                'avg_composite_score': avg_composite,
            }
        return results

    def get_aggregate_stats(self) -> Dict:
        """Compute aggregate stats."""
        if not self.trades:
            return {'total_trades': 0, 'profit_factor': 0, 'total_pnl': 0,
                    'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))

        pf = total_wins / total_losses if total_losses > 0 else float('inf')

        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = float(np.min(dd))

        if len(eq) > 10:
            returns = np.diff(eq) / eq[:-1]
            returns = returns[np.isfinite(returns)]
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(8760)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(self.trades) * 100,
            'profit_factor': pf,
            'total_pnl': sum(t.pnl for t in self.trades),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_hold_hours': np.mean([t.duration_hours for t in self.trades]),
        }

    def print_results(self, compare_results=None):
        """Print results with optional comparison."""
        arch_stats = self.get_per_archetype_stats()
        agg = self.get_aggregate_stats()

        mode_label = self.mode.upper()
        print(f"\n{'=' * 120}")
        print(f"{mode_label} BACKTEST RESULTS")
        print(f"{'=' * 120}")

        header = (f"{'Archetype':<25s} {'Dir':<6s} {'Trades':>7s} {'Wins':>6s} {'WR%':>7s} "
                  f"{'PF':>7s} {'PnL($)':>10s} {'AvgPnL':>8s} {'MaxDD%':>8s} {'AvgHold':>8s}")
        if self.mode == 'composite':
            header += f" {'AvgScore':>9s}"
        print(f"\n{header}")
        print("-" * 120)

        sorted_archs = sorted(arch_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

        for name, s in sorted_archs:
            pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] < 100 else "inf"
            line = (f"{name:<25s} {s['direction']:<6s} {s['trades']:>7d} {s['wins']:>6d} "
                    f"{s['win_rate']:>6.1f}% {pf_str:>7s} "
                    f"${s['total_pnl']:>9,.0f} ${s['avg_pnl']:>7,.0f} "
                    f"{s['max_dd']:>7.1f}% {s['avg_hold_hours']:>7.0f}h")
            if self.mode == 'composite':
                conf = get_confirmation(name)
                line += f" {s['avg_composite_score']:>4.1f}/{conf.max_possible}"
            print(line)

        print("-" * 120)
        agg_pf = f"{agg['profit_factor']:.2f}" if agg['profit_factor'] < 100 else "inf"
        print(f"{'TOTAL':<25s} {'':>6s} {agg['total_trades']:>7d} {agg['winning_trades']:>6d} "
              f"{agg['win_rate']:>6.1f}% {agg_pf:>7s} "
              f"${agg['total_pnl']:>9,.0f} {'':>8s} "
              f"{agg['max_drawdown']:>7.1f}% {agg['avg_hold_hours']:>7.0f}h")
        print(f"\nSharpe: {agg['sharpe_ratio']:.2f} | "
              f"Raw signals: {self.raw_signals} | "
              f"Composite rejected: {self.composite_rejected} | "
              f"Traded: {self.signals_traded}")

        # Composite score distribution
        if self.mode == 'composite' and self.composite_score_dist:
            print(f"\n{'=' * 120}")
            print(f"COMPOSITE SCORE DISTRIBUTION (all signals, including rejected)")
            print(f"{'=' * 120}")
            for arch_name in sorted(self.composite_score_dist.keys()):
                scores = self.composite_score_dist[arch_name]
                conf = get_confirmation(arch_name)
                counts = {}
                for s in scores:
                    counts[s] = counts.get(s, 0) + 1
                dist_str = " | ".join(f"{k}pt:{v}" for k, v in sorted(counts.items()))
                pct_passing = sum(1 for s in scores if s >= conf.min_score) / len(scores) * 100
                print(f"  {arch_name:<25s} min={conf.min_score}/{conf.max_possible} | "
                      f"pass={pct_passing:.0f}% | {dist_str}")

        # Comparison table
        if compare_results:
            print(f"\n{'=' * 120}")
            print(f"COMPARISON: {' vs '.join(r[0] for r in compare_results)}")
            print(f"{'=' * 120}")
            print(f"{'Metric':<20s}", end='')
            for label, _ in compare_results:
                print(f" {label:>14s}", end='')
            print()
            print("-" * (20 + 15 * len(compare_results)))

            metrics = ['total_trades', 'profit_factor', 'total_pnl', 'win_rate',
                       'max_drawdown', 'sharpe_ratio']
            labels = ['Trades', 'PF', 'PnL', 'Win Rate%', 'MaxDD%', 'Sharpe']

            for metric, label in zip(metrics, labels):
                print(f"{label:<20s}", end='')
                for _, stats in compare_results:
                    val = stats.get(metric, 0)
                    if metric == 'total_pnl':
                        print(f" ${val:>13,.0f}", end='')
                    elif metric == 'total_trades':
                        print(f" {val:>14d}", end='')
                    elif metric in ('win_rate', 'max_drawdown'):
                        print(f" {val:>13.1f}%", end='')
                    else:
                        print(f" {val:>14.2f}", end='')
                print()

        print(f"\n{'=' * 120}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Composite Confirmation Backtester')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--feature-store', default=DEFAULT_FEATURE_STORE)
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--initial-cash', type=float, default=100_000.0)
    parser.add_argument('--commission-rate', type=float, default=0.0002)
    parser.add_argument('--slippage-bps', type=float, default=3.0)
    parser.add_argument('--mode', choices=['structural', 'composite', 'production', 'all'],
                       default='composite',
                       help='Mode: structural (raw), composite (N-of-M), production (fusion+CMI), all (compare)')
    parser.add_argument('--archetype', type=str, default=None,
                       help='Comma-separated archetype filter')
    parser.add_argument('--min-score-override', type=int, default=None,
                       help='Override min_score for all archetypes (for sweep testing)')
    parser.add_argument('--max-concurrent', type=int, default=0,
                       help='Max concurrent positions per archetype (0=unlimited)')
    parser.add_argument('--output-dir', default='results/composite')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    arch_filter = None
    if args.archetype:
        arch_filter = [a.strip() for a in args.archetype.split(',')]

    # Load data
    print("Loading feature store...")
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = json.load(f)
    features_df = pd.read_parquet(str(PROJECT_ROOT / args.feature_store))
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index)
    features_df = features_df.sort_index()
    print(f"Loaded {len(features_df):,} bars")

    # Determine modes to run
    modes = ['structural', 'composite'] if args.mode == 'all' else [args.mode]

    # Handle production mode
    if 'production' in modes:
        modes.remove('production')
        # Production uses the main backtester
        print(f"\nRunning PRODUCTION backtester...")
        from bin.backtest_v11_standalone import StandaloneBacktestEngine
        prod_config = json.loads(json.dumps(config))  # deep copy
        prod = StandaloneBacktestEngine(
            config=prod_config,
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            slippage_bps=args.slippage_bps,
            features_df=features_df.copy(),
        )
        prod.run(start_date=args.start_date, end_date=args.end_date)
        prod_stats = prod.get_performance_stats()
        print(f"Production: {prod_stats.get('total_trades', 0)} trades, "
              f"PF={prod_stats.get('profit_factor', 0):.2f}, "
              f"PnL=${prod_stats.get('total_pnl', 0):,.0f}")

    all_results = []

    for mode in modes:
        bt = CompositeBacktester(
            config=json.loads(json.dumps(config)),
            features_df=features_df.copy(),
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            slippage_bps=args.slippage_bps,
            mode=mode,
            archetype_filter=arch_filter,
            min_score_override=args.min_score_override,
            max_concurrent=args.max_concurrent,
        )
        bt.run(start_date=args.start_date, end_date=args.end_date)

        stats = bt.get_aggregate_stats()
        all_results.append((mode, stats))

        # Print per-mode results
        compare = all_results if len(all_results) > 1 else None
        bt.print_results(compare_results=compare)

        # Save trade log
        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if bt.trades:
            trade_data = [{
                'entry_time': t.timestamp_entry,
                'exit_time': t.timestamp_exit,
                'archetype': t.archetype,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'duration_hours': t.duration_hours,
                'fusion_score': t.fusion_score,
                'composite_score': t.composite_score,
                'composite_max': t.composite_max,
                'exit_reason': t.exit_reason,
            } for t in bt.trades]
            trade_df = pd.DataFrame(trade_data)
            trade_path = output_dir / f'trades_{mode}.csv'
            trade_df.to_csv(trade_path, index=False)
            print(f"Trade log saved: {trade_path}")

    # If production was also requested, add to comparison
    if args.mode == 'all':
        # Re-run production for final comparison
        print(f"\nRunning PRODUCTION backtester for comparison...")
        from bin.backtest_v11_standalone import StandaloneBacktestEngine
        prod_config = json.loads(json.dumps(config))
        prod = StandaloneBacktestEngine(
            config=prod_config,
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            slippage_bps=args.slippage_bps,
            features_df=features_df.copy(),
        )
        prod.run(start_date=args.start_date, end_date=args.end_date)
        prod_stats = prod.get_performance_stats()
        all_results.append(('production', prod_stats))

        print(f"\n{'=' * 120}")
        print(f"FINAL COMPARISON: structural vs composite vs production")
        print(f"{'=' * 120}")
        print(f"{'Metric':<20s}", end='')
        for label, _ in all_results:
            print(f" {label:>14s}", end='')
        print()
        print("-" * (20 + 15 * len(all_results)))

        for metric, label in [('total_trades', 'Trades'), ('profit_factor', 'PF'),
                                ('total_pnl', 'PnL'), ('win_rate', 'Win Rate%'),
                                ('max_drawdown', 'MaxDD%'), ('sharpe_ratio', 'Sharpe')]:
            print(f"{label:<20s}", end='')
            for _, stats in all_results:
                val = stats.get(metric, 0)
                if metric == 'total_pnl':
                    print(f" ${val:>13,.0f}", end='')
                elif metric == 'total_trades':
                    print(f" {val:>14d}", end='')
                elif metric in ('win_rate', 'max_drawdown'):
                    print(f" {val:>13.1f}%", end='')
                else:
                    print(f" {val:>14.2f}", end='')
            print()
        print(f"{'=' * 120}")

    print("\nDone.")


if __name__ == '__main__':
    main()
