#!/usr/bin/env python3
"""
V11 Shadow Runner for Bull Machine

Shadow-mode signal generation: fetches real Binance data (or replays historical),
computes features, generates signals using IsolatedArchetypeEngine, and logs results.
Does NOT place any orders.

Usage:
    # Live mode (real Binance data, hourly loop):
    python3 bin/live/v11_shadow_runner.py --live

    # Replay mode (historical feature store, for parity testing):
    python3 bin/live/v11_shadow_runner.py --replay --start-date 2024-01-01 --end-date 2024-03-31

    # Replay from raw candles (test feature computer parity):
    python3 bin/live/v11_shadow_runner.py --replay-raw --start-date 2024-01-01 --end-date 2024-03-31
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.integrations.isolated_archetype_engine import IsolatedArchetypeEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackedPosition:
    """Virtual position for shadow tracking (mirrors backtester)."""
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
    regime_at_entry: str
    atr_at_entry: float
    bars_held: int = 0
    executed_scale_outs: List[float] = field(default_factory=list)
    total_exits_pct: float = 0.0
    trailing_stop: Optional[float] = None
    entry_narrative: Optional[Dict] = None
    entry_factor_attribution: Optional[Dict] = None
    threshold_at_entry: float = 0.0
    risk_temp_at_entry: float = 0.0
    instability_at_entry: float = 0.0
    crisis_prob_at_entry: float = 0.0
    threshold_margin: float = 0.0  # fusion_score - threshold (negative = would have been rejected)
    would_have_passed: bool = True  # False if signal only passed due to bypass

    def to_dict(self):
        d = asdict(self)
        d['entry_time'] = str(d['entry_time'])
        return d

    @classmethod
    def from_dict(cls, d):
        d['entry_time'] = pd.Timestamp(d['entry_time'])
        # Handle state files saved before entry_factor_attribution was added
        if 'entry_factor_attribution' not in d:
            d['entry_factor_attribution'] = None
        # Handle state files saved before threshold metadata was added
        d.setdefault('threshold_at_entry', 0.0)
        d.setdefault('risk_temp_at_entry', 0.0)
        d.setdefault('instability_at_entry', 0.0)
        d.setdefault('crisis_prob_at_entry', 0.0)
        d.setdefault('threshold_margin', 0.0)
        d.setdefault('would_have_passed', True)
        return cls(**d)


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
    factor_attribution: Optional[Dict] = None
    threshold_at_entry: float = 0.0
    risk_temp_at_entry: float = 0.0
    instability_at_entry: float = 0.0
    crisis_prob_at_entry: float = 0.0
    threshold_margin: float = 0.0
    would_have_passed: bool = True
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr_at_entry: float = 0.0


# ---------------------------------------------------------------------------
# Shadow Runner
# ---------------------------------------------------------------------------

class V11ShadowRunner:
    """
    Shadow-mode signal generator using v11 IsolatedArchetypeEngine.

    Replicates the backtester's exact signal flow:
    1. Generate signals from IsolatedArchetypeEngine
    2. Filter: disabled → regime_restrict → fusion_threshold → position_limits
    3. Allocate via PortfolioAllocator
    4. Track virtual positions (entries, exits, PnL)
    5. Log everything to CSV
    """

    def __init__(
        self,
        config_path: str = 'configs/bull_machine_isolated_v11_fixed.json',
        initial_cash: float = 100_000.0,
        commission_rate: float = 0.0004,
        slippage_bps: float = 5.0,
    ):
        # Load config
        config_file = PROJECT_ROOT / config_path
        with open(config_file) as f:
            self.config = json.load(f)

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps

        # Initialize IsolatedArchetypeEngine (same as backtester)
        regime_cfg = self.config.get('regime_classifier', {})
        self.engine = IsolatedArchetypeEngine(
            archetype_config_dir=str(PROJECT_ROOT / self.config.get('archetype_config_dir', 'configs/archetypes/')),
            portfolio_config=self.config.get('portfolio_allocation', {}),
            enable_regime=regime_cfg.get('enabled', False),
            regime_model_path=regime_cfg.get('model_path', None),
            config=self.config,
        )

        # Adaptive fusion config (replaces fusion_thresholds_by_regime)
        self.adaptive_fusion = self.config.get('adaptive_fusion', {})
        # Bypass threshold mode: let ALL signals through for data collection
        self.bypass_threshold = self.adaptive_fusion.get('bypass_threshold', False)
        if self.bypass_threshold:
            logger.warning("[CONFIG] bypass_threshold=True — ALL signals will trade regardless of threshold (data collection mode)")
        # Legacy: fusion_thresholds_by_regime (used as fallback if adaptive_fusion not enabled)
        self.fusion_thresholds_by_regime = self.config.get('fusion_thresholds_by_regime', {})
        self.max_positions_by_regime = self.config.get('max_positions_by_regime', {})
        self.disabled_archetypes = set(self.config.get('disabled_archetypes', []))

        # Track last computed dynamic threshold for heartbeat reporting
        self.last_dynamic_threshold = 0.18
        self.last_risk_temp = 0.0
        self.last_instability = 0.0
        self.last_crisis_prob = 0.0
        self.last_cmi_breakdown = {}  # Detailed CMI component breakdown for dashboard

        # Load optimized CMI weights if available (for dual-threshold comparison)
        self.cmi_weights_optimized = None
        self.last_cmi_comparison = {}
        opt_path = PROJECT_ROOT / 'configs' / 'optimized' / 'cmi_weights_optimized.json'
        if opt_path.exists():
            try:
                with open(opt_path) as f:
                    self.cmi_weights_optimized = json.load(f)
                logger.info(f"[CMI] Loaded optimized weights from {opt_path}")
            except Exception as e:
                logger.warning(f"[CMI] Failed to load optimized weights: {e}")

        # Current (hand-tuned) CMI weights — read from config or use defaults
        cmi_w = self.adaptive_fusion.get('cmi_weights', {})
        self.cmi_weights_current = {
            'risk_temp': cmi_w.get('risk_temp', {'trend_align': 0.45, 'trend_strength': 0.25, 'sentiment': 0.15, 'dd_score': 0.15}),
            'instability': cmi_w.get('instability', {'chop': 0.35, 'adx_weakness': 0.25, 'wick_score': 0.20, 'vol_instab': 0.20}),
            'crisis': cmi_w.get('crisis', {'base_crisis': 0.60, 'vol_shock': 0.20, 'sentiment_crisis': 0.20}),
        }
        raw_allowed = self.config.get('archetype_allowed_regimes', {})
        self.archetype_allowed_regimes = {k: v for k, v in raw_allowed.items() if k != 'notes'}

        # Position sizing
        sizing_cfg = self.config.get('position_sizing', {})
        self.risk_per_trade = sizing_cfg.get('risk_per_trade_pct', 0.02)
        self.max_position_pct = sizing_cfg.get('max_position_size_pct', 0.15)
        self.leverage = self.config.get('leverage', 1.0)

        # State
        self.positions: Dict[str, TrackedPosition] = {}
        self.trades: List[CompletedTrade] = []
        self.bar_index = 0
        self.equity_curve: List[float] = []
        self.running = True

        # Stats
        self.total_signals = 0
        self.signals_allocated = 0
        self.signals_rejected = 0
        # Rejected signal details for dashboard (cleared each bar, written by runner)
        self.last_bar_signals: List[dict] = []

        # Phantom trade tracker — tracks what WOULD have happened for rejected signals
        self.phantom_positions: Dict[str, TrackedPosition] = {}
        self.phantom_trades: List[CompletedTrade] = []
        self.phantom_signals_total = 0

        # Output directory
        self.output_dir = PROJECT_ROOT / 'results' / 'live_signals'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Signal log file
        self.signal_log_path = self.output_dir / 'signals.csv'
        self._init_signal_log()

        logger.info("V11ShadowRunner initialized")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Cash: ${initial_cash:,.0f}")
        logger.info(f"  Commission: {commission_rate*10000:.1f} bps | Slippage: {slippage_bps:.1f} bps")
        logger.info(f"  Disabled: {len(self.disabled_archetypes)} archetypes")
        logger.info(f"  Regime restrictions: {self.archetype_allowed_regimes}")

    def _init_signal_log(self):
        """Initialize signal log CSV with headers."""
        expected_header = (
            'timestamp,archetype,direction,fusion_score,regime,'
            'entry_price,stop_loss,take_profit,action,reason,'
            'threshold,threshold_margin,would_have_passed,'
            'risk_temp,instability,crisis_prob,narrative\n'
        )
        if self.signal_log_path.exists():
            # Check if existing file has old headers (missing threshold cols)
            with open(self.signal_log_path, 'r') as f:
                first_line = f.readline()
            if 'threshold,' not in first_line:
                # Migrate: rename old file and create new with updated headers
                backup = self.signal_log_path.with_suffix('.csv.bak')
                self.signal_log_path.rename(backup)
                logger.info(f"Migrated old signals.csv to {backup} (added threshold columns)")
                with open(self.signal_log_path, 'w') as f:
                    f.write(expected_header)
        else:
            with open(self.signal_log_path, 'w') as f:
                f.write(expected_header)

    # ------------------------------------------------------------------
    # Signal processing (mirrors backtester lines 332-419)
    # ------------------------------------------------------------------

    def _compute_adaptive_threshold(self, features: pd.Series):
        """
        Compute CMI v0 dynamic threshold from features (same as backtester).
        Called every bar to keep state current for heartbeat reporting.
        """
        if not self.adaptive_fusion.get('enabled', False):
            return

        def _get(col, default=0.0):
            val = features.get(col, default)
            try:
                val = float(val)
                if val != val:  # NaN check
                    return float(default)
                return val
            except (TypeError, ValueError):
                return float(default)

        # --- risk_temperature [0-1]: 0=cold/bear, 1=hot/bull ---
        p_above_50 = _get('price_above_ema_50', 0)
        ema_50_200 = _get('ema_50_above_200', 0)
        if p_above_50 and ema_50_200:
            trend_align = 1.0
        elif p_above_50:
            trend_align = 0.6
        elif ema_50_200:
            trend_align = 0.4
        else:
            trend_align = 0.0

        adx = _get('adx', _get('adx_14', 20.0))
        trend_strength = min(adx / 40.0, 1.0)
        fear_greed = _get('fear_greed_norm', 0.5)
        dd_persist = _get('drawdown_persistence', 0.9)
        dd_score = max(1.0 - dd_persist, 0.0)

        # --- risk_temperature using config-driven weights ---
        rt_w = self.cmi_weights_current['risk_temp']
        risk_temp = (rt_w.get('trend_align', 0.45) * trend_align +
                     rt_w.get('trend_strength', 0.25) * trend_strength +
                     rt_w.get('sentiment', 0.15) * fear_greed +
                     rt_w.get('dd_score', 0.15) * dd_score)

        # --- instability_score [0-1]: 0=stable, 1=choppy ---
        chop = _get('chop_score', 0.5)
        adx_weakness = max(1.0 - adx / 40.0, 0.0)
        wick_sc = min(_get('wick_ratio', 1.0) / 5.0, 1.0)
        vol_instab = min(abs(_get('volume_z_7d', 0.0)) / 2.5, 1.0)
        inst_w = self.cmi_weights_current['instability']
        instability = (inst_w.get('chop', 0.35) * chop +
                       inst_w.get('adx_weakness', 0.25) * adx_weakness +
                       inst_w.get('wick_score', 0.20) * wick_sc +
                       inst_w.get('vol_instab', 0.20) * vol_instab)

        # --- crisis_prob [0-1]: pure stress measurement ---
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
        rv = _get('rv_20d', 0.6)
        vol_shock = min(max(rv - 0.8, 0.0) / 0.4, 1.0)
        sentiment_crisis = max(0.0, (0.20 - fear_greed) / 0.20)
        cr_w = self.cmi_weights_current['crisis']
        crisis_prob = (cr_w.get('base_crisis', 0.60) * base_crisis +
                       cr_w.get('vol_shock', 0.20) * vol_shock +
                       cr_w.get('sentiment_crisis', 0.20) * sentiment_crisis)

        # Dynamic threshold
        base_threshold = self.adaptive_fusion.get('base_threshold', 0.18)
        temp_range = self.adaptive_fusion.get('temp_range', 0.35)
        instab_range = self.adaptive_fusion.get('instab_range', 0.15)
        flat_threshold = base_threshold + (1.0 - risk_temp) * temp_range + instability * instab_range

        # --- Dual-threshold: compute optimized threshold for comparison ---
        opt_threshold = None
        if self.cmi_weights_optimized:
            opt_cmi = self.cmi_weights_optimized.get('cmi_weights', {})
            opt_params = self.cmi_weights_optimized.get('threshold_params', {})
            # Recompute risk_temp with optimized weights
            ort_w = opt_cmi.get('risk_temp', rt_w)
            opt_risk_temp = (ort_w.get('trend_align', 0.45) * trend_align +
                             ort_w.get('trend_strength', 0.25) * trend_strength +
                             ort_w.get('sentiment', 0.15) * fear_greed +
                             ort_w.get('dd_score', 0.15) * dd_score)
            # Recompute instability with optimized weights
            oi_w = opt_cmi.get('instability', inst_w)
            opt_instability = (oi_w.get('chop', 0.35) * chop +
                               oi_w.get('adx_weakness', 0.25) * adx_weakness +
                               oi_w.get('wick_score', 0.20) * wick_sc +
                               oi_w.get('vol_instab', 0.20) * vol_instab)
            # Recompute crisis_prob with optimized weights
            oc_w = opt_cmi.get('crisis', cr_w)
            opt_crisis = (oc_w.get('base_crisis', 0.60) * base_crisis +
                          oc_w.get('vol_shock', 0.20) * vol_shock +
                          oc_w.get('sentiment_crisis', 0.20) * sentiment_crisis)
            opt_temp_range = opt_params.get('temp_range', temp_range)
            opt_instab_range = opt_params.get('instab_range', instab_range)
            opt_threshold = base_threshold + (1.0 - opt_risk_temp) * opt_temp_range + opt_instability * opt_instab_range
            self.last_cmi_comparison = {
                "hand_tuned": {
                    "risk_temp": round(risk_temp, 4),
                    "instability": round(instability, 4),
                    "crisis_prob": round(crisis_prob, 4),
                    "threshold": round(flat_threshold, 4),
                },
                "optimized": {
                    "risk_temp": round(opt_risk_temp, 4),
                    "instability": round(opt_instability, 4),
                    "crisis_prob": round(opt_crisis, 4),
                    "threshold": round(opt_threshold, 4),
                },
                "agreement": abs(flat_threshold - opt_threshold) < 0.05,
                "delta": round(opt_threshold - flat_threshold, 4),
            }

        # Store for use in signal filtering and heartbeat
        self.last_dynamic_threshold = flat_threshold
        self.last_risk_temp = risk_temp
        self.last_instability = instability
        self.last_crisis_prob = crisis_prob

        # Store detailed CMI component breakdown for dashboard
        self.last_cmi_breakdown = {
            "risk_temp_components": {
                "trend_align": round(trend_align, 4),
                "trend_strength": round(trend_strength, 4),
                "sentiment": round(fear_greed, 4),
                "dd_score": round(dd_score, 4),
            },
            "risk_temp_weights": dict(rt_w),
            "instability_components": {
                "chop": round(chop, 4),
                "adx_weakness": round(adx_weakness, 4),
                "wick_score": round(wick_sc, 4),
                "vol_instability": round(vol_instab, 4),
            },
            "instability_weights": dict(inst_w),
            "crisis_components": {
                "base_crisis": round(base_crisis, 4),
                "vol_shock": round(vol_shock, 4),
                "sentiment_crisis": round(sentiment_crisis, 4),
                "stress_signals_count": crisis_signals,
            },
            "crisis_weights": dict(cr_w),
            "raw_features": {
                "adx": round(adx, 2),
                "fear_greed_norm": round(fear_greed, 4),
                "chop_score": round(chop, 4),
                "wick_ratio": round(_get('wick_ratio', 1.0), 4),
                "volume_z_7d": round(_get('volume_z_7d', 0.0), 4),
                "drawdown_persistence": round(dd_persist, 4),
                "rv_20d": round(rv, 4),
                "crash_frequency_7d": int(crash_freq),
                "crisis_persistence": round(crisis_persist, 4),
                "price_above_ema_50": int(p_above_50),
                "ema_50_above_200": int(ema_50_200),
            },
            "threshold_config": {
                "base_threshold": base_threshold,
                "temp_range": temp_range,
                "instab_range": instab_range,
                "dynamic_threshold": round(flat_threshold, 4),
            },
        }

    # ------------------------------------------------------------------
    # Signal narrative generation
    # ------------------------------------------------------------------

    def _build_signal_narrative(
        self, archetype: str, features: pd.Series,
        fusion_score: float, threshold: float,
        direction: str, entry_price: float,
        domain_scores: Optional[Dict] = None,
    ) -> Dict:
        """Build a human-readable narrative explaining what triggered a signal.

        Uses domain scores (wyckoff, liquidity, momentum, smc) from the
        archetype's actual fusion computation + real feature values to explain
        WHY the engine took this trade like a trader would explain it.
        """

        ds = domain_scores or {}
        w_score = ds.get('wyckoff', 0.0)
        l_score = ds.get('liquidity', 0.0)
        m_score = ds.get('momentum', 0.0)
        s_score = ds.get('smc', 0.0)

        def _g(col, default=0.0):
            val = features.get(col, default)
            try:
                v = float(val)
                return default if v != v else v
            except (TypeError, ValueError):
                return float(default)

        price_str = f"${entry_price:,.0f}"
        margin = fusion_score - threshold
        regime = str(features.get('regime_label', 'neutral'))
        bypassed = margin < 0

        # Gate values always included
        rsi = round(_g('rsi_14', 50), 1)
        adx = round(_g('adx', _g('adx_14', 20)), 1)
        atr = round(_g('atr_14', 0), 2)
        wick = round(_g('wick_ratio', 1.0), 2)
        boms = round(_g('tf1h_boms_strength', _g('boms_strength', 0)), 3)
        bos_bull = int(_g('bos_bullish', 0))
        bos_bear = int(_g('bos_bearish', 0))
        vol_z = round(_g('volume_z_7d', 0), 2)
        fz = round(_g('funding_Z', 0), 2)
        chop = round(_g('chop_score', 0.5), 3)
        fvg = int(_g('tf1h_fvg_present', 0))
        fg_raw = round(_g('fear_greed_norm', 0.5) * 100, 0)

        gate_values = {
            'rsi_14': rsi, 'adx': adx, 'atr_14': atr, 'wick_ratio': wick,
            'boms_strength': boms, 'bos_bullish': bos_bull, 'bos_bearish': bos_bear,
            'volume_z': vol_z, 'funding_z': fz, 'chop_score': chop,
            'fvg_present': fvg, 'fear_greed': fg_raw,
        }

        # --- Per-archetype headline: what actually happened ---
        dir_word = 'long' if direction == 'long' else 'short'
        support_or_resist = 'support' if direction == 'long' else 'resistance'

        if archetype == 'trap_within_trend':
            headline = (
                f"False breakdown in {'bullish' if direction == 'long' else 'bearish'} "
                f"trend at {price_str}. Wick rejected {support_or_resist} at {wick:.1f}x body, "
                f"trapping {'bears' if direction == 'long' else 'bulls'} who sold the break."
            )
        elif archetype == 'wick_trap':
            headline = (
                f"{'Lower' if direction == 'long' else 'Upper'} wick anomaly ({wick:.1f}x body) "
                f"at {price_str} rejected {support_or_resist}. RSI at {rsi:.0f} "
                f"{'(oversold)' if rsi < 40 else '(neutral)'}, "
                f"{'BOS confirmed structural shift' if (bos_bull and direction == 'long') or (bos_bear and direction == 'short') else 'awaiting BOS confirmation'}."
            )
        elif archetype == 'liquidity_vacuum':
            headline = (
                f"Capitulation spike at {price_str} created liquidity vacuum. "
                f"{'Panic selling' if direction == 'long' else 'Panic buying'} "
                f"exhausted {'sellers' if direction == 'long' else 'buyers'} — "
                f"volume surge {vol_z:+.1f}σ with Wyckoff climax score {w_score:.2f}."
            )
        elif archetype == 'liquidity_sweep':
            headline = (
                f"Price swept {'below' if direction == 'long' else 'above'} {price_str}, "
                f"triggering stop losses. BOMS displacement {boms:.3f} "
                f"{'confirms order flow reversal' if boms > 0.1 else 'weak — marginal signal'}. "
                f"RSI {rsi:.0f}."
            )
        elif archetype == 'retest_cluster':
            headline = (
                f"Fakeout then real move at {price_str}. Initial break shook out weak hands, "
                f"volume {vol_z:+.1f}σ on the retest confirms genuine demand. "
                f"RSI {rsi:.0f}, liquidity recovering."
            )
        elif archetype == 'spring':
            headline = (
                f"Wyckoff spring at {price_str} — price briefly broke {support_or_resist}, "
                f"shaking out {'bears' if direction == 'long' else 'bulls'}, "
                f"then reversed on {'displacement' if boms > 0 else 'wick rejection'} "
                f"(wick {wick:.1f}x body). Wyckoff score {w_score:.2f}."
            )
        elif archetype == 'order_block_retest':
            headline = (
                f"Smart money re-entry at {price_str} — price retested previous order block. "
                f"BOMS {boms:.3f}, "
                f"{'BOS confirms structural continuation' if (bos_bull and direction == 'long') or (bos_bear and direction == 'short') else 'no BOS yet — awaiting confirmation'}. "
                f"Wyckoff score {w_score:.2f}."
            )
        elif archetype == 'failed_continuation':
            headline = (
                f"{'Bearish' if direction == 'long' else 'Bullish'} move failed to continue at {price_str}. "
                f"{'FVG present' if fvg else 'No FVG'}, RSI {rsi:.0f} "
                f"{'weak — sellers exhausted' if rsi < 55 and direction == 'long' else ''}, "
                f"ADX {adx:.0f} {'falling — trend dying' if adx < 20 else 'steady'}."
            )
        elif archetype == 'confluence_breakout':
            headline = (
                f"Tight consolidation broke at {price_str}. ATR compressed (${atr:.0f}), "
                f"BOMS {boms:.3f} {'confirms breakout force' if boms > 0.1 else 'weak breakout'}. "
                f"Coiled range expanding."
            )
        elif archetype == 'fvg_continuation':
            headline = (
                f"Fair Value Gap created on {'bullish' if direction == 'long' else 'bearish'} "
                f"displacement at {price_str}. "
                f"{'BOS confirms structural break' if (bos_bull or bos_bear) else 'Displacement without BOS'}. "
                f"Momentum score {m_score:.2f}."
            )
        elif archetype == 'funding_divergence':
            headline = (
                f"Funding rate divergence at {price_str}: funding Z={fz:+.2f} "
                f"{'(shorts overcrowded — squeeze setup)' if fz < -0.5 else '(funding diverging from price)'}. "
                f"Price holding above lows despite negative funding."
            )
        elif archetype == 'long_squeeze':
            headline = (
                f"Long squeeze setup at {price_str}: funding Z={fz:+.2f} "
                f"{'(longs overcrowded)' if fz > 0.5 else ''}, RSI {rsi:.0f} overheated. "
                f"Liquidation cascade risk building."
            )
        elif archetype == 'liquidity_compression':
            headline = (
                f"Range compression at {price_str}: ATR ${atr:.0f} (low), "
                f"chop {chop:.2f}. Energy building for expansion — "
                f"liquidity score {l_score:.2f}."
            )
        elif archetype == 'exhaustion_reversal':
            headline = (
                f"Momentum exhaustion at {price_str}: RSI {rsi:.0f} "
                f"{'extreme' if rsi > 65 or rsi < 35 else 'elevated'}, "
                f"ATR spiking (${atr:.0f}), volume {vol_z:+.1f}σ. "
                f"{'Selling' if direction == 'long' else 'Buying'} climax — "
                f"{'sellers' if direction == 'long' else 'buyers'} exhausted."
            )
        elif archetype == 'volume_fade_chop':
            headline = (
                f"Volume fade in choppy range at {price_str}. Vol {vol_z:+.1f}σ (dried up), "
                f"RSI {rsi:.0f}, ADX {adx:.0f} (no trend). "
                f"Mean-reversion scalp."
            )
        elif archetype == 'whipsaw':
            headline = (
                f"Whipsaw at {price_str}: wick {wick:.1f}x body in range-bound market. "
                f"Vol {vol_z:+.1f}σ (low), chop {chop:.2f}. "
                f"Fading the {'dip' if direction == 'long' else 'rip'} back to mid-range."
            )
        else:
            headline = (
                f"{archetype.replace('_', ' ').title()} {dir_word} at {price_str}. "
                f"Fusion {fusion_score:.3f}."
            )

        # --- Confluence factors (what supports this trade) ---
        confluence = []

        # Domain-score based confluence (archetype-aware)
        if w_score > 0.3:
            confluence.append(f"Wyckoff structure confirmed ({w_score:.2f})")
        elif w_score > 0.1:
            confluence.append(f"Wyckoff context supportive ({w_score:.2f})")
        if l_score > 0.4:
            confluence.append(f"Strong liquidity signal ({l_score:.2f})")
        elif l_score > 0.2:
            confluence.append(f"Liquidity present ({l_score:.2f})")
        if s_score > 0.3:
            confluence.append(f"SMC structure confirmed ({s_score:.2f})")
        if m_score > 0.4:
            confluence.append(f"Strong momentum ({m_score:.2f})")

        # Feature-based confluence
        if bos_bull and direction == 'long':
            confluence.append('Break of Structure (bullish)')
        if bos_bear and direction == 'short':
            confluence.append('Break of Structure (bearish)')
        if boms > 0.3:
            confluence.append(f"BOMS displacement {boms:.2f}")
        if adx > 25:
            confluence.append(f"Trending market (ADX {adx:.0f})")
        if fvg:
            confluence.append('Fair Value Gap present')
        if abs(vol_z) > 1.5:
            confluence.append(f"Volume anomaly ({vol_z:+.1f}σ)")
        if wick > 2.0:
            confluence.append(f"Wick rejection ({wick:.1f}x body)")

        # --- Risk factors ---
        risk_factors = []
        if abs(fz) > 1.0:
            risk_factors.append(f"Funding {'elevated' if fz > 0 else 'negative'} (Z={fz:+.1f})")
        if chop > 0.7:
            risk_factors.append(f"Choppy market (chop={chop:.2f})")
        if fg_raw < 20:
            risk_factors.append(f"Extreme Fear (F&G={fg_raw:.0f})")
        elif fg_raw > 80:
            risk_factors.append(f"Extreme Greed (F&G={fg_raw:.0f})")
        if adx < 15:
            risk_factors.append(f"No clear trend (ADX {adx:.0f})")
        if bypassed:
            risk_factors.append(f"BYPASSED threshold (fusion {fusion_score:.3f} < threshold {threshold:.3f})")

        # --- Regime context ---
        regime_context = (
            f"{regime.replace('_', ' ').title()} "
            f"(risk_temp={self.last_risk_temp:.2f}), "
            f"threshold={threshold:.3f}, "
            f"{'BYPASSED' if bypassed else 'passed'} with "
            f"{'+' if margin >= 0 else ''}{margin:.3f} margin"
        )

        trigger = headline.split('.')[0] + '.'

        return {
            'headline': headline,
            'summary': headline,
            'trigger': trigger,
            'confluence_factors': confluence,
            'confluence': confluence,
            'regime_context': regime_context,
            'risk_factors': risk_factors,
            'gate_values': gate_values,
            'domain_scores': {
                'wyckoff': round(w_score, 3),
                'liquidity': round(l_score, 3),
                'momentum': round(m_score, 3),
                'smc': round(s_score, 3),
            },
        }

    def process_bar(self, features: pd.Series, timestamp: pd.Timestamp) -> List[dict]:
        """
        Process a single bar through the full signal pipeline.
        Returns list of signal dicts that were acted on.
        """
        self.bar_index += 1
        acted_signals = []
        # Reset per-bar signal tracking for dashboard
        self.last_bar_signals = []

        # Step 1: Update bars_held for open positions
        for pos in self.positions.values():
            pos.bars_held += 1

        # Step 2: Check exits on virtual positions
        self._check_all_exits(features, timestamp)

        # Step 2a: Check exits on phantom positions (counterfactual tracking)
        self._check_phantom_exits(features, timestamp)

        # Step 2.5: Compute adaptive threshold every bar (for heartbeat + filtering)
        self._compute_adaptive_threshold(features)

        # Step 3: Generate signals
        signals = self.engine.get_signals(
            bar=features,
            bar_index=self.bar_index,
        )

        if not signals:
            self._update_equity(features['close'])
            return acted_signals

        self.total_signals += len(signals)

        # Build tracking entries for ALL raw signals
        current_regime = str(features.get('regime_label', 'neutral'))
        for s in signals:
            self.last_bar_signals.append({
                'timestamp': str(timestamp),
                'archetype': s.archetype_id,
                'direction': s.direction,
                'fusion_score': round(s.fusion_score, 4),
                'entry_price': round(s.entry_price, 2) if s.entry_price else 0,
                'regime': current_regime,
                'status': 'pending',  # will be updated below
                'rejection_reason': '',
                'rejection_stage': '',
                'threshold': round(self.last_dynamic_threshold, 4),
                'margin': 0.0,
            })
        sig_index = {id(s): i for i, s in enumerate(signals)}

        # Step 3a: Filter disabled archetypes
        if self.disabled_archetypes:
            surviving = []
            for s in signals:
                if s.archetype_id in self.disabled_archetypes:
                    idx = sig_index[id(s)]
                    self.last_bar_signals[idx]['status'] = 'rejected'
                    self.last_bar_signals[idx]['rejection_reason'] = 'archetype disabled'
                    self.last_bar_signals[idx]['rejection_stage'] = 'disabled'
                    self.signals_rejected += 1
                else:
                    surviving.append(s)
            signals = surviving

        if not signals:
            self._update_equity(features['close'])
            return acted_signals

        # Step 3a.5: Per-archetype regime restrictions
        if self.archetype_allowed_regimes and signals:
            surviving = []
            for s in signals:
                allowed = self.archetype_allowed_regimes.get(s.archetype_id)
                if allowed and current_regime not in allowed:
                    idx = sig_index[id(s)]
                    self.last_bar_signals[idx]['status'] = 'rejected'
                    self.last_bar_signals[idx]['rejection_reason'] = f'regime {current_regime} not in {allowed}'
                    self.last_bar_signals[idx]['rejection_stage'] = 'regime_filter'
                    self.signals_rejected += 1
                else:
                    surviving.append(s)
            signals = surviving

        if not signals:
            self._update_equity(features['close'])
            return acted_signals

        # Step 3b: Adaptive fusion — continuous regime modulation (CMI v0)
        raw_signal_count = len(signals)
        max_pos = 3

        if self.adaptive_fusion.get('enabled', False):
            global_threshold = self.last_dynamic_threshold
            risk_temp = self.last_risk_temp
            instability = self.last_instability
            crisis_prob = self.last_crisis_prob
            c_coeff = self.adaptive_fusion.get('crisis_coefficient', 0.4)
            crisis_penalty = 1.0 - crisis_prob * c_coeff
            per_arch_thresholds = self.adaptive_fusion.get('per_archetype_base_threshold', {})
            base_threshold = self.adaptive_fusion.get('base_threshold', 0.18)
            temp_range = self.adaptive_fusion.get('temp_range', 0.35)
            instab_range = self.adaptive_fusion.get('instab_range', 0.15)

            adjusted_signals = []
            for s in signals:
                adjusted_fusion = s.fusion_score * crisis_penalty
                # Per-archetype dynamic threshold
                arch_base = per_arch_thresholds.get(s.archetype_id, base_threshold)
                arch_threshold = arch_base + (1.0 - risk_temp) * temp_range + instability * instab_range
                idx = sig_index[id(s)]
                self.last_bar_signals[idx]['fusion_score'] = round(adjusted_fusion, 4)
                margin = adjusted_fusion - arch_threshold
                self.last_bar_signals[idx]['margin'] = round(margin, 4)
                # Store threshold metadata on signal for position tracking
                s._threshold_at_entry = arch_threshold
                s._risk_temp_at_entry = risk_temp
                s._instability_at_entry = instability
                s._crisis_prob_at_entry = crisis_prob
                s._threshold_margin = margin
                s._would_have_passed = (adjusted_fusion >= arch_threshold)

                if adjusted_fusion >= arch_threshold:
                    s.fusion_score = adjusted_fusion
                    adjusted_signals.append(s)
                elif self.bypass_threshold:
                    # BYPASS MODE: let signal through but mark it
                    s.fusion_score = adjusted_fusion
                    adjusted_signals.append(s)
                    self.last_bar_signals[idx]['status'] = 'bypassed'
                    self.last_bar_signals[idx]['bypass_note'] = (
                        f'fusion {adjusted_fusion:.3f} < threshold {arch_threshold:.3f} '
                        f'(gap: {margin:.3f}) — BYPASSED for data collection'
                    )
                    logger.info(
                        "[BYPASS] %s: fusion=%.3f < threshold=%.3f (gap=%.3f) — letting through for data collection",
                        s.archetype_id, adjusted_fusion, arch_threshold, margin,
                    )
                else:
                    rej_reason = (
                        f'fusion {adjusted_fusion:.3f} < threshold {arch_threshold:.3f} '
                        f'(gap: {margin:.3f})'
                    )
                    self.last_bar_signals[idx]['status'] = 'rejected'
                    self.last_bar_signals[idx]['rejection_reason'] = rej_reason
                    self.last_bar_signals[idx]['rejection_stage'] = 'adaptive_threshold'
                    self.signals_rejected += 1
                    # Phantom: track what would have happened
                    self._open_phantom(s, features, current_regime, rej_reason, 'adaptive_threshold')
                    logger.debug(
                        "[FILTER] %s rejected: fusion=%.3f < threshold=%.3f (phantom opened)",
                        s.archetype_id, adjusted_fusion, arch_threshold,
                    )
            signals = adjusted_signals

            # Emergency exposure cap
            emergency_threshold = self.adaptive_fusion.get('emergency_crisis_threshold', 0.7)
            emergency_mult = self.adaptive_fusion.get('emergency_size_multiplier', 0.50)
            if crisis_prob > emergency_threshold and signals:
                for s in signals:
                    s.fusion_score *= emergency_mult
                logger.info(
                    "[EMERGENCY] crisis_prob=%.3f > %.2f — applied %.0f%% sizing cap to %d signals",
                    crisis_prob, emergency_threshold, emergency_mult * 100, len(signals),
                )

            # Stress-scaled position limit
            base_max_pos = self.adaptive_fusion.get('base_max_positions', 3)
            stress_level = max(crisis_prob, instability * 0.5)
            max_pos = max(1, round(base_max_pos * (1.0 - 0.5 * stress_level)))

            logger.info(
                "[THRESHOLD] bar=%d | dynamic_threshold=%.3f | risk_temp=%.3f | "
                "instability=%.3f | crisis_prob=%.3f | regime=%s | "
                "raw=%d | passed=%d | max_pos=%d",
                self.bar_index, global_threshold, risk_temp, instability,
                crisis_prob, current_regime, raw_signal_count, len(signals), max_pos,
            )

        elif self.fusion_thresholds_by_regime:
            regime_threshold = self.fusion_thresholds_by_regime.get(current_regime, 0.18)
            surviving = []
            for s in signals:
                if s.fusion_score >= regime_threshold:
                    surviving.append(s)
                else:
                    idx = sig_index[id(s)]
                    self.last_bar_signals[idx]['status'] = 'rejected'
                    self.last_bar_signals[idx]['rejection_reason'] = f'fusion {s.fusion_score:.3f} < regime threshold {regime_threshold:.3f}'
                    self.last_bar_signals[idx]['rejection_stage'] = 'regime_threshold'
                    self.signals_rejected += 1
            signals = surviving
            self.last_dynamic_threshold = regime_threshold

        # Step 3c: Apply position limits
        if signals:
            if not self.adaptive_fusion.get('enabled', False):
                max_pos = self.max_positions_by_regime.get(current_regime, 3) if self.max_positions_by_regime else 3
            if len(self.positions) >= max_pos:
                for s in signals:
                    idx = sig_index[id(s)]
                    rej_reason = f'position limit ({len(self.positions)}/{max_pos})'
                    self.last_bar_signals[idx]['status'] = 'rejected'
                    self.last_bar_signals[idx]['rejection_reason'] = rej_reason
                    self.last_bar_signals[idx]['rejection_stage'] = 'position_limit'
                    self._open_phantom(s, features, current_regime, rej_reason, 'position_limit')
                self.signals_rejected += len(signals)
                signals = []

        # Mark surviving signals as passed (will become allocated or portfolio-rejected)
        for s in signals:
            idx = sig_index[id(s)]
            self.last_bar_signals[idx]['status'] = 'passed'

        if not signals:
            self._update_equity(features['close'])
            return acted_signals

        # Step 4: Portfolio allocation
        current_position_archetypes = [
            pos.archetype for pos in self.positions.values()
        ]
        intents, rejections = self.engine.allocate(
            signals,
            current_positions=current_position_archetypes,
        )
        self.signals_rejected += len(rejections)

        # Update signal tracking for portfolio rejections
        for rej in rejections:
            rej_id = id(rej.signal)
            if rej_id in sig_index:
                idx = sig_index[rej_id]
                rej_reason = f'portfolio: {rej.reason}'
                self.last_bar_signals[idx]['status'] = 'rejected'
                self.last_bar_signals[idx]['rejection_reason'] = rej_reason
                self.last_bar_signals[idx]['rejection_stage'] = 'portfolio_allocator'
                self._open_phantom(rej.signal, features, current_regime, rej_reason, 'portfolio_allocator')

        # Mark allocated signals + build narratives
        for intent in intents:
            alloc_id = id(intent.signal)
            if alloc_id in sig_index:
                idx = sig_index[alloc_id]
                self.last_bar_signals[idx]['status'] = 'allocated'
                sig_meta = intent.signal.metadata or {}
                narrative = self._build_signal_narrative(
                    archetype=intent.signal.archetype_id,
                    features=features,
                    fusion_score=intent.signal.fusion_score,
                    threshold=self.last_dynamic_threshold,
                    direction=intent.signal.direction,
                    entry_price=intent.signal.entry_price,
                    domain_scores={
                        'wyckoff': sig_meta.get('wyckoff_score', 0.0),
                        'liquidity': sig_meta.get('liquidity_score', 0.0),
                        'momentum': sig_meta.get('momentum_score', 0.0),
                        'smc': sig_meta.get('smc_score', 0.0),
                    },
                )
                self.last_bar_signals[idx]['narrative'] = narrative

        # Step 5: Execute allocations (virtual)
        for intent in intents:
            sig = intent.signal
            # Retrieve pre-built narrative
            alloc_id = id(sig)
            narrative = None
            if alloc_id in sig_index:
                narrative = self.last_bar_signals[sig_index[alloc_id]].get('narrative')
            self._open_position(
                timestamp=timestamp,
                archetype=sig.archetype_id,
                direction=sig.direction,
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                fusion_score=sig.fusion_score,
                regime_label=current_regime,
                features=features,
                allocated_size_pct=intent.allocated_size_pct,
                entry_narrative=narrative,
                threshold_at_entry=getattr(sig, '_threshold_at_entry', self.last_dynamic_threshold),
                risk_temp_at_entry=getattr(sig, '_risk_temp_at_entry', self.last_risk_temp),
                instability_at_entry=getattr(sig, '_instability_at_entry', self.last_instability),
                crisis_prob_at_entry=getattr(sig, '_crisis_prob_at_entry', self.last_crisis_prob),
                threshold_margin=getattr(sig, '_threshold_margin', 0.0),
                would_have_passed=getattr(sig, '_would_have_passed', True),
            )
            sig_entry = {
                'timestamp': str(timestamp),
                'archetype': sig.archetype_id,
                'direction': sig.direction,
                'fusion_score': sig.fusion_score,
                'regime': current_regime,
                'entry_price': sig.entry_price,
                'stop_loss': sig.stop_loss,
                'take_profit': sig.take_profit,
                'action': 'ENTRY',
                'reason': f'allocated_{intent.allocation_reason}',
                'threshold': getattr(sig, '_threshold_at_entry', self.last_dynamic_threshold),
                'threshold_margin': getattr(sig, '_threshold_margin', 0.0),
                'would_have_passed': getattr(sig, '_would_have_passed', True),
                'risk_temp': getattr(sig, '_risk_temp_at_entry', self.last_risk_temp),
                'instability': getattr(sig, '_instability_at_entry', self.last_instability),
                'crisis_prob': getattr(sig, '_crisis_prob_at_entry', self.last_crisis_prob),
            }
            # Attach narrative (from _build_signal_narrative) for CSV + dashboard
            if narrative:
                sig_entry['narrative'] = narrative
            acted_signals.append(sig_entry)

        # Log signals to CSV
        for sig_dict in acted_signals:
            self._log_signal(sig_dict)

        self._update_equity(features['close'])
        return acted_signals

    # ------------------------------------------------------------------
    # Factor attribution (approximate from entry conditions)
    # ------------------------------------------------------------------

    def _compute_factor_attribution(self, archetype: str, features, regime_label: str) -> Dict:
        """
        Compute approximate factor attribution for a trade at entry time.

        Uses the archetype's fusion weights to decompose the technical
        contribution, plus macro/regime environment at signal time.

        Returns dict with keys: technical, liquidity, macro, regime
        (each a float 0-1, summing to ~1.0).
        """
        # Default fusion weights by archetype (from YAML configs / dashboard ARCHETYPES)
        default_weights = {
            'trap_within_trend': {'wyckoff': 0.35, 'liquidity': 0.35, 'momentum': 0.15, 'smc': 0.15},
            'liquidity_sweep':   {'wyckoff': 0.30, 'liquidity': 0.40, 'momentum': 0.15, 'smc': 0.15},
            'wick_trap':         {'wyckoff': 0.20, 'liquidity': 0.40, 'momentum': 0.15, 'smc': 0.25},
            'liquidity_compression': {'wyckoff': 0.20, 'liquidity': 0.50, 'momentum': 0.15, 'smc': 0.15},
        }

        # Get archetype fusion weights (from engine config or defaults)
        arch_weights = default_weights.get(archetype, {'wyckoff': 0.25, 'liquidity': 0.35, 'momentum': 0.20, 'smc': 0.20})
        if archetype in self.engine.archetype_configs:
            cfg = self.engine.archetype_configs[archetype]
            if 'fusion_weights' in cfg:
                arch_weights = cfg['fusion_weights']

        # Technical = wyckoff + smc + momentum weighted contributions
        tech_weight = arch_weights.get('wyckoff', 0.0) + arch_weights.get('smc', 0.0) + arch_weights.get('momentum', 0.0)
        liq_weight = arch_weights.get('liquidity', 0.0)

        # Macro contribution: based on how extreme macro signals are at entry
        def _safe(val, default=0.0):
            if val is None:
                return default
            try:
                v = float(val)
                return default if v != v else v
            except (TypeError, ValueError):
                return default

        fg = _safe(features.get('fear_greed_norm', features.get('FEAR_GREED', None)))
        dxy_z = _safe(features.get('DXY_Z', features.get('dxy_z', None)))
        vix_z = _safe(features.get('VIX_Z', features.get('vix_z', None)))

        # Macro extremity: more extreme macro = higher macro contribution
        macro_extremity = 0.0
        macro_count = 0
        if fg is not None and fg != 0.0:
            # F&G: extreme fear (<0.2) or greed (>0.8) contributes more
            macro_extremity += min(abs(fg - 0.5) * 2.0, 1.0)
            macro_count += 1
        if dxy_z is not None and dxy_z != 0.0:
            macro_extremity += min(abs(dxy_z) / 2.0, 1.0)
            macro_count += 1
        if vix_z is not None and vix_z != 0.0:
            macro_extremity += min(abs(vix_z) / 2.0, 1.0)
            macro_count += 1

        if macro_count > 0:
            macro_extremity /= macro_count
        # Scale macro_extremity to a 0-0.30 contribution range
        macro_pct = macro_extremity * 0.30

        # Regime contribution: how aligned the regime is
        risk_temp = self.last_risk_temp
        crisis_prob = self.last_crisis_prob
        # Bull regime = high risk_temp = good alignment = positive regime contribution
        # Bear regime = low risk_temp + high crisis = negative alignment
        if regime_label in ('bull',):
            regime_alignment = min(risk_temp, 1.0)
        elif regime_label in ('bear', 'crisis'):
            regime_alignment = max(1.0 - risk_temp, 0.0) * 0.5
        else:
            regime_alignment = 0.5
        regime_pct = regime_alignment * 0.20

        # Normalize: technical + liquidity get the remaining share
        remaining = max(1.0 - macro_pct - regime_pct, 0.4)
        total_fusion_weight = tech_weight + liq_weight
        if total_fusion_weight > 0:
            tech_pct = remaining * (tech_weight / total_fusion_weight)
            liq_pct = remaining * (liq_weight / total_fusion_weight)
        else:
            tech_pct = remaining * 0.6
            liq_pct = remaining * 0.4

        # Final normalization to ensure sum = 1.0
        total = tech_pct + liq_pct + macro_pct + regime_pct
        if total > 0:
            tech_pct /= total
            liq_pct /= total
            macro_pct /= total
            regime_pct /= total

        return {
            'technical': round(tech_pct, 4),
            'liquidity': round(liq_pct, 4),
            'macro': round(macro_pct, 4),
            'regime': round(regime_pct, 4),
            # Raw entry conditions for deeper analysis
            'entry_conditions': {
                'risk_temperature': round(self.last_risk_temp, 4),
                'instability': round(self.last_instability, 4),
                'crisis_prob': round(self.last_crisis_prob, 4),
                'dynamic_threshold': round(self.last_dynamic_threshold, 4),
                'fear_greed_norm': round(_safe(fg), 4),
                'dxy_z': round(_safe(dxy_z), 4),
                'vix_z': round(_safe(vix_z), 4),
                'fusion_weights': arch_weights,
            },
        }

    # ------------------------------------------------------------------
    # Position management (mirrors backtester)
    # ------------------------------------------------------------------

    def _open_position(
        self,
        timestamp, archetype, direction, entry_price,
        stop_loss, take_profit, fusion_score, regime_label,
        features, allocated_size_pct, entry_narrative=None,
        threshold_at_entry=0.0, risk_temp_at_entry=0.0,
        instability_at_entry=0.0, crisis_prob_at_entry=0.0,
        threshold_margin=0.0, would_have_passed=True,
    ):
        """Open a virtual position."""
        portfolio_value = self.cash
        atr = features.get('atr_14', entry_price * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        # Guard NaN stop/take
        if pd.isna(stop_loss) or stop_loss <= 0:
            stop_loss = entry_price - (atr * 2.0) if direction == 'long' else entry_price + (atr * 2.0)
        if pd.isna(take_profit) or take_profit <= 0:
            take_profit = entry_price + (atr * 4.0) if direction == 'long' else entry_price - (atr * 4.0)

        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        if pd.isna(stop_distance_pct) or stop_distance_pct <= 0:
            stop_distance_pct = 0.025

        risk_dollars = portfolio_value * self.risk_per_trade
        position_size_usd = risk_dollars / stop_distance_pct
        position_size_usd *= (allocated_size_pct / 0.02)
        position_size_usd *= self.leverage
        max_size = portfolio_value * self.max_position_pct * self.leverage
        position_size_usd = min(position_size_usd, max_size)

        commission = position_size_usd * self.commission_rate
        slippage = position_size_usd * (self.slippage_bps / 10000.0)
        total_cost = position_size_usd + commission + slippage

        if total_cost > self.cash:
            self.signals_rejected += 1
            return

        fill_price = entry_price * (1 + self.slippage_bps / 10000.0) if direction == 'long' \
            else entry_price * (1 - self.slippage_bps / 10000.0)

        quantity = position_size_usd / fill_price
        self.cash -= (position_size_usd + commission)

        # Compute factor attribution at entry time
        factor_attribution = self._compute_factor_attribution(archetype, features, regime_label)

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
            entry_narrative=entry_narrative,
            entry_factor_attribution=factor_attribution,
            threshold_at_entry=threshold_at_entry,
            risk_temp_at_entry=risk_temp_at_entry,
            instability_at_entry=instability_at_entry,
            crisis_prob_at_entry=crisis_prob_at_entry,
            threshold_margin=threshold_margin,
            would_have_passed=would_have_passed,
        )
        self.signals_allocated += 1

        logger.info(
            f"[SHADOW ENTRY] {direction.upper()} {archetype} @ ${fill_price:,.2f} | "
            f"Size: ${position_size_usd:,.0f} | Regime: {regime_label} | "
            f"Fusion: {fusion_score:.3f}"
        )

    def _close_position(self, pos_id, exit_price, exit_timestamp, exit_reason="unknown", exit_pct=1.0):
        """Close a virtual position (full or partial)."""
        if pos_id not in self.positions:
            return
        pos = self.positions[pos_id]

        exit_quantity = pos.original_quantity * exit_pct
        exit_quantity = min(exit_quantity, pos.current_quantity)
        if exit_quantity <= 1e-10:
            return

        fill_exit = exit_price * (1 - self.slippage_bps / 10000.0) if pos.direction == 'long' \
            else exit_price * (1 + self.slippage_bps / 10000.0)

        pnl = (fill_exit - pos.entry_price) * exit_quantity if pos.direction == 'long' \
            else (pos.entry_price - fill_exit) * exit_quantity

        exit_value = fill_exit * exit_quantity
        commission = exit_value * self.commission_rate
        pnl -= commission
        self.cash += exit_value - commission

        entry_value = pos.entry_price * exit_quantity
        pnl_pct = (pnl / entry_value * 100) if entry_value > 0 else 0.0
        duration_hours = (exit_timestamp - pos.entry_time).total_seconds() / 3600.0

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
            factor_attribution=pos.entry_factor_attribution,
            threshold_at_entry=pos.threshold_at_entry,
            risk_temp_at_entry=pos.risk_temp_at_entry,
            instability_at_entry=pos.instability_at_entry,
            crisis_prob_at_entry=pos.crisis_prob_at_entry,
            threshold_margin=pos.threshold_margin,
            would_have_passed=pos.would_have_passed,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            atr_at_entry=pos.atr_at_entry,
        ))

        pos.current_quantity -= exit_quantity
        pos.total_exits_pct += exit_pct

        # Log exit signal
        self._log_signal({
            'timestamp': str(exit_timestamp),
            'archetype': pos.archetype,
            'direction': pos.direction,
            'fusion_score': pos.fusion_score,
            'regime': pos.regime_at_entry,
            'entry_price': pos.entry_price,
            'stop_loss': pos.stop_loss,
            'take_profit': pos.take_profit,
            'action': 'EXIT',
            'reason': exit_reason,
            'threshold': pos.threshold_at_entry,
            'threshold_margin': pos.threshold_margin,
            'would_have_passed': pos.would_have_passed,
            'risk_temp': pos.risk_temp_at_entry,
            'instability': pos.instability_at_entry,
            'crisis_prob': pos.crisis_prob_at_entry,
        })

        logger.info(
            f"[SHADOW EXIT] {exit_reason} | {pos.archetype} @ ${fill_exit:,.2f} | "
            f"PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%) | {duration_hours:.1f}h"
        )

        if pos.current_quantity < 1e-10 or pos.total_exits_pct >= 0.99:
            if pos.current_quantity > 1e-10:
                self.cash += pos.current_quantity * fill_exit
            del self.positions[pos_id]

    def _check_all_exits(self, row: pd.Series, ts: pd.Timestamp):
        """Check exits for all virtual positions (mirrors backtester exactly)."""
        for pos_id in list(self.positions.keys()):
            pos = self.positions[pos_id]
            close_price = row['close']
            atr = row.get('atr_14', pos.atr_at_entry)
            if pd.isna(atr) or atr <= 0:
                atr = pos.atr_at_entry

            # 1. Stop loss
            effective_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
            stop_hit = False
            if pos.direction == 'long' and row['low'] <= effective_stop:
                stop_hit = True
                exit_price = effective_stop
            elif pos.direction == 'short' and row['high'] >= effective_stop:
                stop_hit = True
                exit_price = effective_stop

            if stop_hit:
                self._close_position(pos_id, exit_price, ts, exit_reason="stop_loss", exit_pct=1.0)
                continue

            # 2. Scale-out profit targets
            stop_distance = abs(pos.entry_price - pos.stop_loss)
            if stop_distance <= 0:
                stop_distance = atr * 2.5

            unrealized_pnl = (close_price - pos.entry_price) if pos.direction == 'long' \
                else (pos.entry_price - close_price)
            unrealized_r = unrealized_pnl / stop_distance if stop_distance > 0 else 0.0

            archetype_name = pos.archetype
            scale_levels = [0.5, 1.0, 2.0]
            scale_pcts = [0.20, 0.20, 0.30]

            if archetype_name in self.engine.archetypes:
                arch_cfg = self.engine.archetype_configs.get(archetype_name, {})
                exit_cfg = arch_cfg.get('exit_logic', {})
                if 'scale_out_levels' in exit_cfg:
                    scale_levels = exit_cfg['scale_out_levels']
                if 'scale_out_pcts' in exit_cfg:
                    scale_pcts = exit_cfg['scale_out_pcts']

            for level, pct in zip(scale_levels, scale_pcts):
                if unrealized_r >= level and level not in pos.executed_scale_outs:
                    pos.executed_scale_outs.append(level)
                    self._close_position(pos_id, close_price, ts,
                                         exit_reason=f"scale_out_{level:.1f}R", exit_pct=pct)
                    break

            if pos_id not in self.positions:
                continue

            # 3. Time-based exit
            hours_held = (ts - pos.entry_time).total_seconds() / 3600.0
            max_hold = 168
            if archetype_name in self.engine.archetypes:
                arch_cfg = self.engine.archetype_configs.get(archetype_name, {})
                exit_cfg = arch_cfg.get('exit_logic', {})
                max_hold = exit_cfg.get('max_hold_hours', 168)

            if hours_held >= max_hold:
                self._close_position(pos_id, close_price, ts,
                                     exit_reason=f"time_exit_{hours_held:.0f}h", exit_pct=1.0)
                continue

            # 4. Trailing stop update
            trailing_start_r = 1.0
            trailing_atr_mult = 2.0
            if archetype_name in self.engine.archetypes:
                arch_cfg = self.engine.archetype_configs.get(archetype_name, {})
                exit_cfg = arch_cfg.get('exit_logic', {})
                trailing_start_r = exit_cfg.get('trailing_start_r', 1.0)
                trailing_atr_mult = exit_cfg.get('trailing_atr_mult', 2.0)

            if unrealized_r >= trailing_start_r:
                if unrealized_r >= 3.0:
                    effective_mult = trailing_atr_mult * 0.5
                elif unrealized_r >= 2.0:
                    effective_mult = trailing_atr_mult * 0.67
                elif unrealized_r >= 1.0:
                    effective_mult = trailing_atr_mult * 0.83
                else:
                    effective_mult = trailing_atr_mult

                if pos.direction == 'long':
                    new_trail = close_price - (effective_mult * atr)
                    current_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                    if new_trail > current_stop:
                        pos.trailing_stop = new_trail
                else:
                    new_trail = close_price + (effective_mult * atr)
                    current_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                    if new_trail < current_stop:
                        pos.trailing_stop = new_trail

    # ------------------------------------------------------------------
    # Phantom trade tracker — counterfactual tracking for rejected signals
    # ------------------------------------------------------------------

    def _open_phantom(self, signal, features, regime_label, rejection_reason, rejection_stage):
        """Create a phantom position for a rejected signal to track what would have happened."""
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        take_profit = getattr(signal, 'take_profit', None) or 0.0
        atr = features.get('atr_14', 0.0)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.015

        if pd.isna(stop_loss) or stop_loss <= 0:
            stop_loss = entry_price - (atr * 2.0) if signal.direction == 'long' else entry_price + (atr * 2.0)
        if pd.isna(take_profit) or take_profit <= 0:
            take_profit = entry_price + (atr * 4.0) if signal.direction == 'long' else entry_price - (atr * 4.0)

        # Use a fixed notional size for phantom tracking (not real capital)
        phantom_quantity = 1000.0 / entry_price  # $1000 notional

        ts = signal.timestamp if hasattr(signal, 'timestamp') else pd.Timestamp.now(tz='UTC')
        pos_id = f"phantom_{signal.direction}_{signal.archetype_id}_{int(ts.timestamp())}"

        self.phantom_positions[pos_id] = TrackedPosition(
            position_id=pos_id,
            archetype=signal.archetype_id,
            direction=signal.direction,
            entry_price=entry_price,
            entry_time=ts,
            stop_loss=stop_loss,
            take_profit=take_profit,
            original_quantity=phantom_quantity,
            current_quantity=phantom_quantity,
            fusion_score=signal.fusion_score,
            regime_at_entry=regime_label,
            atr_at_entry=atr,
            entry_narrative={
                'rejection_reason': rejection_reason,
                'rejection_stage': rejection_stage,
            },
        )
        self.phantom_signals_total += 1

    def _close_phantom(self, pos_id, exit_price, exit_timestamp, exit_reason="unknown"):
        """Close a phantom position and record the counterfactual outcome."""
        if pos_id not in self.phantom_positions:
            return
        pos = self.phantom_positions[pos_id]

        fill_exit = exit_price * (1 - self.slippage_bps / 10000.0) if pos.direction == 'long' \
            else exit_price * (1 + self.slippage_bps / 10000.0)

        pnl = (fill_exit - pos.entry_price) * pos.original_quantity if pos.direction == 'long' \
            else (pos.entry_price - fill_exit) * pos.original_quantity

        exit_value = fill_exit * pos.original_quantity
        commission = exit_value * self.commission_rate
        entry_commission = pos.entry_price * pos.original_quantity * self.commission_rate
        pnl -= (commission + entry_commission)

        entry_value = pos.entry_price * pos.original_quantity
        pnl_pct = (pnl / entry_value * 100) if entry_value > 0 else 0.0
        duration_hours = (exit_timestamp - pos.entry_time).total_seconds() / 3600.0

        self.phantom_trades.append(CompletedTrade(
            timestamp_entry=pos.entry_time,
            timestamp_exit=exit_timestamp,
            archetype=pos.archetype,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=fill_exit,
            quantity=pos.original_quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_regime=pos.regime_at_entry,
            duration_hours=duration_hours,
            fusion_score=pos.fusion_score,
            exit_reason=exit_reason,
            factor_attribution=pos.entry_narrative,  # Stores rejection reason
        ))
        del self.phantom_positions[pos_id]

    def _check_phantom_exits(self, row: pd.Series, ts: pd.Timestamp):
        """Check exits for all phantom positions (same logic as real positions)."""
        for pos_id in list(self.phantom_positions.keys()):
            pos = self.phantom_positions[pos_id]
            close_price = row['close']
            atr = row.get('atr_14', pos.atr_at_entry)
            if pd.isna(atr) or atr <= 0:
                atr = pos.atr_at_entry

            # 1. Stop loss
            stop_hit = False
            if pos.direction == 'long' and row['low'] <= pos.stop_loss:
                stop_hit = True
                exit_price = pos.stop_loss
            elif pos.direction == 'short' and row['high'] >= pos.stop_loss:
                stop_hit = True
                exit_price = pos.stop_loss

            if stop_hit:
                self._close_phantom(pos_id, exit_price, ts, exit_reason="stop_loss")
                continue

            # 2. Take profit (simplified — full exit at TP for phantoms)
            tp_hit = False
            if pos.direction == 'long' and row['high'] >= pos.take_profit:
                tp_hit = True
                exit_price = pos.take_profit
            elif pos.direction == 'short' and row['low'] <= pos.take_profit:
                tp_hit = True
                exit_price = pos.take_profit

            if tp_hit:
                self._close_phantom(pos_id, exit_price, ts, exit_reason="take_profit")
                continue

            # 3. Time-based exit (same max hold as real)
            hours_held = (ts - pos.entry_time).total_seconds() / 3600.0
            max_hold = 168
            archetype_name = pos.archetype
            if archetype_name in self.engine.archetypes:
                arch_cfg = self.engine.archetype_configs.get(archetype_name, {})
                exit_cfg = arch_cfg.get('exit_logic', {})
                max_hold = exit_cfg.get('max_hold_hours', 168)

            if hours_held >= max_hold:
                self._close_phantom(pos_id, close_price, ts, exit_reason=f"time_exit_{hours_held:.0f}h")

    def get_phantom_summary(self) -> dict:
        """Return phantom trade statistics for heartbeat/dashboard."""
        if not self.phantom_trades:
            return {
                'total_phantom_signals': self.phantom_signals_total,
                'completed_phantom_trades': 0,
                'active_phantom_positions': len(self.phantom_positions),
                'phantom_wins': 0, 'phantom_losses': 0,
                'phantom_win_rate': 0.0, 'phantom_pnl': 0.0,
                'phantom_avg_pnl': 0.0,
                'insight': 'No phantom trades completed yet.',
                'trades': [], 'active': [],
            }

        wins = [t for t in self.phantom_trades if t.pnl > 0]
        losses = [t for t in self.phantom_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.phantom_trades)
        win_rate = len(wins) / len(self.phantom_trades) * 100 if self.phantom_trades else 0

        # Compare phantom vs real performance
        real_wins = [t for t in self.trades if t.pnl > 0]
        real_losses = [t for t in self.trades if t.pnl <= 0]
        real_pnl = sum(t.pnl for t in self.trades)

        if total_pnl > 0 and len(wins) > len(losses):
            insight = (f"Filters are too tight — rejected signals would have netted "
                       f"${total_pnl:.0f} ({len(wins)}W/{len(losses)}L). "
                       f"Consider lowering thresholds.")
        elif total_pnl < real_pnl:
            insight = (f"Filters are working — rejected signals would have lost "
                       f"${total_pnl:.0f} vs real ${real_pnl:.0f}. "
                       f"Current thresholds are protecting capital.")
        else:
            insight = (f"Mixed results — phantom PnL ${total_pnl:.0f} vs real ${real_pnl:.0f}. "
                       f"Need more data for clear signal.")

        # Per-archetype phantom breakdown
        by_arch = {}
        for t in self.phantom_trades:
            if t.archetype not in by_arch:
                by_arch[t.archetype] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
            if t.pnl > 0:
                by_arch[t.archetype]['wins'] += 1
            else:
                by_arch[t.archetype]['losses'] += 1
            by_arch[t.archetype]['pnl'] += t.pnl

        # Serialize recent phantom trades for dashboard
        recent = []
        for t in self.phantom_trades[-20:]:
            recent.append({
                'archetype': t.archetype,
                'direction': t.direction,
                'entry_price': round(t.entry_price, 2),
                'exit_price': round(t.exit_price, 2),
                'pnl': round(t.pnl, 2),
                'pnl_pct': round(t.pnl_pct, 4),
                'fusion_score': round(t.fusion_score, 4),
                'exit_reason': t.exit_reason,
                'duration_hours': round(t.duration_hours, 1),
                'rejection_reason': t.factor_attribution.get('rejection_reason', '') if t.factor_attribution else '',
                'rejection_stage': t.factor_attribution.get('rejection_stage', '') if t.factor_attribution else '',
            })

        # Active phantom positions
        active = []
        for pos in self.phantom_positions.values():
            active.append({
                'archetype': pos.archetype,
                'direction': pos.direction,
                'entry_price': round(pos.entry_price, 2),
                'fusion_score': round(pos.fusion_score, 4),
                'stop_loss': round(pos.stop_loss, 2),
                'take_profit': round(pos.take_profit, 2),
                'rejection_reason': pos.entry_narrative.get('rejection_reason', '') if pos.entry_narrative else '',
            })

        # Fusion score bucket analysis (combines real + phantom for full picture)
        all_trades = list(self.trades) + list(self.phantom_trades)
        fusion_buckets = {}
        for t in all_trades:
            bucket = round(t.fusion_score * 10) / 10  # Round to nearest 0.1
            bucket_key = f"{bucket:.1f}"
            if bucket_key not in fusion_buckets:
                fusion_buckets[bucket_key] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'count': 0}
            fusion_buckets[bucket_key]['count'] += 1
            if t.pnl > 0:
                fusion_buckets[bucket_key]['wins'] += 1
            else:
                fusion_buckets[bucket_key]['losses'] += 1
            fusion_buckets[bucket_key]['total_pnl'] += t.pnl

        return {
            'total_phantom_signals': self.phantom_signals_total,
            'completed_phantom_trades': len(self.phantom_trades),
            'active_phantom_positions': len(self.phantom_positions),
            'phantom_wins': len(wins),
            'phantom_losses': len(losses),
            'phantom_win_rate': round(win_rate, 1),
            'phantom_pnl': round(total_pnl, 2),
            'phantom_avg_pnl': round(total_pnl / len(self.phantom_trades), 2) if self.phantom_trades else 0.0,
            'real_pnl': round(real_pnl, 2),
            'insight': insight,
            'by_archetype': by_arch,
            'trades': recent,
            'active': active,
            'fusion_buckets': fusion_buckets,
        }

    # ------------------------------------------------------------------
    # Equity & logging
    # ------------------------------------------------------------------

    def _update_equity(self, current_price: float):
        equity = self.cash
        for pos in self.positions.values():
            if pos.direction == 'long':
                equity += pos.current_quantity * current_price
            else:
                equity += pos.current_quantity * (2 * pos.entry_price - current_price)
        self.equity_curve.append(equity)

    def _log_signal(self, sig_dict: dict):
        """Append signal to CSV log."""
        # Extract narrative text for CSV (escape commas/newlines)
        narrative_text = ""
        narr = sig_dict.get("narrative")
        if narr and isinstance(narr, dict):
            # Enhanced narrative has 'text', runner narrative has 'summary'
            narrative_text = narr.get("text", narr.get("summary", ""))
        elif narr and isinstance(narr, str):
            narrative_text = narr
        # CSV-safe: quote field if it contains commas or quotes
        narrative_text = narrative_text.replace('"', '""')
        narrative_csv = f'"{narrative_text}"' if narrative_text else ""
        with open(self.signal_log_path, 'a') as f:
            threshold = sig_dict.get('threshold', 0.0)
            t_margin = sig_dict.get('threshold_margin', 0.0)
            whp = sig_dict.get('would_have_passed', True)
            r_temp = sig_dict.get('risk_temp', 0.0)
            instab = sig_dict.get('instability', 0.0)
            crisis = sig_dict.get('crisis_prob', 0.0)
            f.write(
                f"{sig_dict['timestamp']},{sig_dict['archetype']},{sig_dict['direction']},"
                f"{sig_dict['fusion_score']:.4f},{sig_dict['regime']},"
                f"{sig_dict['entry_price']:.2f},{sig_dict['stop_loss']:.2f},"
                f"{sig_dict.get('take_profit', 0):.2f},{sig_dict['action']},{sig_dict['reason']},"
                f"{threshold:.4f},{t_margin:.4f},{whp},"
                f"{r_temp:.4f},{instab:.4f},{crisis:.4f},"
                f"{narrative_csv}\n"
            )

    def save_state(self):
        """Persist positions and stats to disk for restart recovery."""
        # Serialize completed trades
        trades_data = []
        for t in self.trades:
            td = {
                'timestamp_entry': str(t.timestamp_entry),
                'timestamp_exit': str(t.timestamp_exit),
                'archetype': t.archetype,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'entry_regime': t.entry_regime,
                'duration_hours': t.duration_hours,
                'fusion_score': t.fusion_score,
                'exit_reason': t.exit_reason,
                'factor_attribution': t.factor_attribution,
                'threshold_at_entry': t.threshold_at_entry,
                'risk_temp_at_entry': t.risk_temp_at_entry,
                'instability_at_entry': t.instability_at_entry,
                'crisis_prob_at_entry': t.crisis_prob_at_entry,
                'threshold_margin': t.threshold_margin,
                'would_have_passed': t.would_have_passed,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'atr_at_entry': t.atr_at_entry,
            }
            trades_data.append(td)

        # Serialize phantom trades
        phantom_trades_data = []
        for t in self.phantom_trades:
            phantom_trades_data.append({
                'timestamp_entry': str(t.timestamp_entry),
                'timestamp_exit': str(t.timestamp_exit),
                'archetype': t.archetype, 'direction': t.direction,
                'entry_price': t.entry_price, 'exit_price': t.exit_price,
                'quantity': t.quantity, 'pnl': t.pnl, 'pnl_pct': t.pnl_pct,
                'entry_regime': t.entry_regime, 'duration_hours': t.duration_hours,
                'fusion_score': t.fusion_score, 'exit_reason': t.exit_reason,
                'factor_attribution': t.factor_attribution,
            })

        state = {
            'cash': self.cash,
            'bar_index': self.bar_index,
            'total_signals': self.total_signals,
            'signals_allocated': self.signals_allocated,
            'signals_rejected': self.signals_rejected,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'last_cmi_breakdown': self.last_cmi_breakdown,
            'last_cmi_comparison': self.last_cmi_comparison,
            'last_dynamic_threshold': self.last_dynamic_threshold,
            'last_risk_temp': self.last_risk_temp,
            'last_instability': self.last_instability,
            'last_crisis_prob': self.last_crisis_prob,
            'equity_curve': self.equity_curve,
            'trades': trades_data,
            'phantom_positions': {k: v.to_dict() for k, v in self.phantom_positions.items()},
            'phantom_trades': phantom_trades_data,
            'phantom_signals_total': self.phantom_signals_total,
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        state_path = self.output_dir / 'state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved: {len(self.positions)} positions, ${self.cash:,.0f} cash")

    def load_state(self) -> bool:
        """Load saved state for restart recovery. Returns True if loaded."""
        state_path = self.output_dir / 'state.json'
        if not state_path.exists():
            return False

        with open(state_path) as f:
            state = json.load(f)

        self.cash = state['cash']
        self.bar_index = state['bar_index']
        self.total_signals = state['total_signals']
        self.signals_allocated = state['signals_allocated']
        self.signals_rejected = state['signals_rejected']
        self.positions = {
            k: TrackedPosition.from_dict(v) for k, v in state['positions'].items()
        }
        self.last_cmi_breakdown = state.get('last_cmi_breakdown', {})
        self.last_cmi_comparison = state.get('last_cmi_comparison', {})
        self.last_dynamic_threshold = state.get('last_dynamic_threshold', 0.18)
        self.last_risk_temp = state.get('last_risk_temp', 0.0)
        self.last_instability = state.get('last_instability', 0.0)
        self.last_crisis_prob = state.get('last_crisis_prob', 0.0)
        self.equity_curve = state.get('equity_curve', [])
        # Restore completed trades
        for td in state.get('trades', []):
            try:
                t = CompletedTrade(
                    timestamp_entry=pd.Timestamp(td['timestamp_entry']),
                    timestamp_exit=pd.Timestamp(td['timestamp_exit']),
                    archetype=td['archetype'],
                    direction=td['direction'],
                    entry_price=td['entry_price'],
                    exit_price=td['exit_price'],
                    quantity=td['quantity'],
                    pnl=td['pnl'],
                    pnl_pct=td['pnl_pct'],
                    entry_regime=td['entry_regime'],
                    duration_hours=td['duration_hours'],
                    fusion_score=td['fusion_score'],
                    exit_reason=td['exit_reason'],
                    factor_attribution=td.get('factor_attribution', None),
                    threshold_at_entry=td.get('threshold_at_entry', 0.0),
                    risk_temp_at_entry=td.get('risk_temp_at_entry', 0.0),
                    instability_at_entry=td.get('instability_at_entry', 0.0),
                    crisis_prob_at_entry=td.get('crisis_prob_at_entry', 0.0),
                    threshold_margin=td.get('threshold_margin', 0.0),
                    would_have_passed=td.get('would_have_passed', True),
                    stop_loss=td.get('stop_loss', 0.0),
                    take_profit=td.get('take_profit', 0.0),
                    atr_at_entry=td.get('atr_at_entry', 0.0),
                )
                self.trades.append(t)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping trade restore: {e}")
        # Restore phantom positions
        for k, v in state.get('phantom_positions', {}).items():
            try:
                self.phantom_positions[k] = TrackedPosition.from_dict(v)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping phantom position restore: {e}")

        # Restore phantom trades
        for td in state.get('phantom_trades', []):
            try:
                t = CompletedTrade(
                    timestamp_entry=pd.Timestamp(td['timestamp_entry']),
                    timestamp_exit=pd.Timestamp(td['timestamp_exit']),
                    archetype=td['archetype'], direction=td['direction'],
                    entry_price=td['entry_price'], exit_price=td['exit_price'],
                    quantity=td['quantity'], pnl=td['pnl'], pnl_pct=td['pnl_pct'],
                    entry_regime=td['entry_regime'], duration_hours=td['duration_hours'],
                    fusion_score=td['fusion_score'], exit_reason=td['exit_reason'],
                    factor_attribution=td.get('factor_attribution', None),
                    threshold_at_entry=td.get('threshold_at_entry', 0.0),
                    risk_temp_at_entry=td.get('risk_temp_at_entry', 0.0),
                    instability_at_entry=td.get('instability_at_entry', 0.0),
                    crisis_prob_at_entry=td.get('crisis_prob_at_entry', 0.0),
                    threshold_margin=td.get('threshold_margin', 0.0),
                    would_have_passed=td.get('would_have_passed', True),
                    stop_loss=td.get('stop_loss', 0.0),
                    take_profit=td.get('take_profit', 0.0),
                    atr_at_entry=td.get('atr_at_entry', 0.0),
                )
                self.phantom_trades.append(t)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping phantom trade restore: {e}")
        self.phantom_signals_total = state.get('phantom_signals_total', 0)

        logger.info(
            f"State loaded from {state['saved_at']}: "
            f"{len(self.positions)} positions, ${self.cash:,.0f} cash, "
            f"{len(self.equity_curve)} equity points, {len(self.trades)} trades, "
            f"{len(self.phantom_positions)} phantom positions, {len(self.phantom_trades)} phantom trades"
        )
        return True

    def print_summary(self):
        """Print performance summary."""
        print("\n" + "=" * 72)
        print("SHADOW RUNNER SUMMARY")
        print("=" * 72)
        print(f"Total Signals:      {self.total_signals}")
        print(f"Signals Allocated:  {self.signals_allocated}")
        print(f"Signals Rejected:   {self.signals_rejected}")
        print(f"Completed Trades:   {len(self.trades)}")
        print(f"Open Positions:     {len(self.positions)}")

        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]
            total_pnl = sum(t.pnl for t in self.trades)
            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            print(f"\nWin Rate:           {len(wins)}/{len(self.trades)} ({100*len(wins)/len(self.trades):.1f}%)")
            print(f"Profit Factor:      {pf:.3f}")
            print(f"Total PnL:          ${total_pnl:,.2f}")
            print(f"Avg Trade PnL:      ${total_pnl/len(self.trades):,.2f}")

        if self.equity_curve:
            peak = max(self.equity_curve)
            drawdowns = [(peak - e) / peak for e in self.equity_curve]
            max_dd = max(drawdowns) if drawdowns else 0
            print(f"\nFinal Equity:       ${self.equity_curve[-1]:,.2f}")
            print(f"Max Drawdown:       {max_dd*100:.2f}%")
        print("=" * 72)

    # ------------------------------------------------------------------
    # Replay mode (historical feature store)
    # ------------------------------------------------------------------

    def run_replay(
        self,
        feature_store_path: str = 'data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Run in replay mode using the pre-computed feature store.
        This should produce identical signals to the backtester.
        """
        store_path = PROJECT_ROOT / feature_store_path
        logger.info(f"Loading feature store: {store_path}")
        df = pd.read_parquet(store_path)
        logger.info(f"  Shape: {df.shape[0]:,} bars x {df.shape[1]:,} cols")

        # Derive regime labels (same as backtester)
        self._derive_regime_labels(df)

        # Filter date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz='UTC')]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date, tz='UTC')]

        logger.info(f"Replay period: {df.index[0]} to {df.index[-1]} ({len(df):,} bars)")

        t0 = time.time()
        for bar_idx, (ts, row) in enumerate(df.iterrows()):
            self.process_bar(row, ts)

            if bar_idx % 2000 == 0 and bar_idx > 0:
                equity = self.equity_curve[-1] if self.equity_curve else self.initial_cash
                elapsed = time.time() - t0
                logger.info(
                    f"Bar {bar_idx:,}/{len(df):,} | {ts} | "
                    f"Equity: ${equity:,.0f} | Positions: {len(self.positions)} | "
                    f"Trades: {len(self.trades)} | {bar_idx/elapsed:.0f} bars/sec"
                )

        # Close remaining positions
        if self.positions and len(df) > 0:
            last_row = df.iloc[-1]
            last_ts = df.index[-1]
            for pos_id in list(self.positions.keys()):
                self._close_position(pos_id, last_row['close'], last_ts,
                                     exit_reason="replay_end", exit_pct=1.0)

        elapsed = time.time() - t0
        logger.info(f"Replay completed in {elapsed:.1f}s ({len(df)/elapsed:,.0f} bars/sec)")
        self.print_summary()
        self.save_state()

    def _derive_regime_labels(self, df: pd.DataFrame):
        """Derive regime labels from EMA trend alignment (CMI-compatible).

        Uses the same logic as _regime_label_from_risk_temp in the live
        feature computer: labels are derived from trend alignment and
        volatility, producing bull/bear/neutral/crisis.

        This replaces the old SMA-crossover labels (risk_on/risk_off)
        that had zero practical effect on trading.
        """
        if 'close' not in df.columns:
            return

        # Use EMA-based trend alignment (same inputs as CMI risk_temperature)
        # price_above_ema_50 and ema_50_above_200 should already be in the
        # feature store; fall back to computing from close if needed.
        if 'price_above_ema_50' in df.columns and 'ema_50_above_200' in df.columns:
            p_above_50 = df['price_above_ema_50'].fillna(0).astype(float)
            ema_50_200 = df['ema_50_above_200'].fillna(0).astype(float)
        else:
            close = df['close']
            ema50 = close.ewm(span=50, min_periods=20).mean()
            ema200 = close.ewm(span=200, min_periods=50).mean()
            p_above_50 = (close > ema50).astype(float)
            ema_50_200 = (ema50 > ema200).astype(float)

        # Determine if high volatility (for crisis detection)
        if 'atr_percentile' in df.columns and df['atr_percentile'].notna().mean() > 0.5:
            high_vol = df['atr_percentile'] > 0.75
        else:
            ret_vol = df['close'].pct_change().rolling(168).std()
            high_vol = ret_vol > ret_vol.quantile(0.75)

        # Derive labels matching CMI risk_temperature logic
        regimes = pd.Series('neutral', index=df.index)
        regimes[(p_above_50 > 0) & (ema_50_200 > 0)] = 'bull'
        regimes[(p_above_50 > 0) & (ema_50_200 <= 0)] = 'neutral'
        regimes[(p_above_50 <= 0) & (ema_50_200 > 0)] = 'neutral'  # distribution
        regimes[(p_above_50 <= 0) & (ema_50_200 <= 0)] = 'bear'
        regimes[(p_above_50 <= 0) & (ema_50_200 <= 0) & high_vol] = 'crisis'

        df['regime_label'] = regimes
        dist = df['regime_label'].value_counts()
        logger.info(f"Derived regime distribution: {dist.to_dict()}")

    # ------------------------------------------------------------------
    # Live mode (real Binance data)
    # ------------------------------------------------------------------

    def run_live(self):
        """
        Run in live mode: fetch real Binance candles hourly, generate signals.
        """
        try:
            from bin.live.binance_adapter import BinanceAdapter
            from bin.live.live_feature_computer import LiveFeatureComputer
        except ImportError as e:
            logger.error(f"Failed to import live components: {e}")
            logger.error("Make sure ccxt is installed: pip install ccxt")
            sys.exit(1)

        adapter = BinanceAdapter()
        feature_computer = LiveFeatureComputer(warmup_bars=500)

        # Warmup: fetch historical candles
        logger.info("Warming up with 500 historical candles...")
        hist_candles = adapter.fetch_ohlcv_1h(limit=500)
        feature_computer.ingest_candles(hist_candles)
        logger.info(f"Warmup complete: {len(hist_candles)} candles ingested")

        # Load saved state if available
        self.load_state()

        # SIGINT handler
        def shutdown_handler(signum, frame):
            logger.info("Shutdown signal received. Saving state...")
            self.running = False
            self.save_state()
            self.print_summary()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        logger.info("Starting live shadow mode (Ctrl+C to stop)...")
        last_processed_ts = None

        while self.running:
            now = datetime.now(timezone.utc)

            # Wait for candle close (process at XX:01:30 to ensure candle is finalized)
            minutes_past_hour = now.minute
            if minutes_past_hour < 1:
                sleep_secs = (1 - minutes_past_hour) * 60 + 30 - now.second
                if sleep_secs > 0:
                    logger.debug(f"Waiting {sleep_secs}s for candle close...")
                    time.sleep(sleep_secs)
                    continue

            # Fetch latest candle
            try:
                latest_candles = adapter.fetch_ohlcv_1h(limit=2)
                if latest_candles.empty:
                    logger.warning("No candles returned from Binance")
                    time.sleep(60)
                    continue

                # Use the second-to-last candle (the most recently COMPLETED one)
                if len(latest_candles) >= 2:
                    latest_candle = latest_candles.iloc[-2]
                    candle_ts = latest_candles.index[-2]
                else:
                    latest_candle = latest_candles.iloc[-1]
                    candle_ts = latest_candles.index[-1]

                # Skip if already processed
                if last_processed_ts is not None and candle_ts <= last_processed_ts:
                    time.sleep(60)
                    continue

                # Compute features
                features = feature_computer.update(latest_candle.to_dict())

                # Fetch and inject funding rate
                try:
                    funding_rate = adapter.fetch_funding_rate()
                    features['funding_rate'] = funding_rate
                except Exception:
                    pass

                # Process bar
                acted = self.process_bar(features, candle_ts)
                last_processed_ts = candle_ts

                # Heartbeat log
                equity = self.equity_curve[-1] if self.equity_curve else self.initial_cash
                logger.info(
                    f"[HEARTBEAT] {candle_ts} | BTC=${features['close']:,.0f} | "
                    f"Regime: {features.get('regime_label', '?')} | "
                    f"Positions: {len(self.positions)} | "
                    f"Equity: ${equity:,.0f} | Signals: {len(acted)}"
                )

                # Save state periodically
                if self.bar_index % 6 == 0:  # Every 6 hours
                    self.save_state()

            except Exception as e:
                logger.error(f"Error processing candle: {e}", exc_info=True)
                time.sleep(60)
                continue

            # Sleep until next hour
            now = datetime.now(timezone.utc)
            next_hour = now.replace(minute=1, second=30, microsecond=0) + timedelta(hours=1)
            sleep_secs = (next_hour - now).total_seconds()
            if sleep_secs > 0:
                logger.debug(f"Sleeping {sleep_secs:.0f}s until next candle...")
                time.sleep(sleep_secs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='V11 Shadow Runner for Bull Machine')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true', help='Live mode (real Binance data)')
    mode.add_argument('--replay', action='store_true', help='Replay mode (historical feature store)')

    parser.add_argument('--config', type=str,
                        default='configs/bull_machine_isolated_v11_fixed.json',
                        help='Path to config JSON')
    parser.add_argument('--feature-store', type=str,
                        default='data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet',
                        help='Path to feature store (replay mode)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-cash', type=float, default=100_000.0, help='Initial cash ($)')
    parser.add_argument('--commission-rate', type=float, default=0.0004, help='Commission per side')
    parser.add_argument('--slippage-bps', type=float, default=5.0, help='Slippage bps per side')
    parser.add_argument('--verbose', action='store_true', help='Debug logging')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    runner = V11ShadowRunner(
        config_path=args.config,
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
    )

    if args.replay:
        runner.run_replay(
            feature_store_path=args.feature_store,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    elif args.live:
        runner.run_live()


if __name__ == '__main__':
    main()
