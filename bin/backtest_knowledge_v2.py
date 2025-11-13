#!/usr/bin/env python3
"""
Knowledge-Aware Backtest Engine v2.0

Uses ALL 69 features from the MTF feature store intelligently:
- Advanced fusion scoring with PTI, Macro, Wyckoff M1/M2, FRVP
- Smart entry logic (tiered entries, pullback waiting, limit orders)
- Smart exit integration (partial exits, trailing stops, breakeven)
- ATR-based position sizing (1-2% risk per trade)
- Macro regime filtering
- Full knowledge hooks integration

This is the COMPLETE backtest engine that replaces the simplified
optimizer placeholder (bin/optimize_v2_cached.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging
import os
import traceback
import joblib  # ML model loading

# PR#4: Runtime liquidity scoring
from engine.liquidity.score import compute_liquidity_score, compute_liquidity_telemetry

# PR#6A: Archetype expansion (using package import for adapter support)
from engine.archetypes import ArchetypeLogic, ArchetypeTelemetry

# PR#6B: Adaptive Fusion (regime-aware parameter morphing)
try:
    # Use GMM classifier trained on real macro data
    from engine.context.regime_classifier import RegimeClassifier
    from engine.fusion.adaptive import AdaptiveFusion
    from engine.runtime.context import RuntimeContext
    from engine.archetypes.threshold_policy import ThresholdPolicy
    ADAPTIVE_FUSION_AVAILABLE = True
except ImportError:
    ADAPTIVE_FUSION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveHoldEvent:
    """Records when adaptive max-hold extends a position's hold time"""
    trade_id: int
    timestamp: str
    base_bars: int
    extended_bars: int
    wyckoff_phase: str
    wyckoff_score: float
    current_pnl_pct: float
    atr_value: float
    reason: str


@dataclass
class KnowledgeParams:
    """
    Knowledge-aware backtest parameters.

    Expanded from simple fusion weights to include smart entry/exit logic.
    """
    # Domain weights (must sum to ≤ 1.0)
    # OPTIMIZED: Trial 21 weights (BTC 2024)
    wyckoff_weight: float = 0.331       # 33.1% (was 0.30)
    liquidity_weight: float = 0.392     # 39.2% (was 0.30)
    momentum_weight: float = 0.205      # 20.5% (was 0.20)
    macro_weight: float = 0.00          # 0% (was 0.10) - disabled by optimizer
    pti_weight: float = 0.072           # 7.2% (was 0.10) - remaining weight

    # Entry thresholds (tiered) - Adjusted to realistic fusion score range (0.0-0.5)
    tier1_threshold: float = 0.45  # Ultra-high conviction (market entry)
    tier2_threshold: float = 0.40  # High conviction (limit entry on pullback)
    tier3_threshold: float = 0.374  # OPTIMIZED: Trial 21 (was 0.25)

    # Entry modifiers
    require_m1m2_confirmation: bool = True  # Require Wyckoff M1/M2 signal
    require_macro_alignment: bool = True     # Require risk_on regime
    frvp_entry_zone: str = "value_area"      # Enter near POC/value_area
    pullback_depth: float = 0.382            # Wait for 38.2% pullback (Fib)

    # Exit management
    use_smart_exits: bool = True              # Enable smart_exits.py integration
    partial_exit_1: float = 0.33              # Exit 33% at TP1 (+1R)
    partial_exit_2: float = 0.33              # Exit 33% at TP2 (+2R)
    trailing_atr_mult: float = 2.0            # Trail stop 2× ATR from peak
    breakeven_after_tp1: bool = True          # Move stop to breakeven after TP1
    max_hold_bars: int = 168                  # Max 168 hours (7 days)
    adaptive_max_hold: bool = False           # Adjust max_hold based on regime/phase

    # Position sizing
    max_risk_pct: float = 0.02                # Max 2% risk per trade
    atr_stop_mult: float = 2.5                # Stop loss 2.5× ATR below entry
    position_size_method: str = "atr"         # "atr" or "fixed"
    volatility_scaling: bool = True           # Scale down in high VIX

    # Costs
    slippage_bps: float = 2.0                 # 2 basis points
    fee_bps: float = 1.0                      # 1 basis point


@dataclass
class Trade:
    """Trade record with full knowledge context."""
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float  # USD value
    direction: int  # +1 long, -1 short
    entry_fusion_score: float
    entry_reason: str  # "tier1_market", "tier2_pullback", "tier3_scale"

    # Entry knowledge context
    wyckoff_phase: str
    wyckoff_m1_signal: Optional[str]
    wyckoff_m2_signal: Optional[str]
    macro_regime: str
    pti_score_1d: float
    pti_score_1h: float
    frvp_poc_position: str
    entry_archetype: Optional[str] = None  # Track B: archetype that triggered entry

    # Exit tracking
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    partial_exits: List[Dict] = field(default_factory=list)

    # PNL tracking
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0

    # Risk metrics
    atr_at_entry: float = 0.0
    initial_stop: float = 0.0
    peak_profit: float = 0.0
    max_adverse_excursion: float = 0.0

    # ML Export: Store full entry context for feature extraction
    entry_context: Optional[Dict] = None


class KnowledgeAwareBacktest:
    """
    Full knowledge backtest engine using all 69 MTF features.
    """

    def __init__(self, df: pd.DataFrame, params: KnowledgeParams, starting_capital: float = 10000.0, asset: str = 'BTC', runtime_config: Optional[Dict] = None):
        """
        Initialize backtest with feature store and parameters.

        Args:
            df: MTF feature store (69 features)
            params: Knowledge-aware parameters
            starting_capital: Starting equity
            asset: Asset symbol (e.g., 'BTC', 'ETH')
            runtime_config: PR#4 runtime intelligence config (optional)
        """
        self.df = df.copy()
        self.params = params
        self.starting_capital = starting_capital
        self.equity = starting_capital
        self.peak_equity = starting_capital
        self.asset = asset  # Phase 4: Store asset for re-entry window logic

        # PR#4: Runtime intelligence configuration
        self.runtime_config = runtime_config or {}
        self.liquidity_enabled = self.runtime_config.get('runtime', {}).get('runtime_liquidity_enabled', False)
        self.liquidity_scores: List[float] = []  # Track scores for telemetry

        # ML Filter initialization
        self.ml_model = None
        self.ml_feature_names = None
        self.ml_threshold = 0.707
        ml_config = self.runtime_config.get('ml_filter', {})
        if ml_config.get('enabled'):
            try:
                model_path = ml_config['model_path']
                model_data = joblib.load(model_path)
                self.ml_model = model_data['model']
                self.ml_feature_names = model_data['feature_names']
                self.ml_threshold = float(ml_config.get('threshold', model_data.get('threshold', 0.707)))
                logger.info(f"ML Filter: loaded {model_path} (threshold={self.ml_threshold:.3f}, {len(self.ml_feature_names)} features)")
            except Exception as e:
                logger.warning(f"ML Filter: failed to load - {e}")
                self.ml_model = None

        # PR#6B: Adaptive Fusion initialization
        self.regime_classifier = None
        self.adaptive_fusion = None
        adaptive_config = self.runtime_config.get('fusion_adapt', {})
        regime_config = self.runtime_config.get('regime_classifier', {})

        if ADAPTIVE_FUSION_AVAILABLE and adaptive_config.get('enable'):
            try:
                # Load regime classifier
                model_path = regime_config.get('model_path', 'models/regime_classifier_gmm.pkl')
                feature_order = regime_config.get('feature_order', [])
                zero_fill_missing = regime_config.get('zero_fill_missing', False)
                regime_override = regime_config.get('regime_override', None)  # Optional date-based overrides
                self.regime_classifier = RegimeClassifier.load(model_path, feature_order, zero_fill_missing, regime_override)

                # Initialize adaptive fusion coordinator
                self.adaptive_fusion = AdaptiveFusion(self.runtime_config)

                # PR#6B: Initialize ThresholdPolicy for regime-aware threshold management
                # PR-A: Support locked_regime for parity testing
                locked_regime = self.runtime_config.get('locked_regime')  # 'static', 'risk_on', etc.
                self.threshold_policy = ThresholdPolicy(
                    base_cfg=self.runtime_config,
                    regime_profiles=self.runtime_config.get('gates_regime_profiles'),
                    archetype_overrides=self.runtime_config.get('archetype_overrides'),
                    global_clamps=self.runtime_config.get('global_clamps'),
                    locked_regime=locked_regime
                )

                logger.info(f"Adaptive Fusion: ENABLED (ema_alpha={adaptive_config.get('ema_alpha', 0.2)})")
                logger.info(f"ThresholdPolicy: ENABLED (regime-aware archetype thresholds)")
            except Exception as e:
                print(f"🔴 ADAPTIVE FUSION INIT FAILED: {e}")
                print(f"🔴 Traceback:")
                traceback.print_exc()
                logger.warning(f"Adaptive Fusion: failed to initialize - {e}")
                self.regime_classifier = None
                self.adaptive_fusion = None
                self.threshold_policy = None
        else:
            if not ADAPTIVE_FUSION_AVAILABLE:
                logger.info("Adaptive Fusion: DISABLED (modules not available)")
            else:
                logger.info("Adaptive Fusion: DISABLED (enable=false in config)")

            # Still initialize ThresholdPolicy for archetype threshold management
            # even without AdaptiveFusion (will use force_regime or neutral)
            locked_regime = self.runtime_config.get('locked_regime', 'static')
            self.threshold_policy = ThresholdPolicy(
                base_cfg=self.runtime_config,
                regime_profiles=self.runtime_config.get('gates_regime_profiles'),
                archetype_overrides=self.runtime_config.get('archetype_overrides'),
                global_clamps=self.runtime_config.get('global_clamps'),
                locked_regime=locked_regime
            )
            logger.info(f"ThresholdPolicy: ENABLED (static mode, locked_regime={locked_regime})")

        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None

        # Adaptive max-hold instrumentation
        self._adaptive_extension_count = 0
        self._adaptive_events: List[AdaptiveHoldEvent] = []
        self._last_ctx_snapshot: Optional[Dict] = None

        # Phase 4: Re-Entry Tracking
        self._last_exit_bar: Optional[int] = None
        self._last_exit_price: Optional[float] = None
        self._last_exit_reason: Optional[str] = None
        self._last_exit_direction: Optional[int] = None
        self._last_exit_size: Optional[float] = None
        self._reentry_count = 0

        # PR#6A: Archetype system (11-archetype expansion)
        archetype_config = self.runtime_config.get('archetypes', {})
        use_archetypes = archetype_config.get('use_archetypes', False)
        logger.info(f"DIAGNOSTIC: use_archetypes={use_archetypes}, archetype_config_keys={list(archetype_config.keys())}")
        if use_archetypes:
            # Bull Machine v2: Merge top-level config keys needed by ArchetypeLogic
            # (fusion weights, state_aware_gates) into archetype_config
            full_archetype_config = {
                **archetype_config,
                'fusion': self.runtime_config.get('fusion', {}),
                'state_aware_gates': self.runtime_config.get('state_aware_gates', {})
            }
            self.archetype_logic = ArchetypeLogic(full_archetype_config)
            self.archetype_telemetry = ArchetypeTelemetry()
            logger.info("PR#6A: 11-archetype system ENABLED")
        else:
            self.archetype_logic = None
            self.archetype_telemetry = None
            logger.info("DIAGNOSTIC: 11-archetype system DISABLED - falling back to legacy 3-archetype")
            # Fallback: Old 3-archetype tracking (deprecated)
            self._archetype_checks = 0
            self._archetype_a_matches = 0  # Trap Reversal
            self._archetype_b_matches = 0  # OB Retest
            self._archetype_c_matches = 0  # FVG Continuation

        # Diagnostic: Track veto reasons
        self._veto_metrics = {
            'pre_archetype_candidates': 0,
            'archetype_hits': 0,
            'archetype_no_match': 0,
            'veto_final_fusion_gate': 0,
            'veto_crisis_regime': 0,
            'veto_risk_off_regime': 0,
            'veto_ml_filter': 0,
            'veto_archetype_cooldown': 0,
            'veto_max_trades_per_day': 0,
            'veto_monthly_share_cap': 0
        }

        # Step 1 Stabilization: Archetype cooldown and daily trade limits
        self._archetype_last_entry_bar = {}  # Track last entry bar per archetype
        self._trades_today = 0
        self._current_day = None

        # Step 2: Monthly share cap tracking
        self._monthly_archetype_counts = {}  # {(year, month): {archetype: count}}
        self._current_month = None
        self._monthly_total_trades = 0

        # ML Optimization: Exit Strategy Parameters (read from env or use defaults)
        # Phase 2: Pattern Exit Parameters
        self.pattern_confluence_threshold = int(os.getenv('EXIT_PATTERN_CONFLUENCE', '3'))

        # Phase 2: Structure Exit Parameters
        self.structure_min_hold_bars = int(os.getenv('EXIT_STRUCT_MIN_HOLD', '12'))
        self.structure_rsi_long_threshold = int(os.getenv('EXIT_STRUCT_RSI_LONG', '25'))
        self.structure_rsi_short_threshold = int(os.getenv('EXIT_STRUCT_RSI_SHORT', '75'))
        self.structure_vol_zscore_min = float(os.getenv('EXIT_STRUCT_VOL_Z', '1.0'))

        # Phase 3: Trailing Stop Parameters
        self.trailing_stop_base_mult = float(os.getenv('EXIT_TRAILING_BASE', '2.0'))
        self.trailing_stop_trending_mult = float(os.getenv('EXIT_TRAILING_TREND', '2.5'))

        # Phase 4: Re-Entry Parameters
        self.reentry_confluence_threshold = int(os.getenv('EXIT_REENTRY_CONF', '3'))
        self.reentry_window_btc_eth = int(os.getenv('EXIT_REENTRY_WINDOW', '7'))
        self.reentry_fusion_delta = float(os.getenv('EXIT_REENTRY_DELTA', '0.05'))

        # Precompute ATR for position sizing
        self._precompute_atr()

    def _precompute_atr(self):
        """Precompute ATR for the entire dataset."""
        # ATR should already be in feature store, but compute if missing
        if 'atr_14' not in self.df.columns:
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            prev_close = close.shift(1)

            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)

            self.df['atr_14'] = tr.ewm(span=14, adjust=False).mean()

    def compute_advanced_fusion_score(self, row: pd.Series, adapted_params: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Compute advanced fusion score using ALL 69 features.

        Args:
            row: Feature store row
            adapted_params: Optional adaptive fusion parameters (from AdaptiveFusion.update())

        Returns:
            (fusion_score, context_dict)
        """
        context = {}

        # PR#6B: Store regime info in context if adaptive fusion is enabled
        if adapted_params:
            context['regime'] = adapted_params.get('regime', 'neutral')
            context['regime_probs_ema'] = adapted_params.get('regime_probs_ema', {})

        # 1. Wyckoff Governor (1D) - Include M1/M2 signals
        wyckoff_base = row.get('tf1d_wyckoff_score', 0.5)
        m1_strength = row.get('tf1d_m1_signal_strength', 0.0)
        m2_strength = row.get('tf1d_m2_signal_strength', 0.0)

        # Boost Wyckoff score if M1/M2 signals are present
        wyckoff_boost = max(m1_strength, m2_strength) * 0.3  # Up to +0.3 boost
        wyckoff = np.clip(wyckoff_base + wyckoff_boost, 0.0, 1.0)

        context['wyckoff_score'] = wyckoff
        context['wyckoff_phase'] = row.get('tf1d_wyckoff_phase', 'unknown')
        context['m1_signal'] = row.get('tf1d_m1_signal', None)
        context['m2_signal'] = row.get('tf1d_m2_signal', None)

        # 2. Liquidity (BOMS/FVG based)
        boms_strength = row.get('tf1d_boms_strength', 0.0)
        fvg_present = 1.0 if row.get('tf4h_fvg_present', False) else 0.0

        # Normalize BOMS displacement to 0-1 range based on ATR
        boms_disp = row.get('tf4h_boms_displacement', 0.0)
        atr = row.get('atr_14', 500.0)
        disp_normalized = min(boms_disp / (2.0 * atr), 1.0) if atr > 0 else 0.0

        liquidity = (boms_strength + fvg_present + disp_normalized) / 3.0
        context['liquidity_score'] = liquidity

        # 3. Momentum (ADX + RSI + Squiggle)
        adx = row.get('adx_14', 20.0) / 100.0
        rsi = row.get('rsi_14', 50.0)
        rsi_momentum = abs(rsi - 50.0) / 50.0
        squiggle_conf = row.get('tf4h_squiggle_confidence', 0.5)

        momentum = (adx + rsi_momentum + squiggle_conf) / 3.0
        context['momentum_score'] = momentum

        # 4. PTI (Trap Detection) - Inverse scoring (high PTI = avoid)
        pti_1d = row.get('tf1d_pti_score', 0.0)
        pti_1h = row.get('tf1h_pti_score', 0.0)
        pti_combined = max(pti_1d, pti_1h)

        # PTI acts as a penalty (high PTI = likely trap)
        pti_penalty = pti_combined  # 0.0-1.0 scale
        context['pti_score'] = pti_combined
        context['pti_1d'] = pti_1d
        context['pti_1h'] = pti_1h

        # 5. Macro (Regime + Trends)
        macro_regime = row.get('macro_regime', 'neutral')
        macro_vix = row.get('macro_vix_level', 'medium')

        # Regime scoring: risk_on = 1.0, neutral = 0.5, risk_off/crisis = 0.0
        regime_map = {'risk_on': 1.0, 'neutral': 0.5, 'risk_off': 0.2, 'crisis': 0.0}
        regime_score = regime_map.get(macro_regime, 0.5)

        # VIX penalty: low = 1.0, medium = 0.8, high = 0.5, extreme = 0.2
        vix_map = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.2}
        vix_score = vix_map.get(macro_vix, 0.8)

        macro = (regime_score + vix_score) / 2.0
        context['macro_score'] = macro
        context['macro_regime'] = macro_regime
        context['macro_vix'] = macro_vix

        # 6. FRVP (Value Area Positioning)
        frvp_poc_pos = row.get('tf1h_frvp_poc_position', 'middle')

        # Prefer entries near value area (POC)
        poc_map = {'below': 0.3, 'at_poc': 1.0, 'above': 0.3, 'middle': 0.6}
        frvp_score = poc_map.get(frvp_poc_pos, 0.5)
        context['frvp_score'] = frvp_score
        context['frvp_poc_position'] = frvp_poc_pos

        # PR#6B: Weighted fusion score (adaptive or static)
        if adapted_params and adapted_params.get('fusion_weights'):
            # Use adaptive weights (regime-aware blending)
            fusion_weights = adapted_params['fusion_weights']

            # Adaptive fusion uses 4 components: wyckoff, liquidity, momentum, temporal
            # Note: temporal is not computed in this method yet, so default to 0
            temporal_score = context.get('temporal_score', 0.0)

            fusion = (
                fusion_weights.get('wyckoff', 0.0) * wyckoff +
                fusion_weights.get('liquidity', 0.0) * liquidity +
                fusion_weights.get('momentum', 0.0) * momentum +
                fusion_weights.get('temporal', 0.0) * temporal_score
            )
            # Note: Adaptive mode doesn't use PTI penalty - it's regime-aware instead
        else:
            # Fallback to static weights (original logic)
            fusion = (
                self.params.wyckoff_weight * wyckoff +
                self.params.liquidity_weight * liquidity +
                self.params.momentum_weight * momentum +
                self.params.macro_weight * macro +
                (1.0 - self.params.wyckoff_weight - self.params.liquidity_weight -
                 self.params.momentum_weight - self.params.macro_weight) * frvp_score
            )

            # Apply PTI penalty (subtract PTI weight × PTI score)
            fusion -= self.params.pti_weight * pti_penalty

        # Apply fakeout penalty
        if row.get('tf1h_fakeout_detected', False):
            fusion -= 0.1  # -10% penalty for fakeouts

        # Apply governor veto
        if row.get('mtf_governor_veto', False):
            fusion *= 0.3  # Severe penalty for governor veto

        # Clip to [0, 1]
        fusion = np.clip(fusion, 0.0, 1.0)

        context['fusion_score'] = fusion
        return fusion, context

    def classify_entry_archetype(self, row: pd.Series, context: Dict) -> Optional[Tuple[str, float, float]]:
        """
        PR#6A: Classify entry opportunity into archetypes.

        - When use_archetypes=true: Uses 11-archetype system (A-H + K,L,M)
        - When use_archetypes=false: Uses legacy 3-archetype system (A, B, C)

        Returns:
            (archetype_name, threshold, size_multiplier) or None if no archetype matches
        """
        # PR#6A: Use new 11-archetype system if enabled
        if self.archetype_logic and self.archetype_telemetry:
            self.archetype_telemetry.check()

            # BUGFIX: Inject runtime scores into row for archetype detection
            # The archetype logic looks for liquidity_score in the row, but runtime
            # scores are only stored in context. Copy them over.
            row_with_runtime = row.copy()
            if 'liquidity_score' in context:
                row_with_runtime['liquidity_score'] = context['liquidity_score']
            if 'fusion_score' in context:
                row_with_runtime['fusion_score'] = context['fusion_score']

            # PR#6B: Build RuntimeContext with regime-aware thresholds
            # PR-A PARITY FIX: If ThresholdPolicy exists, ALWAYS use RuntimeContext path
            # This ensures locked mode configs don't fall back to legacy path
            if self.threshold_policy:
                # Get adapted_params from context, or use fallback for locked/static mode
                adapted_params = context.get('adapted_params', {})

                # BEAR ARCHETYPE TESTING: Check for forced regime override
                force_regime = self.runtime_config.get('archetypes', {}).get('force_regime', None)
                if force_regime:
                    regime_probs = {force_regime: 1.0}
                    regime_label = force_regime
                    print(f"🔴 [FORCE REGIME] Overriding regime to: {force_regime}")
                    logger.info(f"[FORCE REGIME] Overriding regime to: {force_regime}")
                # BUGFIX: AdaptiveFusion returns 'regime_probs_ema', not 'regime_probs'
                # Default to neutral if no regime info available (locked mode)
                elif adapted_params:
                    regime_probs = adapted_params.get('regime_probs_ema', {'neutral': 1.0})
                    regime_label = adapted_params.get('regime', 'neutral')
                else:
                    regime_probs = {'neutral': 1.0}
                    regime_label = 'neutral'

                # Resolve thresholds using ThresholdPolicy (will use locked mode if configured)
                thresholds = self.threshold_policy.resolve(regime_probs, regime_label)

                # BEAR ARCHETYPE FIX: Prepare metadata for archetypes that need historical context
                current_idx = context.get('current_index', 0)
                prev_row = self.df.iloc[current_idx - 1] if current_idx > 0 else None

                # Build RuntimeContext
                runtime_ctx = RuntimeContext(
                    ts=row.name if hasattr(row, 'name') else context.get('current_index', 0),
                    row=row_with_runtime,
                    regime_probs=regime_probs,
                    regime_label=regime_label,
                    adapted_params=adapted_params,
                    thresholds=thresholds,
                    metadata={
                        'prev_row': prev_row,
                        'df': self.df,
                        'index': current_idx
                    }
                )

                # Call new detect() method with RuntimeContext
                archetype_name, fusion_score, liquidity_score = self.archetype_logic.detect(runtime_ctx)
            else:
                # Fallback to old API if ThresholdPolicy not available (truly legacy configs)
                prev_row = None
                current_idx = context.get('current_index', 0)
                archetype_name, fusion_score, liquidity_score = self.archetype_logic.check_archetype(
                    row=row_with_runtime,
                    prev_row=prev_row,
                    df=self.df,
                    index=current_idx
                )

            # SCORE PROPAGATION FIX: Update context with archetype-specific score
            # This ensures ML filter, position sizing, and all downstream gates use the
            # archetype-specific score instead of the global fusion_score
            if archetype_name:
                context['fusion_score'] = fusion_score  # Overwrite global fusion with archetype-specific score
                context['liquidity_score'] = liquidity_score
                logger.debug(f"[SCORE PROPAGATION] Updated context: fusion={fusion_score:.3f} (archetype-specific)")

            # Track telemetry
            if archetype_name:
                self.archetype_telemetry.count(archetype_name)
                logger.info(f"PR#6A: Archetype {archetype_name.upper()} detected (fusion={fusion_score:.3f}, liq={liquidity_score:.3f})")

                # Map archetype to threshold and size multiplier
                # Size multipliers adjusted based on PR#6A validation results:
                # - trap_within_trend (H): 0.90 → 0.60 (dominated in untuned, lower quality)
                # - wick_trap (K): 1.0 → 1.25 (higher win rate in validation)
                # - volume_exhaustion (L): 0.90 → 1.30 (strong setup, rare but high quality)
                archetype_map = {
                    'trap_reversal': (0.33, 0.75),
                    'order_block_retest': (0.37, 1.0),
                    'fvg_continuation': (0.42, 1.25),
                    'failed_continuation': (0.42, 0.85),
                    'liquidity_compression': (0.35, 1.0),
                    'expansion_exhaustion': (0.38, 0.80),
                    'reaccumulation': (0.40, 1.15),
                    'trap_within_trend': (0.35, 0.60),        # Reduced from 0.90
                    'wick_trap': (0.36, 1.25),                # Increased from 1.0
                    'volume_exhaustion': (0.38, 1.30),        # Increased from 0.90
                    'ratio_coil_break': (0.35, 1.10)
                }

                threshold, size_mult = archetype_map.get(archetype_name, (0.40, 1.0))
                return (archetype_name, threshold, size_mult)

            return None

        # Fallback: Legacy 3-archetype system (deprecated)
        self._archetype_checks += 1

        # Get required features
        pti_trap = row.get('tf1h_pti_trap_type', None)
        pti_score = context.get('pti_score', 0.0)
        boms_disp = row.get('tf4h_boms_displacement', 0.0)
        atr = row.get('atr_14', row['close'] * 0.02)

        bos_bull = row.get('tf1h_bos_bullish', False)
        bos_bear = row.get('tf1h_bos_bearish', False)
        boms_strength = row.get('tf1d_boms_strength', 0.0)

        fvg_1h = row.get('tf1h_fvg_present', False)
        fvg_4h = row.get('tf4h_fvg_present', False)
        tf4h_fusion = row.get('tf4h_fusion_score', 0.0)

        liq_score = context.get('liquidity_score', 0.0)
        wyc_score = context.get('wyckoff_score', 0.0)
        mom_score = context.get('momentum_score', 0.0)
        frvp_pos = context.get('frvp_poc_position', 'middle')

        # Archetype A: Trap Reversal
        if pti_trap is not None and pti_score >= 0.40 and boms_disp >= (0.80 * atr):
            if frvp_pos in ['at_poc', 'middle']:
                self._archetype_a_matches += 1
                return ("trap_reversal", 0.33, 0.75)

        # Archetype B: OB/pHOB Retest
        if (bos_bull or bos_bear) and boms_strength >= 0.30:
            if liq_score >= 0.25 and wyc_score >= 0.35:
                self._archetype_b_matches += 1
                return ("ob_retest", 0.37, 1.0)

        # Archetype C: FVG/Breaker Continuation
        if (fvg_1h or fvg_4h) and boms_disp >= (1.0 * atr):
            if liq_score >= 0.30 and mom_score >= 0.45:
                size_mult = 1.25 if tf4h_fusion >= 0.25 else 1.0
                self._archetype_c_matches += 1
                return ("fvg_continuation", 0.42, size_mult)

        return None

    def calculate_position_size(self, row: pd.Series, fusion_score: float, context: Dict = None) -> float:
        """
        Calculate position size using ATR-based risk management + archetype sizing.

        Target: 1-2% equity at risk per trade.
        """
        if self.params.position_size_method == "fixed":
            # Simple: 95% allocation (old method)
            return self.equity * 0.95

        # ATR-based sizing
        atr = row.get('atr_14', row['close'] * 0.02)  # Default to 2% of price

        # Stop loss distance (2.5× ATR)
        stop_distance = atr * self.params.atr_stop_mult

        # Risk amount (2% of equity)
        risk_dollars = self.equity * self.params.max_risk_pct

        # Position size = risk / stop_distance
        position_size = risk_dollars / (stop_distance / row['close'])

        # Volatility scaling (reduce in high VIX)
        if self.params.volatility_scaling:
            vix_level = row.get('macro_vix_level', 'medium')
            vix_scaling = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.25}
            position_size *= vix_scaling.get(vix_level, 0.8)

        # DYNAMIC SIZING PHASE 1: Fusion-based weight (recentered to observed distribution)
        # Get sizing params from config (with sensible defaults)
        fusion_center = self.runtime_config.get('decision_gates', {}).get('fusion_center', 0.475)
        fusion_gain = self.runtime_config.get('decision_gates', {}).get('fusion_gain', 1.6)
        size_min = self.runtime_config.get('decision_gates', {}).get('size_min', 0.75)
        size_max = self.runtime_config.get('decision_gates', {}).get('size_max', 1.35)

        # Linear recentered formula: weight = 1.0 + gain × (fusion - center)
        fusion_weight = 1.0 + fusion_gain * (fusion_score - fusion_center)
        fusion_weight = max(size_min, min(size_max, fusion_weight))
        position_size *= fusion_weight
        logger.info(f"DYNAMIC SIZING: fusion={fusion_score:.3f} (center={fusion_center:.3f}) → weight={fusion_weight:.2f}x")

        # DYNAMIC SIZING PHASE 2: Regime multipliers
        regime = context.get('macro_regime', 'neutral') if context else 'neutral'
        regime_multipliers = {
            'risk_on': 1.2,
            'neutral': 1.0,
            'risk_off': 0.8,
            'crisis': 0.5
        }
        regime_mult = regime_multipliers.get(regime, 1.0)
        position_size *= regime_mult
        logger.info(f"REGIME MULT: {regime} → {regime_mult:.2f}x")

        # ARCHETYPE SIZING MULTIPLIER (NEW - Phase 3-archetype system)
        if context and 'archetype_size_mult' in context:
            arch_mult = context['archetype_size_mult']
            position_size *= arch_mult
            archetype_name = context.get('entry_archetype', 'unknown')
            logger.info(f"ARCHETYPE SIZING: {archetype_name} × {arch_mult:.2f} → ${position_size:,.0f}")

        # ML SIZING MULTIPLIER (Phase 4: ML-based confidence sizing)
        if context and 'ml_size_mult' in context:
            ml_mult = context['ml_size_mult']
            position_size *= ml_mult
            ml_score = context.get('ml_entry_score', 0.0)
            logger.info(f"ML SIZING: score={ml_score:.3f} × {ml_mult:.2f} → ${position_size:,.0f}")

        # Cap at 95% of equity
        position_size = min(position_size, self.equity * 0.95)

        return position_size

    def _extract_ml_features(self, row: pd.Series, context: Dict) -> np.ndarray:
        """Extract 44 features matching training schema"""
        if not self.ml_feature_names:
            return None

        features = {}

        # Archetype one-hot (match export schema)
        archetype = context.get('entry_archetype', 'unknown')
        for arch_key in ['trap', 'retest', 'continuation', 'failed_continuation',
                         'compression', 'exhaustion', 'reaccumulation', 'trap_within_trend',
                         'wick_trap', 'volume_exhaustion', 'ratio_coil_break', 'false_break_reversal']:
            features[f'archetype_{arch_key}'] = 1 if archetype == arch_key else 0

        # Core scores
        features['entry_fusion_score'] = float(context.get('fusion_score', 0.5))
        features['entry_liquidity_score'] = float(context.get('liquidity_score', 0.0))

        # Macro regime one-hot (match training: macro_regime_risk_on, etc.)
        regime = context.get('macro_regime', 'neutral')
        for r in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            features[f'macro_regime_{r}'] = 1 if regime == r else 0

        # Technicals from row (safe access with defaults)
        def safe_get(col, default=0.0):
            val = row.get(col, default)
            try:
                return float(val) if val is not None and not np.isnan(val) else float(default)
            except:
                return float(default)

        features['vix_z_score'] = safe_get('vix_z_score', 0.0)
        features['btc_volatility_percentile'] = safe_get('btc_volatility_percentile', 0.5)
        features['volume_zscore'] = safe_get('volume_zscore', 0.0)
        features['atr_percentile'] = safe_get('atr_percentile', 0.5)
        features['adx_14'] = safe_get('adx', 0.0)
        features['rsi_14'] = safe_get('rsi', 50.0)
        features['macd_histogram'] = safe_get('macd_histogram', 0.0)

        # MTF fusion
        features['tf1h_fusion'] = safe_get('tf1h_fusion_score', features['entry_fusion_score'])
        features['tf4h_fusion'] = safe_get('tf4h_fusion_score', 0.0)
        features['tf1d_fusion'] = safe_get('tf1d_fusion_score', 0.0)
        features['tf4h_trend_aligned'] = safe_get('tf4h_trend_aligned', 0.0)
        features['tf1d_trend_aligned'] = safe_get('tf1d_trend_aligned', 0.0)
        features['nested_structure_quality'] = safe_get('nested_structure_quality', 0.0)

        # Microstructure
        features['boms_strength'] = safe_get('boms_strength', 0.0)
        features['fvg_quality'] = safe_get('fvg_quality', 0.0)
        features['wyckoff_phase_score'] = safe_get('wyckoff_phase_score', 0.0)
        features['poc_distance'] = safe_get('frvp_poc_distance', 0.5)
        features['lvn_trap_risk'] = safe_get('lvn_trap_risk', 0.0)
        features['liquidity_sweep_strength'] = safe_get('liquidity_sweep_strength', 0.0)

        # Recent performance (defaults if not tracked)
        features['last_3_trades_wr'] = float(context.get('last_3_trades_wr', 0.5))
        features['bars_since_last_trade'] = float(context.get('bars_since_last_trade', 48))
        features['recent_dd_pct'] = float(context.get('recent_dd_pct', 0.0))
        features['streak_length'] = float(context.get('streak_length', 0.0))

        # Timing
        ts = row.name if hasattr(row, 'name') else context.get('timestamp')
        if hasattr(ts, 'hour'):
            hour = int(ts.hour)
            dow = int(ts.weekday())
            doy = int(ts.dayofyear) if hasattr(ts, 'dayofyear') else 1
        else:
            hour, dow, doy = 0, 0, 1

        features['hour_of_day'] = float(hour)
        features['day_of_week'] = float(dow)
        features['days_into_quarter'] = float(doy % 90)

        # Return ordered array matching training
        return np.array([features.get(name, 0.0) for name in self.ml_feature_names])

    def check_entry_conditions(self, row: pd.Series, fusion_score: float, context: Dict) -> Optional[Tuple[str, float]]:
        """
        Check if entry conditions are met using 3-archetype classification system.

        Phase 1: Try archetype classification (context-specific thresholds)
        Phase 2: Fallback to legacy tiered system (safety net for ultra-high fusion)

        Returns:
            (entry_type, entry_price) or None
        """
        # PHASE 1: Try archetype classification first
        self._veto_metrics['pre_archetype_candidates'] += 1
        archetype_result = self.classify_entry_archetype(row, context)

        if archetype_result:
            self._veto_metrics['archetype_hits'] += 1
            archetype_name, threshold, size_mult = archetype_result

            # CRITICAL FIX: Use archetype-specific score from context, not global parameter!
            # The dispatcher calculated archetype-specific scores and stored in context.
            # Using the global fusion_score parameter causes false rejections for VE/OB.
            archetype_fusion = context.get('fusion_score', fusion_score)

            # Add veto tracking for threshold failures (previously silent)
            if archetype_fusion < threshold:
                if 'veto_archetype_threshold' not in self._veto_metrics:
                    self._veto_metrics['veto_archetype_threshold'] = 0
                self._veto_metrics['veto_archetype_threshold'] += 1
                logger.debug(f"[THRESHOLD VETO] {archetype_name} | archetype_score={archetype_fusion:.3f} < threshold={threshold:.3f}")
                # Fall through to PHASE 2 fallback

            # Check if archetype-specific fusion score meets archetype-specific threshold
            elif archetype_fusion >= threshold:
                # ARCHETYPE-SPECIFIC FUSION GATE: Check config for per-archetype gates first
                # This allows VE to have lower gate (0.32) vs trap (0.374)
                archetype_config = self.runtime_config.get('archetypes', {})

                # Try archetype-specific gate (canonical name from classify_entry_archetype)
                archetype_gate = None
                if archetype_name in archetype_config:
                    archetype_gate = archetype_config[archetype_name].get('final_fusion_gate')

                # Fallback to regime-adapted or static gate
                if archetype_gate is None:
                    if 'adapted_params' in context and context['adapted_params'] and context['adapted_params'].get('gates'):
                        final_fusion_gate = context['adapted_params']['gates'].get('final_fusion_floor', 0.374)
                    else:
                        final_fusion_gate = 0.374  # Global fallback
                else:
                    final_fusion_gate = archetype_gate
                    logger.debug(f"[ARCHETYPE GATE] Using {archetype_name}-specific gate: {final_fusion_gate:.3f}")

                # STEP 2: Apply regime-specific gate delta
                regime = context.get('macro_regime', 'neutral')
                routing_config = archetype_config.get('routing', {})
                regime_routing = routing_config.get(regime, {})
                gate_delta = regime_routing.get('final_gate_delta', 0.0)

                if gate_delta != 0.0:
                    original_gate = final_fusion_gate
                    final_fusion_gate += gate_delta
                    logger.debug(f"[REGIME GATE] {regime}: {original_gate:.3f} + {gate_delta:.3f} = {final_fusion_gate:.3f}")

                if archetype_fusion < final_fusion_gate:
                    self._veto_metrics['veto_final_fusion_gate'] += 1
                    logger.info(f"ARCHETYPE REJECTED: {archetype_name} | fusion={archetype_fusion:.3f} < {final_fusion_gate} (final gate)")
                    return None

                # STEP 1 STABILIZATION: Archetype cooldown check
                # Use bar_idx from context (passed from main loop)
                current_bar_idx = context.get('_current_bar_idx', 0)
                # Strip "archetype_" prefix to match config keys (e.g., "archetype_trap_within_trend" → "trap_within_trend")
                config_key = archetype_name.replace('archetype_', '') if archetype_name.startswith('archetype_') else archetype_name
                archetype_specific_config = archetype_config.get(config_key, {})
                cooldown_bars = archetype_specific_config.get('cooldown_bars', 0)

                if cooldown_bars > 0 and archetype_name in self._archetype_last_entry_bar:
                    bars_since_last = current_bar_idx - self._archetype_last_entry_bar[archetype_name]
                    if bars_since_last < cooldown_bars:
                        self._veto_metrics['veto_archetype_cooldown'] += 1
                        logger.debug(f"[COOLDOWN VETO] {archetype_name} | {bars_since_last} bars < {cooldown_bars} required")
                        return None

                # STEP 1 STABILIZATION: Max trades per day check
                max_trades_per_day = archetype_config.get('max_trades_per_day', 999)
                current_day = row['timestamp'].date() if 'timestamp' in row else None

                if current_day is not None:
                    # Reset daily counter if new day
                    if self._current_day != current_day:
                        self._current_day = current_day
                        self._trades_today = 0

                    # Check if we've hit the daily limit
                    if self._trades_today >= max_trades_per_day:
                        self._veto_metrics['veto_max_trades_per_day'] += 1
                        logger.debug(f"[DAILY LIMIT] {self._trades_today} trades today >= {max_trades_per_day} limit")
                        return None

                # STEP 2: Monthly share cap check
                share_caps = archetype_config.get('monthly_share_cap', {})
                if share_caps and 'timestamp' in row:
                    current_month = (row['timestamp'].year, row['timestamp'].month)

                    # Reset monthly counter if new month
                    if self._current_month != current_month:
                        self._current_month = current_month
                        self._monthly_total_trades = 0
                        self._monthly_archetype_counts[current_month] = {}

                    # Check if this archetype has exceeded its monthly share cap
                    archetype_cap = share_caps.get(archetype_name)
                    if archetype_cap is not None and self._monthly_total_trades > 0:
                        monthly_counts = self._monthly_archetype_counts[current_month]
                        archetype_count = monthly_counts.get(archetype_name, 0)
                        current_share = archetype_count / self._monthly_total_trades

                        if current_share >= archetype_cap:
                            self._veto_metrics['veto_monthly_share_cap'] += 1
                            logger.debug(f"[MONTHLY CAP] {archetype_name} share {current_share:.1%} >= cap {archetype_cap:.1%}")
                            return None

                # Store archetype info for position sizing later
                context['entry_archetype'] = archetype_name
                context['archetype_size_mult'] = size_mult

                # Check macro filter (crisis veto applies to all archetypes)
                if context.get('macro_regime') == 'crisis':
                    self._veto_metrics['veto_crisis_regime'] += 1
                    logger.info(f"REGIME FILTER: crisis → entry blocked")
                    return None

                # REGIME FILTER: risk_off requires higher fusion threshold
                regime = context.get('macro_regime', 'neutral')
                if regime == 'risk_off':
                    risk_off_fusion_gate = 0.45  # Stricter than normal
                    if archetype_fusion < risk_off_fusion_gate:
                        self._veto_metrics['veto_risk_off_regime'] += 1
                        logger.info(f"REGIME FILTER: risk_off → fusion={archetype_fusion:.3f} < {risk_off_fusion_gate} (heightened gate)")
                        return None

                # ML META-FILTER: Veto low-quality trades
                if self.ml_model is not None:
                    try:
                        features = self._extract_ml_features(row, context)
                        if features is not None:
                            ml_proba = float(self.ml_model.predict_proba(features.reshape(1, -1))[0][1])
                            context['ml_entry_score'] = ml_proba

                            if ml_proba < self.ml_threshold:
                                self._veto_metrics['veto_ml_filter'] += 1
                                logger.info(f"ML VETO: {archetype_name} | score={ml_proba:.3f} < {self.ml_threshold:.3f}")
                                return None

                            # Optional: ML-based position sizing multiplier
                            if ml_proba >= 0.80:
                                context['ml_size_mult'] = 1.20
                            elif ml_proba >= 0.65:
                                context['ml_size_mult'] = 1.00
                            else:
                                context['ml_size_mult'] = 0.80

                            logger.info(f"ML PASS: {archetype_name} | score={ml_proba:.3f} | size_mult={context.get('ml_size_mult', 1.0):.2f}x")
                    except Exception as e:
                        logger.warning(f"ML filter error (allowing trade): {e}")
                        context['ml_size_mult'] = 1.0
                else:
                    context['ml_size_mult'] = 1.0

                logger.info(f"ARCHETYPE ENTRY: {archetype_name} | fusion={archetype_fusion:.3f} >= {final_fusion_gate} | size_mult={size_mult:.2f}x | regime={regime}")
                return (f"archetype_{archetype_name}", row['close'])
        else:
            # archetype_result is None - no archetype matched
            self._veto_metrics['archetype_no_match'] += 1

        # PHASE 2: Fallback to legacy tiered system (safety net)
        # Only trigger if fusion score is exceptionally high (> 0.45)
        if fusion_score >= 0.45:
            # Ultra-high conviction fallback
            if context.get('macro_regime') not in ['crisis']:
                logger.info(f"LEGACY TIER1 ENTRY: fusion={fusion_score:.3f} (no archetype matched)")
                context['entry_archetype'] = 'legacy_tier1'
                context['archetype_size_mult'] = 1.0
                return ("tier1_market", row['close'])

        return None  # No entry

    def _compute_adaptive_max_hold(self, context: Dict, row: pd.Series, trade: Trade) -> float:
        """
        Compute adaptive max_hold cap based on market regime and Wyckoff phase.

        Enhanced with SPY-specific findings from what-if analysis:
        - M1/M2 liquidity expansion = extend (more institutional flow)
        - Low ATR percentile = extend (stable trends)
        - High macro alignment = extend (favorable environment)
        - Trade profitability = prerequisite for extension

        Logic:
        - Extend cap in markup/markdown phases (strong trends)
        - Shorten cap in accumulation/distribution (choppy/topping)
        - Shorten cap in high volatility (VIX extreme)
        - Shorten cap near FRVP resistance zones (POC/HVN)
        - Shorten cap in crisis macro regime

        Returns:
            Adjusted max_hold in hours
        """
        base_max_hold = self.params.max_hold_bars
        multiplier = 1.0

        # Calculate current trade PnL
        current_price = row['close']
        current_pnl_pct = ((current_price - trade.entry_price) / trade.entry_price * trade.direction) * 100

        # ===== SPY ADAPTIVE LOGIC (Using Actual Wyckoff Phases) =====
        # Snapshot context for debugging
        wyckoff_phase_1d = context.get('wyckoff_phase', 'transition').lower()
        wyckoff_score_1d = row.get('tf1d_wyckoff_score', 0.5)
        current_atr = row.get('atr_14', row.get('atr_20', 2.0))
        low_volatility = current_atr < 1.5

        # Store snapshot for debugging (overwrites each time)
        self._last_ctx_snapshot = {
            'ts': str(row.name),
            'phase': wyckoff_phase_1d,
            'score': wyckoff_score_1d,
            'atr': current_atr,
            'pnl_pct': current_pnl_pct,
        }

        # Extension gate: profitable + favorable conditions
        if current_pnl_pct >= 0.1:  # Trade must be slightly profitable
            # Extension logic based on Wyckoff phase
            reason = None

            # Phase 2.3: M1/M2 Wyckoff extension logic
            # Check for M1 (spring) and M2 (markup) signals in feature store
            m1_spring_detected = row.get('wyckoff_m1_spring', False) or row.get('wyckoff_m1_present', False)
            m2_markup_detected = row.get('wyckoff_m2_markup', False) or row.get('wyckoff_m2_present', False)

            if m2_markup_detected and wyckoff_score_1d >= 0.6:
                # M2 markup: Strong trend continuation signal
                # Extend by +100% (2x multiplier) - let the markup run
                multiplier *= 2.0
                reason = "M2_markup_extension"
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"ADAPTIVE TIME EXIT: M2 markup detected, extending max_hold by +100% "
                          f"(base={base_max_hold}h → {base_max_hold*2}h)")
            elif m1_spring_detected and wyckoff_score_1d >= 0.5:
                # M1 spring: Accumulation transitioning to markup
                # Extend by +50% (1.5x multiplier) - expecting trend development
                multiplier *= 1.5
                reason = "M1_spring_extension"
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"ADAPTIVE TIME EXIT: M1 spring detected, extending max_hold by +50% "
                          f"(base={base_max_hold}h → {base_max_hold*1.5}h)")
            elif 'markup' in wyckoff_phase_1d and wyckoff_score_1d >= 0.6:
                # Strong markup phase (like Nov 2024 rally)
                multiplier *= 3.0 if low_volatility else 2.0
                reason = "strong_markup_low_vol" if low_volatility else "strong_markup"
            elif 'markup' in wyckoff_phase_1d or wyckoff_score_1d >= 0.7:
                # Moderate markup or high Wyckoff score
                multiplier *= 2.0
                reason = "moderate_markup"
            elif low_volatility and current_pnl_pct >= 1.0:
                # Not in markup but profitable + low vol
                multiplier *= 1.5
                reason = "low_vol_profitable"

            # Record extension event AND RETURN EARLY (skip original logic)
            if multiplier > 1.0 and reason:
                extended_bars = int(base_max_hold * multiplier)
                self._adaptive_extension_count += 1
                self._adaptive_events.append(AdaptiveHoldEvent(
                    trade_id=len(self.trades),
                    timestamp=str(row.name),
                    base_bars=base_max_hold,
                    extended_bars=extended_bars,
                    wyckoff_phase=wyckoff_phase_1d,
                    wyckoff_score=wyckoff_score_1d,
                    current_pnl_pct=current_pnl_pct,
                    atr_value=current_atr,
                    reason=reason
                ))

                # Return the extended hold time (skip original logic below)
                return extended_bars

        # ===== ORIGINAL LOGIC (Keep for other contexts) =====
        # 1. Wyckoff Phase adjustment
        wyckoff_phase = context.get('wyckoff_phase', '').lower()
        if 'markup' in wyckoff_phase or 'markdown' in wyckoff_phase:
            multiplier *= 1.5  # Extend 50% in strong trends
        elif 'accumulation' in wyckoff_phase or 'distribution' in wyckoff_phase:
            multiplier *= 0.7  # Shorten 30% in choppy/topping

        # 2. Macro regime adjustment
        macro_regime = context.get('macro_regime', '').lower()
        if macro_regime == 'risk_on':
            multiplier *= 1.2  # Extend 20% in favorable macro
        elif macro_regime == 'crisis':
            multiplier *= 0.5  # Shorten 50% in crisis (exit quickly)

        # 3. Volatility adjustment
        vix_level = row.get('VIX_level', 'normal')
        if vix_level == 'extreme':
            multiplier *= 0.6  # Shorten 40% in high vol
        elif vix_level == 'elevated':
            multiplier *= 0.8  # Shorten 20% in elevated vol

        # 4. FRVP positioning (near resistance)
        frvp_position = context.get('frvp_poc_position', 'neutral')
        if trade.direction > 0:  # Long position
            if 'above_hvn' in frvp_position or 'above_poc' in frvp_position:
                multiplier *= 0.75  # Shorten 25% near overhead resistance
        else:  # Short position
            if 'below_lvn' in frvp_position or 'below_poc' in frvp_position:
                multiplier *= 0.75  # Shorten 25% near support

        # Apply multiplier with floor/ceiling constraints
        adjusted = base_max_hold * multiplier
        min_hold = base_max_hold * 0.5   # Never below 50% of base
        max_hold = base_max_hold * 3.0   # Never above 300% of base (was 200%, increased for SPY)

        return max(min_hold, min(max_hold, adjusted))

    def _check_structure_invalidation(self, row: pd.Series, trade: Trade, current_bar_index: int) -> bool:
        """
        Phase 1.1: Structure Invalidation Exit Check (De-Aggressivized)

        Exit if critical support/resistance structures break WITH CONFLUENCE.

        Emergency Fixes Applied:
        1. Grace period: No exits in first 6 bars after entry
        2. Confluence: Require 2 of 3 structures broken (OB + FVG, not just OB)
        3. Tighter RSI: RSI < 30 (was 40) for genuine momentum breakdown
        4. Body-only close: Wicks don't count, only body midpoint
        5. Wick filter: Ignore liquidation wicks (wick > 3x body)

        Args:
            row: Current bar data with SMC structure levels
            trade: Active trade object
            current_bar_index: Index of current bar for grace period check

        Returns:
            True if 2+ structures invalidated (exit immediately)
        """
        # Phase 1.1 FIX #1: Grace period - no structure exits in first N hours (ML-optimizable)
        bars_held = current_bar_index - trade.entry_bar if hasattr(trade, 'entry_bar') else 999
        if bars_held < self.structure_min_hold_bars:
            return False

        # Tier 2 FIX #3: 4H MTF alignment - only exit on structure breaks if 4H confirms weakness
        wyckoff_4h = row.get('wyckoff_phase_4h', 'unknown')
        if trade.direction == 1:  # Long trade
            # Don't exit longs on structure breaks if 4H is in markup/distribution (strong uptrend)
            if wyckoff_4h in ['markup', 'm2', 'distribution']:
                return False
        else:  # Short trade
            # Don't exit shorts on structure breaks if 4H is in markdown/accumulation (strong downtrend)
            if wyckoff_4h in ['markdown', 'm1', 'accumulation']:
                return False

        # Get structure levels from feature store (1H timeframe)
        ob_low = row.get('tf1h_ob_low', None)
        ob_high = row.get('tf1h_ob_high', None)
        bb_low = row.get('tf1h_bb_low', None)
        bb_high = row.get('tf1h_bb_high', None)
        fvg_low = row.get('tf1h_fvg_low', None)
        fvg_high = row.get('tf1h_fvg_high', None)

        current_close = row['close']
        current_open = row['open']
        current_high = row['high']
        current_low = row['low']

        # Phase 1.1 FIX #5: Wick filter - ignore liquidation spikes
        body_size = abs(current_close - current_open)
        upper_wick = current_high - max(current_open, current_close)
        lower_wick = min(current_open, current_close) - current_low
        if upper_wick > 3 * body_size or lower_wick > 3 * body_size:
            return False  # Liquidation wick, not structural breakdown

        # Phase 1.1 FIX #4: Body-only close (not wick touches)
        body_midpoint = (current_open + current_close) / 2

        # Confluence counter: track how many structures broken
        structure_breaks = 0

        # Long trade checks
        if trade.direction == 1:
            # OB invalidation: Body close below OB boundary by > 0.10% with RECENT BOS
            if ob_low is not None and not pd.isna(ob_low):
                eps_ob = 0.001  # 0.10% epsilon for noise tolerance
                if body_midpoint < ob_low * (1 - eps_ob):
                    # Phase 1.1 FIX #3: Recent BOS only (would need historical check, simplified for now)
                    bos_confirmed = row.get('tf1h_bos_bearish', False)
                    if bos_confirmed:
                        structure_breaks += 1
                        logger.debug(f"Structure break 1/3 (LONG): OB broken at {body_midpoint:.2f} < {ob_low:.2f}")

            # BB penetration: Body below BB boundary by > 0.10%
            if bb_low is not None and not pd.isna(bb_low):
                eps_bb = 0.001  # 0.10%
                if body_midpoint < bb_low * (1 - eps_bb):
                    structure_breaks += 1
                    logger.debug(f"Structure break 2/3 (LONG): BB penetration at {body_midpoint:.2f}")

            # FVG melt: Body through FVG with STRONG momentum (RSI < 25, Tier 2 tightening)
            if fvg_low is not None and not pd.isna(fvg_low):
                eps_fvg = 0.001  # 0.10%
                if body_midpoint < fvg_low * (1 - eps_fvg):
                    rsi = row.get('rsi_14', 50)
                    vol_z = row.get('volume_zscore', 0)

                    # Tier 2 FIX #2: Require RSI below threshold AND above-average volume (ML-optimizable)
                    if rsi < self.structure_rsi_long_threshold and vol_z > self.structure_vol_zscore_min:
                        structure_breaks += 1
                        logger.debug(f"Structure break 3/3 (LONG): FVG melted with strong momentum (RSI={rsi:.1f}, vol_z={vol_z:.2f})")

        # Short trade checks (inverse logic)
        else:
            # OB invalidation: Body close above OB boundary by > 0.10%
            if ob_high is not None and not pd.isna(ob_high):
                eps_ob = 0.001
                if body_midpoint > ob_high * (1 + eps_ob):
                    bos_confirmed = row.get('tf1h_bos_bullish', False)
                    if bos_confirmed:
                        structure_breaks += 1
                        logger.debug(f"Structure break 1/3 (SHORT): OB broken at {body_midpoint:.2f} > {ob_high:.2f}")

            # BB penetration: Body above BB boundary by > 0.10%
            if bb_high is not None and not pd.isna(bb_high):
                eps_bb = 0.001
                if body_midpoint > bb_high * (1 + eps_bb):
                    structure_breaks += 1
                    logger.debug(f"Structure break 2/3 (SHORT): BB penetration at {body_midpoint:.2f}")

            # FVG melt: Body through FVG with STRONG momentum (RSI > 75, Tier 2 tightening)
            if fvg_high is not None and not pd.isna(fvg_high):
                eps_fvg = 0.001
                if body_midpoint > fvg_high * (1 + eps_fvg):
                    rsi = row.get('rsi_14', 50)
                    vol_z = row.get('volume_zscore', 0)

                    # Tier 2 FIX #2: Require RSI above threshold AND above-average volume (ML-optimizable)
                    if rsi > self.structure_rsi_short_threshold and vol_z > self.structure_vol_zscore_min:
                        structure_breaks += 1
                        logger.debug(f"Structure break 3/3 (SHORT): FVG melted with strong momentum (RSI={rsi:.1f}, vol_z={vol_z:.2f})")

        # Phase 1.1 FIX #2: Confluence requirement - need 2 of 3 structures broken
        if structure_breaks >= 2:
            logger.info(f"Structure invalidation ({structure_breaks}/3 structures broken) at {current_close:.2f}")
            return True

        return False

    def _check_reentry_conditions(self, row: pd.Series, fusion_score: float, context: Dict, current_bar_index: int) -> Optional[Tuple[str, float, float]]:
        """
        Phase 4: Check if conditions are met for re-entry after a recent exit.

        Returns:
            Optional tuple of (entry_type, entry_price, reentry_size_multiplier) or None
        """
        # Gate 1: Must have a recent tracked exit
        if self._last_exit_bar is None:
            return None

        # Gate 2: Must be within reentry window (asset-specific)
        asset = self.asset
        if asset == 'SPY':
            reentry_window = 3  # 3 hours for equities
        elif asset in ['BTC', 'ETH']:
            reentry_window = 7  # 7 hours for crypto
        else:
            reentry_window = 5  # Default 5 hours

        bars_since_exit = current_bar_index - self._last_exit_bar

        # Gate 2a: Minimum cooldown - wait at least 1 bar before re-entering
        if bars_since_exit < 1:
            logger.debug(f"PHASE 4 GATE 2a FAIL: Same-bar re-entry not allowed (bars_since_exit={bars_since_exit})")
            return None

        # DEBUG: Track re-entry attempts - log EVERY bar within window
        if bars_since_exit <= reentry_window:
            logger.info(f"PHASE 4 CHECK: bar {current_bar_index}, bars_since_exit={bars_since_exit}/{reentry_window}, "
                       f"fusion={fusion_score:.3f}, exit_reason={self._last_exit_reason}")

        # Gate 2b: Re-entry window expiry
        if bars_since_exit > reentry_window:
            if bars_since_exit == reentry_window + 1:
                logger.info(f"PHASE 4 GATE 2 FAIL: Re-entry window expired ({reentry_window} bars)")
            return None

        # Gate 3: Pullback to structure (OB/FVG within 0.5 ATR) - OPTIONAL if no structures available
        current_price = row['close']
        atr = row.get('atr_14', 0)
        if atr == 0:
            logger.debug(f"PHASE 4 GATE 3 FAIL: ATR is 0")
            return None

        pullback_to_structure = False
        structure_distance = None
        has_structures = False  # Track if any structures exist

        if self._last_exit_direction == 1:  # Was long, check for pullback to support
            ob_low = row.get('tf1h_ob_low', None)
            fvg_low = row.get('tf1h_fvg_low', None)

            if ob_low is not None:
                has_structures = True
                ob_dist = abs(current_price - ob_low) / atr
                structure_distance = ob_dist
                if ob_dist < 0.5:
                    pullback_to_structure = True
                    logger.info(f"PHASE 4 GATE 3 PASS: Pullback to OB support at {ob_low:.2f} (current={current_price:.2f}, dist={ob_dist:.2f} ATR)")

            if fvg_low is not None:
                has_structures = True
                if not pullback_to_structure:
                    fvg_dist = abs(current_price - fvg_low) / atr
                    if structure_distance is None or fvg_dist < structure_distance:
                        structure_distance = fvg_dist
                    if fvg_dist < 0.5:
                        pullback_to_structure = True
                        logger.info(f"PHASE 4 GATE 3 PASS: Pullback to FVG support at {fvg_low:.2f} (current={current_price:.2f}, dist={fvg_dist:.2f} ATR)")

        elif self._last_exit_direction == -1:  # Was short, check for pullback to resistance
            ob_high = row.get('tf1h_ob_high', None)
            fvg_high = row.get('tf1h_fvg_high', None)

            if ob_high is not None:
                has_structures = True
                ob_dist = abs(current_price - ob_high) / atr
                structure_distance = ob_dist
                if ob_dist < 0.5:
                    pullback_to_structure = True
                    logger.info(f"PHASE 4 GATE 3 PASS: Pullback to OB resistance at {ob_high:.2f} (current={current_price:.2f}, dist={ob_dist:.2f} ATR)")

            if fvg_high is not None:
                has_structures = True
                if not pullback_to_structure:
                    fvg_dist = abs(current_price - fvg_high) / atr
                    if structure_distance is None or fvg_dist < structure_distance:
                        structure_distance = fvg_dist
                    if fvg_dist < 0.5:
                        pullback_to_structure = True
                        logger.info(f"PHASE 4 GATE 3 PASS: Pullback to FVG resistance at {fvg_high:.2f} (current={current_price:.2f}, dist={fvg_dist:.2f} ATR)")

        # Gate 3 is OPTIONAL if no structures exist (common in feature store data)
        if not pullback_to_structure and has_structures:
            # Only fail if structures exist but price isn't near them
            if bars_since_exit <= reentry_window:
                dist_str = f"{structure_distance:.2f}" if structure_distance is not None else "N/A"
                logger.info(f"PHASE 4 GATE 3 FAIL: No pullback to structure (closest={dist_str} ATR, need < 0.5)")
            return None
        elif not has_structures:
            logger.info(f"PHASE 4 GATE 3 SKIP: No structures available, proceeding to other gates")

        # Gate 4: Signal recovery (fusion score > tier3_threshold - delta, ML-optimizable)
        tier3_threshold = self.params.tier3_threshold
        recovery_threshold = tier3_threshold - self.reentry_fusion_delta

        if fusion_score < recovery_threshold:
            logger.debug(f"PHASE 4 GATE 4 FAIL: Fusion score too low ({fusion_score:.3f} < {recovery_threshold:.3f})")
            return None

        logger.info(f"PHASE 4 GATE 4 PASS: Fusion score recovered ({fusion_score:.3f} >= {recovery_threshold:.3f})")

        # Gate 5: Confluence checks (RSI, MTF, volume)
        rsi = row.get('rsi_14', 50)
        tf4h_fusion = row.get('tf4h_fusion_score', 0)
        vol_z = row.get('volume_zscore', 0)

        confluence_score = 0
        confluence_details = []

        if self._last_exit_direction == 1:  # Long re-entry
            if rsi > 50:
                confluence_score += 1
                confluence_details.append(f"RSI={rsi:.1f}>50")
            else:
                confluence_details.append(f"RSI={rsi:.1f}≤50")

            if tf4h_fusion > 0.25:
                confluence_score += 1
                confluence_details.append(f"4H_fusion={tf4h_fusion:.3f}>0.25")
            else:
                confluence_details.append(f"4H_fusion={tf4h_fusion:.3f}≤0.25")

            if vol_z > 0.5:
                confluence_score += 1
                confluence_details.append(f"vol_z={vol_z:.2f}>0.5")
            else:
                confluence_details.append(f"vol_z={vol_z:.2f}≤0.5")
        else:  # Short re-entry
            if rsi < 50:
                confluence_score += 1
                confluence_details.append(f"RSI={rsi:.1f}<50")
            else:
                confluence_details.append(f"RSI={rsi:.1f}≥50")

            if tf4h_fusion < -0.25:
                confluence_score += 1
                confluence_details.append(f"4H_fusion={tf4h_fusion:.3f}<-0.25")
            else:
                confluence_details.append(f"4H_fusion={tf4h_fusion:.3f}≥-0.25")

            if vol_z > 0.5:
                confluence_score += 1
                confluence_details.append(f"vol_z={vol_z:.2f}>0.5")
            else:
                confluence_details.append(f"vol_z={vol_z:.2f}≤0.5")

        # Require confluence threshold (Phase 4: ML-optimizable)
        if confluence_score < self.reentry_confluence_threshold:
            logger.info(f"PHASE 4 GATE 5 FAIL: Confluence too low ({confluence_score}/3): {', '.join(confluence_details)}")
            return None

        logger.info(f"PHASE 4 GATE 5 PASS: Confluence sufficient ({confluence_score}/3): {', '.join(confluence_details)}")

        # All gates passed - approve re-entry
        logger.info(f"PHASE 4 RE-ENTRY: bars_since_exit={bars_since_exit}, fusion={fusion_score:.3f}, "
                   f"confluence={confluence_score}/3, rsi={rsi:.1f}, vol_z={vol_z:.2f}")

        # Re-entry at 75% of original size
        reentry_size_multiplier = 0.75
        entry_price = current_price

        return ('phase4_reentry', entry_price, reentry_size_multiplier)

    def check_exit_conditions(self, row: pd.Series, trade: Trade, current_bar_index: int = 0) -> Optional[Tuple[str, float]]:
        """
        Check if exit conditions are met using smart exit logic.

        Args:
            row: Current bar data
            trade: Active trade to check
            current_bar_index: Current bar index (for grace period logic)

        Returns:
            (exit_reason, exit_price) or None
        """
        import logging
        logger = logging.getLogger(__name__)

        current_price = row['close']
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price * trade.direction
        pnl_r = pnl_pct / (self.params.atr_stop_mult * trade.atr_at_entry / trade.entry_price)

        # Compute fusion score and context early (needed for Phase 3 trailing + signal neutralization)
        fusion_score, context = self.compute_advanced_fusion_score(row)

        # 1. Stop loss hit (initial stop)
        if trade.direction == 1:  # Long
            if current_price <= trade.initial_stop:
                return ("stop_loss", trade.initial_stop)
        else:  # Short
            if current_price >= trade.initial_stop:
                return ("stop_loss", trade.initial_stop)

        # 2. Partial exits (Phase 2.1: Enhanced Partial Profit Ladder)
        if self.params.use_smart_exits:
            # TP1: +1R - Take 1/3 off, move stop to BE-ε
            if pnl_r >= 1.0 and not any(p['level'] == 'TP1' for p in trade.partial_exits):
                # Reduce position size by 1/3
                trade.position_size *= (2/3)
                trade.partial_exits.append({'level': 'TP1', 'price': current_price, 'pnl_r': pnl_r})

                # Track last partial bar for pattern exit cooldown
                trade.last_partial_bar = current_bar_index

                # Move stop to breakeven minus small epsilon
                if self.params.breakeven_after_tp1:
                    eps = trade.atr_at_entry * 0.1  # Small epsilon (10% of ATR)
                    trade.initial_stop = trade.entry_price - eps * trade.direction
                    logger.info(f"PARTIAL EXIT TP1 (+{pnl_r:.2f}R): 1/3 position closed @ ${current_price:.2f}, "
                              f"stop → BE-${eps:.2f} (${trade.initial_stop:.2f})")

            # TP2: +2R - Take another 1/3 off (1/2 of remaining), tighten trailing
            if pnl_r >= 2.0 and not any(p['level'] == 'TP2' for p in trade.partial_exits):
                # Reduce remaining position by half (takes another 1/3 of original)
                trade.position_size *= 0.5
                trade.partial_exits.append({'level': 'TP2', 'price': current_price, 'pnl_r': pnl_r})

                # Track last partial bar for pattern exit cooldown
                trade.last_partial_bar = current_bar_index

                # Tighten trailing stop multiplier (ATR × k down to k - 0.5)
                # Store original if not already stored
                if not hasattr(trade, 'original_trailing_mult'):
                    trade.original_trailing_mult = self.params.trailing_atr_mult

                # Tighten by 0.5 (but don't go below 1.5)
                trade.tightened_trailing_mult = max(1.5, self.params.trailing_atr_mult - 0.5)
                logger.info(f"PARTIAL EXIT TP2 (+{pnl_r:.2f}R): 1/3 position closed @ ${current_price:.2f}, "
                          f"trail tightened {self.params.trailing_atr_mult:.1f}× → {trade.tightened_trailing_mult:.1f}×ATR")

        # 2a. Structure Invalidation Exit (Phase 1.1 - De-Aggressivized)
        # Exit if critical SMC structures (OB/BB/FVG) break with confluence
        if self._check_structure_invalidation(row, trade, current_bar_index):
            return ("structure_invalidated", current_price)

        # 3. Trailing stop (Phase 3: Dynamic with PTI regime + FVG proximity)
        if pnl_r > 1.0 and self.params.use_smart_exits:
            atr = row.get('atr_14', trade.atr_at_entry)
            adx = row.get('adx_14', 20)

            # KAMA slope detection (trend momentum)
            kama_rising = False
            if 'kama_10' in row.index:
                try:
                    # Get KAMA from current bar and previous bar
                    current_idx = self.df.index.get_loc(row.name)
                    if current_idx > 0:
                        prev_kama = self.df.iloc[current_idx - 1].get('kama_10', None)
                        curr_kama = row.get('kama_10', None)
                        if prev_kama is not None and curr_kama is not None:
                            kama_rising = curr_kama > prev_kama
                except:
                    kama_rising = False

            vix = row.get('vix', 20)

            # Phase 3: PTI regime factor
            pti_regime = context.get('pti_regime', 'neutral')
            regime_factor = 1.5 if pti_regime == 'chop' else 1.0  # Wider in chop to cut fast

            # Phase 3: FVG proximity factor (tighten near key structures)
            fvg_low = row.get('tf1h_fvg_low', None)
            fvg_high = row.get('tf1h_fvg_high', None)
            structure_factor = 1.0

            if trade.direction == 1 and fvg_low is not None and not pd.isna(fvg_low):
                fvg_proximity = abs(current_price - fvg_low) / atr if atr > 0 else 999
                if fvg_proximity < 2.0:  # Within 2 ATR of FVG support
                    structure_factor = 1.2  # Tighten by 20%
            elif trade.direction == -1 and fvg_high is not None and not pd.isna(fvg_high):
                fvg_proximity = abs(current_price - fvg_high) / atr if atr > 0 else 999
                if fvg_proximity < 2.0:  # Within 2 ATR of FVG resistance
                    structure_factor = 1.2

            # Track B: Look up archetype-specific exit parameters
            archetype_base_trail = None
            if trade.entry_archetype:
                archetype_exits = self.runtime_config.get('archetypes', {}).get('exits', {})
                archetype_params = archetype_exits.get(trade.entry_archetype, archetype_exits.get('_default', {}))
                if 'trail_atr' in archetype_params:
                    archetype_base_trail = archetype_params['trail_atr']
                    logger.info(f"Track B: Using archetype-specific trail_atr={archetype_base_trail} for {trade.entry_archetype}")

            # Determine base ATR multiplier with adaptive logic
            if hasattr(trade, 'tightened_trailing_mult'):
                # Use tightened multiplier from TP2 (Phase 2.1)
                base_mult = trade.tightened_trailing_mult
                reason = "TP2-tightened"
            elif adx > 25 and kama_rising:
                # Strong uptrend: Loosen (let it run)
                base_mult = archetype_base_trail or self.params.trailing_atr_mult
                regime_factor = min(regime_factor, 0.8)  # Override: looser in trends
                reason = "strong-trend"
            elif adx < 20 or vix > 25:
                # Weak trend or high VIX: Tighten
                base_mult = max(1.5, (archetype_base_trail or self.params.trailing_atr_mult) - 0.5)
                regime_factor = max(regime_factor, 1.5)  # Override: tighter in chop/vol
                reason = "weak-trend-or-high-vix"
            else:
                # Normal conditions: Use standard trailing
                base_mult = archetype_base_trail or self.params.trailing_atr_mult
                reason = "standard"

            # Apply Phase 3 regime and structure factors
            atr_mult = base_mult * regime_factor * structure_factor

            # Calculate trailing stop from peak
            trailing_stop = trade.entry_price + (trade.peak_profit - atr_mult * atr) * trade.direction

            # Check if trailing stop hit
            if trade.direction == 1:  # Long
                if current_price <= trailing_stop:
                    logger.info(f"TRAILING STOP HIT (long): price ${current_price:.2f} <= stop ${trailing_stop:.2f} "
                              f"(+{pnl_r:.2f}R, {reason}, {atr_mult:.1f}×ATR, regime={pti_regime}, "
                              f"ADX={adx:.1f}, VIX={vix:.1f}, factors={regime_factor:.1f}×{structure_factor:.1f})")
                    return ("trailing_stop", current_price)
            else:  # Short
                if current_price >= trailing_stop:
                    logger.info(f"TRAILING STOP HIT (short): price ${current_price:.2f} >= stop ${trailing_stop:.2f} "
                              f"(+{pnl_r:.2f}R, {reason}, {atr_mult:.1f}×ATR, regime={pti_regime}, "
                              f"ADX={adx:.1f}, VIX={vix:.1f}, factors={regime_factor:.1f}×{structure_factor:.1f})")
                    return ("trailing_stop", current_price)

        # 4. Fusion score drops (Phase 2.4: Enhanced signal neutralization with regime flip)
        # Note: fusion_score and context already computed at start of method for Phase 3 trailing

        # Check for regime flip (Wyckoff phase transition)
        wyckoff_phase = context.get('wyckoff_phase', '').lower()
        regime_flip_detected = False

        # Detect adverse regime transitions
        if trade.direction == 1:  # Long position
            # Long exits on: markup → distribution (topping), or markup → markdown (reversal)
            if 'distribution' in wyckoff_phase or 'markdown' in wyckoff_phase:
                # Store entry phase on trade if not already stored
                if not hasattr(trade, 'entry_wyckoff_phase'):
                    trade.entry_wyckoff_phase = 'unknown'

                # If we entered during markup/accumulation and now in distribution/markdown
                if trade.entry_wyckoff_phase in ['unknown', 'markup', 'accumulation', 'transition']:
                    regime_flip_detected = True
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"REGIME FLIP DETECTED (long): {trade.entry_wyckoff_phase} → {wyckoff_phase}, "
                              f"fusion={fusion_score:.2f}")

        else:  # Short position
            # Short exits on: markdown → accumulation (bottoming), or markdown → markup (reversal)
            if 'accumulation' in wyckoff_phase or 'markup' in wyckoff_phase:
                if not hasattr(trade, 'entry_wyckoff_phase'):
                    trade.entry_wyckoff_phase = 'unknown'

                if trade.entry_wyckoff_phase in ['unknown', 'markdown', 'distribution', 'transition']:
                    regime_flip_detected = True
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"REGIME FLIP DETECTED (short): {trade.entry_wyckoff_phase} → {wyckoff_phase}, "
                              f"fusion={fusion_score:.2f}")

        # Exit if fusion score drops OR regime flip detected
        if fusion_score < self.params.tier3_threshold:
            if regime_flip_detected:
                return ("signal_neutralized_regime_flip", current_price)
            else:
                return ("signal_neutralized", current_price)
        elif regime_flip_detected:
            # Regime flip alone (fusion still OK but phase changed adversely)
            return ("regime_flip", current_price)

        # 5. Drawdown guard (Phase 2.5: 60% retrace from max runup)
        # Protect profits when PNL retraces >60% from peak after achieving +1R
        if hasattr(trade, 'peak_profit') and trade.peak_profit > 0:
            stop_distance = trade.atr_at_entry * self.params.atr_stop_mult
            peak_pnl_r = trade.peak_profit / stop_distance  # Peak in R-multiple

            if peak_pnl_r >= 1.0:  # Only apply if peak was at least +1R
                # Calculate current profit from entry
                current_pnl = (current_price - trade.entry_price) * trade.direction * trade.position_size
                current_pnl_r = current_pnl / (stop_distance * trade.position_size) if stop_distance > 0 else 0

                # Calculate retrace percentage from peak
                retrace_pct = (peak_pnl_r - current_pnl_r) / peak_pnl_r if peak_pnl_r > 0 else 0

                if retrace_pct > 0.6:  # >60% retrace from peak
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"DRAWDOWN GUARD: Retraced {retrace_pct*100:.1f}% from peak +{peak_pnl_r:.2f}R "
                              f"to +{current_pnl_r:.2f}R, exiting @ ${current_price:.2f}")
                    return ("drawdown_guard", current_price)

        # 6. Pattern exits (Phase 2.6 HARDENED: 2-leg pullback, inside-bar expansion)
        # GUARDS: min bars, ATR magnitude, HTF trend, confluence, profit-state awareness
        try:
            current_idx = self.df.index.get_loc(row.name)

            # GUARD 1: Minimum bars in trade (debounce - no pattern exits in first 8 hours)
            bars_held = current_bar_index - getattr(trade, 'entry_bar', 0)
            if bars_held < 8:
                pass  # Skip pattern checks for first 8 bars

            # GUARD 2: Post-partial cooldown (after TP1/TP2, ignore patterns for 6 bars)
            elif len(trade.partial_exits) > 0:
                last_partial_bar = getattr(trade, 'last_partial_bar', -999)
                if current_bar_index - last_partial_bar < 6:
                    pass  # Cooldown after partial exit

            elif current_idx >= 3:
                # Get ATR for magnitude checks
                atr = row.get('atr_14', trade.atr_at_entry)

                # Get last 4 bars (including current)
                bars = self.df.iloc[current_idx-3:current_idx+1]

                # Initialize confluence score (need 2 of 3 to exit)
                confluence_score = 0
                pattern_detected = False
                pattern_kind = None

                if trade.direction == 1:  # Long position
                    # === 2-leg pullback with ATR magnitude gate ===
                    if len(bars) >= 4:
                        lows = bars['low'].values
                        highs = bars['high'].values

                        # GUARD 3: ATR-scaled magnitude (leg must be >= 0.7 ATR)
                        leg1_size = highs[-3] - lows[-2]
                        leg2_size = highs[-2] - lows[-1]
                        total_pullback = highs[-3] - lows[-1]

                        # Check for descending lows with sufficient magnitude
                        pattern_detected_2leg = (
                            lows[-1] < lows[-2] < lows[-3] and
                            highs[-2] < highs[-3] and
                            total_pullback >= 0.7 * atr  # Significant pullback
                        )

                        if pattern_detected_2leg:
                            pattern_detected = True
                            pattern_kind = "2leg_pullback"
                            confluence_score += 1  # (A) Pattern detected

                    # === Inside-bar expansion with magnitude gate ===
                    if len(bars) >= 3:
                        prev2_high, prev2_low = bars.iloc[-3]['high'], bars.iloc[-3]['low']
                        prev1_high, prev1_low = bars.iloc[-2]['high'], bars.iloc[-2]['low']
                        curr_high, curr_low = bars.iloc[-1]['high'], bars.iloc[-1]['low']

                        # Use body midpoint (not wicks)
                        curr_body_mid = (bars.iloc[-1]['open'] + bars.iloc[-1]['close']) / 2

                        inside_bar = (prev1_high <= prev2_high and prev1_low >= prev2_low)

                        # GUARD 3: Break must be >= 0.3 ATR from inside bar range
                        break_magnitude = prev1_low - curr_low

                        pattern_detected_ib = (
                            inside_bar and
                            curr_body_mid < prev1_low and
                            break_magnitude >= 0.3 * atr
                        )

                        if pattern_detected_ib and not pattern_detected:
                            pattern_detected = True
                            pattern_kind = "inside_bar_expansion"
                            confluence_score += 1  # (A) Pattern detected

                    # GUARD 4: HTF trend block (if 1D Wyckoff = markup and ADX >= 20, require extra confirmation)
                    wyckoff_phase_1d = context.get('wyckoff_phase', '').lower()
                    adx = row.get('adx_14', 15)
                    strong_trend = ('markup' in wyckoff_phase_1d or 'm2' in wyckoff_phase_1d) and adx >= 20

                    if pattern_detected:
                        # (B) Momentum confirmation
                        rsi = row.get('rsi_14', 50)
                        if rsi < 45 or adx < row.get('adx_14_prev', adx):  # RSI weak or ADX declining
                            confluence_score += 1

                        # (C) Fusion score drop
                        fusion_score, _ = self.compute_advanced_fusion_score(row)
                        if fusion_score < self.params.tier3_threshold:
                            confluence_score += 1

                        # Tier 3: Exit if confluence meets threshold (ML-optimizable)
                        if confluence_score >= self.pattern_confluence_threshold:
                            logger.info(f"PATTERN EXIT: {pattern_kind}, confluence={confluence_score}/3, pnl_r={pnl_r:.2f}, "
                                      f"rsi={rsi:.1f}, adx={adx:.1f}")
                            return (f"pattern_exit_{pattern_kind}", current_price)

                else:  # Short position (inverse logic)
                    # === 2-leg rally with ATR magnitude gate ===
                    if len(bars) >= 4:
                        highs = bars['high'].values
                        lows = bars['low'].values

                        leg1_size = lows[-3] - highs[-2]
                        leg2_size = lows[-2] - highs[-1]
                        total_rally = highs[-1] - lows[-3]

                        pattern_detected_2leg = (
                            highs[-1] > highs[-2] > highs[-3] and
                            lows[-2] > lows[-3] and
                            total_rally >= 0.7 * atr
                        )

                        if pattern_detected_2leg:
                            pattern_detected = True
                            pattern_kind = "2leg_pullback"
                            confluence_score += 1

                    # === Inside-bar expansion (bullish) ===
                    if len(bars) >= 3:
                        prev2_high, prev2_low = bars.iloc[-3]['high'], bars.iloc[-3]['low']
                        prev1_high, prev1_low = bars.iloc[-2]['high'], bars.iloc[-2]['low']
                        curr_high, curr_low = bars.iloc[-1]['high'], bars.iloc[-1]['low']

                        curr_body_mid = (bars.iloc[-1]['open'] + bars.iloc[-1]['close']) / 2
                        inside_bar = (prev1_high <= prev2_high and prev1_low >= prev2_low)
                        break_magnitude = curr_high - prev1_high

                        pattern_detected_ib = (
                            inside_bar and
                            curr_body_mid > prev1_high and
                            break_magnitude >= 0.3 * atr
                        )

                        if pattern_detected_ib and not pattern_detected:
                            pattern_detected = True
                            pattern_kind = "inside_bar_expansion"
                            confluence_score += 1

                    # HTF trend block for shorts
                    wyckoff_phase_1d = context.get('wyckoff_phase', '').lower()
                    adx = row.get('adx_14', 15)
                    strong_trend = ('markdown' in wyckoff_phase_1d or 'm4' in wyckoff_phase_1d) and adx >= 20

                    if pattern_detected:
                        # Momentum confirmation (inverse)
                        rsi = row.get('rsi_14', 50)
                        if rsi > 55 or adx < row.get('adx_14_prev', adx):
                            confluence_score += 1

                        # Fusion score drop
                        fusion_score, _ = self.compute_advanced_fusion_score(row)
                        if fusion_score < self.params.tier3_threshold:
                            confluence_score += 1

                        # Tier 3: Exit if confluence meets threshold (ML-optimizable)
                        if confluence_score >= self.pattern_confluence_threshold:
                            logger.info(f"PATTERN EXIT: {pattern_kind}, confluence={confluence_score}/3, pnl_r={pnl_r:.2f}, "
                                      f"rsi={rsi:.1f}, adx={adx:.1f}")
                            return (f"pattern_exit_{pattern_kind}", current_price)

        except Exception as e:
            # Pattern detection failed, continue to other exit checks
            pass

        # 7. PTI reversal detected
        if context.get('pti_score', 0.0) > 0.6:
            return ("pti_reversal", current_price)

        # 8. Macro regime flip
        if context.get('macro_regime') == 'crisis':
            return ("macro_crisis", current_price)

        # 7. Max holding period (adaptive or fixed)
        bars_held = (row.name - trade.entry_time).total_seconds() / 3600  # Hours

        # Track B: Look up archetype-specific max_bars
        archetype_max_bars = None
        if trade.entry_archetype:
            archetype_exits = self.runtime_config.get('archetypes', {}).get('exits', {})
            archetype_params = archetype_exits.get(trade.entry_archetype, archetype_exits.get('_default', {}))
            if 'max_bars' in archetype_params:
                archetype_max_bars = archetype_params['max_bars']
                logger.info(f"Track B: Using archetype-specific max_bars={archetype_max_bars} for {trade.entry_archetype}")

        if self.params.adaptive_max_hold:
            # Adaptive max_hold based on market context
            max_hold_adjusted = self._compute_adaptive_max_hold(context, row, trade)
        else:
            # Fixed max_hold (use archetype-specific if available)
            max_hold_adjusted = archetype_max_bars or self.params.max_hold_bars

        if bars_held >= max_hold_adjusted:
            return ("max_hold", current_price)

        # 8. MTF conflict
        mtf_conflict = row.get('mtf_conflict_score', 0.0)
        if mtf_conflict > 0.7:
            return ("mtf_conflict", current_price)

        return None

    def run(self) -> Dict:
        """
        Run the full knowledge-aware backtest.

        Returns:
            Results dict with trades, metrics, and feature importance.
        """
        logger.info(f"Starting knowledge-aware backtest on {len(self.df)} bars...")

        for bar_idx, (idx, row) in enumerate(self.df.iterrows()):
            # Skip early bars without indicators
            if pd.isna(row.get('atr_14')):
                continue

            # PR#6B: Adaptive Fusion - Classify regime and get adapted parameters
            adapted_params = None
            if self.adaptive_fusion and self.regime_classifier:
                try:
                    # Extract macro features for regime classification
                    macro_row = {feat: row.get(feat, np.nan)
                                for feat in self.runtime_config['regime_classifier']['feature_order']}
                    regime_info = self.regime_classifier.classify(macro_row, timestamp=row.name)
                    adapted_params = self.adaptive_fusion.update(regime_info)

                    # Override ML threshold if adaptive
                    if adapted_params.get('ml_threshold') is not None:
                        self.ml_threshold = adapted_params['ml_threshold']
                except Exception as e:
                    if bar_idx % 1000 == 0:  # Log errors only every 1000 bars to avoid spam
                        logger.warning(f"Adaptive fusion failed at bar {bar_idx}: {e}")
                    adapted_params = None

            # Compute fusion score (with adapted parameters if available)
            fusion_score, context = self.compute_advanced_fusion_score(row, adapted_params)

            # Store adapted_params in context for archetype detection
            context['adapted_params'] = adapted_params

            # PR#4: Compute runtime liquidity score (if enabled)
            if self.liquidity_enabled:
                side = "long"  # TODO: Support short detection based on signal
                liquidity_score = compute_liquidity_score(context, side)
                context['liquidity_score'] = liquidity_score
                self.liquidity_scores.append(liquidity_score)

                # Log telemetry every 500 bars
                if len(self.liquidity_scores) % 500 == 0:
                    telemetry = compute_liquidity_telemetry(self.liquidity_scores)
                    logger.info(f"PR#4 Liquidity Telemetry (n={len(self.liquidity_scores)}): "
                              f"median={telemetry['p50']:.3f}, p75={telemetry['p75']:.3f}, "
                              f"p90={telemetry['p90']:.3f}, nonzero={telemetry['nonzero_pct']:.1f}%")
            else:
                context['liquidity_score'] = 0.0

            # Check for open position
            if self.current_position is not None:
                # Update peak profit and MAE
                current_price = row['close']
                pnl_pct = (current_price - self.current_position.entry_price) / self.current_position.entry_price * self.current_position.direction

                if pnl_pct > self.current_position.peak_profit:
                    self.current_position.peak_profit = pnl_pct

                if pnl_pct < -self.current_position.max_adverse_excursion:
                    self.current_position.max_adverse_excursion = -pnl_pct

                # Check exit conditions (passing bar_idx for Phase 1.1 grace period)
                exit_result = self.check_exit_conditions(row, self.current_position, bar_idx)

                if exit_result:
                    exit_reason, exit_price = exit_result
                    self._close_trade(row, exit_price, exit_reason, bar_idx)

            # Check for new entry (only if no position)
            if self.current_position is None:
                # Phase 4: Check re-entry conditions first (higher priority than new entries)
                reentry_result = self._check_reentry_conditions(row, fusion_score, context, bar_idx)

                if reentry_result:
                    entry_type, entry_price, reentry_size_mult = reentry_result
                    # SCORE PROPAGATION: Use archetype-specific score from context if available
                    archetype_fusion_score = context.get('fusion_score', fusion_score)
                    self._open_trade(row, entry_price, entry_type, archetype_fusion_score, context, bar_idx, reentry_size_mult=reentry_size_mult)
                    self._reentry_count += 1
                else:
                    # Check regular entry conditions
                    # STEP 1 STABILIZATION: Pass bar_idx for cooldown tracking
                    context['_current_bar_idx'] = bar_idx
                    entry_result = self.check_entry_conditions(row, fusion_score, context)

                    if entry_result:
                        entry_type, entry_price = entry_result
                        # SCORE PROPAGATION: Use archetype-specific score from context (updated by classify_entry_archetype)
                        archetype_fusion_score = context.get('fusion_score', fusion_score)
                        self._open_trade(row, entry_price, entry_type, archetype_fusion_score, context, bar_idx)

        # Close any remaining position at end
        if self.current_position is not None:
            last_row = self.df.iloc[-1]
            self._close_trade(last_row, last_row['close'], "end_of_period")

        # DIAGNOSTIC: Print veto metrics summary
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC: ENTRY PIPELINE METRICS")
        logger.info("=" * 60)
        for key, value in self._veto_metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        # Calculate metrics
        return self._calculate_metrics()

    def _open_trade(self, row: pd.Series, entry_price: float, entry_type: str, fusion_score: float, context: Dict, current_bar_index: int, reentry_size_mult: float = 1.0):
        """Open a new trade."""
        position_size = self.calculate_position_size(row, fusion_score, context)

        # Phase 4: Apply re-entry position sizing if applicable
        if reentry_size_mult != 1.0:
            position_size *= reentry_size_mult
            logger.info(f"Phase 4 re-entry: position_size adjusted to {reentry_size_mult:.0%} = ${position_size:.2f}")

        atr = row.get('atr_14', entry_price * 0.02)

        # Calculate initial stop
        stop_distance = atr * self.params.atr_stop_mult
        initial_stop = entry_price - stop_distance  # For long (flip for short)

        trade = Trade(
            entry_time=row.name,
            entry_price=entry_price,
            position_size=position_size,
            direction=1,  # Only long for now
            entry_fusion_score=fusion_score,
            entry_reason=entry_type,
            wyckoff_phase=context.get('wyckoff_phase', 'unknown'),
            wyckoff_m1_signal=context.get('m1_signal'),
            wyckoff_m2_signal=context.get('m2_signal'),
            macro_regime=context.get('macro_regime', 'neutral'),
            pti_score_1d=context.get('pti_1d', 0.0),
            pti_score_1h=context.get('pti_1h', 0.0),
            frvp_poc_position=context.get('frvp_poc_position', 'middle'),
            entry_archetype=context.get('entry_archetype'),  # Track B: store triggering archetype
            atr_at_entry=atr,
            initial_stop=initial_stop,
            entry_context=context.copy()  # ML Export: Store full context for feature extraction
        )

        self.current_position = trade

        # Phase 2.4: Store entry Wyckoff phase for regime flip detection
        trade.entry_wyckoff_phase = context.get('wyckoff_phase', 'unknown').lower()

        # Phase 2.6: Track entry bar for pattern exit debounce
        trade.entry_bar = current_bar_index

        # Phase 4: Set tighter trailing stop for re-entries (1.5× vs normal 2.0×)
        if entry_type == 'phase4_reentry':
            trade.tightened_trailing_mult = 1.5
            trade.original_trailing_mult = self.params.trailing_atr_mult
            logger.info(f"Phase 4 re-entry: trailing stop tightened to 1.5× ATR (vs normal {self.params.trailing_atr_mult:.1f}×)")

        logger.info(f"ENTRY {entry_type}: {row.name} @ ${entry_price:.2f}, size=${position_size:.2f}, fusion={fusion_score:.3f}")

        # STEP 1 STABILIZATION: Update tracking for cooldown and daily limits
        archetype_name = context.get('entry_archetype')
        if archetype_name:
            self._archetype_last_entry_bar[archetype_name] = current_bar_index
        self._trades_today += 1

        # STEP 2: Update monthly archetype tracking
        if self._current_month and archetype_name:
            if self._current_month not in self._monthly_archetype_counts:
                self._monthly_archetype_counts[self._current_month] = {}
            monthly_counts = self._monthly_archetype_counts[self._current_month]
            monthly_counts[archetype_name] = monthly_counts.get(archetype_name, 0) + 1
            self._monthly_total_trades += 1

    def _close_trade(self, row: pd.Series, exit_price: float, exit_reason: str, bar_idx: Optional[int] = None):
        """Close the current trade."""
        trade = self.current_position
        trade.exit_time = row.name
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        # Phase 4: Track this exit for potential re-entry
        # Only track "smart" exits (not hard stops like stop_loss, max_hold)
        reentry_eligible_exits = [
            'signal_neutralized',
            'pattern_exit_2leg_pullback',
            'pattern_exit_inside_bar_expansion',
            'structure_invalidated'
        ]
        if exit_reason in reentry_eligible_exits:
            self._last_exit_bar = bar_idx
            self._last_exit_price = exit_price
            self._last_exit_reason = exit_reason
            self._last_exit_direction = trade.direction
            self._last_exit_size = trade.position_size
            logger.info(f"PHASE 4 EXIT TRACKED: {exit_reason} at bar {bar_idx}, price=${exit_price:.2f}, direction={'LONG' if trade.direction == 1 else 'SHORT'}")

        # Calculate PNL
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * trade.direction
        trade.gross_pnl = trade.position_size * pnl_pct

        # Apply costs
        trade.fees = trade.position_size * (self.params.slippage_bps + self.params.fee_bps) / 10000.0
        trade.net_pnl = trade.gross_pnl - trade.fees

        # Update equity
        self.equity += trade.net_pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.trades.append(trade)
        self.current_position = None

        logger.info(f"EXIT {exit_reason}: {row.name} @ ${exit_price:.2f}, PNL=${trade.net_pnl:.2f}, equity=${self.equity:.2f}")

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_pnl': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'final_equity': self.equity,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'trades': []
            }

        total_pnl = sum(t.net_pnl for t in self.trades)
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        losing_trades = [t for t in self.trades if t.net_pnl < 0]

        gross_profit = sum(t.net_pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.net_pnl for t in losing_trades)) if losing_trades else 1.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0

        # Sharpe (simplified)
        trade_returns = [t.net_pnl / self.starting_capital for t in self.trades]
        sharpe = np.mean(trade_returns) / np.std(trade_returns) if len(trade_returns) > 1 else 0.0
        sharpe = sharpe * np.sqrt(252 / len(self.trades)) if len(self.trades) > 0 else 0.0

        # Max drawdown
        max_dd = (self.peak_equity - min(self.equity, self.peak_equity)) / self.peak_equity

        # ===== ADAPTIVE MAX-HOLD INSTRUMENTATION =====
        if self.params.adaptive_max_hold:
            # Contract test: adaptive must trigger at least once
            if self._adaptive_extension_count == 0:
                print(f"\n❌ ADAPTIVE MAX-HOLD FAILED TO TRIGGER")
                print(f"Last context snapshot: {self._last_ctx_snapshot}")
                raise RuntimeError(
                    f"Adaptive max-hold enabled but NEVER extended a position.\n"
                    f"Expected extensions in markup phases but got ZERO.\n"
                    f"Last seen context: {self._last_ctx_snapshot}\n"
                    f"This indicates the extension logic is not being applied."
                )

            # Dump events to JSON for analysis
            events_file = Path("reports/adaptive_max_hold_events.json")
            events_file.parent.mkdir(parents=True, exist_ok=True)
            with open(events_file, 'w') as f:
                json.dump([asdict(e) for e in self._adaptive_events], f, indent=2)

            logger.info(f"✅ Adaptive max-hold triggered {self._adaptive_extension_count} times")
            logger.info(f"   Events logged to: {events_file}")

        return {
            'total_pnl': total_pnl,
            'total_trades': len(self.trades),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'final_equity': self.equity,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0.0,
            'avg_loss': np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0.0,
            'trades': self.trades
        }

    def _nearest_row(self, ts: pd.Timestamp):
        """Find nearest row in dataframe using asof/searchsorted logic"""
        # Normalize timezone
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)

        # Use searchsorted for fast lookup
        idx = self.df.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return None
        return self.df.iloc[idx]

    def export_trades_with_ml_features(self, output_path: str):
        """
        Export trades with full ML feature set for XGBoost training.

        Extracts 50+ features per trade including:
        - Archetype one-hot encoding
        - Entry fusion/liquidity scores
        - Market state (regime, VIX, volatility)
        - MTF alignment indicators
        - Microstructure features
        - Recent performance metrics
        - Timing features
        - Trade outcome (r_multiple, win/loss)
        """
        print("[ML] Feature Export starting...", flush=True)

        # Early sanity checks
        if not hasattr(self, "trades"):
            print("[ML] ERROR: No trades attribute on backtester", flush=True)
            return

        if not self.trades:
            print("[ML] No trades to export (0 trades)", flush=True)
            return

        if not isinstance(self.df.index, pd.DatetimeIndex):
            print(f"[ML] ERROR: df.index must be DatetimeIndex, got {type(self.df.index)}", flush=True)
            return

        print(f"[ML] Exporting {len(self.trades)} trades...", flush=True)

        # Normalize df timezone to naive UTC
        if self.df.index.tz is not None:
            self.df = self.df.tz_convert("UTC").tz_localize(None)

        records = []
        errors = 0

        for trade_idx, trade in enumerate(self.trades):
            try:
                # Get entry row using safe nearest lookup
                entry_row = self._nearest_row(trade.entry_time)
                if entry_row is None:
                    errors += 1
                    print(f"[ML] WARN: no row found for entry_time={trade.entry_time}", flush=True)
                    continue

                ctx = trade.entry_context or {}

                # Calculate R-multiple (trade outcome)
                r_multiple = trade.net_pnl / (trade.position_size * 0.02) if trade.position_size > 0 else 0.0

                # Calculate archetype one-hot encoding
                archetype_map = {
                    'trap': 'archetype_trap',
                    'order_block_retest': 'archetype_retest',
                    'fvg_continuation': 'archetype_continuation',
                    'failed_continuation': 'archetype_failed_continuation',
                    'compression_coil': 'archetype_compression',
                    'exhaustion_spike': 'archetype_exhaustion',
                    'reaccumulation': 'archetype_reaccumulation',
                    'trap_within_trend': 'archetype_trap_within_trend',
                    'wick_trap': 'archetype_wick_trap',
                    'volume_exhaustion': 'archetype_volume_exhaustion',
                    'ratio_coil_break': 'archetype_ratio_coil_break',
                    'false_break_reversal': 'archetype_false_break_reversal'
                }

                record = {
                    # Trade identification
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'r_multiple': r_multiple,
                    'trade_won': 1 if r_multiple > 0 else 0,

                    # Archetype one-hot (12 features)
                    **{name: 1 if trade.entry_archetype == key else 0
                       for key, name in archetype_map.items()},

                    # Fusion & Liquidity (2 features)
                    'entry_fusion_score': trade.entry_fusion_score,
                    'entry_liquidity_score': ctx.get('liquidity_score', 0.0),

                    # Market State (15 features)
                    'macro_regime_risk_on': 1 if trade.macro_regime == 'risk_on' else 0,
                    'macro_regime_neutral': 1 if trade.macro_regime == 'neutral' else 0,
                    'macro_regime_risk_off': 1 if trade.macro_regime == 'risk_off' else 0,
                    'macro_regime_crisis': 1 if trade.macro_regime == 'crisis' else 0,
                    'vix_z_score': ctx.get('vix_z_score', 0.0),
                    'btc_volatility_percentile': ctx.get('btc_volatility_percentile', 0.5),
                    'volume_zscore': float(entry_row.get('volume_zscore', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'atr_percentile': ctx.get('atr_percentile', 0.5),
                    'adx_14': float(entry_row.get('adx', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'rsi_14': float(entry_row.get('rsi', 50.0)) if isinstance(entry_row, pd.Series) else 50.0,
                    'macd_histogram': float(entry_row.get('macd_histogram', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,

                    # MTF Alignment (12 features)
                    'tf1h_fusion': float(entry_row.get('tf1h_fusion_score', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'tf4h_fusion': float(entry_row.get('tf4h_fusion_score', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'tf1d_fusion': float(entry_row.get('tf1d_fusion_score', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'tf4h_trend_aligned': ctx.get('tf4h_trend_aligned', 0),
                    'tf1d_trend_aligned': ctx.get('tf1d_trend_aligned', 0),
                    'nested_structure_quality': ctx.get('nested_structure_quality', 0.0),

                    # Microstructure (10 features)
                    'boms_strength': float(entry_row.get('boms_strength', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'fvg_quality': float(entry_row.get('fvg_quality', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'wyckoff_phase_score': ctx.get('wyckoff_phase_score', 0.0),
                    'poc_distance': float(entry_row.get('poc_distance', 0.0)) if isinstance(entry_row, pd.Series) else 0.0,
                    'lvn_trap_risk': ctx.get('lvn_trap_risk', 0.0),
                    'liquidity_sweep_strength': ctx.get('liquidity_sweep_strength', 0.0),

                    # Recent Performance (8 features) - Calculate from previous trades
                    'last_3_trades_wr': self._calculate_recent_wr(trade_idx, window=3),
                    'bars_since_last_trade': self._calculate_bars_since_last(trade_idx),
                    'recent_dd_pct': self._calculate_recent_dd(trade_idx),
                    'streak_length': self._calculate_streak_length(trade_idx),

                    # Timing (4 features)
                    'hour_of_day': trade.entry_time.hour,
                    'day_of_week': trade.entry_time.dayofweek,
                    'days_into_quarter': (trade.entry_time.dayofyear - 1) % 90,
                }

                records.append(record)

            except Exception as e:
                errors += 1
                print(f"[ML] ERROR exporting trade #{trade_idx}: {e}", flush=True)
                traceback.print_exc(file=sys.stdout)

        # Check if we have any records to export
        if not records:
            print(f"[ML] No rows exported (errors={errors}). Aborting write.", flush=True)
            return

        # Create DataFrame and export atomically
        df_export = pd.DataFrame(records)
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename (atomic operation)
        tmp_path = out_path.with_suffix(".tmp.csv")
        df_export.to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)

        print(f"[ML] Exported {len(records)} rows → {out_path}", flush=True)
        print(f"[ML] Features: {len(df_export.columns)} columns", flush=True)
        print(f"[ML] Win rate: {df_export['trade_won'].mean()*100:.1f}%", flush=True)

    def _calculate_recent_wr(self, current_idx: int, window: int = 3) -> float:
        """Calculate win rate of last N trades before current trade"""
        if current_idx == 0:
            return 0.0

        start_idx = max(0, current_idx - window)
        recent_trades = self.trades[start_idx:current_idx]

        if not recent_trades:
            return 0.0

        wins = sum(1 for t in recent_trades if t.net_pnl > 0)
        return wins / len(recent_trades)

    def _calculate_bars_since_last(self, current_idx: int) -> int:
        """Calculate bars since previous trade entry"""
        if current_idx == 0:
            return 999  # Large number for first trade

        prev_trade = self.trades[current_idx - 1]
        current_trade = self.trades[current_idx]

        # Calculate time difference in hours (assuming 1H bars)
        time_diff = (current_trade.entry_time - prev_trade.entry_time).total_seconds() / 3600
        return int(time_diff)

    def _calculate_recent_dd(self, current_idx: int) -> float:
        """Calculate recent drawdown percentage before current trade"""
        if current_idx == 0:
            return 0.0

        # Calculate equity before this trade
        equity_before = self.starting_capital + sum(t.net_pnl for t in self.trades[:current_idx])

        # Find peak equity in last 10 trades
        lookback = min(10, current_idx)
        equity_curve = [self.starting_capital]
        for i in range(current_idx - lookback, current_idx):
            equity_curve.append(equity_curve[-1] + self.trades[i].net_pnl)

        peak = max(equity_curve)
        dd_pct = (peak - equity_before) / peak if peak > 0 else 0.0

        return dd_pct

    def _calculate_streak_length(self, current_idx: int) -> int:
        """Calculate current win/loss streak length (positive=wins, negative=losses)"""
        if current_idx == 0:
            return 0

        # Look backward from previous trade
        streak = 0
        last_outcome = self.trades[current_idx - 1].net_pnl > 0

        for i in range(current_idx - 1, -1, -1):
            if (self.trades[i].net_pnl > 0) == last_outcome:
                streak += 1
            else:
                break

        # Make negative if losing streak
        return streak if last_outcome else -streak


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Knowledge-aware backtest using full 69-feature engine')
    parser.add_argument('--asset', required=True, help='Asset (BTC, ETH, SPY)')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-12-31', help='End date')
    parser.add_argument('--config', help='JSON config file with KnowledgeParams')
    parser.add_argument('--export-trades', help='Export trades with ML features to CSV file')

    args = parser.parse_args()

    # Load feature store
    feature_dir = Path('data/features_mtf')
    pattern = f"{args.asset}_1H_*.parquet"
    files = list(feature_dir.glob(pattern))

    if not files:
        print(f"ERROR: No feature store found for {args.asset}")
        sys.exit(1)

    # Select feature store that covers the requested date range
    start_ts = pd.Timestamp(args.start, tz='UTC')
    end_ts = pd.Timestamp(args.end, tz='UTC')

    feature_path = None
    for fpath in sorted(files):
        # Parse date range from filename (e.g., BTC_1H_2022-01-01_to_2023-12-31.parquet)
        fname = fpath.stem  # Remove .parquet
        parts = fname.split('_')
        if len(parts) >= 5 and parts[-2] == 'to':
            try:
                store_start = pd.Timestamp(parts[-3], tz='UTC')
                store_end = pd.Timestamp(parts[-1], tz='UTC')

                # Check if this feature store covers our requested range
                if store_start <= start_ts and store_end >= end_ts:
                    feature_path = fpath
                    break
            except:
                continue

    # Fallback to most recent if no exact match
    if feature_path is None:
        feature_path = sorted(files)[-1]
        print(f"⚠️  No feature store covering {args.start} to {args.end}, using: {feature_path}")
    else:
        print(f"Loading feature store: {feature_path}")

    df = pd.read_parquet(feature_path)

    # PR#6A: Set timestamp as index if it's a column
    if 'timestamp' in df.columns and df.index.name != 'timestamp':
        df = df.set_index('timestamp')

    # PR#6A: Alias k2_fusion_score to fusion_score for archetype logic compatibility
    if 'k2_fusion_score' in df.columns and 'fusion_score' not in df.columns:
        df['fusion_score'] = df['k2_fusion_score']

    # Filter to date range
    start_ts = pd.Timestamp(args.start, tz='UTC')
    end_ts = pd.Timestamp(args.end, tz='UTC')
    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Load params from config or use defaults
    runtime_config = None
    if args.config:
        with open(args.config) as f:
            runtime_config = json.load(f)
        params = KnowledgeParams()  # Use defaults for now
    else:
        params = KnowledgeParams()

    # Run backtest
    backtest = KnowledgeAwareBacktest(df, params, asset=args.asset, runtime_config=runtime_config)
    results = backtest.run()

    # Print results
    print("\n" + "=" * 80)
    print(f"Knowledge-Aware Backtest Results - {args.asset}")
    print("=" * 80)
    print(f"Total PNL: ${results['total_pnl']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Gross Profit: ${results['gross_profit']:.2f}")
    print(f"Gross Loss: ${results['gross_loss']:.2f}")
    print(f"Avg Win: ${results['avg_win']:.2f}")
    print(f"Avg Loss: ${results['avg_loss']:.2f}")

    # PR#6A: Print archetype statistics
    print("\n" + "=" * 80)
    if backtest.archetype_telemetry:
        print("PR#6A: Archetype Entry Statistics (11-Archetype System)")
        print("=" * 80)
        backtest.archetype_telemetry.dump(logger=None)  # Print to stdout
    else:
        print("Archetype Entry Statistics (Legacy 3-Archetype System)")
        print("=" * 80)
        print(f"Total archetype checks: {backtest._archetype_checks}")
        print(f"Archetype A matches (Trap Reversal): {backtest._archetype_a_matches}")
        print(f"Archetype B matches (OB Retest): {backtest._archetype_b_matches}")
        print(f"Archetype C matches (FVG Continuation): {backtest._archetype_c_matches}")
        total_matches = backtest._archetype_a_matches + backtest._archetype_b_matches + backtest._archetype_c_matches
        if backtest._archetype_checks > 0:
            print(f"Total matches: {total_matches} ({100*total_matches/backtest._archetype_checks:.2f}% of checks)")

    print("\n" + "=" * 80)
    print("Trade Log")
    print("=" * 80)
    for i, trade in enumerate(results['trades'], 1):
        print(f"\nTrade {i}: {trade.entry_reason}")
        print(f"  Entry: {trade.entry_time} @ ${trade.entry_price:.2f}")
        print(f"  Exit:  {trade.exit_time} @ ${trade.exit_price:.2f} ({trade.exit_reason})")
        print(f"  PNL: ${trade.net_pnl:.2f} ({trade.net_pnl/trade.position_size:.2%})")
        print(f"  Wyckoff: {trade.wyckoff_phase} (M1={trade.wyckoff_m1_signal}, M2={trade.wyckoff_m2_signal})")
        print(f"  Macro: {trade.macro_regime}, PTI: {trade.pti_score_1d:.3f}/{trade.pti_score_1h:.3f}")
        print(f"  FRVP: {trade.frvp_poc_position}, Fusion: {trade.entry_fusion_score:.3f}")

    # ML Export: Export trades with features if requested
    if args.export_trades:
        print("\n" + "=" * 80)
        print("ML Feature Export")
        print("=" * 80)
        backtest.export_trades_with_ml_features(args.export_trades)
