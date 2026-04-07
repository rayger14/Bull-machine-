"""
Isolated Archetype Engine - Integration Wrapper for v11 Architecture

Integrates the new archetype isolation architecture into Bull Machine backtest:
- Per-archetype fusion calculation (ArchetypeInstance)
- Portfolio allocation with correlation constraints (PortfolioAllocator)
- Existing backtest infrastructure
- Optional ML fusion scoring (XGBoost, opt-in via config)
- Optional Kelly criterion position sizing (opt-in via config)

This is the bridge between new isolation architecture and existing backtest engine.

Architecture Flow:
    Feature Store -> ArchetypeInstances -> Signals -> [ML Fusion] -> PortfolioAllocator -> [Kelly Sizing] -> Positions

Key Components:
- ArchetypeInstance: Self-contained archetype with own fusion
- PortfolioAllocator: Multi-archetype allocation with correlation filtering
- RegimeService: Probabilistic regime detection (optional)
- FusionScorerML: XGBoost-based fusion scoring (optional, opt-in via use_ml_fusion)
- KellyLiteSizer: ML-based position sizing (optional, opt-in via use_kelly_sizing)

Author: Claude Sonnet 4.5
Date: 2026-02-04
Version: v11 (Isolation Architecture + ML Integration)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from engine.archetypes.archetype_instance import ArchetypeInstance, ArchetypeConfig, _safe_float
from engine.archetypes.structural_check import StructuralChecker
from engine.portfolio.archetype_allocator import (
    PortfolioAllocator,
    ArchetypeSignal,
    AllocationMode
)
from engine.config.archetype_config_loader import load_archetype_configs
from engine.context.regime_service import RegimeService, REGIME_MODE_PROBABILISTIC
from engine.models.base import Signal

logger = logging.getLogger(__name__)


class IsolatedArchetypeEngine:
    """
    Engine that runs multiple isolated archetypes with portfolio allocation.

    This is the main integration point for the v11 isolation architecture.

    Optional ML enhancements (all opt-in via config flags):
    - use_ml_fusion: Blend XGBoost fusion scores with static archetype fusion
    - use_kelly_sizing: Use Kelly criterion for dynamic position sizing

    Usage:
        engine = IsolatedArchetypeEngine(
            archetype_config_dir='configs/archetypes/',
            portfolio_config={...},
            config={'use_ml_fusion': True, 'use_kelly_sizing': True}
        )

        for bar in feature_store:
            signals = engine.get_signals(bar)
            allocations = engine.allocate(signals, current_positions)
            # Execute allocations...
    """

    def __init__(
        self,
        archetype_config_dir: str = 'configs/archetypes/',
        portfolio_config: Optional[Dict[str, Any]] = None,
        regime_service: Optional[RegimeService] = None,
        enable_regime: bool = True,
        regime_model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize isolated archetype engine.

        Args:
            archetype_config_dir: Directory with archetype YAML configs
            portfolio_config: Portfolio allocation config
            regime_service: Optional RegimeService instance
            enable_regime: Enable regime detection
            regime_model_path: Path to regime model (if regime_service not provided)
            config: Full config dict (for ML options: use_ml_fusion, use_kelly_sizing,
                    ml_fusion_model_path, kelly_model_path, ml_fusion_blend_weight, etc.)
        """
        self.archetype_config_dir = archetype_config_dir
        self.portfolio_config = portfolio_config or {}
        self.config = config or {}

        # Load archetype configs
        logger.info(f"Loading archetype configs from {archetype_config_dir}")
        self.archetype_configs = load_archetype_configs(
            archetype_config_dir,
            enabled_only=True
        )

        logger.info(f"Loaded {len(self.archetype_configs)} enabled archetypes")

        # Global gate_mode override (from main config, overrides per-YAML settings)
        self.global_gate_mode = self.config.get('gate_mode', None)  # None = use per-YAML

        # ATR overrides from config (used by Optuna to test different stop/TP multipliers)
        # Format: {"wick_trap": {"atr_stop_mult": 3.0, "atr_tp_mult": 5.0}, ...}
        self.atr_overrides = self.config.get('atr_overrides', {})

        # Gate overrides from config (used by Optuna to test different gate threshold values)
        # Format: {"wick_trap": {"volume_zscore": 0.5}, "retest_cluster": {"temporal_confluence_score": 0.2}, ...}
        # Overrides the 'value' field of matching gates by feature name
        self.gate_overrides = self.config.get('gate_overrides', {})

        # Create ArchetypeInstance for each config
        self.archetypes: Dict[str, ArchetypeInstance] = {}
        for name, config_dict in self.archetype_configs.items():
            # Apply global gate_mode override if set
            if self.global_gate_mode is not None:
                config_dict = dict(config_dict)  # Don't mutate original
                config_dict['gate_mode'] = self.global_gate_mode
            # Apply ATR overrides if set (for Optuna optimization)
            if name in self.atr_overrides:
                config_dict = dict(config_dict)
                ps = dict(config_dict.get('position_sizing', {}))
                for k, v in self.atr_overrides[name].items():
                    ps[k] = v
                config_dict['position_sizing'] = ps
            # Apply gate overrides if set (for Optuna optimization of YAML gate thresholds)
            if name in self.gate_overrides:
                config_dict = dict(config_dict)
                gates = [dict(g) for g in config_dict.get('hard_gates', [])]
                overrides = self.gate_overrides[name]
                for i, gate in enumerate(gates):
                    feat = gate.get('feature', '')
                    if feat in overrides:
                        gates[i]['value'] = overrides[feat]
                config_dict['hard_gates'] = gates
            archetype_config = self._convert_to_archetype_config(config_dict)
            self.archetypes[name] = ArchetypeInstance(archetype_config)
            logger.info(
                f"  Initialized: {name} ({config_dict['direction']}) - "
                f"Fusion W={config_dict['fusion_weights']['wyckoff']:.2f}"
            )

        # Initialize regime service
        self.regime_service = regime_service
        if enable_regime and self.regime_service is None and regime_model_path:
            logger.info("Initializing RegimeService in probabilistic mode")
            self.regime_service = RegimeService(
                mode=REGIME_MODE_PROBABILISTIC,
                model_path=regime_model_path
            )

        # Initialize portfolio allocator
        allocation_mode = self.portfolio_config.get('mode', 'MULTI_UNCORRELATED')
        mode_enum = AllocationMode[allocation_mode]

        self.portfolio_allocator = PortfolioAllocator(
            regime_allocator=None,  # TODO: Connect RegimeWeightAllocator if needed
            correlation_matrix=None,  # TODO: Load correlation matrix if available
            correlation_threshold=self.portfolio_config.get('correlation_threshold', 0.7),
            max_simultaneous_positions=self.portfolio_config.get('max_simultaneous_positions', 8),
            allocation_mode=mode_enum,
            enable_correlation_filter=self.portfolio_config.get('enable_correlation_filter', True),
            config=self.portfolio_config
        )

        logger.info(
            f"Portfolio allocator initialized: "
            f"mode={allocation_mode}, "
            f"max_positions={self.portfolio_config.get('max_simultaneous_positions', 3)}"
        )

        # Statistics
        self.stats = {
            'total_bars': 0,
            'total_signals': 0,
            'signals_by_archetype': {name: 0 for name in self.archetypes},
            'allocations': 0,
            'rejections': 0,
            'signals_filtered_by_score': 0,
            'signals_blocked_by_cooling': 0,
            'ml_fusion_applied': 0,
            'kelly_sizing_applied': 0
        }

        # Initialize all runtime attributes via reset() so they exist before
        # the first get_signals() call (ml_fusion_scorer, kelly_sizer, structural_checker)
        self.reset()

    def reset(self):
        """Reset all stateful fields for reuse across WFO/CPCV splits.

        Clears per-archetype cooling state and signal counters so the engine
        can be re-run on a different date range without re-loading YAML configs.
        """
        for arch in self.archetypes.values():
            arch.last_signal_bar = None
        self.stats = {
            'total_bars': 0,
            'total_signals': 0,
            'signals_by_archetype': {name: 0 for name in self.archetypes},
            'allocations': 0,
            'rejections': 0,
            'signals_filtered_by_score': 0,
            'signals_blocked_by_cooling': 0,
            'ml_fusion_applied': 0,
            'kelly_sizing_applied': 0
        }

        # Signal filtering parameters
        self.relative_score_percentile = self.portfolio_config.get('relative_score_percentile', 0.70)
        logger.info(f"Relative score filtering: keep top {self.relative_score_percentile*100:.0f}% of signals")

        # --- ML Fusion Scorer (opt-in) ---
        self.ml_fusion_scorer = None
        self.ml_fusion_blend_weight = self.config.get('ml_fusion_blend_weight', 0.6)
        if self.config.get('use_ml_fusion', False):
            self._init_ml_fusion_scorer()

        # --- Kelly-Lite Position Sizer (opt-in) ---
        self.kelly_sizer = None
        self.kelly_consecutive_losses = 0  # Track for Kelly input
        self.kelly_recent_dd = 0.0  # Track recent drawdown for Kelly input
        if self.config.get('use_kelly_sizing', False):
            self._init_kelly_sizer()

        # --- Structural Pattern Checker (logic.py wiring) ---
        self.structural_checker = None
        structural_cfg = self.config.get('structural_checks', {})
        structural_enabled = structural_cfg.get('enabled', True)
        try:
            from engine import feature_flags
            if not feature_flags.ENABLE_STRUCTURAL_CHECKS:
                structural_enabled = False
        except (ImportError, AttributeError):
            pass

        if structural_enabled:
            mode_context = structural_cfg.get('mode_context', 'backtest')
            gate_params = structural_cfg.get('gate_params', {})
            self.structural_checker = StructuralChecker(
                config={'use_archetypes': True, 'thresholds': {}},
                mode=mode_context,
                gate_params=gate_params,
            )
            logger.info(
                f"[STRUCTURAL] Structural pattern checks ENABLED (mode={mode_context})"
            )
        else:
            logger.info("[STRUCTURAL] Structural pattern checks DISABLED")

    def _init_ml_fusion_scorer(self):
        """
        Initialize the ML fusion scorer if model exists and xgboost is available.

        Loads FusionScorerML from engine.ml.fusion_scorer_ml.
        The scorer blends ML-predicted fusion confidence with static archetype fusion.
        Controlled by config flags:
        - use_ml_fusion: bool (must be True to activate)
        - ml_fusion_model_path: str (default: 'models/fusion_scorer_xgb.pkl')
        - ml_fusion_blend_weight: float (default: 0.6, i.e. 60% ML, 40% static)
        """
        try:
            from engine.ml.fusion_scorer_ml import FusionScorerML

            model_path = self.config.get(
                'ml_fusion_model_path', 'models/fusion_scorer_xgb.pkl'
            )

            if Path(model_path).exists():
                self.ml_fusion_scorer = FusionScorerML.load(model_path)
                logger.info(
                    f"[ML] Loaded ML fusion scorer from {model_path} "
                    f"(blend_weight={self.ml_fusion_blend_weight:.2f})"
                )
            else:
                logger.warning(
                    f"[ML] Fusion model not found at {model_path}, "
                    f"using static fusion weights only"
                )
        except ImportError as e:
            logger.warning(
                f"[ML] Could not import FusionScorerML (xgboost not installed?): {e}. "
                f"Falling back to static fusion weights."
            )
        except Exception as e:
            logger.warning(
                f"[ML] Failed to initialize ML fusion scorer: {e}. "
                f"Falling back to static fusion weights."
            )

    def _init_kelly_sizer(self):
        """
        Initialize the Kelly-Lite position sizer if model exists.

        Loads KellyLiteSizer from engine.ml.kelly_lite_sizer.
        Uses Kelly criterion with guardrails for dynamic position sizing.
        Controlled by config flags:
        - use_kelly_sizing: bool (must be True to activate)
        - kelly_model_path: str (default: 'models/kelly_lite_sizer.pkl')
        - kelly_base_risk_pct: float (default: 0.0075, i.e. 0.75%)
        """
        try:
            from engine.ml.kelly_lite_sizer import KellyLiteSizer

            model_path = self.config.get(
                'kelly_model_path', 'models/kelly_lite_sizer.pkl'
            )
            base_risk_pct = self.config.get('kelly_base_risk_pct', 0.0075)

            if Path(model_path).exists():
                self.kelly_sizer = KellyLiteSizer.load(model_path)
                logger.info(f"[KELLY] Loaded Kelly-Lite sizer from {model_path}")
            else:
                # Initialize without trained model - will use base_risk_pct as fallback
                self.kelly_sizer = KellyLiteSizer(base_risk_pct=base_risk_pct)
                logger.info(
                    f"[KELLY] Kelly-Lite sizer initialized without trained model "
                    f"(base_risk_pct={base_risk_pct:.4f}). "
                    f"Train a model and save to {model_path} for ML-based sizing."
                )
        except ImportError as e:
            logger.warning(
                f"[KELLY] Could not import KellyLiteSizer (sklearn not installed?): {e}. "
                f"Falling back to static position sizing."
            )
        except Exception as e:
            logger.warning(
                f"[KELLY] Failed to initialize Kelly sizer: {e}. "
                f"Falling back to static position sizing."
            )

    def _convert_to_archetype_config(self, config_dict: Dict) -> ArchetypeConfig:
        """Convert loaded YAML config to ArchetypeConfig dataclass."""
        return ArchetypeConfig(
            name=config_dict['name'],
            direction=config_dict['direction'],
            fusion_weights=config_dict['fusion_weights'],
            entry_threshold=config_dict['thresholds'].get('fusion_threshold', 0.35),
            min_liquidity=config_dict['thresholds'].get('liquidity_threshold', 0.12),
            atr_stop_mult=config_dict['position_sizing'].get('atr_stop_mult', 2.5),
            atr_tp_mult=config_dict['position_sizing'].get('atr_tp_mult', 2.5),
            max_hold_hours=config_dict['exit_logic'].get('max_hold_hours', 168),
            trailing_stop=config_dict['exit_logic'].get('trailing_start_r', 0) > 0,
            max_risk_pct=config_dict['position_sizing'].get('risk_per_trade_pct', 0.02),
            regime_weights=config_dict.get('regime_preferences', {}),
            pattern_params=config_dict.get('thresholds', {}),
            hard_gates=config_dict.get('hard_gates', []),
            gate_mode=config_dict.get('gate_mode', 'hard'),
        )

    def _apply_ml_fusion_scoring(
        self,
        archetype_signal: ArchetypeSignal,
        features: Dict,
        regime_label: str,
        timestamp: Optional[pd.Timestamp] = None
    ) -> ArchetypeSignal:
        """
        Blend ML fusion score with static archetype fusion score.

        The ML scorer uses domain scores, macro features, and market structure
        to predict a fusion confidence. This is blended with the static score:
            blended = (ml_weight * ml_score) + ((1 - ml_weight) * static_score)

        Args:
            archetype_signal: Signal with static fusion_score already set
            features: Feature dict from the current bar
            regime_label: Current regime label
            timestamp: Current bar timestamp

        Returns:
            ArchetypeSignal with potentially updated fusion_score
        """
        if self.ml_fusion_scorer is None:
            return archetype_signal

        try:
            # Build domain scores dict from features (matching FusionScorerML interface)
            # The archetype instance already extracted these; we reconstruct from features
            archetype_name = archetype_signal.archetype_id
            archetype = self.archetypes.get(archetype_name)

            if archetype is not None:
                domain_scores = {
                    'wyckoff': archetype._get_wyckoff_score(features),
                    'smc': archetype._get_smc_score(features),
                    'hob': archetype._get_liquidity_score(features),  # HOB maps to liquidity
                    'momentum': archetype._get_momentum_score(features),
                    'temporal': _safe_float(features.get('tf1h_temporal_score', 0.0))
                }
            else:
                # Fallback: use raw features with defaults
                domain_scores = {
                    'wyckoff': features.get('wyckoff_event_confidence', 0.0),
                    'smc': features.get('fusion_smc', 0.0),
                    'hob': features.get('liquidity_score', 0.0),
                    'momentum': features.get('adx_14', 25.0) / 100.0,
                    'temporal': _safe_float(features.get('tf1h_temporal_score', 0.0))
                }

            # Build macro features dict
            macro_features = {
                'rv_20d': features.get('rv_20d', 40.0),
                'rv_60d': features.get('rv_60d', 45.0),
                'vix_proxy': features.get('vix_proxy', features.get('VIX', 20.0)),
                'regime': regime_label
            }

            # Build market features dict
            close = features.get('close', 1.0)
            atr = features.get('atr_14', close * 0.02)
            atr_normalized = atr / close if close > 0 else 0.02

            volume = features.get('volume', 1.0)
            vol_mean = features.get('volume_sma_20', volume)
            volume_ratio = volume / vol_mean if vol_mean > 0 else 1.0

            market_features = {
                'adx': features.get('adx_14', 25.0),
                'atr_normalized': atr_normalized,
                'volume_ratio': volume_ratio
            }

            # Get ML fusion score
            ml_score = self.ml_fusion_scorer.predict_fusion_score(
                domain_scores=domain_scores,
                macro_features=macro_features,
                market_features=market_features,
                timestamp=timestamp
            )

            # Blend ML and static scores
            static_score = archetype_signal.fusion_score
            blend_weight = self.ml_fusion_blend_weight
            blended_score = (blend_weight * ml_score) + ((1.0 - blend_weight) * static_score)

            # Update the signal
            archetype_signal.fusion_score = blended_score

            # Store both scores in metadata for analysis
            archetype_signal.metadata['static_fusion_score'] = static_score
            archetype_signal.metadata['ml_fusion_score'] = ml_score
            archetype_signal.metadata['blended_fusion_score'] = blended_score

            self.stats['ml_fusion_applied'] += 1

            logger.debug(
                f"[ML] {archetype_name}: static={static_score:.3f}, "
                f"ml={ml_score:.3f}, blended={blended_score:.3f} "
                f"(weight={blend_weight:.2f})"
            )

        except Exception as e:
            logger.debug(
                f"[ML] Fusion scoring failed for {archetype_signal.archetype_id}: {e}. "
                f"Keeping static score={archetype_signal.fusion_score:.3f}"
            )

        return archetype_signal

    def get_kelly_risk_pct(
        self,
        fusion_score: float,
        regime: str,
        features: Dict,
        base_risk_pct: float = 0.02
    ) -> float:
        """
        Get Kelly-adjusted risk percentage for position sizing.

        If Kelly sizer is active and has a trained model, uses ML prediction.
        Otherwise returns the base_risk_pct unchanged.

        Args:
            fusion_score: Fusion confidence score for this signal
            regime: Current regime label
            features: Feature dict from the current bar
            base_risk_pct: Fallback risk percentage if Kelly is not available

        Returns:
            Risk percentage (clamped to [0, 0.02] by Kelly guardrails)
        """
        if self.kelly_sizer is None:
            return base_risk_pct

        try:
            kelly_risk = self.kelly_sizer.predict_risk_pct(
                fusion_score=fusion_score,
                regime=regime,
                rv_20d=features.get('rv_20d', 40.0),
                rv_60d=features.get('rv_60d', 45.0),
                vix_proxy=features.get('vix_proxy', features.get('VIX', 20.0)),
                recent_dd=self.kelly_recent_dd,
                expected_r=1.0,  # Default 1R expected
                consecutive_losses=self.kelly_consecutive_losses
            )

            self.stats['kelly_sizing_applied'] += 1

            logger.debug(
                f"[KELLY] regime={regime}, fusion={fusion_score:.3f}, "
                f"consecutive_losses={self.kelly_consecutive_losses}, "
                f"dd={self.kelly_recent_dd:.3f} -> risk={kelly_risk:.4f} "
                f"(base={base_risk_pct:.4f})"
            )

            return kelly_risk

        except Exception as e:
            logger.debug(f"[KELLY] Sizing failed: {e}. Using base_risk_pct={base_risk_pct:.4f}")
            return base_risk_pct

    def update_kelly_state(self, trade_pnl: float, current_drawdown: float = 0.0):
        """
        Update Kelly sizer state after a trade completes.

        Call this after each trade closes to keep the consecutive loss counter
        and drawdown tracker up to date.

        Args:
            trade_pnl: PnL of the completed trade (positive = win, negative = loss)
            current_drawdown: Current portfolio drawdown as a negative fraction
                              (e.g., -0.05 for 5% drawdown)
        """
        if trade_pnl > 0:
            self.kelly_consecutive_losses = 0
        else:
            self.kelly_consecutive_losses += 1

        self.kelly_recent_dd = current_drawdown

    def filter_signals_by_relative_score(
        self,
        signals: List[ArchetypeSignal],
        percentile: Optional[float] = None
    ) -> List[ArchetypeSignal]:
        """
        Keep only signals in top percentile of fusion scores.

        This prevents overtrading by filtering weak signals when many
        archetypes fire simultaneously.

        Args:
            signals: List of candidate signals
            percentile: Keep signals >= this percentile (default from config)

        Returns:
            Filtered list of signals
        """
        if not signals:
            return []

        if percentile is None:
            percentile = self.relative_score_percentile

        # Extract fusion scores
        scores = [s.fusion_score for s in signals]

        # Compute threshold
        threshold = np.percentile(scores, percentile * 100)

        # Filter
        filtered = [s for s in signals if s.fusion_score >= threshold]

        # Log filtering
        if len(filtered) < len(signals):
            removed = len(signals) - len(filtered)
            self.stats['signals_filtered_by_score'] += removed
            logger.info(
                f"[FILTER] Relative score filter removed {removed} weak signals "
                f"(threshold={threshold:.3f}, kept {len(filtered)}/{len(signals)})"
            )

        return filtered

    def _deduplicate_signals(
        self,
        signals: List[ArchetypeSignal],
        mode: str = 'best_per_direction',
    ) -> List[ArchetypeSignal]:
        """
        Deduplicate signals fired on the same bar.

        Modes:
            'best_of_bar': Keep only the single highest-fusion signal.
            'best_per_direction': Keep max 1 long + 1 short per bar.
            'unique_sl_zone': Group by SL zone (within 2% of entry), keep best per zone.
            'disabled': No dedup (pass-through).
        """
        if mode == 'disabled' or len(signals) <= 1:
            return signals

        if mode == 'best_of_bar':
            best = max(signals, key=lambda s: s.fusion_score)
            removed = len(signals) - 1
            if removed > 0:
                self.stats.setdefault('dedup_removed', 0)
                self.stats['dedup_removed'] += removed
                logger.info(
                    f"[DEDUP] best_of_bar: kept {best.archetype_id} "
                    f"(fusion={best.fusion_score:.3f}), removed {removed}"
                )
            return [best]

        elif mode == 'best_per_direction':
            longs = [s for s in signals if s.direction == 'long']
            shorts = [s for s in signals if s.direction == 'short']
            result = []
            if longs:
                best_long = max(longs, key=lambda s: s.fusion_score)
                result.append(best_long)
            if shorts:
                best_short = max(shorts, key=lambda s: s.fusion_score)
                result.append(best_short)
            removed = len(signals) - len(result)
            if removed > 0:
                self.stats.setdefault('dedup_removed', 0)
                self.stats['dedup_removed'] += removed
                kept_names = [s.archetype_id for s in result]
                logger.info(
                    f"[DEDUP] best_per_direction: {len(signals)} -> {len(result)} "
                    f"(kept {kept_names}, removed {removed})"
                )
            return result

        elif mode == 'unique_sl_zone':
            sorted_sigs = sorted(signals, key=lambda s: s.fusion_score, reverse=True)
            kept = []
            used_zones = []  # list of (direction, sl_level)
            for sig in sorted_sigs:
                is_dup = False
                for (d, sl) in used_zones:
                    if d != sig.direction:
                        continue
                    if sig.entry_price > 0:
                        sl_diff_pct = abs(sig.stop_loss - sl) / sig.entry_price
                        if sl_diff_pct < 0.02:  # within 2% = same zone
                            is_dup = True
                            break
                if not is_dup:
                    kept.append(sig)
                    used_zones.append((sig.direction, sig.stop_loss))
            removed = len(signals) - len(kept)
            if removed > 0:
                self.stats.setdefault('dedup_removed', 0)
                self.stats['dedup_removed'] += removed
                kept_names = [s.archetype_id for s in kept]
                logger.info(
                    f"[DEDUP] unique_sl_zone: {len(signals)} -> {len(kept)} "
                    f"(kept {kept_names}, removed {removed})"
                )
            return kept

        return signals

    def get_signals(
        self,
        bar: pd.Series,
        regime_probs: Optional[Dict[str, float]] = None,
        bar_index: Optional[int] = None,
        prev_row: Optional[pd.Series] = None,
        lookback_df: Optional[pd.DataFrame] = None,
    ) -> List[ArchetypeSignal]:
        """
        Generate signals from all archetypes for current bar.

        If ML fusion scoring is enabled, blends ML predictions with static
        archetype fusion scores before filtering.

        Args:
            bar: Current bar (row from feature store)
            regime_probs: Optional regime probabilities (from RegimeService)
            bar_index: Bar index (for cooling period tracking)
            prev_row: Previous bar's features (for structural pattern checks)
            lookback_df: DataFrame of recent bars (for structural checks needing history)

        Returns:
            List of ArchetypeSignal objects (one per firing archetype)
        """
        self.stats['total_bars'] += 1

        # Get regime
        if self.regime_service:
            regime_result = self.regime_service.get_regime(bar.to_dict(), bar.name)
            regime_label = regime_result['regime_label']
            regime_probs = regime_result['regime_probs']
        else:
            regime_label = bar.get('regime_label', bar.get('macro_regime', 'neutral'))
            regime_probs = {regime_label: 1.0}

        # Convert bar Series to dict for archetype detection
        features = bar.to_dict()

        # Collect signals from all archetypes
        signals = []

        for name, archetype in self.archetypes.items():
            # Get signal from archetype (with structural check + cooling period)
            signal = archetype.detect(
                features, regime_label,
                current_bar_idx=bar_index,
                prev_row=prev_row,
                lookback_df=lookback_df,
                structural_checker=self.structural_checker,
            )

            if signal is not None:
                # Convert Signal to ArchetypeSignal
                archetype_signal = ArchetypeSignal(
                    archetype_id=name,
                    direction=signal.direction,
                    confidence=signal.confidence,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    fusion_score=signal.metadata['fusion_score'],
                    regime_label=regime_label,
                    timestamp=bar.name,
                    metadata=signal.metadata
                )

                # Apply ML fusion scoring if enabled (blends with static score)
                if self.ml_fusion_scorer is not None:
                    archetype_signal = self._apply_ml_fusion_scoring(
                        archetype_signal,
                        features,
                        regime_label,
                        timestamp=bar.name
                    )

                signals.append(archetype_signal)
                self.stats['total_signals'] += 1
                self.stats['signals_by_archetype'][name] += 1

                logger.debug(
                    f"[SIGNAL] {name} @ {bar.name}: "
                    f"fusion={archetype_signal.fusion_score:.3f}, "
                    f"confidence={signal.confidence:.3f}"
                )
            elif bar_index is not None and not archetype.can_signal(bar_index):
                # Track cooling period blocks
                self.stats['signals_blocked_by_cooling'] += 1

        # Apply relative score filtering if enabled and multiple signals present
        if len(signals) > 1 and self.relative_score_percentile < 1.0:
            signals = self.filter_signals_by_relative_score(signals)

        # Apply signal deduplication (prevent multiple archetypes on same bar)
        dedup_cfg = self.config.get('signal_dedup', {})
        dedup_mode = dedup_cfg.get('mode', 'disabled')
        if len(signals) > 1 and dedup_mode != 'disabled':
            signals = self._deduplicate_signals(signals, mode=dedup_mode)

        return signals

    def allocate(
        self,
        signals: List[ArchetypeSignal],
        current_positions: List[str],
        regime_probs: Optional[Dict[str, float]] = None
    ):
        """
        Allocate portfolio capital to signals.

        Args:
            signals: List of active signals from archetypes
            current_positions: List of archetype IDs with open positions
            regime_probs: Regime probability distribution

        Returns:
            Tuple of (intents, rejections) from PortfolioAllocator
        """
        intents, rejections = self.portfolio_allocator.allocate(
            signals,
            current_positions,
            regime_probs
        )

        self.stats['allocations'] += len(intents)
        self.stats['rejections'] += len(rejections)

        return intents, rejections

    def get_position_size(
        self,
        archetype_name: str,
        signal: Signal,
        portfolio_value: float,
        regime: str
    ) -> float:
        """
        Get position size for archetype signal.

        Args:
            archetype_name: Archetype identifier
            signal: Entry signal
            portfolio_value: Current portfolio value ($)
            regime: Current regime

        Returns:
            Position size in dollars
        """
        if archetype_name not in self.archetypes:
            logger.warning(f"Unknown archetype: {archetype_name}")
            return 0.0

        archetype = self.archetypes[archetype_name]
        return archetype.get_position_size(portfolio_value, signal, regime)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            'signal_rate': (
                self.stats['total_signals'] / max(self.stats['total_bars'], 1)
            ),
            'allocation_rate': (
                self.stats['allocations'] / max(self.stats['total_signals'], 1)
            ),
            'ml_fusion_active': self.ml_fusion_scorer is not None,
            'kelly_sizing_active': self.kelly_sizer is not None
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_stats()

        lines = [
            "\n" + "="*80,
            "ISOLATED ARCHETYPE ENGINE SUMMARY",
            "="*80,
            f"Archetypes: {len(self.archetypes)}",
            f"Total Bars: {stats['total_bars']:,}",
            f"Total Signals: {stats['total_signals']:,} ({stats['signal_rate']:.3f} per bar)",
            f"Allocations: {stats['allocations']:,} ({stats['allocation_rate']:.1%} of signals)",
            f"Rejections: {stats['rejections']:,}",
            "",
            "ML ENHANCEMENTS:",
            f"  ML Fusion Scoring: {'ACTIVE' if stats['ml_fusion_active'] else 'disabled'}",
            f"  Kelly Sizing:      {'ACTIVE' if stats['kelly_sizing_active'] else 'disabled'}",
        ]

        if stats['ml_fusion_applied'] > 0:
            lines.append(f"  ML Fusion Applied:  {stats['ml_fusion_applied']:,} signals")
        if stats['kelly_sizing_applied'] > 0:
            lines.append(f"  Kelly Sizing Applied: {stats['kelly_sizing_applied']:,} trades")

        lines.append("")
        lines.append("SIGNALS BY ARCHETYPE:")

        # Sort by signal count
        sorted_archetypes = sorted(
            stats['signals_by_archetype'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for name, count in sorted_archetypes:
            if count > 0:
                lines.append(f"  {name:20s}: {count:4d} signals")

        lines.append("="*80)

        return "\n".join(lines)
