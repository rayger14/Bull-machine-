"""
ArchetypeModel - Wrapper for existing archetype system.

Wraps the existing ArchetypeLogic (logic_v2_adapter.py) to implement
the BaseModel interface for clean integration with new architecture.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from engine.models.base import BaseModel, Signal, Position
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext
from engine.archetypes.threshold_policy import ThresholdPolicy
from engine.archetypes.exit_logic import ExitLogic, create_default_exit_config

logger = logging.getLogger(__name__)


class ArchetypeModel(BaseModel):
    """
    Wrapper that adapts the existing archetype system to BaseModel interface.

    This is a thin delegation layer that:
    1. Loads config and initializes ArchetypeLogic
    2. Converts archetype detect() output to Signal objects
    3. Provides simple position sizing based on ATR risk

    Does NOT modify core engine files - just wraps them.
    """

    def __init__(
        self,
        config_path: str,
        archetype_name: str = 'S4',
        name: Optional[str] = None,
        regime_classifier_path: Optional[str] = None,
        regime_allocator: Optional[Any] = None,
        regime_mode: str = 'static',
        regime_service: Optional[Any] = None
    ):
        """
        Initialize archetype model wrapper.

        Args:
            config_path: Path to config JSON (e.g., 'configs/s4_optimized.json')
            archetype_name: Single archetype to use (e.g., 'S4', 'S1', 'trap_within_trend')
            name: Human-readable model name (defaults to archetype_name)
            regime_classifier_path: Optional path to regime classifier model
            regime_allocator: Optional RegimeWeightAllocator for soft gating
            regime_mode: Regime detection mode ('static' | 'probabilistic')
            regime_service: Optional RegimeService instance for probabilistic mode
        """
        super().__init__(name=name or f"Archetype-{archetype_name}")

        self.config_path = Path(config_path)
        self.archetype_name = archetype_name
        self.regime_classifier_path = regime_classifier_path
        self.regime_allocator = regime_allocator
        self.regime_mode = regime_mode
        self.regime_service = regime_service

        # Load configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.full_config = json.load(f)

        # Extract archetype config subsection
        self.archetype_config = self.full_config.get('archetypes', {})

        # Initialize ArchetypeLogic with config
        self.archetype_logic = ArchetypeLogic(self.archetype_config)

        # Initialize ThresholdPolicy for regime-aware thresholds
        # ThresholdPolicy expects full config, not just archetypes section
        self.threshold_policy = ThresholdPolicy(
            base_cfg=self.full_config,
            regime_profiles=self.full_config.get('gates_regime_profiles'),
            archetype_overrides=self.full_config.get('archetype_overrides'),
            global_clamps=self.full_config.get('global_clamps'),
            locked_regime='static'  # Use static mode by default
        )

        # Cache archetype-specific parameters
        self._extract_archetype_params()

        # Simple regime state (neutral by default)
        self.default_regime = 'neutral'

        # Initialize exit logic system
        exit_config = create_default_exit_config()
        # Override with config if provided
        if 'exit_logic' in self.full_config:
            exit_config.update(self.full_config['exit_logic'])
        self.exit_logic = ExitLogic(exit_config)

        logger.info(f"Initialized {self.name} with config: {self.config_path}")
        logger.info(f"Archetype params: {self.archetype_params}")
        logger.info(f"Exit logic initialized with {len(self.exit_logic.exit_rules)} archetype rule sets")
        if self.regime_allocator:
            logger.info("Soft gating enabled with RegimeWeightAllocator")
        if self.regime_service and self.regime_mode == 'probabilistic':
            logger.info("Probabilistic regime mode enabled (3-output system with soft controls)")

    def _extract_archetype_params(self):
        """Extract archetype-specific parameters from config."""
        # Map common archetype names to config keys
        # CRITICAL FIX (2026-01-23): Updated to match logic_v2_adapter.py archetype_map (lines 1500-1522)
        archetype_key_map = {
            # Bear-biased archetypes (short-biased)
            'S1': 'liquidity_vacuum',
            'S2': 'failed_rally',
            'S3': 'whipsaw',
            'S4': 'funding_divergence',
            'S5': 'long_squeeze',
            'S6': 'alt_rotation_down',
            'S7': 'curve_inversion',
            'S8': 'volume_fade_chop',
            # Bull-biased archetypes
            'A': 'trap_reversal',              # FIXED: was 'spring'
            'B': 'order_block_retest',
            'C': 'fvg_continuation',           # FIXED: was 'wick_trap'
            'D': 'failed_continuation',
            'E': 'liquidity_compression',      # FIXED: was 'volume_exhaustion'
            'F': 'expansion_exhaustion',       # FIXED: was 'exhaustion_reversal'
            'G': 're_accumulate',              # FIXED: was 'liquidity_sweep'
            'H': 'trap_within_trend',          # FIXED: was 'momentum_continuation' - THIS WAS THE H BUG!
            'K': 'wick_trap',                  # FIXED: was 'trap_within_trend'
            'L': 'volume_exhaustion',          # FIXED: was 'retest_cluster'
            'M': 'ratio_coil_break',           # FIXED: was 'confluence_breakout'
        }

        # Try to find archetype config
        archetype_key = archetype_key_map.get(self.archetype_name, self.archetype_name)

        # Look in thresholds subdirectory
        thresholds = self.archetype_config.get('thresholds', {})
        self.archetype_params = thresholds.get(archetype_key, {})

        # Also check top-level archetype config
        if archetype_key in self.archetype_config:
            self.archetype_params.update(self.archetype_config[archetype_key])

        # Extract common parameters with defaults
        self.atr_stop_mult = self.archetype_params.get('atr_stop_mult', 2.5)
        self.atr_tp_mult = self.archetype_params.get('atr_tp_mult', 2.5)  # NEW: Profit target multiplier
        self.max_risk_pct = self.archetype_params.get('max_risk_pct', 0.02)
        self.direction = self.archetype_params.get('direction', 'long')

        # Get fusion threshold for confidence mapping
        self.fusion_threshold = self.archetype_params.get('fusion_threshold', 0.35)

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Calibrate model on training data.

        For archetype models, parameters are pre-configured (from Optuna optimization).
        This could run Optuna optimization in the future, but for now it's a no-op.

        Args:
            train_data: Historical data for calibration
            **kwargs: Optional parameters (e.g., n_trials for Optuna)
        """
        logger.info(f"{self.name}: fit() called - using pre-configured parameters")
        logger.info(f"Training data shape: {train_data.shape}")

        # Mark as fitted (parameters already loaded from config)
        self._is_fitted = True

        # Future: Run Optuna optimization here
        # if kwargs.get('optimize', False):
        #     self._run_optuna_optimization(train_data, **kwargs)

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate trading signal for current bar.

        Calls ArchetypeLogic.detect() and converts result to Signal object.

        CRITICAL FIX (2026-01-22): Now position-aware to prevent signal spam
        - If position exists, check for exit conditions FIRST
        - Only generate new entry signals when flat
        - This fixes the 99.7% signal rejection bug (331 signals -> 1 trade)

        Args:
            bar: Current bar data (row from DataFrame)
            position: Current open position (if any)

        Returns:
            Signal object with direction, confidence, entry price, stop loss
        """
        # Get regime label early for all code paths
        context = self._build_runtime_context(bar)
        regime_label = context.regime_label if context else self.default_regime

        # POSITION-AWARE LOGIC: Check for exit conditions first
        if position is not None:
            return self._check_exit_conditions(bar, position, context, regime_label)

        # NO POSITION: Check for entry signals
        # Call archetype logic (returns 4 values: name, fusion, liquidity, direction)
        archetype_name, fusion_score, liquidity_score, direction = self.archetype_logic.detect(context)

        # Apply trade frequency filtering (probabilistic mode only)
        if self.regime_mode == 'probabilistic' and context.metadata.get('regime_result'):
            soft_controls = context.metadata['regime_result'].get('soft_controls', {})
            frequency_mult = soft_controls.get('trade_frequency_multiplier', 1.0)

            # Stochastic filtering: skip signal with probability (1 - frequency_mult)
            # Example: frequency_mult=0.70 means 30% chance to skip
            import random
            if random.random() > frequency_mult:
                logger.debug(
                    f"Signal filtered by frequency multiplier: {frequency_mult:.3f} "
                    f"(instability={context.metadata['regime_result']['instability_score']:.3f})"
                )
                return Signal(
                    direction='hold',
                    confidence=0.0,
                    entry_price=bar['close'],
                    regime_label=regime_label,
                    metadata={
                        'fusion_score': fusion_score,
                        'liquidity_score': liquidity_score,
                        'reason': 'frequency_filter',
                        'frequency_multiplier': frequency_mult
                    }
                )

        # Convert to Signal
        if archetype_name is None:
            # No entry signal
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=bar['close'],
                regime_label=regime_label,
                metadata={
                    'fusion_score': fusion_score,
                    'liquidity_score': liquidity_score,
                    'reason': 'no_archetype_match'
                }
            )

        # Map fusion score to confidence (0.0-1.0)
        # Fusion scores typically range 0.3-0.8, normalize relative to threshold
        confidence = min(1.0, fusion_score / max(self.fusion_threshold, 0.01))

        # Calculate stop loss and take profit
        atr = bar.get('atr_14', bar.get('atr', bar['close'] * 0.02))

        # CRITICAL FIX (2026-01-15): Use direction from archetype detection, not from config
        # This fixes the S5 short execution bug where all trades executed as longs
        # UPDATED (2026-01-23): Use archetype-specific atr_tp_mult instead of hardcoded 2.5
        if direction == 'SHORT':
            stop_loss = bar['close'] + (self.atr_stop_mult * atr)
            signal_direction = 'short'
            # Take profit using archetype-specific multiplier
            take_profit = bar['close'] - (self.atr_stop_mult * atr * self.atr_tp_mult)
        elif direction == 'LONG':
            stop_loss = bar['close'] - (self.atr_stop_mult * atr)
            signal_direction = 'long'
            # Take profit using archetype-specific multiplier
            take_profit = bar['close'] + (self.atr_stop_mult * atr * self.atr_tp_mult)
        elif direction == 'EITHER':
            # For ambiguous archetypes, fall back to config direction
            if self.direction == 'long':
                stop_loss = bar['close'] - (self.atr_stop_mult * atr)
                signal_direction = 'long'
                take_profit = bar['close'] + (self.atr_stop_mult * atr * self.atr_tp_mult)
            else:
                stop_loss = bar['close'] + (self.atr_stop_mult * atr)
                signal_direction = 'short'
                take_profit = bar['close'] - (self.atr_stop_mult * atr * self.atr_tp_mult)
        else:
            # Fallback to config direction if None
            if self.direction == 'long':
                stop_loss = bar['close'] - (self.atr_stop_mult * atr)
                signal_direction = 'long'
                take_profit = bar['close'] + (self.atr_stop_mult * atr * self.atr_tp_mult)
            else:
                stop_loss = bar['close'] + (self.atr_stop_mult * atr)
                signal_direction = 'short'
                take_profit = bar['close'] - (self.atr_stop_mult * atr * self.atr_tp_mult)

        # Build signal metadata
        signal_metadata = {
            'archetype': archetype_name,
            'archetype_direction': direction,  # Store actual archetype direction
            'fusion_score': fusion_score,
            'liquidity_score': liquidity_score,
            'atr': atr,
            'atr_stop_mult': self.atr_stop_mult,
            'atr_tp_mult': self.atr_tp_mult  # NEW: Track profit target multiplier
        }

        # Include regime_result for probabilistic mode
        if context.metadata.get('regime_result'):
            signal_metadata['regime_result'] = context.metadata['regime_result']

        return Signal(
            direction=signal_direction,
            confidence=confidence,
            entry_price=bar['close'],
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime_label=regime_label,
            metadata=signal_metadata
        )

    def _check_exit_conditions(
        self,
        bar: pd.Series,
        position: Position,
        context: RuntimeContext,
        regime_label: str
    ) -> Signal:
        """
        Check if current position should be exited using comprehensive exit logic.

        Uses ExitLogic system with archetype-specific exit rules:
        1. Invalidation exits (pattern breaks, structure fails)
        2. Profit target exits (R-multiple scale-outs)
        3. Time-based exits (max hold period per archetype)
        4. Reason-gone exits (entry condition reverses)
        5. Trailing stop updates

        Args:
            bar: Current bar data
            position: Open position
            context: Runtime context with archetype state
            regime_label: Current regime

        Returns:
            Signal with direction='hold' and exit metadata if should exit,
            otherwise 'hold' signal to maintain position
        """
        close = bar['close']
        atr = bar.get('atr_14', bar.get('atr', bar['close'] * 0.02))

        # Get archetype from position metadata
        archetype = position.metadata.get('archetype', self.archetype_name)

        # Check exit logic (comprehensive archetype-specific rules)
        exit_signal = self.exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype=archetype,
            context=context
        )

        if exit_signal:
            logger.info(
                f"EXIT SIGNAL: {exit_signal.exit_type} @ {close:.2f} "
                f"(archetype={archetype}, reason={exit_signal.reason}, "
                f"exit_pct={exit_signal.exit_pct:.1%})"
            )
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                regime_label=regime_label,
                metadata={
                    'reason': exit_signal.exit_type,
                    'exit_pct': exit_signal.exit_pct,
                    'exit_reason': exit_signal.reason,
                    'stop_update': exit_signal.stop_update
                }
            )

        # FALLBACK: Legacy exit logic (keep for backward compatibility)
        # This should rarely trigger now that ExitLogic is comprehensive

        # 1. CHECK TAKE PROFIT
        if position.take_profit is not None:
            if position.direction == 'long' and close >= position.take_profit:
                logger.info(
                    f"TAKE PROFIT HIT: long @ {position.entry_price:.2f} -> "
                    f"{close:.2f} (target: {position.take_profit:.2f})"
                )
                return Signal(
                    direction='hold',
                    confidence=0.0,
                    entry_price=close,
                    regime_label=regime_label,
                    metadata={'reason': 'profit_target', 'price': close}
                )
            elif position.direction == 'short' and close <= position.take_profit:
                logger.info(
                    f"TAKE PROFIT HIT: short @ {position.entry_price:.2f} -> "
                    f"{close:.2f} (target: {position.take_profit:.2f})"
                )
                return Signal(
                    direction='hold',
                    confidence=0.0,
                    entry_price=close,
                    regime_label=regime_label,
                    metadata={'reason': 'profit_target', 'price': close}
                )

        # 2. CHECK TIME-BASED EXIT (max 7 days)
        hours_in_position = (bar.name - position.entry_time).total_seconds() / 3600
        max_hold_hours = 168  # 7 days

        if hours_in_position >= max_hold_hours:
            logger.info(
                f"TIME EXIT: position held {hours_in_position:.1f}h (max: {max_hold_hours}h), "
                f"entry={position.entry_price:.2f}, current={close:.2f}"
            )
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=close,
                regime_label=regime_label,
                metadata={'reason': 'time_exit', 'hours': hours_in_position}
            )

        # 3. CHECK OPPOSITE SIGNAL (reversal)
        archetype_name, fusion_score, liquidity_score, direction = self.archetype_logic.detect(context)

        if archetype_name is not None:
            # Got a new archetype signal
            # Check if it's opposite direction
            if position.direction == 'long' and direction == 'SHORT':
                logger.info(
                    f"REVERSAL EXIT: long position closed due to SHORT signal "
                    f"(archetype={archetype_name}, fusion={fusion_score:.2f})"
                )
                return Signal(
                    direction='hold',
                    confidence=0.0,
                    entry_price=close,
                    regime_label=regime_label,
                    metadata={
                        'reason': 'signal',
                        'reversal_archetype': archetype_name,
                        'fusion_score': fusion_score
                    }
                )
            elif position.direction == 'short' and direction == 'LONG':
                logger.info(
                    f"REVERSAL EXIT: short position closed due to LONG signal "
                    f"(archetype={archetype_name}, fusion={fusion_score:.2f})"
                )
                return Signal(
                    direction='hold',
                    confidence=0.0,
                    entry_price=close,
                    regime_label=regime_label,
                    metadata={
                        'reason': 'signal',
                        'reversal_archetype': archetype_name,
                        'fusion_score': fusion_score
                    }
                )

        # 4. NO EXIT CONDITIONS MET - HOLD POSITION
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=close,
            regime_label=regime_label,
            metadata={
                'reason': 'holding_position',
                'hours_in_trade': hours_in_position,
                'unrealized_r': self._calculate_unrealized_r(position, close, atr)
            }
        )

    def _calculate_unrealized_r(self, position: Position, current_price: float, atr: float) -> float:
        """
        Calculate unrealized R-multiple for position.

        Args:
            position: Open position
            current_price: Current market price
            atr: Current ATR

        Returns:
            R-multiple (e.g., 1.5 = 1.5x risk captured)
        """
        stop_distance = abs(position.entry_price - position.stop_loss)

        if position.direction == 'long':
            pnl = current_price - position.entry_price
        else:
            pnl = position.entry_price - current_price

        if stop_distance > 0:
            return pnl / stop_distance
        return 0.0

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """
        Calculate position size using ATR-based risk management with soft gating.

        This implements BOTH:
        1. Regime weight scaling (soft gating) using SQUARE-ROOT SPLIT
        2. Regime risk budget caps

        SQUARE-ROOT SPLIT FIX:
        To prevent double-weight bug (w² instead of w), we apply sqrt(regime_weight)
        at the sizing layer. The score layer also applies sqrt(regime_weight),
        giving combined impact: sqrt(w) * sqrt(w) = w (correct!)

        Formula:
            Base Size = (Portfolio Value × Risk %) / Stop Distance %
            Regime Weight = RegimeAllocator.get_weight(archetype, regime)
            Sqrt Weight = sqrt(Regime Weight)  [SQUARE-ROOT SPLIT]
            Position Size = Base Size × Sqrt Weight × Confidence
            Final Size = min(Position Size, Regime Budget Available)

        Args:
            bar: Current bar data
            signal: Entry signal with stop loss

        Returns:
            Position size in quote currency ($)
        """
        # For now, assume fixed portfolio size
        # In production, this would come from account state
        portfolio_value = 10000.0  # $10k default

        # Calculate stop distance as % of entry price
        stop_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price

        # Get regime early for regime-based risk sizing
        # Try multiple column names for regime (macro_regime, regime_label, regime)
        regime = signal.metadata.get('regime',
                                      bar.get('macro_regime',
                                              bar.get('regime_label',
                                                      bar.get('regime', self.default_regime))))

        # TEMPORARY: Disable regime sizing for comparison test
        # regime_risk_pct = self.max_risk_pct  # Fixed 2% risk

        # Regime-based risk percentages (Task #8)
        # Adjust base risk by market regime:
        # - Crisis: 3% (capitalize on extreme volatility)
        # - Risk-off: 2% (standard risk)
        # - Risk-on: 1.5% (reduce exposure in complacent markets)
        # - Neutral: 2% (fallback to standard)
        regime_risk_map = {
            'crisis': 0.03,
            'risk_off': 0.02,
            'risk_on': 0.015,
            'neutral': 0.02
        }
        # TESTING: Disable to compare
        regime_risk_pct = self.max_risk_pct  # Fixed 2% risk
        # regime_risk_pct = regime_risk_map.get(regime, self.max_risk_pct)

        # Risk amount in dollars (now regime-adjusted)
        risk_dollars = portfolio_value * regime_risk_pct

        # Base position size calculation
        # Example: $10k portfolio, 2% risk = $200 risk
        #          Stop 5% away = $200 / 0.05 = $4000 position
        base_position_size = risk_dollars / stop_distance_pct

        # Cap at reasonable max (e.g., 12% of portfolio)
        max_position = portfolio_value * 0.12
        base_position_size = min(base_position_size, max_position)

        # Convert to percentage for soft gating
        base_size_pct = base_position_size / portfolio_value

        # Map archetype name to internal key
        archetype_key_map = {
            'S1': 'liquidity_vacuum',
            'S4': 'funding_divergence',
            'B': 'order_block_retest',
            'C': 'wick_trap_moneytaur',
            'K': 'trap_within_trend',
        }
        archetype_key = archetype_key_map.get(self.archetype_name, self.archetype_name)

        # Apply soft gating if regime allocator is available
        if self.regime_allocator:
            # PROBABILISTIC MODE: Use regime_probs for smooth blending
            if self.regime_mode == 'probabilistic' and 'regime_result' in signal.metadata:
                regime_probs = signal.metadata['regime_result'].get('regime_probs', {regime: 1.0})

                # Get probabilistic regime weight (weighted blend across regime probabilities)
                # For ENTRIES: bypass_entry_filtering flag controls whether to apply regime scaling
                regime_weight = self.regime_allocator.compute_weight_probabilistic(
                    edge=0.5,  # Placeholder - would use historical edge if available
                    N=50,      # Placeholder - would use historical trade count if available
                    archetype=archetype_key,
                    regime_probs=regime_probs,
                    is_entry=True  # Mark as entry decision (bypass flag applies here)
                )

                logger.debug(
                    f"Probabilistic regime weight: archetype={archetype_key}, "
                    f"regime_probs={regime_probs}, weight={regime_weight:.3f}"
                )
            else:
                # DISCRETE MODE: Use single regime label
                regime_weight = self.regime_allocator.get_weight(archetype_key, regime)

            # SQUARE-ROOT SPLIT: Apply sqrt(regime_weight) to prevent double-weight bug
            # This is the sizing layer - score layer also applies sqrt(regime_weight)
            # Combined impact: sqrt(w) * sqrt(w) = w (correct!)
            import math
            sqrt_weight = math.sqrt(regime_weight)

            # Apply sqrt regime weight to size
            size_pct = base_size_pct * sqrt_weight

            # Apply confidence scaling
            size_pct *= signal.confidence

            # Apply regime risk budget cap
            size_pct, was_capped = self.regime_allocator.apply_regime_budget_cap(
                regime, size_pct
            )

            # Convert back to dollar amount
            position_size = portfolio_value * size_pct

            logger.info(
                f"Soft gating ({'PROBABILISTIC' if self.regime_mode == 'probabilistic' else 'DISCRETE'}) applied: "
                f"archetype={archetype_key}, regime={'blend' if self.regime_mode == 'probabilistic' else regime}, "
                f"regime_risk={regime_risk_pct*100:.1f}%, "
                f"base_size_pct={base_size_pct:.1%}, regime_weight={regime_weight:.2f}, "
                f"sqrt_weight={sqrt_weight:.3f}, confidence={signal.confidence:.2f}, "
                f"final_size_pct={size_pct:.1%}, position_size=${position_size:,.0f}, "
                f"budget_capped={was_capped}"
            )
        else:
            # No soft gating - use base size
            position_size = base_position_size

            logger.debug(
                f"Position sizing (no soft gating): portfolio=${portfolio_value:,.0f}, "
                f"regime={regime}, regime_risk={regime_risk_pct*100:.1f}%, "
                f"stop_dist={stop_distance_pct*100:.2f}%, "
                f"size=${position_size:,.0f}"
            )

        # Apply probabilistic regime multipliers if available
        if 'regime_result' in signal.metadata:
            soft_controls = signal.metadata['regime_result'].get('soft_controls', {})
            position_multiplier = soft_controls.get('position_size_multiplier', 1.0)

            # Scale position by multiplier
            original_size = position_size
            position_size *= position_multiplier

            logger.info(
                f"Probabilistic position scaling applied: "
                f"crisis_prob={signal.metadata['regime_result']['crisis_prob']:.3f}, "
                f"risk_temperature={signal.metadata['regime_result']['risk_temperature']:.3f}, "
                f"multiplier={position_multiplier:.3f}, "
                f"original=${original_size:,.0f}, "
                f"scaled=${position_size:,.0f}"
            )

        return position_size

    def _build_runtime_context(self, bar: pd.Series) -> RuntimeContext:
        """
        Build RuntimeContext for archetype detection.

        CRITICAL FIX: This method now enriches the bar with runtime-computed scores
        (liquidity_score, fusion_score) before passing to RuntimeContext, matching
        the production backtester's pattern. Without this, archetypes run "blind"
        because they can't access these essential signals.

        Args:
            bar: Current bar data from feature store

        Returns:
            RuntimeContext with enriched row, regime state, and thresholds
        """
        # STEP 1: Compute runtime scores that archetypes need
        # (Production backtester does this in _compute_fusion_score method)

        # Create enriched copy of bar with runtime scores
        row_with_runtime = bar.copy()

        # 1. Liquidity Score - Check if already in feature store, otherwise compute
        if 'liquidity_score' not in bar or pd.isna(bar.get('liquidity_score')):
            # Derive from BOMS/FVG components (matching production logic)
            boms_strength = bar.get('tf1d_boms_strength', 0.0)
            fvg_present = 1.0 if bar.get('tf4h_fvg_present', False) else 0.0

            # Normalize BOMS displacement to 0-1 range based on ATR
            boms_disp = bar.get('tf4h_boms_displacement', 0.0)
            atr = bar.get('atr_14', bar.get('atr', bar['close'] * 0.02))
            disp_normalized = min(boms_disp / (2.0 * atr), 1.0) if atr > 0 else 0.0

            liquidity_score = (boms_strength + fvg_present + disp_normalized) / 3.0
        else:
            liquidity_score = bar['liquidity_score']

        row_with_runtime['liquidity_score'] = liquidity_score

        # 2. Fusion Score - Compute weighted blend of domain scores
        # Wyckoff component
        wyckoff_m1 = 1.0 if bar.get('tf1d_m1_signal') is not None else 0.0
        wyckoff_m2 = 1.0 if bar.get('tf1d_m2_signal') is not None else 0.0
        wyckoff_score = (wyckoff_m1 + wyckoff_m2) / 2.0

        # Momentum component
        adx = bar.get('adx_14', 20.0) / 100.0
        rsi = bar.get('rsi_14', 50.0)
        rsi_momentum = abs(rsi - 50.0) / 50.0
        squiggle_conf = bar.get('tf4h_squiggle_confidence', 0.5)
        momentum_score = (adx + rsi_momentum + squiggle_conf) / 3.0

        # Macro component
        macro_regime = bar.get('macro_regime', self.default_regime)
        macro_vix = bar.get('macro_vix_level', 'medium')
        regime_map = {'risk_on': 1.0, 'neutral': 0.5, 'risk_off': 0.2, 'crisis': 0.0}
        regime_score = regime_map.get(macro_regime, 0.5)
        vix_map = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.2}
        vix_score = vix_map.get(macro_vix, 0.8)
        macro_score = (regime_score + vix_score) / 2.0

        # FRVP component
        frvp_poc_pos = bar.get('tf1h_frvp_poc_position', 'middle')
        poc_map = {'below': 0.3, 'at_poc': 1.0, 'above': 0.3, 'middle': 0.6}
        frvp_score = poc_map.get(frvp_poc_pos, 0.5)

        # PTI component (acts as penalty)
        pti_1d = bar.get('tf1d_pti_score', 0.0)
        pti_1h = bar.get('tf1h_pti_score', 0.0)
        pti_combined = max(pti_1d, pti_1h)

        # Weighted fusion calculation
        fusion_score = (
            0.30 * wyckoff_score +
            0.30 * liquidity_score +
            0.20 * momentum_score +
            0.10 * macro_score +
            0.10 * frvp_score
        )

        # Apply PTI penalty
        fusion_score -= 0.10 * pti_combined

        # Apply fakeout penalty
        if bar.get('tf1h_fakeout_detected', False):
            fusion_score -= 0.1

        # Apply governor veto
        if bar.get('mtf_governor_veto', False):
            fusion_score *= 0.3

        # Clip to [0, 1]
        fusion_score = max(0.0, min(1.0, fusion_score))

        row_with_runtime['fusion_score'] = fusion_score

        # STEP 2: Determine regime
        # Choose regime detection mode based on configuration
        regime_result = None  # Store full result for probabilistic mode

        if self.regime_mode == 'probabilistic' and self.regime_service:
            # Use RegimeService for probabilistic regime detection
            regime_result = self.regime_service.get_regime(bar.to_dict(), bar.name)
            regime_label = regime_result['regime_label']
            regime_probs = regime_result['regime_probs']

            logger.debug(
                f"[PROBABILISTIC] crisis_prob={regime_result['crisis_prob']:.3f}, "
                f"risk_temperature={regime_result['risk_temperature']:.3f}, "
                f"instability_score={regime_result['instability_score']:.3f}, "
                f"regime_label={regime_label}"
            )
        else:
            # Static mode: Get regime from data if available, otherwise use default
            if 'macro_regime' in bar.index:
                regime_label = bar['macro_regime']
            else:
                regime_label = self.default_regime

            # DRAWDOWN OVERRIDE: Force 'crisis' regime during severe drawdowns
            # This bypasses the hard regime filter in ArchetypeLogic.detect()
            # which would otherwise block archetypes like S1 even when drawdown override should apply
            if 'capitulation_depth' in bar.index:
                capitulation_depth = bar['capitulation_depth']
                # If drawdown > 15%, treat as crisis regardless of macro regime
                # This matches S1's internal drawdown_override_pct threshold (10%)
                # but we use 15% here to be more conservative at the wrapper level
                if capitulation_depth < -0.15:
                    logger.debug(
                        f"[WRAPPER] Drawdown override: capitulation_depth={capitulation_depth:.2%}, "
                        f"forcing regime='crisis' (was '{regime_label}')"
                    )
                    regime_label = 'crisis'

            regime_probs = {regime_label: 1.0}

        # STEP 3: Get adapted thresholds from ThresholdPolicy
        thresholds = self.threshold_policy.resolve(
            regime_probs=regime_probs,
            regime_label=regime_label
        )

        # STEP 4: Build RuntimeContext with ENRICHED row
        # This is the critical fix - we pass row_with_runtime instead of raw bar
        # For probabilistic mode, include regime_result in metadata
        context_metadata = {}
        if regime_result is not None:
            context_metadata['regime_result'] = regime_result

        # Get regime_config from loaded config (for soft controls flag)
        regime_config = self.full_config.get('regime_classifier', {})

        return RuntimeContext(
            ts=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            row=row_with_runtime,  # FIXED: Pass enriched row with runtime scores
            regime_probs=regime_probs,
            regime_label=regime_label,
            adapted_params={},  # Could add adaptive fusion params here
            thresholds=thresholds,
            regime_config=regime_config,  # FIXED: Pass regime config for soft controls
            metadata=context_metadata
        )

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters for logging/comparison.

        Returns:
            Dictionary of archetype parameters
        """
        return {
            'archetype_name': self.archetype_name,
            'direction': self.direction,
            'fusion_threshold': self.fusion_threshold,
            'atr_stop_mult': self.atr_stop_mult,
            'max_risk_pct': self.max_risk_pct,
            'config_path': str(self.config_path),
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get internal model state for debugging.

        Returns:
            Dictionary of internal state
        """
        base_state = super().get_state()
        base_state.update({
            'archetype_name': self.archetype_name,
            'default_regime': self.default_regime,
            'has_archetype_params': bool(self.archetype_params)
        })
        return base_state

    def set_regime(self, regime: str):
        """
        Manually set regime for testing purposes.

        Args:
            regime: Regime label ('risk_on', 'neutral', 'risk_off', 'crisis')
        """
        valid_regimes = ['risk_on', 'neutral', 'risk_off', 'crisis']
        if regime not in valid_regimes:
            raise ValueError(f"Invalid regime: {regime}. Must be one of {valid_regimes}")

        self.default_regime = regime
        logger.info(f"{self.name}: regime set to '{regime}'")
