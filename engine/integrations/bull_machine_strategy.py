"""
Bull Machine Strategy Adapter for Event-Driven Backtesting

Wraps Bull Machine decision logic (RegimeService + Archetypes + Domain Boosts + Soft Gating)
in an event-driven strategy interface compatible with the EventEngine.

Architecture:
    - EventEngine: Handles event loop, fills, OMS, portfolio accounting
    - BullMachineStrategy: Wraps Bull Machine decision logic
    - RegimeService: Brainstem (regime detection)
    - ArchetypeLogic: Pattern detection (Spring, Order Block, etc.)
    - Domain Boosts: Wyckoff/SMC/Temporal signals (already in archetype scores)
    - Soft Gating: Regime-conditioned position sizing
    - Circuit Breaker: Risk management

Key Design:
    - Preserves existing Bull Machine "soul" (decision logic)
    - Replaces plumbing (backtest mechanics) with production-grade event engine
    - NO changes to core engine files
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from pathlib import Path
import json

from engine.integrations.event_engine import (
    BaseStrategy, EventEngine, Bar, Order, OrderSide
)
from engine.context.regime_service import RegimeService
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext
from engine.archetypes.threshold_policy import ThresholdPolicy
from engine.portfolio.regime_allocator import RegimeWeightAllocator
import yaml

logger = logging.getLogger(__name__)


class BullMachineStrategy(BaseStrategy):
    """
    Bull Machine strategy adapter for event-driven backtesting.

    Preserves the engine "soul":
    - RegimeService (brainstem - regime detection)
    - ArchetypeLogic (pattern detection)
    - Domain boosts (Wyckoff/SMC/Temporal - already in scores)
    - Soft gating (regime-conditioned sizing)
    - Circuit breaker (risk management)

    Event-driven interface:
    - on_bar: Main signal generation logic
    - on_order_filled: Track fills
    - on_position_closed: Update circuit breaker
    """

    def __init__(
        self,
        config_path: str,
        feature_store_path: Optional[str] = None,
        archetype_name: Optional[str] = None,
        enable_soft_gating: bool = True,
        enable_circuit_breaker: bool = True,
        base_position_size_usd: float = 1000.0,
        max_positions: int = 1,
        feature_buffer_size: int = 200
    ):
        """
        Initialize Bull Machine strategy.

        Args:
            config_path: Path to config JSON (e.g., 'configs/s4_multi_objective_production.json')
            feature_store_path: Path to complete feature store parquet. If None, uses default.
            archetype_name: Optional - use only this archetype (e.g., 'S4'). If None, use all enabled.
            enable_soft_gating: Apply regime-conditioned position sizing
            enable_circuit_breaker: Enable circuit breaker risk management
            base_position_size_usd: Base position size in USD
            max_positions: Maximum concurrent positions
            feature_buffer_size: Number of bars to keep in feature buffer
        """
        super().__init__(name=f"BullMachine-{archetype_name or 'Multi'}")

        self.config_path = Path(config_path)
        self.archetype_name = archetype_name
        self.enable_soft_gating = enable_soft_gating
        self.enable_circuit_breaker = enable_circuit_breaker
        self.base_position_size_usd = base_position_size_usd
        self.max_positions = max_positions
        self.feature_buffer_size = feature_buffer_size

        # Position tracking for exit logic
        self.open_positions: Dict[str, Dict[str, Any]] = {}  # position_id -> position_state
        self.exit_logic = None  # Initialized in _initialize_components

        # Feature store path
        if feature_store_path is None:
            feature_store_path = 'data/features_mtf/BTC_1H_LATEST.parquet'  # Updated to use latest fixed feature store
        self.feature_store_path = Path(feature_store_path)

        # Load configuration
        self._load_config()

        # Load feature store (CRITICAL: Real features, not placeholders)
        self._load_feature_store()

        # Load archetype directions from registry
        self._load_archetype_directions()

        # Initialize Bull Machine components (SOUL)
        self._initialize_components()

        # Feature buffer (for indicators that need history)
        self.feature_buffer: List[Dict[str, Any]] = []

        # Circuit breaker state
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        self.circuit_breaker_threshold = 3

        # Performance tracking
        self.signals_generated = 0
        self.signals_rejected = 0
        self.rejection_reasons: Dict[str, int] = {}

        self.logger.info(f"Initialized {self.name}")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Feature store: {self.feature_store_path} ({len(self.features_df):,} bars, {len(self.features_df.columns)} columns)")
        self.logger.info(f"Soft gating: {enable_soft_gating}, Circuit breaker: {enable_circuit_breaker}")
        self.logger.info(f"Base position: ${base_position_size_usd}, Max positions: {max_positions}")

    def _load_config(self):
        """Load configuration from JSON."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.full_config = json.load(f)

        self.logger.info(f"Loaded config: {self.full_config.get('version', 'unknown')}")

    def _load_feature_store(self):
        """Load complete feature store for real feature computation."""
        if not self.feature_store_path.exists():
            raise FileNotFoundError(
                f"Feature store not found: {self.feature_store_path}\n"
                f"Run: python3 bin/build_complete_feature_store.py"
            )

        self.logger.info(f"Loading feature store: {self.feature_store_path}")
        self.features_df = pd.read_parquet(self.feature_store_path)

        # Ensure datetime index
        if not isinstance(self.features_df.index, pd.DatetimeIndex):
            self.features_df.index = pd.to_datetime(self.features_df.index)

        # Sort by time
        self.features_df = self.features_df.sort_index()

        self.logger.info(
            f"Feature store loaded: {len(self.features_df):,} bars, "
            f"{len(self.features_df.columns)} columns, "
            f"{self.features_df.index.min()} to {self.features_df.index.max()}"
        )

    def _load_archetype_directions(self):
        """Load archetype direction metadata from registry."""
        registry_path = Path('archetype_registry.yaml')

        if not registry_path.exists():
            self.logger.warning("Archetype registry not found - using default direction=long for all")
            self.archetype_directions = {}
            return

        try:
            with open(registry_path, 'r') as f:
                registry = yaml.safe_load(f)

            self.archetype_directions = {}
            for archetype in registry.get('archetypes', []):
                arch_id = archetype.get('id')
                direction = archetype.get('direction', 'long')
                if arch_id:
                    self.archetype_directions[arch_id] = direction

            self.logger.info(
                f"Loaded archetype directions from registry: {len(self.archetype_directions)} archetypes"
            )
        except Exception as e:
            self.logger.error(f"Failed to load archetype registry: {e}")
            self.archetype_directions = {}

    def _initialize_components(self):
        """Initialize Bull Machine components (SOUL)."""
        # 1. RegimeService (BRAINSTEM)
        regime_config = self.full_config.get('regime_classifier', {})
        self.regime_service = RegimeService(
            mode='hybrid',  # Use hybrid mode (crisis rules + ML for normal regimes)
            model_path=regime_config.get('model_path'),
            enable_event_override=True,
            enable_hysteresis=True
        )
        self.logger.info("RegimeService initialized (hybrid mode)")

        # 2. ArchetypeLogic (PATTERN DETECTION)
        archetype_config = self.full_config.get('archetypes', {})
        self.archetype_logic = ArchetypeLogic(archetype_config)
        self.logger.info("ArchetypeLogic initialized")

        # 3. ThresholdPolicy (REGIME-AWARE THRESHOLDS)
        self.threshold_policy = ThresholdPolicy(
            base_cfg=self.full_config,
            regime_profiles=self.full_config.get('gates_regime_profiles'),
            archetype_overrides=self.full_config.get('archetype_overrides'),
            global_clamps=self.full_config.get('global_clamps'),
            locked_regime='static'
        )
        self.logger.info("ThresholdPolicy initialized")

        # 4. RegimeWeightAllocator (SOFT GATING)
        if self.enable_soft_gating:
            try:
                # Try to load edge table path from config
                allocator_config_path = 'configs/regime_allocator_config.json'
                edge_table_path = 'results/archetype_regime_edge_table.csv'

                # Check if config exists and has edge_table_path
                if Path(allocator_config_path).exists():
                    import json
                    with open(allocator_config_path, 'r') as f:
                        allocator_cfg = json.load(f)
                    edge_table_path = allocator_cfg.get('edge_table_path', edge_table_path)

                # Check if edge table exists
                if not Path(edge_table_path).exists():
                    self.logger.warning(
                        f"Edge table not found at {edge_table_path}. "
                        f"Soft gating disabled. Run edge computation script to enable."
                    )
                    self.regime_allocator = None
                else:
                    self.regime_allocator = RegimeWeightAllocator(
                        edge_table_path=edge_table_path,
                        config_path=allocator_config_path
                    )
                    self.logger.info("RegimeWeightAllocator initialized (soft gating enabled)")
            except Exception as e:
                self.logger.warning(f"Could not load RegimeWeightAllocator: {e}")
                self.regime_allocator = None
        else:
            self.regime_allocator = None

        # 5. ExitLogic (ARCHETYPE-SPECIFIC EXITS)
        from engine.archetypes.exit_logic import ExitLogic, create_default_exit_config

        self.logger.info("Initializing ExitLogic...")
        exit_config = create_default_exit_config()
        if 'exit_logic' in self.full_config:
            exit_config.update(self.full_config['exit_logic'])
        self.exit_logic = ExitLogic(exit_config)
        self.logger.info("✓ ExitLogic initialized (archetype-specific exit rules active)")

    def on_start(self, engine: EventEngine):
        """Called when backtest starts."""
        self.logger.info(f"{self.name} started")
        self.logger.info(f"Backtesting {len(engine.bars)} bars")

    def on_bar(self, bar: Bar, engine: EventEngine):
        """
        Main signal generation logic (called every bar).

        Flow:
        1. Build features from bar
        2. Get regime (RegimeService)
        3. Check exits for open positions (EXIT FIRST)
        4. Evaluate archetypes (pattern detection)
        5. Apply domain boosts (already in archetype score)
        6. Soft gating (regime-conditioned sizing)
        7. Circuit breaker check
        8. Submit order (EventEngine handles execution)
        """
        # 1. Add bar to feature buffer
        self._add_bar_to_buffer(bar)

        # Need enough data for indicators
        if len(self.feature_buffer) < 50:
            return

        # 2. Get regime state (BRAINSTEM)
        features = self._compute_features()
        regime_result = self.regime_service.get_regime(features, bar.timestamp)

        regime_label = regime_result['regime_label']
        regime_confidence = regime_result.get('regime_confidence', 0.5)
        regime_probs = regime_result.get('regime_probs', {})

        # 3. Build RuntimeContext
        context = self._build_runtime_context(
            features=features,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            regime_probs=regime_probs,
            bar=bar
        )

        # 3.5. Check exits for all open positions (EXIT FIRST, BEFORE ENTRIES)
        for position_id in list(self.open_positions.keys()):
            if position_id in engine.portfolio.positions:
                self._check_exit_conditions(bar, engine, context, regime_label, position_id)

        # 4. Evaluate archetypes (PATTERN DETECTION)
        archetype_result = self._evaluate_archetypes(context)

        if not archetype_result:
            return  # No signal

        matched, score, meta, direction = archetype_result
        archetype_name = meta.get('archetype', 'unknown')

        self.signals_generated += 1

        # 5. Soft gating (REGIME-CONDITIONED SIZING)
        position_size_usd = self._compute_position_size(
            base_score=score,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            archetype_name=archetype_name
        )

        # 6. Circuit breaker check
        if self._circuit_breaker_check(regime_label):
            self.signals_rejected += 1
            self._track_rejection("circuit_breaker")
            self.logger.warning(f"Circuit breaker active - rejecting {archetype_name} signal")
            return

        # 7. Check max positions
        if len(engine.portfolio.positions) >= self.max_positions:
            self.signals_rejected += 1
            self._track_rejection("max_positions")
            return

        # 8. Submit order (EVENT ENGINE HANDLES EXECUTION)
        side = OrderSide.BUY if direction == "long" else OrderSide.SELL

        # Generate unique position ID
        position_id = f"{direction.lower()}_{archetype_name}_{int(bar.timestamp.timestamp())}"

        # Pre-populate position tracking (will be confirmed in on_order_filled)
        self.open_positions[position_id] = {
            'entry_price': bar.close,  # Will be updated with actual fill price
            'entry_time': bar.timestamp,
            'archetype': archetype_name,
            'direction': direction,
            'regime_at_entry': regime_label,
            'fusion_score': score,
            'initial_stop': None,
            'trailing_stop': None,
            'executed_exit_types': set()
        }

        order = engine.submit_order(side, position_size_usd, position_id=position_id)

        if order:
            self.logger.info(
                f"Signal: {archetype_name} {direction.upper()} @ ${bar.close:.2f} | "
                f"Regime: {regime_label} ({regime_confidence:.2f}) | "
                f"Score: {score:.2f} | Size: ${position_size_usd:.2f}"
            )
        else:
            # Order rejected (insufficient cash) - remove from tracking
            del self.open_positions[position_id]
            self.signals_rejected += 1
            self._track_rejection("insufficient_cash")

    def _add_bar_to_buffer(self, bar: Bar):
        """Add bar to feature buffer."""
        self.feature_buffer.append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

        # Maintain buffer size
        if len(self.feature_buffer) > self.feature_buffer_size:
            self.feature_buffer.pop(0)

    def _compute_features(self) -> Dict[str, Any]:
        """
        Lookup features from pre-loaded feature store.

        Returns dict of all features for current timestamp.
        Raises KeyError if timestamp not in feature store.
        """
        if len(self.feature_buffer) == 0:
            raise ValueError("Feature buffer is empty - call _add_bar_to_buffer first")

        # Get current timestamp from latest bar in buffer
        timestamp = self.feature_buffer[-1]['timestamp']

        # Convert to pandas Timestamp (handle both int ns and datetime)
        if isinstance(timestamp, (int, float)):
            ts = pd.Timestamp(timestamp, unit='ns', tz='UTC')
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')

        # Lookup features from pre-loaded store
        try:
            row = self.features_df.loc[ts]
        except KeyError:
            # Timestamp not in feature store - find nearest
            self.logger.warning(f"Timestamp {ts} not in feature store, finding nearest")

            # Find nearest timestamp within 1 hour
            time_diffs = abs(self.features_df.index - ts)
            nearest_idx = time_diffs.argmin()
            nearest_ts = self.features_df.index[nearest_idx]

            if time_diffs.iloc[nearest_idx] > pd.Timedelta('1H'):
                raise KeyError(
                    f"No feature store entry within 1H of {ts}\n"
                    f"Nearest: {nearest_ts} ({time_diffs.iloc[nearest_idx]} away)\n"
                    f"Feature store range: {self.features_df.index.min()} to {self.features_df.index.max()}"
                )

            row = self.features_df.iloc[nearest_idx]
            self.logger.debug(f"Using nearest timestamp: {nearest_ts}")

        # Convert row to dict (all features)
        features = row.to_dict()

        # Add OHLCV from current bar (in case not in feature store)
        features.update({
            'close': self.feature_buffer[-1]['close'],
            'high': self.feature_buffer[-1]['high'],
            'low': self.feature_buffer[-1]['low'],
            'open': self.feature_buffer[-1]['open'],
            'volume': self.feature_buffer[-1]['volume'],
        })

        return features

    def _build_runtime_context(
        self,
        features: Dict[str, Any],
        regime_label: str,
        regime_confidence: float,
        regime_probs: Dict[str, float],
        bar: Bar
    ) -> RuntimeContext:
        """Build RuntimeContext for archetype evaluation."""
        # Convert features dict to Series
        row = pd.Series(features)

        # Get adapted parameters (placeholder - would use AdaptiveFusion in production)
        adapted_params = {
            'gates': {},
            'fusion_weights': {},
            'ml_threshold': 0.5,
        }

        # Get thresholds from ThresholdPolicy
        thresholds = self.threshold_policy.get_all_thresholds(regime_label)

        # Get regime_config from loaded config (for soft controls flag)
        regime_config = self.full_config.get('regime_classifier', {})

        # Build context
        context = RuntimeContext(
            ts=bar.timestamp,
            row=row,
            regime_probs=regime_probs,
            regime_label=regime_label,
            adapted_params=adapted_params,
            thresholds=thresholds,
            regime_config=regime_config,  # FIXED: Pass regime config for soft controls
            metadata={
                'regime_confidence': regime_confidence,
            }
        )

        return context

    def _evaluate_archetypes(self, context: RuntimeContext) -> Optional[tuple]:
        """
        Evaluate archetypes in priority order.

        Returns:
            (matched, score, metadata, direction) if signal found, None otherwise
        """
        # If specific archetype requested, use only that one
        if self.archetype_name:
            archetype_methods = [self._get_archetype_method(self.archetype_name)]
        else:
            # Try all enabled archetypes in priority order
            archetype_methods = [
                ('A', self.archetype_logic._check_A),  # Spring/UTAD
                ('B', self.archetype_logic._check_B),  # Order Block
                ('C', self.archetype_logic._check_C),  # BOS/CHOCH
                ('D', self.archetype_logic._check_D),  # Failed Continuation
                ('E', self.archetype_logic._check_E),  # Volume Exhaustion
                ('F', self.archetype_logic._check_F),  # Exhaustion Reversal
                ('G', self.archetype_logic._check_G),  # Liquidity Sweep
                ('H', self.archetype_logic._check_H),  # Momentum Continuation
                ('K', self.archetype_logic._check_K),  # Trap Within Trend
                ('S1', self.archetype_logic._check_S1),  # Liquidity Vacuum
                ('S4', self.archetype_logic._check_S4),  # Funding Divergence
                ('S5', self.archetype_logic._check_S5),  # Long Squeeze
            ]

        for name, method in archetype_methods:
            if method is None:
                continue

            try:
                result = method(context)

                # Handle both 3-tuple (matched, score, meta) and 4-tuple (matched, score, meta, direction) returns
                if len(result) == 3:
                    matched, score, meta = result
                    # Get direction from registry, default to 'long'
                    direction = self.archetype_directions.get(name, 'long')
                elif len(result) == 4:
                    matched, score, meta, direction = result
                else:
                    self.logger.error(f"Archetype {name} returned unexpected tuple length: {len(result)}")
                    continue

                if matched:
                    # Add archetype name to metadata
                    meta['archetype'] = name
                    return (matched, score, meta, direction)
            except Exception as e:
                self.logger.error(f"Archetype {name} error: {e}", exc_info=True)
                continue

        return None

    def _get_archetype_method(self, archetype_name: str):
        """Get archetype check method by name."""
        method_map = {
            'A': self.archetype_logic._check_A,
            'B': self.archetype_logic._check_B,
            'C': self.archetype_logic._check_C,
            'D': self.archetype_logic._check_D,
            'E': self.archetype_logic._check_E,
            'F': self.archetype_logic._check_F,
            'G': self.archetype_logic._check_G,
            'H': self.archetype_logic._check_H,
            'K': self.archetype_logic._check_K,
            'S1': self.archetype_logic._check_S1,
            'S4': self.archetype_logic._check_S4,
            'S5': self.archetype_logic._check_S5,
        }
        return (archetype_name, method_map.get(archetype_name))

    def _compute_position_size(
        self,
        base_score: float,
        regime_label: str,
        regime_confidence: float,
        archetype_name: str
    ) -> float:
        """
        Apply soft gating to position size.

        Combines:
        1. Base score (archetype confidence)
        2. Regime multiplier (crisis = 0.3x, risk_off = 0.6x, neutral = 0.8x, risk_on = 1.0x)
        3. Regime confidence scaling
        4. Optional: RegimeWeightAllocator for dynamic weights
        """
        # 1. Start with base size
        size = self.base_position_size_usd

        # 2. Apply score scaling
        size *= base_score

        # 3. Apply regime multiplier (SOFT GATING)
        if self.enable_soft_gating:
            regime_multiplier = {
                'crisis': 0.3,
                'risk_off': 0.6,
                'neutral': 0.8,
                'risk_on': 1.0
            }.get(regime_label, 0.5)

            size *= regime_multiplier

            # 4. Apply confidence scaling
            confidence_multiplier = 0.5 + (regime_confidence * 0.5)
            size *= confidence_multiplier

            # 5. Optional: Use RegimeWeightAllocator
            if self.regime_allocator:
                try:
                    allocator_weight = self.regime_allocator.get_weight(
                        archetype=archetype_name,
                        regime=regime_label
                    )
                    size *= allocator_weight
                except Exception as e:
                    self.logger.warning(f"RegimeWeightAllocator error: {e}")

        # 6. Enforce minimum size
        size = max(size, 100.0)  # Min $100 position

        return size

    def _circuit_breaker_check(self, regime_label: str) -> bool:
        """
        Check if circuit breaker should halt trading.

        Logic:
        - Halt in crisis if 3+ consecutive losses
        - Halt in any regime if 5+ consecutive losses
        """
        if not self.enable_circuit_breaker:
            return False

        # Crisis threshold: 3 losses
        if regime_label == 'crisis' and self.consecutive_losses >= 3:
            self.circuit_breaker_active = True
            return True

        # General threshold: 5 losses
        if self.consecutive_losses >= 5:
            self.circuit_breaker_active = True
            return True

        return False

    def _track_rejection(self, reason: str):
        """Track rejection reason."""
        if reason not in self.rejection_reasons:
            self.rejection_reasons[reason] = 0
        self.rejection_reasons[reason] += 1

    def on_order_filled(self, order: Order, engine: EventEngine):
        """Called when an order is filled."""
        self.logger.info(f"Order filled: {order}")

        # Track position state for exit logic
        if order.position_id and not order.is_exit:
            # This is an entry order - update with actual fill price
            if order.position_id in self.open_positions:
                # Update entry price with actual fill price (may differ from close due to slippage)
                self.open_positions[order.position_id]['entry_price'] = order.fill_price
                self.logger.info(f"Position entry confirmed: {order.position_id} @ ${order.fill_price:.2f}")
            else:
                # Shouldn't happen, but handle just in case
                self.logger.warning(f"Order filled for untracked position: {order.position_id}")
        elif order.is_exit and order.position_id in self.open_positions:
            # This is an exit order - check if position fully closed
            if order.position_id not in engine.portfolio.positions:
                # Position fully closed, remove from tracking
                del self.open_positions[order.position_id]
                self.logger.info(f"Position closed: {order.position_id}")

    def _check_exit_conditions(self, bar: Bar, engine: EventEngine, context: RuntimeContext, regime_label: str, position_id: str):
        """
        Check comprehensive exit conditions using ExitLogic system.

        This checks:
        1. Invalidation exits (pattern breaks)
        2. Profit target exits (R-multiple scale-outs)
        3. Time-based exits (max hold period)
        4. Reason-gone exits (entry condition reverses)
        5. Trailing stops (profit protection)
        """
        if position_id not in self.open_positions:
            return

        if position_id not in engine.portfolio.positions:
            return

        position_state = self.open_positions[position_id]
        engine_position = engine.portfolio.positions[position_id]

        # Import Position here to avoid circular dependency
        from engine.models.base import Position

        # Get archetype from position state
        archetype = position_state.get('archetype', 'unknown')

        # Calculate position size in USD
        position_size_usd = engine_position.quantity * bar.close

        # Create Position object for ExitLogic
        # Calculate initial stop loss (2 ATR below entry for longs)
        atr = context.row.get('atr_14')
        if atr is None or atr == 0 or pd.isna(atr):
            atr = 0.02 * bar.close  # Default to 2% if ATR missing

        initial_stop = position_state.get('initial_stop')
        if initial_stop is None or pd.isna(initial_stop):
            if position_state.get('direction') == 'long':
                initial_stop = position_state['entry_price'] - (2 * atr)
            else:
                initial_stop = position_state['entry_price'] + (2 * atr)
            position_state['initial_stop'] = initial_stop

        # Use trailing stop if set, otherwise initial stop
        current_stop = position_state.get('trailing_stop', initial_stop)
        if current_stop is None or pd.isna(current_stop):
            current_stop = initial_stop

        position = Position(
            direction=position_state.get('direction', 'long'),
            entry_price=position_state['entry_price'],
            entry_time=position_state['entry_time'],
            size=position_size_usd,
            stop_loss=current_stop,
            take_profit=None,
            regime_label=position_state.get('regime_at_entry', 'neutral'),
            metadata={
                'archetype': archetype,
                'fusion_score': position_state.get('fusion_score', 0.5),
                'initial_stop': initial_stop,
                'executed_exit_types': position_state.get('executed_exit_types', set())
            }
        )

        # Create bar Series with timestamp as name (exit_logic uses bar.name)
        bar_series = pd.Series({
            'close': bar.close,
            'high': bar.high,
            'low': bar.low,
            'open': bar.open,
            'volume': bar.volume
        }, name=bar.timestamp)

        # Check exit logic
        exit_signal = self.exit_logic.check_exit(
            bar=bar_series,
            position=position,
            archetype=archetype,
            context=context
        )

        if exit_signal and exit_signal.exit_type != 'none':
            # Calculate exit quantity
            exit_quantity = engine_position.quantity * exit_signal.exit_pct

            # Submit exit order
            side = OrderSide.SELL if position_state.get('direction') == 'long' else OrderSide.BUY
            exit_size_usd = exit_quantity * bar.close

            self.logger.info(
                f"[EXIT] {exit_signal.exit_type} - {archetype} @ ${bar.close:.2f} | "
                f"Exit {exit_signal.exit_pct*100:.0f}% | "
                f"Reason: {exit_signal.reason}"
            )

            # Submit exit order with is_exit flag
            engine.submit_order(side, exit_size_usd, position_id=position_id, is_exit=True)

            # Track executed exit type
            position_state['executed_exit_types'].add(exit_signal.exit_type)

            # Update stop if provided
            if exit_signal.stop_update:
                position_state['trailing_stop'] = exit_signal.stop_update

    def on_stop(self, engine: EventEngine):
        """Called when backtest ends."""
        stats = engine.get_performance_stats()

        self.logger.info("=" * 80)
        self.logger.info(f"{self.name} - BACKTEST COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Signals generated: {self.signals_generated}")
        self.logger.info(f"Signals rejected: {self.signals_rejected}")
        if self.rejection_reasons:
            self.logger.info(f"Rejection reasons: {self.rejection_reasons}")
        self.logger.info(f"Total trades: {stats['total_trades']}")
        self.logger.info(f"Win rate: {stats['win_rate']:.2f}%")
        self.logger.info(f"Profit factor: {stats['profit_factor']:.2f}")
        self.logger.info(f"Total PnL: ${stats['total_pnl']:,.2f}")
        self.logger.info(f"Total return: {stats['total_return']:.2f}%")
        self.logger.info(f"Max drawdown: {stats['max_drawdown']:.2f}%")
        self.logger.info(f"Sharpe ratio: {stats['sharpe_ratio']:.2f}")
        self.logger.info("=" * 80)

        # Update circuit breaker based on final trades
        if len(engine.portfolio.trades) > 0:
            last_trades = engine.portfolio.trades[-5:]
            consecutive_losses = 0
            for trade in reversed(last_trades):
                if trade.realized_pnl < 0:
                    consecutive_losses += 1
                else:
                    break
            self.consecutive_losses = consecutive_losses
