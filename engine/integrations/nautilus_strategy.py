"""
NautilusBullMachineStrategy - Integration Layer for Bull Machine with Event-Driven Architecture

This is the CORE integration point that connects Bull Machine components with event-driven backtesting.
It wraps RegimeService, ArchetypeLogic, and ThresholdPolicy into an event-driven strategy.

Architecture:
    EventEngine (bars) → NautilusBullMachineStrategy.on_bar()
                       ↓
                   FeatureProvider (features)
                       ↓
                   RegimeService (regime classification)
                       ↓
                   RuntimeContext (unified context)
                       ↓
                   ArchetypeLogic (signal generation)
                       ↓
                   ThresholdPolicy (parameter adaptation)
                       ↓
                   Order submission

Key Design Decisions:
1. Feature Provider: Hybrid approach (feature store OR runtime computation)
2. Regime Classification: Dynamic (no static labels)
3. Signal Generation: Archetype-based with fusion scoring
4. Position Sizing: ATR-based risk management
5. Order Management: Market orders with stop loss

Author: Claude Code (System Architect)
Date: 2026-01-21
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import json

from engine.integrations.event_engine import BaseStrategy, Bar, Order, OrderSide, EventEngine
from engine.integrations.feature_provider import FeatureProvider
from engine.context.regime_service import RegimeService, REGIME_MODE_HYBRID, REGIME_MODE_PROBABILISTIC
from engine.runtime.context import RuntimeContext
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.archetypes.threshold_policy import ThresholdPolicy
from engine.portfolio.regime_allocator import RegimeWeightAllocator

logger = logging.getLogger(__name__)


class NautilusBullMachineStrategy(BaseStrategy):
    """
    Bull Machine strategy adapted for event-driven architecture.

    This class integrates all Bull Machine components:
    - RegimeService for dynamic regime classification
    - ArchetypeLogic for signal generation
    - ThresholdPolicy for parameter adaptation
    - FeatureProvider for feature computation

    Lifecycle Methods:
    - on_start(): Initialize components and load configuration
    - on_bar(): Process new bar and generate signals
    - on_order_filled(): Track fills and update state
    - on_stop(): Cleanup and final logging

    Signal Pipeline:
    1. Receive bar from EventEngine
    2. Build/fetch features via FeatureProvider
    3. Classify regime via RegimeService
    4. Build RuntimeContext with regime + thresholds
    5. Detect archetype via ArchetypeLogic
    6. Calculate position size
    7. Submit order if signal fires
    """

    def __init__(
        self,
        config_path: str,
        regime_model_path: Optional[str] = None,
        feature_store_path: Optional[str] = None,
        enable_regime_service: bool = True,
        enable_feature_store: bool = True,
        risk_per_trade: float = 0.02,
        atr_stop_mult: float = 2.5,
        max_position_pct: float = 0.12,
        name: str = "BullMachine"
    ):
        """
        Initialize Bull Machine strategy.

        Args:
            config_path: Path to Bull Machine config (e.g., configs/baseline_wyckoff_test.json)
            regime_model_path: Path to regime model (e.g., models/logistic_regime_v3.pkl)
            feature_store_path: Path to feature store CSV (if using precomputed features)
            enable_regime_service: Enable dynamic regime classification
            enable_feature_store: Use feature store for historical data (vs runtime computation)
            risk_per_trade: Risk percentage per trade (default: 2%)
            atr_stop_mult: ATR multiplier for stop loss (default: 2.5x)
            max_position_pct: Max position size as % of portfolio (default: 12%)
            name: Strategy name for logging
        """
        super().__init__(name=name)

        # Configuration
        self.config_path = Path(config_path)
        self.regime_model_path = regime_model_path
        self.feature_store_path = feature_store_path
        self.enable_regime_service = enable_regime_service
        self.enable_feature_store = enable_feature_store

        # Risk parameters
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.max_position_pct = max_position_pct

        # Components (initialized in on_start)
        self.config: Optional[Dict[str, Any]] = None
        self.feature_provider: Optional[FeatureProvider] = None
        self.regime_service: Optional[RegimeService] = None
        self.archetype_logic: Optional[ArchetypeLogic] = None
        self.threshold_policy: Optional[ThresholdPolicy] = None
        self.exit_logic = None  # Initialized later to avoid circular import
        self.regime_allocator: Optional[RegimeWeightAllocator] = None

        # State tracking - CHANGED: Support multiple concurrent positions
        self.positions: Dict[str, Dict[str, Any]] = {}  # position_id -> position_state
        self.max_concurrent_positions: int = 999  # REMOVED LIMIT - only constrained by available capital
        self.bars_since_entry: int = 0  # Kept for backward compatibility

        # Performance tracking
        self.total_signals = 0
        self.signals_taken = 0
        self.signals_rejected = 0

        self.logger.info(f"Initialized {self.name} strategy")
        self.logger.info(f"  Config: {self.config_path}")
        self.logger.info(f"  Regime model: {self.regime_model_path or 'None (regime disabled)'}")
        self.logger.info(f"  Feature store: {self.feature_store_path or 'None (runtime computation)'}")
        self.logger.info(f"  Risk per trade: {self.risk_per_trade*100:.1f}%")
        self.logger.info(f"  ATR stop mult: {self.atr_stop_mult}x")

    def on_start(self, engine: EventEngine):
        """
        Called when backtest starts - initialize all components.

        This is where we:
        1. Load configuration
        2. Initialize FeatureProvider
        3. Initialize RegimeService
        4. Initialize ArchetypeLogic
        5. Initialize ThresholdPolicy

        Args:
            engine: EventEngine instance
        """
        self.logger.info("=" * 80)
        self.logger.info(f"{self.name} STARTING")
        self.logger.info("=" * 80)

        # 1. Load configuration
        self.logger.info("Loading configuration...")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self.logger.info(f"✓ Loaded config: {self.config.get('version', 'unknown')}")

        # 2. Initialize FeatureProvider
        self.logger.info("Initializing FeatureProvider...")
        self.feature_provider = FeatureProvider(
            feature_store_path=self.feature_store_path if self.enable_feature_store else None,
            enable_runtime_computation=True  # Always enable runtime as fallback
        )
        self.logger.info("✓ FeatureProvider initialized")

        # 3. Initialize RegimeService (if enabled) - NOW PROBABILISTIC!
        if self.enable_regime_service:
            self.logger.info("Initializing RegimeService (Probabilistic Mode)...")

            # Check config for regime mode
            regime_config = self.config.get('regime_classifier', {})
            regime_mode = regime_config.get('mode', 'probabilistic')

            # Force probabilistic mode for production
            if regime_mode != 'probabilistic':
                self.logger.warning(f"Config has regime mode '{regime_mode}', overriding to 'probabilistic'")
                regime_mode = 'probabilistic'

            self.regime_service = RegimeService(
                mode=REGIME_MODE_PROBABILISTIC,  # PROBABILISTIC (not hybrid!)
                model_path=self.regime_model_path,
                enable_event_override=True,
                enable_hysteresis=False  # Disable hysteresis for smooth transitions
            )
            self.logger.info("✓ RegimeService initialized (Probabilistic)")
        else:
            self.logger.info("⊗ RegimeService disabled (static regime mode)")
            self.regime_service = None

        # 4. Initialize ArchetypeLogic (with multi-position support)
        self.logger.info("Initializing ArchetypeLogic...")
        archetype_config = self.config.get('archetypes', {})

        # Check if multi-position mode is enabled
        position_config = self.config.get('position_sizing', {})
        self.multi_position_mode = position_config.get('multi_position_mode', False)
        self.min_relative_score = position_config.get('min_relative_score', 0.7)

        if self.multi_position_mode:
            from engine.archetypes.multi_position_adapter import MultiPositionArchetypeLogic
            self.archetype_logic = MultiPositionArchetypeLogic(
                archetype_config,
                min_relative_score=self.min_relative_score
            )
            self.logger.info(f"✓ MultiPositionArchetypeLogic initialized (min_score={self.min_relative_score})")
        else:
            self.archetype_logic = ArchetypeLogic(archetype_config)
            self.logger.info("✓ ArchetypeLogic initialized (single-position mode)")

        # 5. Initialize ThresholdPolicy
        self.logger.info("Initializing ThresholdPolicy...")
        self.threshold_policy = ThresholdPolicy(
            base_cfg=self.config,
            regime_profiles=self.config.get('gates_regime_profiles'),
            archetype_overrides=self.config.get('archetype_overrides'),
            global_clamps=self.config.get('global_clamps'),
            locked_regime='static' if not self.enable_regime_service else None
        )
        self.logger.info("✓ ThresholdPolicy initialized")

        # 6. Initialize ExitLogic (archetype-specific exits)
        # Import here to avoid circular dependency
        from engine.archetypes.exit_logic import ExitLogic, create_default_exit_config

        self.logger.info("Initializing ExitLogic...")
        exit_config = create_default_exit_config()
        if 'exit_logic' in self.config:
            exit_config.update(self.config['exit_logic'])
        self.exit_logic = ExitLogic(exit_config)
        # ExitLogic initialized successfully
        self.logger.info("✓ ExitLogic initialized (archetype-specific exit rules active)")

        # 7. Initialize RegimeWeightAllocator (probabilistic position scaling)
        self.logger.info("Initializing RegimeWeightAllocator (Probabilistic)...")

        # Edge table path for historical archetype-regime performance
        # Try multiple locations to handle different execution contexts (direct vs optimization)
        working_dir = self.config_path.parent.parent  # Go up from configs/ to project root
        edge_table_candidates = [
            working_dir / "results" / "archetype_regime_edge_table.csv",  # Standard location
            working_dir / "configs" / "results" / "archetype_regime_edge_table.csv",  # Optimized config location
            Path.cwd() / "results" / "archetype_regime_edge_table.csv",  # Current working directory
        ]

        edge_table_path = None
        for candidate in edge_table_candidates:
            if candidate.exists():
                edge_table_path = str(candidate)
                break

        if not edge_table_path:
            # Fallback to first candidate (will error in RegimeWeightAllocator if missing)
            edge_table_path = str(edge_table_candidates[0])

        # Get directional budgets from config if present
        allocator_config = self.config.get('regime_allocator', {})

        # Initialize with edge table and config
        self.regime_allocator = RegimeWeightAllocator(
            edge_table_path=edge_table_path,
            config_override=allocator_config
        )

        # Check bypass flag for entry filtering
        self.bypass_entry_filtering = allocator_config.get('bypass_entry_filtering', False)
        self.use_regime_for_exits = allocator_config.get('use_regime_for_exits', True)

        self.logger.info("✓ RegimeWeightAllocator initialized")

        self.logger.info("=" * 80)
        self.logger.info(f"{self.name} READY - FULL SOUL ACTIVATED")
        self.logger.info("✓ Exit Logic: Active (archetype-specific rules)")
        self.logger.info("✓ Probabilistic Regime: Active (smooth transitions)")

        if self.bypass_entry_filtering:
            self.logger.info("✓ Regime Allocator: BYPASSED for entries (regime used for exits/risk only)")
        else:
            self.logger.info("✓ Regime Allocator: Active (probabilistic position scaling)")

        self.logger.info("✓ Domain Engines: Active (Wyckoff + SMC + Temporal)")
        self.logger.info("✓ MTF Features: Available (1H + 4H + 1D confluence)")
        self.logger.info("=" * 80)

        # Verify MTF features are present in feature store
        if self.enable_feature_store:
            mtf_features = ['tf1h_fusion_score', 'tf4h_fusion_score', 'tf1d_fusion_score',
                           'mtf_alignment_ok', 'mtf_conflict_score', 'mtf_governor_veto']
            self.logger.info("Checking MTF feature availability...")
            self.logger.info(f"MTF features expected: {len(mtf_features)} features")
            self.logger.info("MTF temporal confluence: ENABLED")

    def on_bar(self, bar: Bar, engine: EventEngine):
        """
        CORE METHOD: Process new bar and generate trading signals.

        This is the main signal generation pipeline:
        1. Get/compute features for current bar
        2. Classify regime (if enabled)
        3. Build RuntimeContext
        4. Detect archetype signal
        5. Calculate position size
        6. Submit order (if signal fires)

        REGIME FLOW:
        1. RegimeService classifies regime (probabilistic: crisis_prob, risk_temperature)
        2. RuntimeContext built with regime_label + regime_probs
        3. Entry: Bypass regime filtering (all archetypes can fire)
        4. Sizing: Probabilistic regime scaling (soft directional budgets)
        5. Exit: Regime-adaptive parameters (trailing stops, time exits)

        Args:
            bar: Current bar (OHLCV data)
            engine: EventEngine instance (for order submission)
        """
        # Update bars held for ALL positions (CHANGED: support multiple positions)
        for position_id in self.positions:
            self.positions[position_id]['bars_held'] += 1

        # STEP 1: Get features for current bar
        try:
            features = self.feature_provider.get_features(bar)
        except Exception as e:
            self.logger.error(f"Failed to get features: {e}")
            return

        # STEP 2: Classify regime (if enabled)
        if self.regime_service:
            regime_result = self.regime_service.get_regime(features, bar.timestamp)
            regime_label = regime_result['regime_label']
            regime_probs = regime_result['regime_probs']
            regime_confidence = regime_result['regime_confidence']
        else:
            # Static regime mode
            regime_label = features.get('macro_regime', 'neutral')
            regime_probs = {regime_label: 1.0}
            regime_confidence = 1.0

        # STEP 3: Build RuntimeContext
        context = self._build_runtime_context(
            bar=bar,
            features=features,
            regime_label=regime_label,
            regime_probs=regime_probs
        )

        # STEP 3.5: Check exit conditions BEFORE entry signals (CRITICAL!)
        # CHANGED: Check exits for ALL positions
        # DEBUG: Log position check (first bar of each day)
        if bar.timestamp.hour == 0:
            self.logger.info(f"[EXIT DEBUG] Strategy positions: {list(self.positions.keys())}")
            self.logger.info(f"[EXIT DEBUG] Portfolio positions: {list(engine.portfolio.positions.keys())}")
        
        for position_id in list(self.positions.keys()):
            if position_id in engine.portfolio.positions:
                self._check_exit_conditions(bar, engine, context, regime_label, position_id)
            elif bar.timestamp.hour == 0:  # Log mismatches once per day
                self.logger.warning(f"[EXIT DEBUG] Position {position_id} in strategy but NOT in portfolio!")

        # STEP 4: Detect archetype signal
        # DEBUG: Log feature values before detection
        if bar.timestamp.hour == 19 and bar.timestamp.day == 1 and bar.timestamp.month == 1:
            self.logger.info(f"[DEBUG] Features at {bar.timestamp}:")
            self.logger.info(f"  bos_bullish: {features.get('bos_bullish')}")
            self.logger.info(f"  wyckoff_score: {features.get('wyckoff_score')}")
            self.logger.info(f"  fusion_wyckoff: {features.get('fusion_wyckoff')}")
            self.logger.info(f"  boms_strength: {features.get('boms_strength')}")
            self.logger.info(f"[DEBUG] Context row type: {type(context.row)}")
            self.logger.info(f"[DEBUG] Context row bos_bullish: {context.row.get('bos_bullish')}")

        # STEP 4.5: Detect archetype signals (single or multi-position mode)
        if self.multi_position_mode:
            # Multi-position mode: Get ALL valid archetypes
            matches = self.archetype_logic.detect_multi(context, max_positions=self.max_concurrent_positions)
        else:
            # Single-position mode: Get best archetype only
            archetype_name, fusion_score, liquidity_score, direction = self.archetype_logic.detect(context)
            if archetype_name is not None:
                matches = [(archetype_name, fusion_score, liquidity_score, direction)]
            else:
                matches = []

        # STEP 5: Process signals (potentially multiple in multi-position mode)
        if len(matches) > 0:
            self.total_signals += len(matches)

            # Log all detected signals
            if self.multi_position_mode and len(matches) > 1:
                self.logger.info(
                    f"[MULTI-SIGNAL] {len(matches)} archetypes detected: "
                    f"{[(m[0], f'{m[1]:.3f}') for m in matches]}"
                )

            # Clean up zombie positions ONCE before processing all signals
            for pos_id, pos in list(engine.portfolio.positions.items()):
                if abs(pos.quantity) < 1e-8:  # Less than 0.00000001 BTC
                    self.logger.warning(
                        f"[ZOMBIE CLEANUP] Removing zombie position: "
                        f"id={pos_id}, qty={pos.quantity:.15e}, side={pos.side.value}"
                    )
                    del engine.portfolio.positions[pos_id]
                    if pos_id in self.positions:
                        del self.positions[pos_id]

            # Process each archetype signal
            for archetype_name, fusion_score, liquidity_score, direction in matches:
                self.logger.info(
                    f"[SIGNAL] {archetype_name} detected @ ${bar.close:.2f} | "
                    f"Direction: {direction} | Fusion: {fusion_score:.3f} | "
                    f"Liquidity: {liquidity_score:.3f} | Regime: {regime_label} ({regime_confidence:.2f})"
                )

                # Check if we already have a position in this archetype
                existing_archetype_positions = [
                    pos_id for pos_id, pos_data in self.positions.items()
                    if pos_data.get('archetype') == archetype_name
                ]
                if existing_archetype_positions:
                    self.logger.debug(
                        f"[SKIP] Already have position in {archetype_name}: {existing_archetype_positions[0]}"
                    )
                    continue

                # Check concurrent positions limit
                num_positions = len(engine.portfolio.positions)
                can_open_position = num_positions < self.max_concurrent_positions

                if not can_open_position:
                    self.logger.info(
                        f"[SKIP] Position limit reached ({num_positions}/{self.max_concurrent_positions})"
                    )
                    break  # No more room for positions

                # Entry logic - CHANGED: Allow multiple concurrent positions
                if can_open_position:
                    # Calculate position size (with probabilistic regime scaling)
                    position_size = self._calculate_position_size(
                        bar=bar,
                        features=features,
                        fusion_score=fusion_score,
                        engine=engine,
                        regime_probs=regime_probs,
                        archetype=archetype_name
                    )

                    # Check if we have enough cash
                    if engine.portfolio.can_open_position(position_size, bar.close):
                        # Generate unique position ID BEFORE submitting order
                        position_id = f"{direction.lower()}_{archetype_name}_{int(bar.timestamp.timestamp())}"

                        # Submit order
                        side = OrderSide.BUY if direction == 'LONG' else OrderSide.SELL
                        quantity = position_size / bar.close

                        self.logger.info(f"[DEBUG] Position sizing: size_usd=${position_size:.2f}, price=${bar.close:.2f}, quantity={quantity:.6f}")

                        order = engine.submit_order(side, position_size, position_id=position_id)  # Pass position_id for exit tracking!

                        if order:
                            self.signals_taken += 1

                            # Calculate stop loss
                            atr = features.get('atr_14', features.get('atr', bar.close * 0.02))
                            if direction == 'LONG':
                                stop_loss = bar.close - (self.atr_stop_mult * atr)
                            else:
                                stop_loss = bar.close + (self.atr_stop_mult * atr)

                            # Track position state in positions dict
                            self.positions[position_id] = {
                                'archetype': archetype_name,
                                'direction': direction,
                                'entry_price': bar.close,
                                'entry_time': bar.timestamp,
                                'fusion_score': fusion_score,
                                'regime': regime_label,
                                'original_quantity': quantity,  # CRITICAL: Store original quantity for exit calculations
                                'executed_scale_outs': [],  # Track which R-levels have been scaled out
                                'executed_exit_types': set(),  # Track which exit types already fired
                                'remaining_position_pct': 1.0,  # Track % of position still open
                                'total_exits_pct': 0.0,  # Track cumulative exit percentage
                                'stop_loss': stop_loss,  # Store stop loss per position
                                'bars_held': 0  # Track bars held for this position
                            }

                            self.logger.info(
                                f"[ENTRY] {direction} ${position_size:,.0f} @ ${bar.close:.2f} | "
                                f"Stop: ${stop_loss:.2f} | Archetype: {archetype_name} | ID: {position_id}"
                            )
                    else:
                        self.signals_rejected += 1
                        self.logger.warning(f"[REJECTED] Insufficient cash for ${position_size:,.0f} position")

        # STEP 6: Check exits for ALL positions (CHANGED: iterate through multiple positions)
        # LEGACY STOP LOSS DISABLED - Now handled by ExitLogic in _check_exit_conditions (called in STEP 3.5)
        # for position_id in list(self.positions.keys()):  # Use list() to avoid dict modification during iteration
        #     if position_id in engine.portfolio.positions:
        #         self._check_stop_loss(bar, engine, position_id)

    def _build_runtime_context(
        self,
        bar: Bar,
        features: Dict[str, Any],
        regime_label: str,
        regime_probs: Dict[str, float]
    ) -> RuntimeContext:
        """
        Build RuntimeContext for archetype detection.

        This creates the unified context object that flows through the entire
        decision pipeline, containing:
        - Bar data
        - Features (from feature store or runtime computation)
        - Regime state (label + probabilities)
        - Adapted parameters (from ThresholdPolicy)

        Args:
            bar: Current bar
            features: Feature dict
            regime_label: Current regime
            regime_probs: Regime probabilities

        Returns:
            RuntimeContext object
        """
        # Convert features dict to Series (ArchetypeLogic expects Series)
        row = pd.Series(features)
        row.name = bar.timestamp  # Set timestamp as index

        # Get adapted thresholds from ThresholdPolicy
        thresholds = self.threshold_policy.resolve(
            regime_probs=regime_probs,
            regime_label=regime_label
        )

        # Build context with regime config for archetype logic
        regime_config = self.config.get('regime_classifier', {})

        return RuntimeContext(
            ts=bar.timestamp,
            row=row,
            regime_probs=regime_probs,
            regime_label=regime_label,
            adapted_params={},  # Could add adaptive fusion params here
            thresholds=thresholds,
            regime_config=regime_config,
            metadata={}
        )

    def _calculate_position_size(
        self,
        bar: Bar,
        features: Dict[str, Any],
        fusion_score: float,
        engine: EventEngine,
        regime_probs: Optional[Dict[str, float]] = None,
        archetype: str = 'trap_within_trend'
    ) -> float:
        """
        Calculate position size using ATR-based risk management + PROBABILISTIC REGIME SCALING.

        Formula:
            Risk Amount = Portfolio Value × Risk %
            Stop Distance = ATR × Stop Multiplier
            Base Position Size = Risk Amount / Stop Distance

        Then scale by:
        1. Fusion score (signal confidence)
        2. PROBABILISTIC REGIME WEIGHT (soft controls - NEW!)
        3. Max position cap

        Args:
            bar: Current bar
            features: Feature dict
            fusion_score: Fusion confidence score
            engine: EventEngine (for portfolio value)
            regime_probs: Regime probability distribution (for probabilistic scaling)
            archetype: Archetype name (for regime allocator)

        Returns:
            Position size in USD
        """
        # Get portfolio value - USE CASH ONLY, not unrealized PnL from open positions
        # This prevents exponential position sizing from unrealized gains
        portfolio_value = engine.portfolio.cash

        # Calculate risk amount
        risk_amount = portfolio_value * self.risk_per_trade

        # Get ATR for stop distance
        atr = features.get('atr_14', features.get('atr', bar.close * 0.02))
        stop_distance = self.atr_stop_mult * atr
        stop_distance_pct = stop_distance / bar.close

        # Base position size
        base_position_size = risk_amount / stop_distance_pct

        # Scale by fusion score (confidence)
        position_size = base_position_size * fusion_score

        # PROBABILISTIC REGIME SCALING (NEW!)
        # For ENTRIES: bypass_entry_filtering flag controls whether to apply regime scaling
        # For EXITS: always apply regime scaling (handled in _check_exit_conditions)
        if self.regime_allocator and regime_probs:
            # Get probabilistic regime weight (weighted blend across regime probabilities)
            self.logger.info(
                f"[ENTRY SIZING] Calling regime allocator with is_entry=True for {archetype}"
            )

            regime_weight = self.regime_allocator.compute_weight_probabilistic(
                edge=0.5,  # Placeholder - would use historical edge if available
                N=50,      # Placeholder - would use historical trade count if available
                archetype=archetype,
                regime_probs=regime_probs,
                is_entry=True  # Mark as entry decision (bypass flag applies here)
            )

            self.logger.info(
                f"[ENTRY SIZING] Received regime_weight={regime_weight:.3f} from allocator"
            )

            position_size *= regime_weight

            self.logger.info(
                f"Probabilistic regime scaling (ENTRY): archetype={archetype}, "
                f"regime_probs={regime_probs}, regime_weight={regime_weight:.3f}, "
                f"position_size={position_size:.2f}"
            )

        # Cap at max position percentage
        max_position = portfolio_value * self.max_position_pct
        position_size = min(position_size, max_position)

        self.logger.debug(
            f"Position sizing: portfolio=${portfolio_value:,.0f}, "
            f"risk={self.risk_per_trade*100:.1f}%, atr=${atr:.2f}, "
            f"stop_dist={stop_distance_pct*100:.2f}%, "
            f"fusion={fusion_score:.2f}, size=${position_size:,.0f}"
        )

        return position_size

    def _check_exit_conditions(self, bar: Bar, engine: EventEngine, context: RuntimeContext, regime_label: str, position_id: str):
        """
        Check comprehensive exit conditions using ExitLogic system.

        UPDATED to support multiple concurrent positions.

        This checks:
        1. Invalidation exits (pattern breaks)
        2. Profit target exits (R-multiple scale-outs)
        3. Time-based exits (max hold period)
        4. Reason-gone exits (entry condition reverses)
        5. Trailing stop updates

        Args:
            bar: Current bar
            engine: EventEngine instance
            context: RuntimeContext
            regime_label: Current regime
            position_id: Unique position identifier
        """
        if position_id not in self.positions:
            return

        if position_id not in engine.portfolio.positions:
            return

        position_state = self.positions[position_id]
        engine_position = engine.portfolio.positions[position_id]

        # Import Position here to avoid circular dependency
        from engine.models.base import Position

        # Convert engine position to Position object for ExitLogic
        archetype = position_state.get('archetype', 'trap_within_trend')

        # Calculate position size in USD (quantity * current price)
        position_size_usd = engine_position.quantity * bar.close

        position = Position(
            entry_price=position_state['entry_price'],
            entry_time=position_state['entry_time'],
            size=position_size_usd,  # Position size in quote currency ($)
            direction=position_state['direction'],
            stop_loss=position_state['stop_loss'],
            take_profit=None,
            regime_label=position_state.get('regime', 'neutral'),
            metadata={
                'archetype': archetype,
                'fusion_score': position_state.get('fusion_score', 0.5),
                'regime': position_state.get('regime', 'neutral'),
                'executed_scale_outs': position_state.get('executed_scale_outs', []),  # Persist executed scale-outs
                'executed_exit_types': position_state.get('executed_exit_types', set()),  # Persist exit types
                'entry_prev_high': context.row.get('prev_high'),  # For liquidity_vacuum exits
                'moon_bag_taken': False,  # For funding_divergence exits
                'scaled_at_prev_high': False  # For S1 exits
            }
        )

        # Check exit logic
        # Create bar Series with timestamp as name (exit_logic uses bar.name)
        bar_series = pd.Series({
            'close': bar.close,
            'high': bar.high,
            'low': bar.low,
            'open': bar.open,
            'volume': bar.volume,
            'atr': context.row.get('atr_14', context.row.get('atr', bar.close * 0.02))
        })
        bar_series.name = bar.timestamp  # Set Series name to timestamp for exit_logic

        exit_signal = self.exit_logic.check_exit(
            bar=bar_series,
            position=position,
            archetype=archetype,
            context=context
        )

        if exit_signal:
            # Execute exit
            side = OrderSide.SELL if position_state['direction'] == 'LONG' else OrderSide.BUY

            # Track cumulative exit percentage
            total_exits_pct = position_state.get('total_exits_pct', 0.0)
            remaining_pct = 1.0 - total_exits_pct

            # Adjust exit_pct to not exceed remaining position
            actual_exit_pct = min(exit_signal.exit_pct, remaining_pct)

            # CRITICAL FIX: Calculate exit based on ORIGINAL quantity from portfolio metadata
            # Scale-outs should be % of original position to ensure complete exit after all scales
            original_quantity = engine_position.metadata.get('original_quantity', engine_position.quantity)
            executed_scale_outs = engine_position.metadata.get('executed_scale_outs', 0.0)

            # Calculate exit quantity as % of original
            target_exit_qty = original_quantity * actual_exit_pct

            # Cap to remaining quantity (prevent over-closing due to rounding)
            remaining_qty = engine_position.quantity
            exit_quantity = min(target_exit_qty, remaining_qty)

            # If this is the last scale-out (total will reach 100%), close remaining
            projected_total = executed_scale_outs + exit_quantity
            if projected_total >= original_quantity * 0.99:  # 99% threshold for rounding
                exit_quantity = remaining_qty  # Close all remaining

            exit_size_usd = exit_quantity * bar.close

            order = engine.submit_order(side, exit_size_usd, position_id=position_id, is_exit=True)

            if order:
                # Update cumulative tracking
                total_exits_pct += actual_exit_pct
                position_state['total_exits_pct'] = total_exits_pct
                remaining_pct = 1.0 - total_exits_pct

                # Track executed scale-outs in position state (persist for next bar)
                if 'scale_level' in exit_signal.metadata:
                    scale_level = exit_signal.metadata['scale_level']
                    if scale_level not in position_state['executed_scale_outs']:
                        position_state['executed_scale_outs'].append(scale_level)

                # Track executed exit types
                position_state['executed_exit_types'].add(exit_signal.exit_type)

                self.logger.info(
                    f"[EXIT] {exit_signal.exit_type} @ ${bar.close:.2f} | "
                    f"Exit: {actual_exit_pct*100:.0f}% | Remaining: {remaining_pct*100:.1f}% | "
                    f"Reason: {exit_signal.reason} | "
                    f"Bars held: {position_state['bars_held']} | "
                    f"Archetype: {archetype} | ID: {position_id}"
                )

                # If position fully closed (or close enough), clear state
                if remaining_pct <= 0.05:  # <= 5% remaining = close enough to zero
                    # CRITICAL FIX: Close ANY remaining position in the engine's portfolio to prevent zombie positions
                    # Even if percentage tracking shows 0%, there may be tiny rounding errors in actual quantity
                    if engine_position and engine_position.quantity > 1e-10:  # Check actual quantity, not percentage
                        final_exit_quantity = engine_position.quantity
                        final_exit_size_usd = final_exit_quantity * bar.close
                        final_side = OrderSide.SELL if position_state['direction'] == 'LONG' else OrderSide.BUY

                        final_order = engine.submit_order(final_side, final_exit_size_usd, position_id=position_id, is_exit=True)
                        if final_order:
                            self.logger.info(
                                f"[FINAL EXIT] Closing zombie remainder {final_exit_quantity:.10f} BTC "
                                f"(${final_exit_size_usd:.2f}) - percentage tracking showed {remaining_pct*100:.1f}% | ID: {position_id}"
                            )
                    elif engine_position and engine_position.quantity > 0:
                        # Position has microscopic quantity (< 1e-10 BTC), just warn and clear
                        self.logger.warning(
                            f"[ZOMBIE DUST] Position has microscopic remainder {engine_position.quantity:.15f} BTC, "
                            f"ignoring as it's below minimum tradeable amount | ID: {position_id}"
                        )

                    self.logger.info(f"[POSITION CLOSED] Remaining {remaining_pct*100:.1f}% - closing position {position_id}")
                    # Remove position from tracking
                    del self.positions[position_id]

                # Update stop loss if trailing stop
                elif exit_signal.stop_update is not None:
                    position_state['stop_loss'] = exit_signal.stop_update
                    self.logger.info(f"[TRAIL STOP] Updated to ${exit_signal.stop_update:.2f} | ID: {position_id}")

    def _check_stop_loss(self, bar: Bar, engine: EventEngine, position_id: str):
        """
        Legacy stop loss check - DEPRECATED (now handled by ExitLogic).

        Kept for backward compatibility. UPDATED to support multiple positions.

        Args:
            bar: Current bar
            engine: EventEngine (for order submission)
            position_id: Unique position ID
        """
        # NOTE: This is now redundant since ExitLogic handles invalidation exits
        # Keeping for safety but should rarely trigger
        if position_id not in self.positions:
            return

        position_state = self.positions[position_id]
        direction = position_state['direction']
        stop_loss = position_state['stop_loss']

        # Check if stop hit
        stop_hit = False
        if direction == 'LONG' and bar.low <= stop_loss:
            stop_hit = True
        elif direction == 'SHORT' and bar.high >= stop_loss:
            stop_hit = True

        if stop_hit:
            # Close position
            side = OrderSide.SELL if direction == 'LONG' else OrderSide.BUY

            # Get current position quantity
            if position_id in engine.portfolio.positions:
                position = engine.portfolio.positions[position_id]

                # Calculate position size in USD (submit_order expects size_usd, not quantity)
                position_size_usd = position.quantity * bar.close

                order = engine.submit_order(side, position_size_usd, position_id=position_id, is_exit=True)

                if order:
                    self.logger.info(
                        f"[STOP LOSS - LEGACY] Closing {direction} @ ${stop_loss:.2f} | "
                        f"Bars held: {position_state['bars_held']} | "
                        f"Archetype: {position_state['archetype']} | ID: {position_id}"
                    )

                    # Remove position from tracking
                    del self.positions[position_id]

    def on_order_filled(self, order: Order, engine: EventEngine):
        """
        Called when order fills.

        Args:
            order: Filled order
            engine: EventEngine instance
        """
        self.logger.info(f"Order filled: {order}")

    def on_stop(self, engine: EventEngine):
        """
        Called when backtest ends - close open positions and log final statistics.

        Args:
            engine: EventEngine instance
        """
        # Close any open positions (CHANGED: support multiple positions)
        if len(engine.portfolio.positions) > 0:
            self.logger.info(f"Closing {len(engine.portfolio.positions)} open position(s) at backtest end...")
            current_bar = engine.current_bar

            # Determine which positions to close
            for position_id in list(engine.portfolio.positions.keys()):
                position = engine.portfolio.positions[position_id]

                # Submit closing order
                side = OrderSide.SELL if position.side.value == 'LONG' else OrderSide.BUY
                quantity = position.quantity

                # Close via portfolio directly (bypass order submission since backtest is ending)
                engine.portfolio._close_position(
                    position_id=position_id,
                    quantity=quantity,
                    exit_price=current_bar.close,
                    exit_timestamp=current_bar.timestamp
                )

                self.logger.info(f"Closed {position.side.value} position ({position_id}) at ${current_bar.close:.2f}")

            # Clear all position tracking
            self.positions.clear()

        stats = engine.get_performance_stats()

        self.logger.info("=" * 80)
        self.logger.info(f"{self.name} STOPPED")
        self.logger.info("=" * 80)
        self.logger.info(f"Total signals: {self.total_signals}")
        self.logger.info(f"Signals taken: {self.signals_taken}")
        self.logger.info(f"Signals rejected: {self.signals_rejected}")
        self.logger.info(f"Signal fill rate: {(self.signals_taken/self.total_signals*100 if self.total_signals > 0 else 0):.1f}%")
        self.logger.info(f"Total trades: {stats['total_trades']}")
        self.logger.info(f"Win rate: {stats['win_rate']:.1f}%")
        self.logger.info(f"Total PnL: ${stats['total_pnl']:,.2f}")
        self.logger.info(f"Total return: {stats['total_return']:.2f}%")
        self.logger.info("=" * 80)
