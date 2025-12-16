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
        regime_classifier_path: Optional[str] = None
    ):
        """
        Initialize archetype model wrapper.

        Args:
            config_path: Path to config JSON (e.g., 'configs/s4_optimized.json')
            archetype_name: Single archetype to use (e.g., 'S4', 'S1', 'trap_within_trend')
            name: Human-readable model name (defaults to archetype_name)
            regime_classifier_path: Optional path to regime classifier model
        """
        super().__init__(name=name or f"Archetype-{archetype_name}")

        self.config_path = Path(config_path)
        self.archetype_name = archetype_name
        self.regime_classifier_path = regime_classifier_path

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

        logger.info(f"Initialized {self.name} with config: {self.config_path}")
        logger.info(f"Archetype params: {self.archetype_params}")

    def _extract_archetype_params(self):
        """Extract archetype-specific parameters from config."""
        # Map common archetype names to config keys
        archetype_key_map = {
            'S1': 'liquidity_vacuum',
            'S2': 'failed_rally',
            'S3': 'whipsaw',
            'S4': 'funding_divergence',
            'S5': 'long_squeeze',
            'A': 'spring',
            'B': 'order_block_retest',
            'C': 'wick_trap',
            'D': 'failed_continuation',
            'E': 'volume_exhaustion',
            'F': 'exhaustion_reversal',
            'G': 'liquidity_sweep',
            'H': 'momentum_continuation',
            'K': 'trap_within_trend',
            'L': 'retest_cluster',
            'M': 'confluence_breakout',
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

        Args:
            bar: Current bar data (row from DataFrame)
            position: Current open position (if any)

        Returns:
            Signal object with direction, confidence, entry price, stop loss
        """
        # Build RuntimeContext for archetype detection
        context = self._build_runtime_context(bar)

        # Call archetype logic
        archetype_name, fusion_score, liquidity_score = self.archetype_logic.detect(context)

        # Convert to Signal
        if archetype_name is None:
            # No entry signal
            return Signal(
                direction='hold',
                confidence=0.0,
                entry_price=bar['close'],
                metadata={
                    'fusion_score': fusion_score,
                    'liquidity_score': liquidity_score,
                    'reason': 'no_archetype_match'
                }
            )

        # Map fusion score to confidence (0.0-1.0)
        # Fusion scores typically range 0.3-0.8, normalize relative to threshold
        confidence = min(1.0, fusion_score / max(self.fusion_threshold, 0.01))

        # Calculate stop loss
        atr = bar.get('atr_14', bar.get('atr', bar['close'] * 0.02))

        if self.direction == 'long':
            stop_loss = bar['close'] - (self.atr_stop_mult * atr)
            signal_direction = 'long'
        else:
            stop_loss = bar['close'] + (self.atr_stop_mult * atr)
            signal_direction = 'short'

        return Signal(
            direction=signal_direction,
            confidence=confidence,
            entry_price=bar['close'],
            stop_loss=stop_loss,
            metadata={
                'archetype': archetype_name,
                'fusion_score': fusion_score,
                'liquidity_score': liquidity_score,
                'atr': atr,
                'atr_stop_mult': self.atr_stop_mult
            }
        )

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """
        Calculate position size using ATR-based risk management.

        Formula: Position Size = (Portfolio Value × Risk %) / Stop Distance %

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

        # Risk amount in dollars
        risk_dollars = portfolio_value * self.max_risk_pct

        # Position size calculation
        # Example: $10k portfolio, 2% risk = $200 risk
        #          Stop 5% away = $200 / 0.05 = $4000 position
        position_size = risk_dollars / stop_distance_pct

        # Cap at reasonable max (e.g., 15% of portfolio)
        max_position = portfolio_value * 0.15
        position_size = min(position_size, max_position)

        logger.debug(
            f"Position sizing: portfolio=${portfolio_value:,.0f}, "
            f"risk={self.max_risk_pct*100:.1f}%, "
            f"stop_dist={stop_distance_pct*100:.2f}%, "
            f"size=${position_size:,.0f}"
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
        # Get regime from data if available, otherwise use default
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
        return RuntimeContext(
            ts=bar.name if hasattr(bar, 'name') else pd.Timestamp.now(),
            row=row_with_runtime,  # FIXED: Pass enriched row with runtime scores
            regime_probs=regime_probs,
            regime_label=regime_label,
            adapted_params={},  # Could add adaptive fusion params here
            thresholds=thresholds,
            metadata={}
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
