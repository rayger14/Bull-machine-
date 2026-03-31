"""
Archetype-Specific Exit Logic System (Smart Exits V2)

Implements comprehensive exit rules for the v17 Whale Footprint architecture:
1. Hard stop-loss (inline, fill-at-stop-level)
2. Composite invalidation (5-feature scoring, threshold 4/5, wick_trap + retest_cluster)
3. Distress half-exit (50% exit when underwater + 4/5 distress signals)
4. R-multiple scale-out targets (per-archetype R-ladders)
5. Time-based exits (per-archetype max_hold_hours)
6. Reason-gone exits (funding flip, OI reversal)
7. Chop-aware trailing stops (0.75x at chop>0.45, 0.88x at chop>0.35)
8. Runner position (keep remainder for extended moves)

Priority chain: hard_stop → invalidation → distress → profit_targets →
               time_exit → reason_gone → trailing_stop → runner

Each archetype has exit config in create_default_exit_config() and per-archetype
YAML files (configs/archetypes/*.yaml). Regime exit scaling is DISABLED (all
factors = 1.0) — backtesting showed regime-scaled exits were net negative.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
import pandas as pd

# Use TYPE_CHECKING to avoid circular import
# Position is only used for type hints
if TYPE_CHECKING:
    from engine.models.base import Position

from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)

# Canonical name mapping: ExitLogic internal names ↔ YAML names
# ExitLogic was written with old names; YAMLs use new names
_ARCHETYPE_ALIASES = {
    'spring': 'trap_reversal',      # YAML 'spring' → ExitLogic 'trap_reversal'
    'trap_reversal': 'trap_reversal',  # Already canonical
}


def _resolve_exit_name(archetype: str) -> str:
    """Resolve archetype name to the canonical name used in ExitLogic exit_rules."""
    return _ARCHETYPE_ALIASES.get(archetype, archetype)


class ExitType(Enum):
    """Exit signal types ordered by priority."""
    INVALIDATION = "invalidation"  # Highest priority - pattern broken
    DISTRESS = "distress"  # Feature-based early partial exit for underwater positions
    PROFIT_TARGET = "profit_target"  # Scale-out at targets
    TIME_EXIT = "time_exit"  # Max hold period
    REASON_GONE = "reason_gone"  # Entry condition reversed
    TRAILING_STOP = "trailing_stop"  # Protect profits
    NONE = "none"  # No exit


@dataclass
class ExitSignal:
    """
    Exit signal with type, size, and optional stop update.

    Attributes:
        exit_type: Type of exit (invalidation, profit_target, etc.)
        exit_pct: Percentage of position to exit (0.2 = 20%, 1.0 = 100%)
        stop_update: Optional new stop loss level (for trailing)
        reason: Human-readable explanation
        confidence: Exit confidence (0.0-1.0)
        metadata: Additional context
    """
    exit_type: str
    exit_pct: float  # 0.2 = 20% partial, 1.0 = 100% full exit
    stop_update: Optional[float] = None
    reason: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_full_exit(self) -> bool:
        """Check if this is a full exit (100%)."""
        return self.exit_pct >= 1.0

    @property
    def is_partial_exit(self) -> bool:
        """Check if this is a partial exit (<100%)."""
        return 0 < self.exit_pct < 1.0


class ExitLogic:
    """
    Archetype-specific exit logic coordinator.

    Checks exit conditions in priority order:
    1. Invalidation V2 (composite feature-based structural breakdown)
    2. Distress half-exit (feature-based early partial exit for underwater positions)
    3. Profit targets (scale-outs)
    4. Time-based (max hold period)
    5. Reason-gone (entry condition reversed)
    6. Trailing stops (profit protection + chop-aware tightening)
    7. Runner exit (wider trailing for remaining position after all scale-outs)
    """

    def __init__(self, config: Dict):
        """
        Initialize exit logic with configuration.

        Args:
            config: Full configuration dict with exit rules per archetype
        """
        self.config = config
        self.exit_rules = self._build_exit_rules()

        # Global exit settings
        self.enable_scale_outs = config.get('enable_scale_outs', True)
        self.enable_time_exits = config.get('enable_time_exits', True)
        self.enable_trailing = config.get('enable_trailing', True)

        logger.info(f"ExitLogic initialized with {len(self.exit_rules)} archetype rule sets")

    def _build_exit_rules(self) -> Dict[str, Dict]:
        """
        Build archetype-specific exit rules from config.

        Returns:
            Dict mapping archetype name to exit rule set
        """
        # FIXED: Check both 'exit_rules' (nested) and top-level keys (after update())
        # After exit_config.update(user_config), archetype configs are at TOP LEVEL
        exit_config = self.config.get('exit_rules', {})

        logger.debug(f"[EXIT_LOGIC BUILD] Loading exit rules, {len(exit_config)} archetypes")

        # Default exit rules (fallback for archetypes without specific rules)
        default_rules = {
            'max_hold_hours': 168,  # 7 days
            'scale_out_levels': [0.5, 1.0, 2.0],  # R-multiples
            'scale_out_pcts': [0.2, 0.2, 0.3],  # Exit 20%, 20%, 30% at each level
            'trailing_start_r': 0.5,  # Start trailing after +0.5R (earlier protection)
            'trailing_atr_mult': 2.0,  # Trail 2 ATR behind peak
            'runner_pct': 0.0,  # No runner by default (0 = disabled)
            'runner_trailing_atr': 3.0,  # Runner uses wider trailing stop
            'invalidation_checks': False,  # Disabled: -$54K net in backtest (wick/spring invalidation all losers)
            'reason_gone_checks': False   # Disabled: needs tuning, currently hurts more than helps
        }

        rules = {}

        # Check for global trailing_atr_mult override
        global_trailing_mult = self.config.get('trailing_atr_mult')
        if global_trailing_mult:
            default_rules['trailing_atr_mult'] = global_trailing_mult

        # Build rules for each archetype
        for archetype_key in self._get_all_archetypes():
            # Start with defaults
            archetype_rules = default_rules.copy()

            # FIXED: Check both nested exit_config AND top-level self.config
            # Priority: exit_rules > top-level config
            config_override_applied = False

            # Check nested exit_rules first (from create_default_exit_config)
            if archetype_key in exit_config:
                archetype_rules.update(exit_config[archetype_key])
                config_override_applied = True
                logger.debug(f"  {archetype_key}: loaded from exit_rules")

            # Check top-level config (from user config after update())
            elif archetype_key in self.config:
                archetype_rules.update(self.config[archetype_key])
                config_override_applied = True
                logger.info(f"  {archetype_key}: loaded from TOP LEVEL config (user override)")

            rules[archetype_key] = archetype_rules

        return rules

    def _get_all_archetypes(self) -> list:
        """Get list of all supported archetypes (canonical ExitLogic names)."""
        return [
            # Active production archetypes
            'wick_trap',
            'liquidity_sweep',
            'retest_cluster',
            'liquidity_vacuum',
            'trap_within_trend',
            'funding_divergence',
            'long_squeeze',
            'order_block_retest',
            'fvg_continuation',
            'failed_continuation',
            'liquidity_compression',
            'confluence_breakout',
            'exhaustion_reversal',
            'oi_divergence',
            'trap_reversal',  # = spring (alias)
            'whipsaw',
            'volume_fade_chop',
            # Legacy names (for backward compat)
            'failed_rally',
            'alt_rotation_down',
            'curve_inversion',
            'expansion_exhaustion',
            're_accumulate',
            'volume_exhaustion',
            'ratio_coil_break',
        ]

    def check_exit(
        self,
        bar: pd.Series,
        position: "Position",
        archetype: str,
        context: RuntimeContext
    ) -> Optional[ExitSignal]:
        """
        Check if exit conditions met for current position.

        Checks exit conditions in priority order:
        1. Invalidation (highest priority)
        2. Profit targets
        3. Time-based
        4. Reason-gone
        5. Trailing stops

        Args:
            bar: Current bar data
            position: Open position
            archetype: Archetype name (e.g., 'liquidity_vacuum')
            context: Runtime context with thresholds and regime state

        Returns:
            ExitSignal if exit triggered, None otherwise
        """
        # Resolve archetype name (spring → trap_reversal for rule lookup)
        canonical = _resolve_exit_name(archetype)

        # Get archetype-specific rules
        rules = self.exit_rules.get(canonical) or self.exit_rules.get(archetype)
        if not rules:
            logger.warning(f"No exit rules for archetype '{archetype}' (canonical: '{canonical}'), using defaults")
            rules = self.exit_rules.get('wick_trap', {})  # Use wick_trap as default

        # Calculate position metrics
        hours_in_position = (bar.name - position.entry_time).total_seconds() / 3600
        atr = bar.get('atr_14', bar.get('atr', bar['close'] * 0.02))
        unrealized_r = self._calculate_unrealized_r(position, bar['close'], atr)

        # Store metrics in metadata for all checks
        exit_context = {
            'hours_in_position': hours_in_position,
            'unrealized_r': unrealized_r,
            'atr': atr,
            'archetype': archetype
        }

        # Apply regime-adaptive exit adjustments to scale-out levels
        regime = getattr(context, 'regime_label', 'neutral')
        regime_adj = self._get_regime_exit_adjustments(regime, rules)

        # Build regime-adjusted rules copy (do not mutate original)
        adjusted_rules = dict(rules)
        scale_level_mult = regime_adj.get('scale_level_multiplier', 1.0)
        if scale_level_mult != 1.0:
            # Adjust scale-out R-levels by regime multiplier
            if 'profit_targets' in adjusted_rules:
                adjusted_rules['profit_targets'] = [
                    {**pt, 'r_multiple': pt['r_multiple'] * scale_level_mult}
                    for pt in adjusted_rules['profit_targets']
                ]
            if 'scale_out_levels' in adjusted_rules:
                adjusted_rules['scale_out_levels'] = [
                    lvl * scale_level_mult
                    for lvl in adjusted_rules['scale_out_levels']
                ]

        # 1. CHECK INVALIDATION V2 (highest priority — composite scoring)
        if rules.get('invalidation_checks', False):
            if invalidation := self._check_invalidation(bar, position, rules, context, exit_context):
                return invalidation

        # 2. CHECK DISTRESS HALF-EXIT (feature-based early partial exit)
        if distress := self._check_distress_exit(bar, position, rules, context, exit_context):
            return distress

        # 3. CHECK PROFIT TARGETS / SCALE-OUTS (regime-adjusted levels)
        if self.enable_scale_outs:
            if profit_target := self._check_profit_targets(bar, position, adjusted_rules, context, exit_context):
                return profit_target

        # 4. CHECK TIME-BASED EXIT
        if self.enable_time_exits:
            if time_exit := self._check_time_based(bar, position, rules, context, exit_context):
                return time_exit

        # 5. CHECK REASON-GONE (entry condition reversed)
        if rules.get('reason_gone_checks', False):
            if reason_gone := self._check_reason_gone(bar, position, rules, context, exit_context):
                return reason_gone

        # 6. UPDATE TRAILING STOP (no exit, just update)
        if self.enable_trailing:
            self._update_trailing_stop(bar, position, rules, context, exit_context)

        # 7. CHECK RUNNER EXIT (trailing stop for runner portion after all scale-outs)
        if runner_exit := self._check_runner_exit(bar, position, rules, exit_context):
            return runner_exit

        return None

    def _calculate_unrealized_r(self, position: "Position", current_price: float, atr: float) -> float:
        """
        Calculate unrealized R-multiple for position.

        R-multiple = (Current PnL) / (Initial Risk)

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

    def _get_adjusted_trailing_mult(self, base_mult: float, r_achieved: float) -> float:
        """
        Progressively tighten trailing stops as profit increases.

        Based on research: crypto requires 3.0-4.0x ATR base multiplier,
        with progressive tightening to lock in gains at profit milestones.

        Progression:
        - Base (0-1R): 3.0x ATR (wide to survive volatility)
        - After 1R: 2.5x ATR (tighten slightly)
        - After 2R: 2.0x ATR (moderate tightening)
        - After 3R+: 1.5x ATR (lock in significant gains)

        Args:
            base_mult: Base trailing multiplier (typically 3.0 for crypto)
            r_achieved: Unrealized R-multiple achieved

        Returns:
            Adjusted trailing multiplier
        """
        if r_achieved >= 3.0:
            return base_mult * 0.5  # 3.0 * 0.5 = 1.5x ATR
        elif r_achieved >= 2.0:
            return base_mult * 0.67  # 3.0 * 0.67 = 2.0x ATR
        elif r_achieved >= 1.0:
            return base_mult * 0.83  # 3.0 * 0.83 = 2.5x ATR
        else:
            return base_mult  # 3.0x ATR (base)


    def _get_regime_adjusted_params(self, base_params: dict, regime: str, r_multiple: float) -> dict:
        """
        Adjust exit parameters based on current market regime.

        NOTE: All regime factors are set to 1.0 (DISABLED). Backtesting showed
        regime-scaled time exits and trailing multipliers were net negative.
        The proven per-archetype config values (168h max hold, fixed R-ladders)
        outperform regime-adaptive scaling. Keeping this method as a no-op
        hook for future experimentation.

        Args:
            base_params: Base exit parameters from config
            regime: Current regime ('risk_on', 'neutral', 'risk_off', 'crisis')
            r_multiple: Current R-multiple achieved

        Returns:
            Adjusted parameters dict (currently unchanged from base_params)
        """
        # Regime adjustment factors
        # NOTE: Regime-adaptive time exits were net negative in backtesting.
        # The old inline exit code had NO regime time scaling and performed better.
        # Keeping trailing mult neutral too — per-archetype config is sufficient.
        regime_factors = {
            'risk_on': {
                'trailing_mult_factor': 1.0,
                'time_exit_factor': 1.0,
            },
            'neutral': {
                'trailing_mult_factor': 1.0,
                'time_exit_factor': 1.0,
            },
            'risk_off': {
                'trailing_mult_factor': 1.0,
                'time_exit_factor': 1.0,
            },
            'crisis': {
                'trailing_mult_factor': 1.0,
                'time_exit_factor': 1.0,
            }
        }

        # Get factors for current regime (default to neutral if unknown)
        factors = regime_factors.get(regime, regime_factors['neutral'])

        # Apply factors to base parameters
        adjusted = {
            'trailing_atr_mult': base_params.get('trailing_atr_mult', 3.0) * factors['trailing_mult_factor'],
            'max_hold_hours': base_params.get('max_hold_hours', 168) * factors['time_exit_factor'],
        }

        return adjusted

    def _get_regime_exit_adjustments(self, regime: str, rules: Dict) -> Dict:
        """
        Scale exit parameters based on market regime.

        Provides multipliers that widen or tighten exit parameters depending
        on the current regime. In risk_on (bull), holds are longer and targets
        wider to let winners run. In risk_off/crisis, holds are shorter and
        targets tighter to protect capital.

        Args:
            regime: Current regime label ('risk_on', 'neutral', 'risk_off', 'crisis')
            rules: Base archetype exit rules

        Returns:
            Dict with multiplier keys:
                - max_hold_multiplier: Scale max hold period
                - scale_level_multiplier: Scale profit target R-levels
                - trailing_atr_multiplier: Scale trailing stop width
        """
        adjustments = {
            'risk_on': {
                'max_hold_multiplier': 1.0,     # No hold extension (per-archetype config is enough)
                'scale_level_multiplier': 1.0,   # Scale-out levels are absolute, never adjusted
                'trailing_atr_multiplier': 1.0,  # No trailing change
            },
            'neutral': {
                'max_hold_multiplier': 1.0,
                'scale_level_multiplier': 1.0,
                'trailing_atr_multiplier': 1.0,
            },
            'risk_off': {
                'max_hold_multiplier': 0.5,     # Shorter holds in bear
                'scale_level_multiplier': 1.0,   # Scale-out levels are absolute
                'trailing_atr_multiplier': 1.0,
            },
            'crisis': {
                'max_hold_multiplier': 0.25,    # Very short in crisis
                'scale_level_multiplier': 1.0,   # Scale-out levels are absolute
                'trailing_atr_multiplier': 1.0,
            },
        }
        return adjustments.get(regime, adjustments['neutral'])

    def _all_scaleouts_executed(self, position, rules: Dict) -> bool:
        """
        Check if all configured scale-out levels have been executed.

        Compares the list of executed scale-out levels stored in position
        metadata against the configured scale-out levels in the rules.

        Args:
            position: Open position with metadata tracking executed scale-outs
            rules: Archetype exit rules with scale-out configuration

        Returns:
            True if all scale-out levels have been executed, False otherwise
        """
        executed_scales = position.metadata.get('executed_scale_outs', [])

        # Get configured scale levels from profit_targets or scale_out_levels
        profit_targets = rules.get('profit_targets', [])
        if profit_targets:
            configured_levels = [pt['r_multiple'] for pt in profit_targets]
        else:
            configured_levels = rules.get('scale_out_levels', [0.5, 1.0, 2.0])

        if not configured_levels:
            return False

        # Check if all configured levels have been executed
        return all(level in executed_scales for level in configured_levels)

    def _check_runner_exit(self, bar: pd.Series, position, rules: Dict, exit_context: Dict) -> Optional[ExitSignal]:
        """
        Check if the runner portion should exit (trailing stop only).

        The runner concept: after all configured scale-outs are done
        (e.g., 25%+35%+20% = 80% exited), the remaining percentage stays
        open as a "runner" with a wider trailing stop. This lets winners
        run on trending moves.

        The runner only exits when its dedicated trailing stop is hit.
        The runner trailing stop ratchets up but never down.

        Args:
            bar: Current bar data
            position: Open position
            rules: Archetype exit rules (must include runner_pct, runner_trailing_atr)
            exit_context: Shared exit context with atr, unrealized_r, etc.

        Returns:
            ExitSignal to close runner if trailing stop hit, None otherwise
        """
        runner_pct = rules.get('runner_pct', 0.0)
        if runner_pct <= 0:
            return None  # No runner configured

        # Runner uses wider trailing stop
        runner_trailing_atr = rules.get('runner_trailing_atr', 3.0)

        # Check if all scale-outs are done
        if not self._all_scaleouts_executed(position, rules):
            return None

        atr = exit_context.get('atr', 0)
        current_price = bar['close']
        direction = getattr(position, 'direction', 'long')

        # Initialize runner trailing stop if not set
        if not hasattr(position, 'runner_trailing_stop') or position.runner_trailing_stop is None:
            if direction == 'long':
                position.runner_trailing_stop = current_price - runner_trailing_atr * atr
            else:
                position.runner_trailing_stop = current_price + runner_trailing_atr * atr
            position.metadata['runner_active'] = True
            logger.info(
                f"[RUNNER] Activated for {exit_context.get('archetype', 'unknown')}: "
                f"runner_pct={runner_pct:.0%}, trailing_stop={position.runner_trailing_stop:.2f}, "
                f"atr_mult={runner_trailing_atr:.1f}"
            )

        # Check if runner trailing stop is hit
        if direction == 'long':
            if current_price <= position.runner_trailing_stop:
                return ExitSignal(
                    exit_type=ExitType.TRAILING_STOP.value,
                    exit_pct=runner_pct,
                    reason=f"Runner trailing stop hit at {position.runner_trailing_stop:.2f} "
                           f"(atr_mult={runner_trailing_atr:.1f})",
                    confidence=1.0,
                    metadata={
                        **exit_context,
                        'runner_exit': True,
                        'runner_trailing_stop': position.runner_trailing_stop,
                    }
                )
            # Ratchet trailing stop up (never down)
            new_stop = current_price - runner_trailing_atr * atr
            if new_stop > position.runner_trailing_stop:
                old_stop = position.runner_trailing_stop
                position.runner_trailing_stop = new_stop
                logger.debug(
                    f"[RUNNER] Trailing stop updated: {old_stop:.2f} -> {new_stop:.2f}"
                )
        else:  # short
            if current_price >= position.runner_trailing_stop:
                return ExitSignal(
                    exit_type=ExitType.TRAILING_STOP.value,
                    exit_pct=runner_pct,
                    reason=f"Runner trailing stop hit at {position.runner_trailing_stop:.2f} "
                           f"(atr_mult={runner_trailing_atr:.1f})",
                    confidence=1.0,
                    metadata={
                        **exit_context,
                        'runner_exit': True,
                        'runner_trailing_stop': position.runner_trailing_stop,
                    }
                )
            # Ratchet trailing stop down (never up) for shorts
            new_stop = current_price + runner_trailing_atr * atr
            if new_stop < position.runner_trailing_stop:
                old_stop = position.runner_trailing_stop
                position.runner_trailing_stop = new_stop
                logger.debug(
                    f"[RUNNER] Trailing stop updated: {old_stop:.2f} -> {new_stop:.2f}"
                )

        return None

    def _compute_invalidation_score(self, bar: pd.Series) -> float:
        """
        Compute composite invalidation score from multiple bearish features.

        Research: Multi-indicator composite exits outperform single-indicator by 20-40%.
        Uses 5 orthogonal structural signals (BOS + RSI + EMA slope + volume).

        Args:
            bar: Current bar with feature data

        Returns:
            Score 0.0-5.0 (higher = more bearish structural breakdown)
        """
        score = 0.0

        # 1H Bearish BOS fired
        bos_bearish = bar.get('tf1h_bos_bearish', 0.0)
        if bos_bearish is not None and bos_bearish == bos_bearish and bos_bearish > 0:
            score += 1.0

        # 4H Bearish BOS (stronger timeframe confirmation)
        tf4h_bos = bar.get('tf4h_bos_bearish', 0.0)
        if tf4h_bos is not None and tf4h_bos == tf4h_bos and tf4h_bos > 0:
            score += 1.0

        # RSI collapsed into oversold
        rsi = bar.get('rsi_14', 50.0)
        if rsi is not None and rsi == rsi and rsi < 30:
            score += 1.0

        # EMA slope trending down
        ema_slope = bar.get('ema_slope_21', 0.0)
        if ema_slope is not None and ema_slope == ema_slope and ema_slope < -0.001:
            score += 1.0

        # Volume dried up (no buying interest)
        vol_ratio = bar.get('volume_ratio', 1.0)
        if vol_ratio is not None and vol_ratio == vol_ratio and vol_ratio < 0.5:
            score += 1.0

        return score

    def _check_invalidation(
        self,
        bar: pd.Series,
        position: "Position",
        rules: Dict,
        context: RuntimeContext,
        exit_context: Dict
    ) -> Optional[ExitSignal]:
        """
        Check pattern invalidation conditions.

        V2: Composite feature-based scoring for wick_trap and retest_cluster.
        Other archetypes use structural checks (funding flip, OI reversal, etc.).

        Research backing: Multi-indicator composite exits outperform single-indicator
        by 20-40% (LiteFinance 2026, MDPI TP/SL Strategies).

        Args:
            bar: Current bar
            position: Open position
            rules: Archetype rules
            context: Runtime context
            exit_context: Shared exit context

        Returns:
            ExitSignal if invalidation detected, None otherwise
        """
        archetype = _resolve_exit_name(exit_context['archetype'])

        # --- COMPOSITE INVALIDATION V2 (wick_trap, retest_cluster) ---
        # Instead of single price-level check (which was 100% losers),
        # use multi-feature structural breakdown score >= 3/5
        if archetype in ('wick_trap', 'retest_cluster'):
            score = self._compute_invalidation_score(bar)
            if score >= 4.0:
                return ExitSignal(
                    exit_type=ExitType.INVALIDATION.value,
                    exit_pct=1.0,
                    reason=f"Composite invalidation: {archetype} structural breakdown score={score:.1f}/5.0",
                    confidence=min(0.7 + score * 0.06, 1.0),
                    metadata={**exit_context, 'invalidation_score': score}
                )
            return None

        # --- STRUCTURAL INVALIDATION (other archetypes) ---

        # S1 (Liquidity Vacuum) - Previous low taken out
        if archetype == 'liquidity_vacuum':
            prev_low = position.metadata.get('entry_prev_low')
            if prev_low and bar['close'] < prev_low:
                return ExitSignal(
                    exit_type=ExitType.INVALIDATION.value,
                    exit_pct=1.0,
                    reason=f"S1 invalidation: previous low {prev_low:.2f} taken out",
                    confidence=1.0,
                    metadata={**exit_context, 'prev_low': prev_low}
                )

        # S4 (Funding Divergence) - Funding flip reversal
        elif archetype == 'funding_divergence':
            funding_z = bar.get('funding_z_score', 0.0)
            entry_funding_z = position.metadata.get('entry_funding_z', 0.0)

            if entry_funding_z < -5.0 and funding_z > -1.0:
                return ExitSignal(
                    exit_type=ExitType.INVALIDATION.value,
                    exit_pct=1.0,
                    reason=f"S4 invalidation: funding normalized ({entry_funding_z:.2f} -> {funding_z:.2f})",
                    confidence=0.9,
                    metadata={**exit_context, 'funding_z': funding_z}
                )

        # S5 (Long Squeeze) - OI divergence reversal
        elif archetype == 'long_squeeze':
            oi_delta_pct = bar.get('oi_delta_pct', 0.0)
            entry_oi_delta = position.metadata.get('entry_oi_delta', 0.0)

            if entry_oi_delta < -10.0 and oi_delta_pct > 5.0:
                return ExitSignal(
                    exit_type=ExitType.INVALIDATION.value,
                    exit_pct=1.0,
                    reason=f"S5 invalidation: OI rebounded ({entry_oi_delta:.1f}% -> {oi_delta_pct:.1f}%)",
                    confidence=0.85,
                    metadata={**exit_context, 'oi_delta_pct': oi_delta_pct}
                )

        return None

    def _check_distress_exit(
        self,
        bar: pd.Series,
        position: "Position",
        rules: Dict,
        context: RuntimeContext,
        exit_context: Dict
    ) -> Optional[ExitSignal]:
        """
        Feature-based early partial exit for positions showing distress signals.

        Research backing:
        - de Prado triple barrier + feature-based 4th dimension
        - Lagged features (RSI, MA slope, volatility) reliably predict drawdowns
        - Our data: 100% of losses are SL exits, losers hit SL 2.8x faster
        - dd_score < 0.10 = 67% loss rate in our 914-trade backtest

        Trigger conditions (ALL must be true):
        1. Trade is underwater (unrealized PnL < -0.2R)
        2. Position held 3-24 hours (not too early, not near SL)
        3. Composite distress score >= 3 of 5 features

        Action: Exit 50% at close price. Remaining 50% keeps original SL.
        Cooldown: Max 1 distress exit per position.

        Args:
            bar: Current bar with feature data
            position: Open position
            rules: Archetype rules
            context: Runtime context
            exit_context: Shared exit context

        Returns:
            ExitSignal with exit_pct=0.5 if distress detected, None otherwise
        """
        if not rules.get('distress_exit_enabled', False):
            return None

        # Already used distress exit on this position
        if position.metadata.get('distress_exit_used', False):
            return None

        unrealized_r = exit_context['unrealized_r']
        hours = exit_context['hours_in_position']

        # Gate 1: Must be underwater (< -0.2R)
        if unrealized_r >= -0.2:
            return None

        # Gate 2: Held 3-24 hours (not too early, not near SL already)
        if hours < 3.0 or hours > 24.0:
            return None

        # Gate 3: Composite distress score >= 3/5
        score = 0.0
        details = []

        # dd_score < 0.10 (deep drawdown regime)
        # dd_score = max(1 - drawdown_persistence, 0) — computed from feature store
        dd_persist = bar.get('drawdown_persistence', 0.5)
        if dd_persist is not None and dd_persist == dd_persist:  # NaN guard
            dd_score = max(1.0 - dd_persist, 0.0)
        else:
            dd_score = 0.5  # neutral default
        if dd_score < 0.10:
            score += 1.0
            details.append(f"dd={dd_score:.2f}")

        # chop_score > 0.40 (choppy market)
        chop = bar.get('chop_score', 0.3)
        if chop is not None and chop == chop and chop > 0.40:
            score += 1.0
            details.append(f"chop={chop:.2f}")

        # rsi_14 < 35 for longs (momentum collapsed)
        rsi = bar.get('rsi_14', 50.0)
        direction = getattr(position, 'direction', 'long')
        if direction == 'long':
            if rsi is not None and rsi == rsi and rsi < 35:
                score += 1.0
                details.append(f"rsi={rsi:.1f}")
        else:  # short
            if rsi is not None and rsi == rsi and rsi > 65:
                score += 1.0
                details.append(f"rsi={rsi:.1f}")

        # ema_slope_21 < -0.001 (trend turned against for longs)
        ema_slope = bar.get('ema_slope_21', 0.0)
        if ema_slope is not None and ema_slope == ema_slope:
            if (direction == 'long' and ema_slope < -0.001) or \
               (direction == 'short' and ema_slope > 0.001):
                score += 1.0
                details.append(f"ema_slope={ema_slope:.4f}")

        # volume_ratio < 0.5 (buying dried up)
        vol_ratio = bar.get('volume_ratio', 1.0)
        if vol_ratio is not None and vol_ratio == vol_ratio and vol_ratio < 0.5:
            score += 1.0
            details.append(f"vol_ratio={vol_ratio:.2f}")

        if score >= 4.0:
            position.metadata['distress_exit_used'] = True
            return ExitSignal(
                exit_type=ExitType.DISTRESS.value,
                exit_pct=0.5,
                reason=f"Distress half-exit: score={score:.0f}/5 ({', '.join(details)}), R={unrealized_r:.2f}",
                confidence=min(0.6 + score * 0.08, 1.0),
                metadata={**exit_context, 'distress_score': score, 'distress_details': details}
            )

        return None

    def _check_profit_targets(
        self,
        bar: pd.Series,
        position: "Position",
        rules: Dict,
        context: RuntimeContext,
        exit_context: Dict
    ) -> Optional[ExitSignal]:
        """
        Check scale-out and profit target conditions.

        Implements R-multiple based scale-outs:
        - Default: Scale out 20% at 0.5R, 20% at 1.0R, 30% at 2.0R
        - Archetype-specific adjustments

        Args:
            bar: Current bar
            position: Open position
            rules: Archetype rules
            context: Runtime context
            exit_context: Shared exit context

        Returns:
            ExitSignal if profit target hit, None otherwise
        """
        unrealized_r = exit_context['unrealized_r']

        # Read profit_targets from config
        profit_targets = rules.get('profit_targets', [])
        if profit_targets:
            scale_levels = [pt['r_multiple'] for pt in profit_targets]
            scale_pcts = [pt['exit_pct'] for pt in profit_targets]
        else:
            scale_levels = rules.get('scale_out_levels', [0.5, 1.0, 2.0])
            scale_pcts = rules.get('scale_out_pcts', [0.2, 0.2, 0.3])

        # Track which scale-outs already executed
        executed_scales = position.metadata.get('executed_scale_outs', [])

        # Check each scale level in order
        for level, pct in zip(scale_levels, scale_pcts):
            if unrealized_r >= level and level not in executed_scales:
                is_first_scaleout = len(executed_scales) == 0

                # Mark this scale level as executed
                executed_scales.append(level)
                position.metadata['executed_scale_outs'] = executed_scales

                # BREAK-EVEN STOP: after the first scale-out fires, move stop to entry.
                # This eliminates the payoff asymmetry: we've locked partial profit, so
                # the remaining position should never lose more than 0. The trailing stop
                # only moves UP from here, so break-even becomes a permanent floor.
                if is_first_scaleout:
                    entry = position.entry_price
                    direction = getattr(position, 'direction', 'long')
                    if direction == 'long' and entry > position.stop_loss:
                        old_stop = position.stop_loss
                        position.stop_loss = entry
                        position.metadata['breakeven_stop_activated'] = True
                        logger.info(
                            f"[BREAK-EVEN] First scale-out at {level:.1f}R — stop moved to entry: "
                            f"{old_stop:.2f} -> {entry:.2f}"
                        )
                    elif direction == 'short' and entry < position.stop_loss:
                        old_stop = position.stop_loss
                        position.stop_loss = entry
                        position.metadata['breakeven_stop_activated'] = True
                        logger.info(
                            f"[BREAK-EVEN] First scale-out at {level:.1f}R — stop moved to entry: "
                            f"{old_stop:.2f} -> {entry:.2f}"
                        )

                return ExitSignal(
                    exit_type=ExitType.PROFIT_TARGET.value,
                    exit_pct=pct,
                    reason=f"Scale-out at {level:.1f}R (exit {pct*100:.0f}%)",
                    confidence=1.0,
                    metadata={**exit_context, 'scale_level': level}
                )

        # Check archetype-specific profit targets
        archetype = _resolve_exit_name(exit_context['archetype'])

        # S1 (Liquidity Vacuum) - Rally to previous high
        if archetype == 'liquidity_vacuum':
            prev_high = position.metadata.get('entry_prev_high')
            if prev_high and bar['close'] >= prev_high * 0.98:  # Within 2% of previous high
                # Check if already scaled out at this level
                if not position.metadata.get('scaled_at_prev_high', False):
                    position.metadata['scaled_at_prev_high'] = True
                    return ExitSignal(
                        exit_type=ExitType.PROFIT_TARGET.value,
                        exit_pct=0.3,  # Scale out 30%
                        reason=f"S1 target: rallied to previous high {prev_high:.2f}",
                        confidence=0.9,
                        metadata={**exit_context, 'prev_high': prev_high}
                    )

        # S4 (Funding Divergence) - Moon-bag at all-time high
        elif archetype == 'funding_divergence':
            if unrealized_r >= 3.0:  # +3R profit
                # Check if already took moon-bag
                if not position.metadata.get('moon_bag_taken', False):
                    position.metadata['moon_bag_taken'] = True
                    return ExitSignal(
                        exit_type=ExitType.PROFIT_TARGET.value,
                        exit_pct=0.1,  # Keep 10% moon-bag
                        reason="S4 moon-bag: keep 10% for extended rally",
                        confidence=0.8,
                        metadata={**exit_context, 'moon_bag': True}
                    )

        return None

    def _check_time_based(
        self,
        bar: pd.Series,
        position: "Position",
        rules: Dict,
        context: RuntimeContext,
        exit_context: Dict
    ) -> Optional[ExitSignal]:
        """
        Check max hold period with regime-adaptive adjustment.

        Different archetypes have different time horizons:
        - Quick momentum plays: 24-48 hours
        - Structural setups: 5-7 days
        - Macro plays: 10-14 days

        Regime adjustments (from 2025 research):
        - Risk-on: 1.5x longer holds (let winners run)
        - Neutral: 1.0x (normal)
        - Risk-off: 0.5x shorter holds (reduce exposure)
        - Crisis: 0.25x shortest holds (exit quickly)

        Args:
            bar: Current bar
            position: Open position
            rules: Archetype rules
            context: Runtime context
            exit_context: Shared exit context

        Returns:
            ExitSignal if max hold period exceeded, None otherwise
        """
        hours_in_position = exit_context['hours_in_position']
        unrealized_r = exit_context['unrealized_r']

        # Get regime from context
        regime = context.regime_label

        # Apply regime adjustment to max hold period
        adjusted_params = self._get_regime_adjusted_params(rules, regime, unrealized_r)
        max_hold = adjusted_params['max_hold_hours']

        if hours_in_position >= max_hold:
            return ExitSignal(
                exit_type=ExitType.TIME_EXIT.value,
                exit_pct=1.0,
                reason=f"Time exit ({regime} regime): held {hours_in_position:.1f}h (max: {max_hold:.0f}h)",
                confidence=0.8,
                metadata={**exit_context, 'max_hold_hours': max_hold, 'regime': regime}
            )

        return None

    def _check_reason_gone(
        self,
        bar: pd.Series,
        position: "Position",
        rules: Dict,
        context: RuntimeContext,
        exit_context: Dict
    ) -> Optional[ExitSignal]:
        """
        Check if entry reason invalidated.

        Archetype-specific reason-gone checks:
        - S4: Funding flip (reason gone)
        - S5: OI rebound (reason gone)
        - Volume archetypes: Volume fade check

        Args:
            bar: Current bar
            position: Open position
            rules: Archetype rules
            context: Runtime context
            exit_context: Shared exit context

        Returns:
            ExitSignal if reason gone, None otherwise
        """
        archetype = _resolve_exit_name(exit_context['archetype'])

        # S4 (Funding Divergence) - Already checked in invalidation
        # S5 (Long Squeeze) - Already checked in invalidation

        # Volume-based archetypes - Check volume fade
        volume_archetypes = ['volume_fade_chop', 'volume_exhaustion', 'expansion_exhaustion']
        if archetype in volume_archetypes:
            volume = bar.get('volume', 0)
            entry_volume = position.metadata.get('entry_volume', volume)

            # Volume faded to <50% of entry level
            if volume < entry_volume * 0.5:
                return ExitSignal(
                    exit_type=ExitType.REASON_GONE.value,
                    exit_pct=1.0,
                    reason=f"Volume fade: {volume:.0f} < 50% of entry ({entry_volume:.0f})",
                    confidence=0.85,
                    metadata={**exit_context, 'volume': volume, 'entry_volume': entry_volume}
                )

        # Momentum archetypes - Check ADX fade
        momentum_archetypes = ['trap_within_trend', 'fvg_continuation']
        if archetype in momentum_archetypes:
            adx = bar.get('adx_14', 20.0)
            entry_adx = position.metadata.get('entry_adx', adx)

            # ADX dropped below 25 (trend weakening)
            if entry_adx > 25 and adx < 25:
                return ExitSignal(
                    exit_type=ExitType.REASON_GONE.value,
                    exit_pct=0.5,  # Partial exit on momentum fade
                    reason=f"Momentum fade: ADX {adx:.1f} < 25 (was {entry_adx:.1f})",
                    confidence=0.75,
                    metadata={**exit_context, 'adx': adx, 'entry_adx': entry_adx}
                )

        return None

    def _update_trailing_stop(
        self,
        bar: pd.Series,
        position: "Position",
        rules: Dict,
        context: RuntimeContext,
        exit_context: Dict
    ) -> None:
        """
        Update trailing stop with progressive tightening and regime adaptation.

        Trailing logic:
        - Activate after reaching trailing_start_r (default +1R)
        - Start with wide 3.0x ATR for crypto volatility
        - Progressive tightening: 3.0x -> 2.5x -> 2.0x -> 1.5x as profit increases
        - Regime adjustments: widen in bear/crisis, tighten in bull
        - Archetype-specific adjustments for certain patterns

        Research-based: Crypto requires 3-4x ATR (vs 1.5-2.5x for forex)
        due to 5x higher volatility. Progressive tightening provides 15%
        performance boost vs fixed stops. 2025 study showed regime adaptation
        achieved Sharpe 2.24 in bear markets.

        Args:
            bar: Current bar
            position: Open position
            rules: Archetype rules
            context: Runtime context
            exit_context: Shared exit context
        """
        unrealized_r = exit_context['unrealized_r']
        atr = exit_context['atr']

        # Only trail if profitable enough
        trailing_start = rules.get('trailing_start_r', 0.5)
        if unrealized_r < trailing_start:
            return

        # Get regime from context
        regime = context.regime_label

        # Apply regime adjustment to get regime-adapted base multiplier
        adjusted_params = self._get_regime_adjusted_params(rules, regime, unrealized_r)
        regime_adjusted_mult = adjusted_params['trailing_atr_mult']

        # Apply progressive tightening based on R achieved
        # Note: We apply progressive tightening to the regime-adjusted base
        trailing_mult = self._get_adjusted_trailing_mult(regime_adjusted_mult, unrealized_r)

        # Chop-aware tightening: reduce trailing width in choppy markets
        # Research: Clare et al. (2013) — volatility-adaptive stops reduce max DD 45-65%
        # Choppy markets = higher whipsaw risk → tighter stops capture profit before reversal
        chop = bar.get('chop_score', 0.3)
        if chop is not None and chop == chop:  # NaN guard
            if chop > 0.45:
                trailing_mult *= 0.75  # 25% tighter in high chop
            elif chop > 0.35:
                trailing_mult *= 0.88  # 12% tighter in moderate chop

        # Archetype-specific tightening overrides
        archetype = _resolve_exit_name(exit_context['archetype'])

        # S1 (Liquidity Vacuum) - Extra tightening after 50% rally
        if archetype == 'liquidity_vacuum':
            rally_pct = (bar['close'] - position.entry_price) / position.entry_price
            if position.direction == 'long' and rally_pct > 0.5:
                trailing_mult = min(trailing_mult, 1.5)  # Cap at 1.5 ATR

        # S5 (Long Squeeze) - Tighten on ADX fade (momentum loss)
        elif archetype == 'long_squeeze':
            adx = bar.get('adx_14', 20.0)
            if adx < 25:
                trailing_mult = min(trailing_mult, 1.5)  # Cap at 1.5 ATR

        # Calculate new trailing stop
        if position.direction == 'long':
            new_stop = bar['close'] - (trailing_mult * atr)
            # Only move stop up, never down
            if new_stop > position.stop_loss:
                old_stop = position.stop_loss
                position.stop_loss = new_stop
                logger.debug(
                    f"Trailing stop updated ({regime} regime): {old_stop:.2f} -> {new_stop:.2f} "
                    f"(R={unrealized_r:.2f}, regime_mult={regime_adjusted_mult:.1f}, final_mult={trailing_mult:.1f})"
                )
        else:  # Short
            new_stop = bar['close'] + (trailing_mult * atr)
            # Only move stop down, never up
            if new_stop < position.stop_loss:
                old_stop = position.stop_loss
                position.stop_loss = new_stop
                logger.debug(
                    f"Trailing stop updated ({regime} regime): {old_stop:.2f} -> {new_stop:.2f} "
                    f"(R={unrealized_r:.2f}, regime_mult={regime_adjusted_mult:.1f}, final_mult={trailing_mult:.1f})"
                )


def create_default_exit_config() -> Dict:
    """
    Create default exit configuration with archetype-specific rules.

    Returns:
        Default exit configuration dict
    """
    return {
        'enable_scale_outs': True,
        'enable_time_exits': True,
        'enable_trailing': True,

        'exit_rules': {
            # --- Existing 7 archetypes (with unique exit profiles) ---

            # Liquidity Vacuum — structural setup, 5 day hold
            'liquidity_vacuum': {
                'max_hold_hours': 120,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.2, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 2.0,
                'runner_pct': 0.15,
                'runner_trailing_atr': 3.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Funding Divergence — macro play, 10 day hold, moon-bag
            'funding_divergence': {
                'max_hold_hours': 240,
                'scale_out_levels': [0.5, 1.0, 2.0, 3.0],
                'scale_out_pcts': [0.2, 0.2, 0.2, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.5,
                'runner_pct': 0.10,
                'runner_trailing_atr': 3.5,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Long Squeeze — fast momentum short, 3 day hold
            'long_squeeze': {
                'max_hold_hours': 72,
                'scale_out_levels': [0.5, 1.0, 1.5],
                'scale_out_pcts': [0.3, 0.3, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Spring/Trap Reversal — Wyckoff reversal, 7 day hold
            'trap_reversal': {
                'max_hold_hours': 168,
                'scale_out_levels': [1.0, 2.0, 3.0],
                'scale_out_pcts': [0.3, 0.3, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.0,
                'runner_pct': 0.10,
                'runner_trailing_atr': 3.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Order Block Retest — structural, 4 day hold
            'order_block_retest': {
                'max_hold_hours': 96,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.25, 0.25, 0.4],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Trap Within Trend — proven config, 7 day hold
            'trap_within_trend': {
                'max_hold_hours': 168,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.2, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.0,
                'runner_pct': 0.15,
                'runner_trailing_atr': 2.5,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Wick Trap — proven config, 7 day hold + composite invalidation V2
            'wick_trap': {
                'max_hold_hours': 168,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.2, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.0,
                'runner_pct': 0.15,
                'runner_trailing_atr': 2.5,
                'invalidation_checks': True,  # V2 composite scoring (not price-level)
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # --- NEW archetypes (missing from original) ---

            # Liquidity Sweep — proven config, 7 day hold
            'liquidity_sweep': {
                'max_hold_hours': 168,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.2, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.0,
                'runner_pct': 0.15,
                'runner_trailing_atr': 2.5,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Retest Cluster — proven config, 7 day hold + composite invalidation V2
            'retest_cluster': {
                'max_hold_hours': 168,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.2, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.0,
                'runner_pct': 0.15,
                'runner_trailing_atr': 2.5,
                'invalidation_checks': True,  # V2 composite scoring (not price-level)
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # FVG Continuation — fast momentum, 3 day hold
            'fvg_continuation': {
                'max_hold_hours': 72,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.3, 0.3, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Failed Continuation — medium reversal, 4 day hold
            'failed_continuation': {
                'max_hold_hours': 96,
                'scale_out_levels': [0.5, 1.0, 1.5],
                'scale_out_pcts': [0.25, 0.35, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Liquidity Compression — structural buildup, 5 day hold
            'liquidity_compression': {
                'max_hold_hours': 120,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.3, 0.4],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 2.0,
                'runner_pct': 0.10,
                'runner_trailing_atr': 2.5,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Confluence Breakout — breakout momentum, 4 day hold
            'confluence_breakout': {
                'max_hold_hours': 96,
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.25, 0.25, 0.4],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Exhaustion Reversal — fast reversal, 2 day hold
            'exhaustion_reversal': {
                'max_hold_hours': 48,
                'scale_out_levels': [0.5, 1.0, 1.5],
                'scale_out_pcts': [0.3, 0.4, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # OI Divergence — whale exit detection, 2 day hold
            'oi_divergence': {
                'max_hold_hours': 48,
                'scale_out_levels': [1.0, 1.5, 2.5],
                'scale_out_pcts': [0.3, 0.3, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Whipsaw — fast reversal, 2 day hold
            'whipsaw': {
                'max_hold_hours': 48,
                'scale_out_levels': [0.5, 1.0, 1.5],
                'scale_out_pcts': [0.3, 0.3, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },

            # Volume Fade Chop — fast mean-reversion, 2 day hold
            'volume_fade_chop': {
                'max_hold_hours': 48,
                'scale_out_levels': [0.5, 1.0, 1.5],
                'scale_out_pcts': [0.3, 0.4, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'runner_pct': 0.0,
                'runner_trailing_atr': 2.0,
                'invalidation_checks': False,
                'reason_gone_checks': False,
                'distress_exit_enabled': True,
            },
        }
    }


# Example usage
if __name__ == '__main__':
    # Demo configuration
    config = create_default_exit_config()

    # Initialize exit logic
    exit_logic = ExitLogic(config)

    # Demo: Check S1 exit
    from engine.models.base import Position
    import pandas as pd

    position = Position(
        direction='long',
        entry_price=50000.0,
        entry_time=pd.Timestamp('2024-01-01 10:00'),
        size=1000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        metadata={
            'entry_prev_low': 48500.0,
            'entry_prev_high': 51000.0,
            'archetype': 'liquidity_vacuum'
        }
    )

    bar = pd.Series({
        'open': 50800.0,
        'high': 51200.0,
        'low': 50700.0,
        'close': 51000.0,
        'volume': 1000000,
        'atr_14': 500.0
    }, name=pd.Timestamp('2024-01-01 12:00'))

    # Mock context
    from engine.runtime.context import RuntimeContext
    context = RuntimeContext(
        ts=bar.name,
        row=bar,
        regime_probs={'risk_off': 1.0},
        regime_label='risk_off',
        adapted_params={},
        thresholds={}
    )

    # Check exit
    exit_signal = exit_logic.check_exit(
        bar=bar,
        position=position,
        archetype='liquidity_vacuum',
        context=context
    )

    if exit_signal:
        print(f"Exit triggered: {exit_signal.exit_type}")
        print(f"Exit {exit_signal.exit_pct*100:.0f}% of position")
        print(f"Reason: {exit_signal.reason}")
    else:
        print("No exit conditions met - hold position")
