"""
Archetype-Specific Exit Logic System

Implements comprehensive exit rules based on extracted trading knowledge:
1. Invalidation exits: Pattern/structure breaks
2. Scale-out exits: Rally targets, previous highs
3. Profit protection: Moon-bags, trailing stops
4. Time-based exits: Max hold period per archetype
5. Reason-gone exits: Funding flip, OI reversal, volume fade

Each archetype has specific exit conditions tailored to its entry logic.
Exit signals are checked in priority order to prevent double-triggers.

REGIME USAGE:
- ENTRY: Bypassed (bypass_entry_filtering=true) - all archetypes allowed
- EXIT: Active - regime adjusts stops, time limits, profit targets
- SIZING: Active via risk_temperature (probabilistic scaling)

This aligns with research: regime for risk management, not entry filtering.
Research shows regime lag causes 93% signal waste when used for binary filtering,
but regime-adaptive exits achieved Sharpe 2.24 in bear markets (2025 study).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, TYPE_CHECKING
from enum import Enum
import pandas as pd

# Use TYPE_CHECKING to avoid circular import
# Position is only used for type hints
if TYPE_CHECKING:
    from engine.models.base import Position

from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


class ExitType(Enum):
    """Exit signal types ordered by priority."""
    INVALIDATION = "invalidation"  # Highest priority - pattern broken
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
    1. Invalidation (pattern/structure breaks)
    2. Profit targets (scale-outs)
    3. Time-based (max hold period)
    4. Reason-gone (entry condition reversed)
    5. Trailing stops (profit protection)
    """

    def __init__(self, config: Dict):
        """
        Initialize exit logic with configuration.

        Args:
            config: Full configuration dict with exit rules per archetype
        """
        # DEBUG: Log what config structure we receive
        print(f"\n[EXIT_LOGIC INIT DEBUG] Config keys received: {list(config.keys())}")
        if 'archetypes' in config:
            print(f"[EXIT_LOGIC INIT DEBUG] archetypes found with {len(config['archetypes'])} entries")
            print(f"[EXIT_LOGIC INIT DEBUG] archetype keys: {list(config['archetypes'].keys())}")
            if 'failed_rally' in config['archetypes']:
                print(f"[EXIT_LOGIC INIT DEBUG] failed_rally config: {config['archetypes']['failed_rally']}")
        else:
            print(f"[EXIT_LOGIC INIT DEBUG] No 'archetypes' key found in config!")
        if 'exit_rules' in config:
            print(f"[EXIT_LOGIC INIT DEBUG] exit_rules found with {len(config['exit_rules'])} entries")

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

        logger.info(f"[EXIT_LOGIC BUILD] Loading exit rules from config")
        logger.info(f"  Config has 'exit_rules' key: {'exit_rules' in self.config}")
        logger.info(f"  Config top-level keys: {list(self.config.keys())}")

        print(f"[EXIT_LOGIC BUILD DEBUG] exit_config has {len(exit_config)} archetypes from 'exit_rules'")
        if 'failed_rally' in exit_config:
            print(f"[EXIT_LOGIC BUILD DEBUG] failed_rally in exit_config: {exit_config['failed_rally']}")

        # Default exit rules (fallback for archetypes without specific rules)
        default_rules = {
            'max_hold_hours': 168,  # 7 days
            'scale_out_levels': [0.5, 1.0, 2.0],  # R-multiples
            'scale_out_pcts': [0.2, 0.2, 0.3],  # Exit 20%, 20%, 30% at each level
            'trailing_start_r': 0.5,  # Start trailing after +0.5R (earlier protection)
            'trailing_atr_mult': 2.0,  # Trail 2 ATR behind peak
            'runner_pct': 0.0,  # No runner by default (0 = disabled)
            'runner_trailing_atr': 3.0,  # Runner uses wider trailing stop
            'invalidation_checks': True,
            'reason_gone_checks': True
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

            # DEBUG: Log final rules for failed_rally
            if archetype_key == 'failed_rally':
                print(f"[EXIT_LOGIC BUILD DEBUG] Final failed_rally rules keys: {list(archetype_rules.keys())}")
                print(f"[EXIT_LOGIC BUILD DEBUG] Final failed_rally has profit_targets: {'profit_targets' in archetype_rules}")
                if 'profit_targets' in archetype_rules:
                    print(f"[EXIT_LOGIC BUILD DEBUG] profit_targets value: {archetype_rules['profit_targets']}")

            # DEBUG: Log trap_reversal and whipsaw rules at initialization
            if archetype_key == 'trap_reversal':
                logger.info(f"[EXIT_LOGIC INIT] trap_reversal exit rules:")
                logger.info(f"  max_hold_hours: {archetype_rules.get('max_hold_hours')}")
                logger.info(f"  scale_out_levels: {archetype_rules.get('scale_out_levels')}")
                logger.info(f"  trailing_start_r: {archetype_rules.get('trailing_start_r')}")
                logger.info(f"  trailing_atr_mult: {archetype_rules.get('trailing_atr_mult')}")
                logger.info(f"  Config override applied: {config_override_applied}")

            if archetype_key == 'whipsaw':
                logger.info(f"[EXIT_LOGIC INIT] whipsaw exit rules:")
                logger.info(f"  max_hold_hours: {archetype_rules.get('max_hold_hours')}")
                logger.info(f"  scale_out_levels: {archetype_rules.get('scale_out_levels')}")
                logger.info(f"  trailing_start_r: {archetype_rules.get('trailing_start_r')}")
                logger.info(f"  trailing_atr_mult: {archetype_rules.get('trailing_atr_mult')}")
                logger.info(f"  Config override applied: {config_override_applied}")

            rules[archetype_key] = archetype_rules

        return rules

    def _get_all_archetypes(self) -> list:
        """Get list of all supported archetypes."""
        return [
            # Bear-biased archetypes (short)
            'liquidity_vacuum',  # S1
            'failed_rally',  # S2
            'whipsaw',  # S3
            'funding_divergence',  # S4
            'long_squeeze',  # S5
            'alt_rotation_down',  # S6
            'curve_inversion',  # S7
            'volume_fade_chop',  # S8
            # Bull-biased archetypes (long)
            'trap_reversal',  # A (Spring)
            'order_block_retest',  # B
            'fvg_continuation',  # C
            'failed_continuation',  # D
            'liquidity_compression',  # E
            'expansion_exhaustion',  # F
            're_accumulate',  # G
            'trap_within_trend',  # H
            'wick_trap',  # K
            'volume_exhaustion',  # L
            'ratio_coil_break',  # M
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
        # Get archetype-specific rules
        rules = self.exit_rules.get(archetype)
        if not rules:
            logger.warning(f"No exit rules for archetype '{archetype}', using defaults")
            rules = self.exit_rules.get('liquidity_vacuum')  # Use S1 as default

        # DEBUG: Log exit rules being used for trap_reversal (every position check)
        if archetype == 'trap_reversal':
            if not hasattr(self, '_trap_reversal_rules_logged'):
                logger.info(f"[EXIT RULES DEBUG] trap_reversal rules loaded:")
                logger.info(f"  max_hold_hours: {rules.get('max_hold_hours', 'MISSING')}")
                logger.info(f"  scale_out_levels: {rules.get('scale_out_levels', 'MISSING')}")
                logger.info(f"  trailing_start_r: {rules.get('trailing_start_r', 'MISSING')}")
                logger.info(f"  trailing_atr_mult: {rules.get('trailing_atr_mult', 'MISSING')}")
                self._trap_reversal_rules_logged = True

        # Calculate position metrics
        hours_in_position = (bar.name - position.entry_time).total_seconds() / 3600
        atr = bar.get('atr_14', bar.get('atr', bar['close'] * 0.02))
        unrealized_r = self._calculate_unrealized_r(position, bar['close'], atr)

        # DEBUG: Log every 100th check to see if this is being called
        if not hasattr(self, '_check_counter'):
            self._check_counter = 0
        self._check_counter += 1

        if self._check_counter % 100 == 0 or hours_in_position > 72:
            logger.info(f"[EXIT CHECK #{self._check_counter}] Archetype: {archetype}, Hours: {hours_in_position:.1f}, R: {unrealized_r:.2f}, "
                       f"enable_time: {self.enable_time_exits}, enable_scale: {self.enable_scale_outs}, max_hold: {rules.get('max_hold_hours', 'N/A')}")

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

        # 1. CHECK INVALIDATION (highest priority)
        if rules.get('invalidation_checks', True):
            if invalidation := self._check_invalidation(bar, position, rules, context, exit_context):
                return invalidation

        # 2. CHECK PROFIT TARGETS / SCALE-OUTS (regime-adjusted levels)
        if self.enable_scale_outs:
            if profit_target := self._check_profit_targets(bar, position, adjusted_rules, context, exit_context):
                return profit_target

        # 3. CHECK TIME-BASED EXIT
        if self.enable_time_exits:
            if time_exit := self._check_time_based(bar, position, rules, context, exit_context):
                return time_exit

        # 4. CHECK REASON-GONE (entry condition reversed)
        if rules.get('reason_gone_checks', True):
            if reason_gone := self._check_reason_gone(bar, position, rules, context, exit_context):
                return reason_gone

        # 5. UPDATE TRAILING STOP (no exit, just update)
        if self.enable_trailing:
            self._update_trailing_stop(bar, position, rules, context, exit_context)

        # 6. CHECK RUNNER EXIT (trailing stop for runner portion after all scale-outs)
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

        Research: 2025 study achieved Sharpe 2.24 in bear markets via regime adaptation.
        Dynamic parameter adjustment prevents premature exits in volatile conditions
        and protects capital in downturns.

        Args:
            base_params: Base exit parameters from config
                - trailing_atr_mult: Base trailing stop multiplier (e.g., 3.0)
                - max_hold_hours: Base max hold period in hours (e.g., 168)
            regime: Current regime ('risk_on', 'neutral', 'risk_off', 'crisis')
            r_multiple: Current R-multiple achieved (for adaptive logic)

        Returns:
            Adjusted parameters dict with:
                - trailing_atr_mult: Regime-adjusted trailing multiplier
                - max_hold_hours: Regime-adjusted max hold period

        Regime Factors:
            risk_on (bullish):
                - Tighter trails (0.67x = 2.0 ATR from 3.0 base) - lock in profits faster
                - Longer holds (1.5x = 67d from 45d base) - let winners run

            neutral (balanced):
                - Normal parameters (1.0x factors) - standard risk management

            risk_off (bearish):
                - Wider trails (1.17x = 3.5 ATR from 3.0 base) - avoid whipsaws
                - Shorter holds (0.5x = 22d from 45d base) - reduce exposure

            crisis (extreme):
                - Widest trails (1.33x = 4.0 ATR from 3.0 base) - survive volatility
                - Shortest holds (0.25x = 11d from 45d base) - exit quickly

        Example:
            Base: trailing_atr_mult=3.0, max_hold_hours=1080 (45 days)
            Risk-off regime: trailing_atr_mult=3.5, max_hold_hours=540 (22 days)
            Crisis regime: trailing_atr_mult=4.0, max_hold_hours=270 (11 days)
        """
        # Regime adjustment factors based on 2025 research
        regime_factors = {
            'risk_on': {
                'trailing_mult_factor': 0.67,  # Tighter trails (2.0x from 3.0x base)
                'time_exit_factor': 1.5,       # Longer holds (45d → 67d)
            },
            'neutral': {
                'trailing_mult_factor': 1.0,   # Normal (3.0x base)
                'time_exit_factor': 1.0,       # Normal holds
            },
            'risk_off': {
                'trailing_mult_factor': 1.17,  # Wider trails (3.5x from 3.0x base)
                'time_exit_factor': 0.5,       # Shorter holds (45d → 22d)
            },
            'crisis': {
                'trailing_mult_factor': 1.33,  # Widest trails (4.0x from 3.0x base)
                'time_exit_factor': 0.25,      # Shortest holds (45d → 11d)
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
                'max_hold_multiplier': 1.5,     # Hold longer in bull
                'scale_level_multiplier': 1.3,   # Wider targets in bull
                'trailing_atr_multiplier': 1.2,  # Wider trailing in bull
            },
            'neutral': {
                'max_hold_multiplier': 1.0,
                'scale_level_multiplier': 1.0,
                'trailing_atr_multiplier': 1.0,
            },
            'risk_off': {
                'max_hold_multiplier': 0.7,     # Shorter holds in bear
                'scale_level_multiplier': 0.8,   # Tighter targets
                'trailing_atr_multiplier': 0.8,
            },
            'crisis': {
                'max_hold_multiplier': 0.5,     # Very short in crisis
                'scale_level_multiplier': 0.6,
                'trailing_atr_multiplier': 0.6,
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

        Archetype-specific invalidation logic:
        - S1: Previous low taken out
        - S4: Funding flip reversal (funding_Z > -1.0 after < -5.0)
        - S5: OI divergence reversal
        - A: Spring invalidation (close below spring low)
        - B: Close under order block support
        - H: Clean close below support
        - K: Wick invalidation (close below wick low)

        Args:
            bar: Current bar
            position: Open position
            rules: Archetype rules
            context: Runtime context
            exit_context: Shared exit context

        Returns:
            ExitSignal if invalidation detected, None otherwise
        """
        archetype = exit_context['archetype']

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

            # If entered on extreme negative funding (< -5.0), exit if funding normalizes (> -1.0)
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

            # If entered on OI drop (< -10%), exit if OI rebounds (> +5%)
            if entry_oi_delta < -10.0 and oi_delta_pct > 5.0:
                return ExitSignal(
                    exit_type=ExitType.INVALIDATION.value,
                    exit_pct=1.0,
                    reason=f"S5 invalidation: OI rebounded ({entry_oi_delta:.1f}% -> {oi_delta_pct:.1f}%)",
                    confidence=0.85,
                    metadata={**exit_context, 'oi_delta_pct': oi_delta_pct}
                )

        # A (Spring/Trap Reversal) - Spring invalidation
        elif archetype == 'trap_reversal':
            spring_low = position.metadata.get('entry_spring_low')
            if spring_low and bar['close'] < spring_low:
                return ExitSignal(
                    exit_type=ExitType.INVALIDATION.value,
                    exit_pct=1.0,
                    reason=f"Spring invalidation: close below spring low {spring_low:.2f}",
                    confidence=1.0,
                    metadata={**exit_context, 'spring_low': spring_low}
                )

        # B (Order Block Retest) - Close under OB support
        elif archetype == 'order_block_retest':
            ob_low = position.metadata.get('entry_ob_low')
            if ob_low and bar['close'] < ob_low:
                # Check for rejection confirmation (pin bar)
                body_size = abs(bar['close'] - bar['open'])
                lower_wick = min(bar['open'], bar['close']) - bar['low']
                wick_ratio = lower_wick / (body_size + lower_wick) if (body_size + lower_wick) > 0 else 0

                if wick_ratio > 0.5:  # Strong rejection wick
                    return ExitSignal(
                        exit_type=ExitType.INVALIDATION.value,
                        exit_pct=1.0,
                        reason=f"OB invalidation: rejection at {ob_low:.2f} (wick ratio {wick_ratio:.2f})",
                        confidence=0.95,
                        metadata={**exit_context, 'ob_low': ob_low, 'wick_ratio': wick_ratio}
                    )

        # H (Trap Within Trend) - Clean close below support
        elif archetype == 'trap_within_trend':
            support_level = position.metadata.get('entry_support_level')
            if support_level and bar['close'] < support_level:
                # Check for "clean" close (not just a wick)
                body_close_below = min(bar['open'], bar['close']) < support_level
                if body_close_below:
                    return ExitSignal(
                        exit_type=ExitType.INVALIDATION.value,
                        exit_pct=1.0,
                        reason=f"H invalidation: clean close below support {support_level:.2f}",
                        confidence=1.0,
                        metadata={**exit_context, 'support_level': support_level}
                    )

        # K (Wick Trap) - Wick invalidation
        elif archetype == 'wick_trap':
            wick_low = position.metadata.get('entry_wick_low')
            if wick_low and bar['close'] < wick_low:
                return ExitSignal(
                    exit_type=ExitType.INVALIDATION.value,
                    exit_pct=1.0,
                    reason=f"Wick invalidation: close below wick low {wick_low:.2f}",
                    confidence=1.0,
                    metadata={**exit_context, 'wick_low': wick_low}
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

        # DEBUG: Log what's in rules to understand config loading
        archetype_name = position.metadata.get('archetype', 'unknown')
        print(f"\n[EXIT_LOGIC DEBUG] _check_profit_targets called for {archetype_name}")
        print(f"[EXIT_LOGIC DEBUG] rules keys: {list(rules.keys())}")
        print(f"[EXIT_LOGIC DEBUG] full rules: {rules}")

        # CRITICAL FIX: Read profit_targets from config, not hardcoded defaults
        profit_targets = rules.get('profit_targets', [])
        if profit_targets:
            # Extract from config format: [{"r_multiple": 0.5, "exit_pct": 0.30}, ...]
            scale_levels = [pt['r_multiple'] for pt in profit_targets]
            scale_pcts = [pt['exit_pct'] for pt in profit_targets]
            print(f"[EXIT_LOGIC DEBUG] Using profit_targets from config: {profit_targets}")
            print(f"[EXIT_LOGIC DEBUG] scale_levels: {scale_levels}, scale_pcts: {scale_pcts}")
        else:
            # Fallback to old format or hardcoded defaults
            scale_levels = rules.get('scale_out_levels', [0.5, 1.0, 2.0])
            scale_pcts = rules.get('scale_out_pcts', [0.2, 0.2, 0.3])
            print(f"[EXIT_LOGIC DEBUG] Using HARDCODED defaults: scale_levels={scale_levels}, scale_pcts={scale_pcts}")

        # Track which scale-outs already executed
        executed_scales = position.metadata.get('executed_scale_outs', [])

        # Check each scale level in order
        for level, pct in zip(scale_levels, scale_pcts):
            if unrealized_r >= level and level not in executed_scales:
                # Mark this scale level as executed
                executed_scales.append(level)
                position.metadata['executed_scale_outs'] = executed_scales

                return ExitSignal(
                    exit_type=ExitType.PROFIT_TARGET.value,
                    exit_pct=pct,
                    reason=f"Scale-out at {level:.1f}R (exit {pct*100:.0f}%)",
                    confidence=1.0,
                    metadata={**exit_context, 'scale_level': level}
                )

        # Check archetype-specific profit targets
        archetype = exit_context['archetype']

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
        archetype = exit_context['archetype']

        # Get regime from context
        regime = context.regime_label

        # Apply regime adjustment to max hold period
        adjusted_params = self._get_regime_adjusted_params(rules, regime, unrealized_r)
        max_hold = adjusted_params['max_hold_hours']

        # DEBUG: Log trap_reversal time checks
        if archetype == 'trap_reversal' and hours_in_position > 20:
            base_max_hold = rules.get('max_hold_hours', 'MISSING')
            logger.info(f"[TIME EXIT CHECK] trap_reversal: hours={hours_in_position:.1f}, "
                       f"base_max={base_max_hold}, adjusted_max={max_hold:.0f}, "
                       f"regime={regime}, R={unrealized_r:.2f}")

        if hours_in_position >= max_hold:
            # DEBUG: Log every trap_reversal time exit
            if archetype == 'trap_reversal':
                logger.warning(f"[TIME EXIT TRIGGERED] trap_reversal position closed: "
                              f"held {hours_in_position:.1f}h >= max {max_hold:.0f}h, "
                              f"base_max_hold={rules.get('max_hold_hours')}, "
                              f"R={unrealized_r:.2f}, regime={regime}")

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
        archetype = exit_context['archetype']

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

        # Archetype-specific tightening overrides
        archetype = exit_context['archetype']

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
            # S1 (Liquidity Vacuum)
            'liquidity_vacuum': {
                'max_hold_hours': 120,  # 5 days
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.2, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 2.0,
                'invalidation_checks': True,
                'reason_gone_checks': False
            },

            # S4 (Funding Divergence)
            'funding_divergence': {
                'max_hold_hours': 240,  # 10 days (macro play)
                'scale_out_levels': [0.5, 1.0, 2.0, 3.0],
                'scale_out_pcts': [0.2, 0.2, 0.2, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.5,
                'invalidation_checks': True,
                'reason_gone_checks': True
            },

            # S5 (Long Squeeze)
            'long_squeeze': {
                'max_hold_hours': 72,  # 3 days (fast momentum)
                'scale_out_levels': [0.5, 1.0, 1.5],
                'scale_out_pcts': [0.3, 0.3, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'invalidation_checks': True,
                'reason_gone_checks': True
            },

            # A (Spring/Trap Reversal)
            'trap_reversal': {
                'max_hold_hours': 168,  # 7 days
                'scale_out_levels': [1.0, 2.0, 3.0],
                'scale_out_pcts': [0.3, 0.3, 0.3],
                'trailing_start_r': 1.0,
                'trailing_atr_mult': 2.0,
                'invalidation_checks': True,
                'reason_gone_checks': False
            },

            # B (Order Block Retest)
            'order_block_retest': {
                'max_hold_hours': 96,  # 4 days
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.25, 0.25, 0.4],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'invalidation_checks': True,
                'reason_gone_checks': False
            },

            # H (Trap Within Trend)
            'trap_within_trend': {
                'max_hold_hours': 120,  # 5 days
                'scale_out_levels': [0.5, 1.0, 2.0],
                'scale_out_pcts': [0.2, 0.3, 0.4],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 2.0,
                'invalidation_checks': True,
                'reason_gone_checks': True
            },

            # K (Wick Trap)
            'wick_trap': {
                'max_hold_hours': 48,  # 2 days (fast reversal)
                'scale_out_levels': [0.5, 1.0, 1.5],
                'scale_out_pcts': [0.3, 0.4, 0.3],
                'trailing_start_r': 0.5,
                'trailing_atr_mult': 1.5,
                'invalidation_checks': True,
                'reason_gone_checks': False
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
