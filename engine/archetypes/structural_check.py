"""
Structural Pattern Check Bridge — Wires logic.py into the ArchetypeInstance pipeline.

Each archetype has a REAL structural pattern check in logic.py (_check_A through _check_S8).
This module bridges the generic ArchetypeInstance.detect() to those specific checks.

Architecture:
    structural_check(logic.py) → cooling → YAML gates → fusion_score → threshold → Signal

If the structural check returns False, the archetype doesn't fire — making each
archetype a genuine independent strategy with pattern-specific recognition.

Feature Store Status:
    All previously-frozen features (fusion_smc, tf4h_fvg_present, tf4h_squiggle_confidence,
    tf1h_frvp_distance_to_poc, tf4h_choch_flag) have been patched with real computed values
    via bin/patch_frozen_features.py. FROZEN_ARCHETYPES is now empty — all structural checks
    are enforced in both backtest and live modes.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from engine.archetypes.logic import ArchetypeLogic

logger = logging.getLogger(__name__)

# YAML archetype name → logic.py letter code
NAME_TO_LETTER: Dict[str, str] = {
    'spring': 'A',
    'order_block_retest': 'B',
    'fvg_continuation': 'C',
    'failed_continuation': 'D',
    'liquidity_compression': 'E',
    'exhaustion_reversal': 'F',
    'liquidity_sweep': 'G',
    'trap_within_trend': 'H',
    'wick_trap': 'K',
    'retest_cluster': 'L',
    'confluence_breakout': 'M',
    'liquidity_vacuum': 'S1',
    'whipsaw': 'S3',
    'funding_divergence': 'S4',
    'long_squeeze': 'S5',
    'volume_fade_chop': 'S8',
}

# Archetypes whose structural checks depend on frozen features.
# Previously: {'A', 'B', 'C', 'G', 'M', 'K', 'D', 'L'} — all had dependencies
# on frozen tf1d_boms_strength, tf4h_boms_displacement, tf1h_frvp_distance_to_poc,
# fusion_smc, tf4h_fvg_present, or tf4h_squiggle_confidence.
#
# After patch_frozen_features.py: ALL features now have real computed values.
# FROZEN_ARCHETYPES is empty — all structural checks enforced everywhere.
#
FROZEN_ARCHETYPES: set = set()


class StructuralChecker:
    """
    Bridge between ArchetypeInstance and ArchetypeLogic structural pattern checks.

    Usage:
        checker = StructuralChecker(config={}, mode='backtest')
        passed, reason = checker.check_structure(
            'wick_trap', row, prev_row, lookback_df, bar_index
        )
        if not passed:
            # Archetype doesn't fire — pattern not structurally valid
    """

    def __init__(self, config: Optional[dict] = None, mode: str = 'backtest',
                 gate_params: Optional[Dict[str, float]] = None):
        """
        Initialize structural checker.

        Args:
            config: Config dict passed to ArchetypeLogic (thresholds, enable flags).
                    Defaults to permissive config with all archetypes enabled.
            mode: 'backtest' or 'live'. Controls frozen feature bypass.
            gate_params: Optional dict of structural gate thresholds to override
                        hardcoded defaults (e.g., {'wick_pct_K': 0.30, 'vol_z_L': 1.5}).
        """
        if config is None:
            config = {'use_archetypes': True}

        # Ensure use_archetypes is True so checks don't short-circuit
        config['use_archetypes'] = True

        # Enable all archetypes by default (structural check is opt-out via frozen bypass)
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M']:
            config.setdefault(f'enable_{letter}', True)
        for s in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']:
            config.setdefault(f'enable_{s}', True)

        self.logic = ArchetypeLogic(config)
        self.mode = mode
        self.gate_params = gate_params or {}
        self.frozen_bypass_enabled = (mode == 'backtest')

        # Stats tracking
        self.stats = {
            'total_checks': 0,
            'passed': 0,
            'rejected': 0,
            'frozen_bypassed': 0,
            'no_check': 0,
            'errors': 0,
            'rejections_by_archetype': {},
        }

        logger.info(
            f"[STRUCTURAL] Initialized: mode={mode}, "
            f"frozen_bypass={'ON' if self.frozen_bypass_enabled else 'OFF'}"
        )

    def check_structure(
        self,
        archetype_name: str,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        lookback_df: Optional[pd.DataFrame],
        bar_index: int,
    ) -> Tuple[bool, str]:
        """
        Run structural pattern check for the named archetype.

        Args:
            archetype_name: YAML archetype name (e.g., 'wick_trap')
            row: Current bar as pd.Series
            prev_row: Previous bar (or None if first bar)
            lookback_df: DataFrame of recent bars for lookback checks (or None)
            bar_index: Current bar index in the DataFrame

        Returns:
            (passed, reason) — passed=True if structure matches or check skipped
        """
        self.stats['total_checks'] += 1

        # Map name to letter code
        letter = NAME_TO_LETTER.get(archetype_name)
        if letter is None:
            self.stats['no_check'] += 1
            return True, "no_structural_check"

        # Skip frozen-feature archetypes in backtest mode
        if self.frozen_bypass_enabled and letter in FROZEN_ARCHETYPES:
            self.stats['frozen_bypassed'] += 1
            return True, "frozen_bypass"

        # Get the check method
        check_fn = getattr(self.logic, f'_check_{letter}', None)
        if check_fn is None:
            self.stats['no_check'] += 1
            return True, "no_check_method"

        # Build lookback_df if not provided (some checks need it)
        if lookback_df is None:
            # Create minimal DataFrame from just the current row
            lookback_df = pd.DataFrame([row])
            bar_index = 0

        try:
            # Call the structural check with fusion_score=1.0 (bypass internal fusion gate)
            # Pass gate_params for tunable threshold overrides
            passed = check_fn(row, prev_row, lookback_df, bar_index,
                            fusion_score=1.0, gate_params=self.gate_params)
        except Exception as e:
            # Don't let structural check errors block signals
            logger.warning(
                f"[STRUCTURAL] Error in _check_{letter} for {archetype_name}: {e}"
            )
            self.stats['errors'] += 1
            return True, f"error:{e}"

        if passed:
            self.stats['passed'] += 1
            return True, "structural_passed"
        else:
            self.stats['rejected'] += 1
            self.stats['rejections_by_archetype'][archetype_name] = (
                self.stats['rejections_by_archetype'].get(archetype_name, 0) + 1
            )
            return False, f"structural_{letter}_failed"

    def get_summary(self) -> dict:
        """Return stats summary for logging."""
        return {
            'total_checks': self.stats['total_checks'],
            'passed': self.stats['passed'],
            'rejected': self.stats['rejected'],
            'frozen_bypassed': self.stats['frozen_bypassed'],
            'errors': self.stats['errors'],
            'rejection_rate': (
                self.stats['rejected'] / max(self.stats['total_checks'], 1) * 100
            ),
            'top_rejections': dict(
                sorted(
                    self.stats['rejections_by_archetype'].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ),
        }
