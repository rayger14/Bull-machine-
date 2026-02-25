"""
Runtime Context - Single Source of Truth for Decision-Making

Immutable context object passed through the entire decision pipeline.
Contains regime state, adapted parameters, and computed thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd
import logging

# PR-A DEBUG: Track threshold logging
_threshold_log_count = 0
_MAX_THRESHOLD_LOGS = 10
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeContext:
    """
    Immutable runtime context for a single bar.

    Attributes:
        ts: Current timestamp
        row: Feature row with runtime fields (fusion, liquidity, etc.)
        regime_probs: Probability distribution over regimes
        regime_label: Argmax regime after hysteresis
        adapted_params: From AdaptiveFusion (weights/gates/exits/sizing/ml_threshold)
        thresholds: Per-archetype thresholds from ThresholdPolicy
        regime_config: Regime classifier configuration (mode, use_soft_controls, etc.)
        metadata: Optional additional context
    """
    ts: Any
    row: pd.Series
    regime_probs: Dict[str, float]
    regime_label: str
    adapted_params: Dict[str, Any]
    thresholds: Dict[str, Dict[str, float]]
    regime_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_threshold(self, archetype: str, param: str, default: float = 0.0) -> float:
        """
        Safely get threshold for archetype parameter with alias resolution.

        **PHASE 1 ALIAS FIX**: Resolves both short (fusion, pti) and long
        (fusion_threshold, pti_score_threshold) parameter names to support
        legacy configs during migration.

        **PHASE 2 ARCHETYPE ALIAS FIX**: Resolves archetype display names to canonical
        threshold names (e.g., "coil_break" → "confluence_breakout").

        **PERFORMANCE**: Logging only occurs on cache misses and errors to reduce I/O overhead.

        Args:
            archetype: Archetype name (e.g., 'order_block_retest', 'coil_break')
            param: Parameter name (short or long form)
            default: Fallback value if not found

        Returns:
            Threshold value

        Examples:
            get_threshold('spring', 'fusion_threshold') → looks for 'fusion_threshold' OR 'fusion'
            get_threshold('coil_break', 'fusion_threshold') → resolves to 'confluence_breakout' first
        """
        from engine.runtime.param_aliases import resolve_canonical, get_all_aliases, ARCHETYPE_SLUG_ALIASES

        global _threshold_log_count

        # PHASE 2 FIX: Resolve archetype slug aliases first
        canonical_archetype = ARCHETYPE_SLUG_ALIASES.get(archetype, archetype)

        # Resolve canonical parameter name (fusion → fusion_threshold)
        canonical = resolve_canonical(canonical_archetype, param)

        # Try canonical archetype name first, then all known aliases
        arch_data = self.thresholds.get(canonical_archetype, {})
        value = None
        found = False  # Track if we found the value in config

        if canonical in arch_data:
            value = arch_data[canonical]
            found = True
        else:
            # Try all aliases (including short forms from legacy configs)
            for alias in get_all_aliases(archetype, canonical):
                if alias in arch_data:
                    value = arch_data[alias]
                    found = True
                    break

        # If not found, use default
        if not found:
            value = default

        # PERFORMANCE: Only log errors/warnings, not successful lookups
        # PHASE 1 FIX: Always warn on critical failures (not just first N calls)
        if not self.thresholds:
            logger.warning(
                f"[PHASE1] CRITICAL: Thresholds dict is EMPTY! "
                f"get_threshold('{archetype}', '{param}') using default={default}"
            )
        elif canonical_archetype not in self.thresholds:
            # Show both original and resolved names for debugging
            alias_info = f" (resolved to '{canonical_archetype}')" if canonical_archetype != archetype else ""
            logger.warning(
                f"[PHASE2] Archetype '{archetype}'{alias_info} NOT FOUND in thresholds! "
                f"Available: {list(self.thresholds.keys())[:5]}... "
                f"Using default={default} for '{param}'"
            )
        elif not found:
            # Only warn if we did NOT find the parameter in config
            alias_info = f" (resolved to '{canonical_archetype}')" if canonical_archetype != archetype else ""
            logger.warning(
                f"[PHASE2] Parameter '{param}' (canonical: '{canonical}') NOT FOUND in archetype '{archetype}'{alias_info}! "
                f"Available params: {list(self.thresholds[canonical_archetype].keys())} "
                f"Using default={default}"
            )
        # REMOVED: Success logging - was 215k INFO calls causing 383s of I/O overhead

        # PR-A DEBUG: Log threshold resolution for first few calls only
        if _threshold_log_count < _MAX_THRESHOLD_LOGS:
            threshold_keys = list(self.thresholds.keys()) if self.thresholds else []
            alias_info = f" → '{canonical_archetype}'" if canonical_archetype != archetype else ""
            logger.info(
                f"[PR-A THRESHOLD DEBUG] get_threshold('{archetype}'{alias_info}, '{param}', default={default}) "
                f"→ {value} | thresholds={'EMPTY' if not self.thresholds else f'{len(threshold_keys)} archetypes'} "
                f"| arch_data={arch_data if arch_data else 'NOT_FOUND'} | canonical_param='{canonical}'"
            )
            _threshold_log_count += 1

        return value

    def get_adapted_param(self, key: str, default: Any = None) -> Any:
        """
        Safely get adapted parameter from adaptive fusion.

        Args:
            key: Parameter key (e.g., 'gates', 'fusion_weights')
            default: Fallback value if not found

        Returns:
            Parameter value
        """
        return self.adapted_params.get(key, default)
