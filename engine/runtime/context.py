"""
Runtime Context - Single Source of Truth for Decision-Making

Immutable context object passed through the entire decision pipeline.
Contains regime state, adapted parameters, and computed thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
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
        metadata: Optional additional context
    """
    ts: Any
    row: pd.Series
    regime_probs: Dict[str, float]
    regime_label: str
    adapted_params: Dict[str, Any]
    thresholds: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_threshold(self, archetype: str, param: str, default: float = 0.0) -> float:
        """
        Safely get threshold for archetype parameter with alias resolution.

        **PHASE 1 ALIAS FIX**: Resolves both short (fusion, pti) and long
        (fusion_threshold, pti_score_threshold) parameter names to support
        legacy configs during migration.

        Args:
            archetype: Archetype name (e.g., 'order_block_retest')
            param: Parameter name (short or long form)
            default: Fallback value if not found

        Returns:
            Threshold value

        Examples:
            get_threshold('spring', 'fusion_threshold') → looks for 'fusion_threshold' OR 'fusion'
            get_threshold('spring', 'fusion') → looks for 'fusion_threshold' OR 'fusion'
        """
        from engine.runtime.param_aliases import resolve_canonical, get_all_aliases

        global _threshold_log_count

        # Resolve canonical parameter name (fusion → fusion_threshold)
        canonical = resolve_canonical(archetype, param)

        # Try canonical name first, then all known aliases
        arch_data = self.thresholds.get(archetype, {})
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

        # PHASE 1 FIX: Always warn on critical failures (not just first N calls)
        if not self.thresholds:
            logger.warning(
                f"[PHASE1] CRITICAL: Thresholds dict is EMPTY! "
                f"get_threshold('{archetype}', '{param}') using default={default}"
            )
        elif archetype not in self.thresholds:
            logger.warning(
                f"[PHASE1] Archetype '{archetype}' NOT FOUND in thresholds! "
                f"Available: {list(self.thresholds.keys())[:5]}... "
                f"Using default={default} for '{param}'"
            )
        elif not found:
            # Only warn if we did NOT find the parameter in config
            logger.warning(
                f"[PHASE1] Parameter '{param}' (canonical: '{canonical}') NOT FOUND in archetype '{archetype}'! "
                f"Available params: {list(self.thresholds[archetype].keys())} "
                f"Using default={default}"
            )
        else:
            # SUCCESS: Found param via alias resolution
            logger.info(
                f"[ParamEcho] {archetype}.{canonical} → {value} "
                f"(requested='{param}', matched in config)"
            )

        # PR-A DEBUG: Log threshold resolution for first few calls
        if _threshold_log_count < _MAX_THRESHOLD_LOGS:
            threshold_keys = list(self.thresholds.keys()) if self.thresholds else []
            logger.info(
                f"[PR-A THRESHOLD DEBUG] get_threshold('{archetype}', '{param}', default={default}) "
                f"→ {value} | thresholds={'EMPTY' if not self.thresholds else f'{len(threshold_keys)} archetypes'} "
                f"| arch_data={arch_data if arch_data else 'NOT_FOUND'} | canonical='{canonical}'"
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
