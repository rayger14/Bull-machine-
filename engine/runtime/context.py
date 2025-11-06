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
        Safely get threshold for archetype parameter.

        Args:
            archetype: Archetype name (e.g., 'order_block_retest')
            param: Parameter name (e.g., 'fusion', 'liquidity')
            default: Fallback value if not found

        Returns:
            Threshold value
        """
        global _threshold_log_count
        value = self.thresholds.get(archetype, {}).get(param, default)

        # PR-A DEBUG: Log threshold resolution for first few calls
        if _threshold_log_count < _MAX_THRESHOLD_LOGS:
            threshold_keys = list(self.thresholds.keys()) if self.thresholds else []
            arch_data = self.thresholds.get(archetype, {}) if self.thresholds else {}
            logger.info(
                f"[PR-A THRESHOLD DEBUG] get_threshold('{archetype}', '{param}', default={default}) "
                f"→ {value} | thresholds={'EMPTY' if not self.thresholds else f'{len(threshold_keys)} archetypes'} "
                f"| arch_data={arch_data if arch_data else 'NOT_FOUND'}"
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
