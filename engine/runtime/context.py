"""
Runtime Context - Single Source of Truth for Decision-Making

Immutable context object passed through the entire decision pipeline.
Contains regime state, adapted parameters, and computed thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd


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
        return self.thresholds.get(archetype, {}).get(param, default)

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
