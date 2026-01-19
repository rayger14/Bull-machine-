"""
Regime Hysteresis - Stub Implementation
========================================

This is a stub implementation to satisfy imports in RegimeService.

The actual RegimeHysteresis implementation was documented but never
committed to the repository. This stub provides no-op behavior
so that RegimeService can function without hysteresis.

Architecture:
- No-op implementation (pass-through)
- Returns input regime unchanged
- Maintains minimal state for interface compatibility

Author: Claude Code (System Restoration)
Date: 2026-01-19
"""

import logging
from typing import Dict, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeHysteresis:
    """
    Stub implementation of RegimeHysteresis.

    This provides no-op pass-through behavior to satisfy imports
    while the actual hysteresis implementation is being restored.

    Expected interface (from RegimeService):
    - apply_hysteresis(regime: str, probs: Dict, timestamp) -> Dict
    - reset()
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RegimeHysteresis stub.

        Args:
            config: Optional config dict (ignored in stub)
        """
        self.config = config or {}
        self.current_regime: Optional[str] = None
        self.last_transition_time: Optional[pd.Timestamp] = None

        logger.warning("RegimeHysteresis stub initialized - NO ACTUAL HYSTERESIS")
        logger.warning("Regime transitions will occur immediately without smoothing")

    def apply_hysteresis(
        self,
        regime: str,
        probs: Dict[str, float],
        timestamp: pd.Timestamp = None
    ) -> Dict:
        """
        Apply hysteresis to regime transition (stub: no-op).

        Args:
            regime: Proposed regime label
            probs: Regime probabilities
            timestamp: Current timestamp

        Returns:
            Dict with:
                - 'regime': str (regime label, unchanged)
                - 'probs': Dict[str, float] (probabilities, unchanged)
                - 'transition': bool (True if regime changed)
                - 'dwell_time': Optional[float] (time in current regime)
        """
        # Track transitions
        transition = (self.current_regime != regime) if self.current_regime is not None else True

        if transition:
            self.last_transition_time = timestamp

        self.current_regime = regime

        # Calculate dwell time if timestamp available
        dwell_time = None
        if self.last_transition_time and timestamp:
            try:
                dwell_time = (timestamp - self.last_transition_time).total_seconds() / 3600  # hours
            except Exception:
                dwell_time = None

        return {
            'regime': regime,
            'probs': probs,
            'transition': transition,
            'dwell_time': dwell_time,
            'hysteresis_applied': False  # Marker that this is stub
        }

    def reset(self):
        """Reset hysteresis state."""
        self.current_regime = None
        self.last_transition_time = None
        logger.info("RegimeHysteresis stub reset")
