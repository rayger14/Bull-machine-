"""
Direction Balance Integration Layer

Integrates DirectionBalanceTracker with backtesting engine and archetypes.
Provides helper functions for signal metadata enrichment and risk scaling.

Author: Bull Machine Risk Team
Version: 1.0
Date: 2025-12-19
"""

from typing import Dict, Any, Optional
import logging

from engine.risk.direction_balance import DirectionBalanceTracker
from engine.archetypes.base_archetype import SignalType, ArchetypeEntry

logger = logging.getLogger(__name__)


class DirectionBalanceIntegration:
    """
    Integration layer between DirectionBalanceTracker and trading system.

    Provides:
    1. Signal metadata enrichment with direction balance
    2. Confidence scaling based on directional imbalance
    3. Position entry/exit hooks for tracking updates
    4. Backtest-compatible interface
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize direction balance integration.

        Args:
            config: Configuration dict passed to DirectionBalanceTracker
        """
        self.tracker = DirectionBalanceTracker(config)
        self.enabled = self.tracker.enabled

        if self.enabled:
            logger.info("Direction Balance Integration initialized")

    # ============================================================================
    # SIGNAL PROCESSING
    # ============================================================================

    def enrich_signal_metadata(
        self,
        signal_direction: str,
        base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich signal metadata with direction balance information.

        Args:
            signal_direction: "long" or "short"
            base_metadata: Existing signal metadata

        Returns:
            Enriched metadata with direction balance fields
        """
        if not self.enabled:
            return base_metadata

        # Get current position metadata
        position_meta = self.tracker.get_position_metadata()

        # Calculate risk scale factor for this direction
        scale_factor = self.tracker.get_risk_scale_factor(signal_direction)

        # Merge into metadata
        enriched = base_metadata.copy()
        enriched.update(position_meta)
        enriched['direction_risk_scale'] = round(scale_factor, 3)

        return enriched

    def apply_direction_scaling(
        self,
        entry: ArchetypeEntry,
        archetype_id: str
    ) -> ArchetypeEntry:
        """
        Apply direction-based confidence scaling to archetype entry.

        Args:
            entry: Original ArchetypeEntry from archetype
            archetype_id: Archetype identifier

        Returns:
            Modified ArchetypeEntry with scaled confidence
        """
        if not self.enabled:
            return entry

        # Get direction string
        direction = "long" if entry.signal == SignalType.LONG else "short"

        # Check if should veto
        should_veto, veto_reason = self.tracker.should_veto_signal(direction)
        if should_veto:
            logger.warning(f"[{archetype_id}] Signal vetoed: {veto_reason}")
            # Return flat signal
            return ArchetypeEntry(
                signal=SignalType.FLAT,
                confidence=0.0,
                entry_price=None,
                metadata={
                    **entry.metadata,
                    'direction_veto': True,
                    'direction_veto_reason': veto_reason
                }
            )

        # Get scale factor
        scale_factor = self.tracker.get_risk_scale_factor(direction)

        # Apply scaling if needed
        if scale_factor < 1.0:
            scaled_confidence = entry.confidence * scale_factor
            logger.info(
                f"[{archetype_id}] Direction scaling applied: "
                f"{entry.confidence:.3f} → {scaled_confidence:.3f} "
                f"(scale={scale_factor:.3f})"
            )

            # Create scaled entry
            return ArchetypeEntry(
                signal=entry.signal,
                confidence=scaled_confidence,
                entry_price=entry.entry_price,
                metadata={
                    **entry.metadata,
                    'direction_scaled': True,
                    'direction_scale_factor': scale_factor,
                    'original_confidence': entry.confidence,
                    **self.tracker.get_position_metadata()
                }
            )

        # No scaling needed
        return ArchetypeEntry(
            signal=entry.signal,
            confidence=entry.confidence,
            entry_price=entry.entry_price,
            metadata={
                **entry.metadata,
                'direction_scaled': False,
                'direction_scale_factor': 1.0,
                **self.tracker.get_position_metadata()
            }
        )

    # ============================================================================
    # POSITION LIFECYCLE HOOKS
    # ============================================================================

    def on_position_entry(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        archetype_id: str,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Hook called when new position is opened.

        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            size: Position size in $
            entry_price: Entry price
            archetype_id: Archetype that generated signal
            confidence: Signal confidence
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        self.tracker.add_position(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            archetype_id=archetype_id,
            confidence=confidence,
            metadata=metadata
        )

    def on_position_exit(self, symbol: str) -> None:
        """
        Hook called when position is closed.

        Args:
            symbol: Trading symbol
        """
        if not self.enabled:
            return

        self.tracker.remove_position(symbol)

    def on_position_size_change(self, symbol: str, new_size: float) -> None:
        """
        Hook called when position size changes (partial exit, scaling).

        Args:
            symbol: Trading symbol
            new_size: New position size in $
        """
        if not self.enabled:
            return

        self.tracker.update_position_size(symbol, new_size)

    # ============================================================================
    # QUERIES & MONITORING
    # ============================================================================

    def get_current_balance_summary(self) -> Dict[str, Any]:
        """
        Get current direction balance summary.

        Returns:
            Dict with balance metrics for logging/monitoring
        """
        if not self.enabled:
            return {"enabled": False}

        balance = self.tracker.get_current_balance()
        return balance.to_dict()

    def log_balance_summary(self):
        """Log detailed balance summary."""
        if not self.enabled:
            return

        self.tracker.log_summary()

    def get_archetype_bias(self, archetype_id: str) -> str:
        """
        Get directional bias for specific archetype.

        Args:
            archetype_id: Archetype identifier

        Returns:
            "long", "short", or "balanced"
        """
        if not self.enabled:
            return "balanced"

        directions = self.tracker.get_archetype_directions(archetype_id)
        total = directions['long'] + directions['short']

        if total == 0:
            return "balanced"
        elif directions['long'] > directions['short'] * 1.5:
            return "long"
        elif directions['short'] > directions['long'] * 1.5:
            return "short"
        else:
            return "balanced"

    # ============================================================================
    # BACKTEST INTEGRATION
    # ============================================================================

    def reset_for_backtest(self):
        """Reset tracker state for new backtest run."""
        if not self.enabled:
            return

        self.tracker.reset()
        logger.info("Direction balance tracker reset for backtest")

    def cleanup_old_data(self):
        """Clean up old history data to prevent memory growth."""
        if not self.enabled:
            return

        self.tracker.clear_old_history()

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    def update_config(self, new_config: Dict) -> None:
        """
        Update tracker configuration.

        Args:
            new_config: New configuration dict
        """
        # Reinitialize tracker with new config
        self.tracker = DirectionBalanceTracker(new_config)
        self.enabled = self.tracker.enabled

        logger.info(f"Direction balance configuration updated: enabled={self.enabled}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_direction_integration(config: Optional[Dict] = None) -> DirectionBalanceIntegration:
    """
    Factory function to create DirectionBalanceIntegration.

    Args:
        config: Configuration dict with direction balance settings

    Returns:
        Configured DirectionBalanceIntegration instance
    """
    return DirectionBalanceIntegration(config)


def get_default_config(
    enabled: bool = True,
    imbalance_threshold: float = 0.70,
    scale_mode: str = 'soft'
) -> Dict:
    """
    Get default direction balance configuration.

    Args:
        enabled: Enable direction tracking
        imbalance_threshold: Threshold for imbalance (0.70 = 70%)
        scale_mode: "soft" (scale confidence) or "hard" (veto)

    Returns:
        Configuration dict
    """
    return {
        "enable": enabled,
        "imbalance_threshold": imbalance_threshold,
        "scale_mode": scale_mode,
        "scale_factor_mild": 0.75,
        "scale_factor_severe": 0.50,
        "scale_factor_extreme": 0.25,
        "history_window_hours": 168,  # 1 week
        "log_frequency_minutes": 60,
        "log_dir": "logs/direction_balance"
    }
