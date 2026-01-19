"""
Direction Balance Tracking Integration for Backtesting Engine

Provides hooks to integrate DirectionBalanceTracker into backtesting pipeline.
Enables position direction monitoring and risk scaling during backtests.

Integration Points:
1. Before position entry: apply_direction_scaling()
2. After fill: on_position_entry()
3. On exit: on_position_exit()
4. Periodic monitoring: log_balance_summary()

Usage:
    from engine.backtesting.direction_hooks import DirectionBacktestHooks

    # Initialize once per backtest
    direction_hooks = DirectionBacktestHooks(
        enabled=True,
        imbalance_threshold=0.70,
        scale_mode='soft'
    )

    # Before generating signal
    scaled_entry = direction_hooks.apply_direction_scaling(
        original_entry,
        archetype_id='S1'
    )

    # After position opened
    if position_opened:
        direction_hooks.on_position_entry(
            symbol=symbol,
            direction='long',
            size=position_size,
            entry_price=entry_price,
            archetype_id='S1'
        )

    # On position close
    if position_closed:
        direction_hooks.on_position_exit(symbol)

    # Get monitoring metrics
    balance = direction_hooks.get_current_balance()

Author: Bull Machine Backend Architect (Claude Code)
Date: 2025-12-19
"""

from typing import Dict, Optional, Any
from pathlib import Path
import logging

from engine.risk.direction_integration import (
    DirectionBalanceIntegration,
    get_default_config
)

logger = logging.getLogger(__name__)


class DirectionBacktestHooks:
    """
    Backtest integration layer for direction balance tracking.

    This class provides a simple interface to integrate direction tracking
    into existing backtesting code with minimal modifications.

    Features:
    - Automatic signal scaling based on direction imbalance
    - Metadata enrichment for post-backtest analysis
    - Periodic monitoring and logging
    - Per-archetype direction breakdown
    - Shadow mode support (monitoring without scaling)
    """

    def __init__(
        self,
        enabled: bool = True,
        imbalance_threshold: float = 0.70,
        scale_mode: str = 'soft',
        shadow_mode: bool = False,
        log_frequency_bars: int = 100,
        log_dir: Optional[str] = None
    ):
        """
        Initialize direction tracking hooks.

        Args:
            enabled: Enable direction tracking (if False, all methods are no-ops)
            imbalance_threshold: Threshold for imbalance (0.70 = 70% one direction)
            scale_mode: 'soft' (scale confidence) or 'hard' (veto trades)
            shadow_mode: If True, log direction balance but don't apply scaling
            log_frequency_bars: Log balance every N bars
            log_dir: Directory for direction balance logs
        """
        self.enabled = enabled
        self.shadow_mode = shadow_mode
        self.log_frequency_bars = log_frequency_bars
        self.bar_count = 0

        if not enabled:
            logger.info("Direction tracking disabled")
            self.integration = None
            return

        # Create config
        config = get_default_config(
            enabled=True,
            imbalance_threshold=imbalance_threshold,
            scale_mode=scale_mode
        )

        if log_dir:
            config['log_dir'] = log_dir

        # Initialize integration layer
        self.integration = DirectionBalanceIntegration(config)

        mode_str = "SHADOW MODE (monitoring only)" if shadow_mode else f"{scale_mode.upper()} MODE"
        logger.info(f"Direction tracking enabled: {mode_str}, threshold={imbalance_threshold}")

    def apply_direction_scaling(
        self,
        entry: Any,
        archetype_id: str,
        symbol: Optional[str] = None
    ) -> Any:
        """
        Apply direction-based scaling to archetype entry signal.

        This should be called BEFORE position entry to scale confidence
        or veto the trade based on current direction balance.

        Args:
            entry: ArchetypeEntry object with signal, confidence, metadata
            archetype_id: Archetype identifier (e.g., 'S1', 'S4')
            symbol: Symbol being traded (optional, for logging)

        Returns:
            Modified ArchetypeEntry with:
            - Scaled confidence (or vetoed to FLAT)
            - Enriched metadata with direction_balance info
        """
        if not self.enabled or self.integration is None:
            return entry

        # In shadow mode, don't scale but still enrich metadata
        if self.shadow_mode:
            return self._shadow_mode_enrich(entry, archetype_id, symbol)

        # Apply actual scaling
        scaled_entry = self.integration.apply_direction_scaling(entry, archetype_id)

        # Log if scaling occurred
        if hasattr(scaled_entry, 'metadata') and scaled_entry.metadata.get('direction_scaled'):
            scale_factor = scaled_entry.metadata.get('direction_scale_factor', 1.0)
            original_conf = scaled_entry.metadata.get('original_confidence', 0)
            logger.debug(
                f"Direction scaling applied to {archetype_id} {symbol or ''}: "
                f"{original_conf:.2f} -> {scaled_entry.confidence:.2f} "
                f"(scale={scale_factor:.2f})"
            )

        return scaled_entry

    def on_position_entry(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        archetype_id: str,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """
        Track position entry in direction balance system.

        Call this AFTER a position is successfully opened.

        Args:
            symbol: Symbol (e.g., 'BTC-USD')
            direction: 'long' or 'short'
            size: Position size in $
            entry_price: Entry price
            archetype_id: Archetype that generated signal
            confidence: Signal confidence
            metadata: Additional metadata
        """
        if not self.enabled or self.integration is None:
            return

        self.integration.on_position_entry(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            archetype_id=archetype_id,
            confidence=confidence,
            metadata=metadata or {}
        )

        # Periodic logging
        self.bar_count += 1
        if self.bar_count % self.log_frequency_bars == 0:
            self.log_balance_summary()

    def on_position_exit(self, symbol: str):
        """
        Remove position from direction tracking.

        Call this when a position is closed.

        Args:
            symbol: Symbol of position being closed
        """
        if not self.enabled or self.integration is None:
            return

        self.integration.on_position_exit(symbol)

    def get_current_balance(self) -> Optional[Dict]:
        """
        Get current direction balance metrics.

        Returns:
            Dictionary with:
            - long_count: Number of long positions
            - short_count: Number of short positions
            - direction_ratio: Long exposure ratio (0.0-1.0)
            - is_imbalanced: True if >70% one direction
            - archetype_breakdown: Per-archetype direction counts
        """
        if not self.enabled or self.integration is None:
            return None

        return self.integration.get_current_balance_summary()

    def log_balance_summary(self):
        """Log current direction balance to console and file"""
        if not self.enabled or self.integration is None:
            return

        self.integration.log_balance_summary()

    def get_archetype_bias(self, archetype_id: str) -> str:
        """
        Get direction bias for specific archetype.

        Args:
            archetype_id: Archetype identifier

        Returns:
            'long', 'short', or 'balanced'
        """
        if not self.enabled or self.integration is None:
            return 'balanced'

        return self.integration.get_archetype_bias(archetype_id)

    def reset_for_backtest(self):
        """Reset direction tracking state for new backtest run"""
        if not self.enabled or self.integration is None:
            return

        self.integration.reset_for_backtest()
        self.bar_count = 0
        logger.info("Direction tracking reset for new backtest")

    def get_monitoring_metrics(self) -> Dict:
        """
        Get comprehensive monitoring metrics for dashboard.

        Returns:
            Dictionary with:
            - current_balance: Latest balance snapshot
            - imbalance_events: Count of imbalance occurrences
            - scaling_events: Count of times scaling was applied
            - vetoed_trades: Count of vetoed trades (hard mode)
        """
        if not self.enabled or self.integration is None:
            return {
                'enabled': False,
                'shadow_mode': False
            }

        balance = self.get_current_balance()

        return {
            'enabled': True,
            'shadow_mode': self.shadow_mode,
            'current_balance': balance,
            'bar_count': self.bar_count,
            'imbalance_threshold': self.integration.config.get('imbalance_threshold', 0.70),
            'scale_mode': self.integration.config.get('scale_mode', 'soft')
        }

    def _shadow_mode_enrich(self, entry: Any, archetype_id: str, symbol: Optional[str]) -> Any:
        """
        In shadow mode, enrich metadata with direction balance
        but don't apply any scaling.
        """
        balance = self.get_current_balance()

        if hasattr(entry, 'metadata') and balance:
            if not isinstance(entry.metadata, dict):
                entry.metadata = {}

            entry.metadata['direction_balance_shadow'] = balance
            entry.metadata['direction_tracking_mode'] = 'shadow'

            # Calculate what WOULD have been scaled
            from engine.archetypes.base_archetype import SignalType
            if hasattr(entry, 'signal') and entry.signal in (SignalType.LONG, SignalType.SHORT):
                direction = 'long' if entry.signal == SignalType.LONG else 'short'
                would_scale = self.integration.tracker.get_risk_scale_factor(direction)
                entry.metadata['direction_would_scale'] = would_scale

                if would_scale < 1.0:
                    logger.info(
                        f"SHADOW: {archetype_id} {symbol or ''} would be scaled "
                        f"{entry.confidence:.2f} -> {entry.confidence * would_scale:.2f}"
                    )

        return entry


def create_direction_hooks_from_config(config: Dict) -> DirectionBacktestHooks:
    """
    Create DirectionBacktestHooks from backtest config.

    Looks for 'direction_tracking' section in config:
    {
        "direction_tracking": {
            "enabled": true,
            "imbalance_threshold": 0.70,
            "scale_mode": "soft",
            "shadow_mode": false
        }
    }

    Args:
        config: Backtest configuration dictionary

    Returns:
        DirectionBacktestHooks instance
    """
    dt_config = config.get('direction_tracking', {})

    return DirectionBacktestHooks(
        enabled=dt_config.get('enabled', False),
        imbalance_threshold=dt_config.get('imbalance_threshold', 0.70),
        scale_mode=dt_config.get('scale_mode', 'soft'),
        shadow_mode=dt_config.get('shadow_mode', False),
        log_frequency_bars=dt_config.get('log_frequency_bars', 100)
    )


# Example integration into backtest loop
def example_backtest_integration():
    """
    Example showing how to integrate direction hooks into backtest loop.

    This is pseudo-code demonstrating the integration pattern.

    Example Code (for documentation):
    ```python
    from engine.archetypes.base_archetype import ArchetypeEntry, SignalType

    # Initialize direction tracking
    direction_hooks = DirectionBacktestHooks(
        enabled=True,
        imbalance_threshold=0.70,
        scale_mode='soft',
        shadow_mode=False
    )

    # Reset for new backtest
    direction_hooks.reset_for_backtest()

    # Backtest loop
    for bar in historical_data:
        original_entry = generate_archetype_signal(bar)
        scaled_entry = direction_hooks.apply_direction_scaling(
            entry=original_entry,
            archetype_id='S1',
            symbol='BTC-USD'
        )

        if scaled_entry.signal == SignalType.FLAT:
            continue

        position_opened = execute_trade(scaled_entry)

        if position_opened:
            direction_hooks.on_position_entry(
                symbol='BTC-USD',
                direction='long' if scaled_entry.signal == SignalType.LONG else 'short',
                size=1000.0,
                entry_price=50000.0,
                archetype_id='S1',
                confidence=scaled_entry.confidence
            )

        for position in open_positions:
            if should_exit(position):
                close_position(position)
                direction_hooks.on_position_exit(position.symbol)

    final_balance = direction_hooks.get_current_balance()
    print(f"Final balance: {final_balance}")
    ```
    """
    raise NotImplementedError("This is example/documentation code only")


if __name__ == '__main__':
    # Demo
    print("Direction Backtest Hooks Module")
    print("=" * 60)
    print("\nThis module provides hooks to integrate direction tracking")
    print("into backtesting engines.")
    print("\nSee example_backtest_integration() for usage pattern.")
