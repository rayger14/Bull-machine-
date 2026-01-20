"""
Direction Balance Tracker for Position Monitoring

Tracks long/short position balance to manage directional risk exposure.
Provides position sizing adjustments based on portfolio direction imbalance.

DESIGN GOALS:
1. Real-time tracking of long/short position counts and exposure
2. Direction balance ratio calculation (long/(long+short))
3. Per-archetype direction breakdown for analysis
4. Integration with existing metadata system
5. Minimal performance overhead (<1ms per update)

RISK MANAGEMENT:
- Detects one-sided exposure (>70% long or >70% short)
- Scales down new signals when imbalanced
- Works alongside circuit breakers and drawdown persistence
- Soft scaling (reduces confidence) vs hard veto (configurable)

MONITORING:
- Direction balance logged periodically
- Historical balance tracking (rolling window)
- Export to metadata for post-trade analysis

Author: Bull Machine Risk Team
Version: 1.0
Date: 2025-12-19
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DirectionType(Enum):
    """Position direction types."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class PositionSnapshot:
    """Snapshot of a single position for direction tracking."""
    symbol: str
    direction: DirectionType
    size: float  # Position size in $
    entry_price: float
    entry_time: datetime
    archetype_id: str
    confidence: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class DirectionBalance:
    """Direction balance metrics at a point in time."""
    timestamp: datetime
    long_count: int
    short_count: int
    long_exposure: float  # Total $ in long positions
    short_exposure: float  # Total $ in short positions
    total_exposure: float
    direction_ratio: float  # long_exposure / total_exposure (0.0 = all short, 1.0 = all long)
    balance_pct: float  # Deviation from 50/50 (0% = balanced, 100% = all one side)
    is_imbalanced: bool  # True if >70% in one direction
    archetype_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "long_count": self.long_count,
            "short_count": self.short_count,
            "long_exposure": round(self.long_exposure, 2),
            "short_exposure": round(self.short_exposure, 2),
            "total_exposure": round(self.total_exposure, 2),
            "direction_ratio": round(self.direction_ratio, 3),
            "balance_pct": round(self.balance_pct, 1),
            "is_imbalanced": self.is_imbalanced,
            "archetype_breakdown": self.archetype_breakdown
        }


class DirectionBalanceTracker:
    """
    Tracks portfolio direction balance and provides risk scaling.

    Usage:
        tracker = DirectionBalanceTracker(config)

        # On position entry
        tracker.add_position(symbol, direction, size, archetype_id, metadata)

        # Before new signal
        scale_factor = tracker.get_risk_scale_factor(new_direction)
        adjusted_confidence = base_confidence * scale_factor

        # On position exit
        tracker.remove_position(symbol)

        # Get current balance
        balance = tracker.get_current_balance()
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize direction balance tracker.

        Args:
            config: Configuration dict with settings
                - enable: Enable direction tracking (default: True)
                - imbalance_threshold: Threshold for imbalance (default: 0.70)
                - scale_mode: "soft" (scale confidence) or "hard" (veto) (default: "soft")
                - scale_factor_mild: Scale for mild imbalance (60-70%) (default: 0.75)
                - scale_factor_severe: Scale for severe imbalance (>70%) (default: 0.50)
                - scale_factor_extreme: Scale for extreme imbalance (>85%) (default: 0.25)
                - history_window_hours: History to keep (default: 168 = 1 week)
                - log_frequency_minutes: How often to log balance (default: 60)
        """
        if config is None:
            config = {}
        self.config = config

        # Settings
        self.enabled = self.config.get('enable', True)
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.70)
        self.scale_mode = self.config.get('scale_mode', 'soft')
        self.scale_factor_mild = self.config.get('scale_factor_mild', 0.75)
        self.scale_factor_severe = self.config.get('scale_factor_severe', 0.50)
        self.scale_factor_extreme = self.config.get('scale_factor_extreme', 0.25)
        self.history_window_hours = self.config.get('history_window_hours', 168)
        self.log_frequency_minutes = self.config.get('log_frequency_minutes', 60)

        # State
        self.positions: Dict[str, PositionSnapshot] = {}  # symbol -> position
        self.balance_history: List[DirectionBalance] = []
        self.last_log_time: Optional[datetime] = None

        # Logging
        self.log_dir = Path(config.get("log_dir", "logs/direction_balance"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.enabled:
            logger.info(
                f"Direction Balance Tracker initialized: "
                f"imbalance_threshold={self.imbalance_threshold:.0%}, "
                f"scale_mode={self.scale_mode}"
            )
        else:
            logger.info("Direction Balance Tracker initialized but DISABLED")

    # ============================================================================
    # POSITION MANAGEMENT
    # ============================================================================

    def add_position(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        archetype_id: str,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> DirectionBalance:
        """
        Add new position to tracking.

        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            size: Position size in $
            entry_price: Entry price
            archetype_id: Archetype that generated signal
            confidence: Signal confidence [0.0, 1.0]
            metadata: Additional position metadata

        Returns:
            Updated DirectionBalance
        """
        if not self.enabled:
            return self._create_empty_balance()

        # Convert string to enum
        dir_type = DirectionType.LONG if direction.lower() == 'long' else DirectionType.SHORT

        # Create position snapshot
        position = PositionSnapshot(
            symbol=symbol,
            direction=dir_type,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            archetype_id=archetype_id,
            confidence=confidence,
            metadata=metadata or {}
        )

        # Add to tracking
        self.positions[symbol] = position

        # Calculate new balance
        balance = self._calculate_balance()
        self.balance_history.append(balance)

        # Periodic logging
        self._log_if_needed(balance)

        logger.debug(
            f"Added {direction} position: {symbol}, size=${size:.2f}, "
            f"archetype={archetype_id}, balance={balance.direction_ratio:.1%}"
        )

        return balance

    def remove_position(self, symbol: str) -> DirectionBalance:
        """
        Remove position from tracking (on exit).

        Args:
            symbol: Trading symbol

        Returns:
            Updated DirectionBalance
        """
        if not self.enabled:
            return self._create_empty_balance()

        if symbol in self.positions:
            position = self.positions.pop(symbol)
            logger.debug(
                f"Removed {position.direction.value} position: {symbol}, "
                f"held for {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f}h"
            )

        # Calculate new balance
        balance = self._calculate_balance()
        self.balance_history.append(balance)

        # Periodic logging
        self._log_if_needed(balance)

        return balance

    def update_position_size(self, symbol: str, new_size: float) -> DirectionBalance:
        """
        Update position size (for partial exits, scaling).

        Args:
            symbol: Trading symbol
            new_size: New position size in $

        Returns:
            Updated DirectionBalance
        """
        if not self.enabled:
            return self._create_empty_balance()

        if symbol in self.positions:
            self.positions[symbol].size = new_size

            # Calculate new balance
            balance = self._calculate_balance()
            self.balance_history.append(balance)

            return balance

        return self._calculate_balance()

    # ============================================================================
    # BALANCE CALCULATION
    # ============================================================================

    def _calculate_balance(self) -> DirectionBalance:
        """
        Calculate current direction balance from active positions.

        Returns:
            DirectionBalance with all metrics
        """
        if not self.positions:
            return self._create_empty_balance()

        # Separate long/short positions
        long_positions = [p for p in self.positions.values() if p.direction == DirectionType.LONG]
        short_positions = [p for p in self.positions.values() if p.direction == DirectionType.SHORT]

        # Count positions
        long_count = len(long_positions)
        short_count = len(short_positions)

        # Calculate exposure
        long_exposure = sum(p.size for p in long_positions)
        short_exposure = sum(p.size for p in short_positions)
        total_exposure = long_exposure + short_exposure

        # Calculate direction ratio
        if total_exposure > 0:
            direction_ratio = long_exposure / total_exposure
        else:
            direction_ratio = 0.5  # Neutral if no exposure

        # Calculate balance percentage (deviation from 50/50)
        balance_pct = abs(direction_ratio - 0.5) * 2.0 * 100  # 0-100%

        # Check if imbalanced (>= threshold to include exactly 70% as imbalanced)
        # Convert to Python bool to avoid numpy bool JSON serialization issues
        is_imbalanced = bool(direction_ratio >= self.imbalance_threshold or direction_ratio <= (1.0 - self.imbalance_threshold))

        # Per-archetype breakdown
        archetype_breakdown = {}
        for position in self.positions.values():
            arch_id = position.archetype_id
            if arch_id not in archetype_breakdown:
                archetype_breakdown[arch_id] = {"long": 0, "short": 0}

            if position.direction == DirectionType.LONG:
                archetype_breakdown[arch_id]["long"] += 1
            else:
                archetype_breakdown[arch_id]["short"] += 1

        return DirectionBalance(
            timestamp=datetime.now(),
            long_count=long_count,
            short_count=short_count,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            total_exposure=total_exposure,
            direction_ratio=direction_ratio,
            balance_pct=balance_pct,
            is_imbalanced=is_imbalanced,
            archetype_breakdown=archetype_breakdown
        )

    def _create_empty_balance(self) -> DirectionBalance:
        """Create empty balance (no positions)."""
        return DirectionBalance(
            timestamp=datetime.now(),
            long_count=0,
            short_count=0,
            long_exposure=0.0,
            short_exposure=0.0,
            total_exposure=0.0,
            direction_ratio=0.5,
            balance_pct=0.0,
            is_imbalanced=False,
            archetype_breakdown={}
        )

    # ============================================================================
    # RISK SCALING
    # ============================================================================

    def get_risk_scale_factor(self, new_direction: str) -> float:
        """
        Get risk scale factor for new signal based on current balance.

        Args:
            new_direction: Direction of new signal ("long" or "short")

        Returns:
            Scale factor [0.0, 1.0]
            - 1.0 = no scaling (balanced or counter-directional)
            - 0.75 = mild imbalance (60-70%)
            - 0.50 = severe imbalance (70-85%)
            - 0.25 = extreme imbalance (>85%)
            - 0.0 = hard veto (if scale_mode='hard')
        """
        if not self.enabled:
            return 1.0

        balance = self._calculate_balance()

        # If no positions, allow full size
        if balance.total_exposure == 0:
            return 1.0

        # Determine if new signal adds to imbalance or reduces it
        new_dir_is_long = new_direction.lower() == 'long'

        # Calculate what direction ratio would be if we take this trade
        # (Approximate - assumes similar position size as current avg)
        avg_position_size = balance.total_exposure / max(1, len(self.positions))
        projected_long = balance.long_exposure + (avg_position_size if new_dir_is_long else 0)
        projected_short = balance.short_exposure + (0 if new_dir_is_long else avg_position_size)
        projected_total = projected_long + projected_short
        projected_ratio = projected_long / projected_total if projected_total > 0 else 0.5

        # Check imbalance severity
        imbalance_severity = abs(projected_ratio - 0.5) * 2.0  # 0.0 = balanced, 1.0 = all one side

        # Scale based on severity (aligned with mission brief expectations)
        # Imbalance severity thresholds:
        # - 0.70+ (85%+ one side) = extreme → 0.25x
        # - 0.40+ (70%+ one side) = severe → 0.50x
        # - 0.20+ (60%+ one side) = mild → 0.75x
        # - <0.20 (<60%) = balanced → 1.0x

        if imbalance_severity >= 0.70:
            # Extreme imbalance (>85% one side)
            scale = 0.0 if self.scale_mode == 'hard' else self.scale_factor_extreme
            logger.warning(
                f"EXTREME direction imbalance: {projected_ratio:.0%} long after new {new_direction} signal "
                f"(scaling to {scale:.0%})"
            )
        elif imbalance_severity >= 0.40:
            # Severe imbalance (70-85% one side)
            scale = 0.0 if self.scale_mode == 'hard' else self.scale_factor_severe
            logger.warning(
                f"SEVERE direction imbalance: {projected_ratio:.0%} long after new {new_direction} signal "
                f"(scaling to {scale:.0%})"
            )
        elif imbalance_severity >= 0.20:
            # Mild imbalance (60-70% one side)
            scale = self.scale_factor_mild
            logger.debug(
                f"Mild direction imbalance: {projected_ratio:.0%} long after new {new_direction} signal "
                f"(scaling to {scale:.0%})"
            )
        else:
            # Balanced or counter-directional (reduces imbalance)
            scale = 1.0

        return scale

    def should_veto_signal(self, new_direction: str) -> Tuple[bool, str]:
        """
        Check if signal should be vetoed due to extreme imbalance.

        Only active if scale_mode='hard'.

        Args:
            new_direction: Direction of new signal ("long" or "short")

        Returns:
            (should_veto, reason)
        """
        if not self.enabled or self.scale_mode != 'hard':
            return False, ""

        scale = self.get_risk_scale_factor(new_direction)

        if scale == 0.0:
            balance = self._calculate_balance()
            return True, (
                f"Extreme direction imbalance: {balance.direction_ratio:.0%} long "
                f"(threshold: {self.imbalance_threshold:.0%})"
            )

        return False, ""

    # ============================================================================
    # GETTERS & QUERIES
    # ============================================================================

    def get_current_balance(self) -> DirectionBalance:
        """Get current direction balance."""
        return self._calculate_balance()

    def get_position_metadata(self) -> Dict:
        """
        Get position metadata for inclusion in signal metadata.

        Returns:
            Dict with direction balance metrics for logging
        """
        if not self.enabled:
            return {}

        balance = self._calculate_balance()

        return {
            "direction_balance": {
                "long_count": balance.long_count,
                "short_count": balance.short_count,
                "long_exposure": round(balance.long_exposure, 2),
                "short_exposure": round(balance.short_exposure, 2),
                "direction_ratio": round(balance.direction_ratio, 3),
                "balance_pct": round(balance.balance_pct, 1),
                "is_imbalanced": balance.is_imbalanced
            }
        }

    def get_archetype_directions(self, archetype_id: str) -> Dict[str, int]:
        """
        Get long/short breakdown for specific archetype.

        Args:
            archetype_id: Archetype identifier

        Returns:
            {"long": count, "short": count}
        """
        breakdown = {"long": 0, "short": 0}

        for position in self.positions.values():
            if position.archetype_id == archetype_id:
                if position.direction == DirectionType.LONG:
                    breakdown["long"] += 1
                else:
                    breakdown["short"] += 1

        return breakdown

    def get_balance_history(
        self,
        hours: int = 24
    ) -> List[DirectionBalance]:
        """
        Get balance history for last N hours.

        Args:
            hours: Hours of history to retrieve

        Returns:
            List of DirectionBalance snapshots
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [b for b in self.balance_history if b.timestamp >= cutoff]

    # ============================================================================
    # LOGGING & MONITORING
    # ============================================================================

    def _log_if_needed(self, balance: DirectionBalance):
        """Log balance periodically."""
        now = datetime.now()

        # Check if we should log
        should_log = (
            self.last_log_time is None or
            (now - self.last_log_time).total_seconds() / 60 >= self.log_frequency_minutes
        )

        if should_log:
            logger.info(
                f"[Direction Balance] "
                f"Long: {balance.long_count} ({balance.long_exposure:.0f}$), "
                f"Short: {balance.short_count} ({balance.short_exposure:.0f}$), "
                f"Ratio: {balance.direction_ratio:.0%}, "
                f"Imbalanced: {balance.is_imbalanced}"
            )

            # Save to log file
            self._save_balance(balance)

            self.last_log_time = now

    def _save_balance(self, balance: DirectionBalance):
        """Save balance snapshot to log file."""
        log_file = self.log_dir / f"direction_balance_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(balance.to_dict()) + "\n")

    def log_summary(self):
        """Log summary of direction tracking."""
        balance = self._calculate_balance()

        logger.info("=" * 60)
        logger.info("DIRECTION BALANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Positions: {len(self.positions)}")
        logger.info(f"Long Positions: {balance.long_count} (${balance.long_exposure:.2f})")
        logger.info(f"Short Positions: {balance.short_count} (${balance.short_exposure:.2f})")
        logger.info(f"Direction Ratio: {balance.direction_ratio:.1%} long")
        logger.info(f"Balance Deviation: {balance.balance_pct:.1%}")
        logger.info(f"Imbalanced: {balance.is_imbalanced}")

        if balance.archetype_breakdown:
            logger.info("\nPer-Archetype Breakdown:")
            for archetype, counts in balance.archetype_breakdown.items():
                logger.info(f"  {archetype}: {counts['long']}L / {counts['short']}S")

        logger.info("=" * 60)

    # ============================================================================
    # CLEANUP
    # ============================================================================

    def clear_old_history(self):
        """Remove old balance history to prevent memory growth."""
        if not self.balance_history:
            return

        cutoff = datetime.now() - timedelta(hours=self.history_window_hours)
        self.balance_history = [
            b for b in self.balance_history
            if b.timestamp >= cutoff
        ]

        logger.debug(f"Cleaned balance history: {len(self.balance_history)} snapshots remaining")

    def reset(self):
        """Reset tracker (clear all positions and history)."""
        self.positions.clear()
        self.balance_history.clear()
        self.last_log_time = None
        logger.info("Direction Balance Tracker reset")
