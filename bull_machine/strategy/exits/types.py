"""
Exit Signal Types and Data Structures
Defines the core types used throughout the exit signal system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import pandas as pd


class ExitType(Enum):
    """Types of exit signals."""
    CHOCH_AGAINST = "choch_against"
    MOMENTUM_FADE = "momentum_fade"
    TIME_STOP = "time_stop"
    MTF_DESYNC = "mtf_desync"
    MANUAL = "manual"


class ExitAction(Enum):
    """Exit actions that can be taken."""
    FULL_EXIT = "full_exit"           # Close entire position
    PARTIAL_EXIT = "partial_exit"     # Close portion of position
    TIGHTEN_STOP = "tighten_stop"     # Move stop closer
    FLIP_POSITION = "flip_position"   # Close and reverse


@dataclass
class ExitSignal:
    """
    Standardized exit signal format.
    """
    timestamp: pd.Timestamp
    symbol: str
    exit_type: ExitType
    action: ExitAction
    confidence: float  # 0.0 to 1.0
    urgency: float     # 0.0 to 1.0 (how quickly to act)

    # Action-specific parameters
    exit_percentage: Optional[float] = None  # For partial exits (0.0-1.0)
    new_stop_price: Optional[float] = None   # For stop tightening
    flip_bias: Optional[str] = None          # For position flips ("long"/"short")

    # Supporting data
    reasons: List[str] = None
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.context is None:
            self.context = {}


@dataclass
class CHoCHContext:
    """Context data for CHoCH-Against detection."""
    timeframe: str
    direction: str  # "bearish" or "bullish"
    break_price: float
    confirmation_price: float
    structure_strength: float  # 0.0 to 1.0
    volume_confirmation: bool


@dataclass
class MomentumContext:
    """Context data for momentum fade detection."""
    current_rsi: float
    rsi_divergence: bool
    volume_decline: float  # Percentage decline
    velocity_slowdown: float  # Rate of price change decline
    timeframes_affected: List[str]


@dataclass
class TimeStopContext:
    """Context data for time-based stops."""
    bars_in_trade: int
    max_bars_allowed: int
    time_decay_factor: float  # How much confidence decays over time
    performance_vs_time: float  # PnL relative to time spent


class ExitEvaluationResult:
    """Result of exit signal evaluation."""

    def __init__(self):
        self.signals: List[ExitSignal] = []
        self.max_confidence = 0.0
        self.max_urgency = 0.0
        self.dominant_signal: Optional[ExitSignal] = None

    def add_signal(self, signal: ExitSignal):
        """Add an exit signal and update metrics."""
        self.signals.append(signal)

        if signal.confidence > self.max_confidence:
            self.max_confidence = signal.confidence
            self.dominant_signal = signal

        if signal.urgency > self.max_urgency:
            self.max_urgency = signal.urgency

    def has_signals(self) -> bool:
        """Check if any signals were generated."""
        return len(self.signals) > 0

    def get_strongest_signal(self) -> Optional[ExitSignal]:
        """Get the signal with highest confidence."""
        return self.dominant_signal

    def get_most_urgent_signal(self) -> Optional[ExitSignal]:
        """Get the signal with highest urgency."""
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.urgency)