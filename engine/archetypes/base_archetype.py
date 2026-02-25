"""
BaseArchetype - Abstract Base Class for All Archetypes

Defines the contract that all concrete archetypes must implement.
Ensures consistency, enforceability, and enables systematic validation.

Design Principles:
- Immutability: RuntimeContext is immutable (frozen dataclass)
- Separation of Concerns: Score, veto, and entry are separate stages
- Explicit over Implicit: All behavior must be declared via abstract methods
- Type Safety: Full type hints for all methods
- Observability: diagnostics() for debugging and auditing

Author: System Architect (Claude Code)
Date: 2025-12-12
Version: 2.0
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class SignalType(Enum):
    """Trade signal types"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    def __str__(self):
        return self.value


class MaturityLevel(Enum):
    """
    Archetype maturity states.

    Lifecycle: STUB → DEVELOPMENT → CALIBRATED → PRODUCTION
    """
    STUB = "stub"               # Placeholder, no implementation
    DEVELOPMENT = "development" # Implementation started, not validated
    CALIBRATED = "calibrated"   # Validated on historical data
    PRODUCTION = "production"   # Live-ready, battle-tested

    def __str__(self):
        return self.value


@dataclass
class ArchetypeScore:
    """
    Structured scoring output from score() method.

    Provides both aggregate score and component breakdown for interpretability.
    """
    total_score: float                      # [0.0, 1.0] - Overall confidence
    component_scores: Dict[str, float]      # Breakdown by feature domain
    reasons: List[str]                      # Human-readable scoring factors
    metadata: Dict[str, Any]                # Additional context

    def __post_init__(self):
        """Validate score range"""
        if not 0.0 <= self.total_score <= 1.0:
            raise ValueError(f"total_score must be in [0.0, 1.0], got {self.total_score}")


@dataclass
class ArchetypeVeto:
    """
    Structured veto output from veto() method.

    Vetos are HARD safety disqualifiers that block trades regardless of score.
    """
    is_vetoed: bool
    reason: str
    veto_type: str  # "hard_stop" | "safety" | "regime_mismatch" | "feature_missing"

    @staticmethod
    def no_veto():
        """Convenience constructor for no veto"""
        return ArchetypeVeto(is_vetoed=False, reason="", veto_type="")


@dataclass
class ArchetypeEntry:
    """
    Structured entry signal from entry() method.

    Contains signal direction, confidence, and trade execution parameters.
    """
    signal: SignalType
    confidence: float                       # [0.0, 1.0]
    entry_price: Optional[float]            # None = market order, else limit price
    metadata: Dict[str, Any]                # Stop loss, take profit, position size hints

    def __post_init__(self):
        """Validate confidence range"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaseArchetype(ABC):
    """
    Abstract base class for all archetypes.

    All concrete archetypes MUST implement this interface.
    The registry validates compliance at load time.

    Design Pattern: Template Method + Strategy
    - Template: Core workflow (score → veto → entry)
    - Strategy: Each archetype implements its own pattern logic

    Required Attributes (override in subclass):
        ARCHETYPE_ID: str          # e.g., "S1"
        ARCHETYPE_NAME: str        # e.g., "Liquidity Vacuum"
        MATURITY: MaturityLevel    # e.g., MaturityLevel.PRODUCTION
        DIRECTION: SignalType      # Primary signal direction
        REGIME_TAGS: List[str]     # e.g., ["risk_off", "crisis"]
        REQUIRES_ENGINES: List[str]  # e.g., ["wyckoff", "smc", "temporal"]

    Required Methods (must implement):
        required_features() → List[str]
        score(context) → ArchetypeScore
        veto(context) → ArchetypeVeto
        entry(context) → ArchetypeEntry

    Optional Methods (default implementations provided):
        exit(context) → Optional[Dict]
        diagnostics(context) → Dict
    """

    # ========================================================================
    # CLASS ATTRIBUTES (Override in subclass)
    # ========================================================================

    ARCHETYPE_ID: str = None
    ARCHETYPE_NAME: str = None
    MATURITY: MaturityLevel = MaturityLevel.STUB
    DIRECTION: SignalType = None
    REGIME_TAGS: List[str] = []
    REQUIRES_ENGINES: List[str] = []

    def __init__(self):
        """
        Initialize archetype.

        Validates required class attributes are set.
        """
        if self.ARCHETYPE_ID is None:
            raise ValueError(
                f"{self.__class__.__name__} must set ARCHETYPE_ID class attribute"
            )
        if self.ARCHETYPE_NAME is None:
            raise ValueError(
                f"{self.__class__.__name__} must set ARCHETYPE_NAME class attribute"
            )
        if self.DIRECTION is None:
            raise ValueError(
                f"{self.__class__.__name__} must set DIRECTION class attribute"
            )

        logger.info(
            f"Initialized {self.ARCHETYPE_ID} ({self.ARCHETYPE_NAME}) - "
            f"Maturity: {self.MATURITY.value}, Direction: {self.DIRECTION.value}"
        )

    # ========================================================================
    # REQUIRED METHODS (Must implement in subclass)
    # ========================================================================

    @abstractmethod
    def required_features(self) -> List[str]:
        """
        Return list of features this archetype requires.

        Features are categorized as:
        - critical: Archetype cannot function without these
        - recommended: Degraded performance without these
        - optional: Nice to have, graceful fallback available

        Returns:
            List of feature names from feature store

        Example:
            return [
                'liquidity_score',      # critical
                'volume_zscore',        # critical
                'wick_lower_ratio',     # critical
                'VIX_Z',                # recommended
                'funding_Z',            # optional
            ]

        Note:
            This is used by FeatureRealityGate for pre-backtest validation.
            List ALL features used in score(), veto(), and entry() methods.
        """
        pass

    @abstractmethod
    def score(self, context) -> ArchetypeScore:
        """
        Calculate archetype confidence score.

        This is the PATTERN RECOGNITION stage - how well does current
        market state match this archetype's canonical pattern?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            ArchetypeScore with total_score [0.0, 1.0] and breakdown

        Example:
            score = ArchetypeScore(
                total_score=0.72,
                component_scores={
                    'liquidity_drain': 0.85,
                    'volume_panic': 0.78,
                    'wick_rejection': 0.65,
                    'crisis_context': 0.60
                },
                reasons=[
                    'Liquidity drained 44% below 7d avg',
                    'Volume panic z-score: 2.8',
                    'Deep lower wick: 48% of candle'
                ],
                metadata={'pattern_quality': 'high'}
            )

        Design Notes:
            - score() should be PURE (no side effects)
            - Use context.row for feature access
            - Use context.get_threshold() for regime-aware thresholds
            - Return 0.0 if pattern not present at all
            - Return [0.4, 0.8] for typical pattern matches
            - Return [0.8, 1.0] for high-conviction setups
        """
        pass

    @abstractmethod
    def veto(self, context) -> ArchetypeVeto:
        """
        Hard safety disqualifiers.

        This is the SAFETY GATE stage - are there conditions that make
        this trade unsafe regardless of score?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            ArchetypeVeto indicating whether trade is blocked

        Example:
            # Liquidity Vacuum only trades in risk_off/crisis regimes
            if context.regime_label not in ['risk_off', 'crisis']:
                return ArchetypeVeto(
                    is_vetoed=True,
                    reason='Liquidity Vacuum requires risk_off or crisis regime',
                    veto_type='regime_mismatch'
                )

            # All checks passed
            return ArchetypeVeto.no_veto()

        Common Veto Reasons:
            - Regime mismatch (e.g., bear pattern in bull regime)
            - Missing critical features
            - Hard stop conditions (e.g., max drawdown exceeded)
            - Extreme market conditions (e.g., circuit breaker triggered)
            - Position limits (e.g., max open trades reached)

        Design Notes:
            - veto() should be FAST (called before score())
            - Use veto() for HARD constraints only
            - Use low score for SOFT preferences
            - Return ArchetypeVeto.no_veto() if all checks pass
        """
        pass

    @abstractmethod
    def entry(self, context) -> ArchetypeEntry:
        """
        Generate entry signal with confidence and metadata.

        This is the EXECUTION stage - given score passed and no veto,
        what is the exact entry specification?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            ArchetypeEntry with signal, confidence, and trade parameters

        Example:
            entry = ArchetypeEntry(
                signal=SignalType.LONG,
                confidence=0.72,
                entry_price=None,  # Market order
                metadata={
                    'stop_loss_pct': -0.025,
                    'take_profit_pct': 0.08,
                    'position_size_mult': 1.0,
                    'max_hold_bars': 72,
                    'entry_reason': 'Capitulation reversal setup'
                }
            )

        Metadata Fields (recommended):
            - stop_loss_pct: Stop loss as % from entry (negative)
            - take_profit_pct: Take profit as % from entry (positive)
            - position_size_mult: Position size multiplier (default 1.0)
            - max_hold_bars: Maximum hold time in bars
            - entry_reason: Human-readable entry justification
            - partial_exit_1_pct: First partial exit level (optional)
            - partial_exit_2_pct: Second partial exit level (optional)
            - trailing_stop_mult: Trailing stop ATR multiplier (optional)

        Design Notes:
            - entry() is only called if score() passed threshold and veto() returned False
            - confidence should match score().total_score (or be adjusted based on context)
            - entry_price=None means market order (immediate execution)
            - entry_price=float means limit order (may not fill)
            - metadata is passed to position manager for execution
        """
        pass

    # ========================================================================
    # OPTIONAL METHODS (Default implementations provided)
    # ========================================================================

    def exit(self, context) -> Optional[Dict]:
        """
        Optional exit logic override.

        Most archetypes use standard exit logic (trailing stop, max hold).
        Override this if archetype has specific exit conditions.

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            None (use default exits) OR Dict with exit signal

        Example:
            # Exit if liquidity recovers above 7d average
            if context.row.get('liquidity_drain_pct', 0.0) > 0.10:
                return {
                    'exit_signal': True,
                    'exit_reason': 'Liquidity recovered above baseline',
                    'exit_price': None  # Market exit
                }
            return None  # Continue holding

        Exit Dict Fields:
            - exit_signal: bool (True = exit now)
            - exit_reason: str (human-readable reason)
            - exit_price: Optional[float] (None = market, float = limit)
            - partial_pct: Optional[float] (partial exit %, default 100%)

        Design Notes:
            - exit() is called every bar while position is open
            - Return None to use default exit logic (recommended)
            - Override only if archetype has specific exit logic
            - Do NOT implement stop loss here (use metadata in entry())
        """
        return None

    def diagnostics(self, context) -> Dict:
        """
        Output what archetype looked at for auditing.

        This enables EXPLAINABILITY - what features did archetype use
        to make its decision?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            Dict with diagnostic information

        Example:
            return {
                'timestamp': context.ts,
                'archetype': self.ARCHETYPE_ID,
                'regime': context.regime_label,
                'features_used': {
                    'liquidity_score': context.row['liquidity_score'],
                    'liquidity_drain_pct': context.row['liquidity_drain_pct'],
                    'volume_zscore': context.row['volume_zscore'],
                    'wick_lower_ratio': context.row['wick_lower_ratio']
                },
                'thresholds_applied': {
                    'liquidity_drain_min': -0.30,
                    'volume_zscore_min': 2.0,
                    'wick_lower_min': 0.30
                },
                'score_components': {...},
                'veto_checks': {...}
            }

        Design Notes:
            - diagnostics() is called AFTER entry() if trade is taken
            - Used for post-trade analysis and debugging
            - Should include all features, thresholds, and intermediate calculations
            - Override this method to add archetype-specific diagnostics
        """
        return {
            'timestamp': context.ts,
            'archetype_id': self.ARCHETYPE_ID,
            'archetype_name': self.ARCHETYPE_NAME,
            'maturity': self.MATURITY.value,
            'regime': context.regime_label,
            'regime_probs': context.regime_probs
        }

    # ========================================================================
    # UTILITY METHODS (Available to all archetypes)
    # ========================================================================

    def get_threshold(self, context, param: str, default: float = 0.0) -> float:
        """
        Get archetype-specific threshold from RuntimeContext.

        Handles regime-aware threshold resolution via ThresholdPolicy.
        Uses archetype ID (lowercased) as lookup key.

        Args:
            context: RuntimeContext with thresholds
            param: Parameter name (e.g., 'fusion_threshold')
            default: Fallback value if not found

        Returns:
            Threshold value

        Example:
            fusion_threshold = self.get_threshold(context, 'fusion_threshold', 0.40)
        """
        return context.get_threshold(self.ARCHETYPE_ID.lower(), param, default)

    def validate_features(self, context) -> Tuple[bool, List[str]]:
        """
        Validate required features are present in context.

        Args:
            context: RuntimeContext with row data

        Returns:
            (all_present, missing_features)

        Example:
            all_present, missing = self.validate_features(context)
            if not all_present:
                logger.warning(f"Missing features: {missing}")
                return ArchetypeVeto(
                    is_vetoed=True,
                    reason=f"Missing critical features: {', '.join(missing)}",
                    veto_type='feature_missing'
                )
        """
        required = self.required_features()
        missing = [f for f in required if f not in context.row.index]
        return (len(missing) == 0, missing)

    def __repr__(self):
        """String representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"id={self.ARCHETYPE_ID}, "
            f"name={self.ARCHETYPE_NAME}, "
            f"maturity={self.MATURITY.value}, "
            f"direction={self.DIRECTION.value})"
        )


# ============================================================================
# Stub Archetype Base Class
# ============================================================================

class StubArchetype(BaseArchetype):
    """
    Base class for stub archetypes (not yet implemented).

    Stub archetypes raise NotImplementedError for all methods.
    This allows them to be registered in the registry while clearly
    indicating they are not ready for use.

    Usage:
        class SpringUTADArchetype(StubArchetype):
            ARCHETYPE_ID = "A"
            ARCHETYPE_NAME = "Spring / UTAD"
            MATURITY = MaturityLevel.STUB
            DIRECTION = SignalType.LONG
            REGIME_TAGS = ["risk_on", "neutral"]
            REQUIRES_ENGINES = ["wyckoff", "pti"]

        # No need to implement methods - StubArchetype raises NotImplementedError
    """

    def required_features(self) -> List[str]:
        raise NotImplementedError(
            f"{self.ARCHETYPE_ID} ({self.ARCHETYPE_NAME}) is a stub archetype - "
            f"not yet implemented"
        )

    def score(self, context) -> ArchetypeScore:
        raise NotImplementedError(
            f"{self.ARCHETYPE_ID} ({self.ARCHETYPE_NAME}) is a stub archetype - "
            f"not yet implemented"
        )

    def veto(self, context) -> ArchetypeVeto:
        raise NotImplementedError(
            f"{self.ARCHETYPE_ID} ({self.ARCHETYPE_NAME}) is a stub archetype - "
            f"not yet implemented"
        )

    def entry(self, context) -> ArchetypeEntry:
        raise NotImplementedError(
            f"{self.ARCHETYPE_ID} ({self.ARCHETYPE_NAME}) is a stub archetype - "
            f"not yet implemented"
        )
