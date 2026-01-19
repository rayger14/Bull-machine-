"""
Regime Hysteresis - Production Implementation
==============================================

Implements dual-threshold hysteresis with minimum dwell time enforcement
to prevent excessive regime transitions (flickering).

Architecture:
- Dual threshold mechanism (high confidence to enter, lower to stay)
- Per-regime minimum dwell time enforcement
- Optional EWMA probability smoothing
- Transition tracking and comprehensive logging
- Stateful (maintains current regime and transition history)

Target Performance:
- 10-40 regime transitions per year (vs 590+ without hysteresis)
- Prevents regime flickering during uncertain periods
- Maintains regime stability while detecting true changes

References:
- docs/archive/2026-01_integration/HYSTERESIS_FIX_FINAL_REPORT.md
- docs/archive/2026-01_regime_detection/LOGISTIC_REGRESSION_REGIME_DETECTION_RESEARCH_REPORT.md

Author: Claude Code (Backend Architect)
Date: 2026-01-19
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RegimeHysteresis:
    """
    Production-ready hysteresis implementation for regime detection.

    Prevents excessive regime transitions using:
    1. Dual thresholds: High confidence needed to ENTER new regime,
       lower threshold allows STAYING in current regime
    2. Minimum dwell time: Per-regime minimum hours before allowing transition
    3. Optional EWMA smoothing: Exponentially weighted moving average of probabilities

    Typical Configuration:
        {
            'enter_threshold': 0.65,     # Need 65% confidence to switch
            'exit_threshold': 0.50,      # Below 50% = uncertain, stay put
            'min_duration_hours': {
                'crisis': 6,             # Crisis regime: 6 hours minimum
                'risk_off': 24,          # Risk-off: 24 hours minimum
                'neutral': 12,           # Neutral: 12 hours minimum
                'risk_on': 48            # Risk-on: 48 hours minimum
            },
            'ewma_alpha': 0.3,           # Decay factor for smoothing
            'enable_ewma': False         # Optional smoothing
        }

    State Management:
    - current_regime: Currently active regime
    - regime_start_time: When current regime began
    - ewma_probs: Smoothed probabilities (if EWMA enabled)
    - transition_count: Total transitions tracked
    - transition_history: List of (timestamp, old_regime, new_regime) tuples
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RegimeHysteresis.

        Args:
            config: Configuration dictionary with:
                - enter_threshold (float): Confidence needed to enter new regime (default: 0.65)
                - exit_threshold (float): Below this = stay in current regime (default: 0.50)
                - min_duration_hours (dict): Per-regime minimum dwell times (default: all 24h)
                - ewma_alpha (float): EWMA decay factor (default: 0.3)
                - enable_ewma (bool): Enable probability smoothing (default: False)
        """
        # Default configuration (conservative - prevents excessive transitions)
        self.config = config or {}

        # Dual threshold parameters
        self.enter_threshold = self.config.get('enter_threshold', 0.65)
        self.exit_threshold = self.config.get('exit_threshold', 0.50)

        # Minimum dwell time per regime (hours)
        default_durations = {
            'crisis': 6,
            'risk_off': 24,
            'neutral': 12,
            'risk_on': 48
        }
        self.min_duration_hours = self.config.get('min_duration_hours', default_durations)

        # EWMA smoothing parameters
        self.enable_ewma = self.config.get('enable_ewma', False)
        self.ewma_alpha = self.config.get('ewma_alpha', 0.3)

        # Validate configuration
        self._validate_config()

        # State variables
        self.current_regime: Optional[str] = None
        self.regime_start_time: Optional[pd.Timestamp] = None
        self.ewma_probs: Optional[Dict[str, float]] = None

        # Transition tracking
        self.transition_count = 0
        self.transition_history = []  # List of (timestamp, old_regime, new_regime)

        # Logging
        logger.info("=" * 80)
        logger.info("RegimeHysteresis initialized (PRODUCTION)")
        logger.info(f"  Enter threshold: {self.enter_threshold:.2f}")
        logger.info(f"  Exit threshold: {self.exit_threshold:.2f}")
        logger.info(f"  Min dwell times: {self.min_duration_hours}")
        logger.info(f"  EWMA smoothing: {'ENABLED' if self.enable_ewma else 'DISABLED'}")
        if self.enable_ewma:
            logger.info(f"  EWMA alpha: {self.ewma_alpha}")
        logger.info("=" * 80)

    def _validate_config(self):
        """Validate configuration parameters."""
        # Threshold validation
        if not 0.0 <= self.enter_threshold <= 1.0:
            raise ValueError(f"enter_threshold must be in [0, 1], got {self.enter_threshold}")
        if not 0.0 <= self.exit_threshold <= 1.0:
            raise ValueError(f"exit_threshold must be in [0, 1], got {self.exit_threshold}")
        if self.exit_threshold > self.enter_threshold:
            logger.warning(
                f"exit_threshold ({self.exit_threshold}) > enter_threshold ({self.enter_threshold}). "
                f"This may cause instability. Recommended: exit < enter"
            )

        # Dwell time validation
        for regime, hours in self.min_duration_hours.items():
            if hours < 0:
                raise ValueError(f"min_duration_hours[{regime}] must be >= 0, got {hours}")

        # EWMA validation
        if self.enable_ewma:
            if not 0.0 < self.ewma_alpha <= 1.0:
                raise ValueError(f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}")

    def apply_hysteresis(
        self,
        regime: str,
        probs: Dict[str, float],
        timestamp: pd.Timestamp = None
    ) -> Dict:
        """
        Apply hysteresis to regime transition.

        Logic:
        1. If first call: Initialize with regime if prob > enter_threshold
        2. If in locked period (< min_dwell_time): Stay in current regime
        3. If new regime proposed:
           a. Check if new regime prob > enter_threshold (strong signal)
           b. Check if current regime prob < exit_threshold (weak hold)
           c. If both true: Allow transition
        4. Optional: Apply EWMA smoothing to probabilities

        Args:
            regime: Proposed regime label from classifier
            probs: Regime probabilities {regime_name: probability}
            timestamp: Current timestamp (for dwell time calculation)

        Returns:
            Dict with:
                - 'regime' (str): Final regime (may differ from input)
                - 'probs' (Dict[str, float]): Potentially smoothed probabilities
                - 'transition' (bool): True if regime changed
                - 'dwell_time' (float): Hours in current regime
                - 'hysteresis_applied' (bool): True if override occurred
                - 'reason' (str): Human-readable explanation of decision
        """
        # Apply EWMA smoothing if enabled
        if self.enable_ewma:
            probs = self._apply_ewma_smoothing(probs)

        # Handle first call (initialization)
        if self.current_regime is None:
            return self._initialize_regime(regime, probs, timestamp)

        # Calculate dwell time
        dwell_time_hours = self._calculate_dwell_time(timestamp)

        # Check minimum dwell time constraint
        if not self._check_dwell_time(timestamp):
            # LOCKED: Must stay in current regime
            return {
                'regime': self.current_regime,
                'probs': probs,
                'transition': False,
                'dwell_time': dwell_time_hours,
                'hysteresis_applied': True,
                'reason': f"Locked in {self.current_regime} (dwell time {dwell_time_hours:.1f}h < {self._get_min_dwell():.1f}h min)"
            }

        # Check if regime change is proposed
        if regime == self.current_regime:
            # No change proposed, continue in current regime
            return {
                'regime': self.current_regime,
                'probs': probs,
                'transition': False,
                'dwell_time': dwell_time_hours,
                'hysteresis_applied': False,
                'reason': f"Stable in {self.current_regime}"
            }

        # Regime change proposed - check thresholds
        new_regime_prob = probs.get(regime, 0.0)
        current_regime_prob = probs.get(self.current_regime, 0.0)

        # Decision logic:
        # 1. New regime must have strong confidence (> enter_threshold)
        # 2. Current regime must be weak (< exit_threshold) OR new regime is very strong
        allow_transition = False
        reason = ""

        if new_regime_prob >= self.enter_threshold:
            # Strong signal for new regime
            if current_regime_prob < self.exit_threshold:
                # Current regime is weak
                allow_transition = True
                reason = f"Transition: {regime} prob {new_regime_prob:.3f} > enter_threshold {self.enter_threshold:.3f}, {self.current_regime} prob {current_regime_prob:.3f} < exit_threshold {self.exit_threshold:.3f}"
            else:
                # Current regime still strong - stay put despite new signal
                allow_transition = False
                reason = f"Stay in {self.current_regime}: Current prob {current_regime_prob:.3f} >= exit_threshold {self.exit_threshold:.3f} (new {regime} prob {new_regime_prob:.3f})"
        else:
            # Weak signal for new regime - stay in current
            allow_transition = False
            reason = f"Stay in {self.current_regime}: New regime {regime} prob {new_regime_prob:.3f} < enter_threshold {self.enter_threshold:.3f}"

        if allow_transition:
            # Execute transition
            old_regime = self.current_regime
            self.current_regime = regime
            self.regime_start_time = timestamp
            self.transition_count += 1

            # Log transition
            self.transition_history.append((timestamp, old_regime, regime))
            logger.info(f"✓ REGIME TRANSITION #{self.transition_count}: {old_regime} → {regime} @ {timestamp}")
            logger.info(f"  Probabilities: {old_regime}={current_regime_prob:.3f}, {regime}={new_regime_prob:.3f}")
            logger.info(f"  Dwell time in {old_regime}: {dwell_time_hours:.1f} hours")

            return {
                'regime': regime,
                'probs': probs,
                'transition': True,
                'dwell_time': 0.0,  # Just transitioned
                'hysteresis_applied': False,
                'reason': reason
            }
        else:
            # Stay in current regime
            return {
                'regime': self.current_regime,
                'probs': probs,
                'transition': False,
                'dwell_time': dwell_time_hours,
                'hysteresis_applied': True,
                'reason': reason
            }

    def _initialize_regime(
        self,
        regime: str,
        probs: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> Dict:
        """
        Initialize first regime.

        Args:
            regime: Proposed initial regime
            probs: Regime probabilities
            timestamp: Current timestamp

        Returns:
            Hysteresis result dict
        """
        regime_prob = probs.get(regime, 0.0)

        # Check if confidence is sufficient for initialization
        if regime_prob >= self.enter_threshold:
            self.current_regime = regime
            self.regime_start_time = timestamp
            self.transition_count = 0

            logger.info(f"✓ REGIME INITIALIZED: {regime} (prob={regime_prob:.3f}) @ {timestamp}")

            return {
                'regime': regime,
                'probs': probs,
                'transition': True,
                'dwell_time': 0.0,
                'hysteresis_applied': False,
                'reason': f"Initial regime {regime} with prob {regime_prob:.3f} >= enter_threshold {self.enter_threshold:.3f}"
            }
        else:
            # Low confidence - default to proposed regime but mark as uncertain
            self.current_regime = regime
            self.regime_start_time = timestamp
            self.transition_count = 0

            logger.warning(f"⚠ REGIME INITIALIZED (LOW CONFIDENCE): {regime} (prob={regime_prob:.3f} < {self.enter_threshold:.3f}) @ {timestamp}")
            logger.warning(f"  Defaulting to proposed regime despite low confidence")

            return {
                'regime': regime,
                'probs': probs,
                'transition': True,
                'dwell_time': 0.0,
                'hysteresis_applied': False,
                'reason': f"Initial regime {regime} (low confidence {regime_prob:.3f})"
            }

    def _check_dwell_time(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if minimum dwell time has been satisfied for current regime.

        Args:
            timestamp: Current timestamp

        Returns:
            True if dwell time satisfied (can transition), False if locked
        """
        if self.regime_start_time is None or timestamp is None:
            # No dwell time tracking - allow transition
            return True

        min_dwell = self._get_min_dwell()
        dwell_time = self._calculate_dwell_time(timestamp)

        return dwell_time >= min_dwell

    def _calculate_dwell_time(self, timestamp: pd.Timestamp) -> float:
        """
        Calculate hours in current regime.

        Args:
            timestamp: Current timestamp

        Returns:
            Hours in current regime (0.0 if cannot calculate)
        """
        if self.regime_start_time is None or timestamp is None:
            return 0.0

        try:
            delta = timestamp - self.regime_start_time
            hours = delta.total_seconds() / 3600.0
            return max(0.0, hours)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Failed to calculate dwell time: {e}")
            return 0.0

    def _get_min_dwell(self) -> float:
        """
        Get minimum dwell time for current regime.

        Returns:
            Minimum hours required in current regime
        """
        if self.current_regime is None:
            return 0.0

        # Look up min dwell for current regime (default to 24h if not specified)
        return self.min_duration_hours.get(self.current_regime, 24.0)

    def _apply_ewma_smoothing(self, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Apply exponentially weighted moving average to probabilities.

        Formula: smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

        Args:
            probs: Raw probabilities from classifier

        Returns:
            Smoothed probabilities
        """
        if self.ewma_probs is None:
            # First call - initialize with raw probabilities
            self.ewma_probs = probs.copy()
            return probs

        # Apply EWMA
        smoothed = {}
        for regime in probs.keys():
            raw_prob = probs.get(regime, 0.0)
            prev_prob = self.ewma_probs.get(regime, raw_prob)

            # EWMA formula
            smoothed[regime] = self.ewma_alpha * raw_prob + (1 - self.ewma_alpha) * prev_prob

        # Normalize to ensure probabilities sum to 1.0
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        # Update state
        self.ewma_probs = smoothed

        return smoothed

    def reset(self):
        """
        Reset hysteresis state.

        Used for:
        - Starting new backtest
        - Clearing state after training
        - Debugging/testing
        """
        logger.info("Resetting RegimeHysteresis state")

        # Reset state
        self.current_regime = None
        self.regime_start_time = None
        self.ewma_probs = None

        # Log statistics before reset
        if self.transition_count > 0:
            logger.info(f"  Total transitions before reset: {self.transition_count}")
            if len(self.transition_history) > 0:
                logger.info(f"  First transition: {self.transition_history[0]}")
                logger.info(f"  Last transition: {self.transition_history[-1]}")

        # Reset tracking
        self.transition_count = 0
        self.transition_history = []

    def get_statistics(self) -> Dict:
        """
        Get hysteresis statistics.

        Returns:
            Dict with:
                - current_regime (str): Current regime
                - dwell_time_hours (float): Hours in current regime
                - total_transitions (int): Total transitions tracked
                - transition_history (list): List of transitions
                - config (dict): Current configuration
        """
        return {
            'current_regime': self.current_regime,
            'dwell_time_hours': self._calculate_dwell_time(pd.Timestamp.now()),
            'total_transitions': self.transition_count,
            'transition_history': self.transition_history.copy(),
            'config': {
                'enter_threshold': self.enter_threshold,
                'exit_threshold': self.exit_threshold,
                'min_duration_hours': self.min_duration_hours,
                'enable_ewma': self.enable_ewma,
                'ewma_alpha': self.ewma_alpha if self.enable_ewma else None
            }
        }

    def validate_transitions_per_year(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
        """
        Validate transitions per year metric.

        Target: 10-40 transitions/year for crypto

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dict with validation results
        """
        if start_date is None or end_date is None:
            logger.warning("Cannot validate transitions/year: missing dates")
            return {'valid': False, 'reason': 'missing dates'}

        try:
            time_span_years = (end_date - start_date).days / 365.25

            if time_span_years <= 0:
                return {'valid': False, 'reason': 'invalid time span'}

            transitions_per_year = self.transition_count / time_span_years

            # Target: 10-40 transitions/year
            if transitions_per_year < 10:
                status = 'TOO_FEW'
                message = f"⚠ Too few transitions ({transitions_per_year:.0f}/year < 10 target)"
            elif transitions_per_year > 40:
                status = 'TOO_MANY'
                message = f"❌ Too many transitions ({transitions_per_year:.0f}/year > 40 target)"
            else:
                status = 'OPTIMAL'
                message = f"✓ Transitions within target range ({transitions_per_year:.0f}/year, target: 10-40)"

            logger.info(message)

            return {
                'valid': True,
                'status': status,
                'transitions_per_year': transitions_per_year,
                'total_transitions': self.transition_count,
                'time_span_years': time_span_years,
                'message': message
            }

        except Exception as e:
            logger.error(f"Failed to validate transitions: {e}")
            return {'valid': False, 'reason': str(e)}
