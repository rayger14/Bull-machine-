"""
PortfolioAllocator - Multi-archetype portfolio allocation with correlation constraints

Handles portfolio-level allocation decisions for simultaneous archetype signals:
- Priority queue allocation (highest-edge signals first)
- Correlation constraints (avoid taking correlated signals simultaneously)
- Regime-conditioned budgets (integrates with RegimeAllocator)
- Deterministic allocation (no execution order dependency)

This solves coupling issues #2 and #6:
- Issue #2: Correlated archetypes reduce diversification
- Issue #6: Execution order affects which archetypes get capital

Algorithm:
1. Collect all active signals from archetypes
2. Compute regime-conditioned edge for each signal
3. Sort by priority (edge × confidence)
4. Allocate greedily, skipping correlated signals (correlation > threshold)
5. Apply directional budget caps (long/short limits per regime)
6. Return deterministic allocation plan

Integration:
- Works with existing RegimeAllocator for regime weights
- Can be integrated into nautilus_strategy.py or backtesting engine
- Supports both single-position (pick best) and multi-position modes

Author: Claude Sonnet 4.5
Date: 2026-02-04
Version: 1.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AllocationMode(Enum):
    """Portfolio allocation modes."""
    SINGLE_BEST = "single_best"      # Take only highest priority signal
    MULTI_UNCORRELATED = "multi_uncorrelated"  # Take multiple uncorrelated signals
    MULTI_ALL = "multi_all"          # Take all signals (no correlation filter)


@dataclass
class ArchetypeSignal:
    """
    Signal from a single archetype with metadata for allocation.

    This is the input to the portfolio allocator - one per archetype
    that has an active signal on the current bar.
    """
    archetype_id: str              # Archetype identifier (e.g., 'S1', 'H', 'spring')
    direction: str                 # 'long' or 'short'
    confidence: float              # Signal confidence [0, 1]
    entry_price: float             # Entry price
    stop_loss: float               # Stop loss level
    take_profit: Optional[float]   # Take profit level (optional)
    fusion_score: float            # Archetype-specific fusion score
    regime_label: str              # Current regime ('risk_on', 'neutral', etc.)
    timestamp: pd.Timestamp        # Signal timestamp
    metadata: Dict = field(default_factory=dict)  # Additional metadata


@dataclass
class AllocationIntent:
    """
    Intent to allocate capital to an archetype signal.

    This is the output of the portfolio allocator - represents
    a decision to take a position with specified size.
    """
    signal: ArchetypeSignal
    allocated_size_pct: float      # % of portfolio to allocate
    priority_score: float          # Priority score (for ranking)
    regime_weight: float           # Regime weight applied
    allocation_reason: str         # Why this was allocated
    metadata: Dict = field(default_factory=dict)


@dataclass
class RejectionReason:
    """Reason why a signal was rejected during allocation."""
    signal: ArchetypeSignal
    reason: str                    # Rejection reason
    details: Dict = field(default_factory=dict)


class PortfolioAllocator:
    """
    Portfolio-level allocator for multi-archetype signals.

    Core Responsibilities:
    1. Prioritize signals by regime-conditioned edge
    2. Filter correlated signals (avoid redundant positions)
    3. Apply regime-directional budgets (long/short caps)
    4. Return deterministic allocation plan

    Key Properties:
    - Deterministic: Same signals → same allocation (no randomness)
    - Regime-aware: Integrates with RegimeAllocator for weights
    - Correlation-aware: Avoids taking highly correlated positions
    - Budget-constrained: Respects directional and total exposure limits
    """

    def __init__(
        self,
        regime_allocator: Optional[object] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        correlation_threshold: float = 0.7,
        max_simultaneous_positions: int = 8,
        allocation_mode: AllocationMode = AllocationMode.MULTI_UNCORRELATED,
        enable_correlation_filter: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize portfolio allocator.

        Args:
            regime_allocator: RegimeWeightAllocator instance (optional)
                If provided, uses regime weights for prioritization.
                If None, uses equal weights.
            correlation_matrix: Archetype correlation matrix (optional)
                DataFrame with archetype IDs as index/columns.
                If None, no correlation filtering applied.
            correlation_threshold: Max correlation to allow simultaneous positions
                Signals with correlation > threshold won't be taken together.
                Default: 0.7 (strong correlation)
            max_simultaneous_positions: Max number of simultaneous positions
                Limits portfolio to N concurrent trades.
                Default: 8
            allocation_mode: How to allocate capital
                - SINGLE_BEST: Take only highest priority signal
                - MULTI_UNCORRELATED: Take multiple uncorrelated signals
                - MULTI_ALL: Take all signals (no correlation filter)
            enable_correlation_filter: Enable/disable correlation filtering
                Useful for A/B testing impact of correlation constraints.
            config: Optional configuration dict for dynamic sizing and other settings.
                Supports keys:
                - risk_per_trade_pct: Base risk per trade (default 0.02)
                - dynamic_sizing_enabled: Enable fusion-score-based sizing (default False)
                - fusion_size_tiers: List of {min_fusion, multiplier} dicts
        """
        self.regime_allocator = regime_allocator
        self.correlation_matrix = correlation_matrix
        self.correlation_threshold = correlation_threshold
        self.max_simultaneous_positions = max_simultaneous_positions
        self.allocation_mode = allocation_mode
        self.enable_correlation_filter = enable_correlation_filter
        self.config = config or {}

        logger.info(
            f"[PortfolioAllocator] Initialized: "
            f"mode={allocation_mode.value}, "
            f"max_positions={max_simultaneous_positions}, "
            f"corr_threshold={correlation_threshold:.2f}, "
            f"corr_filter_enabled={enable_correlation_filter}"
        )

        # Validate correlation matrix if provided
        if correlation_matrix is not None:
            self._validate_correlation_matrix(correlation_matrix)

    def _validate_correlation_matrix(self, corr_matrix: pd.DataFrame):
        """Validate correlation matrix structure."""
        # Check symmetric
        if not corr_matrix.equals(corr_matrix.T):
            logger.warning("Correlation matrix is not symmetric - forcing symmetry")
            # Average with transpose to ensure symmetry
            self.correlation_matrix = (corr_matrix + corr_matrix.T) / 2

        # Check diagonal is 1.0
        diag_values = np.diag(corr_matrix.values)
        if not np.allclose(diag_values, 1.0, atol=0.01):
            logger.warning("Correlation matrix diagonal is not 1.0 - normalizing")
            # Set diagonal to 1.0
            for i in range(len(corr_matrix)):
                self.correlation_matrix.iloc[i, i] = 1.0

        logger.info(
            f"[PortfolioAllocator] Correlation matrix validated: "
            f"{len(corr_matrix)} archetypes"
        )

    def _compute_dynamic_size(self, signal, config: Dict) -> float:
        """
        Dynamic position sizing based on CMI confidence signals.

        Uses the three proven positive predictors from 1,127-trade backtest analysis:
          - dd_score   (r=+0.167***): drawdown regime health
          - risk_temp  (r=+0.126***): market temperature
          - trend_align (r=+0.105***): EMA trend alignment

        NOTE: fusion_score (r=-0.102) was the previous input — it is ANTI-predictive
        and has been removed. Sizing up on high-fusion signals produced worse outcomes.

        Risk scale:
          - Baseline: 2% risk per trade (config risk_per_trade_pct)
          - Range: 0.5% (very low confidence) to 4% (high confidence)
          - Universal good condition (dd>0.20, rt>0.45, ta>0.9): ~3-4%
          - Universal bad condition (ta<0.5, chop>0.40): ~0.5-1%

        Quality formula:
          dd_norm   = clamp(dd_score / 0.20,  0.0, 1.5)   # 0.20 = median threshold
          rt_norm   = clamp(risk_temp / 0.45,  0.0, 1.3)   # 0.45 = median threshold
          ta_norm   = clamp(trend_align / 0.90, 0.0, 1.2)  # 0.90 = strong alignment
          quality   = dd_norm * rt_norm * ta_norm           # raw [0, ~2.34]
          multiplier = clamp(quality / 1.17, 0.25, 2.0)    # normalize: 1.17 = neutral product

        Args:
            signal: ArchetypeSignal with metadata containing dd_score, risk_temp, trend_align
            config: Configuration dict with sizing parameters

        Returns:
            Position size as fraction of portfolio risk (e.g., 0.02 = 2% risk)
        """
        base_size = config.get('risk_per_trade_pct', 0.02)

        if not config.get('dynamic_sizing_enabled', False):
            return base_size

        metadata = getattr(signal, 'metadata', {}) or {}
        dd_score = metadata.get('dd_score', None)
        risk_temp = metadata.get('risk_temp', None)
        trend_align = metadata.get('trend_align', None)

        # Fall back to base size if CMI values not injected (legacy path / live without CMI)
        if dd_score is None or risk_temp is None or trend_align is None:
            logger.debug(
                f"[PortfolioAllocator] Dynamic size: {signal.archetype_id} "
                f"CMI values not available — using base_size={base_size:.3%}"
            )
            return base_size

        # Normalize each component relative to its "good enough" threshold
        dd_norm = min(dd_score / 0.20, 1.5)
        rt_norm = min(risk_temp / 0.45, 1.3)
        ta_norm = min(trend_align / 0.90, 1.2)

        # Composite quality: product of normalized components
        # Neutral market (dd=0.20, rt=0.45, ta=0.90) → quality = 1.0 * 1.0 * 1.0 = 1.0
        # Strong market  (dd=0.35, rt=0.65, ta=1.00) → quality ≈ 1.5*1.3*1.11 ≈ 2.16
        # Weak market    (dd=0.05, rt=0.25, ta=0.50) → quality ≈ 0.25*0.56*0.56 ≈ 0.08
        quality = dd_norm * rt_norm * ta_norm

        # Multiplier: neutral (quality=1.0) → 1.0x, scale 0.25x-2.0x
        # Divide by neutral product (1.0) then clamp
        multiplier = max(0.25, min(quality, 2.0))

        computed_size = base_size * multiplier

        logger.debug(
            f"[PortfolioAllocator] Dynamic size: {signal.archetype_id} "
            f"dd={dd_score:.3f} rt={risk_temp:.3f} ta={trend_align:.3f} "
            f"quality={quality:.3f} multiplier={multiplier:.2f} "
            f"size={computed_size:.3%}"
        )

        return computed_size

    def allocate(
        self,
        signals: List[ArchetypeSignal],
        current_positions: List[str],
        regime_probs: Optional[Dict[str, float]] = None
    ) -> Tuple[List[AllocationIntent], List[RejectionReason]]:
        """
        Allocate capital to signals with correlation and budget constraints.

        This is the main allocation logic - deterministic and correlation-aware.

        Args:
            signals: List of active signals from archetypes
            current_positions: List of archetype IDs with open positions
                Used to prevent duplicate positions in same archetype.
            regime_probs: Regime probability distribution (optional)
                If provided and regime_allocator supports probabilistic mode,
                uses blended regime weights.

        Returns:
            Tuple of:
            - List[AllocationIntent]: Signals to take (with sizes)
            - List[RejectionReason]: Signals rejected (with reasons)
        """
        if not signals:
            return [], []

        logger.info(
            f"[PortfolioAllocator] Allocating {len(signals)} signals, "
            f"{len(current_positions)} current positions"
        )

        # Step 1: Filter out archetypes with existing positions
        available_signals = self._filter_duplicate_archetypes(signals, current_positions)

        if not available_signals:
            rejections = [
                RejectionReason(
                    signal=sig,
                    reason="duplicate_archetype",
                    details={"current_positions": current_positions}
                )
                for sig in signals
            ]
            return [], rejections

        # Step 2: Compute priority scores (regime-weighted edge)
        scored_signals = self._compute_priority_scores(
            available_signals,
            regime_probs
        )

        # Step 3: Sort by priority (descending)
        scored_signals.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Allocate based on mode
        if self.allocation_mode == AllocationMode.SINGLE_BEST:
            return self._allocate_single_best(scored_signals)
        elif self.allocation_mode == AllocationMode.MULTI_UNCORRELATED:
            return self._allocate_multi_uncorrelated(scored_signals, regime_probs)
        elif self.allocation_mode == AllocationMode.MULTI_ALL:
            return self._allocate_multi_all(scored_signals, regime_probs)
        else:
            raise ValueError(f"Unknown allocation mode: {self.allocation_mode}")

    def _filter_duplicate_archetypes(
        self,
        signals: List[ArchetypeSignal],
        current_positions: List[str]
    ) -> List[ArchetypeSignal]:
        """
        Filter out signals from archetypes that already have open positions.

        This prevents double-counting the same archetype.
        """
        filtered = [
            sig for sig in signals
            if sig.archetype_id not in current_positions
        ]

        removed = len(signals) - len(filtered)
        if removed > 0:
            logger.debug(
                f"[PortfolioAllocator] Filtered {removed} duplicate archetypes"
            )

        return filtered

    def _compute_priority_scores(
        self,
        signals: List[ArchetypeSignal],
        regime_probs: Optional[Dict[str, float]]
    ) -> List[Tuple[ArchetypeSignal, float, float]]:
        """
        Compute priority scores for signals using regime weights.

        Priority = regime_weight × confidence

        Returns:
            List of (signal, priority_score, regime_weight) tuples
        """
        scored = []

        for signal in signals:
            # Get regime weight
            if self.regime_allocator:
                if regime_probs is not None:
                    # Probabilistic mode
                    regime_weight = self.regime_allocator.compute_weight_probabilistic(
                        edge=0.5,  # Placeholder - use historical edge if available
                        N=50,      # Placeholder - use historical trade count
                        archetype=signal.archetype_id,
                        regime_probs=regime_probs,
                        is_entry=True
                    )
                else:
                    # Discrete mode
                    regime_weight = self.regime_allocator.get_weight(
                        signal.archetype_id,
                        signal.regime_label
                    )
            else:
                # No regime allocator - equal weight
                regime_weight = 1.0

            # Compute priority score
            priority = regime_weight * signal.confidence

            scored.append((signal, priority, regime_weight))

            logger.debug(
                f"[PortfolioAllocator] {signal.archetype_id}: "
                f"confidence={signal.confidence:.3f}, "
                f"regime_weight={regime_weight:.3f}, "
                f"priority={priority:.3f}"
            )

        return scored

    def _allocate_single_best(
        self,
        scored_signals: List[Tuple[ArchetypeSignal, float, float]]
    ) -> Tuple[List[AllocationIntent], List[RejectionReason]]:
        """
        Allocate to only the highest priority signal.

        This is the simplest mode - take best signal, reject rest.
        """
        if not scored_signals:
            return [], []

        # Take highest priority
        best_signal, priority, regime_weight = scored_signals[0]

        size_pct = self._compute_dynamic_size(best_signal, self.config)

        intent = AllocationIntent(
            signal=best_signal,
            allocated_size_pct=size_pct,
            priority_score=priority,
            regime_weight=regime_weight,
            allocation_reason="highest_priority",
            metadata={"rank": 1, "total_candidates": len(scored_signals)}
        )

        # Reject rest
        rejections = [
            RejectionReason(
                signal=sig,
                reason="not_highest_priority",
                details={"rank": i+1, "priority": pri}
            )
            for i, (sig, pri, _) in enumerate(scored_signals[1:], start=1)
        ]

        logger.info(
            f"[PortfolioAllocator] Single best: {best_signal.archetype_id} "
            f"(priority={priority:.3f})"
        )

        return [intent], rejections

    def _allocate_multi_uncorrelated(
        self,
        scored_signals: List[Tuple[ArchetypeSignal, float, float]],
        regime_probs: Optional[Dict[str, float]]
    ) -> Tuple[List[AllocationIntent], List[RejectionReason]]:
        """
        Allocate to multiple signals, filtering correlated ones.

        Algorithm:
        1. Take highest priority signal
        2. For each remaining signal (in priority order):
           - Check correlation with already-selected signals
           - If correlation < threshold, select it
           - Stop when max_simultaneous_positions reached

        This is the core correlation-aware allocation logic.
        """
        if not scored_signals:
            return [], []

        intents = []
        rejections = []
        selected_archetypes = []

        # Get regime for directional budgets
        regime_label = scored_signals[0][0].regime_label

        # Get directional budgets from regime allocator
        if self.regime_allocator:
            directional_budgets = self.regime_allocator.REGIME_DIRECTIONAL_BUDGETS.get(
                regime_label,
                {'long': 0.5, 'short': 0.5}
            )
        else:
            directional_budgets = {'long': 0.5, 'short': 0.5}

        # Track allocated exposure by direction
        long_allocated = 0.0
        short_allocated = 0.0

        for i, (signal, priority, regime_weight) in enumerate(scored_signals):
            # Check max positions limit
            if len(intents) >= self.max_simultaneous_positions:
                rejections.append(RejectionReason(
                    signal=signal,
                    reason="max_positions_reached",
                    details={
                        "max_positions": self.max_simultaneous_positions,
                        "rank": i+1
                    }
                ))
                continue

            # Check correlation with already-selected signals
            if self.enable_correlation_filter and self.correlation_matrix is not None:
                is_correlated, corr_details = self._check_correlation(
                    signal.archetype_id,
                    selected_archetypes
                )

                if is_correlated:
                    rejections.append(RejectionReason(
                        signal=signal,
                        reason="correlated_with_existing",
                        details=corr_details
                    ))
                    continue

            # Check directional budget
            direction = signal.direction
            size_pct = self._compute_dynamic_size(signal, self.config)

            if direction == 'long':
                if (long_allocated + size_pct) > directional_budgets['long']:
                    rejections.append(RejectionReason(
                        signal=signal,
                        reason="long_budget_exceeded",
                        details={
                            "long_budget": directional_budgets['long'],
                            "long_allocated": long_allocated,
                            "requested": size_pct
                        }
                    ))
                    continue
                long_allocated += size_pct
            else:  # short
                if (short_allocated + size_pct) > directional_budgets['short']:
                    rejections.append(RejectionReason(
                        signal=signal,
                        reason="short_budget_exceeded",
                        details={
                            "short_budget": directional_budgets['short'],
                            "short_allocated": short_allocated,
                            "requested": size_pct
                        }
                    ))
                    continue
                short_allocated += size_pct

            # Accept signal
            intent = AllocationIntent(
                signal=signal,
                allocated_size_pct=size_pct,
                priority_score=priority,
                regime_weight=regime_weight,
                allocation_reason="uncorrelated_and_priority",
                metadata={
                    "rank": i+1,
                    "total_candidates": len(scored_signals),
                    "selected_count": len(intents) + 1
                }
            )

            intents.append(intent)
            selected_archetypes.append(signal.archetype_id)

            logger.info(
                f"[PortfolioAllocator] Selected #{len(intents)}: "
                f"{signal.archetype_id} ({direction}) "
                f"priority={priority:.3f}, "
                f"size={size_pct:.1%}"
            )

        logger.info(
            f"[PortfolioAllocator] Multi-uncorrelated allocation: "
            f"{len(intents)} selected, {len(rejections)} rejected, "
            f"long={long_allocated:.1%}, short={short_allocated:.1%}"
        )

        return intents, rejections

    def _allocate_multi_all(
        self,
        scored_signals: List[Tuple[ArchetypeSignal, float, float]],
        regime_probs: Optional[Dict[str, float]]
    ) -> Tuple[List[AllocationIntent], List[RejectionReason]]:
        """
        Allocate to all signals (no correlation filter).

        This mode takes all signals up to max_simultaneous_positions,
        respecting only directional budgets.

        Useful for comparison testing vs correlation-filtered allocation.
        """
        if not scored_signals:
            return [], []

        intents = []
        rejections = []

        # Get regime for directional budgets
        regime_label = scored_signals[0][0].regime_label

        # Get directional budgets
        if self.regime_allocator:
            directional_budgets = self.regime_allocator.REGIME_DIRECTIONAL_BUDGETS.get(
                regime_label,
                {'long': 0.5, 'short': 0.5}
            )
        else:
            directional_budgets = {'long': 0.5, 'short': 0.5}

        # Track allocated exposure
        long_allocated = 0.0
        short_allocated = 0.0

        for i, (signal, priority, regime_weight) in enumerate(scored_signals):
            # Check max positions limit
            if len(intents) >= self.max_simultaneous_positions:
                rejections.append(RejectionReason(
                    signal=signal,
                    reason="max_positions_reached",
                    details={"max_positions": self.max_simultaneous_positions}
                ))
                continue

            # Check directional budget
            direction = signal.direction
            size_pct = self._compute_dynamic_size(signal, self.config)

            if direction == 'long':
                if (long_allocated + size_pct) > directional_budgets['long']:
                    rejections.append(RejectionReason(
                        signal=signal,
                        reason="long_budget_exceeded",
                        details={
                            "long_budget": directional_budgets['long'],
                            "long_allocated": long_allocated
                        }
                    ))
                    continue
                long_allocated += size_pct
            else:  # short
                if (short_allocated + size_pct) > directional_budgets['short']:
                    rejections.append(RejectionReason(
                        signal=signal,
                        reason="short_budget_exceeded",
                        details={
                            "short_budget": directional_budgets['short'],
                            "short_allocated": short_allocated
                        }
                    ))
                    continue
                short_allocated += size_pct

            # Accept signal
            intent = AllocationIntent(
                signal=signal,
                allocated_size_pct=size_pct,
                priority_score=priority,
                regime_weight=regime_weight,
                allocation_reason="multi_all_mode",
                metadata={"rank": i+1, "total_candidates": len(scored_signals)}
            )

            intents.append(intent)

        logger.info(
            f"[PortfolioAllocator] Multi-all allocation: "
            f"{len(intents)} selected, {len(rejections)} rejected"
        )

        return intents, rejections

    def _check_correlation(
        self,
        archetype_id: str,
        selected_archetypes: List[str]
    ) -> Tuple[bool, Dict]:
        """
        Check if archetype is correlated with any already-selected archetype.

        Args:
            archetype_id: Archetype to check
            selected_archetypes: List of already-selected archetype IDs

        Returns:
            Tuple of:
            - bool: True if correlated (above threshold)
            - dict: Details (max_correlation, correlated_with)
        """
        if not selected_archetypes:
            return False, {}

        if self.correlation_matrix is None:
            return False, {}

        # Check if archetype in correlation matrix
        if archetype_id not in self.correlation_matrix.index:
            logger.warning(
                f"[PortfolioAllocator] {archetype_id} not in correlation matrix - "
                f"assuming uncorrelated"
            )
            return False, {}

        # Check correlation with each selected archetype
        max_correlation = 0.0
        correlated_with = None

        for other_id in selected_archetypes:
            if other_id not in self.correlation_matrix.columns:
                continue

            corr = abs(self.correlation_matrix.loc[archetype_id, other_id])

            if corr > max_correlation:
                max_correlation = corr
                correlated_with = other_id

        is_correlated = max_correlation >= self.correlation_threshold

        details = {
            "max_correlation": max_correlation,
            "correlated_with": correlated_with,
            "threshold": self.correlation_threshold,
            "selected_archetypes": selected_archetypes
        }

        if is_correlated:
            logger.debug(
                f"[PortfolioAllocator] {archetype_id} correlated with {correlated_with} "
                f"(corr={max_correlation:.3f} > threshold={self.correlation_threshold:.3f})"
            )

        return is_correlated, details

    def get_allocation_summary(
        self,
        intents: List[AllocationIntent],
        rejections: List[RejectionReason]
    ) -> str:
        """
        Get human-readable allocation summary.

        Useful for logging and debugging allocation decisions.
        """
        lines = [
            "\n" + "="*80,
            "PORTFOLIO ALLOCATION SUMMARY",
            "="*80,
            f"Mode: {self.allocation_mode.value}",
            f"Max Positions: {self.max_simultaneous_positions}",
            f"Correlation Threshold: {self.correlation_threshold:.2f}",
            f"Correlation Filter Enabled: {self.enable_correlation_filter}",
            ""
        ]

        # Allocated signals
        if intents:
            lines.append(f"ALLOCATED SIGNALS ({len(intents)}):")
            for i, intent in enumerate(intents, 1):
                sig = intent.signal
                lines.append(
                    f"  {i}. {sig.archetype_id} ({sig.direction}): "
                    f"size={intent.allocated_size_pct:.1%}, "
                    f"priority={intent.priority_score:.3f}, "
                    f"regime_weight={intent.regime_weight:.3f}, "
                    f"confidence={sig.confidence:.3f}"
                )
        else:
            lines.append("ALLOCATED SIGNALS: None")

        lines.append("")

        # Rejected signals
        if rejections:
            lines.append(f"REJECTED SIGNALS ({len(rejections)}):")

            # Group by reason
            by_reason = {}
            for rej in rejections:
                if rej.reason not in by_reason:
                    by_reason[rej.reason] = []
                by_reason[rej.reason].append(rej)

            for reason, rejects in by_reason.items():
                lines.append(f"  {reason} ({len(rejects)}):")
                for rej in rejects:
                    details_str = ", ".join(
                        f"{k}={v}" for k, v in rej.details.items()
                        if k not in ['selected_archetypes']
                    )
                    lines.append(
                        f"    - {rej.signal.archetype_id} ({rej.signal.direction}): "
                        f"{details_str}"
                    )
        else:
            lines.append("REJECTED SIGNALS: None")

        lines.append("="*80)

        return "\n".join(lines)
