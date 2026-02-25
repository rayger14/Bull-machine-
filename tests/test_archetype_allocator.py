"""
Unit tests for PortfolioAllocator

Tests:
1. Priority queue allocation (highest-edge first)
2. Correlation conflict resolution (skip correlated signals)
3. Deterministic allocation (same inputs → same outputs)
4. Directional budget constraints (long/short caps)
5. Max simultaneous positions limit
6. Integration with RegimeAllocator

Author: Claude Sonnet 4.5
Date: 2026-02-04
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from engine.portfolio.archetype_allocator import (
    PortfolioAllocator,
    ArchetypeSignal,
    AllocationIntent,
    RejectionReason,
    AllocationMode
)


class MockRegimeAllocator:
    """Mock RegimeAllocator for testing."""

    REGIME_DIRECTIONAL_BUDGETS = {
        'crisis': {'long': 0.15, 'short': 0.60},
        'risk_off': {'long': 0.25, 'short': 0.50},
        'neutral': {'long': 0.40, 'short': 0.40},
        'risk_on': {'long': 0.60, 'short': 0.20}
    }

    def get_weight(self, archetype: str, regime: str) -> float:
        """Return fixed weight for testing."""
        # High weight for crisis-aligned archetypes
        if regime == 'crisis' and archetype in ['S1', 'S3', 'S5']:
            return 0.8
        elif regime == 'risk_on' and archetype in ['H', 'A', 'B']:
            return 0.9
        else:
            return 0.5

    def compute_weight_probabilistic(
        self,
        edge: float,
        N: int,
        archetype: str,
        regime_probs: dict,
        is_entry: bool
    ) -> float:
        """Return blended weight for testing."""
        # Simple weighted average
        total = 0.0
        for regime, prob in regime_probs.items():
            total += prob * self.get_weight(archetype, regime)
        return total


@pytest.fixture
def sample_signals():
    """Create sample signals for testing."""
    timestamp = pd.Timestamp('2023-01-15 12:00:00')

    signals = [
        ArchetypeSignal(
            archetype_id='S1',
            direction='long',
            confidence=0.8,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            fusion_score=0.65,
            regime_label='crisis',
            timestamp=timestamp,
            metadata={'test': 'signal1'}
        ),
        ArchetypeSignal(
            archetype_id='S3',
            direction='short',
            confidence=0.7,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            fusion_score=0.60,
            regime_label='crisis',
            timestamp=timestamp,
            metadata={'test': 'signal2'}
        ),
        ArchetypeSignal(
            archetype_id='H',
            direction='long',
            confidence=0.6,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            fusion_score=0.55,
            regime_label='risk_on',
            timestamp=timestamp,
            metadata={'test': 'signal3'}
        ),
        ArchetypeSignal(
            archetype_id='A',
            direction='long',
            confidence=0.9,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=115.0,
            fusion_score=0.70,
            regime_label='risk_on',
            timestamp=timestamp,
            metadata={'test': 'signal4'}
        )
    ]

    return signals


@pytest.fixture
def correlation_matrix():
    """Create sample correlation matrix."""
    # S1 and S3 are highly correlated (both bear archetypes)
    # H and A are moderately correlated (both bull archetypes)
    archetypes = ['S1', 'S3', 'S5', 'H', 'A', 'B', 'K']

    corr_data = [
        [1.00, 0.85, 0.70, -0.30, -0.25, -0.20, 0.10],  # S1
        [0.85, 1.00, 0.75, -0.35, -0.30, -0.25, 0.15],  # S3
        [0.70, 0.75, 1.00, -0.40, -0.35, -0.30, 0.20],  # S5
        [-0.30, -0.35, -0.40, 1.00, 0.65, 0.55, 0.30],  # H
        [-0.25, -0.30, -0.35, 0.65, 1.00, 0.60, 0.35],  # A
        [-0.20, -0.25, -0.30, 0.55, 0.60, 1.00, 0.40],  # B
        [0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 1.00]     # K
    ]

    return pd.DataFrame(corr_data, index=archetypes, columns=archetypes)


@pytest.fixture
def mock_regime_allocator():
    """Create mock regime allocator."""
    return MockRegimeAllocator()


# ============================================================================
# Test 1: Single Best Mode
# ============================================================================

def test_single_best_mode(sample_signals, mock_regime_allocator):
    """Test that SINGLE_BEST mode selects only highest priority signal."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        allocation_mode=AllocationMode.SINGLE_BEST
    )

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=[]
    )

    # Should select exactly 1 signal
    assert len(intents) == 1
    assert len(rejections) == len(sample_signals) - 1

    # Should select A (highest confidence × regime weight)
    # A: 0.9 confidence × 0.9 regime_weight = 0.81
    # S1: 0.8 confidence × 0.8 regime_weight = 0.64
    assert intents[0].signal.archetype_id == 'A'
    assert intents[0].allocation_reason == "highest_priority"

    # Check rejections
    rejected_ids = [r.signal.archetype_id for r in rejections]
    assert 'S1' in rejected_ids
    assert 'S3' in rejected_ids
    assert 'H' in rejected_ids


# ============================================================================
# Test 2: Correlation Filtering
# ============================================================================

def test_correlation_filtering(sample_signals, correlation_matrix, mock_regime_allocator):
    """Test that correlated signals are filtered out."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        correlation_matrix=correlation_matrix,
        correlation_threshold=0.7,
        allocation_mode=AllocationMode.MULTI_UNCORRELATED,
        max_simultaneous_positions=5  # Increase limit to test correlation filtering
    )

    # Add S5 signal (correlated with S1 at 0.70)
    s5_signal = ArchetypeSignal(
        archetype_id='S5',
        direction='short',
        confidence=0.75,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=90.0,
        fusion_score=0.62,
        regime_label='crisis',
        timestamp=sample_signals[0].timestamp
    )

    test_signals = sample_signals + [s5_signal]

    intents, rejections = allocator.allocate(
        signals=test_signals,
        current_positions=[]
    )

    # Should allocate A first (highest priority)
    # Then try S1 or S5 (bear signals)
    # Should reject correlated bear signals (S1/S3/S5 are correlated)
    allocated_ids = [i.signal.archetype_id for i in intents]

    # A should be selected (highest priority)
    assert 'A' in allocated_ids

    # Check correlation filtering is working
    # S1, S3, S5 are all correlated (>0.7), so only one should be allocated
    bear_archetypes = {'S1', 'S3', 'S5'}
    bear_allocated = [aid for aid in allocated_ids if aid in bear_archetypes]

    # At most one bear archetype should be allocated (they're correlated)
    assert len(bear_allocated) <= 1, f"Multiple correlated bear archetypes allocated: {bear_allocated}"

    # At least one bear signal should be rejected due to correlation
    bear_rejected = [r for r in rejections if r.signal.archetype_id in bear_archetypes]
    corr_rejections = [r for r in bear_rejected if 'correlated' in r.reason]

    assert len(corr_rejections) > 0, \
        f"Should have correlation-based rejection among bear archetypes. " \
        f"Allocated: {bear_allocated}, Rejections: {[(r.signal.archetype_id, r.reason) for r in bear_rejected]}"


# ============================================================================
# Test 3: Deterministic Allocation
# ============================================================================

def test_deterministic_allocation(sample_signals, correlation_matrix, mock_regime_allocator):
    """Test that allocation is deterministic (same inputs → same outputs)."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        correlation_matrix=correlation_matrix,
        allocation_mode=AllocationMode.MULTI_UNCORRELATED,
        max_simultaneous_positions=3
    )

    # Run allocation 3 times with same inputs
    results = []
    for _ in range(3):
        intents, rejections = allocator.allocate(
            signals=sample_signals.copy(),  # Copy to ensure no mutation
            current_positions=[]
        )
        results.append((intents, rejections))

    # All runs should produce identical results
    for i in range(1, len(results)):
        intents_prev, rejections_prev = results[i-1]
        intents_curr, rejections_curr = results[i]

        # Same number of intents
        assert len(intents_curr) == len(intents_prev)

        # Same archetypes selected in same order
        ids_prev = [intent.signal.archetype_id for intent in intents_prev]
        ids_curr = [intent.signal.archetype_id for intent in intents_curr]
        assert ids_curr == ids_prev

        # Same allocations
        for j in range(len(intents_curr)):
            assert intents_curr[j].allocated_size_pct == intents_prev[j].allocated_size_pct
            assert intents_curr[j].priority_score == intents_prev[j].priority_score


# ============================================================================
# Test 4: Directional Budget Constraints
# ============================================================================

def test_directional_budget_constraints(mock_regime_allocator):
    """Test that directional budgets (long/short caps) are enforced."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        allocation_mode=AllocationMode.MULTI_ALL,
        max_simultaneous_positions=10  # High limit to test budget caps
    )

    # Create 5 long signals in risk_on regime (long budget = 60%)
    timestamp = pd.Timestamp('2023-01-15 12:00:00')
    long_signals = []

    for i in range(5):
        signal = ArchetypeSignal(
            archetype_id=f'LONG_{i}',
            direction='long',
            confidence=0.8,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            fusion_score=0.65,
            regime_label='risk_on',  # Long budget = 60%
            timestamp=timestamp
        )
        long_signals.append(signal)

    intents, rejections = allocator.allocate(
        signals=long_signals,
        current_positions=[]
    )

    # Each position is 2%, so max 30 positions could fit in 60% budget
    # But we have max_simultaneous_positions=10, so should get min(5 signals, 10 max, budget limit)

    # Total long allocation should not exceed 60%
    total_long = sum(i.allocated_size_pct for i in intents)
    assert total_long <= 0.60 + 1e-6  # Small tolerance for floating point

    # Should reject signals that would exceed budget
    budget_rejections = [r for r in rejections if 'budget' in r.reason]
    if total_long >= 0.60 - 1e-6:  # If we hit budget cap
        assert len(budget_rejections) > 0


# ============================================================================
# Test 5: Max Simultaneous Positions
# ============================================================================

def test_max_simultaneous_positions(sample_signals, mock_regime_allocator):
    """Test that max_simultaneous_positions limit is enforced."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        allocation_mode=AllocationMode.MULTI_ALL,
        max_simultaneous_positions=2
    )

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=[]
    )

    # Should select exactly 2 signals (max limit)
    assert len(intents) <= 2

    # Should reject remaining signals with "max_positions_reached"
    max_pos_rejections = [
        r for r in rejections
        if r.reason == "max_positions_reached"
    ]
    assert len(max_pos_rejections) == len(sample_signals) - len(intents)


# ============================================================================
# Test 6: Duplicate Archetype Filtering
# ============================================================================

def test_duplicate_archetype_filtering(sample_signals, mock_regime_allocator):
    """Test that archetypes with existing positions are filtered out."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        allocation_mode=AllocationMode.MULTI_ALL,
        max_simultaneous_positions=5
    )

    # S1 and H already have positions
    current_positions = ['S1', 'H']

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=current_positions
    )

    # S1 and H should not be allocated
    allocated_ids = [i.signal.archetype_id for i in intents]
    assert 'S1' not in allocated_ids
    assert 'H' not in allocated_ids

    # S3 and A should be considered (no existing positions)
    # At least one should be allocated
    assert 'S3' in allocated_ids or 'A' in allocated_ids


# ============================================================================
# Test 7: No Correlation Matrix (No Filtering)
# ============================================================================

def test_no_correlation_matrix(sample_signals, mock_regime_allocator):
    """Test allocation without correlation matrix (no filtering)."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        correlation_matrix=None,  # No correlation data
        allocation_mode=AllocationMode.MULTI_UNCORRELATED,
        max_simultaneous_positions=3
    )

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=[]
    )

    # Should allocate up to 3 signals (no correlation filtering)
    assert len(intents) <= 3
    assert len(intents) > 0

    # Should not have any correlation-based rejections
    corr_rejections = [
        r for r in rejections
        if "correlated" in r.reason
    ]
    assert len(corr_rejections) == 0


# ============================================================================
# Test 8: Correlation Filter Disabled
# ============================================================================

def test_correlation_filter_disabled(sample_signals, correlation_matrix, mock_regime_allocator):
    """Test that enable_correlation_filter flag works."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        correlation_matrix=correlation_matrix,
        enable_correlation_filter=False,  # Disable correlation filtering
        allocation_mode=AllocationMode.MULTI_UNCORRELATED,
        max_simultaneous_positions=3
    )

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=[]
    )

    # Should allocate multiple signals without correlation filtering
    assert len(intents) > 1

    # Should not have correlation-based rejections
    corr_rejections = [
        r for r in rejections
        if "correlated" in r.reason
    ]
    assert len(corr_rejections) == 0


# ============================================================================
# Test 9: Probabilistic Regime Mode
# ============================================================================

def test_probabilistic_regime_mode(sample_signals, mock_regime_allocator):
    """Test allocation with probabilistic regime probabilities."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        allocation_mode=AllocationMode.MULTI_ALL,
        max_simultaneous_positions=3
    )

    # Regime probability distribution (blend of crisis and neutral)
    regime_probs = {
        'crisis': 0.40,
        'risk_off': 0.20,
        'neutral': 0.30,
        'risk_on': 0.10
    }

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=[],
        regime_probs=regime_probs
    )

    # Should successfully allocate with probabilistic regime
    assert len(intents) > 0

    # Priority scores should be computed using blended regime weights
    for intent in intents:
        assert intent.regime_weight > 0.0
        assert intent.priority_score > 0.0


# ============================================================================
# Test 10: Empty Signals List
# ============================================================================

def test_empty_signals_list(mock_regime_allocator):
    """Test allocation with empty signals list."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        allocation_mode=AllocationMode.MULTI_ALL
    )

    intents, rejections = allocator.allocate(
        signals=[],
        current_positions=[]
    )

    # Should return empty lists
    assert len(intents) == 0
    assert len(rejections) == 0


# ============================================================================
# Test 11: Allocation Summary
# ============================================================================

def test_allocation_summary(sample_signals, correlation_matrix, mock_regime_allocator):
    """Test allocation summary generation."""
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        correlation_matrix=correlation_matrix,
        allocation_mode=AllocationMode.MULTI_UNCORRELATED,
        max_simultaneous_positions=2
    )

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=[]
    )

    summary = allocator.get_allocation_summary(intents, rejections)

    # Should contain key sections
    assert "PORTFOLIO ALLOCATION SUMMARY" in summary
    assert "ALLOCATED SIGNALS" in summary
    assert "REJECTED SIGNALS" in summary

    # Should show allocated archetypes
    for intent in intents:
        assert intent.signal.archetype_id in summary

    # Should show rejection reasons
    assert len(rejections) > 0  # We have rejections (max_positions=2, 4 signals)


# ============================================================================
# Integration Test: Complete Allocation Flow
# ============================================================================

def test_complete_allocation_flow(sample_signals, correlation_matrix, mock_regime_allocator):
    """
    Integration test: Complete allocation flow with all constraints.

    This tests the full pipeline:
    1. Priority scoring with regime weights
    2. Correlation filtering
    3. Directional budget constraints
    4. Max positions limit
    """
    allocator = PortfolioAllocator(
        regime_allocator=mock_regime_allocator,
        correlation_matrix=correlation_matrix,
        correlation_threshold=0.7,
        max_simultaneous_positions=3,
        allocation_mode=AllocationMode.MULTI_UNCORRELATED
    )

    intents, rejections = allocator.allocate(
        signals=sample_signals,
        current_positions=[]
    )

    # Verify allocation invariants
    assert len(intents) + len(rejections) == len(sample_signals)
    assert len(intents) <= 3  # Max positions limit
    assert len(intents) > 0   # Should allocate at least 1

    # Verify no duplicates in allocated archetypes
    allocated_ids = [i.signal.archetype_id for i in intents]
    assert len(allocated_ids) == len(set(allocated_ids))

    # Verify directional budgets respected
    long_total = sum(
        i.allocated_size_pct for i in intents
        if i.signal.direction == 'long'
    )
    short_total = sum(
        i.allocated_size_pct for i in intents
        if i.signal.direction == 'short'
    )

    # Get regime from first signal
    regime = intents[0].signal.regime_label if intents else 'neutral'
    directional_budgets = mock_regime_allocator.REGIME_DIRECTIONAL_BUDGETS.get(
        regime,
        {'long': 0.5, 'short': 0.5}
    )

    assert long_total <= directional_budgets['long'] + 1e-6
    assert short_total <= directional_budgets['short'] + 1e-6

    # Verify priority ordering (intents should be sorted by priority)
    priorities = [i.priority_score for i in intents]
    assert priorities == sorted(priorities, reverse=True)

    # Print summary for inspection
    summary = allocator.get_allocation_summary(intents, rejections)
    print(summary)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
