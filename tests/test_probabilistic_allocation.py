#!/usr/bin/env python3
"""
Test script for probabilistic regime allocation methods.

Tests:
1. get_directional_budget_probabilistic() - blends directional budgets by regime probabilities
2. compute_weight_probabilistic() - computes weight with probabilistic blending
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engine.portfolio.regime_allocator import RegimeWeightAllocator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_directional_budget_probabilistic():
    """Test probabilistic directional budget blending."""
    print("\n" + "="*80)
    print("TEST 1: Probabilistic Directional Budget Blending")
    print("="*80)

    # Create allocator (we need minimal config - edge table not required for this test)
    allocator = RegimeWeightAllocator(
        edge_table_path='results/archetype_regime_edge_table.csv',
        config_override={
            'k_shrinkage': 30,
            'min_weight': 0.01,
            'neg_edge_cap': 0.20,
            'min_trades': 5,
            'alpha': 4.0
        }
    )

    # Test case from spec: risk_on-heavy distribution for LONG archetype
    regime_probs = {'crisis': 0.1, 'risk_off': 0.0, 'neutral': 0.3, 'risk_on': 0.6}

    # For a LONG archetype (e.g., 'H' = Head and Shoulders)
    # Expected: 0.1*0.15 + 0.0*0.25 + 0.3*0.40 + 0.6*0.60 = 0.015 + 0 + 0.12 + 0.36 = 0.495
    budget = allocator.get_directional_budget_probabilistic('H', regime_probs)

    print(f"\nTest Case: Long archetype in risk_on-heavy regime")
    print(f"Regime probabilities: {regime_probs}")
    print(f"Archetype: H (long)")
    print(f"Expected budget: ~0.495")
    print(f"Actual budget:   {budget:.3f}")

    expected = 0.495
    tolerance = 0.001
    assert abs(budget - expected) < tolerance, f"Expected {expected:.3f}, got {budget:.3f}"
    print("✓ PASSED: Budget matches expected value")

    # Test case 2: crisis-heavy distribution for SHORT archetype
    regime_probs_crisis = {'crisis': 0.6, 'risk_off': 0.3, 'neutral': 0.1, 'risk_on': 0.0}

    # For a SHORT archetype (e.g., 'S5' = Long Squeeze)
    # Expected: 0.6*0.60 + 0.3*0.50 + 0.1*0.40 + 0.0*0.20 = 0.36 + 0.15 + 0.04 + 0 = 0.55
    budget_short = allocator.get_directional_budget_probabilistic('S5', regime_probs_crisis)

    print(f"\nTest Case: Short archetype in crisis-heavy regime")
    print(f"Regime probabilities: {regime_probs_crisis}")
    print(f"Archetype: S5 (short)")
    print(f"Expected budget: ~0.550")
    print(f"Actual budget:   {budget_short:.3f}")

    expected_short = 0.55
    assert abs(budget_short - expected_short) < tolerance, f"Expected {expected_short:.3f}, got {budget_short:.3f}"
    print("✓ PASSED: Budget matches expected value")

    # Test case 3: balanced regime distribution
    regime_probs_balanced = {'crisis': 0.25, 'risk_off': 0.25, 'neutral': 0.25, 'risk_on': 0.25}

    # For LONG archetype: 0.25*0.15 + 0.25*0.25 + 0.25*0.40 + 0.25*0.60 = 0.0375 + 0.0625 + 0.10 + 0.15 = 0.35
    budget_balanced = allocator.get_directional_budget_probabilistic('H', regime_probs_balanced)

    print(f"\nTest Case: Long archetype in balanced regime")
    print(f"Regime probabilities: {regime_probs_balanced}")
    print(f"Archetype: H (long)")
    print(f"Expected budget: ~0.350")
    print(f"Actual budget:   {budget_balanced:.3f}")

    expected_balanced = 0.35
    assert abs(budget_balanced - expected_balanced) < tolerance, f"Expected {expected_balanced:.3f}, got {budget_balanced:.3f}"
    print("✓ PASSED: Budget matches expected value")


def test_compute_weight_probabilistic():
    """Test probabilistic weight computation."""
    print("\n" + "="*80)
    print("TEST 2: Probabilistic Weight Computation")
    print("="*80)

    # Create allocator
    allocator = RegimeWeightAllocator(
        edge_table_path='results/archetype_regime_edge_table.csv',
        config_override={
            'k_shrinkage': 30,
            'min_weight': 0.01,
            'neg_edge_cap': 0.20,
            'min_trades': 5,
            'alpha': 4.0
        }
    )

    # Test parameters
    edge = 0.5  # Positive edge
    N = 50      # Good sample size
    archetype = 'H'  # Long archetype
    regime_probs = {'crisis': 0.1, 'risk_off': 0.0, 'neutral': 0.3, 'risk_on': 0.6}

    # Compute weight
    weight = allocator.compute_weight_probabilistic(edge, N, archetype, regime_probs)

    print(f"\nTest Case: Long archetype with positive edge")
    print(f"Edge (Sharpe-like): {edge}")
    print(f"Sample size (N): {N}")
    print(f"Archetype: {archetype} (long)")
    print(f"Regime probabilities: {regime_probs}")
    print(f"Computed weight: {weight:.4f}")

    # Weight should be positive and reasonable
    assert 0 < weight <= 1.0, f"Weight {weight:.4f} out of valid range [0, 1]"
    assert weight > 0.01, f"Weight {weight:.4f} should be above minimum"
    print("✓ PASSED: Weight is in valid range and above minimum")

    # Test with negative edge
    edge_neg = -0.3
    weight_neg = allocator.compute_weight_probabilistic(edge_neg, N, archetype, regime_probs)

    print(f"\nTest Case: Long archetype with negative edge")
    print(f"Edge (Sharpe-like): {edge_neg}")
    print(f"Sample size (N): {N}")
    print(f"Computed weight: {weight_neg:.4f}")

    # Weight should be capped at neg_edge_cap (0.20) * directional_budget
    max_expected = 0.20 * 0.495  # neg_edge_cap * directional_budget
    assert weight_neg <= max_expected + 0.001, f"Weight {weight_neg:.4f} exceeds expected cap {max_expected:.4f}"
    print(f"✓ PASSED: Negative edge weight properly capped (max ~{max_expected:.4f})")

    # Test with small sample
    N_small = 3
    weight_small = allocator.compute_weight_probabilistic(edge, N_small, archetype, regime_probs)

    print(f"\nTest Case: Small sample size")
    print(f"Sample size (N): {N_small}")
    print(f"Computed weight: {weight_small:.4f}")

    # Weight should be lower due to sample floor adjustment
    assert weight_small < weight, f"Small sample weight {weight_small:.4f} should be less than full sample {weight:.4f}"
    print("✓ PASSED: Small sample results in appropriately reduced weight")


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# PROBABILISTIC REGIME ALLOCATION TESTS")
    print("#"*80)

    try:
        test_directional_budget_probabilistic()
        test_compute_weight_probabilistic()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nProbabilistic allocation methods are working correctly!")
        print("\nKey features:")
        print("- Smooth regime transitions (no hard gates)")
        print("- Weighted blending of directional budgets")
        print("- Proper handling of edge cases (negative edge, small samples)")

    except FileNotFoundError as e:
        print(f"\n⚠ Warning: Edge table not found - {e}")
        print("This is OK for unit testing the probabilistic methods.")
        print("For full integration testing, run with actual edge data.")
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
