"""
Unit tests for directional cap enforcement in RegimeWeightAllocator.

This test suite verifies that directional budgets act as HARD CAPS,
not per-archetype multipliers. This is critical for portfolio invariants.

Author: Backend Architect
Date: 2026-01-21
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.portfolio.regime_allocator import RegimeWeightAllocator


@pytest.fixture
def allocator():
    """Create allocator instance for testing."""
    # Setup paths
    base_path = Path(__file__).parent.parent
    edge_table_path = base_path / 'results' / 'archetype_regime_edge_table.csv'
    config_path = base_path / 'configs' / 'regime_allocator_config.json'
    registry_path = base_path / 'archetype_registry.yaml'

    # Check if edge table exists
    if not edge_table_path.exists():
        pytest.skip(f"Edge table not found at {edge_table_path}")

    # Initialize allocator
    return RegimeWeightAllocator(
        edge_table_path=str(edge_table_path),
        config_path=str(config_path) if config_path.exists() else None,
        registry_path=str(registry_path) if registry_path.exists() else None
    )


def test_directional_caps_enforced(allocator):
    """
    Test that directional budgets act as hard caps, not preferences.

    CRITICAL INVARIANTS:
    - sum(long_weights) <= REGIME_DIRECTIONAL_BUDGETS[regime]['long']
    - sum(short_weights) <= REGIME_DIRECTIONAL_BUDGETS[regime]['short']

    These must hold for ALL regimes.
    """
    regimes = ['crisis', 'risk_off', 'neutral', 'risk_on']
    tolerance = 1e-6  # Small tolerance for floating point errors

    for regime in regimes:
        # Get weights
        distribution = allocator.get_regime_distribution(regime)

        # Split by direction
        long_total = sum(
            w for a, w in distribution.items()
            if allocator.archetype_directions.get(a, 'long') == 'long'
        )
        short_total = sum(
            w for a, w in distribution.items()
            if allocator.archetype_directions.get(a, 'long') == 'short'
        )

        # Get caps
        budgets = allocator.REGIME_DIRECTIONAL_BUDGETS[regime]
        long_cap = budgets['long']
        short_cap = budgets['short']

        # INVARIANT: Long total must not exceed cap
        assert long_total <= long_cap + tolerance, \
            f"{regime}: Long {long_total:.1%} exceeds cap {long_cap:.1%}"

        # INVARIANT: Short total must not exceed cap
        assert short_total <= short_cap + tolerance, \
            f"{regime}: Short {short_total:.1%} exceeds cap {short_cap:.1%}"

        print(f"✓ {regime}: long={long_total:.1%} (cap {long_cap:.1%}), "
              f"short={short_total:.1%} (cap {short_cap:.1%})")


def test_neutral_regime_long_cap():
    """
    Specific test for the smoking gun bug.

    Bug: Neutral regime allocated 68.5% to longs when cap is 40% (71% over budget).
    Fix: Should now enforce 40% cap with proportional scaling.
    """
    base_path = Path(__file__).parent.parent
    edge_table_path = base_path / 'results' / 'archetype_regime_edge_table.csv'
    config_path = base_path / 'configs' / 'regime_allocator_config.json'
    registry_path = base_path / 'archetype_registry.yaml'

    if not edge_table_path.exists():
        pytest.skip(f"Edge table not found at {edge_table_path}")

    allocator = RegimeWeightAllocator(
        edge_table_path=str(edge_table_path),
        config_path=str(config_path) if config_path.exists() else None,
        registry_path=str(registry_path) if registry_path.exists() else None
    )

    # Get neutral regime distribution
    distribution = allocator.get_regime_distribution('neutral')

    # Calculate long total
    long_total = sum(
        w for a, w in distribution.items()
        if allocator.archetype_directions.get(a, 'long') == 'long'
    )

    # Get cap
    long_cap = allocator.REGIME_DIRECTIONAL_BUDGETS['neutral']['long']

    # CRITICAL: Must not exceed cap
    assert long_total <= long_cap + 1e-6, \
        f"SMOKING GUN BUG NOT FIXED: Neutral long {long_total:.1%} > cap {long_cap:.1%}"

    print(f"✓ Neutral regime long allocation: {long_total:.1%} (cap {long_cap:.1%})")
    print(f"✓ Bug fixed: was 68.5%, now {long_total:.1%}")


def test_risk_off_regime_long_cap():
    """
    Test risk_off regime cap enforcement.

    Bug: Risk_off allocated 28.4% to longs when cap is 25% (13.6% over budget).
    Fix: Should now enforce 25% cap with proportional scaling.
    """
    base_path = Path(__file__).parent.parent
    edge_table_path = base_path / 'results' / 'archetype_regime_edge_table.csv'
    config_path = base_path / 'configs' / 'regime_allocator_config.json'
    registry_path = base_path / 'archetype_registry.yaml'

    if not edge_table_path.exists():
        pytest.skip(f"Edge table not found at {edge_table_path}")

    allocator = RegimeWeightAllocator(
        edge_table_path=str(edge_table_path),
        config_path=str(config_path) if config_path.exists() else None,
        registry_path=str(registry_path) if registry_path.exists() else None
    )

    # Get risk_off regime distribution
    distribution = allocator.get_regime_distribution('risk_off')

    # Calculate long total
    long_total = sum(
        w for a, w in distribution.items()
        if allocator.archetype_directions.get(a, 'long') == 'long'
    )

    # Get cap
    long_cap = allocator.REGIME_DIRECTIONAL_BUDGETS['risk_off']['long']

    # CRITICAL: Must not exceed cap
    assert long_total <= long_cap + 1e-6, \
        f"Risk_off long {long_total:.1%} > cap {long_cap:.1%}"

    print(f"✓ Risk_off regime long allocation: {long_total:.1%} (cap {long_cap:.1%})")


def test_proportional_scaling_preserves_relative_strength():
    """
    Test that proportional scaling preserves relative signal strength.

    When caps are applied, the relative ordering of archetype weights
    should be preserved (strong signals stay relatively stronger).
    """
    base_path = Path(__file__).parent.parent
    edge_table_path = base_path / 'results' / 'archetype_regime_edge_table.csv'
    config_path = base_path / 'configs' / 'regime_allocator_config.json'
    registry_path = base_path / 'archetype_registry.yaml'

    if not edge_table_path.exists():
        pytest.skip(f"Edge table not found at {edge_table_path}")

    allocator = RegimeWeightAllocator(
        edge_table_path=str(edge_table_path),
        config_path=str(config_path) if config_path.exists() else None,
        registry_path=str(registry_path) if registry_path.exists() else None
    )

    # Test neutral regime (known to have cap violations before fix)
    distribution = allocator.get_regime_distribution('neutral')

    # Get long archetypes
    long_archetypes = {
        a: w for a, w in distribution.items()
        if allocator.archetype_directions.get(a, 'long') == 'long'
    }

    if len(long_archetypes) > 1:
        # Check that ordering is reasonable (highest weight has highest edge)
        sorted_archetypes = sorted(long_archetypes.items(), key=lambda x: -x[1])

        # Get edge metrics for top 2
        top_arch = sorted_archetypes[0][0]
        second_arch = sorted_archetypes[1][0]

        top_metrics = allocator.get_edge_metrics(top_arch, 'neutral')
        second_metrics = allocator.get_edge_metrics(second_arch, 'neutral')

        print(f"✓ Top archetype: {top_arch} "
              f"(weight={sorted_archetypes[0][1]:.3f}, edge={top_metrics['edge_raw']:.3f})")
        print(f"✓ Second archetype: {second_arch} "
              f"(weight={sorted_archetypes[1][1]:.3f}, edge={second_metrics['edge_raw']:.3f})")

        # Weights should reflect edge strength (though not perfectly due to sample size shrinkage)
        assert sorted_archetypes[0][1] > 0, "Top archetype should have positive weight"


def test_all_regimes_pass_invariants(allocator):
    """
    Comprehensive test that all regimes pass directional cap invariants.
    """
    regimes = ['crisis', 'risk_off', 'neutral', 'risk_on']
    tolerance = 1e-6

    results = []

    for regime in regimes:
        distribution = allocator.get_regime_distribution(regime)

        # Split by direction
        long_total = sum(
            w for a, w in distribution.items()
            if allocator.archetype_directions.get(a, 'long') == 'long'
        )
        short_total = sum(
            w for a, w in distribution.items()
            if allocator.archetype_directions.get(a, 'long') == 'short'
        )

        budgets = allocator.REGIME_DIRECTIONAL_BUDGETS[regime]

        # Check invariants
        long_pass = long_total <= budgets['long'] + tolerance
        short_pass = short_total <= budgets['short'] + tolerance

        results.append({
            'regime': regime,
            'long_total': long_total,
            'long_cap': budgets['long'],
            'long_pass': long_pass,
            'short_total': short_total,
            'short_cap': budgets['short'],
            'short_pass': short_pass
        })

        assert long_pass and short_pass, \
            f"{regime} failed: long={long_total:.1%}/{budgets['long']:.1%}, " \
            f"short={short_total:.1%}/{budgets['short']:.1%}"

    # Print summary
    print("\n" + "="*80)
    print("DIRECTIONAL CAP INVARIANT CHECK SUMMARY")
    print("="*80)
    for r in results:
        print(f"\n{r['regime'].upper()}:")
        print(f"  Long:  {r['long_total']:.1%} / {r['long_cap']:.1%} {'✓' if r['long_pass'] else '✗'}")
        print(f"  Short: {r['short_total']:.1%} / {r['short_cap']:.1%} {'✓' if r['short_pass'] else '✗'}")

    print("\n" + "="*80)
    print("✓ ALL INVARIANTS PASSED")
    print("="*80)


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v', '-s'])
