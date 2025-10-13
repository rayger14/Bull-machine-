"""
Unit tests for v1.8.5 Fibonacci Levels - Bull Machine

Tests negative fib extensions, confluence detection, and bonus calculation.
"""

import pytest
import pandas as pd
import numpy as np
from engine.liquidity.fib_levels import (
    calculate_fib_levels,
    check_fib_confluence,
    calculate_fib_bonus
)


def test_calculate_fib_levels_basic():
    """Test basic fib level calculation with positive ratios only."""
    config = {
        'fib_levels': [0.618, 0.786],
        'negative_fibs_enabled': False
    }

    levels = calculate_fib_levels(100.0, 80.0, config)

    # Should only have positive retracements
    assert len(levels) == 2
    assert all(l >= 80.0 for l in levels)  # All above swing low
    assert all(l <= 100.0 for l in levels)  # All below swing high

    # Check specific calculations
    # 100 - 0.618 * 20 = 87.64
    # 100 - 0.786 * 20 = 84.28
    assert abs(levels[0] - 84.28) < 0.01
    assert abs(levels[1] - 87.64) < 0.01


def test_calculate_fib_levels_negative():
    """Test negative fib extensions below swing low."""
    config = {
        'fib_levels': [0.618, 0.786],
        'negative_fibs': [-0.272, -0.618],
        'negative_fibs_enabled': True
    }

    levels = calculate_fib_levels(100.0, 80.0, config)

    # Should have 4 levels: 2 negative + 2 positive
    assert len(levels) == 4

    # Negative levels should be below swing low
    negative_levels = [l for l in levels if l < 80.0]
    assert len(negative_levels) == 2

    # -0.618 * 20 = -12.36 → 80 - 12.36 = 67.64
    # -0.272 * 20 = -5.44 → 80 - 5.44 = 74.56
    assert any(abs(l - 67.64) < 0.01 for l in levels)
    assert any(abs(l - 74.56) < 0.01 for l in levels)


def test_fib_levels_sorted():
    """Ensure levels are returned in sorted order."""
    config = {
        'fib_levels': [0.618, 0.786],
        'negative_fibs': [-0.272, -0.618],
        'negative_fibs_enabled': True
    }

    levels = calculate_fib_levels(100.0, 80.0, config)

    # Verify sorted ascending
    assert levels == sorted(levels)


def test_check_fib_confluence_hit():
    """Test confluence detection when price is within tolerance."""
    config = {'fib_tolerance': 0.005}  # 0.5%
    levels = [92.36, 95.72, 100.0]

    # Price within 0.5% of 95.72
    is_at, closest, dist = check_fib_confluence(95.80, levels, config)

    assert is_at is True
    assert abs(closest - 95.72) < 0.01
    assert dist < 0.005


def test_check_fib_confluence_miss():
    """Test confluence detection when price is outside tolerance."""
    config = {'fib_tolerance': 0.005}
    levels = [92.36, 95.72, 100.0]

    # Price far from any level
    is_at, closest, dist = check_fib_confluence(97.0, levels, config)

    assert is_at is False
    assert dist >= 0.005


def test_calculate_fib_bonus_at_level():
    """Test bonus calculation when price is at fib level."""
    # 0.618 retracement: 100 - 0.618 * 20 = 87.64
    df = pd.DataFrame({
        'high': [100] * 60,
        'low': [80] * 60,
        'close': [87.64] * 60  # Exactly at 0.618 retracement
    })

    config = {
        'liquidity': {
            'fib_levels': [0.618],
            'negative_fibs': [-0.272],
            'fib_tolerance': 0.01,
            'negative_fibs_enabled': True
        }
    }

    bonus = calculate_fib_bonus(df, config)

    assert bonus == 0.15  # Max bonus


def test_calculate_fib_bonus_not_at_level():
    """Test bonus calculation when price is away from fib levels."""
    df = pd.DataFrame({
        'high': [100] * 60,
        'low': [80] * 60,
        'close': [90.0] * 60  # Not near any fib level
    })

    config = {
        'liquidity': {
            'fib_levels': [0.618, 0.786],
            'negative_fibs': [-0.272, -0.618],
            'fib_tolerance': 0.005,
            'negative_fibs_enabled': True
        }
    }

    bonus = calculate_fib_bonus(df, config)

    assert bonus == 0.0


def test_fib_bonus_bounds():
    """Ensure bonus is always in [0.0, 0.15] range."""
    df = pd.DataFrame({
        'high': np.random.uniform(90, 110, 100),
        'low': np.random.uniform(70, 90, 100),
        'close': np.random.uniform(80, 100, 100)
    })

    config = {
        'liquidity': {
            'fib_levels': [0.618, 0.786],
            'negative_fibs': [-0.272],
            'fib_tolerance': 0.01,
            'negative_fibs_enabled': True
        }
    }

    bonus = calculate_fib_bonus(df, config)

    assert 0.0 <= bonus <= 0.15


def test_performance_fib_calculation():
    """Microbench: fib calculation should be fast (<2ms target)."""
    import time

    config = {
        'fib_levels': [0.618, 0.786, 0.886],
        'negative_fibs': [-0.272, -0.618],
        'negative_fibs_enabled': True
    }

    start = time.time()
    for _ in range(1000):
        calculate_fib_levels(100.0, 80.0, config)
    elapsed_ms = (time.time() - start) * 1000

    avg_ms = elapsed_ms / 1000
    assert avg_ms < 2.0, f"Fib calculation too slow: {avg_ms:.3f}ms (target <2ms)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
