#!/usr/bin/env python3
"""
Unit Tests for Runtime Liquidity Scorer

Tests:
1. Monotonicity: Higher BOMS strength should not reduce score
2. Bounded output: All scores in [0, 1]
3. Safety: No exceptions on missing fields
4. Distribution: Realistic scores with expected quantiles

Run:
    pytest -q tests/test_liquidity_score.py
    pytest -v tests/test_liquidity_score.py::test_monotonic_strength
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.liquidity.score import (
    compute_liquidity_score,
    compute_liquidity_telemetry,
    _clip01,
    _sigmoid01
)


# =============================================================================
# Helper Functions Tests
# =============================================================================

def test_clip01_basic():
    """Test _clip01 clips values to [0, 1]."""
    assert _clip01(0.5) == 0.5
    assert _clip01(-0.5) == 0.0
    assert _clip01(1.5) == 1.0
    assert _clip01(None) == 0.0
    assert _clip01(float('nan')) == 0.0


def test_sigmoid01_basic():
    """Test _sigmoid01 maps to (0, 1)."""
    # Neutral point
    assert abs(_sigmoid01(0.0) - 0.5) < 0.01

    # Positive z increases score
    assert _sigmoid01(2.0) > 0.8
    assert _sigmoid01(-2.0) < 0.2

    # Extreme values don't crash
    assert 0.0 <= _sigmoid01(100.0) <= 1.0
    assert 0.0 <= _sigmoid01(-100.0) <= 1.0


# =============================================================================
# Monotonicity Tests
# =============================================================================

def test_monotonic_strength():
    """Higher tf1d_boms_strength should increase or maintain score."""
    base_ctx = {
        'close': 100.0,
        'high': 100.5,
        'low': 99.5,
        'atr': 1.0,
        'volume_zscore': 0.0,
        'tf4h_boms_displacement': 0.5,
        'fvg_present': True,
        'tf1d_boms_strength': 0.2
    }

    score1 = compute_liquidity_score(base_ctx, 'long')

    # Increase BOMS strength
    base_ctx['tf1d_boms_strength'] = 0.6
    score2 = compute_liquidity_score(base_ctx, 'long')

    assert score2 >= score1, f"Higher BOMS strength reduced score: {score1} -> {score2}"


def test_monotonic_displacement():
    """Higher displacement should increase or maintain score."""
    base_ctx = {
        'close': 100.0,
        'high': 100.5,
        'low': 99.5,
        'atr': 1.0,
        'tf1d_boms_strength': 0.5,
        'tf4h_boms_displacement': 0.2
    }

    score1 = compute_liquidity_score(base_ctx, 'long')

    # Increase displacement
    base_ctx['tf4h_boms_displacement'] = 0.8
    score2 = compute_liquidity_score(base_ctx, 'long')

    assert score2 >= score1, f"Higher displacement reduced score: {score1} -> {score2}"


def test_monotonic_fvg_quality():
    """Higher FVG quality should increase or maintain score."""
    base_ctx = {
        'close': 100.0,
        'high': 100.5,
        'low': 99.5,
        'tf1d_boms_strength': 0.5,
        'fvg_quality': 0.3
    }

    score1 = compute_liquidity_score(base_ctx, 'long')

    # Increase FVG quality
    base_ctx['fvg_quality'] = 0.8
    score2 = compute_liquidity_score(base_ctx, 'long')

    assert score2 >= score1, f"Higher FVG quality reduced score: {score1} -> {score2}"


# =============================================================================
# Bounded Output Tests
# =============================================================================

def test_score_bounded():
    """All scores must be in [0, 1] range."""
    test_cases = [
        # Strong setup
        {
            'close': 60000.0, 'high': 60500.0, 'low': 59800.0,
            'tf1d_boms_strength': 0.9, 'tf4h_boms_displacement': 1500.0,
            'fvg_quality': 0.95, 'volume_zscore': 2.0, 'atr': 800.0,
            'tf4h_fusion_score': 0.85
        },
        # Weak setup
        {
            'close': 60000.0, 'high': 60050.0, 'low': 59950.0,
            'tf1d_boms_strength': 0.05, 'volume_zscore': -1.5
        },
        # Minimal context
        {
            'close': 60000.0, 'high': 60100.0, 'low': 59900.0
        },
        # Extreme values
        {
            'close': 100000.0, 'high': 105000.0, 'low': 95000.0,
            'tf1d_boms_strength': 1.0, 'tf4h_boms_displacement': 10000.0,
            'volume_zscore': 5.0, 'atr': 2000.0
        }
    ]

    for ctx in test_cases:
        score_long = compute_liquidity_score(ctx, 'long')
        score_short = compute_liquidity_score(ctx, 'short')

        assert 0.0 <= score_long <= 1.0, f"Long score out of bounds: {score_long}"
        assert 0.0 <= score_short <= 1.0, f"Short score out of bounds: {score_short}"


def test_no_nan_inf():
    """Scores must not be NaN or Inf."""
    import math

    ctx = {
        'close': 60000.0,
        'high': 60500.0,
        'low': 59800.0,
        'tf1d_boms_strength': 0.5
    }

    score = compute_liquidity_score(ctx, 'long')

    assert not math.isnan(score), "Score is NaN"
    assert not math.isinf(score), "Score is Inf"


# =============================================================================
# Safety / Robustness Tests
# =============================================================================

def test_missing_fields_no_crash():
    """Missing fields should use fallbacks without raising exceptions."""
    # Empty context (only OHLC)
    ctx_minimal = {
        'close': 100.0,
        'high': 101.0,
        'low': 99.0
    }

    try:
        score = compute_liquidity_score(ctx_minimal, 'long')
        assert 0.0 <= score <= 1.0
    except Exception as e:
        pytest.fail(f"Minimal context raised exception: {e}")

    # Partial context
    ctx_partial = {
        'close': 100.0,
        'high': 101.0,
        'low': 99.0,
        'tf1d_boms_strength': 0.6,
        # Missing: displacement, FVG, volume, etc.
    }

    try:
        score = compute_liquidity_score(ctx_partial, 'long')
        assert 0.0 <= score <= 1.0
    except Exception as e:
        pytest.fail(f"Partial context raised exception: {e}")


def test_none_values_safe():
    """None values in context should be handled gracefully."""
    ctx = {
        'close': 100.0,
        'high': 101.0,
        'low': 99.0,
        'tf1d_boms_strength': None,  # Explicit None
        'tf4h_boms_displacement': None,
        'fvg_quality': None,
        'volume_zscore': None,
        'atr': None
    }

    try:
        score = compute_liquidity_score(ctx, 'long')
        assert 0.0 <= score <= 1.0
    except Exception as e:
        pytest.fail(f"None values raised exception: {e}")


def test_zero_denominator_safe():
    """Zero values that could cause division by zero should be safe."""
    ctx = {
        'close': 0.0,  # Edge case
        'high': 0.0,
        'low': 0.0,
        'atr': 0.0,  # Could cause division by zero
        'tf4h_boms_displacement': 100.0
    }

    try:
        score = compute_liquidity_score(ctx, 'long')
        assert 0.0 <= score <= 1.0
    except (ZeroDivisionError, ValueError) as e:
        pytest.fail(f"Zero denominators raised exception: {e}")


# =============================================================================
# Distribution / Signal Quality Tests
# =============================================================================

def test_realistic_distribution():
    """Score distribution should be reasonable across varied setups."""
    import random
    random.seed(42)

    scores = []
    for _ in range(200):
        ctx = {
            'close': random.uniform(50000, 70000),
            'high': random.uniform(50500, 70500),
            'low': random.uniform(49500, 69500),
            'tf1d_boms_strength': random.uniform(0.0, 1.0),
            'tf4h_boms_displacement': random.uniform(0.0, 2000.0),
            'fvg_quality': random.uniform(0.0, 1.0) if random.random() > 0.3 else 0.0,
            'volume_zscore': random.uniform(-2.0, 2.0),
            'atr': random.uniform(500.0, 1500.0),
            'tf4h_fusion_score': random.uniform(0.0, 1.0) if random.random() > 0.5 else 0.0
        }

        score = compute_liquidity_score(ctx, 'long')
        scores.append(score)

    # Compute statistics
    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    median = sorted_scores[n // 2]
    p25 = sorted_scores[n // 4]
    p75 = sorted_scores[3 * n // 4]
    p90 = sorted_scores[int(n * 0.9)]

    # Sanity checks (not strict targets yet, pre-calibration)
    assert 0.30 <= median <= 0.70, f"Median {median:.3f} outside reasonable range"
    assert p75 > median, f"p75 {p75:.3f} not greater than median {median:.3f}"
    assert p90 > p75, f"p90 {p90:.3f} not greater than p75 {p75:.3f}"

    # Non-zero rate should be high (most setups have some liquidity)
    nonzero_count = sum(1 for s in scores if s > 0.0)
    nonzero_pct = (nonzero_count / len(scores)) * 100
    assert nonzero_pct > 80.0, f"Non-zero rate {nonzero_pct:.1f}% too low"


def test_strong_vs_weak_separation():
    """Strong setups should score significantly higher than weak setups."""
    # Strong setup
    ctx_strong = {
        'close': 60000.0,
        'high': 60500.0,
        'low': 59800.0,
        'tf1d_boms_strength': 0.85,
        'tf4h_boms_displacement': 1200.0,
        'fvg_quality': 0.9,
        'fresh_bos_flag': True,
        'volume_zscore': 1.5,
        'atr': 800.0,
        'tf4h_fusion_score': 0.75,
        'range_eq': 59900.0,  # In discount for long
        'tod_boost': 0.8
    }

    # Weak setup
    ctx_weak = {
        'close': 60000.0,
        'high': 60100.0,
        'low': 59900.0,
        'tf1d_boms_strength': 0.1,
        'tf4h_boms_displacement': 50.0,
        'fvg_present': False,
        'volume_zscore': -1.0,
        'atr': 800.0,
        'tf4h_fusion_score': 0.1,
        'range_eq': 59950.0,
        'tod_boost': 0.3
    }

    score_strong = compute_liquidity_score(ctx_strong, 'long')
    score_weak = compute_liquidity_score(ctx_weak, 'long')

    # Strong should be at least 0.3 points higher
    assert score_strong - score_weak >= 0.30, (
        f"Insufficient separation: strong={score_strong:.3f}, weak={score_weak:.3f}"
    )


# =============================================================================
# Telemetry Tests
# =============================================================================

def test_telemetry_basic():
    """Telemetry should compute correct statistics."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = compute_liquidity_telemetry(scores)

    assert abs(stats['p50'] - 0.5) < 0.05, f"Median {stats['p50']:.3f} != 0.5"
    assert stats['p25'] < stats['p50'] < stats['p75'], "Percentiles not ordered"
    assert stats['nonzero_pct'] == 100.0, "All scores non-zero"
    assert abs(stats['mean'] - 0.5) < 0.05, f"Mean {stats['mean']:.3f} != 0.5"


def test_telemetry_empty():
    """Telemetry should handle empty score list."""
    stats = compute_liquidity_telemetry([])

    assert stats['p50'] == 0.0
    assert stats['nonzero_pct'] == 0.0
    assert stats['mean'] == 0.0


def test_telemetry_window():
    """Telemetry should respect window size."""
    scores = list(range(1000))  # 0 to 999
    stats = compute_liquidity_telemetry(scores, window_size=100)

    # Should only use last 100 scores (900-999)
    assert stats['mean'] > 900, f"Mean {stats['mean']:.1f} not from last 100 scores"


# =============================================================================
# Configuration Override Tests
# =============================================================================

def test_custom_weights():
    """Custom weight configuration should be respected."""
    ctx = {
        'close': 60000.0,
        'high': 60500.0,
        'low': 59800.0,
        'tf1d_boms_strength': 0.8,
        'tf4h_boms_displacement': 1000.0,
        'fvg_quality': 0.7,
        'volume_zscore': 1.0,
        'atr': 800.0
    }

    # Default weights
    score_default = compute_liquidity_score(ctx, 'long')

    # Custom weights (boost strength pillar)
    cfg_custom = {
        'wS': 0.50,  # Increased from 0.35
        'wC': 0.25,  # Decreased from 0.30
        'wL': 0.15,  # Decreased from 0.20
        'wP': 0.10   # Decreased from 0.15
    }
    score_custom = compute_liquidity_score(ctx, 'long', cfg=cfg_custom)

    # Should be different (higher due to strong BOMS strength)
    assert score_custom != score_default
    assert score_custom > score_default, "Custom weights didn't boost strength pillar"


def test_custom_caps():
    """Custom cap configuration should be respected (verified via non-crash)."""
    ctx = {
        'close': 60000.0,
        'high': 60500.0,
        'low': 59800.0,
        'tf4h_boms_displacement': 800.0,  # Moderate displacement
        'atr': 1000.0,
        'tf1d_boms_strength': 0.5  # Baseline strength
    }

    # Test that different caps don't crash and produce valid scores
    cfg_low_cap = {'disp_cap': 400.0}
    score_low = compute_liquidity_score(ctx, 'long', cfg=cfg_low_cap)

    cfg_high_cap = {'disp_cap': 2000.0}
    score_high = compute_liquidity_score(ctx, 'long', cfg=cfg_high_cap)

    # Both should be valid scores (bounded, no NaN)
    assert 0.0 <= score_low <= 1.0
    assert 0.0 <= score_high <= 1.0


# =============================================================================
# Side (Long/Short) Tests
# =============================================================================

def test_side_affects_positioning():
    """Long vs short side should affect discount/premium positioning."""
    ctx = {
        'close': 60000.0,
        'high': 60500.0,
        'low': 59800.0,
        'range_eq': 60500.0,  # Close is below EQ
        'tf1d_boms_strength': 0.5,
        'volume_zscore': 0.5
    }

    score_long = compute_liquidity_score(ctx, 'long')
    score_short = compute_liquidity_score(ctx, 'short')

    # Long should benefit from being below EQ (in discount)
    # Short should not benefit (not in premium)
    # Scores won't be equal due to positioning pillar
    assert score_long != score_short, "Long/short scores identical despite positioning"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    # Run all tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
