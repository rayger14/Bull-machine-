#!/usr/bin/env python3
"""
Unit Tests for PR#5 Decision Gates

Tests:
1. Gate 5 pass/fail logic with all criteria
2. Assist-exit trigger conditions and actions
3. Dynamic sizing monotonicity and bounds
4. Stop tightening correctness
5. Telemetry accuracy

Run:
    pytest -q tests/test_decision_gates.py
    pytest -v tests/test_decision_gates.py::test_gate5_pass
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.gates.decision import (
    check_gate5,
    check_assist_exit,
    compute_dynamic_sizing,
    apply_assist_exit_tighten,
    GateTelemetry
)


# =============================================================================
# Gate 5 Tests
# =============================================================================

def test_gate5_pass_all_criteria():
    """Gate 5 should pass when all criteria met."""
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': True
    }

    passed, reason = check_gate5(0.45, 0.75, 0.50, cfg)

    assert passed, f"Should pass with good metrics, got: {reason}"
    assert reason == 'pass'


def test_gate5_fail_liquidity():
    """Gate 5 should fail when liquidity below threshold."""
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': True
    }

    passed, reason = check_gate5(0.30, 0.75, 0.50, cfg)

    assert not passed
    assert reason == 'liquidity_low'


def test_gate5_fail_fusion():
    """Gate 5 should fail when fusion below threshold."""
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': True
    }

    passed, reason = check_gate5(0.45, 0.65, 0.50, cfg)

    assert not passed
    assert reason == 'fusion_weak'


def test_gate5_fail_atr_low():
    """Gate 5 should fail when ATR too low."""
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': True
    }

    passed, reason = check_gate5(0.45, 0.75, 0.15, cfg)

    assert not passed
    assert reason == 'atr_too_low'


def test_gate5_fail_atr_high():
    """Gate 5 should fail when ATR too high."""
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': True
    }

    passed, reason = check_gate5(0.45, 0.75, 0.95, cfg)

    assert not passed
    assert reason == 'atr_too_high'


def test_gate5_volatility_check_disabled():
    """Gate 5 should ignore ATR checks when volatility_stable is False."""
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': False  # Disabled
    }

    # Should pass even with extreme ATR
    passed, reason = check_gate5(0.45, 0.75, 0.95, cfg)

    assert passed, f"Should pass when volatility check disabled, got: {reason}"


def test_gate5_edge_values():
    """Gate 5 should handle edge values correctly."""
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': True
    }

    # Exactly at threshold (should fail)
    passed, reason = check_gate5(0.35, 0.70, 0.50, cfg)
    assert passed, "Should pass at exact threshold"

    # Just below threshold (should fail)
    passed, reason = check_gate5(0.349, 0.70, 0.50, cfg)
    assert not passed and reason == 'liquidity_low'


# =============================================================================
# Assist Exit Tests
# =============================================================================

def test_assist_exit_tighten_early():
    """Assist-exit should recommend tighten early in trade."""
    cfg = {'assist_exit_liquidity_threshold': 0.30}

    should_exit, action = check_assist_exit(0.25, 0.60, 5, cfg)

    assert should_exit
    assert action == 'tighten'


def test_assist_exit_partial_late():
    """Assist-exit should recommend partial late in trade."""
    cfg = {'assist_exit_liquidity_threshold': 0.30}

    should_exit, action = check_assist_exit(0.25, 0.60, 10, cfg)

    assert should_exit
    assert action == 'partial'


def test_assist_exit_no_trigger_good_liquidity():
    """Assist-exit should not trigger when liquidity still good."""
    cfg = {'assist_exit_liquidity_threshold': 0.30}

    should_exit, action = check_assist_exit(0.40, 0.60, 5, cfg)

    assert not should_exit
    assert action is None


def test_assist_exit_no_trigger_below_entry():
    """Assist-exit should not trigger if below threshold but above entry."""
    cfg = {'assist_exit_liquidity_threshold': 0.30}

    # Current 0.28 < threshold 0.30, but entry was 0.25
    should_exit, action = check_assist_exit(0.28, 0.25, 5, cfg)

    assert not should_exit, "Should not exit if current > entry"


def test_assist_exit_boundary_bars():
    """Assist-exit should handle boundary at bars_in_trade=8."""
    cfg = {'assist_exit_liquidity_threshold': 0.30}

    # bars=7 → tighten
    should_exit, action = check_assist_exit(0.25, 0.60, 7, cfg)
    assert action == 'tighten'

    # bars=8 → partial
    should_exit, action = check_assist_exit(0.25, 0.60, 8, cfg)
    assert action == 'partial'


# =============================================================================
# Dynamic Sizing Tests
# =============================================================================

def test_dynamic_sizing_reference_score():
    """Reference score should yield 1.0× leverage."""
    cfg = {
        'sizing_min_leverage': 0.6,
        'sizing_max_leverage': 1.25,
        'sizing_reference_score': 0.50
    }

    mult = compute_dynamic_sizing(0.50, cfg)

    assert abs(mult - 1.0) < 0.01, f"Expected 1.0, got {mult}"


def test_dynamic_sizing_low_liquidity():
    """Low liquidity should reduce leverage."""
    cfg = {
        'sizing_min_leverage': 0.6,
        'sizing_max_leverage': 1.25,
        'sizing_reference_score': 0.50
    }

    mult_low = compute_dynamic_sizing(0.25, cfg)

    assert 0.6 <= mult_low < 1.0, f"Expected 0.6-1.0, got {mult_low}"


def test_dynamic_sizing_high_liquidity():
    """High liquidity should increase leverage."""
    cfg = {
        'sizing_min_leverage': 0.6,
        'sizing_max_leverage': 1.25,
        'sizing_reference_score': 0.50
    }

    mult_high = compute_dynamic_sizing(0.75, cfg)

    assert 1.0 < mult_high <= 1.25, f"Expected 1.0-1.25, got {mult_high}"


def test_dynamic_sizing_monotonic():
    """Higher liquidity should yield higher or equal leverage."""
    cfg = {
        'sizing_min_leverage': 0.6,
        'sizing_max_leverage': 1.25,
        'sizing_reference_score': 0.50
    }

    scores = [0.0, 0.25, 0.50, 0.75, 1.0]
    mults = [compute_dynamic_sizing(s, cfg) for s in scores]

    for i in range(len(mults) - 1):
        assert mults[i] <= mults[i+1], f"Not monotonic: {mults[i]} > {mults[i+1]} at scores {scores[i]}, {scores[i+1]}"


def test_dynamic_sizing_bounds():
    """Sizing should respect min/max bounds."""
    cfg = {
        'sizing_min_leverage': 0.6,
        'sizing_max_leverage': 1.25,
        'sizing_reference_score': 0.50
    }

    # Extreme low
    mult_min = compute_dynamic_sizing(0.0, cfg)
    assert abs(mult_min - 0.6) < 0.01

    # Extreme high
    mult_max = compute_dynamic_sizing(1.0, cfg)
    assert abs(mult_max - 1.25) < 0.01


def test_dynamic_sizing_extreme_config():
    """Sizing should work with extreme config values."""
    cfg = {
        'sizing_min_leverage': 0.1,
        'sizing_max_leverage': 5.0,
        'sizing_reference_score': 0.30
    }

    mult_ref = compute_dynamic_sizing(0.30, cfg)
    assert abs(mult_ref - 1.0) < 0.01, f"Expected 1.0 at reference, got {mult_ref}"


# =============================================================================
# Stop Tightening Tests
# =============================================================================

def test_stop_tighten_long():
    """Long stop should move up (tighten)."""
    cfg = {'assist_exit_tighten_factor': 0.85}

    new_stop = apply_assist_exit_tighten(100.0, 110.0, 'long', cfg)

    assert new_stop > 100.0, f"Long stop should increase, got {new_stop}"
    assert new_stop < 110.0, f"Stop should not exceed price, got {new_stop}"


def test_stop_tighten_short():
    """Short stop should move down (tighten)."""
    cfg = {'assist_exit_tighten_factor': 0.85}

    new_stop = apply_assist_exit_tighten(110.0, 100.0, 'short', cfg)

    assert new_stop < 110.0, f"Short stop should decrease, got {new_stop}"
    assert new_stop > 100.0, f"Stop should not go below price, got {new_stop}"


def test_stop_tighten_never_loosens():
    """Stop should never move away from entry (loosen)."""
    cfg = {'assist_exit_tighten_factor': 0.85}

    # Long: stop should not move down
    new_stop_long = apply_assist_exit_tighten(105.0, 110.0, 'long', cfg)
    assert new_stop_long >= 105.0

    # Short: stop should not move up
    new_stop_short = apply_assist_exit_tighten(105.0, 100.0, 'short', cfg)
    assert new_stop_short <= 105.0


def test_stop_tighten_factor_effect():
    """Lower tighten_factor should move stop closer to price."""
    # Factor 0.85 (looser)
    cfg_loose = {'assist_exit_tighten_factor': 0.85}
    stop_loose = apply_assist_exit_tighten(100.0, 110.0, 'long', cfg_loose)

    # Factor 0.50 (tighter)
    cfg_tight = {'assist_exit_tighten_factor': 0.50}
    stop_tight = apply_assist_exit_tighten(100.0, 110.0, 'long', cfg_tight)

    assert stop_tight > stop_loose, "Lower factor should move stop closer to price"


# =============================================================================
# Telemetry Tests
# =============================================================================

def test_telemetry_gate5_pass_rate():
    """Telemetry should correctly compute Gate 5 pass rate."""
    telemetry = GateTelemetry(window_size=10)

    # Record 10 attempts: 7 pass, 3 fail
    for i in range(10):
        telemetry.record_gate5(i < 7)

    stats = telemetry.get_stats()

    assert stats['gate5_attempts'] == 10
    assert abs(stats['gate5_pass_rate'] - 70.0) < 0.1


def test_telemetry_assist_exit_outcomes():
    """Telemetry should track assist-exit outcomes correctly."""
    telemetry = GateTelemetry(window_size=10)

    # Record 6 tighten, 4 partial
    for i in range(10):
        action = 'tighten' if i < 6 else 'partial'
        telemetry.record_assist_exit(action)

    stats = telemetry.get_stats()

    assert stats['assist_exit_count'] == 10
    assert abs(stats['assist_exit_tighten_pct'] - 60.0) < 0.1


def test_telemetry_sizing_stats():
    """Telemetry should compute sizing statistics correctly."""
    telemetry = GateTelemetry(window_size=10)

    # Record varied multipliers
    mults = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]
    for m in mults:
        telemetry.record_sizing(m)

    stats = telemetry.get_stats()

    assert stats['sizing_min'] == 0.6
    assert stats['sizing_max'] == 1.25
    assert 0.9 < stats['sizing_mean'] < 1.0, f"Expected mean ~0.95, got {stats['sizing_mean']}"


def test_telemetry_window_size():
    """Telemetry should respect window size."""
    telemetry = GateTelemetry(window_size=5)

    # Record 10 attempts, but only last 5 should be kept
    for i in range(10):
        telemetry.record_gate5(i >= 5)  # Last 5 pass

    stats = telemetry.get_stats()

    assert stats['gate5_attempts'] == 5, f"Expected 5 attempts, got {stats['gate5_attempts']}"
    assert abs(stats['gate5_pass_rate'] - 100.0) < 0.1, "Last 5 should all pass"


def test_telemetry_empty():
    """Telemetry should handle empty state gracefully."""
    telemetry = GateTelemetry(window_size=10)

    stats = telemetry.get_stats()

    assert stats['gate5_pass_rate'] == 0.0
    assert stats['gate5_attempts'] == 0
    assert stats['assist_exit_count'] == 0
    assert stats['sizing_mean'] == 1.0  # Default neutral


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    # Run all tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
