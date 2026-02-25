#!/usr/bin/env python3
"""
PR#5: Decision Gates for Adaptive Re-entry and Exits

Provides:
1. Gate 5: Re-entry filter based on liquidity + fusion + volatility stability
2. Assist Exits: Gradual exit tightening when liquidity deteriorates
3. Dynamic Position Sizing: Leverage modulation (0.6-1.25×) based on liquidity score

Architecture:
- Pure functions (no state except telemetry accumulator)
- All gates behind config flags (default OFF)
- Telemetry tracking for pass/fail rates and outcomes

Author: PR#5 - Adaptive Decision Gates
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from collections import deque


class GateTelemetry:
    """
    Telemetry tracker for decision gates.

    Tracks:
    - Gate 5 pass/fail counts and rates
    - Assist-exit trigger counts and outcomes
    - Dynamic sizing distribution
    """

    def __init__(self, window_size: int = 500):
        """
        Initialize telemetry tracker.

        Args:
            window_size: Number of recent events to track
        """
        self.window_size = window_size

        # Gate 5 tracking
        self.gate5_attempts: deque = deque(maxlen=window_size)
        self.gate5_passes: deque = deque(maxlen=window_size)

        # Assist-exit tracking
        self.assist_exit_triggers: deque = deque(maxlen=window_size)
        self.assist_exit_outcomes: deque = deque(maxlen=window_size)  # 'tighten' or 'partial'

        # Dynamic sizing tracking
        self.sizing_multipliers: deque = deque(maxlen=window_size)

    def record_gate5(self, passed: bool) -> None:
        """Record a Gate 5 attempt and outcome."""
        self.gate5_attempts.append(1)
        self.gate5_passes.append(1 if passed else 0)

    def record_assist_exit(self, outcome: str) -> None:
        """Record an assist-exit trigger and outcome."""
        self.assist_exit_triggers.append(1)
        self.assist_exit_outcomes.append(outcome)

    def record_sizing(self, multiplier: float) -> None:
        """Record a dynamic sizing multiplier."""
        self.sizing_multipliers.append(multiplier)

    def get_stats(self) -> Dict[str, Any]:
        """
        Compute telemetry statistics.

        Returns:
            Dict with:
                - gate5_pass_rate: Percentage of Gate 5 passes
                - gate5_attempts: Total attempts in window
                - assist_exit_count: Total assist-exit triggers
                - assist_exit_tighten_pct: % of tighten outcomes
                - sizing_mean: Average sizing multiplier
                - sizing_min: Minimum sizing multiplier
                - sizing_max: Maximum sizing multiplier
        """
        gate5_pass_rate = 0.0
        if self.gate5_attempts:
            gate5_pass_rate = (sum(self.gate5_passes) / len(self.gate5_attempts)) * 100.0

        assist_exit_tighten_pct = 0.0
        if self.assist_exit_outcomes:
            tighten_count = sum(1 for o in self.assist_exit_outcomes if o == 'tighten')
            assist_exit_tighten_pct = (tighten_count / len(self.assist_exit_outcomes)) * 100.0

        sizing_mean = sum(self.sizing_multipliers) / len(self.sizing_multipliers) if self.sizing_multipliers else 1.0
        sizing_min = min(self.sizing_multipliers) if self.sizing_multipliers else 1.0
        sizing_max = max(self.sizing_multipliers) if self.sizing_multipliers else 1.0

        return {
            'gate5_pass_rate': gate5_pass_rate,
            'gate5_attempts': len(self.gate5_attempts),
            'assist_exit_count': len(self.assist_exit_triggers),
            'assist_exit_tighten_pct': assist_exit_tighten_pct,
            'sizing_mean': sizing_mean,
            'sizing_min': sizing_min,
            'sizing_max': sizing_max
        }


def check_gate5(
    liquidity_score: float,
    fusion_score: float,
    atr_percentile: float,
    cfg: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Gate 5: Re-entry quality filter.

    Checks:
    1. Liquidity score ≥ threshold (default 0.35)
    2. Fusion score ≥ threshold (default 0.70)
    3. Volatility stable (ATR between min/max percentiles)

    Args:
        liquidity_score: Runtime liquidity score (0-1)
        fusion_score: Fusion confidence score (0-1)
        atr_percentile: Current ATR percentile (0-1)
        cfg: Configuration dict with gate5_* parameters

    Returns:
        Tuple of (passed: bool, reason: str)

    Example:
        >>> cfg = {'gate5_liquidity_threshold': 0.35, 'gate5_fusion_threshold': 0.70,
        ...        'gate5_atr_percentile_min': 0.25, 'gate5_atr_percentile_max': 0.85}
        >>> check_gate5(0.40, 0.75, 0.50, cfg)
        (True, 'pass')
        >>> check_gate5(0.30, 0.75, 0.50, cfg)
        (False, 'liquidity_low')
    """
    liq_thresh = cfg.get('gate5_liquidity_threshold', 0.35)
    fus_thresh = cfg.get('gate5_fusion_threshold', 0.70)
    atr_min = cfg.get('gate5_atr_percentile_min', 0.25)
    atr_max = cfg.get('gate5_atr_percentile_max', 0.85)

    # Check 1: Liquidity
    if liquidity_score < liq_thresh:
        return (False, 'liquidity_low')

    # Check 2: Fusion
    if fusion_score < fus_thresh:
        return (False, 'fusion_weak')

    # Check 3: Volatility stability (optional)
    if cfg.get('gate5_volatility_stable', True):
        if atr_percentile < atr_min:
            return (False, 'atr_too_low')
        if atr_percentile > atr_max:
            return (False, 'atr_too_high')

    return (True, 'pass')


def check_assist_exit(
    current_liquidity: float,
    entry_liquidity: float,
    bars_in_trade: int,
    cfg: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Assist Exit: Check if liquidity deterioration warrants exit action.

    Logic:
    - If current liquidity drops below threshold AND below entry liquidity:
      - If bars_in_trade < 8: recommend 'tighten' (move stop closer)
      - If bars_in_trade >= 8: recommend 'partial' (take partial profit)

    Args:
        current_liquidity: Current bar liquidity score (0-1)
        entry_liquidity: Liquidity score at entry (0-1)
        bars_in_trade: Number of bars since entry
        cfg: Configuration dict with assist_exit_* parameters

    Returns:
        Tuple of (should_exit: bool, action: Optional[str])
        action is 'tighten' or 'partial' if should_exit=True

    Example:
        >>> cfg = {'assist_exit_liquidity_threshold': 0.30}
        >>> check_assist_exit(0.25, 0.60, 5, cfg)
        (True, 'tighten')
        >>> check_assist_exit(0.25, 0.60, 10, cfg)
        (True, 'partial')
        >>> check_assist_exit(0.40, 0.60, 5, cfg)
        (False, None)
    """
    threshold = cfg.get('assist_exit_liquidity_threshold', 0.30)

    # Check if liquidity deteriorated significantly
    if current_liquidity < threshold and current_liquidity < entry_liquidity:
        # Early in trade: tighten stops
        if bars_in_trade < 8:
            return (True, 'tighten')
        # Later in trade: take partial profits
        else:
            return (True, 'partial')

    return (False, None)


def compute_dynamic_sizing(
    liquidity_score: float,
    cfg: Dict[str, Any]
) -> float:
    """
    Compute dynamic position sizing multiplier based on liquidity score.

    Maps liquidity score to leverage multiplier:
    - Low liquidity (< reference): reduce leverage (0.6-1.0×)
    - High liquidity (> reference): increase leverage (1.0-1.25×)
    - Reference score (default 0.50): 1.0× leverage

    Formula:
        multiplier = min_lev + (max_lev - min_lev) * normalized_score
        where normalized_score maps [0, 1] to [0, 1] with reference at 0.5

    Args:
        liquidity_score: Runtime liquidity score (0-1)
        cfg: Configuration dict with sizing_* parameters

    Returns:
        Leverage multiplier in [sizing_min_leverage, sizing_max_leverage]

    Example:
        >>> cfg = {'sizing_min_leverage': 0.6, 'sizing_max_leverage': 1.25,
        ...        'sizing_reference_score': 0.50}
        >>> compute_dynamic_sizing(0.50, cfg)  # Reference score
        1.0
        >>> compute_dynamic_sizing(0.75, cfg)  # High liquidity
        1.125
        >>> compute_dynamic_sizing(0.25, cfg)  # Low liquidity
        0.8
    """
    min_lev = cfg.get('sizing_min_leverage', 0.6)
    max_lev = cfg.get('sizing_max_leverage', 1.25)
    ref_score = cfg.get('sizing_reference_score', 0.50)

    # Clamp liquidity score
    liq = max(0.0, min(1.0, liquidity_score))

    # Linear mapping: reference score → 1.0×
    # Below reference: scale linearly from min_lev to 1.0
    # Above reference: scale linearly from 1.0 to max_lev

    if liq <= ref_score:
        # Map [0, ref] → [min_lev, 1.0]
        t = liq / ref_score if ref_score > 0 else 0.0
        multiplier = min_lev + (1.0 - min_lev) * t
    else:
        # Map [ref, 1.0] → [1.0, max_lev]
        t = (liq - ref_score) / (1.0 - ref_score) if (1.0 - ref_score) > 0 else 0.0
        multiplier = 1.0 + (max_lev - 1.0) * t

    # Ensure bounds
    return max(min_lev, min(max_lev, multiplier))


def apply_assist_exit_tighten(
    current_stop: float,
    current_price: float,
    side: str,
    cfg: Dict[str, Any]
) -> float:
    """
    Apply stop-loss tightening for assist-exit.

    Moves stop closer to current price by tighten_factor (default 0.85).

    Args:
        current_stop: Current stop-loss price
        current_price: Current market price
        side: 'long' or 'short'
        cfg: Configuration dict with assist_exit_tighten_factor

    Returns:
        New tightened stop-loss price

    Example:
        >>> cfg = {'assist_exit_tighten_factor': 0.85}
        >>> # Long trade: stop at 100, price at 110
        >>> apply_assist_exit_tighten(100.0, 110.0, 'long', cfg)
        101.5  # 85% closer to current price
    """
    tighten_factor = cfg.get('assist_exit_tighten_factor', 0.85)

    if side == 'long':
        # Move stop up (tighten)
        distance = current_price - current_stop
        new_stop = current_stop + distance * (1.0 - tighten_factor)
        return max(current_stop, new_stop)  # Never loosen
    else:  # short
        # Move stop down (tighten)
        distance = current_stop - current_price
        new_stop = current_stop - distance * (1.0 - tighten_factor)
        return min(current_stop, new_stop)  # Never loosen


# Example usage / testing
if __name__ == '__main__':
    print("PR#5 Decision Gates - Smoke Test\n")

    # Test configuration
    cfg = {
        'gate5_liquidity_threshold': 0.35,
        'gate5_fusion_threshold': 0.70,
        'gate5_atr_percentile_min': 0.25,
        'gate5_atr_percentile_max': 0.85,
        'gate5_volatility_stable': True,
        'assist_exit_liquidity_threshold': 0.30,
        'assist_exit_tighten_factor': 0.85,
        'sizing_min_leverage': 0.6,
        'sizing_max_leverage': 1.25,
        'sizing_reference_score': 0.50
    }

    # Test Gate 5
    print("=== Gate 5 Tests ===")

    # Test 1: Pass all checks
    passed, reason = check_gate5(0.45, 0.75, 0.50, cfg)
    print(f"✓ Test 1 (good setup): {passed} ({reason})")
    assert passed, "Should pass with good liquidity, fusion, and ATR"

    # Test 2: Fail liquidity
    passed, reason = check_gate5(0.30, 0.75, 0.50, cfg)
    print(f"✓ Test 2 (low liquidity): {passed} ({reason})")
    assert not passed and reason == 'liquidity_low'

    # Test 3: Fail fusion
    passed, reason = check_gate5(0.45, 0.65, 0.50, cfg)
    print(f"✓ Test 3 (weak fusion): {passed} ({reason})")
    assert not passed and reason == 'fusion_weak'

    # Test 4: Fail ATR (too low)
    passed, reason = check_gate5(0.45, 0.75, 0.15, cfg)
    print(f"✓ Test 4 (ATR too low): {passed} ({reason})")
    assert not passed and reason == 'atr_too_low'

    # Test 5: Fail ATR (too high)
    passed, reason = check_gate5(0.45, 0.75, 0.95, cfg)
    print(f"✓ Test 5 (ATR too high): {passed} ({reason})")
    assert not passed and reason == 'atr_too_high'

    print()

    # Test Assist Exit
    print("=== Assist Exit Tests ===")

    # Test 6: Trigger tighten (early in trade)
    should_exit, action = check_assist_exit(0.25, 0.60, 5, cfg)
    print(f"✓ Test 6 (early exit): {should_exit} ({action})")
    assert should_exit and action == 'tighten'

    # Test 7: Trigger partial (late in trade)
    should_exit, action = check_assist_exit(0.25, 0.60, 10, cfg)
    print(f"✓ Test 7 (late exit): {should_exit} ({action})")
    assert should_exit and action == 'partial'

    # Test 8: No exit (liquidity still good)
    should_exit, action = check_assist_exit(0.40, 0.60, 5, cfg)
    print(f"✓ Test 8 (no exit): {should_exit} ({action})")
    assert not should_exit

    print()

    # Test Dynamic Sizing
    print("=== Dynamic Sizing Tests ===")

    # Test 9: Low liquidity → reduced leverage
    mult_low = compute_dynamic_sizing(0.25, cfg)
    print(f"✓ Test 9 (low liquidity 0.25): {mult_low:.3f}× leverage")
    assert 0.6 <= mult_low < 1.0, f"Expected 0.6-1.0, got {mult_low}"

    # Test 10: Reference liquidity → 1.0× leverage
    mult_ref = compute_dynamic_sizing(0.50, cfg)
    print(f"✓ Test 10 (reference 0.50): {mult_ref:.3f}× leverage")
    assert abs(mult_ref - 1.0) < 0.01, f"Expected ~1.0, got {mult_ref}"

    # Test 11: High liquidity → increased leverage
    mult_high = compute_dynamic_sizing(0.75, cfg)
    print(f"✓ Test 11 (high liquidity 0.75): {mult_high:.3f}× leverage")
    assert 1.0 < mult_high <= 1.25, f"Expected 1.0-1.25, got {mult_high}"

    # Test 12: Extreme low → min leverage
    mult_min = compute_dynamic_sizing(0.0, cfg)
    print(f"✓ Test 12 (extreme low 0.0): {mult_min:.3f}× leverage")
    assert abs(mult_min - 0.6) < 0.01, f"Expected 0.6, got {mult_min}"

    # Test 13: Extreme high → max leverage
    mult_max = compute_dynamic_sizing(1.0, cfg)
    print(f"✓ Test 13 (extreme high 1.0): {mult_max:.3f}× leverage")
    assert abs(mult_max - 1.25) < 0.01, f"Expected 1.25, got {mult_max}"

    print()

    # Test Stop Tightening
    print("=== Stop Tightening Tests ===")

    # Test 14: Long tighten
    new_stop = apply_assist_exit_tighten(100.0, 110.0, 'long', cfg)
    print(f"✓ Test 14 (long tighten): stop 100 → {new_stop:.2f} (price 110)")
    assert new_stop > 100.0, "Stop should move up"

    # Test 15: Short tighten
    new_stop = apply_assist_exit_tighten(110.0, 100.0, 'short', cfg)
    print(f"✓ Test 15 (short tighten): stop 110 → {new_stop:.2f} (price 100)")
    assert new_stop < 110.0, "Stop should move down"

    print()

    # Test Telemetry
    print("=== Telemetry Tests ===")

    telemetry = GateTelemetry(window_size=10)

    # Record some events
    for i in range(15):
        telemetry.record_gate5(i % 3 != 0)  # ~66% pass rate
        if i % 4 == 0:
            telemetry.record_assist_exit('tighten' if i % 8 == 0 else 'partial')
        telemetry.record_sizing(0.8 + 0.1 * (i % 5))

    stats = telemetry.get_stats()
    print(f"✓ Gate 5 pass rate: {stats['gate5_pass_rate']:.1f}%")
    print(f"✓ Gate 5 attempts: {stats['gate5_attempts']}")
    print(f"✓ Assist-exit count: {stats['assist_exit_count']}")
    print(f"✓ Assist-exit tighten %: {stats['assist_exit_tighten_pct']:.1f}%")
    print(f"✓ Sizing mean: {stats['sizing_mean']:.3f}×")
    print(f"✓ Sizing range: [{stats['sizing_min']:.3f}, {stats['sizing_max']:.3f}]")

    print("\n✅ All smoke tests passed!")
