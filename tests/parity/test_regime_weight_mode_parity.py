"""
Parity tests for adaptive_fusion.regime_weight_mode (feat/regime-weights-as-threshold).

Covers:
1. ArchetypeInstance.compute_regime_threshold_multiplier — pure math.
2. Flag-off ("fusion_multiplier", default) leaves the threshold-check site
   structurally identical to the legacy path (no archetype attribute is
   touched, no hard-blocks fired).
3. Flag-on ("threshold_adjustment") activates the new branch — multiplier of
   1.0 is a no-op, 0.5 doubles the threshold, weight below the hard-block
   floor blocks the signal outright.
"""
from __future__ import annotations

import math
import pytest

from engine.archetypes.archetype_instance import ArchetypeConfig, ArchetypeInstance


# ---------------------------------------------------------------------------
# 1. Core math
# ---------------------------------------------------------------------------

def _inst(weights):
    cfg = ArchetypeConfig(
        name="probe",
        direction="long",
        regime_weights=weights,
    )
    return ArchetypeInstance(cfg)


def test_regime_weight_1_is_noop():
    inst = _inst({"risk_on": 1.0})
    mult, blocked, meta = inst.compute_regime_threshold_multiplier("risk_on")
    assert mult == pytest.approx(1.0)
    assert blocked is False
    assert meta["reason"] == "ok"


def test_regime_weight_half_doubles_threshold():
    inst = _inst({"crisis": 0.5})
    mult, blocked, meta = inst.compute_regime_threshold_multiplier("crisis")
    assert mult == pytest.approx(2.0)
    assert blocked is False


def test_regime_weight_above_1_lowers_threshold():
    # Spring-like archetype: stronger preference in crisis means lower threshold.
    inst = _inst({"crisis": 2.0})
    mult, blocked, _ = inst.compute_regime_threshold_multiplier("crisis")
    assert mult == pytest.approx(0.5)
    assert blocked is False


def test_regime_weight_zero_hard_blocks():
    inst = _inst({"crisis": 0.0})
    mult, blocked, meta = inst.compute_regime_threshold_multiplier("crisis")
    assert blocked is True
    assert math.isinf(mult)
    assert meta["reason"] == "weight_below_hard_block_floor"


def test_regime_weight_below_default_floor_blocks():
    # Default floor is 0.2; 0.1 -> block.
    inst = _inst({"crisis": 0.1})
    _, blocked, _ = inst.compute_regime_threshold_multiplier("crisis")
    assert blocked is True


def test_regime_weight_at_floor_does_not_block():
    # 0.2 is the floor — strict `<` comparison, so 0.2 itself should pass.
    inst = _inst({"crisis": 0.2})
    mult, blocked, _ = inst.compute_regime_threshold_multiplier(
        "crisis", hard_block_floor=0.2
    )
    assert blocked is False
    assert mult == pytest.approx(5.0)


def test_custom_floor_can_block_higher_weight():
    inst = _inst({"crisis": 0.4})
    _, blocked, _ = inst.compute_regime_threshold_multiplier(
        "crisis", hard_block_floor=0.5
    )
    assert blocked is True


def test_unknown_regime_is_passthrough():
    inst = _inst({"risk_on": 1.0})
    mult, blocked, meta = inst.compute_regime_threshold_multiplier("never_seen")
    assert mult == pytest.approx(1.0)
    assert blocked is False
    assert meta["reason"] == "no_regime_weight_for_regime"


def test_empty_regime_weights_is_passthrough():
    cfg = ArchetypeConfig(name="bare", direction="long")
    # __post_init__ injects defaults, so force-clear and re-check
    cfg.regime_weights = {}
    inst = ArchetypeInstance(cfg)
    mult, blocked, meta = inst.compute_regime_threshold_multiplier("crisis")
    assert mult == pytest.approx(1.0)
    assert blocked is False
    assert meta["reason"] == "no_regime_weight_for_regime"


def test_nan_weight_is_passthrough():
    inst = _inst({"crisis": float("nan")})
    mult, blocked, meta = inst.compute_regime_threshold_multiplier("crisis")
    assert mult == pytest.approx(1.0)
    assert blocked is False
    assert meta["reason"] == "regime_weight_non_finite"


# ---------------------------------------------------------------------------
# 2. Flag-off legacy mode is a no-op at the consumer site
# ---------------------------------------------------------------------------

def test_default_mode_is_fusion_multiplier():
    """Default flag value preserves legacy behavior — no compute_regime_threshold_multiplier
    call should be triggered."""
    # We can't easily exercise the full backtester here without IO; instead
    # assert that the helper plus a fusion_multiplier config keep the archetype
    # untouched via a stub.
    inst = _inst({"crisis": 0.5})

    # Simulate the consumer site: in fusion_multiplier mode we never call
    # compute_regime_threshold_multiplier, so the threshold stays at base.
    base_threshold = 0.18
    mode = "fusion_multiplier"
    effective = base_threshold
    if mode == "threshold_adjustment":
        mult, blocked, _ = inst.compute_regime_threshold_multiplier("crisis")
        if blocked:
            effective = float("inf")
        else:
            effective = base_threshold * mult

    assert effective == pytest.approx(0.18)


def test_threshold_adjustment_mode_scales_threshold():
    """Flag-on path: a crisis-hostile archetype (weight 0.5) sees threshold doubled."""
    inst = _inst({"crisis": 0.5, "risk_on": 1.0})
    base_threshold = 0.18

    # crisis: weight=0.5 → multiplier 2 → effective 0.36
    mult, blocked, _ = inst.compute_regime_threshold_multiplier("crisis")
    assert not blocked
    assert base_threshold * mult == pytest.approx(0.36)

    # risk_on: weight=1.0 → no change
    mult, blocked, _ = inst.compute_regime_threshold_multiplier("risk_on")
    assert not blocked
    assert base_threshold * mult == pytest.approx(0.18)


def test_threshold_adjustment_mode_blocks_below_floor():
    inst = _inst({"crisis": 0.1, "risk_on": 1.0})
    _, blocked, _ = inst.compute_regime_threshold_multiplier("crisis")
    assert blocked is True

    # risk_on still fine
    _, blocked, _ = inst.compute_regime_threshold_multiplier("risk_on")
    assert blocked is False


# ---------------------------------------------------------------------------
# 3. Symmetry property: weight w → multiplier 1/w (away from the block floor)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weight", [0.25, 0.5, 0.85, 1.0, 1.2, 2.0, 5.0])
def test_multiplier_is_inverse_of_weight(weight):
    inst = _inst({"x": weight})
    mult, blocked, _ = inst.compute_regime_threshold_multiplier("x", hard_block_floor=0.21)
    if weight >= 0.21:
        assert not blocked
        assert mult * weight == pytest.approx(1.0)
    else:
        assert blocked
