#!/usr/bin/env python3
"""
Validate v1.4.2 hotfix implementations
Test the three key changes: wider stops, regime filter, distribution exits
"""

import pandas as pd
import numpy as np
from bull_machine.modules.risk.dynamic_risk import calculate_stop_loss, wyckoff_state
from bull_machine.scoring.fusion import FusionEngineV141
from bull_machine.strategy.exits.advanced_rules import distribution_exit
import json


def test_wider_stops_markup_phases():
    """Test that markup phases (D, E) get 2.0x stops vs 1.5x for others."""
    print("🧪 Testing phase-aware stop calculation...")

    # Create synthetic data
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "high": [50000, 51000, 52000, 53000, 54000] * 4,  # 20 bars
            "low": [49000, 50000, 51000, 52000, 53000] * 4,
            "close": [49500, 50500, 51500, 52500, 53500] * 4,
        }
    )

    # Test different phases by manipulating range position
    test_cases = [
        ("Phase D (Markup)", [54000] * 20, [49000] * 20, [53000] * 20, 2.0),  # High close in range
        ("Phase E (Markdown)", [54000] * 20, [49000] * 20, [50000] * 20, 2.0),  # Low close in range
        (
            "Phase C (Consolidation)",
            [54000] * 20,
            [49000] * 20,
            [51500] * 20,
            1.5,
        ),  # Mid close in range
    ]

    for case_name, highs, lows, closes, expected_mult in test_cases:
        df_test = df.copy()
        df_test["high"] = highs
        df_test["low"] = lows
        df_test["close"] = closes

        phase_info = wyckoff_state(df_test)
        # Use current close as entry price for consistency
        entry_price = df_test.iloc[-1]["close"]
        stop_price = calculate_stop_loss(df_test, "long", entry_price, 0.5, 500)

        # Calculate implied multiplier
        stop_distance = entry_price - stop_price
        implied_mult = stop_distance / 500  # ATR = 500

        print(
            f"  {case_name}: Phase={phase_info['phase']}, Multiplier={implied_mult:.1f} (expected {expected_mult})"
        )

        # Allow some tolerance for depth/vol adjustments
        if phase_info["phase"] in ("D", "E"):
            assert implied_mult > 1.8, (
                f"Markup/Markdown should have >1.8x multiplier, got {implied_mult:.1f}"
            )
        else:
            assert implied_mult < 1.8, (
                f"Consolidation should have <1.8x multiplier, got {implied_mult:.1f}"
            )

    print("✅ Phase-aware stops working correctly")


def test_regime_filter_enhancement():
    """Test that high-vol A/C phases are allowed through."""
    print("🧪 Testing enhanced regime filter...")

    # Load config
    with open("configs/v141/profile_balanced.json") as f:
        config = json.load(f)

    engine = FusionEngineV141(config)

    test_cases = [
        ("Low-vol A phase", 0.7, {"phase": "A"}, [1000] * 20, False),  # Should be vetoed
        (
            "High-vol A phase",
            0.7,
            {"phase": "A"},
            [1000] * 19 + [2500],
            True,
        ),  # Should pass due to volume
        (
            "High Wyckoff A phase",
            0.9,
            {"phase": "A"},
            [1000] * 20,
            True,
        ),  # Should pass due to score
        ("Normal B phase", 0.7, {"phase": "B"}, [1000] * 20, True),  # Should pass (not A/C)
    ]

    for case_name, wyckoff_score, context, volumes, should_pass in test_cases:
        # Create test data for each case
        df = pd.DataFrame({"volume": volumes})

        result = engine.regime_filter(df, wyckoff_score, context)
        print(f"  {case_name}: {result} (expected {should_pass})")
        assert result == should_pass, f"{case_name} failed"

    print("✅ Enhanced regime filter working correctly")


def test_relaxed_distribution_exits():
    """Test that distribution exits use 1.4x volume ratio (vs 1.5x)."""
    print("🧪 Testing relaxed distribution exits...")

    # Create test data with consistent volume for SMA calculation
    base_volumes = [1000] * 20  # 20 bars for rolling mean

    test_cases = [
        ("1.3x volume", 1300, False),  # Below 1.4x threshold
        ("1.4x volume", 1400, True),  # At new threshold
        ("1.5x volume", 1500, True),  # Above threshold
    ]

    for case_name, test_volume, should_trigger in test_cases:
        # Create test data for each case
        df = pd.DataFrame({"volume": base_volumes})

        result = distribution_exit(df, test_volume, 0.3)  # displacement < 0.4
        triggered = result.get("metadata", {}).get("distribution_detected", False)
        vol_ratio = test_volume / 1000.0  # Base volume is 1000
        print(
            f"  {case_name}: vol_ratio={vol_ratio:.1f}, triggered={triggered} (expected {should_trigger})"
        )
        assert triggered == should_trigger, f"{case_name} failed"

    print("✅ Relaxed distribution exits working correctly")


def test_configuration_integration():
    """Test that all hotfix parameters are properly integrated."""
    print("🧪 Testing configuration integration...")

    with open("configs/v141/profile_balanced.json") as f:
        config = json.load(f)

    # Check hotfix parameters
    assert config["stop_loss"]["base_atr_multipliers"]["markup_phases"] == 2.0
    assert config["regime_filter"]["min_vol_ratio"] == 1.2
    assert config["exit_thresholds"]["distribution_exit"]["volume_ratio"] == 1.4
    assert config["exit_thresholds"]["distribution_exit"]["displacement"] == 0.4

    print("✅ All hotfix parameters correctly configured")


def main():
    """Run all v1.4.2 validation tests."""
    print("🚀 Bull Machine v1.4.2 Hotfix Validation")
    print("=" * 50)

    try:
        test_wider_stops_markup_phases()
        test_regime_filter_enhancement()
        test_relaxed_distribution_exits()
        test_configuration_integration()

        print("\n" + "=" * 50)
        print("🎉 ALL v1.4.2 HOTFIX VALIDATIONS PASSED")
        print("✅ Wider stops for markup phases: IMPLEMENTED")
        print("✅ Enhanced regime filter: IMPLEMENTED")
        print("✅ Relaxed distribution exits: IMPLEMENTED")
        print("✅ Configuration integration: COMPLETE")

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
