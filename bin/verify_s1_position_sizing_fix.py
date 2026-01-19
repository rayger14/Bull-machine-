#!/usr/bin/env python3
"""
Verify S1 Position Sizing Fix
Tests that archetype_weight reduction from 2.5 to 1.0 fixes the -75% drawdown issue.
"""

import json
import sys
from pathlib import Path

def verify_config_fix(config_path: str = "configs/s1_v2_production.json"):
    """Verify the position sizing fix in the S1 config."""

    print("=" * 70)
    print("S1 POSITION SIZING FIX VERIFICATION")
    print("=" * 70)

    # Load config
    with open(config_path) as f:
        cfg = json.load(f)

    # Extract key parameters
    archetype_weight = cfg['archetypes']['thresholds']['liquidity_vacuum']['archetype_weight']
    base_risk = cfg['risk']['base_risk_pct']
    max_risk = cfg['archetypes']['thresholds']['liquidity_vacuum']['max_risk_pct']

    print(f"\nCONFIG: {config_path}")
    print("-" * 70)
    print(f"archetype_weight: {archetype_weight}")
    print(f"base_risk_pct: {base_risk*100:.2f}%")
    print(f"max_risk_pct: {max_risk*100:.2f}%")

    # Calculate effective risk
    # In backtest_knowledge_v2.py, position sizing formula:
    # 1. Base position from risk_dollars / stop_pct
    # 2. Multiply by fusion_weight (0.75-1.35, avg ~1.0)
    # 3. Multiply by regime_mult (0.5-1.2, depends on regime)
    # 4. Multiply by archetype_weight
    # 5. Cap at 95% equity

    # Conservative estimate (neutral regime, avg fusion)
    fusion_mult = 1.0  # Average
    regime_mult = 1.0  # Neutral regime
    effective_risk = base_risk * fusion_mult * regime_mult * archetype_weight

    print(f"\nEFFECTIVE RISK CALCULATION:")
    print("-" * 70)
    print(f"Base risk: {base_risk*100:.2f}%")
    print(f"× Fusion multiplier (avg): {fusion_mult}")
    print(f"× Regime multiplier (neutral): {regime_mult}")
    print(f"× Archetype weight: {archetype_weight}")
    print(f"= Effective risk per trade: {effective_risk*100:.2f}%")

    # Calculate expected drawdown for 10 consecutive losses
    n_losses = 10
    dd_compound = 100 - (1 - effective_risk)**n_losses * 100
    dd_linear = n_losses * effective_risk * 100

    print(f"\nEXPECTED DRAWDOWN ({n_losses} consecutive losses):")
    print("-" * 70)
    print(f"Compounding formula: {dd_compound:.1f}%")
    print(f"Linear approximation: {dd_linear:.1f}%")

    # Worst case (risk_on regime + high fusion)
    worst_fusion = 1.35
    worst_regime = 1.2
    worst_risk = base_risk * worst_fusion * worst_regime * archetype_weight
    worst_dd = 100 - (1 - worst_risk)**n_losses * 100

    print(f"\nWORST CASE (risk_on + high fusion):")
    print("-" * 70)
    print(f"Effective risk: {worst_risk*100:.2f}%")
    print(f"Expected DD: {worst_dd:.1f}%")

    # Verification
    print(f"\n" + "=" * 70)
    print("VERIFICATION RESULTS:")
    print("=" * 70)

    target_dd = 20.0
    issues = []

    if archetype_weight > 1.5:
        issues.append(f"❌ archetype_weight too high: {archetype_weight} (should be ≤ 1.5)")
    else:
        print(f"✓ archetype_weight: {archetype_weight} (acceptable)")

    if dd_compound > target_dd:
        issues.append(f"❌ Expected DD exceeds target: {dd_compound:.1f}% > {target_dd}%")
    else:
        print(f"✓ Expected DD within target: {dd_compound:.1f}% ≤ {target_dd}%")

    if worst_dd > 30:
        issues.append(f"⚠️  Worst case DD high: {worst_dd:.1f}% (acceptable but risky)")
    else:
        print(f"✓ Worst case DD acceptable: {worst_dd:.1f}%")

    if effective_risk > 0.025:
        issues.append(f"❌ Effective risk too high: {effective_risk*100:.2f}% > 2.5%")
    else:
        print(f"✓ Effective risk safe: {effective_risk*100:.2f}% ≤ 2.5%")

    print()
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\n❌ FIX INCOMPLETE - Requires further adjustment")
        return False
    else:
        print("✓ ALL CHECKS PASSED")
        print(f"✓ Expected max drawdown: {dd_compound:.1f}% (target: <{target_dd}%)")
        print("✓ Position sizing fix VERIFIED")
        return True

def compare_before_after():
    """Compare old vs new configuration."""
    print("\n" + "=" * 70)
    print("BEFORE/AFTER COMPARISON")
    print("=" * 70)

    scenarios = [
        ("BEFORE (archetype_weight=2.5)", 2.5, 0.02),
        ("AFTER  (archetype_weight=1.0)", 1.0, 0.02),
    ]

    for name, arch_w, base_r in scenarios:
        eff_risk = arch_w * base_r
        dd_10 = 100 - (1 - eff_risk)**10 * 100
        print(f"\n{name}:")
        print(f"  Effective risk: {eff_risk*100:.1f}%")
        print(f"  DD (10 losses): {dd_10:.1f}%")
        print(f"  Status: {'❌ UNSAFE' if dd_10 > 20 else '✓ SAFE'}")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/s1_v2_production.json"

    success = verify_config_fix(config_path)
    compare_before_after()

    sys.exit(0 if success else 1)
