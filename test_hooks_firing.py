#!/usr/bin/env python3
"""
Test Knowledge v2.0 Hooks - Verify hooks fire correctly

This script tests that the fusion hooks can read features and apply
adjustments correctly. It's a sanity check before running full tests.
"""

import pandas as pd
import numpy as np
from engine.fusion.knowledge_hooks import apply_knowledge_hooks

print("=" * 70)
print("Testing Knowledge v2.0 Hooks")
print("=" * 70)

# Load feature store
print("\n📊 Loading feature store v2.0...")
df = pd.read_parquet("data/features_v2/ETH_1H_2024-07-01_to_2024-09-30.parquet")
print(f"   Loaded {len(df)} bars with {len(df.columns)} columns")

# Test configuration
config = {
    'knowledge_v2': {
        'enabled': True,
        'shadow_mode': False  # Apply adjustments for testing
    }
}

# Find bars with interesting features
print("\n🔍 Looking for bars with active features...")

# Find bars with high PTI score
high_pti = df[df['pti_score'] > 0.3]
print(f"   High PTI bars: {len(high_pti)}")

# Find bars with fakeouts
fakeouts = df[df['fakeout_detected'] == True]
print(f"   Fakeout bars: {len(fakeouts)}")

# Find bars with BOMS
boms_bars = df[df['boms_detected'] == True]
print(f"   BOMS bars: {len(boms_bars)}")

# Find bars with squiggle entry windows
squiggle_bars = df[df['squiggle_entry_window'] == True]
print(f"   Squiggle entry bars: {len(squiggle_bars)}")

# Test hooks on a few sample bars
print("\n🧪 Testing hooks on sample bars...")

test_indices = [100, 500, 1000, 1500, 2000]
results = []

for idx in test_indices:
    if idx >= len(df):
        continue

    # Get features for this bar
    feats = df.iloc[idx].to_dict()
    current_price = feats.get('close', 2500.0)

    # Test with neutral fusion score
    base_score = 0.65

    try:
        adjusted_score, threshold_delta, risk_mult, reasons = apply_knowledge_hooks(
            fusion_score=base_score,
            feats=feats,
            current_price=current_price,
            config=config
        )

        results.append({
            'index': idx,
            'timestamp': df.index[idx],
            'base_score': base_score,
            'adjusted_score': adjusted_score,
            'score_delta': adjusted_score - base_score,
            'threshold_delta': threshold_delta,
            'risk_mult': risk_mult,
            'reasons': reasons
        })

    except Exception as e:
        print(f"   ❌ Error at index {idx}: {e}")

# Display results
print("\n📋 Hook Test Results:")
print(f"{'Index':<8} {'Timestamp':<20} {'Base':<6} {'Adj':<6} {'ΔScore':<8} {'ΔThresh':<8} {'Risk':<6} {'Reasons'}")
print("-" * 120)

for r in results:
    reasons_str = ', '.join(r['reasons'][:2]) if r['reasons'] else 'None'
    if len(reasons_str) > 40:
        reasons_str = reasons_str[:37] + "..."

    print(f"{r['index']:<8} {str(r['timestamp'])[:19]:<20} "
          f"{r['base_score']:<6.3f} {r['adjusted_score']:<6.3f} "
          f"{r['score_delta']:+7.3f} {r['threshold_delta']:+7.3f} "
          f"{r['risk_mult']:<6.2f} {reasons_str}")

# Summary statistics
print("\n📊 Summary Statistics:")
if results:
    score_deltas = [r['score_delta'] for r in results]
    threshold_deltas = [r['threshold_delta'] for r in results]
    risk_mults = [r['risk_mult'] for r in results]

    print(f"   Score Delta:     mean={np.mean(score_deltas):+.4f}, std={np.std(score_deltas):.4f}, range=[{min(score_deltas):+.3f}, {max(score_deltas):+.3f}]")
    print(f"   Threshold Delta: mean={np.mean(threshold_deltas):+.4f}, std={np.std(threshold_deltas):.4f}, range=[{min(threshold_deltas):+.3f}, {max(threshold_deltas):+.3f}]")
    print(f"   Risk Multiplier: mean={np.mean(risk_mults):.3f}, std={np.std(risk_mults):.3f}, range=[{min(risk_mults):.2f}, {max(risk_mults):.2f}]")

    # Count reason types
    all_reasons = []
    for r in results:
        all_reasons.extend(r['reasons'])

    if all_reasons:
        print(f"\n   Total adjustments: {len(all_reasons)}")
        print("   Reason codes:")
        reason_types = {}
        for reason in all_reasons:
            code = reason.split(':')[0] if ':' in reason else reason
            reason_types[code] = reason_types.get(code, 0) + 1

        for code, count in sorted(reason_types.items(), key=lambda x: -x[1]):
            print(f"      {code}: {count}")
else:
    print("   No results to summarize")

# Test with extreme features
print("\n🎯 Testing with extreme feature values...")

extreme_test = {
    'close': 2500.0,
    'hob_score': 0.7,  # Trigger HOB bonus
    'pti_score': 0.8,  # High PTI
    'pti_trap_type': 'bullish_trap',
    'pti_confidence': 0.8,
    'pti_reversal_likely': True,
    'pti_rsi_divergence': 0.8,
    'pti_volume_exhaustion': 0.6,
    'pti_wick_trap': 0.7,
    'pti_failed_breakout': 0.5,
    'fakeout_detected': True,
    'fakeout_intensity': 0.75,
    'fakeout_direction': 'bullish_fakeout',
    'fakeout_return_speed': 2,
    'macro_regime': 'risk_off',
    'macro_correlation_score': -0.6,
    'boms_detected': False,
    'squiggle_entry_window': False,
    'range_outcome': 'none',
    'frvp_current_position': 'in_va',
    'frvp_distance_to_poc': 0.0,
    'conflict_score': 0.3,
    'structure_alignment': True,
}

print("\n   Extreme scenario: High PTI + Fakeout + Risk-off macro")
adj_score, thresh_delta, risk_mult, reasons = apply_knowledge_hooks(
    fusion_score=0.70,
    feats=extreme_test,
    current_price=2500.0,
    config=config
)

print(f"   Base score:  0.700")
print(f"   Adjusted:    {adj_score:.3f} (Δ {adj_score - 0.70:+.3f})")
print(f"   Threshold:   {thresh_delta:+.3f}")
print(f"   Risk mult:   {risk_mult:.2f}x")
print(f"   Reasons:     {', '.join(reasons)}")

# Verify safety bounds
print("\n🛡️  Verifying safety bounds...")
assert -0.30 <= (adj_score - 0.70) <= 0.30, "Score delta exceeded bounds!"
assert -0.10 <= thresh_delta <= 0.10, "Threshold delta exceeded bounds!"
assert 0.5 <= risk_mult <= 1.5, "Risk multiplier exceeded bounds!"
print("   ✅ All safety bounds respected")

print("\n" + "=" * 70)
print("✅ Hook Testing Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Run shadow mode test with optimizer")
print("2. Check decision logs for rule__* codes")
print("3. Compare baseline vs v2 active")
print("4. Run ablation tests to isolate best hooks")
