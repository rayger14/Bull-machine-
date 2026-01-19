#!/usr/bin/env python3
"""
Diagnose why 6 archetypes produce ZERO signals in Q1 2023 smoke test.
"""
import pandas as pd
import numpy as np
import sys

# Load data (using the same file as smoke test)
df = pd.read_parquet('/Users/raymondghandchi/Bull-machine-/Bull-machine-/data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

# Ensure datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

# Filter to Q1 2023 (smoke test period)
q1_2023 = df[(df.index >= '2023-01-01') & (df.index < '2023-04-01')].copy()
print(f"Q1 2023 data: {len(q1_2023)} bars\n")
print("="*80)

# ============================================================================
# S1 - Liquidity Vacuum
# ============================================================================
print("\n### S1 - LIQUIDITY VACUUM ###\n")

# Check V2 features
s1_v2_features = ['capitulation_depth', 'crisis_composite', 'volume_climax_last_3b', 'wick_exhaustion_last_3b']
s1_has_v2 = all(f in df.columns for f in s1_v2_features)
print(f"Has V2 features: {s1_has_v2}")

if s1_has_v2:
    # V2 Logic Analysis
    cap_depth = q1_2023['capitulation_depth']
    crisis = q1_2023['crisis_composite']
    vol_climax = q1_2023['volume_climax_last_3b']
    wick_exhaust = q1_2023['wick_exhaustion_last_3b']

    print(f"\nV2 Hard Gates (default thresholds):")
    print(f"  capitulation_depth < -0.20: {(cap_depth < -0.20).sum()} bars ({100*(cap_depth < -0.20).sum()/len(q1_2023):.1f}%)")
    print(f"  crisis_composite >= 0.40: {(crisis >= 0.40).sum()} bars ({100*(crisis >= 0.40).sum()/len(q1_2023):.1f}%)")
    print(f"  volume_climax_3b > 0.25: {(vol_climax > 0.25).sum()} bars ({100*(vol_climax > 0.25).sum()/len(q1_2023):.1f}%)")
    print(f"  wick_exhaustion_3b > 0.30: {(wick_exhaust > 0.30).sum()} bars ({100*(wick_exhaust > 0.30).sum()/len(q1_2023):.1f}%)")

    # Both V2 hard gates
    both_gates = (cap_depth < -0.20) & (crisis >= 0.40)
    print(f"\n  BOTH V2 gates pass: {both_gates.sum()} bars ({100*both_gates.sum()/len(q1_2023):.1f}%)")

    # With exhaustion OR gate
    exhaustion_or = (vol_climax > 0.25) | (wick_exhaust > 0.30)
    full_v2 = both_gates & exhaustion_or
    print(f"  Full V2 logic (gates + exhaustion): {full_v2.sum()} bars ({100*full_v2.sum()/len(q1_2023):.1f}%)")

    print(f"\nV2 Value ranges:")
    print(f"  capitulation_depth: [{cap_depth.min():.3f}, {cap_depth.max():.3f}], median={cap_depth.median():.3f}")
    print(f"  crisis_composite: [{crisis.min():.3f}, {crisis.max():.3f}], median={crisis.median():.3f}")
    print(f"  volume_climax_3b: [{vol_climax.min():.3f}, {vol_climax.max():.3f}], median={vol_climax.median():.3f}")
    print(f"  wick_exhaustion_3b: [{wick_exhaust.min():.3f}, {wick_exhaust.max():.3f}], median={wick_exhaust.median():.3f}")

# V1 Fallback Analysis
liq = q1_2023['liquidity_score']
vol_z = q1_2023['volume_zscore']
wick = q1_2023['wick_lower_ratio']

print(f"\nV1 Fallback Gates:")
print(f"  liquidity_score < 0.20: {(liq < 0.20).sum()} bars ({100*(liq < 0.20).sum()/len(q1_2023):.1f}%)")
print(f"  volume_zscore > 1.5: {(vol_z > 1.5).sum()} bars ({100*(vol_z > 1.5).sum()/len(q1_2023):.1f}%)")
print(f"  wick_lower_ratio > 0.28: {(wick > 0.28).sum()} bars ({100*(wick > 0.28).sum()/len(q1_2023):.1f}%)")

v1_gate1 = (liq < 0.20)
v1_gate2 = (vol_z > 1.5) | (wick > 0.28)
v1_full = v1_gate1 & v1_gate2
print(f"  Full V1 logic: {v1_full.sum()} bars ({100*v1_full.sum()/len(q1_2023):.1f}%)")

# ============================================================================
# S5 - Long Squeeze
# ============================================================================
print("\n" + "="*80)
print("\n### S5 - LONG SQUEEZE ###\n")

funding_z = q1_2023['funding_Z']
rsi = q1_2023['rsi_14']
liq_s5 = q1_2023['liquidity_score']
bos_bullish = q1_2023.get('tf1h_bos_bullish', pd.Series([False]*len(q1_2023)))

print(f"Gates (default thresholds):")
print(f"  funding_Z >= 1.2: {(funding_z >= 1.2).sum()} bars ({100*(funding_z >= 1.2).sum()/len(q1_2023):.1f}%)")
print(f"  rsi_14 >= 70: {(rsi >= 70).sum()} bars ({100*(rsi >= 70).sum()/len(q1_2023):.1f}%)")
print(f"  liquidity_score < 0.25: {(liq_s5 < 0.25).sum()} bars ({100*(liq_s5 < 0.25).sum()/len(q1_2023):.1f}%)")
print(f"  tf1h_bos_bullish = False (NOT vetoed): {(~bos_bullish).sum()} bars ({100*(~bos_bullish).sum()/len(q1_2023):.1f}%)")

gate1 = (funding_z >= 1.2)
gate2 = (rsi >= 70)
gate3 = (liq_s5 < 0.25)
veto = (~bos_bullish)
full_s5 = gate1 & gate2 & gate3 & veto
print(f"\n  All gates pass: {full_s5.sum()} bars ({100*full_s5.sum()/len(q1_2023):.1f}%)")

print(f"\nValue ranges:")
print(f"  funding_Z: [{funding_z.min():.3f}, {funding_z.max():.3f}], median={funding_z.median():.3f}")
print(f"  rsi_14: [{rsi.min():.1f}, {rsi.max():.1f}], median={rsi.median():.1f}")
print(f"  liquidity_score: [{liq_s5.min():.3f}, {liq_s5.max():.3f}], median={liq_s5.median():.3f}")

# ============================================================================
# A - Spring (PTI Trap)
# ============================================================================
print("\n" + "="*80)
print("\n### A - SPRING (PTI TRAP) ###\n")

# Check for PTI features
pti_features = ['pti_trap_type', 'pti_score', 'boms_disp', 'atr']
pti_available = {f: f in df.columns for f in pti_features}
print(f"Feature availability:")
for f, avail in pti_available.items():
    print(f"  {f}: {avail}")

# Check for tf1h variants
if 'tf1h_pti_trap_type' in df.columns:
    pti_trap = q1_2023['tf1h_pti_trap_type']
    pti_score_col = q1_2023.get('tf1h_pti_score', pd.Series([0.0]*len(q1_2023)))
    print(f"\n  Using tf1h_pti_trap_type (found)")
    print(f"  Spring traps: {(pti_trap == 'spring').sum()} bars")
    print(f"  UTAD traps: {(pti_trap == 'utad').sum()} bars")
    print(f"  Any trap: {(pti_trap.isin(['spring', 'utad'])).sum()} bars")
    if 'tf1h_pti_score' in df.columns:
        print(f"  pti_score > 0.40: {(pti_score_col > 0.40).sum()} bars")
else:
    print(f"\n  MISSING: pti_trap_type not found (checked both 'pti_trap_type' and 'tf1h_pti_trap_type')")

# Check for displacement
if 'tf4h_boms_displacement' in df.columns:
    disp = q1_2023['tf4h_boms_displacement']
    atr_col = q1_2023.get('atr_14', q1_2023.get('atr_20', pd.Series([1.0]*len(q1_2023))))
    print(f"\n  Using tf4h_boms_displacement")
    print(f"  displacement > 0.8*ATR: {(disp > 0.8 * atr_col).sum()} bars")

# ============================================================================
# C - BOS/CHOCH (Wick Trap)
# ============================================================================
print("\n" + "="*80)
print("\n### C - BOS/CHOCH (WICK TRAP) ###\n")

bos_bull = q1_2023.get('tf1h_bos_bullish', pd.Series([False]*len(q1_2023)))
choch = q1_2023.get('tf1h_choch_flag', pd.Series([False]*len(q1_2023)))

print(f"Feature availability:")
print(f"  tf1h_bos_bullish: {'tf1h_bos_bullish' in df.columns}")
print(f"  tf1h_choch_flag: {'tf1h_choch_flag' in df.columns}")

if 'tf1h_bos_bullish' in df.columns:
    print(f"\nGates:")
    print(f"  tf1h_bos_bullish: {bos_bull.sum()} bars ({100*bos_bull.sum()/len(q1_2023):.1f}%)")
    if 'tf1h_choch_flag' in df.columns:
        print(f"  tf1h_choch_flag: {choch.sum()} bars ({100*choch.sum()/len(q1_2023):.1f}%)")
        both = bos_bull & choch
        print(f"  BOTH: {both.sum()} bars ({100*both.sum()/len(q1_2023):.1f}%)")
    else:
        print(f"  MISSING: tf1h_choch_flag")

# Check for tf4h_choch_flag as alternative
if 'tf4h_choch_flag' in df.columns:
    choch_4h = q1_2023['tf4h_choch_flag']
    print(f"\n  Alternative: tf4h_choch_flag exists: {choch_4h.sum()} bars ({100*choch_4h.sum()/len(q1_2023):.1f}%)")

# ============================================================================
# M - Coil Break (Confluence Breakout)
# ============================================================================
print("\n" + "="*80)
print("\n### M - COIL BREAK (CONFLUENCE BREAKOUT) ###\n")

atr_pct = q1_2023.get('atr_percentile', None)
bos_4h = q1_2023.get('tf4h_bos_bullish', pd.Series([False]*len(q1_2023)))

print(f"Feature availability:")
print(f"  atr_percentile: {'atr_percentile' in df.columns}")
print(f"  tf4h_bos_bullish: {'tf4h_bos_bullish' in df.columns}")

if atr_pct is not None:
    print(f"\nGates:")
    print(f"  atr_percentile <= 0.25: {(atr_pct <= 0.25).sum()} bars ({100*(atr_pct <= 0.25).sum()/len(q1_2023):.1f}%)")
    print(f"  atr_percentile range: [{atr_pct.min():.3f}, {atr_pct.max():.3f}], median={atr_pct.median():.3f}")
else:
    print(f"\n  MISSING: atr_percentile not found")
    # Try to compute from available ATR
    if 'atr_14' in df.columns:
        atr_14 = q1_2023['atr_14']
        print(f"  atr_14 range: [{atr_14.min():.3f}, {atr_14.max():.3f}], median={atr_14.median():.3f}")

# Check for BOS signals
if 'tf4h_bos_bullish' not in df.columns:
    print(f"\n  MISSING: tf4h_bos_bullish")
    # Check for alternatives
    if 'tf4h_boms_direction' in df.columns:
        boms_dir = q1_2023['tf4h_boms_direction']
        print(f"  Alternative: tf4h_boms_direction (bullish): {(boms_dir == 'bullish').sum()} bars")

# ============================================================================
# S8 - Fakeout Exhaustion (Volume Fade Chop)
# ============================================================================
print("\n" + "="*80)
print("\n### S8 - FAKEOUT EXHAUSTION (VOLUME FADE CHOP) ###\n")

vol_z_s8 = q1_2023['volume_zscore']
atr_pct_s8 = q1_2023.get('atr_percentile', None)

print(f"Feature availability:")
print(f"  volume_zscore: {'volume_zscore' in df.columns}")
print(f"  atr_percentile: {'atr_percentile' in df.columns}")

print(f"\nGates (default thresholds):")
print(f"  volume_zscore <= -0.5: {(vol_z_s8 <= -0.5).sum()} bars ({100*(vol_z_s8 <= -0.5).sum()/len(q1_2023):.1f}%)")

if atr_pct_s8 is not None:
    print(f"  atr_percentile <= 0.35: {(atr_pct_s8 <= 0.35).sum()} bars ({100*(atr_pct_s8 <= 0.35).sum()/len(q1_2023):.1f}%)")
    both_s8 = (vol_z_s8 <= -0.5) & (atr_pct_s8 <= 0.35)
    print(f"  BOTH gates: {both_s8.sum()} bars ({100*both_s8.sum()/len(q1_2023):.1f}%)")
else:
    print(f"  MISSING: atr_percentile (cannot compute full logic)")

print(f"\nValue ranges:")
print(f"  volume_zscore: [{vol_z_s8.min():.3f}, {vol_z_s8.max():.3f}], median={vol_z_s8.median():.3f}")
if atr_pct_s8 is not None:
    print(f"  atr_percentile: [{atr_pct_s8.min():.3f}, {atr_pct_s8.max():.3f}], median={atr_pct_s8.median():.3f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("\n### SUMMARY ###\n")

print("Root Causes:")
print("  S1: V2 logic requires BOTH cap_depth < -20% AND crisis >= 0.40 (AND gate blocking)")
print("  S5: Triple gate (funding_Z >= 1.2 AND rsi >= 70 AND liquidity < 0.25) too strict")
print("  A: MISSING features (pti_trap_type, pti_score, boms_disp)")
print("  C: MISSING feature (tf1h_choch_flag)")
print("  M: MISSING features (atr_percentile, tf4h_bos_bullish)")
print("  S8: MISSING feature (atr_percentile)")

print("\nPriority:")
print("  1. A, C, M, S8: Add missing features (feature engineering)")
print("  2. S1: Relax V2 gates (use OR logic or lower thresholds)")
print("  3. S5: Relax one gate (funding OR rsi, not AND)")
