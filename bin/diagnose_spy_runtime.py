#!/usr/bin/env python3
"""
Diagnostic script for SPY runtime fusion/liquidity distributions.
Goal: Understand SPY's fusion levels to set appropriate archetype thresholds.
"""
import pandas as pd
import numpy as np
import json

# Load SPY feature store
spy = pd.read_parquet('data/features_mtf/SPY_1H_2023-01-01_to_2024-12-31.parquet')
print(f"Loaded SPY feature store: {len(spy)} bars")
print()

# Load config for weights
with open('configs/profile_eth_seed.json', 'r') as f:
    config = json.load(f)

# Compute runtime fusion (simplified version matching backtest logic)
def compute_runtime_fusion(row, config):
    """Simplified fusion computation for diagnostics."""
    weights = config.get('fusion', {}).get('weights', {})
    w_wyk = weights.get('wyckoff', 0.30)
    w_liq = weights.get('liquidity', 0.30)
    w_mom = weights.get('momentum', 0.25)
    w_smc = weights.get('smc', 0.15)

    # Components
    wyk = row.get('tf1d_wyckoff_score', 0.5)
    mom = row.get('rsi_14', 50) / 100.0
    smc = 1.0 if row.get('tf1h_fvg_present', False) else 0.0

    # Weighted sum (liquidity component skipped for simplicity)
    fusion = w_wyk * wyk + w_mom * mom + w_smc * smc
    return fusion

# Compute fusion for each bar
print("Computing runtime fusion scores...")
spy['runtime_fusion'] = spy.apply(lambda row: compute_runtime_fusion(row, config), axis=1)

# ─── Fusion Distribution ───
print("="*80)
print("RUNTIME FUSION DISTRIBUTION")
print("="*80)
fusion_mean = spy['runtime_fusion'].mean()
fusion_std = spy['runtime_fusion'].std()
fusion_p25 = spy['runtime_fusion'].quantile(0.25)
fusion_p50 = spy['runtime_fusion'].quantile(0.50)
fusion_p75 = spy['runtime_fusion'].quantile(0.75)
fusion_p90 = spy['runtime_fusion'].quantile(0.90)

print(f"Mean:     {fusion_mean:.3f}")
print(f"Std Dev:  {fusion_std:.3f}")
print(f"p25:      {fusion_p25:.3f}")
print(f"p50:      {fusion_p50:.3f}")
print(f"p75:      {fusion_p75:.3f}")
print(f"p90:      {fusion_p90:.3f}")
print()

# ─── Stored k2_fusion_score (for comparison) ───
if 'k2_fusion_score' in spy.columns:
    print("="*80)
    print("STORED k2_fusion_score (for comparison)")
    print("="*80)
    k2_mean = spy['k2_fusion_score'].mean()
    k2_std = spy['k2_fusion_score'].std()
    print(f"Mean:     {k2_mean:.3f}")
    print(f"Std Dev:  {k2_std:.3f}")
    print(f"p90:      {spy['k2_fusion_score'].quantile(0.90):.3f}")
    print()

# ─── Runtime Liquidity (if available) ───
if 'liquidity_score' in spy.columns:
    print("="*80)
    print("RUNTIME LIQUIDITY DISTRIBUTION")
    print("="*80)
    liq_median = spy['liquidity_score'].median()
    liq_p75 = spy['liquidity_score'].quantile(0.75)
    liq_p90 = spy['liquidity_score'].quantile(0.90)
    liq_nonzero_pct = (spy['liquidity_score'] > 0).sum() / len(spy) * 100

    print(f"Median:      {liq_median:.3f}")
    print(f"p75:         {liq_p75:.3f}")
    print(f"p90:         {liq_p90:.3f}")
    print(f"Nonzero (%): {liq_nonzero_pct:.1f}%")
    print()

# ─── Archetype Threshold Recommendations ───
print("="*80)
print("ARCHETYPE THRESHOLD RECOMMENDATIONS (SPY)")
print("="*80)
print(f"Conservative (p75):  fusion_floor = {fusion_p75:.2f}")
print(f"Moderate (p50):      fusion_floor = {fusion_p50:.2f}")
print(f"Aggressive (p25):    fusion_floor = {fusion_p25:.2f}")
print()
print(f"For seed config targeting 20-40 trades/year:")
print(f"  - entry_threshold_confidence: {fusion_p75:.2f} - {fusion_p90:.2f}")
print(f"  - archetype B fusion: {fusion_p25:.2f} - {fusion_p50:.2f}")
print()

# ─── Bars Above Thresholds ───
print("="*80)
print("BARS EXCEEDING FUSION THRESHOLDS")
print("="*80)
for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
    pct = (spy['runtime_fusion'] >= threshold).sum() / len(spy) * 100
    print(f"  fusion >= {threshold:.2f}: {pct:5.1f}%")
print()

print("✅ Diagnostic complete!")
print(f"📊 Recommendation: SPY fusion appears {'similar to ETH' if abs(fusion_mean - 0.277) < 0.05 else 'different from ETH'}")
print(f"   Use ETH-like ranges if mean ~0.28, or BTC-like if mean ~0.35")
