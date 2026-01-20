#!/usr/bin/env python3
"""
ETH Runtime Fusion/Liquidity Diagnostic Script

Emulates the backtest engine's runtime computation to check:
1. Fusion score distributions (expect mean ~0.25-0.35, p75 > 0.35)
2. Liquidity score distributions (expect median ~0.15-0.25)
3. How many bars would pass archetype precheck
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.fusion.domain_fusion import analyze_fusion
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.context.loader import load_macro_data

def compute_runtime_fusion_simple(row, config):
    """Simplified fusion computation for diagnostics."""
    weights = config.get('fusion', {}).get('weights', {})
    w_wyk = weights.get('wyckoff', 0.30)
    w_liq = weights.get('liquidity', 0.30)
    w_mom = weights.get('momentum', 0.25)
    w_smc = weights.get('smc', 0.15)

    # Read scores from row
    wyk = row.get('tf1d_wyckoff_score', 0.5)
    # Liquidity computed separately via runtime module
    mom = row.get('rsi_14', 50) / 100.0  # Normalize RSI
    smc = 1.0 if row.get('tf1h_fvg_present', False) else 0.0

    fusion = w_wyk * wyk + w_mom * mom + w_smc * smc
    return fusion

def main():
    print("=" * 80)
    print("ETH Runtime Fusion/Liquidity Diagnostic")
    print("=" * 80)

    # Load ETH feature store
    eth_path = 'data/features_mtf/ETH_1H_2024-01-01_to_2025-10-17.parquet'
    df = pd.read_parquet(eth_path)

    # Filter to 2024 only
    df_2024 = df[(df.index >= '2024-01-01') & (df.index <= '2024-12-31')]
    print(f"\nLoaded {len(df_2024)} bars (2024 only)")

    # Load config
    with open('configs/profile_production.json') as f:
        config = json.load(f)

    # 1. Check stored fusion scores (should be placeholder)
    print("\n" + "=" * 80)
    print("1. STORED FUSION SCORES (Feature Store)")
    print("=" * 80)

    if 'k2_fusion_score' in df_2024.columns:
        k2_fusion = df_2024['k2_fusion_score']
        print(f"\nk2_fusion_score:")
        print(f"  mean:   {k2_fusion.mean():.3f}")
        print(f"  median: {k2_fusion.median():.3f}")
        print(f"  std:    {k2_fusion.std():.3f}")
        print(f"  p25:    {k2_fusion.quantile(0.25):.3f}")
        print(f"  p75:    {k2_fusion.quantile(0.75):.3f}")
        print(f"  p90:    {k2_fusion.quantile(0.90):.3f}")

        if k2_fusion.std() < 0.01:
            print(f"  ⚠️ CONSTANT VALUE - This is placeholder data, not real fusion!")

    # 2. Simulate runtime fusion computation
    print("\n" + "=" * 80)
    print("2. RUNTIME FUSION (Simulated)")
    print("=" * 80)

    runtime_fusion = []
    for idx, row in df_2024.iterrows():
        fusion = compute_runtime_fusion_simple(row, config)
        runtime_fusion.append(fusion)

    runtime_fusion = np.array(runtime_fusion)
    print(f"\nRuntime fusion (simplified):")
    print(f"  mean:   {runtime_fusion.mean():.3f}")
    print(f"  median: {np.median(runtime_fusion):.3f}")
    print(f"  std:    {runtime_fusion.std():.3f}")
    print(f"  p25:    {np.percentile(runtime_fusion, 25):.3f}")
    print(f"  p75:    {np.percentile(runtime_fusion, 75):.3f}")
    print(f"  p90:    {np.percentile(runtime_fusion, 90):.3f}")

    # Check if runtime fusion looks healthy
    if runtime_fusion.mean() > 0.20 and runtime_fusion.std() > 0.05:
        print(f"  ✅ Distribution looks healthy (varied, not constant)")
    else:
        print(f"  ⚠️ Distribution may be problematic")

    # Count how many bars exceed common thresholds
    print(f"\nBars exceeding thresholds:")
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
        count = np.sum(runtime_fusion >= threshold)
        pct = 100 * count / len(runtime_fusion)
        print(f"  fusion >= {threshold:.2f}: {count:4d} bars ({pct:5.1f}%)")

    # 3. Check component scores
    print("\n" + "=" * 80)
    print("3. FUSION COMPONENTS")
    print("=" * 80)

    print(f"\ntf1d_wyckoff_score:")
    wyk = df_2024['tf1d_wyckoff_score']
    print(f"  mean:   {wyk.mean():.3f}")
    print(f"  p25/50/75: {wyk.quantile(0.25):.3f} / {wyk.median():.3f} / {wyk.quantile(0.75):.3f}")

    print(f"\nrsi_14 (momentum proxy):")
    rsi = df_2024['rsi_14']
    print(f"  mean:   {rsi.mean():.1f}")
    print(f"  p25/50/75: {rsi.quantile(0.25):.1f} / {rsi.median():.1f} / {rsi.quantile(0.75):.1f}")

    print(f"\ntf1h_fvg_present (SMC):")
    fvg = df_2024['tf1h_fvg_present']
    print(f"  True:   {fvg.sum()} bars ({100*fvg.sum()/len(fvg):.1f}%)")
    print(f"  False:  {(~fvg).sum()} bars ({100*(~fvg).sum()/len(fvg):.1f}%)")

    # 4. Liquidity (placeholder - would need runtime module)
    print("\n" + "=" * 80)
    print("4. LIQUIDITY (Note: Requires runtime computation)")
    print("=" * 80)
    print("\nℹ️  Liquidity is computed at runtime by the runtime_liquidity module")
    print("   It's not stored in the feature store")
    print("   Expected: median ~0.15-0.25, similar to BTC")
    print("   If ETH runtime liq < 0.10, the precheck will block entries")

    # 5. Archetype precheck simulation
    print("\n" + "=" * 80)
    print("5. ARCHETYPE PRECHECK SIMULATION")
    print("=" * 80)

    # Use BTC thresholds
    fusion_floor = config.get('fusion', {}).get('entry_threshold_confidence', 0.35)
    min_liq = config.get('archetypes', {}).get('thresholds', {}).get('min_liquidity', 0.14)

    print(f"\nBTC thresholds:")
    print(f"  final_fusion_floor: {fusion_floor:.2f}")
    print(f"  min_liquidity: {min_liq:.2f}")

    # Count how many bars would pass fusion precheck
    passing_fusion = np.sum(runtime_fusion >= fusion_floor)
    print(f"\nBars passing fusion precheck (>= {fusion_floor:.2f}):")
    print(f"  {passing_fusion} / {len(runtime_fusion)} = {100*passing_fusion/len(runtime_fusion):.2f}%")

    if passing_fusion == 0:
        print(f"\n❌ PROBLEM: 0 bars pass BTC fusion threshold!")
        print(f"   Recommendation: Lower fusion_floor to 0.20-0.25 for ETH")
    elif passing_fusion < 100:
        print(f"\n⚠️  Very few bars pass - may need lower threshold")
    else:
        print(f"\n✅ Enough bars pass for archetype detection")

    # 6. Recommendations
    print("\n" + "=" * 80)
    print("6. RECOMMENDATIONS")
    print("=" * 80)

    # Find threshold that captures ~10-15% of bars
    target_percentile = 85  # Top 15%
    suggested_fusion_floor = np.percentile(runtime_fusion, target_percentile)

    print(f"\nSuggested ETH thresholds (to capture top 15% of bars):")
    print(f"  final_fusion_floor: {suggested_fusion_floor:.2f} (BTC uses 0.35)")
    print(f"  min_liquidity: 0.10 (BTC uses 0.14)")
    print(f"\nArchetype-specific fusion thresholds (suggest -25% from BTC):")
    print(f"  B_fusion: 0.24 (BTC: 0.32)")
    print(f"  L_fusion: 0.27 (BTC: 0.36)")
    print(f"  K_fusion: 0.27 (BTC: 0.36)")

    bars_at_suggested = np.sum(runtime_fusion >= suggested_fusion_floor)
    print(f"\nExpected bars with suggested threshold: {bars_at_suggested} ({100*bars_at_suggested/len(runtime_fusion):.1f}%)")
    print(f"  Target: 5-20 trades/year → need ~0.5-2% of bars to fire")

if __name__ == '__main__':
    main()
