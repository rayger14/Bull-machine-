#!/usr/bin/env python3
"""Debug script to understand why archetypes aren't matching."""

import pandas as pd
import json
from engine.archetypes.logic_v2_adapter import ArchetypeLogic

# Load the config
with open('configs/profile_experimental.json') as f:
    config = json.load(f)

# Create archetype logic
logic = ArchetypeLogic(config['archetypes'])

print(f"use_archetypes: {logic.use_archetypes}")
print(f"min_liquidity: {logic.min_liquidity}")
print(f"Enabled archetypes: {logic.enabled}")
print()

# Load feature store
df = pd.read_parquet('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet')
print(f"Feature store loaded: {len(df)} rows")
print(f"Columns: {list(df.columns)[:10]}...")
print()

# Check first 100 rows for archetype matches
matches = []
for idx in range(min(100, len(df))):
    row = df.iloc[idx]
    prev_row = df.iloc[idx-1] if idx > 0 else None

    archetype, fusion, liq = logic.check_archetype(row, prev_row, df, idx)
    if archetype:
        matches.append((idx, archetype, fusion, liq))
        print(f"Row {idx}: {archetype}, fusion={fusion:.3f}, liq={liq:.3f}")

print(f"\nTotal matches in first 100 rows: {len(matches)}")

# Sample some liquidity scores to understand distribution
liq_scores = []
for idx in range(min(1000, len(df))):
    row = df.iloc[idx]
    liq = logic._liquidity_score(row)
    liq_scores.append(liq)

print(f"\nLiquidity score distribution (first 1000 rows):")
print(f"  Mean: {pd.Series(liq_scores).mean():.4f}")
print(f"  Median: {pd.Series(liq_scores).median():.4f}")
print(f"  Min: {pd.Series(liq_scores).min():.4f}")
print(f"  Max: {pd.Series(liq_scores).max():.4f}")
print(f"  > 0.05: {(pd.Series(liq_scores) > 0.05).sum()} ({100 * (pd.Series(liq_scores) > 0.05).sum() / len(liq_scores):.1f}%)")

# Check fusion scores too
fusion_scores = []
for idx in range(min(1000, len(df))):
    row = df.iloc[idx]
    fusion = logic._fusion(row)
    fusion_scores.append(fusion)

print(f"\nFusion score distribution (first 1000 rows):")
print(f"  Mean: {pd.Series(fusion_scores).mean():.4f}")
print(f"  Median: {pd.Series(fusion_scores).median():.4f}")
print(f"  Min: {pd.Series(fusion_scores).min():.4f}")
print(f"  Max: {pd.Series(fusion_scores).max():.4f}")

# Check specific archetype thresholds
print(f"\nArchetype fusion thresholds:")
for arch in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M']:
    thresh_key = f"thresh_{arch}"
    if hasattr(logic, thresh_key):
        thresh_dict = getattr(logic, thresh_key)
        fusion_thresh = thresh_dict.get('fusion', 0)
        print(f"  {arch}: fusion >= {fusion_thresh}")
