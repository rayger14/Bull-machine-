#!/usr/bin/env python3
"""
Diagnose why domain engines aren't affecting S1 behavior
"""

import sys
import pandas as pd
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.runtime.context import RuntimeContext
from engine.archetypes.logic_v2_adapter import ArchetypeLogic

# Load feature store
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

# Load configs
with open('configs/variants/s1_core.json') as f:
    config_core = json.load(f)

with open('configs/variants/s1_full.json') as f:
    config_full = json.load(f)

# Initialize archetype logic for both configs
logic_core = ArchetypeLogic(config_core)
logic_full = ArchetypeLogic(config_full)

print("=" * 80)
print("DOMAIN ENGINE DIAGNOSTIC")
print("=" * 80)
print()

# Test on first 100 rows
test_rows = df.head(1000)

print(f"Testing on {len(test_rows)} rows...")
print()

# Track signals
core_signals = []
full_signals = []

for idx, row in test_rows.iterrows():
    # Build RuntimeContext for CORE variant
    ctx_core = RuntimeContext(
        ts=row.name,
        row=row,
        regime_probs={'neutral': 1.0},
        regime_label='neutral',
        adapted_params={},
        thresholds=config_core.get('archetypes', {}).get('thresholds', {}),
        metadata={
            'feature_flags': config_core.get('feature_flags', {}),
            'df': df,
            'index': idx
        }
    )

    # Build RuntimeContext for FULL variant
    ctx_full = RuntimeContext(
        ts=row.name,
        row=row,
        regime_probs={'neutral': 1.0},
        regime_label='neutral',
        adapted_params={},
        thresholds=config_full.get('archetypes', {}).get('thresholds', {}),
        metadata={
            'feature_flags': config_full.get('feature_flags', {}),
            'df': df,
            'index': idx
        }
    )

    # Check liquidity_vacuum archetype for both
    matched_core, score_core, metadata_core = logic_core._check_S1_v2_binary(ctx_core)
    matched_full, score_full, metadata_full = logic_full._check_S1_v2_binary(ctx_full)

    if matched_core:
        core_signals.append({
            'ts': row.name,
            'score': score_core,
            'domain_boost': metadata_core.get('domain_boost', 1.0),
            'domain_signals': metadata_core.get('domain_signals', [])
        })

    if matched_full:
        full_signals.append({
            'ts': row.name,
            'score': score_full,
            'domain_boost': metadata_full.get('domain_boost', 1.0),
            'domain_signals': metadata_full.get('domain_signals', [])
        })

print(f"CORE variant signals: {len(core_signals)}")
print(f"FULL variant signals: {len(full_signals)}")
print()

if core_signals:
    print("CORE variant first 3 signals:")
    for sig in core_signals[:3]:
        print(f"  {sig['ts']}: score={sig['score']:.3f}, boost={sig['domain_boost']:.2f}, signals={sig['domain_signals']}")
    print()

if full_signals:
    print("FULL variant first 3 signals:")
    for sig in full_signals[:3]:
        print(f"  {sig['ts']}: score={sig['score']:.3f}, boost={sig['domain_boost']:.2f}, signals={sig['domain_signals']}")
    print()

# Check feature_flags
print("Feature flags check:")
print(f"  CORE enable_wyckoff: {config_core.get('feature_flags', {}).get('enable_wyckoff')}")
print(f"  CORE enable_smc: {config_core.get('feature_flags', {}).get('enable_smc')}")
print(f"  FULL enable_wyckoff: {config_full.get('feature_flags', {}).get('enable_wyckoff')}")
print(f"  FULL enable_smc: {config_full.get('feature_flags', {}).get('enable_smc')}")
print()

# Check if domain features have any True values in the data
print("Domain feature signal counts (in test data):")
for feat in ['wyckoff_spring_a', 'wyckoff_sc', 'tf1h_bos_bullish', 'tf4h_bos_bullish', 'smc_demand_zone']:
    if feat in df.columns:
        count = (df[feat] == True).sum() if df[feat].dtype == bool else (df[feat] > 0).sum()
        print(f"  {feat}: {count} signals")

print()
print("=" * 80)
print("DIAGNOSIS:")
if len(core_signals) == len(full_signals) and all(c['domain_boost'] == f['domain_boost'] for c, f in zip(core_signals, full_signals)):
    print("❌ BUG CONFIRMED: Core and Full have IDENTICAL domain_boost values!")
    print("   This means feature_flags are NOT controlling domain engine activation.")
else:
    print("✅ Domain engines ARE affecting scores differently!")
print("=" * 80)
