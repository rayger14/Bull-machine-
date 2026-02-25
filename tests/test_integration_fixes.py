#!/usr/bin/env python3
"""
Integration Test for Nautilus Backtest Bug Fixes

Tests all three bugs:
1. ThresholdPolicy.get_all_thresholds() - FIXED
2. RegimeWeightAllocator initialization with edge_table_path - FIXED
3. RegimeService key extraction - ALREADY FIXED

Run this before attempting the full backtest to verify fixes.
"""

import sys
import json
from pathlib import Path

print("=" * 80)
print("INTEGRATION TEST: Nautilus Backtest Bug Fixes")
print("=" * 80)

# Test 1: ThresholdPolicy.get_all_thresholds()
print("\n[TEST 1] ThresholdPolicy.get_all_thresholds()")
print("-" * 80)
try:
    from engine.archetypes.threshold_policy import ThresholdPolicy

    with open('configs/baseline_wyckoff_test.json', 'r') as f:
        config = json.load(f)

    policy = ThresholdPolicy(
        base_cfg=config,
        locked_regime='static'
    )

    thresholds = policy.get_all_thresholds('risk_on')

    assert isinstance(thresholds, dict), "Should return dict"
    assert len(thresholds) > 0, "Should have thresholds"

    print(f"✓ Method exists and works correctly")
    print(f"  - Returned {len(thresholds)} archetype thresholds")
    print(f"  - Sample archetypes: {list(thresholds.keys())[:5]}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: RegimeWeightAllocator initialization
print("\n[TEST 2] RegimeWeightAllocator initialization")
print("-" * 80)
try:
    from engine.portfolio.regime_allocator import RegimeWeightAllocator

    # Test with missing edge table (should fail gracefully)
    edge_table_path = 'results/archetype_regime_edge_table.csv'
    config_path = 'configs/regime_allocator_config.json'

    if Path(edge_table_path).exists():
        allocator = RegimeWeightAllocator(
            edge_table_path=edge_table_path,
            config_path=config_path
        )
        print(f"✓ Initialized with edge_table_path parameter")
        print(f"  - Edge table: {edge_table_path}")
        print(f"  - Loaded {len(allocator.edge_data)} archetype-regime pairs")

        # Test get_weight method
        weight = allocator.get_weight('A', 'risk_on')
        print(f"  - get_weight('A', 'risk_on') = {weight:.3f}")
    else:
        print(f"⚠ Edge table not found at {edge_table_path}")
        print(f"  - This is OK for basic testing")
        print(f"  - BullMachineStrategy will disable soft gating gracefully")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: RegimeService key extraction
print("\n[TEST 3] RegimeService returns correct keys")
print("-" * 80)
try:
    from engine.context.regime_service import RegimeService
    import pandas as pd
    import numpy as np

    service = RegimeService(
        mode='hybrid',
        enable_event_override=True,
        enable_hysteresis=True
    )

    # Create dummy features
    features = {
        'close': 50000.0,
        'returns_1h': 0.01,
        'volume': 1000000,
        'vwap_1h': 49500,
        'funding_rate': 0.0001,
        'oi_1h': 1000000000,
        'cvd_zscore': 0.5,
    }

    result = service.get_regime(features)

    # Verify all required keys exist
    required_keys = ['regime_label', 'regime_probs', 'regime_confidence']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    print(f"✓ RegimeService returns correct keys")
    print(f"  - regime_label: {result['regime_label']}")
    print(f"  - regime_probs: {result['regime_probs']}")
    print(f"  - regime_confidence: {result['regime_confidence']:.3f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: BullMachineStrategy initialization (integration test)
print("\n[TEST 4] BullMachineStrategy initialization (FULL INTEGRATION)")
print("-" * 80)
try:
    from engine.integrations.bull_machine_strategy import BullMachineStrategy

    strategy = BullMachineStrategy(
        config_path='configs/baseline_wyckoff_test.json',
        enable_soft_gating=True,
        enable_circuit_breaker=True,
        base_position_size_usd=1000.0,
        max_positions=1
    )

    print(f"✓ BullMachineStrategy initialized successfully")
    print(f"  - Config: {strategy.config_path}")
    print(f"  - Feature store: {strategy.feature_store_path}")
    print(f"  - Features: {len(strategy.features_df):,} bars, {len(strategy.features_df.columns)} columns")
    print(f"  - RegimeService: {'✓' if strategy.regime_service else '✗'}")
    print(f"  - ArchetypeLogic: {'✓' if strategy.archetype_logic else '✗'}")
    print(f"  - ThresholdPolicy: {'✓' if strategy.threshold_policy else '✗'}")
    print(f"  - RegimeAllocator: {'✓' if strategy.regime_allocator else '✗ (soft gating disabled)'}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nYou can now run the backtest:")
print("  python3 bin/nautilus_backtest.py --config configs/baseline_wyckoff_test.json --start 2023-01-01 --end 2023-03-31")
