#!/usr/bin/env python3
"""Test M on exact smoke test period (2023H2)"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext

def main():
    # Same config as smoke test
    config = {
        'version': 'smoke_test_M',
        'use_archetypes': True,
        'adaptive_fusion': False,
        'regime_classifier': {
            'zero_fill_missing': True,
            'regime_override': {'default': 'neutral'}
        },
        'ml_filter': {'enabled': False},
        'fusion': {
            'entry_threshold_confidence': 0.0,
            'weights': {'wyckoff': 0.0, 'liquidity': 0.0, 'momentum': 0.0, 'smc': 0.0}
        },
        'archetypes': {
            'use_archetypes': True,
            'max_trades_per_day': 0,
            'thresholds': {'min_liquidity': 0.0},
        },
        'feature_flags': {
            'enable_wyckoff': True,
            'enable_smc': True,
            'enable_temporal': True,
            'enable_hob': True,
            'enable_fusion': True,
            'enable_macro': True,
        },
        'temporal_fusion': {'enabled': True, 'use_confluence': True},
        'wyckoff_events': {'enabled': True, 'log_events': False},
        'smc_engine': {'enabled': True},
        'hob_engine': {'enabled': True},
        'enable_M': True,
    }

    # Load data
    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    # EXACT period from smoke test
    test_df = df['2023-08-01':'2023-12-31']
    print(f"Testing M on {len(test_df)} bars (2023-08-01 to 2023-12-31)")
    print(f"Expected from smoke test: 0 signals")
    print(f"Expected from diagnostic: 52-54 signals\n")

    # Create logic
    logic = ArchetypeLogic(config)

    signals = []

    for idx, (ts, row) in enumerate(test_df.iterrows()):
        ctx = RuntimeContext(
            ts=ts,
            row=row,
            regime_probs={'neutral': 1.0},
            regime_label='neutral',
            adapted_params={},
            thresholds=config.get('archetypes', {}).get('thresholds', {}),
            metadata={
                'feature_flags': config.get('feature_flags', {}),
                'df': df,
                'index': ts,
            }
        )

        matched, score, meta = logic._check_M(ctx)

        if matched and score > 0:
            signals.append((ts, score))

    print(f"ACTUAL RESULT: {len(signals)} signals")

    if len(signals) > 0:
        print("\nFirst 10 signals:")
        for ts, score in signals[:10]:
            print(f"  {ts}: score={score:.3f}")

        print(f"\n✅ PATTERN M WORKS! {len(signals)} signals detected")
        print(f"   → Smoke test issue is elsewhere (likely test harness problem)")
    else:
        print("\n❌ ZERO SIGNALS (matches smoke test)")
        print("   → Need to debug WHY it works standalone but not in smoke test")

if __name__ == '__main__':
    main()
