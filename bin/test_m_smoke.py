#!/usr/bin/env python3
"""Minimal smoke test for M only"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext

def main():
    # Create minimal config (same as smoke test)
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
    test_df = df['2023-01-01':'2023-04-01']

    print(f"Testing M on {len(test_df)} bars (Q1 2023)")

    # Create logic
    logic = ArchetypeLogic(config)

    signals = 0
    errors = []

    for idx, (ts, row) in enumerate(test_df.head(500).iterrows()):
        try:
            # Create context (SAME as smoke test)
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

            # Call _check_M
            matched, score, meta = logic._check_M(ctx)

            if matched and score > 0:
                signals += 1
                if signals <= 5:
                    print(f"Signal #{signals} at {ts}: score={score:.3f}")

        except Exception as e:
            errors.append(str(e))
            if len(errors) <= 3:
                print(f"ERROR at {ts}: {e}")

    print(f"\nResults:")
    print(f"  Signals: {signals}")
    print(f"  Errors: {len(errors)}")

    if signals == 0 and len(errors) == 0:
        print("\n❌ NO ERRORS BUT NO SIGNALS!")
        print("   → Pattern logic is blocking signals (check thresholds)")

        # Debug: Test base pattern directly
        print("\nTesting base pattern (_pattern_M) directly:")
        base_signals = 0
        for idx, (ts, row) in enumerate(test_df.head(500).iterrows()):
            ctx = RuntimeContext(
                ts=ts,
                row=row,
                regime_probs={'neutral': 1.0},
                regime_label='neutral',
                adapted_params={},
                thresholds=config.get('archetypes', {}).get('thresholds', {}),
                metadata={'feature_flags': config.get('feature_flags', {})}
            )
            base_result = logic._pattern_M(ctx)
            if base_result:
                base_signals += 1

        print(f"  Base pattern fires: {base_signals} times")
        if base_signals > 0:
            print("  → Base pattern works, but _check_M blocks it")
            print("  → Check fusion threshold or domain engine vetoes")

if __name__ == '__main__':
    main()
