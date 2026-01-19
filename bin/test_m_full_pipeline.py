#!/usr/bin/env python3
"""Test full M pipeline to find where signals are blocked"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext

def main():
    print("=" * 80)
    print("PATTERN M FULL PIPELINE TEST")
    print("=" * 80)

    # Load data
    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    print(f"\nLoaded {len(df)} bars")

    # Find bars that should pass base pattern
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    base_matches = df[(df['atr_pct'] < 0.50) & (df['tf4h_bos_bullish'] == True)]
    print(f"Base pattern matches: {len(base_matches)}")

    # Test on Q1 2023
    q1_2023 = df['2023-01-01':'2023-04-01']
    q1_base = q1_2023[(q1_2023['atr_pct'] < 0.50) & (q1_2023['tf4h_bos_bullish'] == True)]
    print(f"\nQ1 2023 base matches: {len(q1_base)}")

    if len(q1_base) == 0:
        print("No base matches in Q1 2023, trying 2023H2...")
        test_df = df['2023-08-01':'2023-12-31']
        test_base = test_df[(test_df['atr_pct'] < 0.50) & (test_df['tf4h_bos_bullish'] == True)]
        print(f"2023H2 base matches: {len(test_base)}")
    else:
        test_df = q1_2023
        test_base = q1_base

    if len(test_base) == 0:
        print("No base matches found in any test regime!")
        return

    # Create config
    config = {
        'thresholds': {
            'coil_break': {
                'atr_pct_max': 0.50,
                'fusion_threshold': 0.40
            }
        },
        'feature_flags': {}
    }

    logic = ArchetypeLogic(config)

    # Test first 10 base matches
    print("\n" + "=" * 80)
    print("TESTING FIRST 10 BASE MATCHES THROUGH FULL PIPELINE")
    print("=" * 80)

    base_pass = 0
    fusion_blocked = 0
    domain_blocked = 0
    final_pass = 0

    for idx, row in test_base.head(10).iterrows():
        print(f"\n{idx} (close={row['close']:.2f}, atr_pct={row['atr_pct']:.3f}%):")

        # Create context (row must be pandas Series, not dict)
        ctx = RuntimeContext(
            ts=idx,
            row=row,  # Pass pandas Series, not dict
            regime_probs={'neutral': 1.0},
            regime_label='neutral',
            adapted_params={},
            thresholds=config.get('thresholds', {}),
            metadata={'feature_flags': {}}
        )

        # Test base pattern
        try:
            base_result = logic._pattern_M(ctx)
            if base_result:
                matched, score, tags = base_result
                print(f"  ✅ Base pattern: score={score:.3f}, tags={tags}")
                base_pass += 1

                # Test full check
                final_matched, final_score, meta = logic._check_M(ctx)

                if final_matched:
                    print(f"  ✅ FINAL PASS: score={final_score:.3f}")
                    final_pass += 1
                else:
                    reason = meta.get('reason', 'unknown')
                    print(f"  ❌ BLOCKED: {reason}")

                    if 'fusion' in reason.lower():
                        fusion_blocked += 1
                        print(f"     base_score={meta.get('base_score', 'N/A')}")
                        print(f"     final_score={meta.get('final_score', 'N/A')}")
                        print(f"     fusion_threshold={meta.get('fusion_threshold', 'N/A')}")
                    else:
                        domain_blocked += 1
                        print(f"     meta={meta}")
            else:
                print(f"  ❌ Base pattern: FAILED")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Base pattern pass: {base_pass}/10")
    print(f"Fusion blocked: {fusion_blocked}/10")
    print(f"Domain blocked: {domain_blocked}/10")
    print(f"Final pass: {final_pass}/10")

    if fusion_blocked > 0:
        print("\n❌ PRIMARY ISSUE: Fusion threshold blocking signals")
        print("   → Base pattern fires but score too low after domain engines")
        print("   → FIX: Lower fusion_threshold from 0.40 to 0.30")
    elif domain_blocked > 0:
        print("\n❌ PRIMARY ISSUE: Domain engine vetos blocking signals")
        print("   → Check domain engine logic for M archetype")

    # Test all regimes
    print("\n" + "=" * 80)
    print("FULL REGIME TEST")
    print("=" * 80)

    regimes = [
        ("Q1 2023", "2023-01-01", "2023-04-01"),
        ("2022 Crisis", "2022-01-01", "2022-12-31"),
        ("2023H2", "2023-08-01", "2023-12-31")
    ]

    for name, start, end in regimes:
        regime_df = df[start:end]
        regime_base = regime_df[(regime_df['atr_pct'] < 0.50) & (regime_df['tf4h_bos_bullish'] == True)]

        print(f"\n{name}:")
        print(f"  Total bars: {len(regime_df)}")
        print(f"  Base matches: {len(regime_base)}")

        if len(regime_base) == 0:
            print(f"  Final signals: 0 (no base matches)")
            continue

        signals = 0
        for idx, row in regime_base.iterrows():
            ctx = RuntimeContext(
                ts=idx,
                row=row,  # Pass pandas Series, not dict
                regime_probs={'neutral': 1.0},
                regime_label='neutral',
                adapted_params={},
                thresholds=config.get('thresholds', {}),
                metadata={'feature_flags': {}}
            )

            try:
                matched, score, meta = logic._check_M(ctx)
                if matched:
                    signals += 1
            except:
                pass

        print(f"  Final signals: {signals} ({signals/len(regime_base)*100:.1f}% pass rate)")

if __name__ == '__main__':
    main()
