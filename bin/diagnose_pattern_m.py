#!/usr/bin/env python3
"""
Diagnostic script for Archetype M (Confluence Breakout / Coil Break)

Investigates why M produces 0 signals across all market regimes.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext

def main():
    print("=" * 80)
    print("ARCHETYPE M (COIL BREAK) DIAGNOSTIC")
    print("=" * 80)

    # Load feature data
    feature_path = Path(__file__).parent.parent / "data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet"
    print(f"\n1. Loading features from: {feature_path}")

    if not feature_path.exists():
        print(f"❌ Feature file not found: {feature_path}")
        return

    df = pd.read_parquet(feature_path)
    print(f"✅ Loaded {len(df)} rows")

    # Test on all 3 regimes
    regimes = [
        ("Q1 2023 Bull Recovery", "2023-01-01", "2023-04-01"),
        ("2022 Crisis", "2022-01-01", "2022-12-31"),
        ("2023H2 Mixed", "2023-08-01", "2023-12-31")
    ]

    # Create minimal config for testing
    config = {
        'thresholds': {
            'coil_break': {
                'atr_pct_max': 0.50,  # Current default
                'fusion_threshold': 0.40
            }
        },
        'feature_flags': {}
    }
    adapter = ArchetypeLogic(config)

    # Check feature availability FIRST
    print("\n" + "=" * 80)
    print("2. FEATURE AVAILABILITY CHECK")
    print("=" * 80)

    required_features = ['atr_14', 'atr_20', 'tf4h_bos_bullish', 'close']
    optional_features = ['atr_percentile', 'poc_distance', 'volume_zscore']

    print("\nRequired Features:")
    for feat in required_features:
        exists = feat in df.columns
        status = "✅" if exists else "❌"
        if exists:
            null_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  {status} {feat}: {null_pct:.1f}% null")
        else:
            print(f"  {status} {feat}: MISSING")

    print("\nOptional Features:")
    for feat in optional_features:
        exists = feat in df.columns
        status = "✅" if exists else "⚠️"
        if exists:
            null_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  {status} {feat}: {null_pct:.1f}% null")
        else:
            print(f"  {status} {feat}: MISSING (fallback logic should handle)")

    # Analyze pattern M logic requirements
    print("\n" + "=" * 80)
    print("3. PATTERN M LOGIC REQUIREMENTS")
    print("=" * 80)
    print("""
From _pattern_M() code:
  1. atr = g(r, 'atr_14', g(r, 'atr_20', 0))
  2. atr_pct = (atr / close) * 100
  3. atr_pct_max = get_threshold('coil_break', 'atr_pct_max', 0.50)  # Default: 0.50%
  4. tf4h_bos_bullish = g(r, 'tf4h_bos_bullish', False)
  5. Pattern fires if: atr_pct < atr_pct_max AND tf4h_bos_bullish

HYPOTHESIS: Either atr_pct is never < 0.50%, OR tf4h_bos_bullish is always False
""")

    # Statistical analysis across all regimes
    print("\n" + "=" * 80)
    print("4. STATISTICAL ANALYSIS ACROSS ALL DATA")
    print("=" * 80)

    # Calculate atr_pct for entire dataset
    if 'atr_14' in df.columns and 'close' in df.columns:
        df_clean = df.dropna(subset=['atr_14', 'close'])
        df_clean = df_clean[df_clean['close'] > 0]

        if len(df_clean) > 0:
            df_clean['atr_pct'] = (df_clean['atr_14'] / df_clean['close']) * 100

            print(f"\nATR Percentage Statistics (n={len(df_clean)}):")
            print(f"  Min:  {df_clean['atr_pct'].min():.3f}%")
            print(f"  25th: {df_clean['atr_pct'].quantile(0.25):.3f}%")
            print(f"  50th: {df_clean['atr_pct'].median():.3f}%")
            print(f"  75th: {df_clean['atr_pct'].quantile(0.75):.3f}%")
            print(f"  Max:  {df_clean['atr_pct'].max():.3f}%")

            # How many bars pass atr_pct < 0.50% gate?
            low_vol_bars = (df_clean['atr_pct'] < 0.50).sum()
            low_vol_pct = (low_vol_bars / len(df_clean)) * 100
            print(f"\n  Bars with atr_pct < 0.50%: {low_vol_bars} ({low_vol_pct:.1f}%)")

            if low_vol_bars == 0:
                print("\n❌ CRITICAL: 0 bars have atr_pct < 0.50%")
                print("   → Threshold 0.50% is TOO STRICT (below minimum volatility)")
                print("   → Need to relax to ~1.5% or ~2.0% to match realistic compression")

    # Check tf4h_bos_bullish availability
    if 'tf4h_bos_bullish' in df.columns:
        bos_true = df['tf4h_bos_bullish'].sum()
        bos_pct = (bos_true / len(df)) * 100
        print(f"\ntf4h_bos_bullish Statistics:")
        print(f"  True count: {bos_true} ({bos_pct:.1f}%)")
        print(f"  False count: {len(df) - bos_true} ({100-bos_pct:.1f}%)")

        if bos_true == 0:
            print("\n❌ CRITICAL: tf4h_bos_bullish is ALWAYS False")
    else:
        print("\n❌ CRITICAL: tf4h_bos_bullish feature MISSING")

    # Test pattern matching on each regime
    print("\n" + "=" * 80)
    print("5. PATTERN MATCHING TEST (SAMPLE)")
    print("=" * 80)

    for regime_name, start_date, end_date in regimes:
        regime_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)].copy()

        if len(regime_df) == 0:
            print(f"\n{regime_name}: No data")
            continue

        print(f"\n{regime_name} ({start_date} to {end_date}):")
        print(f"  Total bars: {len(regime_df)}")

        # Sample first 500 bars
        sample_size = min(500, len(regime_df))
        signals = 0
        blocks = []

        for idx, row in regime_df.head(sample_size).iterrows():
            ctx = RuntimeContext(
                row=row,
                metadata={'feature_flags': {}},
                thresholds={}
            )

            try:
                matched, score, meta = adapter._check_M(ctx)
                if matched:
                    signals += 1
                    print(f"  ✅ Signal at {row.get('timestamp')}: score={score:.3f}")

                # Track blocking reasons
                if not matched and isinstance(meta, dict) and 'reason' in meta:
                    blocks.append(meta['reason'])

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"  Signals in first {sample_size} bars: {signals}")

        if signals == 0 and blocks:
            from collections import Counter
            block_counts = Counter(blocks)
            print(f"  Blocking reasons (top 5):")
            for reason, count in block_counts.most_common(5):
                pct = (count / sample_size) * 100
                print(f"    - {reason}: {count} ({pct:.1f}%)")

    # Deep dive: Check actual gate failures
    print("\n" + "=" * 80)
    print("6. GATE FAILURE ANALYSIS")
    print("=" * 80)

    # Sample from Q1 2023 (100 bars)
    q1_2023 = df[(df['timestamp'] >= '2023-01-01') & (df['timestamp'] < '2023-04-01')].head(100)

    if len(q1_2023) > 0:
        print(f"\nAnalyzing first 100 bars of Q1 2023:")

        gate1_pass = 0  # atr_pct < 0.50
        gate2_pass = 0  # tf4h_bos_bullish == True
        both_pass = 0

        for idx, row in q1_2023.iterrows():
            atr = row.get('atr_14', row.get('atr_20', 0))
            close = row.get('close', 1)
            atr_pct = (atr / close) * 100 if close > 0 else 999

            tf4h_bos = row.get('tf4h_bos_bullish', False)

            g1 = atr_pct < 0.50
            g2 = tf4h_bos == True

            if g1:
                gate1_pass += 1
            if g2:
                gate2_pass += 1
            if g1 and g2:
                both_pass += 1

        print(f"  Gate 1 (atr_pct < 0.50%): {gate1_pass}/100 bars pass ({gate1_pass}%)")
        print(f"  Gate 2 (tf4h_bos_bullish): {gate2_pass}/100 bars pass ({gate2_pass}%)")
        print(f"  Both gates pass: {both_pass}/100 bars ({both_pass}%)")

        if gate1_pass == 0:
            print("\n❌ ROOT CAUSE: Gate 1 (atr_pct < 0.50%) blocks ALL signals")
            print("   FIX: Increase atr_pct_max threshold to 1.5% or 2.0%")
        elif gate2_pass == 0:
            print("\n❌ ROOT CAUSE: Gate 2 (tf4h_bos_bullish) blocks ALL signals")
            print("   FIX: Make tf4h_bos_bullish optional or check feature availability")
        elif both_pass == 0:
            print("\n❌ ROOT CAUSE: Gates never align (AND gate too strict)")
            print("   FIX: Consider OR logic or relax thresholds")

    print("\n" + "=" * 80)
    print("7. RECOMMENDED FIXES")
    print("=" * 80)

    print("""
Based on diagnostic results, apply ONE of these fixes:

FIX 1: Relax ATR threshold (MOST LIKELY)
  - Current: atr_pct_max = 0.50% (bottom 25% volatility)
  - Problem: Real volatility compression is 1-2%, not 0.5%
  - Solution: Change default to 1.5% or 2.0%
  - Code: line 2106 in logic_v2_adapter.py
    atr_pct_max = context.get_threshold('coil_break', 'atr_pct_max', 1.50)  # Was 0.50

FIX 2: Make tf4h_bos_bullish optional (IF MISSING)
  - Current: Requires tf4h_bos_bullish == True
  - Problem: Feature may be missing or always False
  - Solution: Use OR gate with alternative breakout signal
  - Code: line 2109
    if atr_pct < atr_pct_max and (
        self.g(r, 'tf4h_bos_bullish', False) OR
        self.g(r, 'tf1h_bos_bullish', False) OR
        self.g(r, 'smc_demand_zone', False)
    ):

FIX 3: Use BB Width for volatility (FALLBACK)
  - Problem: ATR-based compression detection may not capture all coils
  - Solution: Add Bollinger Band width as alternative
  - Code: Check if BB width < 0.04 (4% = tight range)

TEST after fix:
  python bin/diagnose_pattern_m.py
  # Should show >0 signals in at least one regime
""")

if __name__ == '__main__':
    main()
