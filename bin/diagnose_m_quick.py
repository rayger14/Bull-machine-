#!/usr/bin/env python3
"""Quick diagnostic for Pattern M gate alignment"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 80)
    print("PATTERN M GATE ALIGNMENT TEST")
    print("=" * 80)

    # Load data
    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    print(f"\nLoaded {len(df)} bars")

    # Calculate atr_pct
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100

    # Test gates
    print("\n" + "=" * 80)
    print("GATE STATISTICS (ALL DATA)")
    print("=" * 80)

    gate1 = df['atr_pct'] < 0.50
    gate2 = df['tf4h_bos_bullish'] == True
    both = gate1 & gate2

    print(f"Gate 1 (atr_pct < 0.50%): {gate1.sum():,} bars ({gate1.sum()/len(df)*100:.1f}%)")
    print(f"Gate 2 (tf4h_bos_bullish): {gate2.sum():,} bars ({gate2.sum()/len(df)*100:.1f}%)")
    print(f"Both gates pass: {both.sum():,} bars ({both.sum()/len(df)*100:.1f}%)")

    if both.sum() == 0:
        print("\n❌ ROOT CAUSE: Gates NEVER align (0 bars pass both)")
        print("   → Gate 1 AND Gate 2 have ZERO overlap")
        print("   → Need to check WHY they don't overlap")

        # Deep dive: When does each gate pass?
        print("\n" + "=" * 80)
        print("TEMPORAL ANALYSIS")
        print("=" * 80)

        # Check if tf4h_bos_bullish only fires during high volatility
        bos_true = df[gate2]
        if len(bos_true) > 0:
            bos_atr_stats = bos_true['atr_pct'].describe()
            print(f"\nWhen tf4h_bos_bullish = True:")
            print(f"  ATR% min:  {bos_atr_stats['min']:.3f}%")
            print(f"  ATR% 25th: {bos_atr_stats['25%']:.3f}%")
            print(f"  ATR% median: {bos_atr_stats['50%']:.3f}%")
            print(f"  ATR% 75th: {bos_atr_stats['75%']:.3f}%")
            print(f"  ATR% max:  {bos_atr_stats['max']:.3f}%")

            # How close do we get?
            closest = bos_true['atr_pct'].min()
            print(f"\n  Closest to 0.50% threshold: {closest:.3f}%")
            if closest > 0.50:
                print(f"  → BOS always fires during HIGH volatility (min {closest:.3f}% > 0.50%)")
                print(f"  → Fix: Increase atr_pct_max to {closest * 1.2:.2f}% (20% buffer)")

        # Check low volatility periods
        low_vol = df[gate1]
        if len(low_vol) > 0:
            bos_during_low_vol = low_vol['tf4h_bos_bullish'].sum()
            print(f"\nWhen atr_pct < 0.50% (low volatility):")
            print(f"  tf4h_bos_bullish = True: {bos_during_low_vol} times")
            if bos_during_low_vol == 0:
                print(f"  → BOS NEVER fires during low volatility periods")
                print(f"  → Pattern impossible: BOS requires volatility, but M requires low volatility")

    else:
        print(f"\n✅ SUCCESS: {both.sum()} bars pass both gates")
        print("\nSample matches:")
        matches = df[both].head(10)
        for idx, row in matches.iterrows():
            print(f"  {idx}: atr_pct={row['atr_pct']:.3f}%, close={row['close']:.2f}")

    # Test regime-specific
    print("\n" + "=" * 80)
    print("REGIME-SPECIFIC ANALYSIS")
    print("=" * 80)

    regimes = [
        ("Q1 2023", "2023-01-01", "2023-04-01"),
        ("2022 Crisis", "2022-01-01", "2022-12-31"),
        ("2023H2", "2023-08-01", "2023-12-31")
    ]

    for name, start, end in regimes:
        regime_df = df[start:end]
        if len(regime_df) == 0:
            continue

        g1 = (regime_df['atr_pct'] < 0.50).sum()
        g2 = (regime_df['tf4h_bos_bullish'] == True).sum()
        both_r = ((regime_df['atr_pct'] < 0.50) & (regime_df['tf4h_bos_bullish'] == True)).sum()

        print(f"\n{name} ({len(regime_df)} bars):")
        print(f"  Gate 1: {g1} ({g1/len(regime_df)*100:.1f}%)")
        print(f"  Gate 2: {g2} ({g2/len(regime_df)*100:.1f}%)")
        print(f"  Both:   {both_r} ({both_r/len(regime_df)*100:.1f}%)")

    # Recommend fix
    print("\n" + "=" * 80)
    print("RECOMMENDED FIX")
    print("=" * 80)

    if both.sum() == 0:
        # Check if relaxing atr_pct would help
        test_thresholds = [0.60, 0.70, 0.80, 1.00, 1.50, 2.00]
        print("\nTesting relaxed ATR thresholds:")
        for threshold in test_thresholds:
            test_gate1 = df['atr_pct'] < threshold
            test_both = test_gate1 & gate2
            print(f"  atr_pct < {threshold:.2f}%: {test_both.sum()} signals ({test_both.sum()/len(df)*100:.2f}%)")

        # Find optimal threshold
        bos_true = df[gate2]
        if len(bos_true) > 0:
            min_atr_during_bos = bos_true['atr_pct'].min()
            optimal = min_atr_during_bos * 1.1  # 10% buffer
            print(f"\n✅ OPTIMAL THRESHOLD: {optimal:.2f}%")
            print(f"   (Minimum ATR during BOS = {min_atr_during_bos:.2f}%)")

            # Test optimal
            optimal_gate1 = df['atr_pct'] < optimal
            optimal_both = optimal_gate1 & gate2
            print(f"   → Would produce {optimal_both.sum()} signals ({optimal_both.sum()/len(df)*100:.2f}%)")

            print(f"\nFIX CODE (line 2106 in logic_v2_adapter.py):")
            print(f"  OLD: atr_pct_max = context.get_threshold('coil_break', 'atr_pct_max', 0.50)")
            print(f"  NEW: atr_pct_max = context.get_threshold('coil_break', 'atr_pct_max', {optimal:.2f})")

if __name__ == '__main__':
    main()
