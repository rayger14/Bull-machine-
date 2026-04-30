#!/usr/bin/env python3
"""Validate portfolio on 2024 data (out-of-sample)."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from engine.backtesting.engine import BacktestEngine
from engine.models.archetype_model import ArchetypeModel

def test_period(year, config_path, df):
    """Test portfolio for a given year."""
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'

    archetypes = [
        ('B', 'Order Block'),
        ('K', 'Wick Trap'),
        ('A', 'Spring/UTAD'),
        ('S1', 'Liquidity Vacuum')
    ]

    results_list = []
    total_pnl = 0

    for arch_id, arch_name in archetypes:
        model = ArchetypeModel(
            config_path=str(config_path),
            archetype_name=arch_id,
            name=f"{year}_{arch_id}"
        )

        engine = BacktestEngine(
            model=model,
            data=df,
            initial_capital=10000.0,
            commission_pct=0.001
        )

        try:
            results = engine.run(start=start_date, end=end_date, verbose=False)

            results_list.append({
                'archetype': arch_id,
                'trades': results.total_trades,
                'win_rate': results.win_rate,
                'pnl': results.total_pnl,
                'pf': results.profit_factor if results.profit_factor else 0
            })

            total_pnl += results.total_pnl
        except Exception as e:
            print(f"   Error testing {arch_id}: {e}")
            results_list.append({
                'archetype': arch_id,
                'trades': 0,
                'win_rate': 0,
                'pnl': 0,
                'pf': 0
            })

    return results_list, total_pnl

def main():
    print("=" * 80)
    print("2024 VALIDATION (Out-of-Sample)")
    print("=" * 80)

    df = pd.read_parquet(PROJECT_ROOT / 'data/btcusd_1h_features.parquet')
    config_path = PROJECT_ROOT / 'configs/test_optimized_no_funding.json'

    # Check 2024 data availability
    df_2024 = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
    print(f"\n2024 data: {len(df_2024):,} bars")

    if len(df_2024) < 1000:
        print("⚠️  Limited 2024 data - results may not be representative")
    print()

    # Test 2023 (in-sample, for comparison)
    print("Testing 2023 (in-sample baseline)...")
    results_2023, pnl_2023 = test_period('2023', config_path, df)

    # Test 2024 (out-of-sample)
    print("Testing 2024 (out-of-sample validation)...")
    results_2024, pnl_2024 = test_period('2024', config_path, df)

    print("\n" + "=" * 80)
    print("COMPARISON: 2023 vs 2024")
    print("=" * 80)

    print(f"\n{'Archetype':<12} {'2023 Trades':<15} {'2024 Trades':<15} {'2023 PnL':<15} {'2024 PnL':<15}")
    print("-" * 72)

    for r23, r24 in zip(results_2023, results_2024):
        arch = r23['archetype']
        print(f"{arch:<12} {r23['trades']:<15} {r24['trades']:<15} ${r23['pnl']:<14.2f} ${r24['pnl']:<14.2f}")

    total_trades_2023 = sum(r['trades'] for r in results_2023)
    total_trades_2024 = sum(r['trades'] for r in results_2024)

    print("-" * 72)
    print(f"{'TOTAL':<12} {total_trades_2023:<15} {total_trades_2024:<15} ${pnl_2023:<14.2f} ${pnl_2024:<14.2f}")

    print("\n" + "=" * 80)
    print("OVERFITTING ANALYSIS")
    print("=" * 80)

    # Calculate degradation
    pnl_degradation = (pnl_2024 - pnl_2023) / pnl_2023 * 100 if pnl_2023 != 0 else 0

    print(f"\nPortfolio PnL:")
    print(f"   2023 (in-sample): ${pnl_2023:.2f} (33.6% return)")
    print(f"   2024 (out-of-sample): ${pnl_2024:.2f} ({(pnl_2024/10000)*100:.1f}% return)")
    print(f"   Change: ${pnl_2024 - pnl_2023:+.2f} ({pnl_degradation:+.1f}%)")

    print(f"\nTrade Count:")
    print(f"   2023: {total_trades_2023}")
    print(f"   2024: {total_trades_2024}")
    print(f"   Change: {total_trades_2024 - total_trades_2023:+} ({((total_trades_2024 - total_trades_2023)/total_trades_2023*100):+.1f}%)")

    # Archetype-by-archetype degradation
    print(f"\nPer-Archetype Degradation:")
    for r23, r24 in zip(results_2023, results_2024):
        if r23['pnl'] != 0:
            deg = (r24['pnl'] - r23['pnl']) / abs(r23['pnl']) * 100
            print(f"   {r23['archetype']}: {deg:+.1f}% ({r23['pnl']:.0f} → {r24['pnl']:.0f})")

    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    # Overfitting thresholds
    if pnl_degradation >= -10:
        print(f"\n✅ NO OVERFITTING: 2024 performance {pnl_degradation:+.1f}% vs 2023")
        print("   Strategy generalizes well to out-of-sample data")
    elif pnl_degradation >= -30:
        print(f"\n⚠️  MODERATE DEGRADATION: {pnl_degradation:+.1f}% on 2024 data")
        print("   Some overfitting present but acceptable")
        print("   Consider walk-forward validation to refine parameters")
    else:
        print(f"\n❌ SIGNIFICANT OVERFITTING: {pnl_degradation:+.1f}% degradation")
        print("   Strategy may be overfit to 2023 market conditions")
        print("   Recommend parameter simplification or longer training period")

    # Individual archetype flags
    print("\nArchetype-Specific Issues:")
    for r23, r24 in zip(results_2023, results_2024):
        if r23['pnl'] > 100 and r24['pnl'] < 0:
            print(f"   ⚠️  {r23['archetype']}: Profitable in 2023 but negative in 2024")
        elif r24['trades'] < r23['trades'] * 0.5:
            print(f"   ⚠️  {r23['archetype']}: Trade count dropped >50% ({r23['trades']} → {r24['trades']})")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if pnl_degradation >= -10:
        print("\n✅ DEPLOY TO PRODUCTION")
        print("   Current configuration validated on out-of-sample data")
    elif pnl_degradation >= -30:
        print("\n⚠️  CONDITIONAL DEPLOY")
        print("   Monitor closely in paper trading")
        print("   Consider walk-forward optimization for robustness")
    else:
        print("\n❌ DO NOT DEPLOY")
        print("   Significant overfitting detected")
        print("   Revise strategy or collect more training data")

if __name__ == '__main__':
    main()
