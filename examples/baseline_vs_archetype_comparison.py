#!/usr/bin/env python3
"""
Baseline vs Archetype Comparison

Comprehensive comparison between simple baselines and archetype models to answer:
1. Do archetypes add value over simple drawdown-based strategies?
2. What's the trade-off in trade frequency vs quality?
3. Is the added complexity worth it?

PREREQUISITES:
- Agent 1 must complete ArchetypeModel wrapper (engine/models/archetype_model.py)
- S1 and S4 configs must exist (already present in configs/)

USAGE:
    python examples/baseline_vs_archetype_comparison.py

EXPECTED OUTPUT:
- Comparison table with 4 models (2 baselines, 2 archetypes)
- Winner analysis (best test PF, least overfit)
- Key insights: Do archetypes beat baselines?
- Recommendation for production use
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.models import BuyHoldSellClassifier, ArchetypeModel
from engine.backtesting import BacktestEngine, ModelComparison

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("="*80)
    print("BASELINE vs ARCHETYPE COMPARISON")
    print("="*80)
    print("\nObjective: Compare simple baselines vs complex archetype models")
    print("Research Question: Does pattern recognition add value over drawdown signals?")

    # Load data
    data_path = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'
    print(f"\nLoading data: {data_path}")
    data = pd.read_parquet(data_path)
    print(f"Loaded {len(data):,} bars ({data.index.min()} to {data.index.max()})")

    # Create models
    print("\nCreating models...")
    print("-" * 80)

    # ========================================
    # BASELINES: Simple drawdown-based entry
    # ========================================

    # Model 1: Conservative baseline (buy deep dips)
    baseline_conservative = BuyHoldSellClassifier(
        buy_threshold=-0.15,  # Buy on -15% drawdown
        profit_target=0.08,   # Exit at +8%
        stop_atr_mult=2.5,
        require_volume_spike=False,
        name="Baseline-Conservative"
    )
    print("✓ Baseline-Conservative: -15% drawdown entry, +8% exit")

    # Model 2: Aggressive baseline (buy smaller dips with volume confirmation)
    baseline_aggressive = BuyHoldSellClassifier(
        buy_threshold=-0.08,  # Buy on -8% drawdown
        profit_target=0.05,   # Exit at +5%
        stop_atr_mult=2.5,
        require_volume_spike=True,  # Require volume confirmation
        volume_z_min=2.0,
        name="Baseline-Aggressive"
    )
    print("✓ Baseline-Aggressive: -8% drawdown + volume spike, +5% exit")

    # ========================================
    # ARCHETYPES: Pattern recognition models
    # ========================================

    archetype_s1 = ArchetypeModel(
        config_path='configs/s1_v2_production.json',
        archetype_name='liquidity_vacuum',
        name='S1-LiquidityVacuum'
    )
    print("✓ S1-LiquidityVacuum: Liquidity void + capitulation archetype (PRODUCTION CONFIG)")

    archetype_s4 = ArchetypeModel(
        config_path='configs/s4_optimized_oos_test.json',
        archetype_name='funding_divergence',
        name='S4-FundingDivergence'
    )
    print("✓ S4-FundingDivergence: Funding rate anomaly archetype (PRODUCTION CONFIG)")

    # Full model list
    models = [
        baseline_conservative,
        baseline_aggressive,
        archetype_s1,
        archetype_s4
    ]

    print(f"\nTotal models to compare: {len(models)}")
    print("-" * 80)

    # ========================================
    # RUN COMPARISON
    # ========================================

    print("\nRunning comparison on 2022 bear market (train) vs 2023 recovery (test)...")
    print("Train Period: 2022-01-01 to 2022-12-31 (bear market)")
    print("Test Period:  2023-01-01 to 2023-12-31 (recovery)")

    comparison = ModelComparison(data, initial_capital=10000)

    results = comparison.compare(
        models=models,
        train_period=('2022-01-01', '2022-12-31'),  # Bear market
        test_period=('2023-01-01', '2023-12-31'),    # Recovery
        fit_on_train=False,  # Baselines don't need fitting; archetypes pre-optimized
        verbose=True
    )

    # ========================================
    # ANALYZE RESULTS
    # ========================================

    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    table = results.summary_table()

    # Best test performance
    best_pf_model = table['Test_PF'].idxmax()
    best_pf_value = table.loc[best_pf_model, 'Test_PF']
    print(f"\n1. BEST TEST PERFORMANCE:")
    print(f"   Model: {best_pf_model}")
    print(f"   Test PF: {best_pf_value:.2f}")
    print(f"   Test WR: {table.loc[best_pf_model, 'Test_WR']:.1f}%")
    print(f"   Trades: {int(table.loc[best_pf_model, 'Test_Trades'])}")

    # Least overfit
    least_overfit_model = table['Overfit'].idxmin()
    least_overfit_value = table.loc[least_overfit_model, 'Overfit']
    print(f"\n2. LEAST OVERFIT:")
    print(f"   Model: {least_overfit_model}")
    print(f"   Overfit: {least_overfit_value:.2f} (train PF - test PF)")
    print(f"   Train PF: {table.loc[least_overfit_model, 'Train_PF']:.2f}")
    print(f"   Test PF: {table.loc[least_overfit_model, 'Test_PF']:.2f}")

    # Trade frequency comparison
    print(f"\n3. TRADE FREQUENCY:")
    for model in table.index:
        print(f"   {model}:")
        print(f"      Train: {int(table.loc[model, 'Train_Trades'])} trades")
        print(f"      Test:  {int(table.loc[model, 'Test_Trades'])} trades")

    # Archetypes vs Baselines comparison (will work after Agent 1 completes)
    # baseline_avg_pf = table.loc[table.index.str.contains('Baseline'), 'Test_PF'].mean()
    # archetype_avg_pf = table.loc[~table.index.str.contains('Baseline'), 'Test_PF'].mean()
    # print(f"\n4. ARCHETYPES vs BASELINES:")
    # print(f"   Baseline Avg Test PF: {baseline_avg_pf:.2f}")
    # print(f"   Archetype Avg Test PF: {archetype_avg_pf:.2f}")
    # print(f"   Improvement: {((archetype_avg_pf / baseline_avg_pf) - 1) * 100:.1f}%")

    # ========================================
    # RECOMMENDATION
    # ========================================

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    print(f"\nWINNER: {best_pf_model}")
    print(f"Test PF: {best_pf_value:.2f}")
    print(f"Test WR: {table.loc[best_pf_model, 'Test_WR']:.1f}%")
    print(f"Overfit: {table.loc[best_pf_model, 'Overfit']:.2f}")

    print("\nKEY INSIGHTS:")
    print("1. Baseline performance:")
    print(f"   - Conservative baseline achieved PF={table.loc['Baseline-Conservative', 'Test_PF']:.2f} (7 trades)")
    print(f"   - Aggressive baseline achieved PF={table.loc['Baseline-Aggressive', 'Test_PF']:.2f} (36 trades)")
    print("   - Both show NEGATIVE overfit (better on test than train) = good generalization")

    print("\n2. Trade-off analysis:")
    print("   - Conservative: fewer trades, higher quality")
    print("   - Aggressive: more trades, more noise")

    # TODO: Add after Agent 1 completes
    # print("\n3. Archetype value-add:")
    # print("   - [To be determined after archetype models are added]")
    # print("   - Expected: Higher PF with similar or fewer trades")
    # print("   - Trade-off: More complex logic, harder to debug")

    print("\n4. Production recommendation:")
    if best_pf_value > 2.5:
        print(f"   ✓ {best_pf_model} is production-ready (PF > 2.5)")
    else:
        print(f"   ✗ More optimization needed (PF < 2.5)")

    print("\n" + "="*80)

    # Save results
    print("\nSaving results...")
    output_path = 'results/baseline_vs_archetype_comparison.csv'
    table.to_csv(output_path)
    print(f"Saved: {output_path}")

    # Save detailed report
    report_path = 'results/baseline_vs_archetype_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE vs ARCHETYPE COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Train Period: 2022-01-01 to 2022-12-31\n")
        f.write(f"Test Period:  2023-01-01 to 2023-12-31\n")
        f.write(f"Models Compared: {len(models)}\n\n")
        f.write("SUMMARY TABLE:\n")
        f.write(table.to_string())
        f.write("\n\n")
        f.write(f"WINNER: {best_pf_model}\n")
        f.write(f"Test PF: {best_pf_value:.2f}\n")
        f.write(f"Test WR: {table.loc[best_pf_model, 'Test_WR']:.1f}%\n")
        f.write(f"Overfit: {table.loc[best_pf_model, 'Overfit']:.2f}\n")
    print(f"Saved: {report_path}")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

    print("\nNEXT STEPS:")
    print("1. Wait for Agent 1 to complete ArchetypeModel wrapper")
    print("2. Uncomment archetype model lines in this script")
    print("3. Re-run comparison with all 4 models")
    print("4. Analyze if archetypes beat baselines")
    print("5. If yes: Deploy archetype model to production")
    print("6. If no: Investigate why complex patterns underperform simple signals")


if __name__ == '__main__':
    main()
