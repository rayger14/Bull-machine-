#!/usr/bin/env python3
"""
Model Comparison Demo

Demonstrates clean separation of concerns:
1. Model abstraction (baseline vs archetype)
2. Backtesting framework (model-agnostic)
3. Train/test comparison

Usage:
    python examples/model_comparison_demo.py
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.models import BuyHoldSellClassifier
from engine.backtesting import BacktestEngine, ModelComparison

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    print("="*80)
    print("MODEL COMPARISON DEMO")
    print("="*80)

    # Load data
    data_path = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'
    print(f"\nLoading data: {data_path}")
    data = pd.read_parquet(data_path)
    print(f"Loaded {len(data):,} bars ({data.index.min()} to {data.index.max()})")

    # Create models
    print("\nCreating models...")

    # Model 1: Conservative baseline
    baseline_conservative = BuyHoldSellClassifier(
        buy_threshold=-0.15,  # Buy on -15% drawdown
        profit_target=0.08,   # Exit at +8%
        name="Baseline-Conservative"
    )

    # Model 2: Aggressive baseline
    baseline_aggressive = BuyHoldSellClassifier(
        buy_threshold=-0.08,  # Buy on -8% drawdown
        profit_target=0.05,   # Exit at +5%
        require_volume_spike=True,
        name="Baseline-Aggressive"
    )

    # TODO: Model 3: Archetype S1 (when wrapper is implemented)
    # archetype_s1 = ArchetypeModel(config='configs/s1_optimized.json', name='S1-Optimized')

    models = [baseline_conservative, baseline_aggressive]

    # Run comparison
    print("\nRunning comparison...")
    comparison = ModelComparison(data, initial_capital=10000)

    results = comparison.compare(
        models=models,
        train_period=('2022-01-01', '2022-12-31'),
        test_period=('2023-01-01', '2023-12-31'),
        fit_on_train=False,  # Baselines don't need fitting
        verbose=True
    )

    # Results are printed by comparison.compare(verbose=True)

    # Save results
    print("\nSaving results...")
    table = results.summary_table()
    table.to_csv('results/model_comparison_demo.csv')
    print("Saved: results/model_comparison_demo.csv")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Implement ArchetypeModel wrapper around logic_v2_adapter")
    print("2. Add archetype models to comparison")
    print("3. Implement walk-forward validation")
    print("4. Add more advanced metrics (Sharpe, MDD, etc.)")


if __name__ == '__main__':
    main()
