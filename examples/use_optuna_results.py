#!/usr/bin/env python3
"""
Example: Using Optuna Optimization Results

Demonstrates how to:
1. Load optimized config
2. Extract metadata
3. Compare to baseline
4. Use study object for analysis
"""

import json
import joblib
from pathlib import Path


def load_optimized_config(config_path: str = "configs/auto/best_optuna.json"):
    """Load and display optimized config metadata"""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("="*80)
    print("OPTIMIZED CONFIG METADATA")
    print("="*80)

    if '_optuna_metadata' in config:
        meta = config['_optuna_metadata']

        print(f"\nOptimization Date: {meta.get('optimization_date')}")
        print(f"Asset: {meta.get('asset')}")
        print(f"Period: {meta.get('period')}")
        print(f"Total Trials: {meta.get('trials')}")
        print(f"Successful Trials: {meta.get('successful_trials')}")

        print(f"\nPerformance:")
        print(f"  Profit Factor: {meta.get('profit_factor'):.2f}")
        print(f"  Max Drawdown: {meta.get('max_drawdown'):.1f}%")
        print(f"  Sharpe Ratio: {meta.get('sharpe_ratio'):.2f}")
        print(f"  Total Trades: {meta.get('num_trades')}")

        print(f"\nOptimized Parameters:")
        for param, value in meta.get('optimized_params', {}).items():
            print(f"  {param:20s}: {value:.4f}")
    else:
        print("No metadata found - this may not be an Optuna-optimized config")

    return config


def load_study(study_path: str = "configs/auto/best_optuna_study.pkl"):
    """Load and analyze study object"""

    if not Path(study_path).exists():
        print(f"Study file not found: {study_path}")
        return None

    study = joblib.load(study_path)

    print("\n" + "="*80)
    print("STUDY ANALYSIS")
    print("="*80)

    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Pareto front size: {len(study.best_trials)}")

    print(f"\nTop 3 Pareto-optimal solutions:")
    for i, trial in enumerate(study.best_trials[:3], 1):
        pf, dd = trial.values
        sharpe = trial.user_attrs.get('sharpe_ratio', 0.0)
        trades = trial.user_attrs.get('num_trades', 0)

        print(f"\n  Solution {i}:")
        print(f"    PF: {pf:.2f} | DD: {dd:.1f}% | Sharpe: {sharpe:.2f} | Trades: {trades}")

    return study


def compare_configs(
    baseline_path: str = "configs/profile_default.json",
    optimized_path: str = "configs/auto/best_optuna.json"
):
    """Compare baseline vs optimized config parameters"""

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    with open(optimized_path, 'r') as f:
        optimized = json.load(f)

    print("\n" + "="*80)
    print("CONFIG COMPARISON")
    print("="*80)

    if '_optuna_metadata' not in optimized:
        print("Optimized config missing metadata")
        return

    opt_params = optimized['_optuna_metadata']['optimized_params']

    print(f"\n{'Parameter':<20} {'Baseline':<12} {'Optimized':<12} {'Change':<12}")
    print("-"*60)

    # Compare fusion threshold
    baseline_fusion = baseline.get('fusion', {}).get('entry_threshold_confidence', 'N/A')
    opt_fusion = opt_params.get('fusion_threshold', 'N/A')
    if baseline_fusion != 'N/A' and opt_fusion != 'N/A':
        change = ((opt_fusion - baseline_fusion) / baseline_fusion) * 100
        print(f"{'fusion_threshold':<20} {baseline_fusion:<12.4f} {opt_fusion:<12.4f} {change:+.1f}%")

    # Compare liquidity threshold
    baseline_liq = baseline.get('liquidity', {}).get('min_liquidity', 'N/A')
    opt_liq = opt_params.get('min_liquidity', 'N/A')
    if baseline_liq != 'N/A' and opt_liq != 'N/A':
        change = ((opt_liq - baseline_liq) / baseline_liq) * 100
        print(f"{'min_liquidity':<20} {baseline_liq:<12.4f} {opt_liq:<12.4f} {change:+.1f}%")

    # Show other optimized params
    for param in ['volume_z_min', 'funding_z_min', 'archetype_weight']:
        if param in opt_params:
            print(f"{param:<20} {'N/A':<12} {opt_params[param]:<12.4f} {'New':<12}")


def extract_best_params_for_different_objectives(study_path: str):
    """Extract best parameters for different optimization criteria"""

    study = joblib.load(study_path)

    print("\n" + "="*80)
    print("ALTERNATIVE SOLUTIONS")
    print("="*80)

    # Best PF
    best_pf = max(study.best_trials, key=lambda t: t.values[0])
    print(f"\n1. Highest Profit Factor (PF = {best_pf.values[0]:.2f}):")
    print(f"   Max Drawdown: {best_pf.values[1]:.1f}%")
    print(f"   Parameters: {best_pf.params}")

    # Best DD
    best_dd = min(study.best_trials, key=lambda t: t.values[1])
    print(f"\n2. Lowest Max Drawdown (DD = {best_dd.values[1]:.1f}%):")
    print(f"   Profit Factor: {best_dd.values[0]:.2f}")
    print(f"   Parameters: {best_dd.params}")

    # Best Sharpe (if available)
    pareto_with_sharpe = [t for t in study.best_trials if 'sharpe_ratio' in t.user_attrs]
    if pareto_with_sharpe:
        best_sharpe = max(pareto_with_sharpe, key=lambda t: t.user_attrs['sharpe_ratio'])
        print(f"\n3. Highest Sharpe Ratio (Sharpe = {best_sharpe.user_attrs['sharpe_ratio']:.2f}):")
        print(f"   Profit Factor: {best_sharpe.values[0]:.2f}")
        print(f"   Max Drawdown: {best_sharpe.values[1]:.1f}%")
        print(f"   Parameters: {best_sharpe.params}")


def main():
    """Run all examples"""

    print("\n" + "="*80)
    print("OPTUNA RESULTS USAGE EXAMPLES")
    print("="*80)

    # Example 1: Load optimized config
    try:
        config = load_optimized_config()
    except FileNotFoundError:
        print("\nOptimized config not found. Run optimization first:")
        print("  python3 bin/optuna_thresholds.py --asset ETH --trials 20")
        return

    # Example 2: Load study
    try:
        study = load_study()
    except Exception as e:
        print(f"\nCould not load study: {e}")
        study = None

    # Example 3: Compare configs
    try:
        compare_configs()
    except Exception as e:
        print(f"\nCould not compare configs: {e}")

    # Example 4: Alternative solutions
    if study:
        try:
            extract_best_params_for_different_objectives("configs/auto/best_optuna_study.pkl")
        except Exception as e:
            print(f"\nCould not extract alternatives: {e}")

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nYou can now:")
    print("1. Use configs/auto/best_optuna.json for backtesting")
    print("2. Extract alternative solutions from the Pareto front")
    print("3. Analyze parameter importance")
    print("4. Generate visualizations with bin/analyze_optuna_results.py")
    print()


if __name__ == '__main__':
    main()
