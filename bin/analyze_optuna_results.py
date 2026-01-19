#!/usr/bin/env python3
"""
Analyze Optuna optimization results

Usage:
    python3 bin/analyze_optuna_results.py configs/auto/best_optuna_study.pkl
"""

import sys
from pathlib import Path

try:
    import joblib
    import optuna
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install optuna joblib")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not installed - visualizations disabled")


def analyze_study(study_path: str):
    """Analyze completed Optuna study"""

    # Load study
    print(f"\nLoading study from: {study_path}")
    study = joblib.load(study_path)

    print("\n" + "="*80)
    print("OPTUNA STUDY ANALYSIS")
    print("="*80)

    # Basic stats
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best trials (Pareto front): {len(study.best_trials)}")

    # Pareto front
    print("\n" + "-"*80)
    print("PARETO FRONT (Non-dominated Solutions)")
    print("-"*80)

    for i, trial in enumerate(study.best_trials[:10], 1):
        pf, dd = trial.values
        sharpe = trial.user_attrs.get('sharpe_ratio', 0.0)
        trades = trial.user_attrs.get('num_trades', 0)

        print(f"\nSolution {i}:")
        print(f"  Trial #{trial.number}")
        print(f"  PF: {pf:.2f} | DD: {dd:.1f}% | Sharpe: {sharpe:.2f} | Trades: {trades}")
        print(f"  Parameters:")
        for param, value in sorted(trial.params.items()):
            print(f"    {param:20s}: {value:.4f}")

    # Parameter importance
    print("\n" + "-"*80)
    print("PARAMETER IMPORTANCE")
    print("-"*80)

    try:
        # Importance for first objective (PF)
        importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
        print("\nFor Profit Factor:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {param:20s}: {imp:.4f}")

        # Importance for second objective (DD)
        importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[1])
        print("\nFor Max Drawdown:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {param:20s}: {imp:.4f}")

    except Exception as e:
        print(f"  Could not compute importance: {e}")

    # Best by different metrics
    print("\n" + "-"*80)
    print("BEST SOLUTIONS BY DIFFERENT CRITERIA")
    print("-"*80)

    # Highest PF
    best_pf_trial = max(study.best_trials, key=lambda t: t.values[0])
    print(f"\nHighest Profit Factor:")
    print(f"  PF: {best_pf_trial.values[0]:.2f}")
    print(f"  DD: {best_pf_trial.values[1]:.1f}%")
    print(f"  Params: {best_pf_trial.params}")

    # Lowest DD
    best_dd_trial = min(study.best_trials, key=lambda t: t.values[1])
    print(f"\nLowest Max Drawdown:")
    print(f"  PF: {best_dd_trial.values[0]:.2f}")
    print(f"  DD: {best_dd_trial.values[1]:.1f}%")
    print(f"  Params: {best_dd_trial.params}")

    # Best Sharpe (if available)
    pareto_with_sharpe = [t for t in study.best_trials if 'sharpe_ratio' in t.user_attrs]
    if pareto_with_sharpe:
        best_sharpe_trial = max(pareto_with_sharpe, key=lambda t: t.user_attrs['sharpe_ratio'])
        print(f"\nHighest Sharpe Ratio (on Pareto front):")
        print(f"  PF: {best_sharpe_trial.values[0]:.2f}")
        print(f"  DD: {best_sharpe_trial.values[1]:.1f}%")
        print(f"  Sharpe: {best_sharpe_trial.user_attrs['sharpe_ratio']:.2f}")
        print(f"  Params: {best_sharpe_trial.params}")

    # Balanced (closest to ideal point)
    print(f"\nMost Balanced (closest to ideal PF=∞, DD=0):")
    # Normalize and find closest to (1, 0) in normalized space
    max_pf = max(t.values[0] for t in study.best_trials)
    min_dd = min(t.values[1] for t in study.best_trials)
    max_dd = max(t.values[1] for t in study.best_trials)

    best_balanced = min(
        study.best_trials,
        key=lambda t: ((1 - t.values[0]/max_pf)**2 + ((t.values[1] - min_dd)/(max_dd - min_dd))**2)**0.5
    )
    print(f"  PF: {best_balanced.values[0]:.2f}")
    print(f"  DD: {best_balanced.values[1]:.1f}%")
    print(f"  Params: {best_balanced.params}")

    # Visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\n" + "-"*80)
        print("GENERATING VISUALIZATIONS")
        print("-"*80)

        try:
            # Pareto front plot
            fig = optuna.visualization.matplotlib.plot_pareto_front(
                study,
                target_names=['Profit Factor', 'Max Drawdown (%)']
            )
            plot_path = Path(study_path).parent / "pareto_front.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()

            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plot_path = Path(study_path).parent / "optimization_history.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()

            # Parameter importances
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plot_path = Path(study_path).parent / "param_importances.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()

            # Parallel coordinate plot
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plot_path = Path(study_path).parent / "parallel_coordinate.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()

        except Exception as e:
            print(f"  Visualization error: {e}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    study_path = sys.argv[1]

    if not Path(study_path).exists():
        print(f"ERROR: Study file not found: {study_path}")
        sys.exit(1)

    analyze_study(study_path)


if __name__ == '__main__':
    main()
