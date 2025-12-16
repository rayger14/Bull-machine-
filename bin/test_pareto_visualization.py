#!/usr/bin/env python3
"""
Test script for Pareto frontier visualization.

Creates a synthetic Optuna study with known Pareto frontier for validation.

Usage:
    python bin/test_pareto_visualization.py
"""

import optuna
import numpy as np
import os
import tempfile
from pathlib import Path


def objective(trial):
    """
    Synthetic multi-objective function with known Pareto frontier.

    Objectives:
    1. Maximize profit_factor
    2. Optimize trade_count to range [25, 40]
    3. Minimize max_drawdown
    """
    # Sample parameters
    confidence_threshold = trial.suggest_float('confidence_threshold', 0.5, 0.9)
    risk_per_trade = trial.suggest_float('risk_per_trade', 0.01, 0.05)
    min_rrr = trial.suggest_float('min_rrr', 1.5, 3.0)
    atr_multiplier = trial.suggest_float('atr_multiplier', 1.0, 3.0)

    # Synthetic relationships
    # Higher confidence → higher PF but fewer trades
    base_pf = 0.8 + confidence_threshold * 1.2
    base_trades = 60 - (confidence_threshold * 40)

    # Risk management affects drawdown
    base_dd = -0.15 - (risk_per_trade * 4) + (min_rrr * 0.03)

    # Add noise
    noise = np.random.normal(0, 0.05, 3)

    profit_factor = max(0.5, base_pf + noise[0])
    trade_count = max(5, base_trades + noise[1] * 10)
    max_drawdown = min(-0.05, base_dd + noise[2])

    # Store additional metrics as user attributes
    trial.set_user_attr('win_rate', min(0.95, max(0.3, 0.45 + confidence_threshold * 0.3)))
    trial.set_user_attr('avg_win_loss_ratio', min(4.0, max(1.0, min_rrr * 0.8)))

    return profit_factor, trade_count, max_drawdown


def create_test_study(n_trials: int = 200):
    """Create a test study with synthetic data."""

    # Create temporary database
    db_path = os.path.join('results', 'test_pareto', 'test_study.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove existing study if present
    if os.path.exists(db_path):
        os.remove(db_path)

    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name='test_pareto_frontier',
        storage=storage,
        directions=['maximize', 'maximize', 'minimize'],  # PF max, TC optimize (handled in viz), DD min
        load_if_exists=False
    )

    print(f"Creating test study with {n_trials} trials...")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n✓ Created test study: {db_path}")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)}")

    # Print best trials
    print("\nTop 5 trials by Profit Factor:")
    print(f"{'Trial':<8} {'PF':<8} {'Trades':<10} {'DD':<12}")
    print("-" * 40)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.values[0], reverse=True)

    for trial in sorted_trials[:5]:
        print(f"{trial.number:<8} "
              f"{trial.values[0]:<8.3f} "
              f"{trial.values[1]:<10.1f} "
              f"{trial.values[2]:<12.2%}")

    return db_path


def run_visualization(db_path: str):
    """Run the visualization script on the test study."""
    import subprocess

    output_dir = 'results/test_pareto/visualizations'

    cmd = [
        'python', 'bin/visualize_pareto_frontier.py',
        '--study-name', 'test_pareto_frontier',
        '--db-path', db_path,
        '--output-dir', output_dir
    ]

    print(f"\nRunning visualization script...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n✓ Visualization complete!")
        print(f"\nGenerated files:")
        print(f"  - {output_dir}/pareto_3d.html")
        print(f"  - {output_dir}/pareto_2d_projections.png")
        print(f"  - {output_dir}/parameter_sensitivity.png")
        print(f"  - {output_dir}/pareto_distributions.png")
        print(f"  - {output_dir}/pareto_trials.csv")
        print(f"  - {output_dir}/pareto_analysis_summary.txt")

        print(f"\nOpen interactive plot:")
        print(f"  open {output_dir}/pareto_3d.html")
    else:
        print(f"\n✗ Visualization failed with return code {result.returncode}")


def main():
    print("="*80)
    print("PARETO FRONTIER VISUALIZATION TEST")
    print("="*80 + "\n")

    # Create test study
    db_path = create_test_study(n_trials=200)

    # Run visualization
    run_visualization(db_path)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
