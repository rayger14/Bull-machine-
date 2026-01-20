#!/usr/bin/env python3
"""
Quick test script for Phase 2 optimization

Runs 3 trials (9 backtests total) to validate the optimization pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from bin.optimize_bear_s2_s5_multiobjective import (
    objective,
    export_pareto_frontier,
    generate_report,
    RESULTS_DIR
)

def main():
    """Run quick validation test"""
    print("=" * 80)
    print("PHASE 2 OPTIMIZATION - QUICK TEST")
    print("=" * 80)
    print()
    print("Running 3 trials (9 backtests) to validate pipeline...")
    print()

    # Create test study (in-memory, no persistence)
    study = optuna.create_study(
        study_name="phase2_test",
        directions=["minimize", "minimize", "minimize"],
        storage=None  # In-memory
    )

    # Run 3 trials
    try:
        study.optimize(objective, n_trials=3, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        return

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()

    if len(study.trials) > 0:
        print(f"Completed {len(study.trials)} trials")
        print(f"Pareto solutions: {len(study.best_trials)}")
        print()

        # Show results
        for i, trial in enumerate(study.trials):
            pf = trial.user_attrs.get('mean_pf', 0.0)
            trades = trial.user_attrs.get('mean_annual_trades', 0.0)
            dd = trial.user_attrs.get('mean_max_dd', 0.0)

            print(f"Trial {i}:")
            print(f"  PF: {pf:.2f} | Trades/yr: {trades:.1f} | DD: {dd:.1f}%")
            print(f"  Objectives: {trial.values}")
            print()

        print("✓ Pipeline validation successful!")
        print()
        print("To run full optimization:")
        print("  python3 bin/optimize_bear_s2_s5_multiobjective.py")
    else:
        print("✗ No trials completed")

if __name__ == '__main__':
    main()
