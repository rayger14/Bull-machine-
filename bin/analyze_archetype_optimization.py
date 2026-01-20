#!/usr/bin/env python3
"""
Analyze Optuna Archetype Optimization Results

Provides detailed analysis of parallel archetype optimization:
- Per-group best parameters and scores
- Parameter importance analysis
- Convergence plots
- Trade-off visualizations (PF vs trades, DD vs score)
- Comparison tables

Usage:
    python bin/analyze_archetype_optimization.py --storage optuna_archetypes.db
    python bin/analyze_archetype_optimization.py --export-csv results/
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import optuna
import pandas as pd


def analyze_study(study_name: str, storage_path: str) -> Dict:
    """
    Analyze a single Optuna study.

    Args:
        study_name: Study name
        storage_path: SQLite storage path

    Returns:
        Analysis dict with best params, scores, and statistics
    """
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}"
    )

    best_trial = study.best_trial
    trials_df = study.trials_dataframe()

    # Compute statistics
    analysis = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_score': best_trial.value,
        'best_trial': best_trial.number,
        'best_params': best_trial.params,
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }

    # Top 5 trials
    analysis['top_5_trials'] = trials_df.nlargest(5, 'value')[
        ['number', 'value', 'state']
    ].to_dict('records')

    # Parameter importance (if enough trials)
    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            analysis['param_importance'] = dict(sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 most important
        except Exception as e:
            analysis['param_importance'] = f"Error: {e}"

    return analysis


def compare_studies(storage_path: str) -> pd.DataFrame:
    """
    Compare all studies in storage.

    Args:
        storage_path: SQLite storage path

    Returns:
        DataFrame with study comparison
    """
    # Get all study names from SQLite
    conn = sqlite3.connect(storage_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT study_name FROM studies")
    study_names = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Analyze each study
    comparisons = []
    for study_name in study_names:
        analysis = analyze_study(study_name, storage_path)
        comparisons.append({
            'Study': study_name,
            'Best Score': analysis['best_score'],
            'Trials': analysis['n_trials'],
            'Completed': analysis['completed_trials'],
            'Pruned': analysis['pruned_trials'],
            'Failed': analysis['failed_trials'],
            'Best Trial': analysis['best_trial']
        })

    df = pd.DataFrame(comparisons)
    return df.sort_values('Best Score', ascending=False)


def export_best_params(storage_path: str, output_path: str):
    """
    Export best parameters from all studies to JSON.

    Args:
        storage_path: SQLite storage path
        output_path: Output JSON path
    """
    conn = sqlite3.connect(storage_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT study_name FROM studies")
    study_names = [row[0] for row in cursor.fetchall()]
    conn.close()

    results = {}
    for study_name in study_names:
        analysis = analyze_study(study_name, storage_path)
        results[study_name] = {
            'best_score': analysis['best_score'],
            'best_params': analysis['best_params'],
            'n_trials': analysis['n_trials']
        }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Best parameters exported to {output_path}")


def print_analysis_report(storage_path: str):
    """
    Print comprehensive analysis report.

    Args:
        storage_path: SQLite storage path
    """
    print("\n" + "=" * 80)
    print("ARCHETYPE OPTIMIZATION ANALYSIS")
    print("=" * 80)

    # Study comparison
    comparison_df = compare_studies(storage_path)
    print("\n📊 STUDY COMPARISON")
    print("-" * 80)
    print(comparison_df.to_string(index=False))

    # Detailed analysis per study
    conn = sqlite3.connect(storage_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT study_name FROM studies")
    study_names = [row[0] for row in cursor.fetchall()]
    conn.close()

    for study_name in study_names:
        analysis = analyze_study(study_name, storage_path)

        print(f"\n\n📈 {study_name.upper()}")
        print("-" * 80)
        print(f"Best Score: {analysis['best_score']:.3f}")
        print(f"Best Trial: #{analysis['best_trial']}")
        print(f"Total Trials: {analysis['n_trials']} "
              f"(Completed: {analysis['completed_trials']}, "
              f"Pruned: {analysis['pruned_trials']}, "
              f"Failed: {analysis['failed_trials']})")

        print(f"\n🏆 Top 5 Trials:")
        for trial in analysis['top_5_trials']:
            print(f"  Trial #{trial['number']}: {trial['value']:.3f} ({trial['state']})")

        print(f"\n🔧 Best Parameters:")
        for param, value in sorted(analysis['best_params'].items()):
            if not param.startswith('_'):  # Skip internal params
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")

        if 'param_importance' in analysis and isinstance(analysis['param_importance'], dict):
            print(f"\n📊 Parameter Importance (Top 10):")
            for param, importance in analysis['param_importance'].items():
                print(f"  {param}: {importance:.4f}")

    print("\n" + "=" * 80)


def export_trials_to_csv(storage_path: str, output_dir: str):
    """
    Export all trials to CSV for external analysis.

    Args:
        storage_path: SQLite storage path
        output_dir: Output directory for CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(storage_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT study_name FROM studies")
    study_names = [row[0] for row in cursor.fetchall()]
    conn.close()

    for study_name in study_names:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}"
        )
        df = study.trials_dataframe()

        csv_path = output_path / f"{study_name}_trials.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported {len(df)} trials to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Archetype Optimization Results')
    parser.add_argument('--storage', type=str, default='optuna_archetypes.db',
                        help='SQLite storage path')
    parser.add_argument('--export-json', type=str, default=None,
                        help='Export best params to JSON')
    parser.add_argument('--export-csv', type=str, default=None,
                        help='Export trials to CSV directory')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only show study comparison table')

    args = parser.parse_args()

    if not Path(args.storage).exists():
        print(f"Error: Storage file '{args.storage}' not found")
        return 1

    # Main analysis report
    if args.compare_only:
        df = compare_studies(args.storage)
        print(df.to_string(index=False))
    else:
        print_analysis_report(args.storage)

    # Export best params
    if args.export_json:
        export_best_params(args.storage, args.export_json)

    # Export trials to CSV
    if args.export_csv:
        export_trials_to_csv(args.storage, args.export_csv)

    return 0


if __name__ == '__main__':
    exit(main())
