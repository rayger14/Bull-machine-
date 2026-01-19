#!/usr/bin/env python3
"""
Pareto Frontier Visualization Script

Analyzes multi-objective optimization results from Optuna studies and generates:
- 3D interactive Pareto frontier plots (Plotly)
- 2D projection plots with target zones (Matplotlib)
- Parameter sensitivity heatmaps
- Exportable CSV of Pareto-optimal trials

Usage:
    python bin/visualize_pareto_frontier.py --study-name bear_phase2_tuning --db-path optuna_studies.db
    python bin/visualize_pareto_frontier.py --study-name bear_phase2_tuning --db-path optuna_studies.db --output-dir results/custom_output
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from optuna.visualization import plot_pareto_front
from scipy.spatial import ConvexHull


# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']


def load_study(study_name: str, db_path: str) -> optuna.Study:
    """Load Optuna study from SQLite database."""
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"✓ Loaded study '{study_name}' with {len(study.trials)} trials")
        return study
    except Exception as e:
        raise ValueError(f"Failed to load study '{study_name}' from {db_path}: {e}")


def extract_trial_data(study: optuna.Study) -> pd.DataFrame:
    """Extract trial data into a structured DataFrame."""
    data = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        row = {
            'trial_number': trial.number,
            'profit_factor': trial.values[0] if len(trial.values) > 0 else np.nan,
            'trade_count': trial.values[1] if len(trial.values) > 1 else np.nan,
            'max_drawdown': trial.values[2] if len(trial.values) > 2 else np.nan,
            'state': trial.state.name,
            'duration': trial.duration.total_seconds() if trial.duration else np.nan,
        }

        # Add all parameters
        for key, value in trial.params.items():
            row[f'param_{key}'] = value

        # Add user attributes (additional metrics)
        for key, value in trial.user_attrs.items():
            row[f'attr_{key}'] = value

        data.append(row)

    df = pd.DataFrame(data)
    print(f"✓ Extracted {len(df)} completed trials")
    return df


def identify_pareto_front(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify Pareto-optimal trials.

    Optimization goals:
    - Maximize Profit Factor (values[0])
    - Maximize Trade Count (values[1]) - but we want moderate, not extreme
    - Minimize Max Drawdown (values[2])
    """
    # Filter valid trials
    valid = df[
        df['profit_factor'].notna() &
        df['trade_count'].notna() &
        df['max_drawdown'].notna()
    ].copy()

    if len(valid) == 0:
        raise ValueError("No valid trials with all objectives")

    # For Pareto dominance, we need to flip max_drawdown (minimize -> maximize negative)
    # And we treat trade_count as "closeness to target range [25-40]"
    valid['neg_drawdown'] = -valid['max_drawdown']
    valid['trade_count_score'] = -np.abs(valid['trade_count'] - 32.5)  # Target: 25-40, center at 32.5

    # Identify Pareto front
    pareto_mask = np.ones(len(valid), dtype=bool)

    for i in range(len(valid)):
        if not pareto_mask[i]:
            continue

        # Check if any other trial dominates this one
        for j in range(len(valid)):
            if i == j or not pareto_mask[j]:
                continue

            # Trial j dominates trial i if it's better or equal in all objectives
            # and strictly better in at least one
            pf_better_eq = valid.iloc[j]['profit_factor'] >= valid.iloc[i]['profit_factor']
            tc_better_eq = valid.iloc[j]['trade_count_score'] >= valid.iloc[i]['trade_count_score']
            dd_better_eq = valid.iloc[j]['neg_drawdown'] >= valid.iloc[i]['neg_drawdown']

            pf_strictly_better = valid.iloc[j]['profit_factor'] > valid.iloc[i]['profit_factor']
            tc_strictly_better = valid.iloc[j]['trade_count_score'] > valid.iloc[i]['trade_count_score']
            dd_strictly_better = valid.iloc[j]['neg_drawdown'] > valid.iloc[i]['neg_drawdown']

            dominates = (pf_better_eq and tc_better_eq and dd_better_eq and
                        (pf_strictly_better or tc_strictly_better or dd_strictly_better))

            if dominates:
                pareto_mask[i] = False
                break

    pareto_trials = valid[pareto_mask].copy()
    non_pareto_trials = valid[~pareto_mask].copy()

    print(f"✓ Identified {len(pareto_trials)} Pareto-optimal trials")
    return pareto_trials, non_pareto_trials


def create_3d_interactive_plot(pareto_trials: pd.DataFrame,
                                non_pareto_trials: pd.DataFrame,
                                output_path: str):
    """Create 3D interactive Plotly visualization of Pareto frontier."""

    fig = go.Figure()

    # Non-Pareto trials (gray)
    fig.add_trace(go.Scatter3d(
        x=non_pareto_trials['profit_factor'],
        y=non_pareto_trials['trade_count'],
        z=non_pareto_trials['max_drawdown'],
        mode='markers',
        marker=dict(
            size=4,
            color='lightgray',
            opacity=0.3,
            line=dict(width=0)
        ),
        name='All Trials',
        text=[f"Trial {n}<br>PF: {pf:.3f}<br>Trades: {tc:.0f}<br>DD: {dd:.2%}"
              for n, pf, tc, dd in zip(
                  non_pareto_trials['trial_number'],
                  non_pareto_trials['profit_factor'],
                  non_pareto_trials['trade_count'],
                  non_pareto_trials['max_drawdown']
              )],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Pareto-optimal trials (red)
    fig.add_trace(go.Scatter3d(
        x=pareto_trials['profit_factor'],
        y=pareto_trials['trade_count'],
        z=pareto_trials['max_drawdown'],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            opacity=0.9,
            line=dict(width=1, color='darkred'),
            symbol='diamond'
        ),
        name='Pareto Optimal',
        text=[f"<b>Trial {n}</b><br>PF: {pf:.3f}<br>Trades: {tc:.0f}<br>DD: {dd:.2%}"
              for n, pf, tc, dd in zip(
                  pareto_trials['trial_number'],
                  pareto_trials['profit_factor'],
                  pareto_trials['trade_count'],
                  pareto_trials['max_drawdown']
              )],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Pareto Frontier: Multi-Objective Optimization</b>',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(
                title='<b>Profit Factor</b>',
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True
            ),
            yaxis=dict(
                title='<b>Annual Trade Count</b>',
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True
            ),
            zaxis=dict(
                title='<b>Max Drawdown</b>',
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True,
                tickformat='.0%'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.7,
            y=0.9,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        width=1200,
        height=800,
        template='plotly_white'
    )

    # Save
    fig.write_html(output_path)
    print(f"✓ Saved 3D interactive plot: {output_path}")


def create_2d_projections(pareto_trials: pd.DataFrame,
                          non_pareto_trials: pd.DataFrame,
                          output_dir: str):
    """Create 2D projection plots with target zones."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pareto Frontier: 2D Projections', fontsize=18, fontweight='bold', y=0.995)

    # 1. Profit Factor vs Trade Count
    ax = axes[0, 0]
    ax.scatter(non_pareto_trials['trade_count'], non_pareto_trials['profit_factor'],
               c='lightgray', s=50, alpha=0.4, label='All Trials')
    ax.scatter(pareto_trials['trade_count'], pareto_trials['profit_factor'],
               c='red', s=100, alpha=0.9, marker='D', edgecolors='darkred', linewidths=1.5,
               label='Pareto Optimal', zorder=10)

    # Target zones
    ax.axhline(y=1.3, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target PF > 1.3')
    ax.axvspan(25, 40, alpha=0.1, color='blue', label='Target Trades: 25-40')

    ax.set_xlabel('Annual Trade Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit Factor', fontsize=12, fontweight='bold')
    ax.set_title('Profit Factor vs Trade Count', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 2. Profit Factor vs Max Drawdown
    ax = axes[0, 1]
    ax.scatter(non_pareto_trials['max_drawdown'], non_pareto_trials['profit_factor'],
               c='lightgray', s=50, alpha=0.4, label='All Trials')
    ax.scatter(pareto_trials['max_drawdown'], pareto_trials['profit_factor'],
               c='red', s=100, alpha=0.9, marker='D', edgecolors='darkred', linewidths=1.5,
               label='Pareto Optimal', zorder=10)

    ax.axhline(y=1.3, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target PF > 1.3')
    ax.axvspan(-0.30, -0.15, alpha=0.1, color='blue', label='Target DD: -15% to -30%')

    ax.set_xlabel('Max Drawdown', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit Factor', fontsize=12, fontweight='bold')
    ax.set_title('Profit Factor vs Max Drawdown', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 3. Trade Count vs Max Drawdown
    ax = axes[1, 0]
    ax.scatter(non_pareto_trials['max_drawdown'], non_pareto_trials['trade_count'],
               c='lightgray', s=50, alpha=0.4, label='All Trials')
    ax.scatter(pareto_trials['max_drawdown'], pareto_trials['trade_count'],
               c='red', s=100, alpha=0.9, marker='D', edgecolors='darkred', linewidths=1.5,
               label='Pareto Optimal', zorder=10)

    ax.axhspan(25, 40, alpha=0.1, color='blue', label='Target Trades: 25-40')
    ax.axvspan(-0.30, -0.15, alpha=0.1, color='blue', label='Target DD: -15% to -30%')

    ax.set_xlabel('Max Drawdown', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Trade Count', fontsize=12, fontweight='bold')
    ax.set_title('Trade Count vs Max Drawdown', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 4. Pareto Trial Numbers (for reference)
    ax = axes[1, 1]
    pareto_sorted = pareto_trials.sort_values('profit_factor', ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(pareto_sorted)))

    bars = ax.barh(range(len(pareto_sorted)), pareto_sorted['profit_factor'], color=colors)
    ax.set_yticks(range(len(pareto_sorted)))
    ax.set_yticklabels([f"Trial {n}" for n in pareto_sorted['trial_number']])
    ax.set_xlabel('Profit Factor', fontsize=12, fontweight='bold')
    ax.set_title('Pareto-Optimal Trials Ranked by Profit Factor', fontsize=14, fontweight='bold')
    ax.axvline(x=1.3, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, pf) in enumerate(zip(bars, pareto_sorted['profit_factor'])):
        ax.text(pf + 0.02, i, f'{pf:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pareto_2d_projections.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 2D projections: {output_path}")


def create_parameter_sensitivity_heatmap(all_trials: pd.DataFrame,
                                          pareto_trials: pd.DataFrame,
                                          output_dir: str):
    """Create parameter sensitivity analysis heatmaps."""

    # Get parameter columns
    param_cols = [col for col in all_trials.columns if col.startswith('param_')]

    if len(param_cols) == 0:
        print("⚠ No parameters found, skipping sensitivity analysis")
        return

    # Calculate correlations
    objectives = ['profit_factor', 'trade_count', 'max_drawdown']
    correlations = []

    for param_col in param_cols:
        param_name = param_col.replace('param_', '')
        row = {'parameter': param_name}

        for obj in objectives:
            valid_data = all_trials[[param_col, obj]].dropna()
            if len(valid_data) > 1:
                corr = valid_data[param_col].corr(valid_data[obj])
                row[obj] = corr
            else:
                row[obj] = 0.0

        correlations.append(row)

    corr_df = pd.DataFrame(correlations)

    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(param_cols) * 0.4)))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=18, fontweight='bold')

    # Heatmap
    ax = axes[0]
    pivot = corr_df.set_index('parameter')[objectives]
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Correlation'}, ax=ax, linewidths=0.5)
    ax.set_title('Parameter Correlations with Objectives', fontsize=14, fontweight='bold')
    ax.set_xlabel('Objective', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')

    # Importance bars (absolute correlation average)
    ax = axes[1]
    corr_df['abs_avg'] = corr_df[objectives].abs().mean(axis=1)
    corr_sorted = corr_df.sort_values('abs_avg', ascending=True)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(corr_sorted)))
    bars = ax.barh(corr_sorted['parameter'], corr_sorted['abs_avg'], color=colors)
    ax.set_xlabel('Average |Correlation|', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Importance (Avg Absolute Correlation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, corr_sorted['abs_avg']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'parameter_sensitivity.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved parameter sensitivity: {output_path}")


def create_pareto_values_distribution(pareto_trials: pd.DataFrame,
                                       all_trials: pd.DataFrame,
                                       output_dir: str):
    """Create distribution plots comparing Pareto vs all trials."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Objective Distribution: Pareto Optimal vs All Trials',
                 fontsize=18, fontweight='bold')

    objectives = [
        ('profit_factor', 'Profit Factor', False),
        ('trade_count', 'Annual Trade Count', False),
        ('max_drawdown', 'Max Drawdown', True)
    ]

    for ax, (col, label, is_percent) in zip(axes, objectives):
        # Histograms
        ax.hist(all_trials[col], bins=30, alpha=0.5, color='gray',
                label='All Trials', density=True)
        ax.hist(pareto_trials[col], bins=20, alpha=0.7, color='red',
                label='Pareto Optimal', density=True)

        # Means
        all_mean = all_trials[col].mean()
        pareto_mean = pareto_trials[col].mean()

        ax.axvline(all_mean, color='gray', linestyle='--', linewidth=2,
                   label=f'All Mean: {all_mean:.2%}' if is_percent else f'All Mean: {all_mean:.2f}')
        ax.axvline(pareto_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Pareto Mean: {pareto_mean:.2%}' if is_percent else f'Pareto Mean: {pareto_mean:.2f}')

        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Distribution', fontsize=14, fontweight='bold')

        if is_percent:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pareto_distributions.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution plots: {output_path}")


def export_pareto_trials(pareto_trials: pd.DataFrame, output_path: str):
    """Export Pareto-optimal trials to CSV."""

    # Select key columns
    export_cols = ['trial_number', 'profit_factor', 'trade_count', 'max_drawdown']
    param_cols = [col for col in pareto_trials.columns if col.startswith('param_')]
    export_cols.extend(param_cols)

    export_df = pareto_trials[export_cols].copy()

    # Rename parameter columns
    export_df.columns = [col.replace('param_', '') for col in export_df.columns]

    # Sort by profit factor
    export_df = export_df.sort_values('profit_factor', ascending=False)

    # Save
    export_df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(export_df)} Pareto trials to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("PARETO-OPTIMAL TRIALS SUMMARY")
    print("="*80)
    print(f"\nTop 5 by Profit Factor:")
    print(export_df[['trial_number', 'profit_factor', 'trade_count', 'max_drawdown']].head())
    print("\n" + "="*80)


def generate_summary_report(pareto_trials: pd.DataFrame,
                            all_trials: pd.DataFrame,
                            output_dir: str):
    """Generate text summary report."""

    report_path = os.path.join(output_dir, 'pareto_analysis_summary.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PARETO FRONTIER ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Trials: {len(all_trials)}\n")
        f.write(f"Pareto-Optimal Trials: {len(pareto_trials)} ({len(pareto_trials)/len(all_trials)*100:.1f}%)\n\n")

        f.write("-"*80 + "\n")
        f.write("OBJECTIVE STATISTICS\n")
        f.write("-"*80 + "\n\n")

        objectives = [
            ('profit_factor', 'Profit Factor'),
            ('trade_count', 'Annual Trade Count'),
            ('max_drawdown', 'Max Drawdown')
        ]

        for col, label in objectives:
            f.write(f"{label}:\n")
            f.write(f"  All Trials:     Mean={all_trials[col].mean():.3f}, "
                   f"Std={all_trials[col].std():.3f}, "
                   f"Min={all_trials[col].min():.3f}, "
                   f"Max={all_trials[col].max():.3f}\n")
            f.write(f"  Pareto Optimal: Mean={pareto_trials[col].mean():.3f}, "
                   f"Std={pareto_trials[col].std():.3f}, "
                   f"Min={pareto_trials[col].min():.3f}, "
                   f"Max={pareto_trials[col].max():.3f}\n\n")

        f.write("-"*80 + "\n")
        f.write("TOP 10 PARETO-OPTIMAL TRIALS (by Profit Factor)\n")
        f.write("-"*80 + "\n\n")

        top10 = pareto_trials.nlargest(10, 'profit_factor')
        f.write(f"{'Trial':<8} {'PF':<8} {'Trades':<10} {'DD':<12}\n")
        f.write("-"*40 + "\n")

        for _, row in top10.iterrows():
            f.write(f"{int(row['trial_number']):<8} "
                   f"{row['profit_factor']:<8.3f} "
                   f"{row['trade_count']:<10.1f} "
                   f"{row['max_drawdown']:<12.2%}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TARGET ZONE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # Trials meeting targets
        target_trials = pareto_trials[
            (pareto_trials['profit_factor'] > 1.3) &
            (pareto_trials['trade_count'] >= 25) &
            (pareto_trials['trade_count'] <= 40)
        ]

        f.write(f"Trials meeting targets (PF > 1.3, Trades 25-40): {len(target_trials)}\n")

        if len(target_trials) > 0:
            f.write("\nTrials in target zone:\n")
            f.write(f"{'Trial':<8} {'PF':<8} {'Trades':<10} {'DD':<12}\n")
            f.write("-"*40 + "\n")

            for _, row in target_trials.iterrows():
                f.write(f"{int(row['trial_number']):<8} "
                       f"{row['profit_factor']:<8.3f} "
                       f"{row['trade_count']:<10.1f} "
                       f"{row['max_drawdown']:<12.2%}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✓ Saved summary report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Pareto frontier from Optuna multi-objective optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin/visualize_pareto_frontier.py --study-name bear_phase2_tuning --db-path optuna_studies.db
  python bin/visualize_pareto_frontier.py --study-name bear_phase2_tuning --db-path optuna_studies.db --output-dir results/custom_output
        """
    )

    parser.add_argument('--study-name', type=str, required=True,
                       help='Name of the Optuna study')
    parser.add_argument('--db-path', type=str, required=True,
                       help='Path to SQLite database')
    parser.add_argument('--output-dir', type=str, default='results/phase3_frontier',
                       help='Output directory for results (default: results/phase3_frontier)')

    args = parser.parse_args()

    # Setup
    print("\n" + "="*80)
    print("PARETO FRONTIER VISUALIZATION")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ Output directory: {args.output_dir}\n")

    # Load study
    study = load_study(args.study_name, args.db_path)

    # Extract data
    all_trials = extract_trial_data(study)

    if len(all_trials) == 0:
        print("✗ No completed trials found")
        return

    # Identify Pareto front
    pareto_trials, non_pareto_trials = identify_pareto_front(all_trials)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 3D interactive plot
    html_path = os.path.join(args.output_dir, 'pareto_3d.html')
    create_3d_interactive_plot(pareto_trials, non_pareto_trials, html_path)

    # 2D projections
    create_2d_projections(pareto_trials, non_pareto_trials, args.output_dir)

    # Parameter sensitivity
    create_parameter_sensitivity_heatmap(all_trials, pareto_trials, args.output_dir)

    # Distribution comparison
    create_pareto_values_distribution(pareto_trials, all_trials, args.output_dir)

    # Export CSV
    csv_path = os.path.join(args.output_dir, 'pareto_trials.csv')
    export_pareto_trials(pareto_trials, csv_path)

    # Summary report
    generate_summary_report(pareto_trials, all_trials, args.output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  - pareto_3d.html              : Interactive 3D plot")
    print(f"  - pareto_2d_projections.png   : 2D projection plots")
    print(f"  - parameter_sensitivity.png   : Parameter correlation heatmap")
    print(f"  - pareto_distributions.png    : Objective distributions")
    print(f"  - pareto_trials.csv           : Pareto-optimal trial data")
    print(f"  - pareto_analysis_summary.txt : Text summary report")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
