#!/usr/bin/env python3
"""
Visualization Script for Walk-Forward Validation Results

Generates comprehensive visualizations of validation results:
- PF degradation scatter plots
- Regime performance heatmaps
- Equity curve comparisons
- Distribution analysis

Usage:
    python bin/visualize_validation_results.py \
        --input results/validation/walk_forward/ \
        --output results/validation/walk_forward/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_validation_summary(input_dir: Path) -> pd.DataFrame:
    """Load validation summary CSV"""
    summary_path = input_dir / 'validation_summary.csv'

    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return pd.DataFrame()

    return pd.read_csv(summary_path)


def plot_pf_degradation(summary_df: pd.DataFrame, output_dir: Path):
    """
    Create scatter plot of train PF vs validation/OOS PF.

    Shows degradation patterns and identifies overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Train vs Val
    ax = axes[0]
    passed = summary_df[summary_df['passed']]
    failed = summary_df[~summary_df['passed']]

    if len(passed) > 0:
        ax.scatter(passed['train_pf'], passed['val_pf'],
                  alpha=0.6, s=100, c='green', label='Passed', edgecolors='black')

    if len(failed) > 0:
        ax.scatter(failed['train_pf'], failed['val_pf'],
                  alpha=0.6, s=100, c='red', label='Failed', edgecolors='black', marker='x')

    # Add 1:1 line
    max_pf = max(summary_df['train_pf'].max(), summary_df['val_pf'].max())
    ax.plot([0, max_pf], [0, max_pf], 'k--', alpha=0.3, label='1:1 (no degradation)')

    # Add 0.8× line (20% degradation threshold)
    ax.plot([0, max_pf], [0, max_pf * 0.8], 'r--', alpha=0.3, label='20% degradation limit')

    ax.set_xlabel('Train Profit Factor', fontsize=12)
    ax.set_ylabel('Validation Profit Factor', fontsize=12)
    ax.set_title('Train vs Validation PF', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Train vs OOS
    ax = axes[1]

    if len(passed) > 0:
        ax.scatter(passed['train_pf'], passed['oos_pf'],
                  alpha=0.6, s=100, c='green', label='Passed', edgecolors='black')

    if len(failed) > 0:
        ax.scatter(failed['train_pf'], failed['oos_pf'],
                  alpha=0.6, s=100, c='red', label='Failed', edgecolors='black', marker='x')

    # Add 1:1 line
    max_pf = max(summary_df['train_pf'].max(), summary_df['oos_pf'].max())
    ax.plot([0, max_pf], [0, max_pf], 'k--', alpha=0.3, label='1:1 (no degradation)')

    # Add OOS minimum threshold
    ax.axhline(y=1.1, color='orange', linestyle='--', alpha=0.5, label='OOS PF threshold (1.1)')

    ax.set_xlabel('Train Profit Factor', fontsize=12)
    ax.set_ylabel('OOS Profit Factor', fontsize=12)
    ax.set_title('Train vs Out-of-Sample PF', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'pf_degradation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved PF degradation plot to {output_path}")
    plt.close()


def plot_regime_heatmap(input_dir: Path, output_dir: Path):
    """
    Create heatmap of config performance by regime.

    Shows which configs are robust across different market conditions.
    """
    # Load regime breakdown for all configs
    regime_data = []

    for config_dir in input_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue

        regime_file = config_dir / 'regime_breakdown.csv'
        if not regime_file.exists():
            continue

        regime_df = pd.read_csv(regime_file)

        # Extract OOS regime PF
        for _, row in regime_df.iterrows():
            regime_data.append({
                'config': config_dir.name,
                'regime': row['regime'],
                'pf': row['oos_pf'],
                'trades': row['oos_trades'],
            })

    if len(regime_data) == 0:
        logger.warning("No regime data found")
        return

    regime_df = pd.DataFrame(regime_data)

    # Create pivot table
    pivot = regime_df.pivot(index='config', columns='regime', values='pf')

    # Sort by average PF
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False).drop('avg', axis=1)

    # Take top 20 configs for readability
    pivot = pivot.head(20)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.4)))

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                vmin=0.5, vmax=2.0, cbar_kws={'label': 'Profit Factor'},
                linewidths=0.5, ax=ax)

    ax.set_title('OOS Profit Factor by Regime (Top 20 Configs)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_ylabel('Config ID', fontsize=12)

    plt.tight_layout()
    output_path = output_dir / 'regime_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved regime heatmap to {output_path}")
    plt.close()


def plot_metric_distributions(summary_df: pd.DataFrame, output_dir: Path):
    """
    Plot distributions of key metrics for passed vs failed configs.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('oos_pf', 'OOS Profit Factor'),
        ('oos_sharpe', 'OOS Sharpe Ratio'),
        ('oos_dd', 'OOS Max Drawdown'),
        ('oos_trades', 'OOS Trade Count'),
        ('p_value', 'Permutation p-value'),
        ('val_pf', 'Validation PF'),
    ]

    for ax, (metric, label) in zip(axes.flat, metrics):
        passed = summary_df[summary_df['passed']][metric].dropna()
        failed = summary_df[~summary_df['passed']][metric].dropna()

        if len(passed) > 0:
            ax.hist(passed, bins=20, alpha=0.6, color='green', label='Passed', edgecolor='black')

        if len(failed) > 0:
            ax.hist(failed, bins=20, alpha=0.6, color='red', label='Failed', edgecolor='black')

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'metric_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved metric distributions to {output_path}")
    plt.close()


def plot_period_comparison(summary_df: pd.DataFrame, output_dir: Path):
    """
    Compare performance across train/val/oos periods.
    """
    # Take top 10 passed configs
    top_configs = summary_df[summary_df['passed']].nlargest(10, 'oos_pf')

    if len(top_configs) == 0:
        logger.warning("No passed configs to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_configs))
    width = 0.25

    ax.bar(x - width, top_configs['train_pf'], width, label='Train', alpha=0.8, color='steelblue')
    ax.bar(x, top_configs['val_pf'], width, label='Validation', alpha=0.8, color='orange')
    ax.bar(x + width, top_configs['oos_pf'], width, label='OOS', alpha=0.8, color='green')

    ax.set_xlabel('Config', fontsize=12)
    ax.set_ylabel('Profit Factor', fontsize=12)
    ax.set_title('Top 10 Configs: PF Across Periods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_configs['config_id'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.1, color='red', linestyle='--', alpha=0.5, label='OOS threshold')

    plt.tight_layout()
    output_path = output_dir / 'period_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved period comparison to {output_path}")
    plt.close()


def plot_risk_return_scatter(summary_df: pd.DataFrame, output_dir: Path):
    """
    Risk-return scatter plot (OOS Sharpe vs OOS DD).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    passed = summary_df[summary_df['passed']]
    failed = summary_df[~summary_df['passed']]

    if len(passed) > 0:
        scatter = ax.scatter(passed['oos_dd'], passed['oos_sharpe'],
                           s=passed['oos_pf'] * 100,  # Size by PF
                           alpha=0.6, c='green', label='Passed',
                           edgecolors='black', linewidths=1)

    if len(failed) > 0:
        ax.scatter(failed['oos_dd'], failed['oos_sharpe'],
                  s=failed['oos_pf'] * 100,
                  alpha=0.6, c='red', label='Failed',
                  edgecolors='black', linewidths=1, marker='x')

    ax.set_xlabel('OOS Max Drawdown', fontsize=12)
    ax.set_ylabel('OOS Sharpe Ratio', fontsize=12)
    ax.set_title('Risk-Return Profile (OOS)', fontsize=14, fontweight='bold')

    # Add threshold lines
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='Min Sharpe (0.5)')
    ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.3, label='Max DD (25%)')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add size legend
    sizes = [1.0, 1.5, 2.0]
    for size in sizes:
        ax.scatter([], [], s=size*100, c='gray', alpha=0.6,
                  edgecolors='black', label=f'PF = {size}')

    ax.legend(loc='upper right')

    plt.tight_layout()
    output_path = output_dir / 'risk_return_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved risk-return scatter to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Walk-Forward Validation Results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with validation results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (defaults to input directory)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summary
    logger.info("Loading validation summary...")
    summary_df = load_validation_summary(input_dir)

    if len(summary_df) == 0:
        logger.error("No validation data found")
        return

    logger.info(f"Loaded {len(summary_df)} configs")

    # Generate visualizations
    logger.info("Generating PF degradation plots...")
    plot_pf_degradation(summary_df, output_dir)

    logger.info("Generating regime heatmap...")
    plot_regime_heatmap(input_dir, output_dir)

    logger.info("Generating metric distributions...")
    plot_metric_distributions(summary_df, output_dir)

    logger.info("Generating period comparison...")
    plot_period_comparison(summary_df, output_dir)

    logger.info("Generating risk-return scatter...")
    plot_risk_return_scatter(summary_df, output_dir)

    logger.info(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
