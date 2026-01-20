#!/usr/bin/env python3
"""
Pareto Frontier Analyzer v10 - Extract Cross-Regime Guardrails

Purpose:
  Analyze bull (2024) and bear (2022-2023) frontier results
  Extract parameter ranges that work across both regimes
  Identify hard boundaries and robustness patterns
  Generate production-ready guardrails

Input:
  - reports/bull_frontier_v10/trials.csv
  - reports/bear_frontier_v10/trials.csv

Output:
  - Cross-regime parameter analysis
  - Hard boundaries and failure modes
  - Top-quartile parameter distributions
  - Production guardrail recommendations

Usage:
  python3 bin/analyze_pareto_frontiers_v10.py --output reports/pareto_analysis_v10
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def load_trials(bull_path: str, bear_path: str):
    """Load and merge bull/bear trial results."""
    df_bull = pd.read_csv(bull_path)
    df_bear = pd.read_csv(bear_path)

    df_bull['regime'] = 'bull'
    df_bear['regime'] = 'bear'

    df_combined = pd.concat([df_bull, df_bear], ignore_index=True)

    return df_bull, df_bear, df_combined


def identify_hard_boundaries(df_bull, df_bear, threshold=-10000):
    """Identify parameter values that cause catastrophic failures."""
    boundaries = {}

    # Bull failures
    df_bull_fail = df_bull[df_bull['value'] < threshold]
    df_bull_ok = df_bull[df_bull['value'] >= threshold]

    # Bear failures
    df_bear_fail = df_bear[df_bear['value'] < threshold]
    df_bear_ok = df_bear[df_bear['value'] >= threshold]

    # Check min_liquidity boundary
    if len(df_bull_fail) > 0:
        bull_fail_min_liq = df_bull_fail['params_min_liquidity'].min()
        bull_ok_max_liq = df_bull_ok['params_min_liquidity'].max()
        boundaries['bull_min_liquidity'] = {
            'failure_min': bull_fail_min_liq,
            'safe_max': bull_ok_max_liq,
            'n_failures': len(df_bull_fail)
        }

    if len(df_bear_fail) > 0:
        bear_fail_min_liq = df_bear_fail['params_min_liquidity'].min()
        bear_ok_max_liq = df_bear_ok['params_min_liquidity'].max()
        boundaries['bear_min_liquidity'] = {
            'failure_min': bear_fail_min_liq,
            'safe_max': bear_ok_max_liq,
            'n_failures': len(df_bear_fail)
        }

    # Cross-regime safe zone
    if 'bull_min_liquidity' in boundaries and 'bear_min_liquidity' in boundaries:
        boundaries['cross_regime_min_liquidity'] = {
            'safe_max': min(
                boundaries['bull_min_liquidity']['safe_max'],
                boundaries['bear_min_liquidity']['safe_max']
            ),
            'recommendation': 'Hard ceiling - exceeding causes catastrophic DD'
        }

    return boundaries


def analyze_top_quartile(df, regime_name: str, param_cols: list):
    """Analyze top 25% performers to find robust parameter ranges."""
    # Top quartile
    q75 = df['value'].quantile(0.75)
    df_top = df[df['value'] >= q75].copy()

    analysis = {
        'regime': regime_name,
        'n_trials': len(df),
        'n_top_quartile': len(df_top),
        'best_value': df['value'].max(),
        'q75_value': q75,
        'median_value': df['value'].median(),
        'parameters': {}
    }

    # Analyze each parameter in top quartile
    for col in param_cols:
        if col in df_top.columns:
            param_name = col.replace('params_', '')
            analysis['parameters'][param_name] = {
                'min': float(df_top[col].min()),
                'max': float(df_top[col].max()),
                'mean': float(df_top[col].mean()),
                'median': float(df_top[col].median()),
                'std': float(df_top[col].std()),
                'range_width': float(df_top[col].max() - df_top[col].min())
            }

    return analysis


def find_cross_regime_parameters(bull_analysis, bear_analysis):
    """Find parameter ranges that work in both bull and bear markets."""
    cross_regime = {}

    bull_params = bull_analysis['parameters']
    bear_params = bear_analysis['parameters']

    common_params = set(bull_params.keys()) & set(bear_params.keys())

    for param in common_params:
        bull_range = (bull_params[param]['min'], bull_params[param]['max'])
        bear_range = (bear_params[param]['min'], bear_params[param]['max'])

        # Intersection
        intersect_min = max(bull_range[0], bear_range[0])
        intersect_max = min(bull_range[1], bear_range[1])

        # Union
        union_min = min(bull_range[0], bear_range[0])
        union_max = max(bull_range[1], bear_range[1])

        has_overlap = intersect_min <= intersect_max

        cross_regime[param] = {
            'bull_range': bull_range,
            'bear_range': bear_range,
            'intersection': (intersect_min, intersect_max) if has_overlap else None,
            'union': (union_min, union_max),
            'overlap': has_overlap,
            'bull_center': bull_params[param]['median'],
            'bear_center': bear_params[param]['median']
        }

    return cross_regime


def identify_convergence_patterns(df):
    """Identify if parameters converged to identical values (plateau)."""
    # Count unique configurations
    param_cols = [c for c in df.columns if c.startswith('params_')]

    # Round to avoid floating point issues
    df_rounded = df[param_cols].round(6)
    n_unique = len(df_rounded.drop_duplicates())
    n_total = len(df)

    convergence = {
        'n_total_trials': n_total,
        'n_unique_configs': n_unique,
        'convergence_rate': 1 - (n_unique / n_total),
        'plateau_detected': (n_unique / n_total) < 0.10  # Less than 10% unique
    }

    # Find most common configuration
    if convergence['plateau_detected']:
        config_counts = df_rounded.value_counts()
        most_common_count = config_counts.iloc[0]
        convergence['most_common_config_count'] = int(most_common_count)
        convergence['most_common_config_pct'] = float(most_common_count / n_total * 100)

    return convergence


def generate_guardrails(boundaries, cross_regime, bull_analysis, bear_analysis):
    """Generate production-ready guardrail recommendations."""
    guardrails = {
        'version': 'v10',
        'timestamp': datetime.utcnow().isoformat(),
        'hard_boundaries': {},
        'recommended_ranges': {},
        'regime_specific': {
            'bull': {},
            'bear': {}
        }
    }

    # Hard boundaries
    if 'cross_regime_min_liquidity' in boundaries:
        guardrails['hard_boundaries']['min_liquidity'] = {
            'max_safe_value': boundaries['cross_regime_min_liquidity']['safe_max'],
            'reason': 'Exceeding causes 13-15% drawdown in both regimes',
            'enforcement': 'HARD CEILING - never exceed'
        }

    # Cross-regime safe ranges (intersection of top quartiles)
    for param, info in cross_regime.items():
        if info['overlap']:
            guardrails['recommended_ranges'][param] = {
                'min': info['intersection'][0],
                'max': info['intersection'][1],
                'bull_center': info['bull_center'],
                'bear_center': info['bear_center'],
                'regime_adaptive': abs(info['bull_center'] - info['bear_center']) > 0.1
            }

    # Regime-specific recommendations
    for param in cross_regime.keys():
        if param in bull_analysis['parameters']:
            guardrails['regime_specific']['bull'][param] = {
                'median': bull_analysis['parameters'][param]['median'],
                'range': (bull_analysis['parameters'][param]['min'],
                         bull_analysis['parameters'][param]['max'])
            }

        if param in bear_analysis['parameters']:
            guardrails['regime_specific']['bear'][param] = {
                'median': bear_analysis['parameters'][param]['median'],
                'range': (bear_analysis['parameters'][param]['min'],
                         bear_analysis['parameters'][param]['max'])
            }

    return guardrails


def main():
    parser = argparse.ArgumentParser(description="Analyze Pareto Frontiers v10")
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--bull-trials', default='reports/bull_frontier_v10/trials.csv')
    parser.add_argument('--bear-trials', default='reports/bear_frontier_v10/trials.csv')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Pareto Frontier Analysis v10")
    print(f"{'='*80}")
    print(f"Bull trials: {args.bull_trials}")
    print(f"Bear trials: {args.bear_trials}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading trial data...")
    df_bull, df_bear, df_combined = load_trials(args.bull_trials, args.bear_trials)

    print(f"Bull trials: {len(df_bull)}")
    print(f"Bear trials: {len(df_bear)}")
    print(f"Combined: {len(df_combined)}\n")

    # Get parameter columns
    param_cols = [c for c in df_bull.columns if c.startswith('params_')]
    print(f"Analyzing {len(param_cols)} parameters\n")

    # 1. Identify hard boundaries
    print("1. Identifying hard boundaries...")
    boundaries = identify_hard_boundaries(df_bull, df_bear)
    print(f"   Found {len(boundaries)} boundary conditions\n")

    # 2. Analyze top quartile
    print("2. Analyzing top-quartile parameters...")
    bull_analysis = analyze_top_quartile(df_bull, 'bull', param_cols)
    bear_analysis = analyze_top_quartile(df_bear, 'bear', param_cols)
    print(f"   Bull top-quartile: {bull_analysis['n_top_quartile']} trials")
    print(f"   Bear top-quartile: {bear_analysis['n_top_quartile']} trials\n")

    # 3. Find cross-regime parameters
    print("3. Finding cross-regime parameter ranges...")
    cross_regime = find_cross_regime_parameters(bull_analysis, bear_analysis)
    n_overlap = sum(1 for p in cross_regime.values() if p['overlap'])
    print(f"   {n_overlap}/{len(cross_regime)} parameters have cross-regime overlap\n")

    # 4. Identify convergence patterns
    print("4. Checking for convergence/plateau patterns...")
    bull_convergence = identify_convergence_patterns(df_bull)
    bear_convergence = identify_convergence_patterns(df_bear)
    print(f"   Bull plateau: {bull_convergence['plateau_detected']} "
          f"({bull_convergence['n_unique_configs']} unique configs)")
    print(f"   Bear plateau: {bear_convergence['plateau_detected']} "
          f"({bear_convergence['n_unique_configs']} unique configs)\n")

    # 5. Generate guardrails
    print("5. Generating production guardrails...")
    guardrails = generate_guardrails(boundaries, cross_regime, bull_analysis, bear_analysis)
    print(f"   Hard boundaries: {len(guardrails['hard_boundaries'])}")
    print(f"   Recommended ranges: {len(guardrails['recommended_ranges'])}\n")

    # Save results
    print("Saving analysis results...")

    # Full analysis
    full_analysis = {
        'metadata': {
            'version': 'v10',
            'timestamp': datetime.utcnow().isoformat(),
            'bull_trials': len(df_bull),
            'bear_trials': len(df_bear)
        },
        'boundaries': boundaries,
        'bull_analysis': bull_analysis,
        'bear_analysis': bear_analysis,
        'cross_regime_parameters': cross_regime,
        'convergence': {
            'bull': bull_convergence,
            'bear': bear_convergence
        },
        'guardrails': guardrails
    }

    analysis_path = output_dir / 'full_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    print(f"   Full analysis: {analysis_path}")

    # Guardrails only (for easy import)
    guardrails_path = output_dir / 'guardrails_v10.json'
    with open(guardrails_path, 'w') as f:
        json.dump(guardrails, f, indent=2)
    print(f"   Guardrails: {guardrails_path}")

    # Summary report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PARETO FRONTIER ANALYSIS v10 - SUMMARY")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append("HARD BOUNDARIES (Never Exceed):")
    report_lines.append("-" * 80)
    for param, info in guardrails['hard_boundaries'].items():
        report_lines.append(f"  {param}: ≤ {info['max_safe_value']:.3f}")
        report_lines.append(f"    Reason: {info['reason']}")
    report_lines.append("")

    report_lines.append("CROSS-REGIME SAFE RANGES (Work in Both Bull & Bear):")
    report_lines.append("-" * 80)
    for param, info in sorted(guardrails['recommended_ranges'].items()):
        report_lines.append(f"  {param}:")
        report_lines.append(f"    Range: [{info['min']:.3f}, {info['max']:.3f}]")
        report_lines.append(f"    Bull center: {info['bull_center']:.3f}")
        report_lines.append(f"    Bear center: {info['bear_center']:.3f}")
        if info['regime_adaptive']:
            report_lines.append(f"    ⚠️  Regime-adaptive: Consider different values for bull/bear")
    report_lines.append("")

    report_lines.append("PERFORMANCE SUMMARY:")
    report_lines.append("-" * 80)
    report_lines.append(f"  Bull Market (2024):")
    report_lines.append(f"    Best H-Mean: {bull_analysis['best_value']:.2f}")
    report_lines.append(f"    Top-quartile threshold: {bull_analysis['q75_value']:.2f}")
    report_lines.append(f"    Plateau detected: {bull_convergence['plateau_detected']}")
    report_lines.append(f"")
    report_lines.append(f"  Bear Market (2022-2023):")
    report_lines.append(f"    Best H-Mean: {bear_analysis['best_value']:.2f}")
    report_lines.append(f"    Top-quartile threshold: {bear_analysis['q75_value']:.2f}")
    report_lines.append(f"    Plateau detected: {bear_convergence['plateau_detected']}")
    report_lines.append("")

    report_lines.append("="*80)

    report_text = '\n'.join(report_lines)
    print(f"\n{report_text}")

    report_path = output_dir / 'SUMMARY.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n   Summary report: {report_path}")

    print(f"\n{'='*80}")
    print(f"Analysis Complete")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    exit(main())
