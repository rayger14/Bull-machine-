#!/usr/bin/env python3
"""
Pareto Frontier Exploration for Archetype Portfolio

Instead of guessing constraints, systematically explore what's achievable:
- Run multiple Optuna studies with relaxed constraints
- Map out the Pareto frontier of PF vs DD vs Trade Count
- Identify the best achievable performance

Strategy:
1. Frontier 1: Maximize PF (no constraint)
2. Frontier 2: Maximize PF with DD ≤ 30%
3. Frontier 3: Maximize PF with DD ≤ 25%
4. Frontier 4: Maximize PF with DD ≤ 20%
5. Frontier 5: Maximize PF + minimize individual archetype bleeding

This tells us what's ACTUALLY achievable before committing to constraints.

Usage:
    python3 bin/optuna_frontier_exploration.py --trials 50 --asset BTC
"""

import argparse
import json
import optuna
import subprocess
import tempfile
import re
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Validation windows
WINDOWS = [
    ('2022-01-01', '2022-12-31', '2022'),
    ('2023-01-01', '2023-12-31', '2023'),
    ('2024-01-01', '2024-12-31', '2024'),
    ('2022-01-01', '2024-12-31', 'full')
]

def run_backtest(config_path: str, start_date: str, end_date: str, asset: str = "BTC"):
    """Run backtest and extract metrics including per-archetype breakdown."""
    cmd = [
        "python3", "bin/backtest_knowledge_v2.py",
        "--asset", asset,
        "--start", start_date,
        "--end", end_date,
        "--config", config_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            check=False
        )

        output = result.stdout + result.stderr

        metrics = {
            'pnl': 0.0,
            'trades': 0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'archetype_metrics': {}
        }

        # Overall metrics
        pnl_match = re.search(r'Total PNL:\s+\$?([-\d,\.]+)', output)
        if pnl_match:
            metrics['pnl'] = float(pnl_match.group(1).replace(',', ''))

        trades_match = re.search(r'Total Trades:\s+(\d+)', output)
        if trades_match:
            metrics['trades'] = int(trades_match.group(1))

        wr_match = re.search(r'Win Rate:\s+([\d\.]+)%', output)
        if wr_match:
            metrics['win_rate'] = float(wr_match.group(1))

        dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
        if dd_match:
            metrics['drawdown'] = float(dd_match.group(1))

        pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
        if pf_match:
            metrics['profit_factor'] = float(pf_match.group(1))

        sharpe_match = re.search(r'Sharpe Ratio:\s+([-\d\.]+)', output)
        if sharpe_match:
            metrics['sharpe'] = float(sharpe_match.group(1))

        # Parse archetype distribution
        archetype_section = re.search(
            r'BY ARCHETYPE:(.*?)={60,}',
            output,
            re.DOTALL
        )

        if archetype_section:
            arch_text = archetype_section.group(1)
            arch_blocks = re.finditer(
                r'(\w+):\s+Trades:\s+(\d+).*?Profit Factor:\s+([\d\.]+)',
                arch_text,
                re.DOTALL
            )

            for match in arch_blocks:
                arch_name = match.group(1).lower()
                arch_trades = int(match.group(2))
                arch_pf = float(match.group(3))

                metrics['archetype_metrics'][arch_name] = {
                    'trades': arch_trades,
                    'profit_factor': arch_pf
                }

        return metrics

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {start_date} to {end_date}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def apply_params_to_config(config, params):
    """Apply trial parameters to config."""
    config['archetypes']['trap_within_trend']['final_fusion_gate'] = params['trap_final_fusion_gate']
    config['archetypes']['trap_within_trend']['archetype_weight'] = params['trap_archetype_weight']
    config['archetypes']['trap_within_trend']['cooldown_bars'] = params['trap_cooldown_bars']

    config['archetypes']['volume_exhaustion']['final_fusion_gate'] = params['ve_final_fusion_gate']
    config['archetypes']['volume_exhaustion']['archetype_weight'] = params['ve_archetype_weight']
    config['archetypes']['volume_exhaustion']['cooldown_bars'] = params['ve_cooldown_bars']

    config['archetypes']['order_block_retest']['final_fusion_gate'] = params['ob_final_fusion_gate']
    config['archetypes']['order_block_retest']['archetype_weight'] = params['ob_archetype_weight']
    config['archetypes']['order_block_retest']['cooldown_bars'] = params['ob_cooldown_bars']

    config['archetypes']['max_trades_per_day'] = params['max_trades_per_day']

    # VE exit tuning
    if 'exits' not in config['archetypes']:
        config['archetypes']['exits'] = {}
    if 'L' not in config['archetypes']['exits']:
        config['archetypes']['exits']['L'] = {}

    config['archetypes']['exits']['L']['trail_atr'] = params['ve_trail_atr_mult']
    config['archetypes']['exits']['L']['max_bars'] = params['ve_max_bars']

    return config


def objective_unconstrained(trial, base_config_path, asset):
    """
    Frontier 1: Pure PF maximization (no constraints)
    Objective: Median PF across windows
    """
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # Sample parameters
    params = {
        'trap_final_fusion_gate': trial.suggest_float('trap_final_fusion_gate', 0.38, 0.50),
        'trap_archetype_weight': trial.suggest_float('trap_archetype_weight', 0.5, 1.0),
        'trap_cooldown_bars': trial.suggest_int('trap_cooldown_bars', 8, 15),
        've_final_fusion_gate': trial.suggest_float('ve_final_fusion_gate', 0.30, 0.40),
        've_archetype_weight': trial.suggest_float('ve_archetype_weight', 1.0, 1.5),
        've_cooldown_bars': trial.suggest_int('ve_cooldown_bars', 6, 12),
        've_trail_atr_mult': trial.suggest_float('ve_trail_atr_mult', 0.8, 1.5),
        've_max_bars': trial.suggest_int('ve_max_bars', 40, 80),
        'ob_final_fusion_gate': trial.suggest_float('ob_final_fusion_gate', 0.28, 0.38),
        'ob_archetype_weight': trial.suggest_float('ob_archetype_weight', 1.2, 1.6),
        'ob_cooldown_bars': trial.suggest_int('ob_cooldown_bars', 4, 10),
        'max_trades_per_day': trial.suggest_int('max_trades_per_day', 6, 12),
    }

    config = apply_params_to_config(config, params)

    # Write temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_config = f.name

    try:
        window_metrics = []
        pfs = []

        for start, end, name in WINDOWS:
            print(f"  Trial {trial.number}: Running {name}...")
            metrics = run_backtest(temp_config, start, end, asset)

            if metrics is None:
                return -1000.0

            window_metrics.append((name, metrics))
            pfs.append(metrics['profit_factor'])

        # Pure PF maximization
        median_pf = np.median(pfs)

        # Store all metrics for analysis
        trial.set_user_attr('pfs', pfs)
        trial.set_user_attr('median_pf', median_pf)
        trial.set_user_attr('min_pf', min(pfs))
        trial.set_user_attr('max_pf', max(pfs))

        # Store full window metrics
        full_metrics = next(m for name, m in window_metrics if name == 'full')
        trial.set_user_attr('full_pf', full_metrics['profit_factor'])
        trial.set_user_attr('full_dd', full_metrics['drawdown'])
        trial.set_user_attr('full_trades', full_metrics['trades'])
        trial.set_user_attr('full_sharpe', full_metrics['sharpe'])

        # Store per-archetype metrics
        for arch_name, arch_data in full_metrics['archetype_metrics'].items():
            trial.set_user_attr(f'{arch_name}_pf', arch_data['profit_factor'])
            trial.set_user_attr(f'{arch_name}_trades', arch_data['trades'])

        print(f"  Trial {trial.number}: PF={median_pf:.3f}, DD={full_metrics['drawdown']:.1f}%, Trades={full_metrics['trades']}")

        return median_pf

    finally:
        Path(temp_config).unlink(missing_ok=True)


def objective_dd_constrained(trial, base_config_path, asset, max_dd):
    """
    Frontier 2-4: PF maximization with DD constraint
    Objective: Median PF, penalized if DD exceeds threshold
    """
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    params = {
        'trap_final_fusion_gate': trial.suggest_float('trap_final_fusion_gate', 0.38, 0.50),
        'trap_archetype_weight': trial.suggest_float('trap_archetype_weight', 0.5, 1.0),
        'trap_cooldown_bars': trial.suggest_int('trap_cooldown_bars', 8, 15),
        've_final_fusion_gate': trial.suggest_float('ve_final_fusion_gate', 0.30, 0.40),
        've_archetype_weight': trial.suggest_float('ve_archetype_weight', 1.0, 1.5),
        've_cooldown_bars': trial.suggest_int('ve_cooldown_bars', 6, 12),
        've_trail_atr_mult': trial.suggest_float('ve_trail_atr_mult', 0.8, 1.5),
        've_max_bars': trial.suggest_int('ve_max_bars', 40, 80),
        'ob_final_fusion_gate': trial.suggest_float('ob_final_fusion_gate', 0.28, 0.38),
        'ob_archetype_weight': trial.suggest_float('ob_archetype_weight', 1.2, 1.6),
        'ob_cooldown_bars': trial.suggest_int('ob_cooldown_bars', 4, 10),
        'max_trades_per_day': trial.suggest_int('max_trades_per_day', 6, 12),
    }

    config = apply_params_to_config(config, params)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_config = f.name

    try:
        window_metrics = []
        pfs = []

        for start, end, name in WINDOWS:
            print(f"  Trial {trial.number}: Running {name}...")
            metrics = run_backtest(temp_config, start, end, asset)

            if metrics is None:
                return -1000.0

            window_metrics.append((name, metrics))
            pfs.append(metrics['profit_factor'])

        median_pf = np.median(pfs)
        full_metrics = next(m for name, m in window_metrics if name == 'full')

        # Store metrics
        trial.set_user_attr('pfs', pfs)
        trial.set_user_attr('median_pf', median_pf)
        trial.set_user_attr('min_pf', min(pfs))
        trial.set_user_attr('full_pf', full_metrics['profit_factor'])
        trial.set_user_attr('full_dd', full_metrics['drawdown'])
        trial.set_user_attr('full_trades', full_metrics['trades'])

        for arch_name, arch_data in full_metrics['archetype_metrics'].items():
            trial.set_user_attr(f'{arch_name}_pf', arch_data['profit_factor'])
            trial.set_user_attr(f'{arch_name}_trades', arch_data['trades'])

        # Penalty if DD exceeds threshold
        if full_metrics['drawdown'] > max_dd:
            penalty = (full_metrics['drawdown'] - max_dd) * 0.1  # 0.1 penalty per % over
            print(f"  Trial {trial.number}: DD={full_metrics['drawdown']:.1f}% > {max_dd}% (penalty -{penalty:.2f})")
            return median_pf - penalty

        print(f"  Trial {trial.number}: PF={median_pf:.3f}, DD={full_metrics['drawdown']:.1f}% ✓")
        return median_pf

    finally:
        Path(temp_config).unlink(missing_ok=True)


def objective_no_bleeders(trial, base_config_path, asset):
    """
    Frontier 5: Maximize PF while ensuring no archetype bleeds
    Objective: Median PF with heavy penalty for bleeding archetypes
    """
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    params = {
        'trap_final_fusion_gate': trial.suggest_float('trap_final_fusion_gate', 0.38, 0.50),
        'trap_archetype_weight': trial.suggest_float('trap_archetype_weight', 0.5, 1.0),
        'trap_cooldown_bars': trial.suggest_int('trap_cooldown_bars', 8, 15),
        've_final_fusion_gate': trial.suggest_float('ve_final_fusion_gate', 0.30, 0.40),
        've_archetype_weight': trial.suggest_float('ve_archetype_weight', 1.0, 1.5),
        've_cooldown_bars': trial.suggest_int('ve_cooldown_bars', 6, 12),
        've_trail_atr_mult': trial.suggest_float('ve_trail_atr_mult', 0.8, 1.5),
        've_max_bars': trial.suggest_int('ve_max_bars', 40, 80),
        'ob_final_fusion_gate': trial.suggest_float('ob_final_fusion_gate', 0.28, 0.38),
        'ob_archetype_weight': trial.suggest_float('ob_archetype_weight', 1.2, 1.6),
        'ob_cooldown_bars': trial.suggest_int('ob_cooldown_bars', 4, 10),
        'max_trades_per_day': trial.suggest_int('max_trades_per_day', 6, 12),
    }

    config = apply_params_to_config(config, params)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_config = f.name

    try:
        window_metrics = []
        pfs = []

        for start, end, name in WINDOWS:
            print(f"  Trial {trial.number}: Running {name}...")
            metrics = run_backtest(temp_config, start, end, asset)

            if metrics is None:
                return -1000.0

            window_metrics.append((name, metrics))
            pfs.append(metrics['profit_factor'])

        median_pf = np.median(pfs)
        full_metrics = next(m for name, m in window_metrics if name == 'full')

        trial.set_user_attr('pfs', pfs)
        trial.set_user_attr('median_pf', median_pf)
        trial.set_user_attr('full_pf', full_metrics['profit_factor'])
        trial.set_user_attr('full_dd', full_metrics['drawdown'])
        trial.set_user_attr('full_trades', full_metrics['trades'])

        # Calculate bleeding penalty
        bleeding_penalty = 0.0
        for arch_name, arch_data in full_metrics['archetype_metrics'].items():
            trial.set_user_attr(f'{arch_name}_pf', arch_data['profit_factor'])
            trial.set_user_attr(f'{arch_name}_trades', arch_data['trades'])

            if arch_data['trades'] >= 20 and arch_data['profit_factor'] < 1.0:
                deficit = 1.0 - arch_data['profit_factor']
                bleeding_penalty += deficit * 2.0  # Heavy penalty
                print(f"  Trial {trial.number}: {arch_name} bleeding (PF {arch_data['profit_factor']:.2f})")

        if bleeding_penalty > 0:
            return median_pf - bleeding_penalty

        print(f"  Trial {trial.number}: PF={median_pf:.3f}, No bleeders ✓")
        return median_pf

    finally:
        Path(temp_config).unlink(missing_ok=True)


def run_frontier(frontier_name, objective_fn, n_trials, base_config, asset, output_dir):
    """Run a single frontier exploration."""
    study_name = f"frontier_{frontier_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{output_dir}/frontier_{frontier_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=False
    )

    print("=" * 80)
    print(f"FRONTIER: {frontier_name}")
    print("=" * 80)

    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

    # Save results
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.3f}")
    print(f"\nBest attributes:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")

    # Save summary
    summary_path = output_dir / f'frontier_{frontier_name}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Frontier: {frontier_name}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.3f}\n\n")
        f.write("Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nBest attributes:\n")
        for key, value in study.best_trial.user_attrs.items():
            f.write(f"  {key}: {value}\n")

    return study


def main():
    parser = argparse.ArgumentParser(description='Pareto frontier exploration')
    parser.add_argument('--trials', type=int, default=30, help='Trials per frontier')
    parser.add_argument('--asset', type=str, default='BTC')
    parser.add_argument('--base-config', type=str,
                       default='configs/baseline_btc_bull_regime_routed_v1.json')
    parser.add_argument('--output', type=str,
                       default='results/optuna_frontier_exploration')
    parser.add_argument('--frontiers', type=str, default='all',
                       help='Comma-separated list: unconstrained,dd30,dd25,dd20,no_bleeders or "all"')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frontiers_to_run = args.frontiers.split(',') if args.frontiers != 'all' else ['unconstrained', 'dd30', 'dd25', 'dd20', 'no_bleeders']

    results = {}

    if 'unconstrained' in frontiers_to_run:
        print("\n" + "=" * 80)
        print("FRONTIER 1: UNCONSTRAINED (Pure PF Maximization)")
        print("=" * 80)
        results['unconstrained'] = run_frontier(
            'unconstrained',
            lambda trial: objective_unconstrained(trial, args.base_config, args.asset),
            args.trials,
            args.base_config,
            args.asset,
            output_dir
        )

    if 'dd30' in frontiers_to_run:
        print("\n" + "=" * 80)
        print("FRONTIER 2: DD ≤ 30%")
        print("=" * 80)
        results['dd30'] = run_frontier(
            'dd30',
            lambda trial: objective_dd_constrained(trial, args.base_config, args.asset, 30.0),
            args.trials,
            args.base_config,
            args.asset,
            output_dir
        )

    if 'dd25' in frontiers_to_run:
        print("\n" + "=" * 80)
        print("FRONTIER 3: DD ≤ 25%")
        print("=" * 80)
        results['dd25'] = run_frontier(
            'dd25',
            lambda trial: objective_dd_constrained(trial, args.base_config, args.asset, 25.0),
            args.trials,
            args.base_config,
            args.asset,
            output_dir
        )

    if 'dd20' in frontiers_to_run:
        print("\n" + "=" * 80)
        print("FRONTIER 4: DD ≤ 20%")
        print("=" * 80)
        results['dd20'] = run_frontier(
            'dd20',
            lambda trial: objective_dd_constrained(trial, args.base_config, args.asset, 20.0),
            args.trials,
            args.base_config,
            args.asset,
            output_dir
        )

    if 'no_bleeders' in frontiers_to_run:
        print("\n" + "=" * 80)
        print("FRONTIER 5: NO BLEEDING ARCHETYPES")
        print("=" * 80)
        results['no_bleeders'] = run_frontier(
            'no_bleeders',
            lambda trial: objective_no_bleeders(trial, args.base_config, args.asset),
            args.trials,
            args.base_config,
            args.asset,
            output_dir
        )

    # Create final summary
    print("\n" + "=" * 80)
    print("FRONTIER EXPLORATION COMPLETE")
    print("=" * 80)
    print("\nBest results per frontier:")
    for name, study in results.items():
        print(f"\n{name}:")
        print(f"  Best PF: {study.best_value:.3f}")
        if 'full_dd' in study.best_trial.user_attrs:
            print(f"  DD: {study.best_trial.user_attrs['full_dd']:.1f}%")
        if 'full_trades' in study.best_trial.user_attrs:
            print(f"  Trades: {study.best_trial.user_attrs['full_trades']}")

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
