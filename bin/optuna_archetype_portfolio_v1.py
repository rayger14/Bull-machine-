#!/usr/bin/env python3
"""
Step 5: Constrained Optuna Sweep - Archetype Portfolio Optimization

Optimizes per-archetype parameters with hard constraints:
- Each archetype must have PF ≥ 1.0 (no bleeders)
- Overall portfolio PF ≥ 1.2
- Drawdown ≤ 25%

Search space:
- Per-archetype: gates, weights, cooldowns
- VE exits: trail_atr, max_bars
- Global: max_trades_per_day

Objective: Median PF across rolling windows (2022, 2023, 2024, full)

Usage:
    python3 bin/optuna_archetype_portfolio_v1.py --trials 100 --asset BTC
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
    """
    Run backtest and extract metrics including per-archetype breakdown.
    Returns: dict with overall metrics + archetype_metrics dict
    """
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
            timeout=180,  # 3 min per window
            check=False
        )

        output = result.stdout + result.stderr

        # Extract overall metrics
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

        # Parse archetype distribution (from analyze_archetype_perf.py style output)
        # Look for patterns like "TRAP_WITHIN_TREND:" followed by metrics
        archetype_section = re.search(
            r'BY ARCHETYPE:(.*?)={60,}',
            output,
            re.DOTALL
        )

        if archetype_section:
            arch_text = archetype_section.group(1)

            # Match each archetype block
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


def check_constraints(metrics_list):
    """
    Check hard constraints across all windows.
    Returns: (passed: bool, reason: str)
    """
    # Check each window
    for window_name, metrics in metrics_list:
        if metrics is None:
            return False, f"Failed to run {window_name}"

        # Overall constraints
        if metrics['trades'] < 50:
            return False, f"{window_name}: Too few trades ({metrics['trades']})"

        if metrics['trades'] > 2000:
            return False, f"{window_name}: Too many trades ({metrics['trades']})"

        if metrics['drawdown'] > 25.0:
            return False, f"{window_name}: DD {metrics['drawdown']:.1f}% > 25%"

        # Per-archetype constraint: No bleeders with meaningful sample size
        for arch_name, arch_data in metrics['archetype_metrics'].items():
            if arch_data['trades'] >= 20:  # Meaningful sample
                if arch_data['profit_factor'] < 1.0:
                    return False, f"{window_name}: {arch_name} bleeding (PF {arch_data['profit_factor']:.2f})"

    # Overall PF constraint (on full period)
    full_metrics = next(m for name, m in metrics_list if name == 'full')
    if full_metrics['profit_factor'] < 1.2:
        return False, f"Full period PF {full_metrics['profit_factor']:.2f} < 1.2"

    return True, "All constraints passed"


def objective(trial, base_config_path, asset):
    """
    Optuna objective function with constraints.
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # Sample parameters from search space

    # TRAP: Suppress or improve
    trap_gate = trial.suggest_float('trap_final_fusion_gate', 0.38, 0.50)
    trap_weight = trial.suggest_float('trap_archetype_weight', 0.5, 1.0)
    trap_cooldown = trial.suggest_int('trap_cooldown_bars', 8, 15)

    # VOLUME EXHAUSTION: Fix bleeding
    ve_gate = trial.suggest_float('ve_final_fusion_gate', 0.30, 0.40)
    ve_weight = trial.suggest_float('ve_archetype_weight', 1.0, 1.5)
    ve_cooldown = trial.suggest_int('ve_cooldown_bars', 6, 12)
    ve_trail_atr = trial.suggest_float('ve_trail_atr_mult', 0.8, 1.5)
    ve_max_bars = trial.suggest_int('ve_max_bars', 40, 80)

    # ORDER BLOCK: Expand
    ob_gate = trial.suggest_float('ob_final_fusion_gate', 0.28, 0.38)
    ob_weight = trial.suggest_float('ob_archetype_weight', 1.2, 1.6)
    ob_cooldown = trial.suggest_int('ob_cooldown_bars', 4, 10)

    # GLOBAL
    max_trades_day = trial.suggest_int('max_trades_per_day', 6, 12)

    # Apply to config
    config['archetypes']['trap_within_trend']['final_fusion_gate'] = trap_gate
    config['archetypes']['trap_within_trend']['archetype_weight'] = trap_weight
    config['archetypes']['trap_within_trend']['cooldown_bars'] = trap_cooldown

    config['archetypes']['volume_exhaustion']['final_fusion_gate'] = ve_gate
    config['archetypes']['volume_exhaustion']['archetype_weight'] = ve_weight
    config['archetypes']['volume_exhaustion']['cooldown_bars'] = ve_cooldown

    config['archetypes']['order_block_retest']['final_fusion_gate'] = ob_gate
    config['archetypes']['order_block_retest']['archetype_weight'] = ob_weight
    config['archetypes']['order_block_retest']['cooldown_bars'] = ob_cooldown

    config['archetypes']['max_trades_per_day'] = max_trades_day

    # VE exit tuning
    if 'exits' not in config['archetypes']:
        config['archetypes']['exits'] = {}
    if 'L' not in config['archetypes']['exits']:  # L = volume_exhaustion
        config['archetypes']['exits']['L'] = {}

    config['archetypes']['exits']['L']['trail_atr'] = ve_trail_atr
    config['archetypes']['exits']['L']['max_bars'] = ve_max_bars

    # Write temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_config = f.name

    try:
        # Run on all windows
        window_metrics = []
        pfs = []

        for start, end, name in WINDOWS:
            print(f"  Trial {trial.number}: Running {name}...")
            metrics = run_backtest(temp_config, start, end, asset)

            if metrics is None:
                # Failed - return worst possible score
                return -1000.0

            window_metrics.append((name, metrics))
            pfs.append(metrics['profit_factor'])

            # Early stopping: If any window is catastrophic
            if metrics['profit_factor'] < 0.5:
                print(f"  Trial {trial.number}: Early stop - {name} PF {metrics['profit_factor']:.2f}")
                return -1000.0

        # Check constraints
        passed, reason = check_constraints(window_metrics)

        if not passed:
            print(f"  Trial {trial.number}: FAILED - {reason}")
            # Return negative score proportional to how bad it is
            return min(pfs) - 2.0

        # All constraints passed - compute objective
        # Objective: Median PF (robust to outliers) + bonus for consistency
        median_pf = np.median(pfs)
        min_pf = min(pfs)
        consistency_bonus = 0.2 * min_pf  # Reward if worst year is still good

        objective_value = median_pf + consistency_bonus

        print(f"  Trial {trial.number}: SUCCESS - Objective {objective_value:.3f} (PFs: {[f'{p:.2f}' for p in pfs]})")

        # Store trial results
        trial.set_user_attr('pfs', pfs)
        trial.set_user_attr('median_pf', median_pf)
        trial.set_user_attr('min_pf', min_pf)
        for name, metrics in window_metrics:
            trial.set_user_attr(f'{name}_trades', metrics['trades'])
            trial.set_user_attr(f'{name}_dd', metrics['drawdown'])

        return objective_value

    finally:
        # Cleanup temp config
        Path(temp_config).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Constrained Optuna archetype optimization')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--asset', type=str, default='BTC', help='Asset symbol')
    parser.add_argument('--base-config', type=str,
                       default='configs/baseline_btc_bull_regime_routed_v1.json',
                       help='Base config to start from')
    parser.add_argument('--output', type=str,
                       default='results/optuna_step5_constrained',
                       help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create study
    study_name = f"archetype_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{output_dir}/optuna_study.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    )

    print("="*80)
    print("OPTUNA CONSTRAINED ARCHETYPE PORTFOLIO OPTIMIZATION")
    print("="*80)
    print(f"Base config: {args.base_config}")
    print(f"Trials: {args.trials}")
    print(f"Asset: {args.asset}")
    print(f"Output: {args.output}")
    print(f"Storage: {storage}")
    print("\nConstraints:")
    print("  - Each archetype PF ≥ 1.0 (min 20 trades)")
    print("  - Overall PF ≥ 1.2")
    print("  - Max DD ≤ 25%")
    print("  - Trades: 50-2000 per window")
    print("\nObjective: Median PF + 0.2*MinPF (robustness)")
    print("="*80)

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args.base_config, args.asset),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best objective: {study.best_value:.3f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBest trial attributes:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")

    # Save best config
    with open(args.base_config, 'r') as f:
        best_config = json.load(f)

    # Apply best params
    bp = study.best_params
    best_config['archetypes']['trap_within_trend']['final_fusion_gate'] = bp['trap_final_fusion_gate']
    best_config['archetypes']['trap_within_trend']['archetype_weight'] = bp['trap_archetype_weight']
    best_config['archetypes']['trap_within_trend']['cooldown_bars'] = bp['trap_cooldown_bars']

    best_config['archetypes']['volume_exhaustion']['final_fusion_gate'] = bp['ve_final_fusion_gate']
    best_config['archetypes']['volume_exhaustion']['archetype_weight'] = bp['ve_archetype_weight']
    best_config['archetypes']['volume_exhaustion']['cooldown_bars'] = bp['ve_cooldown_bars']

    best_config['archetypes']['order_block_retest']['final_fusion_gate'] = bp['ob_final_fusion_gate']
    best_config['archetypes']['order_block_retest']['archetype_weight'] = bp['ob_archetype_weight']
    best_config['archetypes']['order_block_retest']['cooldown_bars'] = bp['ob_cooldown_bars']

    best_config['archetypes']['max_trades_per_day'] = bp['max_trades_per_day']

    if 'exits' not in best_config['archetypes']:
        best_config['archetypes']['exits'] = {}
    if 'L' not in best_config['archetypes']['exits']:
        best_config['archetypes']['exits']['L'] = {}

    best_config['archetypes']['exits']['L']['trail_atr'] = bp['ve_trail_atr_mult']
    best_config['archetypes']['exits']['L']['max_bars'] = bp['ve_max_bars']

    # Save
    best_config_path = output_dir / 'best_config.json'
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest config saved to: {best_config_path}")

    # Save study summary
    summary_path = output_dir / 'optimization_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPTUNA OPTIMIZATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Study name: {study_name}\n")
        f.write(f"Trials completed: {len(study.trials)}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best objective: {study.best_value:.3f}\n\n")

        f.write("Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nBest trial performance:\n")
        for key, value in study.best_trial.user_attrs.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 TRIALS\n")
        f.write("="*80 + "\n")

        top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1000, reverse=True)[:10]
        for i, trial in enumerate(top_trials, 1):
            f.write(f"\n{i}. Trial {trial.number}: {trial.value:.3f}\n")
            if 'pfs' in trial.user_attrs:
                f.write(f"   PFs: {trial.user_attrs['pfs']}\n")
            if 'full_trades' in trial.user_attrs:
                f.write(f"   Full trades: {trial.user_attrs['full_trades']}\n")

    print(f"Summary saved to: {summary_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
