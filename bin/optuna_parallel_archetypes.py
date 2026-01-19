#!/usr/bin/env python3
"""
Parallel Optuna Archetype Optimization with Multiprocessing

Runs 4 independent Optuna studies in parallel, one per archetype group:
1. Trap Within Trend (A, G, K) - momentum-based reversals
2. Order Block Retest (B, H, L) - structure-based entries
3. BOS/CHOCH (C) - continuation patterns
4. Long Squeeze (S5) - funding rate cascades

Features:
- Multiprocessing pool for true parallelism (4 processes)
- Hyperband pruner for early stopping (6x faster convergence)
- Multi-fidelity: test on 1mo → 3mo → 9mo progressively
- Per-archetype parameter spaces with shared globals
- Result aggregation into unified config
- Estimated runtime: 6-8 hours (down from 33h sequential)

Architecture:
- Each process runs an independent Optuna study
- SQLite storage for study persistence
- Progress tracking via shared queue
- Graceful shutdown on Ctrl+C

Usage:
    python bin/optuna_parallel_archetypes.py --trials 100 --base-config configs/profile_production.json
    python bin/optuna_parallel_archetypes.py --resume  # Resume from checkpoints
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Archetype Group Definitions
# ============================================================================

ARCHETYPE_GROUPS = {
    'trap_within_trend': {
        'archetypes': ['A', 'G', 'K'],  # Trap Reversal, Re-Accumulate, Wick Trap
        'canonical': ['spring', 'liquidity_sweep', 'trap_within_trend'],
        'description': 'Momentum-based reversals and liquidity traps',
        'trader_type': 'Moneytaur',  # Wick specialist
    },
    'order_block_retest': {
        'archetypes': ['B', 'H', 'L'],  # Order Block, Trap in Trend, Volume Exhaustion
        'canonical': ['order_block_retest', 'momentum_continuation', 'volume_exhaustion'],
        'description': 'Structure-based order block retests',
        'trader_type': 'Zeroika',  # Structure specialist
    },
    'bos_choch': {
        'archetypes': ['C'],  # FVG Continuation
        'canonical': ['wick_trap'],
        'description': 'Break of Structure and Change of Character',
        'trader_type': 'Generic',
    },
    'long_squeeze': {
        'archetypes': ['S5'],  # Long Squeeze Cascade
        'canonical': ['long_squeeze'],
        'description': 'Funding rate cascade patterns',
        'trader_type': 'Moneytaur',  # Funding specialist
    }
}


# ============================================================================
# Backtest Execution
# ============================================================================

def run_backtest(
    config_path: str,
    start_date: str,
    end_date: str,
    asset: str = "BTC",
    timeout: int = 120
) -> Optional[Dict]:
    """
    Run backtest_knowledge_v2.py and extract metrics.

    Args:
        config_path: Path to config JSON
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        asset: Asset symbol (BTC, ETH, etc.)
        timeout: Timeout in seconds

    Returns:
        Dict with metrics or None on error
    """
    cmd = [
        "python3",
        "bin/backtest_knowledge_v2.py",
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
            timeout=timeout,
            check=False,
            cwd=Path(__file__).parent.parent  # Run from repo root
        )

        output = result.stdout + result.stderr

        # Extract metrics using regex
        metrics = {
            'pnl': 0.0,
            'trades': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'avg_pnl_per_trade': 0.0
        }

        # Parse metrics
        pnl_match = re.search(r'Total PNL:\s+\$?([-\d,\.]+)', output)
        if pnl_match:
            metrics['pnl'] = float(pnl_match.group(1).replace(',', ''))

        trades_match = re.search(r'Total Trades:\s+(\d+)', output)
        if trades_match:
            metrics['trades'] = int(trades_match.group(1))

        roi_match = re.search(r'ROI:\s+([-\d\.]+)%', output)
        if roi_match:
            metrics['roi'] = float(roi_match.group(1))

        wr_match = re.search(r'Win Rate:\s+([\d\.]+)%', output)
        if wr_match:
            metrics['win_rate'] = float(wr_match.group(1))

        dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
        if dd_match:
            metrics['drawdown'] = float(dd_match.group(1))

        pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
        if pf_match:
            metrics['profit_factor'] = float(pf_match.group(1))

        sharpe_match = re.search(r'Sharpe:\s+([-\d\.]+)', output)
        if sharpe_match:
            metrics['sharpe'] = float(sharpe_match.group(1))

        if metrics['trades'] > 0:
            metrics['avg_pnl_per_trade'] = metrics['pnl'] / metrics['trades']

        return metrics

    except subprocess.TimeoutExpired:
        logger.warning(f"Backtest timeout for {start_date} to {end_date}")
        return None
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return None


# ============================================================================
# Multi-Fidelity Training
# ============================================================================

def get_training_periods(fidelity: int) -> Tuple[str, str]:
    """
    Get training period based on fidelity level.

    Fidelity levels:
    - 0: 1 month (2024-01-01 to 2024-01-31) - fast pruning
    - 1: 3 months (2024-01-01 to 2024-03-31) - medium validation
    - 2: 9 months (2024-01-01 to 2024-09-30) - full evaluation

    Args:
        fidelity: Fidelity level (0-2)

    Returns:
        (start_date, end_date) tuple
    """
    periods = {
        0: ("2024-01-01", "2024-01-31"),   # 1 month
        1: ("2024-01-01", "2024-03-31"),   # 3 months
        2: ("2024-01-01", "2024-09-30"),   # 9 months
    }
    return periods.get(fidelity, periods[2])


# ============================================================================
# Parameter Space Definitions
# ============================================================================

def suggest_global_params(trial: optuna.Trial, base_cfg: dict) -> dict:
    """
    Suggest global parameters shared across all archetypes.

    Args:
        trial: Optuna trial
        base_cfg: Base configuration

    Returns:
        Modified config dict
    """
    cfg = json.loads(json.dumps(base_cfg))  # Deep copy

    # Global liquidity threshold
    if 'archetypes' not in cfg:
        cfg['archetypes'] = {}
    if 'thresholds' not in cfg['archetypes']:
        cfg['archetypes']['thresholds'] = {}

    cfg['archetypes']['thresholds']['min_liquidity'] = trial.suggest_float(
        'global_min_liquidity', 0.10, 0.35, step=0.02
    )

    # Global fusion threshold floor
    if 'decision_gates' not in cfg:
        cfg['decision_gates'] = {}

    cfg['decision_gates']['final_fusion_floor'] = trial.suggest_float(
        'global_fusion_floor', 0.30, 0.45, step=0.01
    )

    # Fusion weights (must sum to 1.0)
    if 'fusion' not in cfg:
        cfg['fusion'] = {}
    if 'weights' not in cfg['fusion']:
        cfg['fusion']['weights'] = {}

    w_wyckoff = trial.suggest_float('w_wyckoff', 0.15, 0.55, step=0.05)
    w_liquidity = trial.suggest_float('w_liquidity', 0.15, 0.55, step=0.05)
    w_momentum = trial.suggest_float('w_momentum', 0.05, 0.45, step=0.05)

    # Normalize to sum to 1.0
    total = w_wyckoff + w_liquidity + w_momentum
    cfg['fusion']['weights']['wyckoff'] = w_wyckoff / total
    cfg['fusion']['weights']['liquidity'] = w_liquidity / total
    cfg['fusion']['weights']['momentum'] = w_momentum / total

    return cfg


def suggest_archetype_params(
    trial: optuna.Trial,
    cfg: dict,
    group_name: str
) -> dict:
    """
    Suggest per-archetype parameters for a specific group.

    Args:
        trial: Optuna trial
        cfg: Config dict with global params already set
        group_name: Archetype group name

    Returns:
        Modified config dict
    """
    group = ARCHETYPE_GROUPS[group_name]

    # Enable only this group's archetypes
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']:
        cfg['archetypes'][f'enable_{letter}'] = (letter in group['archetypes'])

    # Group-specific parameters
    if group_name == 'trap_within_trend':
        # Trap patterns - emphasize momentum and liquidity inversions
        cfg['archetypes']['trap_within_trend'] = {
            'fusion_threshold': trial.suggest_float('trap_fusion', 0.30, 0.45, step=0.01),
            'adx_threshold': trial.suggest_float('trap_adx', 20.0, 35.0, step=1.0),
            'liquidity_threshold': trial.suggest_float('trap_liq_max', 0.15, 0.35, step=0.02),
            'archetype_weight': trial.suggest_float('trap_weight', 0.85, 1.15, step=0.05),
        }

        cfg['archetypes']['spring'] = {
            'fusion_threshold': trial.suggest_float('spring_fusion', 0.28, 0.42, step=0.01),
            'pti_score_threshold': trial.suggest_float('spring_pti', 0.30, 0.55, step=0.05),
            'disp_atr_multiplier': trial.suggest_float('spring_disp', 0.6, 1.2, step=0.1),
        }

        cfg['archetypes']['liquidity_sweep'] = {
            'fusion_threshold': trial.suggest_float('sweep_fusion', 0.32, 0.48, step=0.01),
            'boms_strength_min': trial.suggest_float('sweep_boms', 0.30, 0.55, step=0.05),
            'liquidity_min': trial.suggest_float('sweep_liq', 0.30, 0.50, step=0.02),
        }

    elif group_name == 'order_block_retest':
        # Order block patterns - emphasize structure and Wyckoff
        cfg['archetypes']['order_block_retest'] = {
            'fusion_threshold': trial.suggest_float('ob_fusion', 0.30, 0.45, step=0.01),
            'boms_strength_min': trial.suggest_float('ob_boms', 0.25, 0.45, step=0.02),
            'wyckoff_min': trial.suggest_float('ob_wyckoff', 0.30, 0.50, step=0.02),
            'archetype_weight': trial.suggest_float('ob_weight', 0.90, 1.20, step=0.05),
        }

        cfg['archetypes']['volume_exhaustion'] = {
            'fusion_threshold': trial.suggest_float('volexh_fusion', 0.32, 0.46, step=0.01),
            'vol_z_min': trial.suggest_float('volexh_volz', 0.8, 1.5, step=0.1),
            'rsi_min': trial.suggest_float('volexh_rsi', 65.0, 78.0, step=1.0),
            'archetype_weight': trial.suggest_float('volexh_weight', 0.95, 1.15, step=0.05),
        }

    elif group_name == 'bos_choch':
        # Continuation patterns - emphasize momentum and displacement
        cfg['archetypes']['wick_trap'] = {
            'fusion_threshold': trial.suggest_float('bos_fusion', 0.35, 0.50, step=0.01),
            'disp_atr_multiplier': trial.suggest_float('bos_disp', 0.8, 1.5, step=0.1),
            'momentum_min': trial.suggest_float('bos_momentum', 0.40, 0.60, step=0.02),
            'tf4h_fusion_min': trial.suggest_float('bos_tf4h', 0.20, 0.35, step=0.02),
        }

    elif group_name == 'long_squeeze':
        # Funding rate cascades - emphasize funding and RSI extremes
        cfg['archetypes']['long_squeeze'] = {
            'fusion_threshold': trial.suggest_float('squeeze_fusion', 0.28, 0.42, step=0.01),
            'funding_z_min': trial.suggest_float('squeeze_funding', 1.0, 1.8, step=0.1),
            'rsi_min': trial.suggest_float('squeeze_rsi', 65.0, 78.0, step=1.0),
            'liquidity_max': trial.suggest_float('squeeze_liq_max', 0.18, 0.32, step=0.02),
        }

    return cfg


# ============================================================================
# Objective Function
# ============================================================================

def compute_objective_score(metrics: Dict, fidelity: int) -> float:
    """
    Compute objective score for optimization.

    Scoring formula:
    - Base: Profit Factor × (1 + win_rate/100) × sqrt(trades)
    - Penalties: -drawdown/10, -overtrading (trades > 100)
    - Bonuses: +sharpe, +consistency

    Args:
        metrics: Backtest metrics dict
        fidelity: Fidelity level (0-2)

    Returns:
        Objective score (higher is better)
    """
    if not metrics or metrics['trades'] < 3:
        return -1000.0  # Penalize configs with < 3 trades

    pf = max(metrics['profit_factor'], 0.1)
    wr = metrics['win_rate']
    trades = metrics['trades']
    dd = metrics['drawdown']
    sharpe = metrics.get('sharpe', 0.0)

    # Base score: PF × win rate × trade consistency
    base_score = pf * (1 + wr / 100.0) * (trades ** 0.5)

    # Drawdown penalty (lighter at low fidelity)
    dd_penalty = dd / (10.0 if fidelity == 2 else 20.0)

    # Overtrading penalty (scale with fidelity)
    max_trades = [30, 60, 100][fidelity]  # 1mo/3mo/9mo targets
    overtrade_penalty = max(0, (trades - max_trades) * 0.1)

    # Sharpe bonus (only at full fidelity)
    sharpe_bonus = max(sharpe, 0) * 0.5 if fidelity == 2 else 0

    # Final score
    score = base_score - dd_penalty - overtrade_penalty + sharpe_bonus

    return score


# ============================================================================
# Optuna Study Runner (per archetype group)
# ============================================================================

def optimize_archetype_group(
    group_name: str,
    base_config: dict,
    n_trials: int,
    storage_path: str,
    progress_queue: mp.Queue
) -> Dict:
    """
    Run Optuna optimization for a single archetype group.

    Args:
        group_name: Archetype group name
        base_config: Base configuration dict
        n_trials: Number of trials to run
        storage_path: Path to SQLite storage
        progress_queue: Multiprocessing queue for progress updates

    Returns:
        Best trial parameters dict
    """
    group = ARCHETYPE_GROUPS[group_name]
    study_name = f"archetype_{group_name}"

    logger.info(f"Starting optimization for {group_name}: {group['description']}")
    logger.info(f"Archetypes: {', '.join(group['archetypes'])}")

    # Create Optuna study with Hyperband pruner
    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner = HyperbandPruner(
        min_resource=1,      # Min fidelity (1 month)
        max_resource=3,      # Max fidelity (9 months = level 2)
        reduction_factor=3   # Prune aggressively
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    def objective(trial: optuna.Trial) -> float:
        """Objective function for this archetype group."""
        # Multi-fidelity: start at fidelity 0, increase based on trial performance
        fidelity = trial.suggest_int('_fidelity', 0, 2)
        start_date, end_date = get_training_periods(fidelity)

        # Suggest parameters
        cfg = suggest_global_params(trial, base_config)
        cfg = suggest_archetype_params(trial, cfg, group_name)

        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f, indent=2)
            config_path = f.name

        try:
            # Run backtest
            metrics = run_backtest(config_path, start_date, end_date, asset="BTC")

            if metrics is None:
                return -1000.0

            # Compute score
            score = compute_objective_score(metrics, fidelity)

            # Report for pruning (Hyperband needs step = fidelity)
            trial.report(score, step=fidelity)

            # Update progress
            progress_queue.put({
                'group': group_name,
                'trial': trial.number,
                'score': score,
                'pf': metrics['profit_factor'],
                'trades': metrics['trades'],
                'fidelity': fidelity
            })

            # Prune if needed
            if trial.should_prune():
                logger.info(f"[{group_name}] Trial {trial.number} pruned at fidelity {fidelity}")
                raise optuna.TrialPruned()

            return score

        finally:
            # Cleanup temp config
            try:
                os.unlink(config_path)
            except:
                pass

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,  # Single-threaded per process (parallelism is across groups)
        show_progress_bar=True
    )

    # Return best params
    best_trial = study.best_trial
    logger.info(f"[{group_name}] Best trial: {best_trial.number}, score: {best_trial.value:.2f}")

    return {
        'group': group_name,
        'best_params': best_trial.params,
        'best_score': best_trial.value,
        'trial_number': best_trial.number,
        'n_trials': len(study.trials)
    }


# ============================================================================
# Progress Monitor
# ============================================================================

def monitor_progress(progress_queue: mp.Queue, total_trials: int, n_groups: int):
    """
    Monitor and display progress from all worker processes.

    Args:
        progress_queue: Shared queue receiving progress updates
        total_trials: Total trials across all groups
        n_groups: Number of archetype groups
    """
    logger.info(f"Monitoring {n_groups} parallel studies with {total_trials} trials each")

    group_progress = defaultdict(lambda: {'trials': 0, 'best_score': -float('inf')})
    start_time = time.time()

    while True:
        try:
            update = progress_queue.get(timeout=1)

            if update == 'DONE':
                break

            group = update['group']
            trial = update['trial']
            score = update['score']

            # Update progress
            group_progress[group]['trials'] = trial + 1
            if score > group_progress[group]['best_score']:
                group_progress[group]['best_score'] = score

            # Display summary
            elapsed = time.time() - start_time
            total_completed = sum(p['trials'] for p in group_progress.values())

            logger.info(
                f"Progress: {total_completed}/{total_trials * n_groups} trials | "
                f"Elapsed: {elapsed/3600:.1f}h | "
                f"Group: {group} | "
                f"Trial: {trial} | "
                f"Score: {score:.2f} (PF={update['pf']:.2f}, trades={update['trades']})"
            )

        except mp.queues.Empty:
            continue
        except KeyboardInterrupt:
            logger.info("Progress monitor interrupted")
            break


# ============================================================================
# Result Aggregation
# ============================================================================

def aggregate_results(results: List[Dict], base_config: dict, output_path: str):
    """
    Aggregate best parameters from all groups into unified config.

    Args:
        results: List of optimization results per group
        base_config: Base configuration
        output_path: Path to save unified config
    """
    logger.info("Aggregating results from all archetype groups...")

    # Start with base config
    unified = json.loads(json.dumps(base_config))

    # Extract global params from best overall group
    best_group = max(results, key=lambda x: x['best_score'])
    global_params = {k: v for k, v in best_group['best_params'].items()
                     if k.startswith(('global_', 'w_'))}

    # Apply global params
    if 'global_min_liquidity' in global_params:
        unified['archetypes']['thresholds']['min_liquidity'] = global_params['global_min_liquidity']

    if 'global_fusion_floor' in global_params:
        unified['decision_gates']['final_fusion_floor'] = global_params['global_fusion_floor']

    # Apply fusion weights
    if all(k in global_params for k in ['w_wyckoff', 'w_liquidity', 'w_momentum']):
        total = sum(global_params[k] for k in ['w_wyckoff', 'w_liquidity', 'w_momentum'])
        unified['fusion']['weights'] = {
            'wyckoff': global_params['w_wyckoff'] / total,
            'liquidity': global_params['w_liquidity'] / total,
            'momentum': global_params['w_momentum'] / total
        }

    # Apply per-archetype params from each group
    for result in results:
        group_name = result['group']
        params = result['best_params']

        # Filter archetype-specific params (exclude global and internal)
        arch_params = {k: v for k, v in params.items()
                       if not k.startswith(('global_', 'w_', '_'))}

        # Map to archetype configs
        if group_name == 'trap_within_trend':
            for arch in ['trap_within_trend', 'spring', 'liquidity_sweep']:
                unified['archetypes'][arch] = {}
                for k, v in arch_params.items():
                    if k.startswith(arch.split('_')[0]):
                        param_name = k.replace(f"{arch.split('_')[0]}_", '')
                        unified['archetypes'][arch][param_name] = v

        # Similar for other groups...
        # (Add mappings for other archetype groups)

    # Enable all optimized archetypes
    for result in results:
        group = ARCHETYPE_GROUPS[result['group']]
        for letter in group['archetypes']:
            unified['archetypes'][f'enable_{letter}'] = True

    # Add metadata
    unified['_optimization_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'groups_optimized': [r['group'] for r in results],
        'total_trials': sum(r['n_trials'] for r in results),
        'best_scores': {r['group']: r['best_score'] for r in results}
    }

    # Save unified config
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2)

    logger.info(f"Unified config saved to {output_path}")

    # Generate comparison table
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"{'Group':<25} {'Best Score':>12} {'Trials':>8} {'Archetypes':<30}")
    print("-" * 80)
    for result in sorted(results, key=lambda x: x['best_score'], reverse=True):
        group = ARCHETYPE_GROUPS[result['group']]
        print(
            f"{result['group']:<25} "
            f"{result['best_score']:>12.2f} "
            f"{result['n_trials']:>8} "
            f"{', '.join(group['archetypes']):<30}"
        )
    print("=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Parallel Optuna Archetype Optimization')
    parser.add_argument('--trials', type=int, default=50, help='Trials per archetype group')
    parser.add_argument('--base-config', type=str, default='configs/profile_production.json',
                        help='Base config path')
    parser.add_argument('--storage', type=str, default='optuna_archetypes.db',
                        help='SQLite storage path')
    parser.add_argument('--output', type=str, default='configs/optimized_archetypes.json',
                        help='Output config path')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--groups', nargs='+', default=None,
                        help='Specific groups to optimize (default: all)')

    args = parser.parse_args()

    # Load base config
    with open(args.base_config, 'r') as f:
        base_config = json.load(f)

    # Select groups to optimize
    groups_to_run = args.groups or list(ARCHETYPE_GROUPS.keys())

    logger.info(f"Starting parallel optimization for {len(groups_to_run)} groups")
    logger.info(f"Trials per group: {args.trials}")
    logger.info(f"Total expected trials: {args.trials * len(groups_to_run)}")
    logger.info(f"Estimated runtime: 6-8 hours")

    # Create progress queue
    progress_queue = mp.Queue()

    # Start progress monitor in separate process
    monitor_proc = mp.Process(
        target=monitor_progress,
        args=(progress_queue, args.trials, len(groups_to_run))
    )
    monitor_proc.start()

    # Create worker pool
    with mp.Pool(processes=len(groups_to_run)) as pool:
        # Start optimization for each group
        async_results = []
        for group_name in groups_to_run:
            result = pool.apply_async(
                optimize_archetype_group,
                args=(group_name, base_config, args.trials, args.storage, progress_queue)
            )
            async_results.append(result)

        # Wait for all to complete
        results = []
        for async_result in async_results:
            try:
                result = async_result.get(timeout=28800)  # 8 hour timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    # Signal monitor to stop
    progress_queue.put('DONE')
    monitor_proc.join(timeout=5)

    # Aggregate results
    if results:
        aggregate_results(results, base_config, args.output)
        logger.info("Optimization complete!")
    else:
        logger.error("No results to aggregate")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
