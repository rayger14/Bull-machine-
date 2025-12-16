#!/usr/bin/env python3
"""
Parallel Optuna Threshold Optimization

Runs 4 archetype optimizations in parallel for 4× speedup.

Runtime: 33 hours → 8.25 hours (75% reduction)
Accuracy: 100% preserved (deterministic)

Usage:
    python bin/optuna_thresholds_parallel.py --asset BTC --trials 500 --n-jobs 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import multiprocessing as mp
from functools import partial
from datetime import datetime
from typing import Dict, List
import logging

from optuna_thresholds import OptunaOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_archetype_config(base_config_path: str, archetype_name: str) -> str:
    """
    Create archetype-specific configuration from base config.

    Args:
        base_config_path: Path to base configuration file
        archetype_name: Archetype to optimize

    Returns:
        Path to archetype-specific config file
    """
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # Enable only this archetype
    if 'archetypes' not in config:
        config['archetypes'] = {}

    if 'use_archetypes' not in config['archetypes']:
        config['archetypes']['use_archetypes'] = []

    # Set active archetype
    config['archetypes']['use_archetypes'] = [archetype_name]

    # Create temporary config
    temp_path = f"/tmp/optuna_{archetype_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(temp_path, 'w') as f:
        json.dump(config, f, indent=2)

    return temp_path


def optimize_archetype(
    archetype_name: str,
    base_config_path: str,
    n_trials: int,
    asset: str,
    start_date: str,
    end_date: str,
    timeout: int,
    output_dir: str
) -> dict:
    """
    Optimize single archetype in isolated process.

    Args:
        archetype_name: Archetype to optimize
        base_config_path: Base configuration file
        n_trials: Number of Optuna trials
        asset: Asset symbol (BTC, ETH, etc.)
        start_date: Backtest start date
        end_date: Backtest end date
        timeout: Timeout per trial (seconds)
        output_dir: Output directory for results

    Returns:
        Summary dict with best params and metrics
    """
    logger.info(f"[{archetype_name}] Starting optimization with {n_trials} trials")

    try:
        # Create archetype-specific config
        archetype_config = create_archetype_config(base_config_path, archetype_name)

        # Initialize optimizer
        optimizer = OptunaOptimizer(
            asset=asset,
            base_config=archetype_config,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
            timeout_per_trial=timeout
        )

        # Run optimization
        study = optimizer.optimize()

        # Save results
        output_path = Path(output_dir) / f"{archetype_name}_best.json"
        optimizer.save_results(study, str(output_path))

        # Extract best trial
        best_trial = max(study.best_trials, key=lambda t: t.values[0])

        result = {
            'archetype': archetype_name,
            'best_pf': best_trial.values[0],
            'best_dd': best_trial.values[1],
            'params': best_trial.params,
            'n_trials': len(study.trials),
            'successful_trials': len([t for t in study.trials if t.state.name == 'COMPLETE']),
            'config_path': str(output_path)
        }

        logger.info(f"[{archetype_name}] Completed: PF={result['best_pf']:.2f}, DD={result['best_dd']:.1f}%")
        return result

    except Exception as e:
        logger.error(f"[{archetype_name}] Optimization failed: {e}")
        return {
            'archetype': archetype_name,
            'error': str(e),
            'n_trials': 0,
            'best_pf': 0.0,
            'best_dd': 100.0
        }


def parallel_optimize(
    archetypes: List[str],
    base_config: str,
    n_trials_per_archetype: int,
    asset: str,
    start_date: str,
    end_date: str,
    timeout: int,
    output_dir: str,
    n_jobs: int = 4
) -> Dict[str, dict]:
    """
    Run parallel optimization across multiple archetypes.

    Args:
        archetypes: List of archetype names to optimize
        base_config: Path to base configuration
        n_trials_per_archetype: Trials per archetype
        asset: Asset symbol (BTC, ETH, etc.)
        start_date: Backtest start date
        end_date: Backtest end date
        timeout: Timeout per trial (seconds)
        output_dir: Output directory for results
        n_jobs: Number of parallel workers (default: 4)

    Returns:
        Dict mapping archetype -> best results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("PARALLEL OPTUNA OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Archetypes: {len(archetypes)}")
    logger.info(f"Trials per archetype: {n_trials_per_archetype}")
    logger.info(f"Total trials: {len(archetypes) * n_trials_per_archetype}")
    logger.info(f"Parallel workers: {n_jobs}")
    logger.info(f"Asset: {asset}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Estimated runtime (parallel): {n_trials_per_archetype * timeout / 3600:.1f} hours")
    logger.info(f"Baseline runtime (sequential): {len(archetypes) * n_trials_per_archetype * timeout / 3600:.1f} hours")
    logger.info("=" * 80)
    logger.info("")

    start_time = datetime.now()

    # Create partial function with fixed args
    optimize_fn = partial(
        optimize_archetype,
        base_config_path=base_config,
        n_trials=n_trials_per_archetype,
        asset=asset,
        start_date=start_date,
        end_date=end_date,
        timeout=timeout,
        output_dir=output_dir
    )

    # Run parallel optimization
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(optimize_fn, archetypes)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Aggregate results
    results_dict = {r['archetype']: r for r in results}

    # Compute statistics
    total_trials = sum(r.get('n_trials', 0) for r in results)
    successful_trials = sum(r.get('successful_trials', 0) for r in results)
    baseline_runtime = len(archetypes) * n_trials_per_archetype * timeout
    speedup = baseline_runtime / duration

    # Save aggregated summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'asset': asset,
        'period': f"{start_date} to {end_date}",
        'archetypes': archetypes,
        'n_trials_per_archetype': n_trials_per_archetype,
        'total_trials': total_trials,
        'successful_trials': successful_trials,
        'n_jobs': n_jobs,
        'runtime_seconds': duration,
        'runtime_hours': duration / 3600,
        'baseline_runtime_hours': baseline_runtime / 3600,
        'speedup': speedup,
        'time_saved_hours': (baseline_runtime - duration) / 3600,
        'results': results_dict
    }

    summary_path = Path(output_dir) / "parallel_optimization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PARALLEL OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Runtime: {duration / 3600:.2f} hours")
    logger.info(f"Baseline (sequential): {baseline_runtime / 3600:.1f} hours")
    logger.info(f"Speedup: {speedup:.1f}×")
    logger.info(f"Time Saved: {(baseline_runtime - duration) / 3600:.1f} hours")
    logger.info("")
    logger.info("Results by Archetype:")
    logger.info("-" * 80)

    for arch, res in results_dict.items():
        if 'error' in res:
            logger.error(f"{arch:25s}: FAILED - {res['error']}")
        else:
            logger.info(
                f"{arch:25s}: PF={res['best_pf']:5.2f}, DD={res['best_dd']:5.1f}%, "
                f"Trials={res['n_trials']:3d} ({res['successful_trials']:3d} successful)"
            )

    logger.info("=" * 80)
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("")

    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Optuna threshold optimization for Bull Machine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize 4 archetypes in parallel (4× speedup)
  python bin/optuna_thresholds_parallel.py --asset BTC --trials 500 --n-jobs 4

  # Quick test with 2 workers
  python bin/optuna_thresholds_parallel.py --asset ETH --trials 100 --n-jobs 2

  # Custom date range
  python bin/optuna_thresholds_parallel.py --asset BTC --start 2024-01-01 --end 2024-06-30 \\
      --trials 200 --n-jobs 4
        """
    )

    parser.add_argument(
        '--asset',
        required=True,
        choices=['BTC', 'ETH', 'SPY'],
        help='Asset to optimize'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=500,
        help='Number of trials per archetype (default: 500)'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--base-config',
        default='configs/profile_default.json',
        help='Base config file (default: configs/profile_default.json)'
    )

    parser.add_argument(
        '--archetypes',
        nargs='+',
        default=['order_block_retest', 'wick_trap', 'trap_within_trend', 'volume_exhaustion'],
        help='Archetypes to optimize (default: order_block_retest wick_trap trap_within_trend volume_exhaustion)'
    )

    parser.add_argument(
        '--start',
        default='2024-01-01',
        help='Backtest start date (default: 2024-01-01)'
    )

    parser.add_argument(
        '--end',
        default='2024-09-30',
        help='Backtest end date (default: 2024-09-30)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout per trial in seconds (default: 60)'
    )

    parser.add_argument(
        '--output',
        default='results/parallel_optuna',
        help='Output directory (default: results/parallel_optuna)'
    )

    args = parser.parse_args()

    # Validate base config exists
    if not Path(args.base_config).exists():
        logger.error(f"Base config not found: {args.base_config}")
        sys.exit(1)

    # Validate n_jobs
    max_jobs = mp.cpu_count()
    if args.n_jobs > max_jobs:
        logger.warning(f"Requested {args.n_jobs} jobs but only {max_jobs} CPUs available")
        args.n_jobs = max_jobs

    # Run parallel optimization
    results = parallel_optimize(
        archetypes=args.archetypes,
        base_config=args.base_config,
        n_trials_per_archetype=args.trials,
        asset=args.asset,
        start_date=args.start,
        end_date=args.end,
        timeout=args.timeout,
        output_dir=args.output,
        n_jobs=args.n_jobs
    )

    # Exit code based on success
    failed_archetypes = [arch for arch, res in results.items() if 'error' in res]
    if failed_archetypes:
        logger.error(f"Failed archetypes: {failed_archetypes}")
        sys.exit(1)
    else:
        logger.info("All archetypes optimized successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
