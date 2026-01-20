#!/usr/bin/env python3
"""
Hyperband/ASHA Optuna Threshold Optimization

Uses adaptive successive halving to prune unpromising trials early.

Multi-fidelity evaluation:
- Rung 0: 1 month (2024-01-01 to 2024-01-31)  → ~6s
- Rung 1: 3 months (2024-01-01 to 2024-03-31) → ~20s
- Rung 2: 9 months (2024-01-01 to 2024-09-30) → ~60s (full)

Expected pruning: 60-70% of trials eliminated at Rung 0 or 1
Runtime: 8.25 hours → 2.2 hours (73% reduction on top of parallel)

Academic Reference:
- Li et al. (2018). "Massively Parallel Hyperband." ICLR 2018.

Usage:
    python bin/optuna_thresholds_hyperband.py --asset BTC --trials 500 --n-jobs 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import tempfile
import subprocess
import re
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import multiprocessing as mp
from functools import partial

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.trial import TrialState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperbandBacktestRunner:
    """
    Multi-fidelity backtest runner with progressive evaluation.

    Evaluates trials on progressively longer time periods:
    - Rung 0: 1 month (cheap evaluation, prune bad params early)
    - Rung 1: 3 months (medium-cost validation)
    - Rung 2: 9 months (full evaluation for best candidates)

    This enables 60-70% pruning rate, reducing effective trials from 500 to ~150-200.
    """

    # Multi-fidelity rungs: (start_date, end_date, months, expected_runtime_sec)
    RUNGS = [
        ("2024-01-01", "2024-01-31", 1, 6),    # Rung 0: 1 month
        ("2024-01-01", "2024-03-31", 3, 20),   # Rung 1: 3 months
        ("2024-01-01", "2024-09-30", 9, 60),   # Rung 2: 9 months (full)
    ]

    def __init__(
        self,
        asset: str,
        base_config_path: str,
        archetype_name: Optional[str] = None,
        timeout: int = 60,
        backtest_script: str = "bin/backtest_knowledge_v2.py"
    ):
        """
        Initialize Hyperband backtest runner.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            base_config_path: Path to base configuration
            archetype_name: Optional archetype to optimize
            timeout: Timeout per trial (seconds)
            backtest_script: Path to backtest script
        """
        self.asset = asset
        self.base_config_path = base_config_path
        self.archetype_name = archetype_name
        self.timeout = timeout
        self.backtest_script = backtest_script
        self.python_exec = sys.executable

        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f)

        # Statistics
        self.trials_by_rung = {i: 0 for i in range(len(self.RUNGS))}
        self.pruned_by_rung = {i: 0 for i in range(len(self.RUNGS))}

    def generate_config(self, params: Dict[str, float]) -> str:
        """
        Generate trial config with suggested parameters.

        Args:
            params: Dict with keys matching optimization parameters

        Returns:
            Path to temporary config file
        """
        # Deep copy base config
        trial_config = json.loads(json.dumps(self.base_config))

        # Update thresholds (same logic as optuna_thresholds.py)
        if 'fusion' in trial_config:
            trial_config['fusion']['entry_threshold_confidence'] = params['fusion_threshold']

        if 'knowledge_v2' in trial_config and 'archetypes' in trial_config['knowledge_v2']:
            archetypes = trial_config['knowledge_v2']['archetypes']
            if self.archetype_name and self.archetype_name in archetypes:
                archetypes[self.archetype_name]['archetype_weight'] = params['archetype_weight']
                archetypes[self.archetype_name]['fusion_threshold'] = params['fusion_threshold']
                if 'funding_z_min' in archetypes[self.archetype_name]:
                    archetypes[self.archetype_name]['funding_z_min'] = params['funding_z_min']

        if 'liquidity' in trial_config:
            trial_config['liquidity']['min_liquidity'] = params['min_liquidity']

        if 'momentum' in trial_config and isinstance(trial_config['momentum'], dict):
            if 'volume_z_min' in trial_config['momentum']:
                trial_config['momentum']['volume_z_min'] = params['volume_z_min']

        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            prefix='optuna_hyperband_'
        )
        json.dump(trial_config, temp_file, indent=2)
        temp_file.close()

        return temp_file.name

    def run_backtest(
        self,
        config_path: str,
        start_date: str,
        end_date: str
    ) -> Optional[Dict[str, float]]:
        """
        Execute backtest subprocess and parse results.

        Args:
            config_path: Path to config file
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dict with metrics or None if failed
        """
        cmd = [
            self.python_exec,
            self.backtest_script,
            "--asset", self.asset,
            "--start", start_date,
            "--end", end_date,
            "--config", config_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False
            )

            if result.returncode != 0:
                return None

            return self._parse_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warning(f"Backtest timeout for {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return None

    def _parse_output(self, output: str) -> Optional[Dict[str, float]]:
        """Parse backtest output to extract metrics."""
        metrics = {
            'profit_factor': 0.0,
            'max_drawdown': 100.0,
            'sharpe_ratio': 0.0,
            'num_trades': 0
        }

        try:
            pf_match = re.search(r'Profit Factor:\s*([\d.]+)', output)
            if pf_match:
                metrics['profit_factor'] = float(pf_match.group(1))

            dd_match = re.search(r'Max Drawdown:\s*([\d.]+)%', output)
            if dd_match:
                metrics['max_drawdown'] = float(dd_match.group(1))

            sharpe_match = re.search(r'Sharpe Ratio:\s*([\d.-]+)', output)
            if sharpe_match:
                metrics['sharpe_ratio'] = float(sharpe_match.group(1))

            trades_match = re.search(r'Total Trades:\s*(\d+)', output)
            if trades_match:
                metrics['num_trades'] = int(trades_match.group(1))

            # Validation
            if metrics['num_trades'] < 3:
                return None
            if metrics['profit_factor'] <= 0:
                return None

            return metrics

        except Exception as e:
            logger.error(f"Error parsing output: {e}")
            return None

    def objective_with_pruning(self, trial: optuna.Trial) -> float:
        """
        Objective function with intermediate reporting for Hyperband pruning.

        Progressive evaluation:
        1. Run 1-month backtest (Rung 0) - if bad, prune immediately
        2. Report intermediate value to Optuna pruner
        3. If pruner decides to continue, run 3-month backtest (Rung 1)
        4. If still promising, run full 9-month backtest (Rung 2)

        Returns:
            Final objective score (higher is better)
        """
        # Suggest parameters
        params = {
            'min_liquidity': trial.suggest_float('min_liquidity', 0.05, 0.30),
            'fusion_threshold': trial.suggest_float('fusion_threshold', 0.20, 0.50),
            'volume_z_min': trial.suggest_float('volume_z_min', 0.5, 2.5),
            'funding_z_min': trial.suggest_float('funding_z_min', 0.5, 2.5),
            'archetype_weight': trial.suggest_float('archetype_weight', 0.8, 2.0),
        }

        # Generate config
        config_path = self.generate_config(params)

        try:
            # Progressive evaluation through rungs
            for rung_idx, (start, end, months, expected_time) in enumerate(self.RUNGS):
                self.trials_by_rung[rung_idx] += 1

                # Run backtest for this rung
                logger.info(
                    f"[Trial {trial.number}] Rung {rung_idx}: {months}mo backtest ({start} to {end})"
                )

                metrics = self.run_backtest(config_path, start, end)

                if metrics is None:
                    # Early failure - prune immediately
                    logger.warning(f"[Trial {trial.number}] Rung {rung_idx}: FAILED")
                    self.pruned_by_rung[rung_idx] += 1
                    raise optuna.TrialPruned()

                # Compute objective score
                pf = metrics['profit_factor']
                dd = metrics['max_drawdown']
                sharpe = metrics['sharpe_ratio']
                trades = metrics['num_trades']

                # Multi-objective score: PF - 0.1 * DD + 0.5 * Sharpe
                # Normalized to favor high PF, low DD, high Sharpe
                score = pf - 0.1 * dd + 0.5 * sharpe

                logger.info(
                    f"[Trial {trial.number}] Rung {rung_idx}: "
                    f"PF={pf:.2f}, DD={dd:.1f}%, Sharpe={sharpe:.2f}, Score={score:.2f}"
                )

                # Report intermediate value to pruner
                trial.report(score, step=rung_idx)

                # Check if should prune (Optuna's Successive Halving decides)
                if trial.should_prune():
                    logger.info(
                        f"[Trial {trial.number}] Rung {rung_idx}: PRUNED "
                        f"(below median of concurrent trials)"
                    )
                    self.pruned_by_rung[rung_idx] += 1
                    raise optuna.TrialPruned()

            # Store final metrics
            trial.set_user_attr('profit_factor', pf)
            trial.set_user_attr('max_drawdown', dd)
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('num_trades', trades)

            logger.info(f"[Trial {trial.number}] COMPLETED - Final score: {score:.2f}")
            return score

        finally:
            # Cleanup temp config
            Path(config_path).unlink(missing_ok=True)


def optimize_archetype_hyperband(
    archetype_name: str,
    base_config_path: str,
    n_trials: int,
    asset: str,
    output_dir: str,
    min_resource: int = 1,
    reduction_factor: int = 3
) -> dict:
    """
    Optimize single archetype using Hyperband pruning.

    Args:
        archetype_name: Archetype to optimize
        base_config_path: Base configuration file
        n_trials: Number of trials
        asset: Asset symbol
        output_dir: Output directory
        min_resource: Minimum resource (rung 0)
        reduction_factor: Pruning aggressiveness (keep 1/reduction_factor)

    Returns:
        Summary dict with results
    """
    logger.info(f"[{archetype_name}] Starting Hyperband optimization ({n_trials} trials)")

    # Create runner
    runner = HyperbandBacktestRunner(
        asset=asset,
        base_config_path=base_config_path,
        archetype_name=archetype_name
    )

    # Create pruner (Successive Halving)
    pruner = SuccessiveHalvingPruner(
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        min_early_stopping_rate=0
    )

    # Create sampler (TPE with multivariate)
    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=10)

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'hyperband_{archetype_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    # Run optimization
    study.optimize(
        runner.objective_with_pruning,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )

    # Compute statistics
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    if not completed:
        logger.error(f"[{archetype_name}] No completed trials!")
        return {
            'archetype': archetype_name,
            'error': 'No completed trials',
            'n_trials': n_trials
        }

    best_trial = study.best_trial

    result = {
        'archetype': archetype_name,
        'best_score': best_trial.value,
        'best_pf': best_trial.user_attrs.get('profit_factor', 0.0),
        'best_dd': best_trial.user_attrs.get('max_drawdown', 100.0),
        'best_sharpe': best_trial.user_attrs.get('sharpe_ratio', 0.0),
        'params': best_trial.params,
        'n_trials': n_trials,
        'n_completed': len(completed),
        'n_pruned': len(pruned),
        'pruning_rate': len(pruned) / n_trials,
        'trials_by_rung': dict(runner.trials_by_rung),
        'pruned_by_rung': dict(runner.pruned_by_rung)
    }

    # Save results
    output_path = Path(output_dir) / f"{archetype_name}_hyperband_best.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"[{archetype_name}] Completed: "
        f"Score={result['best_score']:.2f}, "
        f"PF={result['best_pf']:.2f}, "
        f"DD={result['best_dd']:.1f}%, "
        f"Pruned={result['pruning_rate']*100:.1f}%"
    )

    return result


def parallel_hyperband_optimize(
    archetypes: List[str],
    base_config: str,
    n_trials_per_archetype: int,
    asset: str,
    output_dir: str,
    n_jobs: int = 4
) -> Dict[str, dict]:
    """
    Run parallel Hyperband optimization across archetypes.

    Combines parallel execution with Hyperband pruning for maximum speedup.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("PARALLEL HYPERBAND OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Archetypes: {len(archetypes)}")
    logger.info(f"Trials per archetype: {n_trials_per_archetype}")
    logger.info(f"Parallel workers: {n_jobs}")
    logger.info(f"Expected pruning rate: 60-70%")
    logger.info(f"Estimated runtime: 2-3 hours")
    logger.info("=" * 80)
    logger.info("")

    start_time = datetime.now()

    # Create partial function
    optimize_fn = partial(
        optimize_archetype_hyperband,
        base_config_path=base_config,
        n_trials=n_trials_per_archetype,
        asset=asset,
        output_dir=output_dir
    )

    # Run parallel
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(optimize_fn, archetypes)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Aggregate results
    results_dict = {r['archetype']: r for r in results}

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'asset': asset,
        'archetypes': archetypes,
        'n_trials_per_archetype': n_trials_per_archetype,
        'n_jobs': n_jobs,
        'runtime_hours': duration / 3600,
        'baseline_runtime_hours': 33.3,
        'speedup': 33.3 / (duration / 3600),
        'results': results_dict
    }

    summary_path = Path(output_dir) / "hyperband_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("HYPERBAND OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Runtime: {duration / 3600:.2f} hours")
    logger.info(f"Baseline: 33.3 hours")
    logger.info(f"Speedup: {33.3 / (duration / 3600):.1f}×")
    logger.info("")

    for arch, res in results_dict.items():
        if 'error' in res:
            logger.error(f"{arch}: FAILED")
        else:
            logger.info(
                f"{arch}: Score={res['best_score']:.2f}, "
                f"PF={res['best_pf']:.2f}, DD={res['best_dd']:.1f}%, "
                f"Pruned={res['pruning_rate']*100:.0f}%"
            )

    return results_dict


def main():
    parser = argparse.ArgumentParser(description='Hyperband Optuna optimization')
    parser.add_argument('--asset', required=True, choices=['BTC', 'ETH', 'SPY'])
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--base-config', default='configs/profile_default.json')
    parser.add_argument(
        '--archetypes',
        nargs='+',
        default=['order_block_retest', 'wick_trap', 'trap_within_trend', 'volume_exhaustion']
    )
    parser.add_argument('--output', default='results/hyperband_optuna')

    args = parser.parse_args()

    if not Path(args.base_config).exists():
        logger.error(f"Base config not found: {args.base_config}")
        sys.exit(1)

    results = parallel_hyperband_optimize(
        archetypes=args.archetypes,
        base_config=args.base_config,
        n_trials_per_archetype=args.trials,
        asset=args.asset,
        output_dir=args.output,
        n_jobs=args.n_jobs
    )

    sys.exit(0 if all('error' not in r for r in results.values()) else 1)


if __name__ == '__main__':
    main()
