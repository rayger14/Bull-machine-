#!/usr/bin/env python3
"""
Optuna Threshold Optimization for Bull Machine

Multi-objective optimization of key config thresholds:
- Objective 1: Maximize Profit Factor
- Objective 2: Minimize Max Drawdown

Optimized Parameters:
- min_liquidity: Minimum liquidity score threshold (0.05 - 0.30)
- fusion_threshold: Entry confidence threshold (0.20 - 0.50)
- volume_z_min: Minimum volume z-score (0.5 - 2.5)
- funding_z_min: Minimum funding rate z-score (0.5 - 2.5)
- archetype_weight: Archetype scoring weight (0.8 - 2.0)

Usage:
    python bin/optuna_thresholds.py --asset ETH --trials 500
    python bin/optuna_thresholds.py --asset BTC --trials 100 --base-config configs/profile_btc_seed.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import subprocess
import re
from datetime import datetime
from typing import Dict, Tuple, Optional
import tempfile

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestRunner:
    """Handles subprocess execution of backtest_knowledge_v2.py"""

    def __init__(
        self,
        asset: str,
        start_date: str,
        end_date: str,
        timeout: int = 60,
        backtest_script: str = "bin/backtest_knowledge_v2.py"
    ):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.timeout = timeout
        self.backtest_script = backtest_script
        self.python_exec = sys.executable

    def run(self, config_path: str) -> Optional[Dict[str, float]]:
        """
        Execute backtest and parse results.

        Returns:
            Dict with keys: profit_factor, max_drawdown, sharpe_ratio, num_trades
            None if backtest failed or timed out
        """
        cmd = [
            self.python_exec,
            self.backtest_script,
            "--asset", self.asset,
            "--start", self.start_date,
            "--end", self.end_date,
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
                logger.warning(f"Backtest failed with code {result.returncode}")
                logger.warning(f"STDERR: {result.stderr[:500]}")
                return None

            return self._parse_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warning(f"Backtest timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            return None

    def _parse_output(self, output: str) -> Optional[Dict[str, float]]:
        """
        Parse backtest output to extract metrics.

        Expected output format:
            Total Trades: 42
            Profit Factor: 1.85
            Max Drawdown: 12.3%
            Sharpe Ratio: 1.23
        """
        metrics = {
            'profit_factor': 0.0,
            'max_drawdown': 100.0,  # Start with worst case
            'sharpe_ratio': 0.0,
            'num_trades': 0
        }

        try:
            # Extract Profit Factor
            pf_match = re.search(r'Profit Factor:\s*([\d.]+)', output)
            if pf_match:
                metrics['profit_factor'] = float(pf_match.group(1))

            # Extract Max Drawdown (convert percentage to decimal)
            dd_match = re.search(r'Max Drawdown:\s*([\d.]+)%', output)
            if dd_match:
                metrics['max_drawdown'] = float(dd_match.group(1))

            # Extract Sharpe Ratio
            sharpe_match = re.search(r'Sharpe Ratio:\s*([\d.-]+)', output)
            if sharpe_match:
                metrics['sharpe_ratio'] = float(sharpe_match.group(1))

            # Extract Total Trades
            trades_match = re.search(r'Total Trades:\s*(\d+)', output)
            if trades_match:
                metrics['num_trades'] = int(trades_match.group(1))

            # Validation: Must have at least some trades
            if metrics['num_trades'] < 5:
                logger.warning(f"Too few trades ({metrics['num_trades']}) - rejecting trial")
                return None

            # Validation: PF must be positive
            if metrics['profit_factor'] <= 0:
                logger.warning(f"Invalid PF ({metrics['profit_factor']}) - rejecting trial")
                return None

            return metrics

        except Exception as e:
            logger.error(f"Error parsing output: {e}")
            logger.debug(f"Output sample: {output[:500]}")
            return None


class ConfigGenerator:
    """Generates trial configs by updating base config with suggested params"""

    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()

    def _load_base_config(self) -> dict:
        """Load and validate base config"""
        try:
            with open(self.base_config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded base config: {self.base_config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Base config not found: {self.base_config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in base config: {e}")
            raise

    def generate(self, params: Dict[str, float]) -> str:
        """
        Generate trial config with suggested parameters.

        Args:
            params: Dict with keys matching optimization parameters

        Returns:
            Path to temporary config file
        """
        # Deep copy base config
        trial_config = json.loads(json.dumps(self.base_config))

        # Update thresholds based on config structure
        # Note: Different configs have different structures, so we need to handle both cases

        # Update fusion threshold (multiple possible locations)
        if 'fusion' in trial_config:
            trial_config['fusion']['entry_threshold_confidence'] = params['fusion_threshold']

        # Update archetype settings
        if 'knowledge_v2' in trial_config and 'archetypes' in trial_config['knowledge_v2']:
            archetypes = trial_config['knowledge_v2']['archetypes']
            if 'ob_high' in archetypes:
                archetypes['ob_high']['archetype_weight'] = params['archetype_weight']
                archetypes['ob_high']['fusion_threshold'] = params['fusion_threshold']
                if 'funding_z_min' in archetypes['ob_high']:
                    archetypes['ob_high']['funding_z_min'] = params['funding_z_min']

        # Update liquidity settings
        if 'liquidity' in trial_config:
            trial_config['liquidity']['min_liquidity'] = params['min_liquidity']

        # Update volume settings (if exists in config)
        if 'momentum' in trial_config and isinstance(trial_config['momentum'], dict):
            if 'volume_z_min' in trial_config['momentum']:
                trial_config['momentum']['volume_z_min'] = params['volume_z_min']

        # Alternative structure: nested archetype configs
        if 'archetypes' in trial_config:
            for archetype_name in trial_config['archetypes']:
                archetype = trial_config['archetypes'][archetype_name]
                if isinstance(archetype, dict):
                    if 'archetype_weight' in archetype:
                        archetype['archetype_weight'] = params['archetype_weight']
                    if 'fusion_threshold' in archetype:
                        archetype['fusion_threshold'] = params['fusion_threshold']

        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            prefix='optuna_trial_'
        )

        json.dump(trial_config, temp_file, indent=2)
        temp_file.close()

        return temp_file.name


class OptunaOptimizer:
    """Multi-objective Optuna optimizer for threshold parameters"""

    def __init__(
        self,
        asset: str,
        base_config: str,
        start_date: str = "2024-01-01",
        end_date: str = "2024-09-30",
        n_trials: int = 500,
        timeout_per_trial: int = 60
    ):
        self.asset = asset
        self.base_config = base_config
        self.start_date = start_date
        self.end_date = end_date
        self.n_trials = n_trials
        self.timeout_per_trial = timeout_per_trial

        self.backtest_runner = BacktestRunner(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout_per_trial
        )

        self.config_generator = ConfigGenerator(base_config)

        # Trial tracking
        self.trial_count = 0
        self.successful_trials = 0
        self.failed_trials = 0

        # Best results tracking
        self.best_pf = 0.0
        self.best_dd = 100.0
        self.best_params = {}

    def objective(self, trial: optuna.Trial) -> Tuple[float, float]:
        """
        Optuna objective function.

        Returns:
            Tuple of (profit_factor, max_drawdown) for multi-objective optimization
            - Maximize profit_factor (direction='maximize')
            - Minimize max_drawdown (direction='minimize')
        """
        self.trial_count += 1

        # Suggest parameters
        params = {
            'min_liquidity': trial.suggest_float('min_liquidity', 0.05, 0.30),
            'fusion_threshold': trial.suggest_float('fusion_threshold', 0.20, 0.50),
            'volume_z_min': trial.suggest_float('volume_z_min', 0.5, 2.5),
            'funding_z_min': trial.suggest_float('funding_z_min', 0.5, 2.5),
            'archetype_weight': trial.suggest_float('archetype_weight', 0.8, 2.0),
        }

        # Generate config
        config_path = self.config_generator.generate(params)

        try:
            # Run backtest
            metrics = self.backtest_runner.run(config_path)

            if metrics is None:
                self.failed_trials += 1
                # Return worst possible values to prune this trial
                return 0.0, 100.0

            self.successful_trials += 1

            pf = metrics['profit_factor']
            dd = metrics['max_drawdown']
            sharpe = metrics['sharpe_ratio']
            trades = metrics['num_trades']

            # Update best tracking
            if pf > self.best_pf:
                self.best_pf = pf
                self.best_params = params.copy()
            if dd < self.best_dd:
                self.best_dd = dd

            # Log progress every 50 trials
            if self.trial_count % 50 == 0:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"PROGRESS UPDATE - Trial {self.trial_count}/{self.n_trials}")
                logger.info("=" * 80)
                logger.info(f"Successful: {self.successful_trials} | Failed: {self.failed_trials}")
                logger.info(f"Best PF so far: {self.best_pf:.2f}")
                logger.info(f"Best DD so far: {self.best_dd:.1f}%")
                logger.info(f"Latest Trial Results:")
                logger.info(f"  PF: {pf:.2f} | DD: {dd:.1f}% | Sharpe: {sharpe:.2f} | Trades: {trades}")
                logger.info(f"  Params: {params}")
                logger.info("=" * 80)
                logger.info("")

            # Store metrics in trial user attributes for later analysis
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('num_trades', trades)

            return pf, dd

        finally:
            # Cleanup temp config file
            try:
                Path(config_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp config {config_path}: {e}")

    def optimize(self) -> optuna.Study:
        """
        Run multi-objective optimization.

        Returns:
            Completed Optuna study object
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("OPTUNA THRESHOLD OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Asset: {self.asset}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Base Config: {self.base_config}")
        logger.info(f"Trials: {self.n_trials}")
        logger.info(f"Timeout per trial: {self.timeout_per_trial}s")
        logger.info("=" * 80)
        logger.info("")

        # Create study with multi-objective optimization
        study = optuna.create_study(
            directions=['maximize', 'minimize'],  # maximize PF, minimize DD
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)  # Don't stop on individual trial failures
        )

        return study

    def save_results(self, study: optuna.Study, output_path: str) -> None:
        """
        Save optimization results and best parameters.

        Args:
            study: Completed Optuna study
            output_path: Path to save best config JSON
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total trials: {self.trial_count}")
        logger.info(f"Successful: {self.successful_trials}")
        logger.info(f"Failed: {self.failed_trials}")
        logger.info("")

        # Get Pareto front trials
        pareto_trials = study.best_trials

        logger.info(f"Pareto Front: {len(pareto_trials)} non-dominated solutions")
        logger.info("")

        # Display top solutions
        logger.info("Top 5 Pareto-Optimal Solutions:")
        logger.info("-" * 80)

        for i, trial in enumerate(pareto_trials[:5], 1):
            pf, dd = trial.values
            sharpe = trial.user_attrs.get('sharpe_ratio', 0.0)
            trades = trial.user_attrs.get('num_trades', 0)

            logger.info(f"Solution {i}:")
            logger.info(f"  PF: {pf:.2f} | DD: {dd:.1f}% | Sharpe: {sharpe:.2f} | Trades: {trades}")
            logger.info(f"  Params:")
            for param, value in trial.params.items():
                logger.info(f"    {param}: {value:.4f}")
            logger.info("")

        # Select best solution (highest PF among Pareto front)
        best_trial = max(pareto_trials, key=lambda t: t.values[0])
        best_pf, best_dd = best_trial.values

        logger.info("=" * 80)
        logger.info("SELECTED BEST SOLUTION (Highest PF on Pareto Front):")
        logger.info("=" * 80)
        logger.info(f"Profit Factor: {best_pf:.2f}")
        logger.info(f"Max Drawdown: {best_dd:.1f}%")
        logger.info(f"Sharpe Ratio: {best_trial.user_attrs.get('sharpe_ratio', 0.0):.2f}")
        logger.info(f"Trades: {best_trial.user_attrs.get('num_trades', 0)}")
        logger.info("")
        logger.info("Best Parameters:")
        for param, value in best_trial.params.items():
            logger.info(f"  {param}: {value:.4f}")
        logger.info("=" * 80)
        logger.info("")

        # Generate final config with best params
        final_config_path = self.config_generator.generate(best_trial.params)

        # Read and save to output path
        with open(final_config_path, 'r') as f:
            final_config = json.load(f)

        # Add metadata
        final_config['_optuna_metadata'] = {
            'optimization_date': datetime.now().isoformat(),
            'asset': self.asset,
            'period': f"{self.start_date} to {self.end_date}",
            'trials': self.trial_count,
            'successful_trials': self.successful_trials,
            'profit_factor': best_pf,
            'max_drawdown': best_dd,
            'sharpe_ratio': best_trial.user_attrs.get('sharpe_ratio', 0.0),
            'num_trades': best_trial.user_attrs.get('num_trades', 0),
            'optimized_params': best_trial.params
        }

        # Save to output path
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(final_config, f, indent=2)

        logger.info(f"Best config saved to: {output_path}")

        # Cleanup temp file
        try:
            Path(final_config_path).unlink()
        except Exception:
            pass

        # Save study object for later analysis
        study_path = output_path.replace('.json', '_study.pkl')
        try:
            import joblib
            joblib.dump(study, study_path)
            logger.info(f"Study object saved to: {study_path}")
        except Exception as e:
            logger.warning(f"Failed to save study object: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Optuna-based threshold optimization for Bull Machine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize ETH with 500 trials
  python bin/optuna_thresholds.py --asset ETH --trials 500

  # Quick test with 50 trials
  python bin/optuna_thresholds.py --asset BTC --trials 50 --timeout 30

  # Custom date range and base config
  python bin/optuna_thresholds.py --asset ETH --start 2024-01-01 --end 2024-06-30 \\
      --base-config configs/profile_eth_seed.json --trials 200
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
        help='Number of optimization trials (default: 500)'
    )

    parser.add_argument(
        '--base-config',
        default='configs/profile_default.json',
        help='Base config file to modify (default: configs/profile_default.json)'
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
        default='configs/auto/best_optuna.json',
        help='Output path for best config (default: configs/auto/best_optuna.json)'
    )

    args = parser.parse_args()

    # Validate base config exists
    if not Path(args.base_config).exists():
        logger.error(f"Base config not found: {args.base_config}")
        # Try to find a suitable alternative
        alternatives = [
            f"configs/profile_{args.asset.lower()}_seed.json",
            "configs/profile_default.json",
            f"configs/knowledge_v2/{args.asset}_baseline.json"
        ]
        for alt in alternatives:
            if Path(alt).exists():
                logger.info(f"Using alternative config: {alt}")
                args.base_config = alt
                break
        else:
            logger.error("No suitable base config found!")
            sys.exit(1)

    # Create optimizer
    optimizer = OptunaOptimizer(
        asset=args.asset,
        base_config=args.base_config,
        start_date=args.start,
        end_date=args.end,
        n_trials=args.trials,
        timeout_per_trial=args.timeout
    )

    # Run optimization
    study = optimizer.optimize()

    # Save results
    optimizer.save_results(study, args.output)

    logger.info("")
    logger.info("Optimization complete!")
    logger.info(f"Best config: {args.output}")
    logger.info("")


if __name__ == '__main__':
    main()
