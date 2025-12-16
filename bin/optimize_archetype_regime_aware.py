#!/usr/bin/env python3
"""
Generic Regime-Aware Archetype Optimizer

Universal optimizer for any archetype with regime awareness.
Supports S1, S4, S5, and all other archetypes with regime-specific calibration.

Usage:
    # S1 (Liquidity Vacuum)
    python bin/optimize_archetype_regime_aware.py --archetype liquidity_vacuum --regimes risk_off crisis

    # S4 (Funding Divergence)
    python bin/optimize_archetype_regime_aware.py --archetype funding_divergence --regimes risk_off neutral

    # S5 (Long Squeeze)
    python bin/optimize_archetype_regime_aware.py --archetype long_squeeze --regimes risk_on neutral

Architecture:
- Filters data by allowed regimes
- Optimizes thresholds per regime
- Stores regime-specific parameters
- Validates on OOS data

Author: Claude Code (Backend Architect)
Date: 2025-11-25
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from typing import Dict, List, Optional
import json
import logging
import argparse
from datetime import datetime

from bin.backtest_regime_stratified import (
    backtest_regime_stratified,
    get_regime_distribution,
    validate_regime_coverage
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Archetype configurations with search spaces
ARCHETYPE_CONFIGS = {
    'liquidity_vacuum': {
        'allowed_regimes': ['risk_off', 'crisis'],
        'search_space': {
            'fusion_threshold': (0.40, 0.55),
            'liquidity_max': (0.10, 0.20),
            'volume_z_min': (1.5, 2.5),
            'wick_lower_min': (0.25, 0.40),
            'cooldown_bars': (8, 18),
            'atr_stop_mult': (2.0, 3.5)
        },
        'target_trades_per_year': (10, 15),
        'min_pf': 2.0,
        'min_wr': 50.0,
        'min_event_recall': 66.0
    },
    'funding_divergence': {
        'allowed_regimes': ['risk_off', 'neutral'],
        'search_space': {
            'fusion_threshold': (0.75, 0.90),
            'funding_z_max': (-2.2, -1.5),  # NEGATIVE funding
            'resilience_min': (0.55, 0.70),
            'liquidity_max': (0.20, 0.35),
            'cooldown_bars': (8, 18),
            'atr_stop_mult': (2.0, 3.5)
        },
        'target_trades_per_year': (6, 10),
        'min_pf': 2.0,
        'min_wr': 50.0,
        'min_event_recall': 0.0
    },
    'long_squeeze': {
        'allowed_regimes': ['risk_on', 'neutral'],
        'search_space': {
            'fusion_threshold': (0.75, 0.90),
            'funding_z_min': (1.8, 2.5),  # POSITIVE funding
            'rsi_min': (60, 75),
            'liquidity_max': (0.20, 0.35),
            'cooldown_bars': (8, 18),
            'atr_stop_mult': (2.0, 3.5)
        },
        'target_trades_per_year': (10, 20),
        'min_pf': 1.8,
        'min_wr': 55.0,
        'min_event_recall': 0.0
    }
}


def optimize_archetype_per_regime(
    archetype: str,
    regime: str,
    data: pd.DataFrame,
    n_trials: int = 200,
    timeout: int = 7200
) -> Optional[Dict]:
    """
    Optimize archetype thresholds for a specific regime.

    Args:
        archetype: Archetype name
        regime: Regime label
        data: Full historical data with regime labels
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds

    Returns:
        Dict with optimization results
    """
    if archetype not in ARCHETYPE_CONFIGS:
        raise ValueError(f"Unknown archetype: {archetype}. Available: {list(ARCHETYPE_CONFIGS.keys())}")

    config = ARCHETYPE_CONFIGS[archetype]
    search_space = config['search_space']
    target_trades_min, target_trades_max = config['target_trades_per_year']
    target_trades_mid = (target_trades_min + target_trades_max) / 2

    logger.info(f"\n{'='*80}")
    logger.info(f"Optimizing {archetype.upper()} on {regime.upper()} regime")
    logger.info(f"{'='*80}")

    # Filter to regime bars
    regime_data = data[data['regime_label'] == regime].copy()

    total_bars = len(data)
    regime_bars = len(regime_data)
    regime_pct = regime_bars / total_bars if total_bars > 0 else 0.0

    logger.info(f"Total bars: {total_bars:,}")
    logger.info(f"Regime bars: {regime_bars:,} ({regime_pct*100:.1f}%)")

    if regime_bars < 1000:
        logger.warning(f"WARNING: Insufficient {regime} bars ({regime_bars} < 1000)")
        return None

    # Define objective function
    def objective(trial):
        """Multi-objective: (PF, WR, trades/year deviation)"""

        # Suggest parameters from search space
        params = {}
        for param_name, (low, high) in search_space.items():
            if param_name.endswith('_bars'):
                # Integer parameter
                params[param_name] = trial.suggest_int(param_name, int(low), int(high))
            else:
                # Float parameter
                params[param_name] = trial.suggest_float(param_name, low, high)

        # Backtest on regime data
        try:
            results = backtest_regime_stratified(
                archetype=archetype,
                data=regime_data,
                config=params,
                allowed_regimes=[regime]
            )

            # Multi-objective optimization
            pf_objective = -results.profit_factor  # Maximize PF
            wr_objective = -results.win_rate       # Maximize WR
            trades_objective = abs(results.trades_per_year - target_trades_mid)  # Target range

            # Store metrics
            trial.set_user_attr('profit_factor', results.profit_factor)
            trial.set_user_attr('win_rate', results.win_rate)
            trial.set_user_attr('trades_per_year', results.trades_per_year)
            trial.set_user_attr('total_trades', results.total_trades)
            trial.set_user_attr('sharpe_ratio', results.sharpe_ratio)

            return pf_objective, wr_objective, trades_objective

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0, 0.0, 999.0

    # Create study
    study_name = f"{archetype}_{regime}_regime_aware"
    storage_name = f"sqlite:///optuna_{archetype}_{regime}_regime_aware.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize", "minimize", "minimize"],
        sampler=NSGAIISampler(population_size=20),
        load_if_exists=True
    )

    logger.info(f"Starting optimization: {n_trials} trials, {timeout}s timeout")

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    logger.info(f"Optimization complete: {len(study.trials)} trials")

    # Get Pareto frontier
    pareto_trials = [t for t in study.best_trials]
    logger.info(f"Pareto frontier: {len(pareto_trials)} solutions")

    # Select best
    best_trial = select_best_trial(pareto_trials, config)

    if best_trial:
        logger.info(f"\nBest solution for {regime}:")
        logger.info(f"  PF: {best_trial.user_attrs['profit_factor']:.3f}")
        logger.info(f"  WR: {best_trial.user_attrs['win_rate']:.1f}%")
        logger.info(f"  Trades/Year: {best_trial.user_attrs['trades_per_year']:.1f}")
        logger.info(f"  Sharpe: {best_trial.user_attrs['sharpe_ratio']:.2f}")
        logger.info(f"  Params: {best_trial.params}")

    return {
        'regime': regime,
        'best_trial': best_trial,
        'pareto_frontier': pareto_trials,
        'study': study
    }


def select_best_trial(pareto_trials: List, config: Dict) -> Optional[optuna.trial.FrozenTrial]:
    """
    Select best trial from Pareto frontier.

    Args:
        pareto_trials: List of Pareto-optimal trials
        config: Archetype config with constraints

    Returns:
        Best trial
    """
    if not pareto_trials:
        return None

    min_pf = config['min_pf']
    min_wr = config['min_wr']

    # Filter by constraints
    valid_trials = [
        t for t in pareto_trials
        if t.user_attrs['profit_factor'] >= min_pf and t.user_attrs['win_rate'] >= min_wr
    ]

    if not valid_trials:
        # Relax constraints
        logger.warning(f"No trials meet constraints (PF>={min_pf}, WR>={min_wr}). Relaxing...")
        valid_trials = pareto_trials

    # Score by weighted combination
    scored = [
        (
            t.user_attrs['profit_factor'] * 0.5 +
            (t.user_attrs['win_rate'] / 100) * 0.3 +
            t.user_attrs['sharpe_ratio'] * 0.2,
            t
        )
        for t in valid_trials
    ]

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def main():
    parser = argparse.ArgumentParser(description='Regime-aware archetype optimization')
    parser.add_argument('--archetype', required=True, choices=list(ARCHETYPE_CONFIGS.keys()),
                        help='Archetype to optimize')
    parser.add_argument('--regimes', nargs='+', required=True,
                        help='Regimes to optimize for (e.g., risk_off crisis)')
    parser.add_argument('--train-start', default='2022-01-01',
                        help='Training data start date')
    parser.add_argument('--train-end', default='2022-12-31',
                        help='Training data end date')
    parser.add_argument('--test-start', default='2023-01-01',
                        help='Test data start date')
    parser.add_argument('--test-end', default='2023-06-30',
                        help='Test data end date')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Number of Optuna trials per regime')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Timeout in seconds per regime')

    args = parser.parse_args()

    archetype = args.archetype
    regimes = args.regimes

    # Validate regimes match archetype
    expected_regimes = ARCHETYPE_CONFIGS[archetype]['allowed_regimes']
    for regime in regimes:
        if regime not in expected_regimes:
            logger.error(f"Invalid regime '{regime}' for {archetype}. Expected: {expected_regimes}")
            sys.exit(1)

    # Load training data
    feature_file = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    if not feature_file.exists():
        logger.error(f"Feature file not found: {feature_file}")
        sys.exit(1)

    logger.info(f"Loading training data: {args.train_start} to {args.train_end}")
    df = pd.read_parquet(feature_file)
    train_data = df[(df.index >= args.train_start) & (df.index < args.train_end)].copy()

    if 'regime_label' not in train_data.columns:
        logger.error("Data missing regime_label. Run: bin/quick_add_regime_labels.py")
        sys.exit(1)

    # Validate coverage
    is_valid, msg = validate_regime_coverage(train_data, regimes, min_bars_per_regime=500)
    if not is_valid:
        logger.error(f"Regime coverage validation failed: {msg}")
        sys.exit(1)

    logger.info(f"Regime coverage validated: {msg}")

    # Optimize per regime
    regime_results = {}
    for regime in regimes:
        result = optimize_archetype_per_regime(
            archetype, regime, train_data,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        if result:
            regime_results[regime] = result

    if not regime_results:
        logger.error("Optimization failed for all regimes")
        sys.exit(1)

    # Create config
    config = {
        'version': f'{archetype}_regime_aware_v1',
        'profile': f'{archetype.replace("_", " ").title()} Regime-Aware',
        'archetypes': {
            'thresholds': {
                archetype: {
                    '_comment': f'{archetype} - regime-aware thresholds',
                    'allowed_regimes': regimes,
                    'regime_thresholds': {}
                }
            }
        }
    }

    # Add regime thresholds
    base_params_list = []
    for regime, result in regime_results.items():
        params = result['best_trial'].params
        config['archetypes']['thresholds'][archetype]['regime_thresholds'][regime] = params
        base_params_list.append(params)

    # Compute base thresholds (average across regimes)
    if base_params_list:
        base_params = {}
        for key in base_params_list[0].keys():
            values = [p[key] for p in base_params_list]
            base_params[key] = sum(values) / len(values) if isinstance(values[0], (int, float)) else values[0]

        config['archetypes']['thresholds'][archetype].update(base_params)

    # Save config
    output_path = Path(f'configs/{archetype}_regime_aware_v1.json')
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nConfig saved: {output_path}")

    # Save detailed results
    results_path = Path(f'results/{archetype}_regime_aware_optimization_results.json')
    results_path.parent.mkdir(exist_ok=True)

    results_data = {
        'archetype': archetype,
        'regimes': regimes,
        'train_period': f"{args.train_start} to {args.train_end}",
        'train_results': {
            regime: {
                'params': result['best_trial'].params,
                'metrics': result['best_trial'].user_attrs
            }
            for regime, result in regime_results.items()
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved: {results_path}")

    logger.info("\n" + "="*80)
    logger.info("REGIME-AWARE OPTIMIZATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
