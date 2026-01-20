#!/usr/bin/env python3
"""
S1 (Liquidity Vacuum) Regime-Aware Optimizer

Optimizes S1 thresholds SEPARATELY for each allowed regime (risk_off, crisis).
This is the LEARNING CORTEX implementation - training on relevant data only.

Key Innovation:
- Filters historical bars by regime BEFORE optimization
- Optimizes thresholds per archetype-regime pair
- Stores regime-specific parameters in configs
- Prevents cross-regime contamination

Architecture:
1. Load historical data with regime labels
2. Filter to allowed regimes (risk_off + crisis for S1)
3. Run Optuna optimization on regime-filtered bars
4. Store per-regime thresholds in config
5. Validate on OOS data (regime-stratified)

Expected Results:
- Crisis thresholds: More aggressive (higher risk/reward in extreme events)
- Risk_off thresholds: More conservative (steady bear markets)
- Overall PF improvement: 15-25% vs non-regime-aware

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
from typing import Dict, List, Tuple
import json
import logging
from datetime import datetime

from bin.backtest_regime_stratified import (
    backtest_regime_stratified,
    get_regime_distribution,
    validate_regime_coverage
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global cache for feature data
FEATURE_CACHE = {}

# Ground truth events for S1 validation
GROUND_TRUTH_EVENTS = [
    '2022-05-12',  # LUNA death spiral
    '2022-06-18',  # Market capitulation
    '2022-11-09'   # FTX collapse
]


def load_feature_data_with_regimes(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load feature data with regime labels.

    Args:
        start_date: Start date (ISO format)
        end_date: End date (ISO format)

    Returns:
        DataFrame with regime_label column
    """
    cache_key = f"{start_date}_{end_date}"

    if cache_key not in FEATURE_CACHE:
        feature_file = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

        if not feature_file.exists():
            raise FileNotFoundError(f"Feature store not found: {feature_file}")

        logger.info(f"Loading feature data: {start_date} to {end_date}")
        df = pd.read_parquet(feature_file)
        df = df[(df.index >= start_date) & (df.index <= end_date)].copy()

        # Validate regime labels exist
        if 'regime_label' not in df.columns:
            raise ValueError(
                "Feature data missing regime_label column. "
                "Run regime classifier first: bin/quick_add_regime_labels.py"
            )

        FEATURE_CACHE[cache_key] = df
        logger.info(f"Cached {len(df):,} bars with regime labels")

        # Log regime distribution
        regime_dist = get_regime_distribution(df)
        logger.info("Regime distribution:")
        for regime, pct in sorted(regime_dist.items()):
            logger.info(f"  {regime}: {pct*100:.1f}%")

    return FEATURE_CACHE[cache_key].copy()


def optimize_s1_per_regime(
    regime: str,
    data: pd.DataFrame,
    n_trials: int = 200,
    timeout: int = 7200
) -> Dict:
    """
    Optimize S1 thresholds on specific regime bars only.

    Args:
        regime: 'risk_off' or 'crisis' (S1 allowed regimes)
        data: Full historical dataframe with regime_label column
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds (2 hours default)

    Returns:
        Dict with best parameters and Pareto frontier
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Optimizing S1 on {regime.upper()} regime")
    logger.info(f"{'='*80}")

    # REGIME STRATIFICATION - Filter to regime bars only
    regime_data = data[data['regime_label'] == regime].copy()

    total_bars = len(data)
    regime_bars = len(regime_data)
    regime_pct = regime_bars / total_bars if total_bars > 0 else 0.0

    logger.info(f"Total bars: {total_bars:,}")
    logger.info(f"Regime bars: {regime_bars:,} ({regime_pct*100:.1f}%)")

    if regime_bars < 1000:
        logger.warning(f"WARNING: Too few {regime} bars for optimization ({regime_bars} < 1000)")
        return None

    # OPTUNA OPTIMIZATION on regime-filtered data
    def objective(trial):
        """Multi-objective: (PF, event_recall, trades/year)"""

        # Suggest parameters
        params = {
            'fusion_threshold': trial.suggest_float('fusion_threshold', 0.40, 0.55),
            'liquidity_max': trial.suggest_float('liquidity_max', 0.10, 0.20),
            'volume_z_min': trial.suggest_float('volume_z_min', 1.5, 2.5),
            'wick_lower_min': trial.suggest_float('wick_lower_min', 0.25, 0.40),
            'cooldown_bars': trial.suggest_int('cooldown_bars', 8, 18),
            'atr_stop_mult': trial.suggest_float('atr_stop_mult', 2.0, 3.5)
        }

        # Backtest on REGIME DATA ONLY
        try:
            results = backtest_regime_stratified(
                archetype='liquidity_vacuum',
                data=regime_data,
                config=params,
                allowed_regimes=[regime],
                ground_truth_events=GROUND_TRUTH_EVENTS
            )

            # Multi-objective optimization
            # Objective 1: Maximize PF (minimize negative)
            pf_objective = -results.profit_factor

            # Objective 2: Maximize event recall (minimize negative)
            recall_objective = -results.event_recall

            # Objective 3: Target 10-15 trades/year (minimize deviation)
            target_trades = 12.5
            trades_objective = abs(results.trades_per_year - target_trades)

            # Store metrics for analysis
            trial.set_user_attr('profit_factor', results.profit_factor)
            trial.set_user_attr('win_rate', results.win_rate)
            trial.set_user_attr('event_recall', results.event_recall)
            trial.set_user_attr('trades_per_year', results.trades_per_year)
            trial.set_user_attr('total_trades', results.total_trades)

            return pf_objective, recall_objective, trades_objective

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return worst values
            return 0.0, 0.0, 999.0

    # Create study
    study_name = f"s1_{regime}_regime_aware"
    storage_name = f"sqlite:///optuna_s1_{regime}_regime_aware.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize", "minimize", "minimize"],  # All objectives minimize
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

    # Select best solution (balance PF and recall)
    best_trial = select_best_from_pareto(pareto_trials, regime)

    if best_trial:
        logger.info(f"\nBest solution for {regime}:")
        logger.info(f"  PF: {best_trial.user_attrs['profit_factor']:.3f}")
        logger.info(f"  WR: {best_trial.user_attrs['win_rate']:.1f}%")
        logger.info(f"  Event Recall: {best_trial.user_attrs['event_recall']:.1f}%")
        logger.info(f"  Trades/Year: {best_trial.user_attrs['trades_per_year']:.1f}")
        logger.info(f"  Thresholds: {best_trial.params}")

    return {
        'regime': regime,
        'best_trial': best_trial,
        'pareto_frontier': pareto_trials,
        'study': study
    }


def select_best_from_pareto(
    pareto_trials: List,
    regime: str
) -> optuna.trial.FrozenTrial:
    """
    Select best solution from Pareto frontier.

    Selection criteria:
    - Crisis: Prioritize event recall (must catch LUNA, FTX)
    - Risk_off: Balance PF and recall

    Args:
        pareto_trials: List of Pareto-optimal trials
        regime: Regime name for selection strategy

    Returns:
        Best trial
    """
    if not pareto_trials:
        return None

    # Crisis: Prioritize event recall
    if regime == 'crisis':
        # Sort by recall, then PF
        scored = [
            (
                t.user_attrs['event_recall'] * 0.6 + t.user_attrs['profit_factor'] * 0.4,
                t
            )
            for t in pareto_trials
            if t.user_attrs['event_recall'] >= 66.0  # Must catch 2/3 events minimum
        ]
    else:
        # Risk_off: Balance PF and recall
        scored = [
            (
                t.user_attrs['profit_factor'] * 0.5 + t.user_attrs['event_recall'] * 0.3 + (t.user_attrs['win_rate'] / 100) * 0.2,
                t
            )
            for t in pareto_trials
        ]

    if not scored:
        # Fallback: Just pick highest PF
        scored = [(t.user_attrs['profit_factor'], t) for t in pareto_trials]

    # Sort descending
    scored.sort(reverse=True, key=lambda x: x[0])

    return scored[0][1]


def create_regime_aware_config(
    base_config: Dict,
    risk_off_params: Dict,
    crisis_params: Dict
) -> Dict:
    """
    Create config with per-regime thresholds.

    Args:
        base_config: Base S1 configuration
        risk_off_params: Optimized parameters for risk_off regime
        crisis_params: Optimized parameters for crisis regime

    Returns:
        Config with regime_thresholds structure
    """
    config = base_config.copy()

    # Ensure archetypes.thresholds.liquidity_vacuum exists
    if 'archetypes' not in config:
        config['archetypes'] = {}
    if 'thresholds' not in config['archetypes']:
        config['archetypes']['thresholds'] = {}
    if 'liquidity_vacuum' not in config['archetypes']['thresholds']:
        config['archetypes']['thresholds']['liquidity_vacuum'] = {}

    s1_config = config['archetypes']['thresholds']['liquidity_vacuum']

    # Add metadata
    s1_config['_comment'] = "S1 Liquidity Vacuum - regime-aware thresholds"
    s1_config['allowed_regimes'] = ['risk_off', 'crisis']

    # Add base thresholds (average of regime thresholds)
    s1_config['fusion_threshold'] = (
        risk_off_params['fusion_threshold'] + crisis_params['fusion_threshold']
    ) / 2
    s1_config['liquidity_max'] = (
        risk_off_params['liquidity_max'] + crisis_params['liquidity_max']
    ) / 2
    s1_config['volume_z_min'] = (
        risk_off_params['volume_z_min'] + crisis_params['volume_z_min']
    ) / 2
    s1_config['wick_lower_min'] = (
        risk_off_params['wick_lower_min'] + crisis_params['wick_lower_min']
    ) / 2
    s1_config['cooldown_bars'] = int((
        risk_off_params['cooldown_bars'] + crisis_params['cooldown_bars']
    ) / 2)
    s1_config['atr_stop_mult'] = (
        risk_off_params['atr_stop_mult'] + crisis_params['atr_stop_mult']
    ) / 2

    # Add regime-specific thresholds
    s1_config['regime_thresholds'] = {
        'risk_off': risk_off_params,
        'crisis': crisis_params
    }

    return config


def validate_regime_aware_config(
    config: Dict,
    test_data: pd.DataFrame
) -> Dict:
    """
    Validate regime-aware config on OOS data.

    Args:
        config: Config with regime_thresholds
        test_data: OOS test data with regime labels

    Returns:
        Dict with validation metrics per regime
    """
    logger.info("\n" + "="*80)
    logger.info("OOS VALIDATION (Regime-Stratified)")
    logger.info("="*80)

    results = {}

    s1_params = config['archetypes']['thresholds']['liquidity_vacuum']
    regime_thresholds = s1_params['regime_thresholds']

    for regime, params in regime_thresholds.items():
        logger.info(f"\nValidating {regime} regime on OOS data...")

        try:
            result = backtest_regime_stratified(
                archetype='liquidity_vacuum',
                data=test_data,
                config=params,
                allowed_regimes=[regime],
                ground_truth_events=GROUND_TRUTH_EVENTS
            )

            results[regime] = result.to_dict()

            logger.info(f"{regime} OOS Results:")
            logger.info(f"  PF: {result.profit_factor:.3f}")
            logger.info(f"  WR: {result.win_rate:.1f}%")
            logger.info(f"  Event Recall: {result.event_recall:.1f}%")
            logger.info(f"  Trades: {result.total_trades}")

        except Exception as e:
            logger.error(f"Validation failed for {regime}: {e}")
            results[regime] = None

    return results


def main():
    """Main optimization workflow"""

    # Load training data (2022)
    logger.info("Loading training data (2022)...")
    train_data = load_feature_data_with_regimes('2022-01-01', '2022-12-31')

    # Validate regime coverage
    is_valid, msg = validate_regime_coverage(
        train_data,
        required_regimes=['risk_off', 'crisis'],
        min_bars_per_regime=500
    )

    if not is_valid:
        logger.error(f"Regime coverage validation failed: {msg}")
        return

    logger.info(f"Regime coverage validated: {msg}")

    # Optimize per regime
    risk_off_results = optimize_s1_per_regime('risk_off', train_data, n_trials=200)
    crisis_results = optimize_s1_per_regime('crisis', train_data, n_trials=200)

    if not risk_off_results or not crisis_results:
        logger.error("Optimization failed for one or more regimes")
        return

    # Extract best parameters
    risk_off_params = risk_off_results['best_trial'].params
    crisis_params = crisis_results['best_trial'].params

    # Create regime-aware config
    base_config = {
        'version': 's1_regime_aware_v1',
        'profile': 'S1 Regime-Aware Optimization'
    }

    config = create_regime_aware_config(base_config, risk_off_params, crisis_params)

    # Save config
    output_path = Path('configs/s1_regime_aware_v1.json')
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nRegime-aware config saved: {output_path}")

    # Load test data (2023 H1)
    logger.info("\nLoading test data (2023 H1)...")
    test_data = load_feature_data_with_regimes('2023-01-01', '2023-06-30')

    # OOS validation
    oos_results = validate_regime_aware_config(config, test_data)

    # Save results
    results_path = Path('results/s1_regime_aware_optimization_results.json')
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'train_results': {
                'risk_off': {
                    'params': risk_off_params,
                    'metrics': risk_off_results['best_trial'].user_attrs
                },
                'crisis': {
                    'params': crisis_params,
                    'metrics': crisis_results['best_trial'].user_attrs
                }
            },
            'oos_results': oos_results,
            'config_path': str(output_path)
        }, f, indent=2)

    logger.info(f"\nResults saved: {results_path}")

    logger.info("\n" + "="*80)
    logger.info("REGIME-AWARE OPTIMIZATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
