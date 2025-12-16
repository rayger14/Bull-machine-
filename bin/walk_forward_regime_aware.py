#!/usr/bin/env python3
"""
Walk-Forward Validation with Regime Stratification

Implements institutional-grade walk-forward optimization with regime awareness.
This is the GOLD STANDARD for validating regime-aware threshold optimization.

Key Principles:
1. Train and test on regime-filtered bars ONLY
2. Validate that optimized thresholds generalize to OOS data
3. Compute train/test correlation (OOS consistency metric)
4. Prevent overfitting by requiring stable performance across windows

Architecture:
- Rolling windows: 180-day train, 60-day test
- Regime stratification: Only bars in allowed regimes
- Multi-objective optimization per window
- OOS consistency tracking (correlation between train/test PF)

Expected Results:
- OOS consistency >0.6: Parameters generalize well
- OOS consistency <0.4: Overfitting detected
- Stable PF across windows: Robust strategy

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
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from bin.backtest_regime_stratified import (
    backtest_regime_stratified,
    validate_regime_coverage
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward validation window"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_regime_bars: int
    test_regime_bars: int
    train_pf: float
    test_pf: float
    train_wr: float
    test_wr: float
    train_trades: int
    test_trades: int
    best_params: Dict

    def to_dict(self) -> Dict:
        return {
            'window_id': self.window_id,
            'train_period': f"{self.train_start} to {self.train_end}",
            'test_period': f"{self.test_start} to {self.test_end}",
            'train_regime_bars': self.train_regime_bars,
            'test_regime_bars': self.test_regime_bars,
            'train_pf': round(self.train_pf, 3),
            'test_pf': round(self.test_pf, 3),
            'train_wr': round(self.train_wr, 2),
            'test_wr': round(self.test_wr, 2),
            'train_trades': self.train_trades,
            'test_trades': self.test_trades,
            'best_params': self.best_params
        }


@dataclass
class WalkForwardResults:
    """Complete walk-forward validation results"""
    archetype: str
    allowed_regimes: List[str]
    windows: List[WalkForwardWindow]
    oos_consistency: float  # Correlation between train/test PF
    avg_test_pf: float
    avg_test_wr: float
    stable_performance: bool  # True if std(test_pf) < 0.5

    def to_dict(self) -> Dict:
        return {
            'archetype': self.archetype,
            'allowed_regimes': self.allowed_regimes,
            'num_windows': len(self.windows),
            'oos_consistency': round(self.oos_consistency, 3),
            'avg_test_pf': round(self.avg_test_pf, 3),
            'avg_test_wr': round(self.avg_test_wr, 2),
            'stable_performance': self.stable_performance,
            'windows': [w.to_dict() for w in self.windows]
        }


def walk_forward_regime_aware(
    archetype: str,
    data: pd.DataFrame,
    allowed_regimes: List[str],
    search_space: Dict,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
    n_trials: int = 100,
    min_bars_per_window: int = 500
) -> WalkForwardResults:
    """
    Walk-forward optimization with regime stratification.

    Args:
        archetype: Archetype name
        data: Full historical data with regime_label column
        allowed_regimes: List of allowed regimes for this archetype
        search_space: Parameter search space dict
        train_days: Training window size in days
        test_days: Test window size in days
        step_days: Step size between windows in days
        n_trials: Optuna trials per window
        min_bars_per_window: Minimum regime bars required per window

    Returns:
        WalkForwardResults with OOS consistency metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Walk-Forward Validation: {archetype.upper()}")
    logger.info(f"Regimes: {allowed_regimes}")
    logger.info(f"Train: {train_days}d, Test: {test_days}d, Step: {step_days}d")
    logger.info(f"{'='*80}\n")

    # Generate windows
    start_date = data.index[0]
    end_date = data.index[-1]

    windows = []
    window_id = 0

    current_start = start_date
    while True:
        # Define window boundaries
        train_start = current_start
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)

        # Check if we've run out of data
        if test_end > end_date:
            break

        window_id += 1
        windows.append({
            'id': window_id,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })

        # Step forward
        current_start += timedelta(days=step_days)

    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Run optimization for each window
    results = []

    for window in windows:
        logger.info(f"\n{'='*60}")
        logger.info(f"Window {window['id']}/{len(windows)}")
        logger.info(f"Train: {window['train_start'].date()} to {window['train_end'].date()}")
        logger.info(f"Test: {window['test_start'].date()} to {window['test_end'].date()}")
        logger.info(f"{'='*60}")

        # Extract window data
        train_data = data[
            (data.index >= window['train_start']) & (data.index < window['train_end'])
        ].copy()

        test_data = data[
            (data.index >= window['test_start']) & (data.index < window['test_end'])
        ].copy()

        # Filter to regime bars
        train_regime = train_data[train_data['regime_label'].isin(allowed_regimes)]
        test_regime = test_data[test_data['regime_label'].isin(allowed_regimes)]

        logger.info(f"Train regime bars: {len(train_regime):,}")
        logger.info(f"Test regime bars: {len(test_regime):,}")

        # Skip window if insufficient regime bars
        if len(train_regime) < min_bars_per_window:
            logger.warning(f"Skipping window: insufficient train bars ({len(train_regime)} < {min_bars_per_window})")
            continue

        if len(test_regime) < 100:
            logger.warning(f"Skipping window: insufficient test bars ({len(test_regime)} < 100)")
            continue

        # Optimize on train data
        logger.info("Optimizing on train data...")
        best_params = optimize_window(
            archetype, train_regime, allowed_regimes, search_space, n_trials
        )

        if not best_params:
            logger.warning("Optimization failed for window")
            continue

        # Backtest on train data (in-sample)
        train_result = backtest_regime_stratified(
            archetype=archetype,
            data=train_regime,
            config=best_params['params'],
            allowed_regimes=allowed_regimes
        )

        # Backtest on test data (out-of-sample)
        test_result = backtest_regime_stratified(
            archetype=archetype,
            data=test_regime,
            config=best_params['params'],
            allowed_regimes=allowed_regimes
        )

        logger.info(f"Train: PF={train_result.profit_factor:.3f}, WR={train_result.win_rate:.1f}%, Trades={train_result.total_trades}")
        logger.info(f"Test: PF={test_result.profit_factor:.3f}, WR={test_result.win_rate:.1f}%, Trades={test_result.total_trades}")

        # Store window result
        window_result = WalkForwardWindow(
            window_id=window['id'],
            train_start=str(window['train_start'].date()),
            train_end=str(window['train_end'].date()),
            test_start=str(window['test_start'].date()),
            test_end=str(window['test_end'].date()),
            train_regime_bars=len(train_regime),
            test_regime_bars=len(test_regime),
            train_pf=train_result.profit_factor,
            test_pf=test_result.profit_factor,
            train_wr=train_result.win_rate,
            test_wr=test_result.win_rate,
            train_trades=train_result.total_trades,
            test_trades=test_result.total_trades,
            best_params=best_params['params']
        )

        results.append(window_result)

    # Compute OOS consistency
    if len(results) < 2:
        logger.error("Insufficient windows for OOS consistency calculation")
        raise ValueError("Need at least 2 windows for walk-forward validation")

    train_pfs = [w.train_pf for w in results]
    test_pfs = [w.test_pf for w in results]

    oos_consistency = np.corrcoef(train_pfs, test_pfs)[0, 1]

    # Compute average metrics
    avg_test_pf = np.mean(test_pfs)
    avg_test_wr = np.mean([w.test_wr for w in results])

    # Check stability
    test_pf_std = np.std(test_pfs)
    stable_performance = test_pf_std < 0.5

    logger.info(f"\n{'='*80}")
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Windows completed: {len(results)}")
    logger.info(f"OOS Consistency (train/test correlation): {oos_consistency:.3f}")
    logger.info(f"Average Test PF: {avg_test_pf:.3f}")
    logger.info(f"Average Test WR: {avg_test_wr:.1f}%")
    logger.info(f"Test PF Std Dev: {test_pf_std:.3f}")
    logger.info(f"Stable Performance: {'YES' if stable_performance else 'NO'}")

    if oos_consistency >= 0.6:
        logger.info("STATUS: EXCELLENT - Parameters generalize well")
    elif oos_consistency >= 0.4:
        logger.info("STATUS: ACCEPTABLE - Some generalization")
    else:
        logger.warning("STATUS: OVERFITTING DETECTED - Parameters do not generalize")

    return WalkForwardResults(
        archetype=archetype,
        allowed_regimes=allowed_regimes,
        windows=results,
        oos_consistency=oos_consistency,
        avg_test_pf=avg_test_pf,
        avg_test_wr=avg_test_wr,
        stable_performance=stable_performance
    )


def optimize_window(
    archetype: str,
    train_data: pd.DataFrame,
    allowed_regimes: List[str],
    search_space: Dict,
    n_trials: int
) -> Optional[Dict]:
    """
    Optimize parameters for a single window.

    Args:
        archetype: Archetype name
        train_data: Training data (already regime-filtered)
        allowed_regimes: Allowed regimes
        search_space: Parameter search space
        n_trials: Number of Optuna trials

    Returns:
        Dict with best parameters and metrics
    """

    def objective(trial):
        """Multi-objective: (PF, WR, trades/year)"""

        # Suggest parameters
        params = {}
        for param_name, (low, high) in search_space.items():
            if param_name.endswith('_bars'):
                params[param_name] = trial.suggest_int(param_name, int(low), int(high))
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)

        # Backtest
        try:
            result = backtest_regime_stratified(
                archetype=archetype,
                data=train_data,
                config=params,
                allowed_regimes=allowed_regimes
            )

            # Multi-objective
            pf_obj = -result.profit_factor
            wr_obj = -result.win_rate
            trades_obj = abs(result.trades_per_year - 12.0)  # Target ~12 trades/year

            trial.set_user_attr('profit_factor', result.profit_factor)
            trial.set_user_attr('win_rate', result.win_rate)
            trial.set_user_attr('trades_per_year', result.trades_per_year)

            return pf_obj, wr_obj, trades_obj

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0, 0.0, 999.0

    # Create study (in-memory, no persistence)
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=NSGAIISampler(population_size=10)
    )

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Get best trial
    pareto_trials = study.best_trials
    if not pareto_trials:
        return None

    # Select best by PF
    best_trial = max(pareto_trials, key=lambda t: t.user_attrs['profit_factor'])

    return {
        'params': best_trial.params,
        'metrics': best_trial.user_attrs
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Walk-forward validation with regime awareness')
    parser.add_argument('--archetype', required=True, help='Archetype name')
    parser.add_argument('--regimes', nargs='+', required=True, help='Allowed regimes')
    parser.add_argument('--start-date', default='2022-01-01', help='Start date')
    parser.add_argument('--end-date', default='2023-12-31', help='End date')
    parser.add_argument('--train-days', type=int, default=180, help='Training window size')
    parser.add_argument('--test-days', type=int, default=60, help='Test window size')
    parser.add_argument('--step-days', type=int, default=60, help='Step size')
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials per window')

    args = parser.parse_args()

    # Define search space (archetype-specific)
    SEARCH_SPACES = {
        'liquidity_vacuum': {
            'fusion_threshold': (0.40, 0.55),
            'liquidity_max': (0.10, 0.20),
            'volume_z_min': (1.5, 2.5),
            'wick_lower_min': (0.25, 0.40),
            'cooldown_bars': (8, 18),
            'atr_stop_mult': (2.0, 3.5)
        },
        'funding_divergence': {
            'fusion_threshold': (0.75, 0.90),
            'funding_z_max': (-2.2, -1.5),
            'resilience_min': (0.55, 0.70),
            'liquidity_max': (0.20, 0.35),
            'cooldown_bars': (8, 18),
            'atr_stop_mult': (2.0, 3.5)
        },
        'long_squeeze': {
            'fusion_threshold': (0.75, 0.90),
            'funding_z_min': (1.8, 2.5),
            'rsi_min': (60, 75),
            'liquidity_max': (0.20, 0.35),
            'cooldown_bars': (8, 18),
            'atr_stop_mult': (2.0, 3.5)
        }
    }

    search_space = SEARCH_SPACES.get(args.archetype)
    if not search_space:
        logger.error(f"Unknown archetype: {args.archetype}. Available: {list(SEARCH_SPACES.keys())}")
        sys.exit(1)

    # Load data
    feature_file = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    if not feature_file.exists():
        logger.error(f"Feature file not found: {feature_file}")
        sys.exit(1)

    logger.info(f"Loading data: {args.start_date} to {args.end_date}")
    df = pd.read_parquet(feature_file)
    data = df[(df.index >= args.start_date) & (df.index < args.end_date)].copy()

    if 'regime_label' not in data.columns:
        logger.error("Data missing regime_label. Run: bin/quick_add_regime_labels.py")
        sys.exit(1)

    # Run walk-forward validation
    results = walk_forward_regime_aware(
        archetype=args.archetype,
        data=data,
        allowed_regimes=args.regimes,
        search_space=search_space,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        n_trials=args.n_trials
    )

    # Save results
    output_path = Path(f'results/walk_forward_{args.archetype}_regime_aware.json')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
