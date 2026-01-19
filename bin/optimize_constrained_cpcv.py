#!/usr/bin/env python3
"""
Constrained Multi-Objective Optimization with CPCV (FIXED - No Information Leakage)

CRITICAL FIXES:
1. Proper purging based on feature lookback windows (200-bar EMA + 24h label = ~246 hours = 10 days)
2. Hard constraints FIRST (reject trials instead of penalties)
3. Robust objective = percentile_10 of OOS Sortino (not mean)
4. WFE-like diagnostics per trial
5. Debug prints for every fold

Based on research from:
- López de Prado "Advances in Financial Machine Learning" (Chapter 7: Cross-Validation in Finance)
- awesome-systematic-trading GitHub repo
- Walk-forward validation failure analysis

Author: Claude Code - Refactoring Expert
Date: 2026-01-18
"""

import sys
from pathlib import Path
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import optuna
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bin.backtest_full_engine_replay import FullEngineBacktest

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Only optimizer logs at INFO level


# =============================================================================
# ARCHETYPE CONFIGURATIONS
# =============================================================================

# Map archetype IDs to slugs and reduced parameter spaces
ARCHETYPE_CONFIGS = {
    'H': {
        'slug': 'trap_within_trend',
        'name': 'Trap Within Trend',
        'reduced_params': {
            'fusion_threshold': (0.55, 0.70),
            'trend_weight': (0.35, 0.55),
            'liquidity_weight': (0.25, 0.45),
        },
        'fixed_params': {},
    },
    'B': {
        'slug': 'order_block_retest',
        'name': 'Order Block Retest',
        'reduced_params': {
            'fusion_threshold': (0.30, 0.45),
            'order_block_weight': (0.40, 0.60),
            'liquidity_weight': (0.25, 0.45),
        },
        'fixed_params': {},
    },
    'S1': {
        'slug': 'liquidity_vacuum',
        'name': 'Liquidity Vacuum Reversal',
        'reduced_params': {
            'fusion_threshold': (0.30, 0.45),
            'liquidity_weight': (0.28, 0.38),
            'volume_weight': (0.18, 0.28),
        },
        'fixed_params': {
            'wick_weight': 0.18,
        },
    },
}


# =============================================================================
# CPCV (COMBINATORIAL PURGED CROSS-VALIDATION) - FIXED
# =============================================================================

@dataclass
class CPCVFold:
    """A single CPCV fold with train/test split."""
    fold_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    purge_start: pd.Timestamp
    purge_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_cpcv_folds(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size_pct: float = 0.20,
    max_feature_lookback_bars: int = 200,
    max_label_horizon_bars: int = 24,
    purge_multiplier: float = 1.1
) -> List[CPCVFold]:
    """
    Generate CPCV folds with PROPER PURGING to prevent information leakage.

    Key fixes:
    1. Purge window accounts for FEATURE lookback (200-bar EMA, not just labels)
    2. Embargo = 1% of test set (minimum 24 bars)
    3. No overlap between train features and test labels

    Mathematical basis (López de Prado, 2018):
    - Feature at time t uses data from [t - lookback, t]
    - Label at time t+h uses data from [t, t+h]
    - Purge must be >= (lookback + horizon) to prevent leakage
    - Add 10% safety margin for rounding/alignment errors

    Args:
        data: Full dataset with datetime index
        n_splits: Number of CV folds (default: 5)
        test_size_pct: Size of each test set as % of total data (default: 20%)
        max_feature_lookback_bars: Longest feature window (200 for EMA_200)
        max_label_horizon_bars: Forward-looking label horizon (24h for our labels)
        purge_multiplier: Safety margin (default: 1.1 = 10% buffer)

    Returns:
        List of CPCVFold objects with proper purging
    """
    # CRITICAL FIX: Only use data from 2022-01-02 onwards (when features are populated)
    # Features (adx_14, rsi_14, tf4h_*) only exist from 2022-01-02 to 2024-12-31
    feature_start_date = pd.Timestamp("2022-01-02", tz='UTC')

    start_date = max(data.index.min(), feature_start_date)
    end_date = data.index.max()
    total_duration = end_date - start_date

    logger.info(f"Using date range: {start_date.date()} to {end_date.date()} (features populated)")

    # Calculate purge window (feature lookback + label horizon + safety margin)
    purge_bars = int((max_feature_lookback_bars + max_label_horizon_bars) * purge_multiplier)

    # Assume 1H bars (BTC 1H data)
    purge_duration = pd.Timedelta(hours=purge_bars)

    logger.info(f"CPCV Purging Configuration:")
    logger.info(f"  Feature lookback: {max_feature_lookback_bars} bars")
    logger.info(f"  Label horizon: {max_label_horizon_bars} bars")
    logger.info(f"  Purge window: {purge_bars} bars (~{purge_bars/24:.1f} days)")
    logger.info(f"  Purge multiplier: {purge_multiplier}x (safety margin)")

    # Calculate test window size
    test_duration = total_duration * test_size_pct

    # Calculate step size (distance between test windows)
    # Use non-overlapping approach for clean validation
    step_duration = total_duration / (n_splits + 1)

    folds = []
    for i in range(n_splits):
        # Test window placement
        test_start = start_date + (i + 1) * step_duration
        test_end = test_start + test_duration

        # Ensure test doesn't exceed data range
        if test_end > end_date:
            test_end = end_date

        # Calculate embargo (1% of test set, minimum 24 bars)
        test_bars = len(data[(data.index >= test_start) & (data.index < test_end)])
        embargo_bars = max(24, int(test_bars * 0.01))
        embargo_duration = pd.Timedelta(hours=embargo_bars)

        # Purge window = full purge + embargo
        total_purge = purge_duration + embargo_duration

        # Train on all data before test (excluding purge)
        train_start = start_date
        train_end = test_start - total_purge

        # Purge period
        purge_start = train_end
        purge_end = test_start

        # Ensure we have enough training data
        if train_end <= train_start:
            logger.warning(f"Fold {i+1} has insufficient training data, skipping")
            continue

        # Ensure we have enough test data
        if (test_end - test_start) < pd.Timedelta(days=30):
            logger.warning(f"Fold {i+1} has insufficient test data, skipping")
            continue

        folds.append(CPCVFold(
            fold_number=i + 1,
            train_start=train_start,
            train_end=train_end,
            purge_start=purge_start,
            purge_end=purge_end,
            test_start=test_start,
            test_end=test_end
        ))

    logger.info(f"\nGenerated {len(folds)} CPCV folds with proper purging:")
    for fold in folds:
        train_days = (fold.train_end - fold.train_start).days
        purge_days = (fold.purge_end - fold.purge_start).days
        test_days = (fold.test_end - fold.test_start).days
        logger.info(
            f"  Fold {fold.fold_number}: "
            f"Train={train_days}d, Purge={purge_days}d, Test={test_days}d"
        )

    return folds


# =============================================================================
# CONSTRAINED OBJECTIVE FUNCTION - FIXED
# =============================================================================

@dataclass
class FoldMetrics:
    """Metrics for a single fold."""
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    total_trades: int
    sortino_ratio: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    overfitting_ratio: float  # test_sortino / train_sortino


def objective_with_constraints(
    trial: optuna.Trial,
    archetype_id: str,
    data: pd.DataFrame,
    folds: List[CPCVFold],
    min_trades_total: int = 30,  # Changed 2026-01-19: Reduced from 50 to 30 (diagnostic gates showed extreme sparsity with single archetype optimization)
    max_dd_cap: float = 25.0,
    max_concentration: float = 0.5,  # Top-10-trades / total PnL
) -> float:
    """
    Constrained objective with HARD CONSTRAINTS and robust statistics.

    Key changes:
    1. Hard constraints REJECT trials (return -inf, don't penalize)
    2. Objective = percentile_10 of OOS Sortino (robust to outliers)
    3. WFE-like diagnostics (% profitable folds, dispersion, PSR)
    4. Debug prints for EVERY fold
    5. Overfitting ratio per fold

    Args:
        trial: Optuna trial object
        archetype_id: Archetype ID (H, B, S1)
        data: Full dataset
        folds: CPCV fold definitions
        min_trades_total: HARD: Minimum total trades across all folds
        max_dd_cap: HARD: Maximum drawdown threshold
        max_concentration: HARD: Maximum PnL concentration (optional)

    Returns:
        Objective value = percentile_10(OOS Sortino) (higher is better)
        Returns -inf if any hard constraint violated
    """
    config = ARCHETYPE_CONFIGS[archetype_id]
    archetype_slug = config['slug']

    # Suggest parameters (reduced space)
    params = {}
    for param_name, (low, high) in config['reduced_params'].items():
        params[param_name] = trial.suggest_float(param_name, low, high)

    # Add fixed parameters
    params.update(config['fixed_params'])

    logger.info(f"\n{'='*80}")
    logger.info(f"TRIAL {trial.number}: {archetype_id} ({config['name']})")
    logger.info(f"{'='*80}")
    logger.info(f"Parameters: {params}")
    logger.info("")

    # Evaluate across all CPCV folds
    fold_results: List[FoldMetrics] = []

    for fold in folds:
        logger.info(f"\n--- Fold {fold.fold_number}/{len(folds)} ---")

        # Extract train and test data
        train_data = data[
            (data.index >= fold.train_start) &
            (data.index < fold.train_end)
        ].copy()

        test_data = data[
            (data.index >= fold.test_start) &
            (data.index < fold.test_end)
        ].copy()

        # Skip if insufficient data
        if len(train_data) < 100 or len(test_data) < 30:
            logger.warning(f"  ⚠ Insufficient data (train={len(train_data)}, test={len(test_data)}), skipping")
            continue

        # Log fold date ranges
        logger.info(f"  Train: {fold.train_start.date()} to {fold.train_end.date()} ({len(train_data)} bars)")
        logger.info(f"  Purge: {fold.purge_start.date()} to {fold.purge_end.date()}")
        logger.info(f"  Test:  {fold.test_start.date()} to {fold.test_end.date()} ({len(test_data)} bars)")

        # Create engine config
        engine_config = {
            'symbol': 'BTC',
            'initial_capital': 10000.0,
            'max_positions': 5,
            'position_size_pct': 0.12,
            'fee_pct': 0.0006,
            'slippage_pct': 0.0008,
            'enable_circuit_breakers': True,
            'enable_direction_balance': True,
            'enable_regime_penalties': True,
            'enable_adaptive_regime': True,
            'cooldown_bars': 12,
            # Changed 2026-01-19: Enable all wired systems for production-ready optimization
            'use_regime_detection': True,     # Enable HybridRegimeModel (v1/v2 ensemble)
            'use_soft_gating': True,           # Enable RegimeWeightAllocator (edge table + sqrt soft gating)
            'use_circuit_breaker': True,       # Enable 4-tier safety system (margin/drawdown/consecutive losses/regime veto)
            # Disable all archetypes except the one being optimized
            'enable_A': False,
            'enable_B': archetype_id == 'B',
            'enable_C': False,
            'enable_G': False,
            'enable_H': archetype_id == 'H',
            'enable_K': False,
            'enable_S1': archetype_id == 'S1',
            'enable_S4': False,
            'enable_S5': False,
        }

        archetype_config_overrides = {
            archetype_slug: params
        }

        try:
            # Run backtest on TRAIN set (for overfitting ratio)
            engine_train = FullEngineBacktest(
                config=engine_config,
                archetype_config_overrides=archetype_config_overrides
            )
            train_metrics = engine_train.run(
                data=train_data,
                start_date=str(fold.train_start.date()),
                end_date=str(fold.train_end.date())
            )

            # Run backtest on TEST set (OOS)
            engine_test = FullEngineBacktest(
                config=engine_config,
                archetype_config_overrides=archetype_config_overrides
            )
            test_metrics = engine_test.run(
                data=test_data,
                start_date=str(fold.test_start.date()),
                end_date=str(fold.test_end.date())
            )

            # Extract metrics
            train_sortino = train_metrics.get('sortino_ratio', 0.0)
            train_trades = train_metrics.get('total_trades', 0)
            train_return = train_metrics.get('total_return_pct', 0.0)

            test_sortino = test_metrics.get('sortino_ratio', 0.0)
            test_sharpe = test_metrics.get('sharpe_ratio', 0.0)
            test_trades = test_metrics.get('total_trades', 0)
            test_return = test_metrics.get('total_return_pct', 0.0)
            test_max_dd = test_metrics.get('max_drawdown_pct', 0.0)
            test_win_rate = test_metrics.get('win_rate', 0.0)

            # Calculate overfitting ratio
            overfitting_ratio = test_sortino / train_sortino if train_sortino > 0.01 else 0.0

            # Debug prints
            logger.info(f"  Train: Trades={train_trades:3d}, Return={train_return:6.2f}%, Sortino={train_sortino:5.2f}")
            logger.info(f"  Test:  Trades={test_trades:3d}, Return={test_return:6.2f}%, Sortino={test_sortino:5.2f}")
            logger.info(f"  MaxDD: {test_max_dd:.2f}%, WinRate: {test_win_rate:.1f}%")
            logger.info(f"  Overfitting Ratio: {overfitting_ratio:.3f} (test/train Sortino)")

            fold_results.append(FoldMetrics(
                fold_number=fold.fold_number,
                train_start=str(fold.train_start.date()),
                train_end=str(fold.train_end.date()),
                test_start=str(fold.test_start.date()),
                test_end=str(fold.test_end.date()),
                total_trades=test_trades,
                sortino_ratio=test_sortino,
                sharpe_ratio=test_sharpe,
                total_return=test_return,
                max_drawdown=test_max_dd,
                win_rate=test_win_rate,
                overfitting_ratio=overfitting_ratio
            ))

        except Exception as e:
            logger.error(f"  ❌ Fold {fold.fold_number} failed: {e}")
            continue

    # Check if we have enough folds (require at least 2, or all but 1 if using >2 folds)
    min_successful_folds = max(2, len(folds) - 1)
    if len(fold_results) < min_successful_folds:
        logger.warning(f"  ❌ Only {len(fold_results)} folds succeeded (need >={min_successful_folds}), returning -inf")
        return float('-inf')

    # =============================================================================
    # HARD CONSTRAINTS (reject trials, don't penalize)
    # =============================================================================

    total_trades = sum(f.total_trades for f in fold_results)
    max_dd = max(f.max_drawdown for f in fold_results)
    sortino_values = [f.sortino_ratio for f in fold_results]

    # HARD CONSTRAINT 1: Minimum total trades
    if total_trades < min_trades_total:
        logger.warning(f"  ❌ HARD CONSTRAINT VIOLATED: Total trades {total_trades} < {min_trades_total}")
        return float('-inf')

    # HARD CONSTRAINT 2: Maximum drawdown cap
    if max_dd > max_dd_cap:
        logger.warning(f"  ❌ HARD CONSTRAINT VIOLATED: Max drawdown {max_dd:.2f}% > {max_dd_cap}%")
        return float('-inf')

    # HARD CONSTRAINT 3: PnL concentration (optional)
    # TODO: Implement top-10 trade concentration if needed
    # For now, skip this check

    # =============================================================================
    # ROBUST OBJECTIVE = PERCENTILE_10 of OOS Sortino
    # =============================================================================

    # Use 10th percentile instead of mean (robust to outliers)
    objective_value = np.percentile(sortino_values, 10)

    # =============================================================================
    # WFE-LIKE DIAGNOSTICS
    # =============================================================================

    profitable_folds = sum(1 for f in fold_results if f.total_return > 0)
    profitable_pct = profitable_folds / len(fold_results) * 100
    median_return = np.median([f.total_return for f in fold_results])
    sortino_dispersion = np.std(sortino_values)

    # Probabilistic Sharpe Ratio (PSR) - simplified
    # PSR = probability that Sharpe > 0
    # Approximation: use z-score of mean Sortino
    mean_sortino = np.mean(sortino_values)
    std_sortino = np.std(sortino_values) + 1e-8
    z_score = mean_sortino / (std_sortino / np.sqrt(len(sortino_values)))
    psr = 0.5 * (1 + np.tanh(z_score))  # Approximate CDF

    logger.info(f"\n{'='*80}")
    logger.info(f"TRIAL {trial.number} SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Hard Constraints:")
    logger.info(f"  ✓ Total trades: {total_trades} (>= {min_trades_total})")
    logger.info(f"  ✓ Max drawdown: {max_dd:.2f}% (<= {max_dd_cap}%)")
    logger.info(f"")
    logger.info(f"Robust Objective:")
    logger.info(f"  Percentile_10 Sortino: {objective_value:.3f}")
    logger.info(f"")
    logger.info(f"WFE-Like Diagnostics:")
    logger.info(f"  Profitable folds: {profitable_folds}/{len(fold_results)} ({profitable_pct:.1f}%)")
    logger.info(f"  Median fold return: {median_return:.2f}%")
    logger.info(f"  Sortino dispersion (std): {sortino_dispersion:.3f}")
    logger.info(f"  Probabilistic Sharpe Ratio (PSR): {psr:.3f}")
    logger.info(f"")
    logger.info(f"Fold-by-Fold Sortino: {[f'{s:.2f}' for s in sortino_values]}")
    logger.info(f"{'='*80}\n")

    # Store diagnostics in trial
    trial.set_user_attr('total_trades', total_trades)
    trial.set_user_attr('max_dd', max_dd)
    trial.set_user_attr('profitable_pct', profitable_pct)
    trial.set_user_attr('median_return', median_return)
    trial.set_user_attr('sortino_dispersion', sortino_dispersion)
    trial.set_user_attr('psr', psr)
    trial.set_user_attr('fold_results', [asdict(f) for f in fold_results])

    return objective_value


# =============================================================================
# MAIN OPTIMIZATION LOOP
# =============================================================================

def optimize_archetype_constrained(
    archetype_id: str,
    data: pd.DataFrame,
    n_trials: int = 50,
    n_folds: int = 5,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Run constrained optimization with CPCV for a single archetype.

    Args:
        archetype_id: Archetype ID (H, B, S1)
        data: Full dataset for optimization
        n_trials: Number of TPE trials (default: 50)
        n_folds: Number of CPCV folds (default: 5)
        output_dir: Directory to save results

    Returns:
        Optimization results dict
    """
    config = ARCHETYPE_CONFIGS[archetype_id]
    logger.info(f"\n{'='*80}")
    logger.info(f"CONSTRAINED OPTIMIZATION WITH CPCV: {archetype_id} ({config['name']})")
    logger.info(f"{'='*80}")
    logger.info(f"Parameters: {list(config['reduced_params'].keys())}")
    logger.info(f"Fixed params: {config['fixed_params']}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"CPCV folds: {n_folds}")

    # Generate CPCV folds with proper purging
    folds = generate_cpcv_folds(
        data=data,
        n_splits=n_folds,
        test_size_pct=0.30,  # Changed 2026-01-19: Extended from 0.20 to 0.30 (+50% test exposure for more trading opportunities)
        max_feature_lookback_bars=200,  # EMA_200
        max_label_horizon_bars=24,      # 24h labels
        purge_multiplier=1.1            # 10% safety margin
    )

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(
            seed=42,
            n_startup_trials=20,      # More random exploration
            multivariate=True,        # Account for parameter interactions
            constant_liar=True        # Better for parallel trials
        )
    )

    # Run optimization
    study.optimize(
        lambda trial: objective_with_constraints(
            trial=trial,
            archetype_id=archetype_id,
            data=data,
            folds=folds,
            min_trades_total=20,  # Production: 20 for 5 folds (was 5 for 2-fold smoke test)
            max_dd_cap=25.0
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Extract best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZATION COMPLETE: {archetype_id}")
    logger.info(f"{'='*80}")
    logger.info(f"Best objective (10th percentile Sortino): {best_value:.3f}")
    logger.info(f"Best parameters: {best_params}")

    # Extract best trial diagnostics
    best_trial = study.best_trial
    logger.info(f"\nBest Trial Diagnostics:")

    total_trades = best_trial.user_attrs.get('total_trades')
    max_dd = best_trial.user_attrs.get('max_dd')
    profitable_pct = best_trial.user_attrs.get('profitable_pct')
    median_return = best_trial.user_attrs.get('median_return')
    sortino_dispersion = best_trial.user_attrs.get('sortino_dispersion')
    psr = best_trial.user_attrs.get('psr')

    logger.info(f"  Total trades: {total_trades if total_trades is not None else 'N/A'}")
    logger.info(f"  Max drawdown: {max_dd:.2f}%" if max_dd is not None else "  Max drawdown: N/A")
    logger.info(f"  Profitable folds: {profitable_pct:.1f}%" if profitable_pct is not None else "  Profitable folds: N/A")
    logger.info(f"  Median return: {median_return:.2f}%" if median_return is not None else "  Median return: N/A")
    logger.info(f"  Sortino dispersion: {sortino_dispersion:.3f}" if sortino_dispersion is not None else "  Sortino dispersion: N/A")
    logger.info(f"  PSR: {psr:.3f}" if psr is not None else "  PSR: N/A")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            'archetype': archetype_id,
            'archetype_slug': config['slug'],
            'archetype_name': config['name'],
            'optimization_date': datetime.now().isoformat(),
            'method': 'constrained_cpcv_fixed',
            'n_trials': n_trials,
            'n_folds': n_folds,
            'best_params': best_params,
            'best_objective': best_value,
            'best_trial_diagnostics': {
                'total_trades': best_trial.user_attrs.get('total_trades'),
                'max_dd': best_trial.user_attrs.get('max_dd'),
                'profitable_pct': best_trial.user_attrs.get('profitable_pct'),
                'median_return': best_trial.user_attrs.get('median_return'),
                'sortino_dispersion': best_trial.user_attrs.get('sortino_dispersion'),
                'psr': best_trial.user_attrs.get('psr'),
                'fold_results': best_trial.user_attrs.get('fold_results'),
            },
            'reduced_param_space': config['reduced_params'],
            'fixed_params': config['fixed_params'],
            'purging_config': {
                'max_feature_lookback_bars': 200,
                'max_label_horizon_bars': 24,
                'purge_multiplier': 1.1,
                'total_purge_bars': int((200 + 24) * 1.1),
            }
        }

        output_path = output_dir / f"{archetype_id}_constrained_cpcv_fixed.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"\n✅ Results saved to {output_path}")

    return {
        'archetype_id': archetype_id,
        'best_params': best_params,
        'best_value': best_value,
        'study': study
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Constrained multi-objective optimization with CPCV (FIXED - No Leakage)"
    )
    parser.add_argument(
        '--archetype',
        type=str,
        required=True,
        choices=['H', 'B', 'S1'],
        help="Archetype to optimize (H, B, or S1)"
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help="Number of TPE trials (default: 50)"
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help="Number of CPCV folds (default: 5)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/optimization_constrained_cpcv_fixed',
        help="Output directory for results"
    )

    args = parser.parse_args()

    # === PREFLIGHT CHECKS (Fail Fast) ===
    logger.info("Running preflight checks...")

    # Check 1: Archetype registry
    registry_path = project_root / 'archetype_registry.yaml'
    if not registry_path.exists():
        raise FileNotFoundError(
            f"❌ PREFLIGHT FAILED: Archetype registry not found at {registry_path}\n"
            "This file is required for ArchetypeFactory to load archetype definitions."
        )
    logger.info(f"✓ Archetype registry found: {registry_path}")

    # Check 2: Data file
    data_path = project_root / 'data' / 'features_2018_2024_MERGED.parquet'
    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ PREFLIGHT FAILED: Dataset not found at {data_path}\n"
            "Run feature backfilling scripts to generate this file."
        )
    logger.info(f"✓ Data file found: {data_path}")

    # Check 3: Output directory (create if missing)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory ready: {output_dir}")

    logger.info("✅ All preflight checks passed\n")
    # === END PREFLIGHT CHECKS ===

    # Load data (MERGED has backfilled basic features 2018-2024 + tf4h features 2022-2024)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(data):,} bars ({data.index.min().date()} to {data.index.max().date()})")

    # CRITICAL FIX: Filter to only use populated feature range (2022-01-02 onwards)
    feature_start_date = pd.Timestamp("2022-01-02", tz='UTC')
    data = data[data.index >= feature_start_date]
    logger.info(f"Filtered to feature-populated range: {len(data):,} bars ({data.index.min().date()} to {data.index.max().date()})")

    # Run optimization
    output_dir = Path(args.output_dir)
    result = optimize_archetype_constrained(
        archetype_id=args.archetype,
        data=data,
        n_trials=args.trials,
        n_folds=args.folds,
        output_dir=output_dir
    )

    logger.info(f"\n✅ Optimization complete for {args.archetype}")
    logger.info(f"Best parameters: {result['best_params']}")
    logger.info(f"Best objective (10th percentile Sortino): {result['best_value']:.3f}")


if __name__ == '__main__':
    main()
