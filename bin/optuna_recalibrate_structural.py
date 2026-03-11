#!/usr/bin/env python3
"""
Optuna Re-Calibration for Structural Check Pipeline

After wiring identity-only structural gates (logic.py) into the detection pipeline,
signal distribution changed significantly. This script re-optimizes:

  1. Per-archetype base_threshold (11 archetypes)
  2. Global temp_range, instab_range, crisis_coefficient

to close the PF gap (1.55 → target ~1.78) and reduce MaxDD (13.6% → target ~7%).

Uses the production StandaloneBacktestEngine directly — same engine, same feature store,
same exit logic. Only threshold parameters change between trials.

Usage:
    python3 bin/optuna_recalibrate_structural.py --trials 100
    python3 bin/optuna_recalibrate_structural.py --trials 50 --train-start 2020-01-01 --train-end 2022-12-31

Author: Claude Code
Date: 2026-03-09
"""

import sys
import json
import argparse
import logging
import time
import copy
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

from bin.backtest_v11_standalone import StandaloneBacktestEngine

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

DEFAULT_CONFIG = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"

# Archetypes that were profitable in structural check backtest (identity gates v3)
CALIBRATED_ARCHETYPES = [
    'wick_trap',           # $48.6K PnL, PF > 1.0
    'retest_cluster',      # $38.5K PnL
    'liquidity_sweep',     # $35.0K PnL
    'trap_within_trend',   # $17.3K PnL
    'spring',              # $3.1K PnL
    'failed_continuation', # $3.0K PnL
    'funding_divergence',  # $1.9K PnL
    'long_squeeze',        # $1.5K PnL
    'fvg_continuation',    # $0.2K PnL (marginal)
    'exhaustion_reversal', # $0.2K PnL (marginal)
    'order_block_retest',  # breakeven with structural checks
]

# Search ranges for per-archetype base_threshold
# Lower floor than before (0.03 vs 0.06) because structural checks pre-filter
THRESHOLD_RANGES = {
    'wick_trap':           (0.03, 0.18),  # Was 0.10, high-volume archetype
    'retest_cluster':      (0.03, 0.15),  # Was 0.06, very permissive currently
    'liquidity_sweep':     (0.03, 0.18),  # Was 0.08
    'trap_within_trend':   (0.03, 0.15),  # Was 0.08
    'spring':              (0.03, 0.18),  # Was 0.08
    'failed_continuation': (0.03, 0.18),  # Was 0.08
    'funding_divergence':  (0.03, 0.20),  # Was 0.18 (default), needs exploration
    'long_squeeze':        (0.03, 0.20),  # Was 0.18 (default), short archetype
    'fvg_continuation':    (0.03, 0.20),  # Was 0.18 (default)
    'exhaustion_reversal': (0.03, 0.20),  # Was 0.18 (default)
    'order_block_retest':  (0.03, 0.18),  # Was 0.10
    'liquidity_vacuum':    (0.03, 0.15),  # Was 0.08
}

# Global parameters
GLOBAL_RANGES = {
    'temp_range':        (0.25, 0.55),  # Was 0.38, controls bear penalty
    'instab_range':      (0.08, 0.25),  # Was 0.15, controls chop penalty
    'crisis_coefficient': (0.30, 0.70),  # Was 0.50, controls crisis penalty
}


def load_data(feature_store_path: str) -> pd.DataFrame:
    """Load feature store once (expensive)."""
    logger.info(f"Loading feature store: {feature_store_path}")
    df = pd.read_parquet(feature_store_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logger.info(f"Loaded {len(df):,} bars x {len(df.columns)} cols, {df.index.min()} to {df.index.max()}")
    return df


def run_backtest(
    config: Dict[str, Any],
    features_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0002,
    slippage_bps: float = 3.0,
) -> Dict[str, Any]:
    """Run a single backtest with given config. Returns performance stats."""
    engine = StandaloneBacktestEngine(
        config=config,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_bps=slippage_bps,
        features_df=features_df,
    )
    engine.run(start_date=start_date, end_date=end_date)
    return engine.get_performance_stats()


class StructuralRecalibrationObjective:
    """Optuna objective: maximize profit_factor on train period."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        features_df: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        min_trades: int = 50,
        max_dd_pct: float = 20.0,
        initial_cash: float = 100_000.0,
        commission_rate: float = 0.0002,
        slippage_bps: float = 3.0,
    ):
        self.base_config = base_config
        self.features_df = features_df
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.min_trades = min_trades
        self.max_dd_pct = max_dd_pct
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps

        self.trial_count = 0
        self.best_pf = 0.0
        self.best_trial = -1

    def __call__(self, trial: optuna.Trial) -> float:
        self.trial_count += 1
        t0 = time.time()

        # Build config with trial parameters
        config = copy.deepcopy(self.base_config)

        # Sample per-archetype base thresholds
        per_arch_thresholds = {}
        for arch_name, (lo, hi) in THRESHOLD_RANGES.items():
            val = trial.suggest_float(f'bt_{arch_name}', lo, hi, step=0.01)
            per_arch_thresholds[arch_name] = val

        # Sample global adaptive fusion parameters
        temp_range = trial.suggest_float('temp_range', *GLOBAL_RANGES['temp_range'], step=0.01)
        instab_range = trial.suggest_float('instab_range', *GLOBAL_RANGES['instab_range'], step=0.01)
        crisis_coeff = trial.suggest_float('crisis_coefficient', *GLOBAL_RANGES['crisis_coefficient'], step=0.01)

        # Apply to config
        af = config.setdefault('adaptive_fusion', {})
        af['enabled'] = True
        af['per_archetype_base_threshold'] = per_arch_thresholds
        af['temp_range'] = temp_range
        af['instab_range'] = instab_range
        af['crisis_coefficient'] = crisis_coeff
        # Ensure bypass is OFF for optimization (otherwise all thresholds are equivalent)
        af['bypass_threshold'] = False

        # Run backtest on train period
        try:
            stats = run_backtest(
                config=config,
                features_df=self.features_df,
                start_date=self.train_start,
                end_date=self.train_end,
                initial_cash=self.initial_cash,
                commission_rate=self.commission_rate,
                slippage_bps=self.slippage_bps,
            )
        except Exception as e:
            logger.warning(f"Trial {trial.number} crashed: {e}")
            return -1e9

        elapsed = time.time() - t0

        # Extract metrics
        pf = stats.get('profit_factor', 0.0)
        trades = stats.get('total_trades', 0)
        max_dd = abs(stats.get('max_drawdown', 0.0))
        sharpe = stats.get('sharpe_ratio', 0.0)
        total_pnl = stats.get('total_pnl', 0.0)
        wr = stats.get('win_rate', 0.0)

        # Penalty: too few trades (unreliable PF)
        if trades < self.min_trades:
            logger.info(
                f"Trial {trial.number:3d} | PENALTY: {trades} trades < {self.min_trades} min | "
                f"PF={pf:.2f} | {elapsed:.0f}s"
            )
            return -1e9

        # Penalty: extreme drawdown
        if max_dd > self.max_dd_pct:
            logger.info(
                f"Trial {trial.number:3d} | PENALTY: MaxDD={max_dd:.1f}% > {self.max_dd_pct}% | "
                f"PF={pf:.2f} | trades={trades} | {elapsed:.0f}s"
            )
            return -1e9

        # Penalty: PF must be profitable
        if pf <= 1.0:
            score = pf - 1.0  # Negative score for losing strategies
        else:
            # Composite objective: PF with DD penalty
            # PF is the primary objective, DD is a soft penalty
            dd_penalty = max(0, (max_dd - 8.0)) * 0.02  # Penalty starts above 8% DD
            score = pf - dd_penalty

        if score > self.best_pf:
            self.best_pf = score
            self.best_trial = trial.number

        logger.info(
            f"Trial {trial.number:3d} | PF={pf:.3f} | trades={trades:4d} | WR={wr:.1f}% | "
            f"MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f} | PnL=${total_pnl:,.0f} | "
            f"score={score:.3f} {'*** BEST' if trial.number == self.best_trial else ''} | {elapsed:.0f}s"
        )

        # Store metrics for later analysis
        trial.set_user_attr('profit_factor', pf)
        trial.set_user_attr('total_trades', trades)
        trial.set_user_attr('max_drawdown', max_dd)
        trial.set_user_attr('sharpe_ratio', sharpe)
        trial.set_user_attr('total_pnl', total_pnl)
        trial.set_user_attr('win_rate', wr)

        return score


def validate_best(
    study: optuna.Study,
    base_config: Dict[str, Any],
    features_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0002,
    slippage_bps: float = 3.0,
) -> Dict[str, Any]:
    """Run best trial parameters on OOS test period."""
    best = study.best_trial

    # Build config from best params
    config = copy.deepcopy(base_config)
    af = config.setdefault('adaptive_fusion', {})
    af['enabled'] = True
    af['bypass_threshold'] = False

    per_arch_thresholds = {}
    for arch_name in THRESHOLD_RANGES:
        key = f'bt_{arch_name}'
        if key in best.params:
            per_arch_thresholds[arch_name] = best.params[key]
    af['per_archetype_base_threshold'] = per_arch_thresholds

    if 'temp_range' in best.params:
        af['temp_range'] = best.params['temp_range']
    if 'instab_range' in best.params:
        af['instab_range'] = best.params['instab_range']
    if 'crisis_coefficient' in best.params:
        af['crisis_coefficient'] = best.params['crisis_coefficient']

    # Run OOS backtest
    stats = run_backtest(
        config=config,
        features_df=features_df,
        start_date=test_start,
        end_date=test_end,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_bps=slippage_bps,
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description='Optuna re-calibration for structural check pipeline')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Base config JSON path')
    parser.add_argument('--feature-store', default=DEFAULT_FEATURE_STORE, help='Feature store parquet path')
    parser.add_argument('--train-start', default='2020-01-01', help='Train period start')
    parser.add_argument('--train-end', default='2022-12-31', help='Train period end')
    parser.add_argument('--test-start', default='2023-01-01', help='Test (OOS) period start')
    parser.add_argument('--test-end', default='2024-12-31', help='Test (OOS) period end')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--min-trades', type=int, default=50, help='Minimum trades for valid trial')
    parser.add_argument('--max-dd', type=float, default=20.0, help='Max drawdown % before penalty')
    parser.add_argument('--initial-cash', type=float, default=100_000.0, help='Starting capital')
    parser.add_argument('--commission-rate', type=float, default=0.0002, help='Commission rate')
    parser.add_argument('--slippage-bps', type=float, default=3.0, help='Slippage in basis points')
    parser.add_argument('--output', default='results/optuna_structural_recal', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    print("=" * 80)
    print("OPTUNA RE-CALIBRATION: Structural Check Pipeline")
    print("=" * 80)
    print(f"Config:        {args.config}")
    print(f"Feature Store: {args.feature_store}")
    print(f"Train:         {args.train_start} to {args.train_end}")
    print(f"Test (OOS):    {args.test_start} to {args.test_end}")
    print(f"Trials:        {args.trials}")
    print(f"Parameters:    {len(THRESHOLD_RANGES)} archetypes + {len(GLOBAL_RANGES)} global = {len(THRESHOLD_RANGES) + len(GLOBAL_RANGES)} total")
    print(f"Capital:       ${args.initial_cash:,.0f}")
    print(f"Cost model:    {args.commission_rate*10000:.0f}bps comm + {args.slippage_bps:.0f}bps slip")
    print("=" * 80)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        base_config = json.load(f)

    # Load feature store (expensive — do once)
    features_df = load_data(str(PROJECT_ROOT / args.feature_store))

    # Create objective
    objective = StructuralRecalibrationObjective(
        base_config=base_config,
        features_df=features_df,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        min_trades=args.min_trades,
        max_dd_pct=args.max_dd,
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
    )

    # Create study
    study = optuna.create_study(
        study_name='structural_recalibration',
        direction='maximize',
        sampler=TPESampler(seed=args.seed, n_startup_trials=15),
        pruner=MedianPruner(n_startup_trials=10),
    )

    # Run optimization
    print(f"\nStarting optimization ({args.trials} trials)...\n")
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)
    total_time = time.time() - t0

    # ── Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    best = study.best_trial
    print(f"\nBest Trial: #{best.number}")
    print(f"Best Score: {best.value:.4f}")
    print(f"Total Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Avg Time/Trial: {total_time/args.trials:.0f}s")

    # Train metrics
    train_pf = best.user_attrs.get('profit_factor', 0)
    train_trades = best.user_attrs.get('total_trades', 0)
    train_dd = best.user_attrs.get('max_drawdown', 0)
    train_sharpe = best.user_attrs.get('sharpe_ratio', 0)
    train_pnl = best.user_attrs.get('total_pnl', 0)
    train_wr = best.user_attrs.get('win_rate', 0)

    print(f"\n--- TRAIN PERIOD ({args.train_start} to {args.train_end}) ---")
    print(f"PF:      {train_pf:.3f}")
    print(f"Trades:  {train_trades}")
    print(f"WR:      {train_wr:.1f}%")
    print(f"MaxDD:   {train_dd:.1f}%")
    print(f"Sharpe:  {train_sharpe:.2f}")
    print(f"PnL:     ${train_pnl:,.0f}")

    # Best parameters
    print(f"\n--- BEST PARAMETERS ---")
    print(f"{'Parameter':<35s} {'Value':>8s}  {'Old':>8s}")
    print("-" * 55)
    old_thresholds = base_config.get('adaptive_fusion', {}).get('per_archetype_base_threshold', {})
    old_base = base_config.get('adaptive_fusion', {}).get('base_threshold', 0.18)
    for arch_name in sorted(THRESHOLD_RANGES.keys()):
        key = f'bt_{arch_name}'
        new_val = best.params.get(key, 0)
        old_val = old_thresholds.get(arch_name, old_base)
        change = "=" if abs(new_val - old_val) < 0.005 else ("+" if new_val > old_val else "-")
        print(f"  {arch_name:<33s} {new_val:>6.2f}    {old_val:>6.2f}  {change}")

    old_af = base_config.get('adaptive_fusion', {})
    for gkey in ['temp_range', 'instab_range', 'crisis_coefficient']:
        new_val = best.params.get(gkey, 0)
        old_val = old_af.get(gkey, 0)
        change = "=" if abs(new_val - old_val) < 0.005 else ("+" if new_val > old_val else "-")
        print(f"  {gkey:<33s} {new_val:>6.2f}    {old_val:>6.2f}  {change}")

    # OOS Validation
    print(f"\n--- OOS VALIDATION ({args.test_start} to {args.test_end}) ---")
    oos_stats = validate_best(
        study=study,
        base_config=base_config,
        features_df=features_df,
        test_start=args.test_start,
        test_end=args.test_end,
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
    )

    oos_pf = oos_stats.get('profit_factor', 0)
    oos_trades = oos_stats.get('total_trades', 0)
    oos_dd = abs(oos_stats.get('max_drawdown', 0))
    oos_sharpe = oos_stats.get('sharpe_ratio', 0)
    oos_pnl = oos_stats.get('total_pnl', 0)
    oos_wr = oos_stats.get('win_rate', 0)

    print(f"PF:      {oos_pf:.3f}")
    print(f"Trades:  {oos_trades}")
    print(f"WR:      {oos_wr:.1f}%")
    print(f"MaxDD:   {oos_dd:.1f}%")
    print(f"Sharpe:  {oos_sharpe:.2f}")
    print(f"PnL:     ${oos_pnl:,.0f}")

    # Walk-Forward Efficiency
    wfe = (oos_pf / train_pf * 100) if train_pf > 0 else 0
    print(f"\nWalk-Forward Efficiency: {wfe:.0f}% (target: >70%)")
    if wfe >= 70:
        print("PASS: No overfitting detected")
    elif wfe >= 50:
        print("WARNING: Moderate degradation on OOS — consider widening search ranges")
    else:
        print("FAIL: Significant overfitting — results unreliable")

    # ── Save results ──────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters as config overlay
    overlay = {
        'per_archetype_base_threshold': {},
        'temp_range': best.params.get('temp_range', 0.38),
        'instab_range': best.params.get('instab_range', 0.15),
        'crisis_coefficient': best.params.get('crisis_coefficient', 0.50),
    }
    for arch_name in THRESHOLD_RANGES:
        key = f'bt_{arch_name}'
        if key in best.params:
            overlay['per_archetype_base_threshold'][arch_name] = best.params[key]

    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(overlay, f, indent=2)

    # Save full results
    results = {
        'best_trial': best.number,
        'best_score': best.value,
        'train_metrics': {
            'profit_factor': train_pf,
            'total_trades': train_trades,
            'max_drawdown': train_dd,
            'sharpe_ratio': train_sharpe,
            'total_pnl': train_pnl,
            'win_rate': train_wr,
        },
        'oos_metrics': {
            'profit_factor': oos_pf,
            'total_trades': oos_trades,
            'max_drawdown': oos_dd,
            'sharpe_ratio': oos_sharpe,
            'total_pnl': oos_pnl,
            'win_rate': oos_wr,
        },
        'walk_forward_efficiency': wfe,
        'best_params': overlay,
        'total_trials': args.trials,
        'total_time_s': total_time,
        'train_period': f"{args.train_start} to {args.train_end}",
        'test_period': f"{args.test_start} to {args.test_end}",
    }

    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save top 5 trials
    top_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -1e8],
        key=lambda t: t.value,
        reverse=True,
    )[:5]

    print(f"\n--- TOP 5 TRIALS ---")
    print(f"{'#':>4s} {'Score':>8s} {'PF':>8s} {'Trades':>8s} {'WR':>8s} {'MaxDD':>8s} {'Sharpe':>8s}")
    print("-" * 60)
    for t in top_trials:
        print(
            f"#{t.number:3d}  {t.value:>8.3f}  "
            f"{t.user_attrs.get('profit_factor', 0):>8.3f}  "
            f"{t.user_attrs.get('total_trades', 0):>8d}  "
            f"{t.user_attrs.get('win_rate', 0):>7.1f}%  "
            f"{t.user_attrs.get('max_drawdown', 0):>7.1f}%  "
            f"{t.user_attrs.get('sharpe_ratio', 0):>8.2f}"
        )

    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n--- PARAMETER IMPORTANCE (top 10) ---")
        for i, (param, imp) in enumerate(list(importances.items())[:10]):
            print(f"  {i+1:2d}. {param:<35s} {imp*100:>5.1f}%")
    except Exception:
        pass

    print(f"\nResults saved to: {output_dir}/")
    print(f"  best_params.json          — config overlay (apply to production config)")
    print(f"  optimization_results.json — full results with OOS validation")
    print("=" * 80)


if __name__ == '__main__':
    main()
