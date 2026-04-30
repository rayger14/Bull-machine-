#!/usr/bin/env python3
"""
Optuna Walk-Forward Optimization for Hybrid Mode — Top 2 Archetypes

Optimizes composite confirmation thresholds + CMI base_threshold for
wick_trap and trap_within_trend in the hybrid backtester.

Protocol: Anchored WFO with 2 windows, 40 TPE trials, 8 parameters.
  Window 1: Train 2020-2022, Test 2023
  Window 2: Train 2020-2023, Test 2024

Accept only if: PF > 1.2 in BOTH OOS windows, trades within 30% of baseline.

Usage:
    python3 bin/optuna_hybrid_top2.py --trials 40

Author: Claude Code
Date: 2026-03-17
Branch: feat/composite-confirmation-experiment
"""

import sys
import json
import argparse
import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"

# Target archetypes
TARGET_ARCHETYPES = ['wick_trap', 'trap_within_trend']

# WFO windows
WFO_WINDOWS = [
    {'train_start': '2020-01-01', 'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31', 'label': 'W1 (test 2023)'},
    {'train_start': '2020-01-01', 'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31', 'label': 'W2 (test 2024)'},
]


def run_hybrid_backtest(config, features_df, start_date, end_date, overrides=None):
    """Run hybrid backtest with optional parameter overrides.

    Returns dict with pf, pnl, trades, sharpe, max_dd, win_rate, per_archetype stats.
    """
    from bin.backtest_composite import (
        CompositeBacktester, COMPOSITE_DEFINITIONS, ArchetypeConfirmation,
        Confirmation, get_confirmation
    )

    cfg = copy.deepcopy(config)

    # Apply overrides to composite definitions
    if overrides:
        _apply_overrides(overrides)
        # Also update per_archetype_base_threshold in config
        if 'adaptive_fusion' not in cfg:
            cfg['adaptive_fusion'] = {}
        if 'per_archetype_base_threshold' not in cfg['adaptive_fusion']:
            cfg['adaptive_fusion']['per_archetype_base_threshold'] = {}

        for arch in TARGET_ARCHETYPES:
            key = f'{arch}_base_threshold'
            if key in overrides:
                cfg['adaptive_fusion']['per_archetype_base_threshold'][arch] = overrides[key]

    bt = CompositeBacktester(
        config=cfg,
        features_df=features_df.copy(),
        initial_cash=100_000.0,
        commission_rate=0.0002,
        slippage_bps=3.0,
        mode='hybrid',
        archetype_filter=TARGET_ARCHETYPES,
        vetoes_enabled=True,
    )

    import logging
    logging.disable(logging.WARNING)

    # Suppress prints
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        bt.run(start_date=start_date, end_date=end_date)

    logging.disable(logging.NOTSET)

    agg = bt.get_aggregate_stats()
    per_arch = bt.get_per_archetype_stats()

    # Restore original definitions
    if overrides:
        _restore_defaults()

    return {
        'pf': agg.get('profit_factor', 0),
        'pnl': agg.get('total_pnl', 0),
        'trades': agg.get('total_trades', 0),
        'sharpe': agg.get('sharpe_ratio', 0),
        'max_dd': agg.get('max_drawdown', 0),
        'win_rate': agg.get('win_rate', 0),
        'per_archetype': per_arch,
    }


# Store original definitions for reset
_ORIGINAL_DEFINITIONS = {}

def _save_defaults():
    """Save original composite definitions."""
    from bin.backtest_composite import COMPOSITE_DEFINITIONS
    for arch in TARGET_ARCHETYPES:
        if arch in COMPOSITE_DEFINITIONS:
            _ORIGINAL_DEFINITIONS[arch] = copy.deepcopy(COMPOSITE_DEFINITIONS[arch])

def _restore_defaults():
    """Restore original composite definitions."""
    from bin.backtest_composite import COMPOSITE_DEFINITIONS
    for arch, orig in _ORIGINAL_DEFINITIONS.items():
        COMPOSITE_DEFINITIONS[arch] = orig

def _apply_overrides(overrides):
    """Apply trial overrides to composite definitions."""
    from bin.backtest_composite import (
        COMPOSITE_DEFINITIONS, ArchetypeConfirmation, Confirmation
    )

    # wick_trap overrides
    if 'wick_trap' in COMPOSITE_DEFINITIONS:
        wt = COMPOSITE_DEFINITIONS['wick_trap']
        new_confs = []
        for c in wt.confirmations:
            new_c = copy.deepcopy(c)
            if c.name == 'liquidity_present' and 'wt_liquidity_min' in overrides:
                new_c.value = overrides['wt_liquidity_min']
            elif c.name == 'rsi_oversold_zone' and 'wt_rsi_max' in overrides:
                new_c.value = overrides['wt_rsi_max']
            elif c.name == 'momentum_positive' and 'wt_momentum_min' in overrides:
                new_c.value = overrides['wt_momentum_min']
            elif c.name == 'wyckoff_bullish' and 'wt_wyckoff_min' in overrides:
                new_c.value = overrides['wt_wyckoff_min']
            elif c.name == 'temporal_confluence' and 'wt_temporal_min' in overrides:
                new_c.value = overrides['wt_temporal_min']
            new_confs.append(new_c)
        COMPOSITE_DEFINITIONS['wick_trap'] = ArchetypeConfirmation(
            archetype='wick_trap',
            min_score=wt.min_score,
            max_possible=wt.max_possible,
            confirmations=new_confs,
        )

    # trap_within_trend overrides
    if 'trap_within_trend' in COMPOSITE_DEFINITIONS:
        twt = COMPOSITE_DEFINITIONS['trap_within_trend']
        new_confs = []
        for c in twt.confirmations:
            new_c = copy.deepcopy(c)
            if c.name == 'adx_trending' and 'twt_adx_min' in overrides:
                new_c.value = overrides['twt_adx_min']
            elif c.name == 'trend_aligned' and 'twt_ema_slope_min' in overrides:
                new_c.value = overrides['twt_ema_slope_min']
            elif c.name == 'liquidity_present' and 'twt_liquidity_min' in overrides:
                new_c.value = overrides['twt_liquidity_min']
            elif c.name == 'rsi_not_extreme_high' and 'twt_rsi_max' in overrides:
                new_c.value = overrides['twt_rsi_max']
            new_confs.append(new_c)
        COMPOSITE_DEFINITIONS['trap_within_trend'] = ArchetypeConfirmation(
            archetype='trap_within_trend',
            min_score=twt.min_score,
            max_possible=twt.max_possible,
            confirmations=new_confs,
        )


def create_objective(features_df, config, window):
    """Create Optuna objective for a single WFO window."""

    def objective(trial):
        # 8 parameters: 4 per archetype
        overrides = {
            # wick_trap confirmation thresholds
            'wt_liquidity_min': trial.suggest_float('wt_liquidity_min', 0.1, 0.6),
            'wt_rsi_max': trial.suggest_float('wt_rsi_max', 35.0, 55.0),
            'wt_momentum_min': trial.suggest_float('wt_momentum_min', 0.1, 0.5),
            'wt_wyckoff_min': trial.suggest_float('wt_wyckoff_min', 0.05, 0.25),
            # trap_within_trend confirmation thresholds
            'twt_adx_min': trial.suggest_float('twt_adx_min', 10.0, 25.0),
            'twt_ema_slope_min': trial.suggest_float('twt_ema_slope_min', -0.005, 0.005),
            'twt_liquidity_min': trial.suggest_float('twt_liquidity_min', 0.1, 0.5),
            'twt_rsi_max': trial.suggest_float('twt_rsi_max', 55.0, 75.0),
        }

        # Run on TRAIN period
        result = run_hybrid_backtest(
            config, features_df,
            start_date=window['train_start'],
            end_date=window['train_end'],
            overrides=overrides,
        )

        pf = result['pf']
        trades = result['trades']
        max_dd = abs(result['max_dd'])

        # Hard constraints (return -inf)
        if trades < 30:
            return -1e9
        if max_dd > 20.0:
            return -1e9

        # Score: PF with mild DD penalty
        dd_penalty = max(0, (max_dd - 10.0)) * 0.02
        score = pf - dd_penalty if pf > 1.0 else pf - 1.0

        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna WFO for Hybrid Top-2')
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--feature-store', default=DEFAULT_FEATURE_STORE)
    args = parser.parse_args()

    n_trials = args.trials

    print("=" * 100)
    print("OPTUNA WALK-FORWARD OPTIMIZATION — Hybrid Mode Top-2 Archetypes")
    print(f"Archetypes: {TARGET_ARCHETYPES}")
    print(f"Trials per window: {n_trials}")
    print(f"Parameters: 8 (4 per archetype)")
    print(f"Config: $100K initial, 2bps commission, 3bps slippage, 2% risk/trade")
    print("=" * 100)

    # Load data
    print("\nLoading feature store...")
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = json.load(f)
    features_df = pd.read_parquet(str(PROJECT_ROOT / args.feature_store))
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index)
    features_df = features_df.sort_index()
    print(f"Loaded {len(features_df):,} bars\n")

    # Save defaults
    _save_defaults()

    # Step 1: Baseline
    print("=" * 100)
    print("BASELINE (current thresholds)")
    print("=" * 100)

    baseline_results = {}
    for window in WFO_WINDOWS:
        # Test baseline on TEST period
        result = run_hybrid_backtest(
            config, features_df,
            start_date=window['test_start'],
            end_date=window['test_end'],
        )
        baseline_results[window['label']] = result
        print(f"  {window['label']}: PF={result['pf']:.2f}, PnL=${result['pnl']:,.0f}, "
              f"Trades={result['trades']}, Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.1f}%")
        for arch, stats in result.get('per_archetype', {}).items():
            pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 100 else "inf"
            print(f"    {arch}: PF={pf_str}, Trades={stats['trades']}, PnL=${stats['total_pnl']:,.0f}")

    # Step 2: WFO Optimization
    sampler = TPESampler(
        seed=42,
        n_startup_trials=min(15, n_trials // 3),
        multivariate=True,
    )

    best_params_per_window = {}
    oos_results = {}

    for wi, window in enumerate(WFO_WINDOWS):
        print(f"\n{'=' * 100}")
        print(f"WINDOW {wi+1}: Train {window['train_start']}—{window['train_end']}, "
              f"Test {window['test_start']}—{window['test_end']}")
        print(f"{'=' * 100}")

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'hybrid_top2_w{wi+1}',
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        objective = create_objective(features_df, config, window)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\nBest trial #{best.number}: score={best.value:.4f}")
        print("Parameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v:.4f}")

        best_params_per_window[window['label']] = best.params

        # Validate on OOS (test period)
        print(f"\nOOS Validation ({window['test_start']}—{window['test_end']}):")
        oos = run_hybrid_backtest(
            config, features_df,
            start_date=window['test_start'],
            end_date=window['test_end'],
            overrides=best.params,
        )
        oos_results[window['label']] = oos
        print(f"  PF={oos['pf']:.2f}, PnL=${oos['pnl']:,.0f}, Trades={oos['trades']}, "
              f"Sharpe={oos['sharpe']:.2f}, MaxDD={oos['max_dd']:.1f}%, WR={oos['win_rate']:.1f}%")
        for arch, stats in oos.get('per_archetype', {}).items():
            pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 100 else "inf"
            print(f"    {arch}: PF={pf_str}, Trades={stats['trades']}, PnL=${stats['total_pnl']:,.0f}")

        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\nParameter importance:")
            for k, v in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "█" * int(v * 40)
                print(f"  {k:<25s} {v:>6.1%} {bar}")
        except Exception:
            pass

    # Step 3: WFO Validation Summary
    print(f"\n{'=' * 100}")
    print("WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'=' * 100}")

    all_oos_pf = [r['pf'] for r in oos_results.values()]
    all_oos_trades = [r['trades'] for r in oos_results.values()]
    all_baseline_pf = [r['pf'] for r in baseline_results.values()]
    all_baseline_trades = [r['trades'] for r in baseline_results.values()]

    print(f"\n{'Window':<20s} {'Baseline PF':>12s} {'Optimized PF':>13s} {'Δ PF':>8s} "
          f"{'Baseline Tr':>12s} {'Optimized Tr':>12s}")
    print("-" * 80)
    for label in oos_results:
        bl = baseline_results[label]
        opt = oos_results[label]
        delta_pf = opt['pf'] - bl['pf']
        print(f"{label:<20s} {bl['pf']:>12.2f} {opt['pf']:>13.2f} {delta_pf:>+8.2f} "
              f"{bl['trades']:>12d} {opt['trades']:>12d}")

    min_oos_pf = min(all_oos_pf)

    # Anti-overfit checks
    print(f"\nAnti-Overfit Checks:")

    pf_check = min_oos_pf > 1.2
    print(f"  [{'✓' if pf_check else '✗'}] OOS PF > 1.2 in every window: min={min_oos_pf:.2f}")

    trade_check = True
    for label in oos_results:
        bl_trades = baseline_results[label]['trades']
        opt_trades = oos_results[label]['trades']
        ratio = opt_trades / bl_trades if bl_trades > 0 else 0
        ok = 0.7 <= ratio <= 1.3
        if not ok:
            trade_check = False
        print(f"  [{'✓' if ok else '✗'}] Trade count {label}: {opt_trades} vs baseline {bl_trades} "
              f"(ratio={ratio:.2f}, need 0.7-1.3)")

    # Check parameter consistency across windows
    param_keys = list(best_params_per_window[WFO_WINDOWS[0]['label']].keys())
    print(f"\nParameter consistency across windows:")
    param_stable = True
    for k in param_keys:
        vals = [best_params_per_window[w['label']][k] for w in WFO_WINDOWS]
        cv = np.std(vals) / np.mean(vals) * 100 if np.mean(vals) != 0 else 0
        stable = cv < 50
        if not stable:
            param_stable = False
        print(f"  {k:<25s} W1={vals[0]:.4f} W2={vals[1]:.4f} CV={cv:.0f}% {'✓' if stable else '⚠ unstable'}")

    # Final verdict
    print(f"\n{'=' * 100}")
    if pf_check and trade_check:
        print("VERDICT: ✓ ACCEPT — OOS PF > 1.2 in all windows, trade count stable")

        # Average the best params across windows
        avg_params = {}
        for k in param_keys:
            vals = [best_params_per_window[w['label']][k] for w in WFO_WINDOWS]
            avg_params[k] = np.mean(vals)

        print(f"\nAVERAGED PARAMETERS (for deployment):")
        for k, v in avg_params.items():
            print(f"  {k}: {v:.4f}")

        # Full-period validation with averaged params
        print(f"\nFull-period validation (2020-2024) with averaged params:")
        full = run_hybrid_backtest(
            config, features_df,
            start_date='2020-01-01', end_date='2024-12-31',
            overrides=avg_params,
        )
        bl_full = run_hybrid_backtest(
            config, features_df,
            start_date='2020-01-01', end_date='2024-12-31',
        )
        print(f"  Baseline:   PF={bl_full['pf']:.2f}, PnL=${bl_full['pnl']:,.0f}, "
              f"Trades={bl_full['trades']}, Sharpe={bl_full['sharpe']:.2f}, MaxDD={bl_full['max_dd']:.1f}%")
        print(f"  Optimized:  PF={full['pf']:.2f}, PnL=${full['pnl']:,.0f}, "
              f"Trades={full['trades']}, Sharpe={full['sharpe']:.2f}, MaxDD={full['max_dd']:.1f}%")

        for arch, stats in full.get('per_archetype', {}).items():
            bl_stats = bl_full.get('per_archetype', {}).get(arch, {})
            pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 100 else "inf"
            bl_pf = f"{bl_stats.get('profit_factor', 0):.2f}" if bl_stats.get('profit_factor', 0) < 100 else "inf"
            print(f"    {arch}: PF {bl_pf}→{pf_str}, "
                  f"Trades {bl_stats.get('trades', 0)}→{stats['trades']}, "
                  f"PnL ${bl_stats.get('total_pnl', 0):,.0f}→${stats['total_pnl']:,.0f}")

    elif not pf_check:
        print(f"VERDICT: ✗ REJECT — OOS PF below 1.2 in at least one window (min={min_oos_pf:.2f})")
        print("The composite confirmation thresholds may not have enough discriminating power.")
    else:
        print(f"VERDICT: ✗ REJECT — Trade count changed > 30% in at least one window")
        print("The optimizer found a niche overfit, not a better filter.")

    print(f"{'=' * 100}")

    # Restore
    _restore_defaults()


if __name__ == '__main__':
    main()
