#!/usr/bin/env python3
"""
Parallel Group Optuna Optimizer

Optimizes a SUBSET of parameters while holding others at baseline.
Designed to be run in parallel across 6 groups for ~6x speedup.

Groups:
  1: Top earners base_threshold (wick_trap, retest_cluster, liquidity_sweep, trap_within_trend)
  2: Mid-tier base_threshold (spring, failed_continuation, order_block_retest, liquidity_vacuum)
  3: New archetypes base_threshold (funding_divergence, long_squeeze, fvg_continuation, exhaustion_reversal)
  4: Global CMI params (temp_range, instab_range, crisis_coefficient)
  5: Structural gate thresholds (wick_pct, vol_z, RSI, BOS proximity, funding_z)
  6: ATR stop/TP multipliers (6 active archetypes × 2 params)

Usage:
    python3 bin/optuna_parallel_group.py --group 1 --trials 50
    python3 bin/optuna_parallel_group.py --group 5 --trials 50

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
from typing import Dict, Any

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from bin.backtest_v11_standalone import StandaloneBacktestEngine

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"

# Baseline thresholds (from structural check backtest)
BASELINE_THRESHOLDS = {
    'wick_trap': 0.10,
    'retest_cluster': 0.06,
    'liquidity_sweep': 0.08,
    'trap_within_trend': 0.08,
    'spring': 0.08,
    'failed_continuation': 0.08,
    'order_block_retest': 0.10,
    'liquidity_vacuum': 0.08,
    'funding_divergence': 0.18,
    'long_squeeze': 0.18,
    'fvg_continuation': 0.18,
    'exhaustion_reversal': 0.18,
}

BASELINE_GLOBALS = {
    'temp_range': 0.38,
    'instab_range': 0.15,
    'crisis_coefficient': 0.50,
}

# Baseline structural gate params (current hardcoded defaults in logic.py)
BASELINE_GATES = {
    'wick_pct_K': 0.35,    # wick_trap + trap_within_trend wick threshold
    'wick_pct_G': 0.35,    # liquidity_sweep lower wick threshold
    'vol_z_L': 1.0,        # retest_cluster volume z-score min
    'rsi_upper_L': 70.0,   # retest_cluster RSI overbought
    'rsi_lower_L': 30.0,   # retest_cluster RSI oversold
    'rsi_upper_F': 78.0,   # exhaustion_reversal RSI upper
    'rsi_lower_F': 22.0,   # exhaustion_reversal RSI lower
    'bos_atr_B': 1.5,      # order_block_retest BOS proximity (ATR units)
    'funding_z_S4': -1.0,  # funding_divergence Z threshold
    'funding_z_S5': 1.0,   # long_squeeze Z threshold
}

# Baseline ATR stop/TP multipliers (from configs/archetypes/*.yaml)
BASELINE_ATR = {
    'atr_stop_wick_trap': 3.4,
    'atr_tp_wick_trap': 4.0,
    'atr_stop_retest_cluster': 3.4,
    'atr_tp_retest_cluster': 5.4,
    'atr_stop_liquidity_sweep': 3.4,
    'atr_tp_liquidity_sweep': 5.7,
    'atr_stop_trap_within_trend': 2.9,
    'atr_tp_trap_within_trend': 3.8,
    'atr_stop_spring': 1.3,
    'atr_tp_spring': 3.1,
    'atr_stop_failed_continuation': 1.8,
    'atr_tp_failed_continuation': 6.0,
}

# Group definitions: parameters to optimize + their search ranges
GROUPS = {
    1: {
        'name': 'top_earners',
        'archetypes': {
            'wick_trap':        (0.03, 0.18),
            'retest_cluster':   (0.03, 0.15),
            'liquidity_sweep':  (0.03, 0.18),
            'trap_within_trend':(0.03, 0.15),
        },
        'globals': {},
        'structural_gates': {},
        'atr_params': {},
    },
    2: {
        'name': 'mid_tier',
        'archetypes': {
            'spring':              (0.03, 0.18),
            'failed_continuation': (0.03, 0.18),
            'order_block_retest':  (0.03, 0.18),
            'liquidity_vacuum':    (0.03, 0.15),
        },
        'globals': {},
        'structural_gates': {},
        'atr_params': {},
    },
    3: {
        'name': 'new_archetypes',
        'archetypes': {
            'funding_divergence':  (0.03, 0.20),
            'long_squeeze':        (0.03, 0.20),
            'fvg_continuation':    (0.03, 0.20),
            'exhaustion_reversal': (0.03, 0.20),
        },
        'globals': {},
        'structural_gates': {},
        'atr_params': {},
    },
    4: {
        'name': 'global_cmi',
        'archetypes': {},
        'globals': {
            'temp_range':        (0.25, 0.55),
            'instab_range':      (0.08, 0.25),
            'crisis_coefficient': (0.30, 0.70),
        },
        'structural_gates': {},
        'atr_params': {},
    },
    5: {
        'name': 'structural_gates',
        'archetypes': {},
        'globals': {},
        'structural_gates': {
            'wick_pct_K':    (0.20, 0.50),   # wick_trap + trap_within_trend
            'wick_pct_G':    (0.20, 0.50),   # liquidity_sweep
            'vol_z_L':       (0.5, 2.0),     # retest_cluster vol z-score min
            'rsi_upper_L':   (65.0, 80.0),   # retest_cluster RSI overbought
            'rsi_lower_L':   (20.0, 35.0),   # retest_cluster RSI oversold
            'rsi_upper_F':   (72.0, 85.0),   # exhaustion_reversal RSI upper
            'rsi_lower_F':   (15.0, 28.0),   # exhaustion_reversal RSI lower
            'bos_atr_B':     (0.8, 3.0),     # order_block_retest BOS proximity
            'funding_z_S4':  (-2.0, -0.5),   # funding_divergence Z threshold
            'funding_z_S5':  (0.5, 2.0),     # long_squeeze Z threshold
        },
        'atr_params': {},
    },
    6: {
        'name': 'atr_multipliers',
        'archetypes': {},
        'globals': {},
        'structural_gates': {},
        'atr_params': {
            'atr_stop_wick_trap':          (1.5, 5.0),
            'atr_tp_wick_trap':            (2.0, 8.0),
            'atr_stop_retest_cluster':     (1.5, 5.0),
            'atr_tp_retest_cluster':       (2.0, 8.0),
            'atr_stop_liquidity_sweep':    (1.5, 5.0),
            'atr_tp_liquidity_sweep':      (2.0, 8.0),
            'atr_stop_trap_within_trend':  (1.5, 5.0),
            'atr_tp_trap_within_trend':    (2.0, 8.0),
            'atr_stop_spring':             (0.8, 4.0),
            'atr_tp_spring':               (2.0, 8.0),
            'atr_stop_failed_continuation':(0.8, 4.0),
            'atr_tp_failed_continuation':  (2.0, 8.0),
        },
    },
}


def run_backtest(config, features_df, start_date, end_date, initial_cash=100_000.0,
                 commission_rate=0.0002, slippage_bps=3.0):
    engine = StandaloneBacktestEngine(
        config=config, initial_cash=initial_cash,
        commission_rate=commission_rate, slippage_bps=slippage_bps,
        features_df=features_df,
    )
    engine.run(start_date=start_date, end_date=end_date)
    return engine.get_performance_stats()


class GroupObjective:
    def __init__(self, base_config, features_df, group_def,
                 train_start, train_end, min_trades=50, max_dd=20.0):
        self.base_config = base_config
        self.features_df = features_df
        self.group_def = group_def
        self.train_start = train_start
        self.train_end = train_end
        self.min_trades = min_trades
        self.max_dd = max_dd
        self.best_score = -999
        self.best_trial = -1

    def __call__(self, trial):
        t0 = time.time()
        config = copy.deepcopy(self.base_config)
        af = config.setdefault('adaptive_fusion', {})
        af['enabled'] = True
        af['bypass_threshold'] = False

        # Start with ALL archetypes at baseline
        per_arch = dict(BASELINE_THRESHOLDS)

        # Override this group's archetypes with trial values
        for arch_name, (lo, hi) in self.group_def['archetypes'].items():
            val = trial.suggest_float(f'bt_{arch_name}', lo, hi, step=0.01)
            per_arch[arch_name] = val

        af['per_archetype_base_threshold'] = per_arch

        # Global params: either optimize or hold at baseline
        if self.group_def['globals']:
            for gkey, (lo, hi) in self.group_def['globals'].items():
                val = trial.suggest_float(gkey, lo, hi, step=0.01)
                af[gkey] = val
        else:
            for gkey, gval in BASELINE_GLOBALS.items():
                af[gkey] = gval

        # Structural gate params: inject into structural_checks config section
        structural_gates = self.group_def.get('structural_gates', {})
        if structural_gates:
            gate_params = dict(BASELINE_GATES)  # Start with baseline
            for gkey, (lo, hi) in structural_gates.items():
                val = trial.suggest_float(gkey, lo, hi)
                gate_params[gkey] = val
            sc = config.setdefault('structural_checks', {})
            sc['enabled'] = True
            sc['gate_params'] = gate_params
        else:
            # Ensure structural checks use baseline gates
            sc = config.setdefault('structural_checks', {})
            sc['enabled'] = True
            sc['gate_params'] = dict(BASELINE_GATES)

        # ATR stop/TP multiplier overrides
        atr_params = self.group_def.get('atr_params', {})
        if atr_params:
            atr_overrides = {}
            for pkey, (lo, hi) in atr_params.items():
                val = trial.suggest_float(pkey, lo, hi, step=0.1)
                # Parse "atr_stop_wick_trap" → archetype="wick_trap", field="atr_stop_mult"
                parts = pkey.split('_', 2)  # ['atr', 'stop'/'tp', 'archetype_name']
                field_type = parts[1]  # 'stop' or 'tp'
                arch_name = pkey.replace(f'atr_{field_type}_', '')
                field = f'atr_{field_type}_mult'
                if arch_name not in atr_overrides:
                    atr_overrides[arch_name] = {}
                atr_overrides[arch_name][field] = val
            config['atr_overrides'] = atr_overrides

        try:
            stats = run_backtest(
                config=config, features_df=self.features_df,
                start_date=self.train_start, end_date=self.train_end,
            )
        except Exception as e:
            logger.warning(f"Trial {trial.number} crashed: {e}")
            return -1e9

        elapsed = time.time() - t0
        pf = stats.get('profit_factor', 0.0)
        trades = stats.get('total_trades', 0)
        max_dd = abs(stats.get('max_drawdown', 0.0))
        sharpe = stats.get('sharpe_ratio', 0.0)
        pnl = stats.get('total_pnl', 0.0)
        wr = stats.get('win_rate', 0.0)

        if trades < self.min_trades:
            print(f"  Trial {trial.number:3d} | PENALTY: {trades} trades < {self.min_trades} | {elapsed:.0f}s")
            return -1e9
        if max_dd > self.max_dd:
            print(f"  Trial {trial.number:3d} | PENALTY: MaxDD={max_dd:.1f}% | PF={pf:.2f} | {elapsed:.0f}s")
            return -1e9

        dd_penalty = max(0, (max_dd - 8.0)) * 0.02
        score = pf - dd_penalty if pf > 1.0 else pf - 1.0

        trial.set_user_attr('profit_factor', pf)
        trial.set_user_attr('total_trades', trades)
        trial.set_user_attr('max_drawdown', max_dd)
        trial.set_user_attr('sharpe_ratio', sharpe)
        trial.set_user_attr('total_pnl', pnl)
        trial.set_user_attr('win_rate', wr)

        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_trial = trial.number

        print(
            f"  Trial {trial.number:3d} | PF={pf:.3f} | trades={trades:4d} | WR={wr:.1f}% | "
            f"MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f} | PnL=${pnl:,.0f} | "
            f"score={score:.3f} {'*** BEST' if is_best else ''} | {elapsed:.0f}s"
        )
        return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, required=True, choices=[1,2,3,4,5,6])
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--feature-store', default=DEFAULT_FEATURE_STORE)
    parser.add_argument('--train-start', default='2020-01-01')
    parser.add_argument('--train-end', default='2022-12-31')
    parser.add_argument('--test-start', default='2023-01-01')
    parser.add_argument('--test-end', default='2024-12-31')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    group_def = GROUPS[args.group]
    n_params = (len(group_def['archetypes']) + len(group_def['globals']) +
                len(group_def.get('structural_gates', {})) + len(group_def.get('atr_params', {})))

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    print("=" * 80)
    print(f"GROUP {args.group}: {group_def['name'].upper()} ({n_params} parameters, {args.trials} trials)")
    print("=" * 80)

    if group_def['archetypes']:
        print("Archetypes:")
        for name, (lo, hi) in group_def['archetypes'].items():
            print(f"  {name:<25s}  range=[{lo:.2f}, {hi:.2f}]  baseline={BASELINE_THRESHOLDS[name]:.2f}")
    if group_def['globals']:
        print("Global params:")
        for name, (lo, hi) in group_def['globals'].items():
            print(f"  {name:<25s}  range=[{lo:.2f}, {hi:.2f}]  baseline={BASELINE_GLOBALS[name]:.2f}")
    if group_def.get('structural_gates'):
        print("Structural gates:")
        for name, (lo, hi) in group_def['structural_gates'].items():
            print(f"  {name:<25s}  range=[{lo}, {hi}]  baseline={BASELINE_GATES[name]}")
    if group_def.get('atr_params'):
        print("ATR multipliers:")
        for name, (lo, hi) in group_def['atr_params'].items():
            print(f"  {name:<30s}  range=[{lo}, {hi}]  baseline={BASELINE_ATR[name]}")
    print(f"Train: {args.train_start} to {args.train_end}")
    print("=" * 80)

    # Load data
    print("Loading feature store...")
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        base_config = json.load(f)
    features_df = pd.read_parquet(str(PROJECT_ROOT / args.feature_store))
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index)
    features_df = features_df.sort_index()
    print(f"Loaded {len(features_df):,} bars\n")

    # Create study
    objective = GroupObjective(
        base_config=base_config, features_df=features_df,
        group_def=group_def, train_start=args.train_start,
        train_end=args.train_end,
    )

    study = optuna.create_study(
        study_name=f'group_{args.group}_{group_def["name"]}',
        direction='maximize',
        sampler=TPESampler(seed=args.seed, n_startup_trials=min(10, args.trials // 3)),
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)
    total_time = time.time() - t0

    # Results
    best = study.best_trial
    print(f"\n{'=' * 80}")
    print(f"GROUP {args.group} RESULTS: {group_def['name'].upper()}")
    print(f"{'=' * 80}")
    print(f"Best Trial: #{best.number} | Score: {best.value:.4f}")
    print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min) | {total_time/args.trials:.0f}s/trial")
    print(f"\nTrain: PF={best.user_attrs.get('profit_factor',0):.3f} | "
          f"Trades={best.user_attrs.get('total_trades',0)} | "
          f"WR={best.user_attrs.get('win_rate',0):.1f}% | "
          f"MaxDD={best.user_attrs.get('max_drawdown',0):.1f}% | "
          f"Sharpe={best.user_attrs.get('sharpe_ratio',0):.2f} | "
          f"PnL=${best.user_attrs.get('total_pnl',0):,.0f}")

    print(f"\nBest Parameters:")
    for param, val in sorted(best.params.items()):
        baseline_key = param.replace('bt_', '')
        old = (BASELINE_THRESHOLDS.get(baseline_key) or
               BASELINE_GLOBALS.get(baseline_key) or
               BASELINE_GATES.get(param) or
               BASELINE_ATR.get(param) or '?')
        print(f"  {param:<35s} {val:>6.2f}  (was {old})")

    # OOS Validation
    print(f"\nRunning OOS validation ({args.test_start} to {args.test_end})...")
    oos_config = copy.deepcopy(base_config)
    af = oos_config.setdefault('adaptive_fusion', {})
    af['enabled'] = True
    af['bypass_threshold'] = False
    per_arch = dict(BASELINE_THRESHOLDS)
    for arch_name in group_def['archetypes']:
        key = f'bt_{arch_name}'
        if key in best.params:
            per_arch[arch_name] = best.params[key]
    af['per_archetype_base_threshold'] = per_arch
    for gkey in group_def['globals']:
        if gkey in best.params:
            af[gkey] = best.params[gkey]
    for gkey, gval in BASELINE_GLOBALS.items():
        if gkey not in group_def['globals']:
            af[gkey] = gval

    # Apply structural gate params for OOS
    if group_def.get('structural_gates'):
        gate_params = dict(BASELINE_GATES)
        for gkey in group_def['structural_gates']:
            if gkey in best.params:
                gate_params[gkey] = best.params[gkey]
        sc = oos_config.setdefault('structural_checks', {})
        sc['enabled'] = True
        sc['gate_params'] = gate_params
    else:
        sc = oos_config.setdefault('structural_checks', {})
        sc['enabled'] = True
        sc['gate_params'] = dict(BASELINE_GATES)

    # Apply ATR overrides for OOS
    if group_def.get('atr_params'):
        atr_overrides = {}
        for pkey in group_def['atr_params']:
            if pkey in best.params:
                parts = pkey.split('_', 2)
                field_type = parts[1]
                arch_name = pkey.replace(f'atr_{field_type}_', '')
                field = f'atr_{field_type}_mult'
                if arch_name not in atr_overrides:
                    atr_overrides[arch_name] = {}
                atr_overrides[arch_name][field] = best.params[pkey]
        oos_config['atr_overrides'] = atr_overrides

    oos = run_backtest(oos_config, features_df, args.test_start, args.test_end)
    oos_pf = oos.get('profit_factor', 0)
    train_pf = best.user_attrs.get('profit_factor', 1)
    wfe = (oos_pf / train_pf * 100) if train_pf > 0 else 0

    print(f"OOS:   PF={oos_pf:.3f} | Trades={oos.get('total_trades',0)} | "
          f"WR={oos.get('win_rate',0):.1f}% | MaxDD={abs(oos.get('max_drawdown',0)):.1f}% | "
          f"Sharpe={oos.get('sharpe_ratio',0):.2f} | PnL=${oos.get('total_pnl',0):,.0f}")
    print(f"WFE:   {wfe:.0f}% {'PASS' if wfe >= 70 else 'WARN' if wfe >= 50 else 'FAIL'}")

    # Save
    output_dir = PROJECT_ROOT / f'results/optuna_group_{args.group}_{group_def["name"]}'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'group': args.group,
        'name': group_def['name'],
        'best_trial': best.number,
        'best_score': best.value,
        'best_params': best.params,
        'train_metrics': dict(best.user_attrs),
        'oos_metrics': {k: oos.get(k, 0) for k in ['profit_factor','total_trades','max_drawdown','sharpe_ratio','total_pnl','win_rate']},
        'wfe': wfe,
        'total_time_s': total_time,
        'trials': args.trials,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Top 5
    top = sorted([t for t in study.trials if t.value and t.value > -1e8],
                 key=lambda t: t.value, reverse=True)[:5]
    print(f"\nTop 5:")
    for t in top:
        print(f"  #{t.number:3d} score={t.value:.3f} PF={t.user_attrs.get('profit_factor',0):.3f} "
              f"trades={t.user_attrs.get('total_trades',0)}")

    # Parameter importance
    try:
        imp = optuna.importance.get_param_importances(study)
        print(f"\nParameter Importance:")
        for p, v in imp.items():
            print(f"  {p:<35s} {v*100:>5.1f}%")
    except Exception:
        pass

    print(f"\nSaved to: {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
