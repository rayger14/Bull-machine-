#!/usr/bin/env python3
"""
Optuna Gate Threshold Optimizer

Optimizes YAML gate threshold values added in the strategy notebook gap analysis.
Uses gate_overrides config mechanism to inject trial values into the standalone backtester.

Groups:
  A: Active production archetypes (high trade count — trap_within_trend, retest_cluster,
     liquidity_sweep, failed_continuation, liquidity_vacuum)
  B: Secondary archetypes (low trade count — funding_divergence, order_block_retest,
     confluence_breakout, volume_fade_chop, liquidity_compression)

Usage:
    python3 bin/optuna_optimize_gates.py --group A --trials 60
    python3 bin/optuna_optimize_gates.py --group B --trials 60
    python3 bin/optuna_optimize_gates.py --group A --trials 60 --train-start 2020-01-01 --train-end 2022-12-31

Author: Claude Code
Date: 2026-03-11
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

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.WARNING)

from bin.backtest_v11_standalone import StandaloneBacktestEngine

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"


# ── Gate Parameter Definitions ─────────────────────────────────────────
# Format: {param_name: (archetype, feature, op, current_value, low, high, step)}

GROUP_A_PARAMS = {
    # trap_within_trend (hard mode, 22 trades baseline)
    'twt_wick_lower_ratio':  ('trap_within_trend', 'wick_lower_ratio',  'min', 0.3,  0.05, 0.8,  0.05),
    'twt_bars_since_pivot':  ('trap_within_trend', 'bars_since_pivot',  'max', 55,   20,   120,  5),
    # retest_cluster (soft mode, 297 trades baseline)
    'rc_temporal_conf':      ('retest_cluster',    'temporal_confluence_score', 'min', 0.3, 0.05, 0.6, 0.05),
    # liquidity_sweep (soft mode, 109 trades baseline)
    'ls_wick_lower_ratio':   ('liquidity_sweep',   'wick_lower_ratio',  'min', 1.0,  0.2,  2.0,  0.1),
    # failed_continuation (soft mode, 23 trades baseline)
    'fc_vol_z_max':          ('failed_continuation','volume_zscore',    'max', 2.0,  0.5,  5.0,  0.25),
    'fc_effort_ratio':       ('failed_continuation','effort_result_ratio','max', 1.5, 0.3,  3.0,  0.1),
    # liquidity_vacuum (hard mode, small count)
    'lv_wick_exhaust':       ('liquidity_vacuum',  'wick_exhaustion_last_3b', 'min', 1.0, 0.3, 3.0, 0.1),
}

GROUP_B_PARAMS = {
    # funding_divergence (soft mode, 22 trades baseline)
    'fd_oi_price_div':       ('funding_divergence', 'oi_price_divergence', 'min', 0.01, 0.001, 0.15, 0.005),
    'fd_oi_change_4h':       ('funding_divergence', 'oi_change_4h',       'max', 0.05, -0.1,  0.2,  0.01),
    # order_block_retest (soft mode, 1 trade baseline)
    'obr_fib_time_conf':     ('order_block_retest', 'fib_time_confluence','min', 0.1,  0.01,  0.5,  0.01),
    # confluence_breakout (soft mode)
    'cb_tf4h_fusion':        ('confluence_breakout','tf4h_fusion_score',  'min', 0.2,  0.02,  0.5,  0.02),
    'cb_vol_z':              ('confluence_breakout','volume_zscore',      'min', 0.5,  0.1,   2.5,  0.1),
    # volume_fade_chop (soft mode)
    'vfc_effort_ratio':      ('volume_fade_chop',  'effort_result_ratio','max', 1.5,  0.3,   3.0,  0.1),
    # liquidity_compression (hard mode, rewritten to volume exhaustion)
    'lc_vol_z':              ('liquidity_compression','volume_zscore',   'min', 2.0,  0.5,   4.0,  0.25),
}

GROUPS = {
    'A': {
        'name': 'active_production',
        'params': GROUP_A_PARAMS,
    },
    'B': {
        'name': 'secondary_archetypes',
        'params': GROUP_B_PARAMS,
    },
}


def build_gate_overrides(trial, params_def: Dict) -> Dict[str, Dict[str, float]]:
    """Build gate_overrides dict from Optuna trial samples."""
    overrides = {}
    for param_name, (archetype, feature, op, current, lo, hi, step) in params_def.items():
        if isinstance(current, int):
            val = trial.suggest_int(param_name, int(lo), int(hi), step=int(step))
        else:
            val = trial.suggest_float(param_name, lo, hi, step=step)
        if archetype not in overrides:
            overrides[archetype] = {}
        overrides[archetype][feature] = val
    return overrides


def run_backtest(config, features_df, start_date, end_date, initial_cash=100_000.0,
                 commission_rate=0.0002, slippage_bps=3.0):
    """Run backtest and return performance stats."""
    engine = StandaloneBacktestEngine(
        config=config, initial_cash=initial_cash,
        commission_rate=commission_rate, slippage_bps=slippage_bps,
        features_df=features_df,
    )
    engine.run(start_date=start_date, end_date=end_date)
    return engine.get_performance_stats()


class GateObjective:
    def __init__(self, base_config, features_df, params_def,
                 train_start, train_end, min_trades=50, max_dd=15.0):
        self.base_config = base_config
        self.features_df = features_df
        self.params_def = params_def
        self.train_start = train_start
        self.train_end = train_end
        self.min_trades = min_trades
        self.max_dd = max_dd
        self.best_score = -999
        self.best_trial = -1

    def __call__(self, trial):
        t0 = time.time()
        config = copy.deepcopy(self.base_config)

        # Build gate overrides from trial
        gate_overrides = build_gate_overrides(trial, self.params_def)
        config['gate_overrides'] = gate_overrides

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

        # Score: PF with mild DD penalty above 8%
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
    parser = argparse.ArgumentParser(description='Optimize YAML gate thresholds with Optuna')
    parser.add_argument('--group', type=str, required=True, choices=['A', 'B'],
                       help='Gate group: A=active production, B=secondary archetypes')
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--feature-store', default=DEFAULT_FEATURE_STORE)
    parser.add_argument('--train-start', default='2020-01-01')
    parser.add_argument('--train-end', default='2022-12-31')
    parser.add_argument('--test-start', default='2023-01-01')
    parser.add_argument('--test-end', default='2024-12-31')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-trades', type=int, default=50)
    parser.add_argument('--max-dd', type=float, default=15.0)
    args = parser.parse_args()

    group_def = GROUPS[args.group]
    params_def = group_def['params']
    n_params = len(params_def)

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    print("=" * 90)
    print(f"GATE THRESHOLD OPTIMIZATION: Group {args.group} — {group_def['name']} ({n_params} params, {args.trials} trials)")
    print("=" * 90)

    # Print parameter table
    print(f"\n{'Parameter':<25s} {'Archetype':<25s} {'Feature':<30s} {'Op':<5s} {'Current':>8s} {'Range':>15s}")
    print("-" * 110)
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        print(f"{pname:<25s} {arch:<25s} {feat:<30s} {op:<5s} {curr:>8.3f} [{lo:.3f}, {hi:.3f}]")

    print(f"\nTrain: {args.train_start} to {args.train_end}")
    print(f"Test:  {args.test_start} to {args.test_end}")
    print("=" * 90)

    # Load data
    print("\nLoading feature store...")
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        base_config = json.load(f)
    features_df = pd.read_parquet(str(PROJECT_ROOT / args.feature_store))
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index)
    features_df = features_df.sort_index()
    print(f"Loaded {len(features_df):,} bars\n")

    # Baseline
    print("Running baseline (current YAML values)...")
    t0 = time.time()
    baseline = run_backtest(base_config, features_df, args.train_start, args.train_end)
    print(f"Baseline: PF={baseline.get('profit_factor',0):.3f} | trades={baseline.get('total_trades',0)} | "
          f"WR={baseline.get('win_rate',0):.1f}% | MaxDD={abs(baseline.get('max_drawdown',0)):.1f}% | "
          f"Sharpe={baseline.get('sharpe_ratio',0):.2f} | PnL=${baseline.get('total_pnl',0):,.0f} | "
          f"{time.time()-t0:.0f}s\n")

    # Create study
    objective = GateObjective(
        base_config=base_config, features_df=features_df,
        params_def=params_def, train_start=args.train_start,
        train_end=args.train_end, min_trades=args.min_trades,
        max_dd=args.max_dd,
    )

    study = optuna.create_study(
        study_name=f'gates_{args.group}_{group_def["name"]}',
        direction='maximize',
        sampler=TPESampler(seed=args.seed, n_startup_trials=min(15, args.trials // 3)),
    )

    print(f"Starting optimization ({args.trials} trials)...\n")
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)
    total_time = time.time() - t0

    # Results
    best = study.best_trial
    print(f"\n{'=' * 90}")
    print(f"OPTIMIZATION RESULTS: Group {args.group} — {group_def['name']}")
    print(f"{'=' * 90}")
    print(f"Best Trial: #{best.number} | Score: {best.value:.4f}")
    print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min) | {total_time/args.trials:.0f}s/trial")
    print(f"\nTrain: PF={best.user_attrs.get('profit_factor',0):.3f} | "
          f"Trades={best.user_attrs.get('total_trades',0)} | "
          f"WR={best.user_attrs.get('win_rate',0):.1f}% | "
          f"MaxDD={best.user_attrs.get('max_drawdown',0):.1f}% | "
          f"Sharpe={best.user_attrs.get('sharpe_ratio',0):.2f} | "
          f"PnL=${best.user_attrs.get('total_pnl',0):,.0f}")

    print(f"\nOptimized Parameters:")
    print(f"{'Parameter':<25s} {'Optimized':>10s} {'Was':>10s} {'Change':>10s}")
    print("-" * 60)
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        opt_val = best.params.get(pname, curr)
        if curr != 0:
            change = ((opt_val - curr) / abs(curr)) * 100
            print(f"{pname:<25s} {opt_val:>10.3f} {curr:>10.3f} {change:>+9.1f}%")
        else:
            print(f"{pname:<25s} {opt_val:>10.3f} {curr:>10.3f}       N/A")

    # OOS Validation
    print(f"\nRunning OOS validation ({args.test_start} to {args.test_end})...")
    oos_config = copy.deepcopy(base_config)
    oos_overrides = build_gate_overrides_from_dict(best.params, params_def)
    oos_config['gate_overrides'] = oos_overrides

    oos = run_backtest(oos_config, features_df, args.test_start, args.test_end)
    oos_pf = oos.get('profit_factor', 0)
    train_pf = best.user_attrs.get('profit_factor', 1)
    wfe = (oos_pf / train_pf * 100) if train_pf > 0 else 0

    print(f"OOS:   PF={oos_pf:.3f} | Trades={oos.get('total_trades',0)} | "
          f"WR={oos.get('win_rate',0):.1f}% | MaxDD={abs(oos.get('max_drawdown',0)):.1f}% | "
          f"Sharpe={oos.get('sharpe_ratio',0):.2f} | PnL=${oos.get('total_pnl',0):,.0f}")
    print(f"WFE:   {wfe:.0f}% {'PASS' if wfe >= 70 else 'WARN' if wfe >= 50 else 'FAIL'}")

    # Full period
    print(f"\nRunning full period (2020-2024)...")
    full = run_backtest(oos_config, features_df, '2020-01-01', '2024-12-31')
    print(f"Full:  PF={full.get('profit_factor',0):.3f} | Trades={full.get('total_trades',0)} | "
          f"WR={full.get('win_rate',0):.1f}% | MaxDD={abs(full.get('max_drawdown',0)):.1f}% | "
          f"Sharpe={full.get('sharpe_ratio',0):.2f} | PnL=${full.get('total_pnl',0):,.0f}")

    # Compare vs baseline full period
    print(f"\nBaseline full period (2020-2024)...")
    baseline_full = run_backtest(base_config, features_df, '2020-01-01', '2024-12-31')
    print(f"Base:  PF={baseline_full.get('profit_factor',0):.3f} | Trades={baseline_full.get('total_trades',0)} | "
          f"WR={baseline_full.get('win_rate',0):.1f}% | MaxDD={abs(baseline_full.get('max_drawdown',0)):.1f}% | "
          f"Sharpe={baseline_full.get('sharpe_ratio',0):.2f} | PnL=${baseline_full.get('total_pnl',0):,.0f}")

    # Improvement
    base_pf = baseline_full.get('profit_factor', 1)
    full_pf = full.get('profit_factor', 1)
    if base_pf > 0:
        pf_change = ((full_pf - base_pf) / base_pf) * 100
        print(f"\n{'IMPROVEMENT' if pf_change > 0 else 'REGRESSION'}: PF {base_pf:.3f} → {full_pf:.3f} ({pf_change:+.1f}%)")

    # Save results
    output_dir = PROJECT_ROOT / f'results/optuna_gates_{args.group}_{group_def["name"]}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build YAML update instructions
    yaml_updates = {}
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        opt_val = best.params.get(pname, curr)
        if arch not in yaml_updates:
            yaml_updates[arch] = []
        yaml_updates[arch].append({
            'feature': feat,
            'new_value': opt_val,
            'old_value': curr,
        })

    results = {
        'group': args.group,
        'name': group_def['name'],
        'best_trial': best.number,
        'best_score': best.value,
        'best_params': best.params,
        'train_metrics': dict(best.user_attrs),
        'oos_metrics': {k: oos.get(k, 0) for k in ['profit_factor', 'total_trades', 'max_drawdown', 'sharpe_ratio', 'total_pnl', 'win_rate']},
        'full_metrics': {k: full.get(k, 0) for k in ['profit_factor', 'total_trades', 'max_drawdown', 'sharpe_ratio', 'total_pnl', 'win_rate']},
        'baseline_full_metrics': {k: baseline_full.get(k, 0) for k in ['profit_factor', 'total_trades', 'max_drawdown', 'sharpe_ratio', 'total_pnl', 'win_rate']},
        'wfe': wfe,
        'total_time_s': total_time,
        'trials': args.trials,
        'yaml_updates': yaml_updates,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Top 5
    top = sorted([t for t in study.trials if t.value and t.value > -1e8],
                 key=lambda t: t.value, reverse=True)[:5]
    print(f"\nTop 5 Trials:")
    for t in top:
        print(f"  #{t.number:3d} score={t.value:.3f} PF={t.user_attrs.get('profit_factor',0):.3f} "
              f"trades={t.user_attrs.get('total_trades',0)}")

    # Parameter importance
    try:
        imp = optuna.importance.get_param_importances(study)
        print(f"\nParameter Importance:")
        for p, v in imp.items():
            print(f"  {p:<25s} {v*100:>5.1f}%")
    except Exception:
        pass

    # Print YAML update commands
    print(f"\n{'=' * 90}")
    print("YAML UPDATE INSTRUCTIONS:")
    print(f"{'=' * 90}")
    for arch, updates in yaml_updates.items():
        print(f"\n{arch}.yaml:")
        for u in updates:
            print(f"  {u['feature']}: value {u['old_value']} → {u['new_value']:.4f}")

    print(f"\nResults saved to: {output_dir}/")
    print("=" * 90)


def build_gate_overrides_from_dict(params_dict: Dict, params_def: Dict) -> Dict[str, Dict[str, float]]:
    """Build gate_overrides from a flat params dict (from best trial)."""
    overrides = {}
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        val = params_dict.get(pname, curr)
        if arch not in overrides:
            overrides[arch] = {}
        overrides[arch][feat] = val
    return overrides


if __name__ == '__main__':
    main()
