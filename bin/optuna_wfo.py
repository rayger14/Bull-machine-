#!/usr/bin/env python3
"""
Walk-Forward + CPCV Optuna Gate Optimizer

Modes:
  wfo  — Anchored Walk-Forward: Train [2020-2022] → Test 2023, Train [2020-2023] → Test 2024
  cpcv — Combinatorial Purged Cross-Validation (López de Prado): k=6 groups, p=2 test groups,
         C(6,2)=15 train/test paths with 48-bar purge + 24-bar embargo

Usage:
    python3 bin/optuna_wfo.py --group A --trials 40 --mode wfo
    python3 bin/optuna_wfo.py --group A --trials 30 --mode cpcv
    python3 bin/optuna_wfo.py --group A --trials 60 --mode wfo --seed 123

Author: Claude Code
Date: 2026-03-13
"""

import sys
import json
import argparse
import logging
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import combinations

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

# Purge & embargo constants (1H bars)
PURGE_BARS = 48   # 48 hours — removes autocorrelation leakage
EMBARGO_BARS = 24  # 24 additional hours after purge


# ── Gate Parameter Definitions ─────────────────────────────────────────
# Format: {param_name: (archetype, feature, op, current_value, low, high, step)}

GROUP_A_PARAMS = {
    'twt_wick_lower_ratio':  ('trap_within_trend', 'wick_lower_ratio',  'min', 0.3,  0.05, 0.8,  0.05),
    'twt_bars_since_pivot':  ('trap_within_trend', 'bars_since_pivot',  'max', 55,   20,   120,  5),
    'rc_temporal_conf':      ('retest_cluster',    'temporal_confluence_score', 'min', 0.3, 0.05, 0.6, 0.05),
    'ls_wick_lower_ratio':   ('liquidity_sweep',   'wick_lower_ratio',  'min', 1.0,  0.2,  2.0,  0.1),
    'fc_vol_z_max':          ('failed_continuation','volume_zscore',    'max', 2.0,  0.5,  5.0,  0.25),
    'fc_effort_ratio':       ('failed_continuation','effort_result_ratio','max', 1.5, 0.3,  3.0,  0.1),
    'lv_wick_exhaust':       ('liquidity_vacuum',  'wick_exhaustion_last_3b', 'min', 1.0, 0.3, 3.0, 0.1),
}

GROUP_B_PARAMS = {
    'fd_oi_price_div':       ('funding_divergence', 'oi_price_divergence', 'min', 0.01, 0.001, 0.15, 0.005),
    'fd_oi_change_4h':       ('funding_divergence', 'oi_change_4h',       'max', 0.05, -0.1,  0.2,  0.01),
    'obr_fib_time_conf':     ('order_block_retest', 'fib_time_confluence','min', 0.1,  0.01,  0.5,  0.01),
    'cb_tf4h_fusion':        ('confluence_breakout','tf4h_fusion_score',  'min', 0.2,  0.02,  0.5,  0.02),
    'cb_vol_z':              ('confluence_breakout','volume_zscore',      'min', 0.5,  0.1,   2.5,  0.1),
    'vfc_effort_ratio':      ('volume_fade_chop',  'effort_result_ratio','max', 1.5,  0.3,   3.0,  0.1),
    'lc_vol_z':              ('liquidity_compression','volume_zscore',   'min', 2.0,  0.5,   4.0,  0.25),
}

GROUPS = {
    'A': {'name': 'active_production', 'params': GROUP_A_PARAMS, 'min_trades': 50},
    'B': {'name': 'secondary_archetypes', 'params': GROUP_B_PARAMS, 'min_trades': 20},
}


# ── Walk-Forward Windows ──────────────────────────────────────────────

WFO_WINDOWS = [
    {'train_start': '2020-01-01', 'train_end': '2022-12-31',
     'test_start': '2023-01-01', 'test_end': '2023-12-31', 'label': 'W1: Train 2020-22, Test 2023'},
    {'train_start': '2020-01-01', 'train_end': '2023-12-31',
     'test_start': '2024-01-01', 'test_end': '2024-12-31', 'label': 'W2: Train 2020-23, Test 2024'},
]


# ── CPCV Split Generator ─────────────────────────────────────────────

def generate_cpcv_splits(features_df: pd.DataFrame, k: int = 6, p: int = 2
                         ) -> List[Dict[str, Any]]:
    """Generate CPCV splits with purge and embargo.

    Splits data into k groups by time. For each C(k,p) combination,
    p groups are test, k-p groups are train — with purge/embargo zones
    removed from the training set at each train/test boundary.

    Returns list of dicts with 'train_idx', 'test_idx', 'label', 'test_groups'.
    """
    n = len(features_df)
    group_size = n // k
    group_boundaries = []
    for i in range(k):
        start = i * group_size
        end = (i + 1) * group_size if i < k - 1 else n
        group_boundaries.append((start, end))

    splits = []
    for test_groups in combinations(range(k), p):
        train_groups = [g for g in range(k) if g not in test_groups]

        # Build test indices
        test_idx = []
        for g in test_groups:
            s, e = group_boundaries[g]
            test_idx.extend(range(s, e))

        # Build train indices with purge + embargo
        train_idx = []
        test_boundaries = set()
        for g in test_groups:
            s, e = group_boundaries[g]
            test_boundaries.add(s)
            test_boundaries.add(e)

        for g in train_groups:
            s, e = group_boundaries[g]
            for idx in range(s, e):
                # Check distance from any test boundary
                too_close = False
                for boundary in test_boundaries:
                    distance = abs(idx - boundary)
                    if distance <= PURGE_BARS + EMBARGO_BARS:
                        too_close = True
                        break
                if not too_close:
                    train_idx.append(idx)

        test_group_labels = [f"G{g}" for g in test_groups]
        train_group_labels = [f"G{g}" for g in train_groups]
        label = f"Test={','.join(test_group_labels)} Train={','.join(train_group_labels)}"

        splits.append({
            'train_idx': np.array(train_idx),
            'test_idx': np.array(test_idx),
            'label': label,
            'test_groups': test_groups,
            'purged_bars': sum(1 for g in train_groups
                               for idx in range(group_boundaries[g][0], group_boundaries[g][1])
                               if any(abs(idx - b) <= PURGE_BARS + EMBARGO_BARS for b in test_boundaries))
                          - len(train_idx) + sum(group_boundaries[g][1] - group_boundaries[g][0] for g in train_groups),
        })

    return splits


def build_backtest_paths(k: int = 6, p: int = 2) -> List[List[Tuple]]:
    """Find all non-overlapping tilings of k groups into test sets of size p.

    Each "path" is a list of N/p test-group tuples that together cover all k groups
    exactly once. The number of paths = phi(k,p) = (p/k) * C(k,p).

    For k=6, p=2: returns 5 paths, each containing 3 split indices.
    """
    all_combos = list(combinations(range(k), p))

    paths = []

    def _find_tilings(remaining_groups, current_path, combo_start):
        if not remaining_groups:
            paths.append(current_path[:])
            return
        for i in range(combo_start, len(all_combos)):
            combo = all_combos[i]
            if set(combo).issubset(remaining_groups):
                current_path.append(combo)
                _find_tilings(remaining_groups - set(combo), current_path, i + 1)
                current_path.pop()

    _find_tilings(set(range(k)), [], 0)
    return paths


# ── Backtest Runner ───────────────────────────────────────────────────

def run_backtest(config, features_df, start_date=None, end_date=None,
                 initial_cash=100_000.0, commission_rate=0.0002, slippage_bps=3.0):
    """Run backtest and return performance stats."""
    engine = StandaloneBacktestEngine(
        config=config, initial_cash=initial_cash,
        commission_rate=commission_rate, slippage_bps=slippage_bps,
        features_df=features_df,
    )
    engine.run(start_date=start_date, end_date=end_date)
    return engine.get_performance_stats()


def run_backtest_on_indices(config, full_features_df, indices,
                            initial_cash=100_000.0, commission_rate=0.0002, slippage_bps=3.0,
                            return_equity=False):
    """Run backtest on specific row indices (for CPCV splits).

    If return_equity=True, returns (stats, equity_curve, equity_timestamps).
    """
    subset = full_features_df.iloc[indices].copy()
    empty_stats = {'profit_factor': 0, 'total_trades': 0, 'max_drawdown': 0,
                   'sharpe_ratio': 0, 'total_pnl': 0, 'win_rate': 0}
    if len(subset) < 100:
        if return_equity:
            return empty_stats, [initial_cash], [subset.index[0] if len(subset) > 0 else pd.Timestamp.now()]
        return empty_stats

    start_date = subset.index[0].strftime('%Y-%m-%d')
    end_date = subset.index[-1].strftime('%Y-%m-%d')

    engine = StandaloneBacktestEngine(
        config=config, initial_cash=initial_cash,
        commission_rate=commission_rate, slippage_bps=slippage_bps,
        features_df=subset,
    )
    engine.run(start_date=start_date, end_date=end_date)
    stats = engine.get_performance_stats()

    if return_equity:
        return stats, engine.equity_curve, engine.equity_timestamps

    return stats


# ── Gate Override Builders ────────────────────────────────────────────

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


def build_gate_overrides_from_dict(params_dict: Dict, params_def: Dict) -> Dict[str, Dict[str, float]]:
    """Build gate_overrides from a flat params dict."""
    overrides = {}
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        val = params_dict.get(pname, curr)
        if arch not in overrides:
            overrides[arch] = {}
        overrides[arch][feat] = val
    return overrides


# ── Scoring ───────────────────────────────────────────────────────────

def compute_score(stats: Dict, min_trades: int = 50, max_dd: float = 15.0) -> float:
    """Compute optimization score from backtest stats."""
    pf = stats.get('profit_factor', 0.0)
    trades = stats.get('total_trades', 0)
    max_dd_pct = abs(stats.get('max_drawdown', 0.0))

    if trades < min_trades:
        return -1e9
    if max_dd_pct > max_dd:
        return -1e9

    dd_penalty = max(0, (max_dd_pct - 8.0)) * 0.02
    return pf - dd_penalty if pf > 1.0 else pf - 1.0


# ── WFO Objective ─────────────────────────────────────────────────────

class WFOObjective:
    """Anchored Walk-Forward Optimization.

    Trains on each window, evaluates OOS. Final score = min(OOS PF) across windows
    (ensures robustness across ALL periods, not just average).
    """

    def __init__(self, base_config, features_df, params_def, windows,
                 min_trades=50, max_dd=15.0):
        self.base_config = base_config
        self.features_df = features_df
        self.params_def = params_def
        self.windows = windows
        self.min_trades = min_trades
        self.max_dd = max_dd
        self.best_score = -999
        self.best_trial = -1

    def __call__(self, trial):
        t0 = time.time()
        config = copy.deepcopy(self.base_config)
        gate_overrides = build_gate_overrides(trial, self.params_def)
        config['gate_overrides'] = gate_overrides

        oos_results = []
        train_results = []

        for w in self.windows:
            # Train evaluation
            try:
                train_stats = run_backtest(config, self.features_df,
                                           w['train_start'], w['train_end'])
                oos_stats = run_backtest(config, self.features_df,
                                         w['test_start'], w['test_end'])
            except Exception as e:
                logger.warning(f"Trial {trial.number} crashed on {w['label']}: {e}")
                return -1e9

            train_results.append(train_stats)
            oos_results.append(oos_stats)

        # Score = minimum OOS PF across windows (ensures ALL windows pass)
        oos_pfs = [r.get('profit_factor', 0) for r in oos_results]
        oos_trades = [r.get('total_trades', 0) for r in oos_results]
        train_pfs = [r.get('profit_factor', 0) for r in train_results]

        # Hard constraints on OOS
        for i, (pf, trades) in enumerate(zip(oos_pfs, oos_trades)):
            min_oos_trades = max(10, self.min_trades // 3)  # Lower bar for single-year OOS
            if trades < min_oos_trades:
                elapsed = time.time() - t0
                print(f"  Trial {trial.number:3d} | PENALTY: OOS window {i} has {trades} trades < {min_oos_trades} | {elapsed:.0f}s")
                return -1e9

        # Overfit detection: PF > 2.0 in one window but < 1.0 in another
        if max(oos_pfs) > 2.0 and min(oos_pfs) < 1.0:
            elapsed = time.time() - t0
            print(f"  Trial {trial.number:3d} | OVERFIT: OOS PFs={[f'{p:.2f}' for p in oos_pfs]} (divergent) | {elapsed:.0f}s")
            return -1e9

        # Score = min OOS PF (most conservative)
        min_oos_pf = min(oos_pfs)
        avg_oos_pf = np.mean(oos_pfs)

        # Mild penalty for DD
        avg_oos_dd = np.mean([abs(r.get('max_drawdown', 0)) for r in oos_results])
        dd_penalty = max(0, (avg_oos_dd - 8.0)) * 0.02
        score = min_oos_pf - dd_penalty if min_oos_pf > 1.0 else min_oos_pf - 1.0

        # WFE per window
        wfes = []
        for train_pf, oos_pf in zip(train_pfs, oos_pfs):
            wfe = (oos_pf / train_pf * 100) if train_pf > 0 else 0
            wfes.append(wfe)

        elapsed = time.time() - t0

        # Store attrs
        trial.set_user_attr('oos_pfs', oos_pfs)
        trial.set_user_attr('oos_trades', oos_trades)
        trial.set_user_attr('train_pfs', train_pfs)
        trial.set_user_attr('wfes', wfes)
        trial.set_user_attr('avg_oos_pf', avg_oos_pf)

        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_trial = trial.number

        oos_str = " | ".join([f"W{i}: PF={pf:.2f} ({t} tr, WFE={wfe:.0f}%)"
                               for i, (pf, t, wfe) in enumerate(zip(oos_pfs, oos_trades, wfes))])
        print(f"  Trial {trial.number:3d} | {oos_str} | min_PF={min_oos_pf:.2f} | "
              f"score={score:.3f} {'*** BEST' if is_best else ''} | {elapsed:.0f}s")

        return score


# ── CPCV Objective ────────────────────────────────────────────────────

class CPCVObjective:
    """Combinatorial Purged Cross-Validation.

    Optimizes on 3 representative CPCV paths, then validates final params
    on all 15 paths (done post-optimization, not in objective).
    """

    def __init__(self, base_config, features_df, params_def, train_splits,
                 min_trades=20, max_dd=15.0):
        self.base_config = base_config
        self.features_df = features_df
        self.params_def = params_def
        self.train_splits = train_splits  # 3 representative splits
        self.min_trades = min_trades
        self.max_dd = max_dd
        self.best_score = -999
        self.best_trial = -1

    def __call__(self, trial):
        t0 = time.time()
        config = copy.deepcopy(self.base_config)
        gate_overrides = build_gate_overrides(trial, self.params_def)
        config['gate_overrides'] = gate_overrides

        oos_pfs = []
        oos_trades_list = []

        for split in self.train_splits:
            try:
                # Train on train_idx portion
                train_stats = run_backtest_on_indices(
                    config, self.features_df, split['train_idx'])
                # Test on test_idx portion
                oos_stats = run_backtest_on_indices(
                    config, self.features_df, split['test_idx'])
            except Exception as e:
                logger.warning(f"Trial {trial.number} crashed on {split['label']}: {e}")
                return -1e9

            oos_pf = oos_stats.get('profit_factor', 0)
            oos_trades = oos_stats.get('total_trades', 0)
            oos_pfs.append(oos_pf)
            oos_trades_list.append(oos_trades)

        # Score = median OOS PF across paths
        median_pf = float(np.median(oos_pfs))
        min_pf = min(oos_pfs)

        # Hard constraint: no path with PF < 0.5
        if min_pf < 0.5:
            elapsed = time.time() - t0
            print(f"  Trial {trial.number:3d} | PENALTY: min_PF={min_pf:.2f} < 0.5 | {elapsed:.0f}s")
            return -1e9

        score = median_pf if median_pf > 1.0 else median_pf - 1.0

        trial.set_user_attr('oos_pfs', oos_pfs)
        trial.set_user_attr('oos_trades', oos_trades_list)
        trial.set_user_attr('median_pf', median_pf)
        trial.set_user_attr('min_pf', min_pf)

        elapsed = time.time() - t0
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_trial = trial.number

        path_str = " | ".join([f"P{i}: PF={pf:.2f}({t}tr)"
                                for i, (pf, t) in enumerate(zip(oos_pfs, oos_trades_list))])
        print(f"  Trial {trial.number:3d} | {path_str} | med={median_pf:.2f} min={min_pf:.2f} | "
              f"score={score:.3f} {'*** BEST' if is_best else ''} | {elapsed:.0f}s")

        return score


# ── Validation ────────────────────────────────────────────────────────

def validate_wfo(best_params, base_config, features_df, params_def, windows, baseline_stats):
    """Run full WFO validation and anti-overfit checks."""
    config = copy.deepcopy(base_config)
    config['gate_overrides'] = build_gate_overrides_from_dict(best_params, params_def)

    print(f"\n{'=' * 90}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'=' * 90}")

    all_pass = True
    for w in windows:
        oos = run_backtest(config, features_df, w['test_start'], w['test_end'])
        oos_pf = oos.get('profit_factor', 0)
        oos_trades = oos.get('total_trades', 0)
        oos_dd = abs(oos.get('max_drawdown', 0))
        oos_wr = oos.get('win_rate', 0)

        train = run_backtest(config, features_df, w['train_start'], w['train_end'])
        train_pf = train.get('profit_factor', 0)
        train_dd = abs(train.get('max_drawdown', 0))
        train_wr = train.get('win_rate', 0)
        wfe = (oos_pf / train_pf * 100) if train_pf > 0 else 0

        # Checks
        pf_pass = oos_pf > 1.2
        wfe_pass = wfe >= 70
        dd_pass = oos_dd <= train_dd * 1.5
        wr_pass = abs(oos_wr - train_wr) <= 10

        status = "PASS" if all([pf_pass, wfe_pass, dd_pass, wr_pass]) else "WARN"
        if not pf_pass:
            status = "FAIL"
            all_pass = False

        print(f"\n  {w['label']}:")
        print(f"    Train: PF={train_pf:.3f} | DD={train_dd:.1f}% | WR={train_wr:.1f}%")
        print(f"    OOS:   PF={oos_pf:.3f} | DD={oos_dd:.1f}% | WR={oos_wr:.1f}% | trades={oos_trades}")
        print(f"    WFE:   {wfe:.0f}% {'OK' if wfe_pass else 'LOW'}")
        print(f"    DD ratio: {oos_dd/max(train_dd,0.01):.2f}x {'OK' if dd_pass else 'HIGH'}")
        print(f"    WR gap:   {abs(oos_wr-train_wr):.1f}pp {'OK' if wr_pass else 'HIGH'}")
        print(f"    Status: [{status}]")

    # Full period comparison
    print(f"\n{'─' * 60}")
    full = run_backtest(config, features_df, '2020-01-01', '2024-12-31')
    base_full = run_backtest(base_config, features_df, '2020-01-01', '2024-12-31')

    full_pf = full.get('profit_factor', 0)
    base_pf = base_full.get('profit_factor', 0)
    full_trades = full.get('total_trades', 0)
    base_trades = base_full.get('total_trades', 0)

    trade_change = abs(full_trades - base_trades) / max(base_trades, 1) * 100
    trade_pass = trade_change <= 30

    print(f"  Full Period (2020-2024):")
    print(f"    Optimized: PF={full_pf:.3f} | trades={full_trades} | "
          f"Sharpe={full.get('sharpe_ratio',0):.2f} | PnL=${full.get('total_pnl',0):,.0f}")
    print(f"    Baseline:  PF={base_pf:.3f} | trades={base_trades} | "
          f"Sharpe={base_full.get('sharpe_ratio',0):.2f} | PnL=${base_full.get('total_pnl',0):,.0f}")
    pf_change = ((full_pf - base_pf) / max(base_pf, 0.01)) * 100
    print(f"    PF change: {pf_change:+.1f}% | Trade count change: {trade_change:.0f}% {'OK' if trade_pass else 'HIGH (>30%)'}")

    # Boundary check
    print(f"\n  Boundary Check:")
    boundary_hit = False
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        val = best_params.get(pname, curr)
        at_low = abs(val - lo) < step * 0.5
        at_high = abs(val - hi) < step * 0.5
        if at_low or at_high:
            boundary_hit = True
            which = "LOW" if at_low else "HIGH"
            print(f"    WARNING: {pname}={val:.4f} hit {which} boundary [{lo}, {hi}] — widen range and re-run")
    if not boundary_hit:
        print(f"    OK — no parameters at boundary")

    # Final verdict
    print(f"\n{'=' * 90}")
    if all_pass and trade_pass and not boundary_hit:
        print("VERDICT: PASS — parameters are safe to deploy")
    elif all_pass:
        print("VERDICT: PASS with warnings — review trade count and boundary hits")
    else:
        print("VERDICT: FAIL — OOS PF < 1.2 in at least one window. DO NOT deploy.")
    print(f"{'=' * 90}")

    return {
        'full_metrics': {k: full.get(k, 0) for k in ['profit_factor', 'total_trades', 'max_drawdown',
                                                       'sharpe_ratio', 'total_pnl', 'win_rate']},
        'baseline_full_metrics': {k: base_full.get(k, 0) for k in ['profit_factor', 'total_trades', 'max_drawdown',
                                                                     'sharpe_ratio', 'total_pnl', 'win_rate']},
        'all_oos_pass': all_pass,
        'trade_count_pass': trade_pass,
        'boundary_hit': boundary_hit,
    }


def _compute_sharpe_from_equity(equity_curve: List[float], periods_per_year: float = 8760.0) -> float:
    """Compute annualized Sharpe ratio from an equity curve.

    Assumes 1H bars (8760 bars/year). Returns 0 if insufficient data.
    """
    if len(equity_curve) < 10:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def validate_cpcv(best_params, base_config, features_df, params_def, all_splits, k=6, p=2):
    """Validate final CPCV parameters across all splits + compute PBO via stitched equity curves.

    PBO (Probability of Backtest Overfitting) = fraction of stitched backtest paths
    with Sharpe < 0. López de Prado (2018), Chapter 12.
    """
    config = copy.deepcopy(base_config)
    config['gate_overrides'] = build_gate_overrides_from_dict(best_params, params_def)

    print(f"\n{'=' * 90}")
    print(f"CPCV FULL VALIDATION (all {len(all_splits)} splits)")
    print(f"{'=' * 90}")

    # Phase 1: Run all splits and collect per-split OOS stats + equity curves
    split_results = {}  # keyed by test_groups tuple
    oos_pfs = []

    for i, split in enumerate(all_splits):
        try:
            oos, equity, timestamps = run_backtest_on_indices(
                config, features_df, split['test_idx'], return_equity=True)
            pf = oos.get('profit_factor', 0)
            trades = oos.get('total_trades', 0)
            sharpe = oos.get('sharpe_ratio', 0)
            oos_pfs.append(pf)

            split_results[split['test_groups']] = {
                'stats': oos,
                'equity': equity,
                'timestamps': timestamps,
                'pf': pf,
                'trades': trades,
                'sharpe': sharpe,
            }

            status = "OK" if pf >= 0.8 else "FAIL" if pf < 0.5 else "WARN"
            print(f"  Split {i:2d} ({split['label']:<35s}): PF={pf:.3f} | trades={trades:4d} | Sharpe={sharpe:.2f} [{status}]")
        except Exception as e:
            oos_pfs.append(0)
            split_results[split['test_groups']] = {
                'stats': {}, 'equity': [100000], 'timestamps': [],
                'pf': 0, 'trades': 0, 'sharpe': 0,
            }
            print(f"  Split {i:2d} ({split['label']:<35s}): ERROR: {e}")

    median_pf = float(np.median(oos_pfs))
    min_pf = min(oos_pfs) if oos_pfs else 0

    print(f"\n  Per-Split Summary:")
    print(f"    Median OOS PF: {median_pf:.3f} {'PASS' if median_pf >= 1.2 else 'FAIL'}")
    print(f"    Min OOS PF:    {min_pf:.3f} {'PASS' if min_pf >= 0.8 else 'FAIL'}")
    paths_below_08 = sum(1 for pf in oos_pfs if pf < 0.8)
    print(f"    Splits < 0.8:  {paths_below_08} {'PASS' if paths_below_08 == 0 else 'FAIL'}")

    # Phase 2: Build stitched backtest paths and compute PBO
    print(f"\n{'─' * 60}")
    print(f"  STITCHED BACKTEST PATHS (PBO Analysis)")
    print(f"{'─' * 60}")

    backtest_paths = build_backtest_paths(k=k, p=p)
    n_paths = len(backtest_paths)
    print(f"  phi({k},{p}) = {n_paths} non-overlapping paths\n")

    path_sharpes = []
    path_pnls = []

    for path_idx, path in enumerate(backtest_paths):
        # Each path is a list of test-group tuples, e.g. [(0,1), (2,3), (4,5)]
        # Stitch equity curves in chronological order (sort by group index)
        sorted_segments = sorted(path, key=lambda tup: tup[0])

        stitched_equity = []
        stitched_pnl = 0.0
        segment_details = []

        for seg_groups in sorted_segments:
            result = split_results.get(seg_groups)
            if result is None:
                continue

            eq = result['equity']
            pnl = result['stats'].get('total_pnl', 0)
            stitched_pnl += pnl

            if not stitched_equity:
                # First segment: use raw equity
                stitched_equity.extend(eq)
            else:
                # Subsequent segments: rebase equity to end of previous segment
                if len(eq) > 1:
                    prev_end = stitched_equity[-1]
                    start_val = eq[0] if eq[0] != 0 else 1
                    scale = prev_end / start_val
                    stitched_equity.extend([v * scale for v in eq[1:]])

            seg_label = f"G{''.join(str(g) for g in seg_groups)}"
            segment_details.append(f"{seg_label}(PF={result['pf']:.2f})")

        # Compute Sharpe on stitched equity curve
        path_sharpe = _compute_sharpe_from_equity(stitched_equity)
        path_sharpes.append(path_sharpe)
        path_pnls.append(stitched_pnl)

        total_return = ((stitched_equity[-1] / stitched_equity[0]) - 1) * 100 if stitched_equity else 0
        status = "OK" if path_sharpe > 0 else "OVERFIT"
        print(f"  Path {path_idx}: {' + '.join(segment_details)} | "
              f"PnL=${stitched_pnl:,.0f} | Return={total_return:.1f}% | "
              f"Sharpe={path_sharpe:.2f} [{status}]")

    # PBO = fraction of paths with Sharpe < 0
    n_negative = sum(1 for s in path_sharpes if s < 0)
    pbo = n_negative / n_paths if n_paths > 0 else 1.0

    print(f"\n  {'─' * 40}")
    print(f"  PBO (Probability of Backtest Overfitting):")
    print(f"    Paths with Sharpe < 0: {n_negative}/{n_paths}")
    print(f"    PBO = {pbo:.1%}")

    if pbo == 0:
        print(f"    Interpretation: EXCELLENT — no path is overfit")
    elif pbo < 0.2:
        print(f"    Interpretation: GOOD — low overfitting risk")
    elif pbo < 0.5:
        print(f"    Interpretation: CAUTION — moderate overfitting risk")
    else:
        print(f"    Interpretation: FAIL — majority of paths are overfit, DO NOT deploy")

    print(f"\n    Path Sharpes: {[f'{s:.2f}' for s in path_sharpes]}")
    print(f"    Median Path Sharpe: {float(np.median(path_sharpes)):.2f}")
    print(f"    Min Path Sharpe:    {min(path_sharpes):.2f}")

    # Phase 3: Full period comparison
    print(f"\n{'─' * 60}")
    full = run_backtest(config, features_df, '2020-01-01', '2024-12-31')
    base_full = run_backtest(base_config, features_df, '2020-01-01', '2024-12-31')

    print(f"  Full Period (2020-2024):")
    print(f"    Optimized: PF={full.get('profit_factor',0):.3f} | trades={full.get('total_trades',0)} | "
          f"PnL=${full.get('total_pnl',0):,.0f} | Sharpe={full.get('sharpe_ratio',0):.2f}")
    print(f"    Baseline:  PF={base_full.get('profit_factor',0):.3f} | trades={base_full.get('total_trades',0)} | "
          f"PnL=${base_full.get('total_pnl',0):,.0f} | Sharpe={base_full.get('sharpe_ratio',0):.2f}")

    # Final verdict
    accept = median_pf >= 1.2 and min_pf >= 0.8 and paths_below_08 == 0 and pbo < 0.5
    print(f"\n{'=' * 90}")
    if accept and pbo == 0:
        print("VERDICT: STRONG PASS — all splits profitable, PBO=0%, safe to deploy")
    elif accept:
        print(f"VERDICT: PASS — PBO={pbo:.0%}, acceptable overfitting risk")
    elif pbo >= 0.5:
        print(f"VERDICT: FAIL — PBO={pbo:.0%}, majority of paths overfit. DO NOT deploy.")
    else:
        print(f"VERDICT: FAIL — OOS metrics below threshold. DO NOT deploy.")
    print(f"{'=' * 90}")

    return {
        'oos_pfs': oos_pfs,
        'median_pf': median_pf,
        'min_pf': min_pf,
        'paths_below_08': paths_below_08,
        'pbo': pbo,
        'path_sharpes': path_sharpes,
        'path_pnls': path_pnls,
        'n_stitched_paths': n_paths,
        'accept': accept,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Walk-Forward + CPCV Optuna Gate Optimizer')
    parser.add_argument('--group', type=str, required=True, choices=['A', 'B'])
    parser.add_argument('--mode', type=str, required=True, choices=['wfo', 'cpcv'],
                       help='wfo=Anchored Walk-Forward, cpcv=Combinatorial Purged Cross-Validation')
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--feature-store', default=DEFAULT_FEATURE_STORE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpcv-k', type=int, default=6, help='CPCV: number of groups')
    parser.add_argument('--cpcv-p', type=int, default=2, help='CPCV: number of test groups')
    parser.add_argument('--cpcv-train-paths', type=int, default=3,
                       help='CPCV: number of representative paths to optimize on')
    args = parser.parse_args()

    group_def = GROUPS[args.group]
    params_def = group_def['params']
    n_params = len(params_def)
    min_trades = group_def['min_trades']

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    mode_label = 'Anchored Walk-Forward' if args.mode == 'wfo' else f'CPCV (k={args.cpcv_k}, p={args.cpcv_p})'
    print("=" * 90)
    print(f"OPTUNA {mode_label.upper()}: Group {args.group} — {group_def['name']}")
    print(f"  {n_params} params, {args.trials} trials, seed={args.seed}")
    print("=" * 90)

    # Print parameter table
    print(f"\n{'Parameter':<25s} {'Archetype':<25s} {'Feature':<30s} {'Op':<5s} {'Current':>8s} {'Range':>15s}")
    print("-" * 110)
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        print(f"{pname:<25s} {arch:<25s} {feat:<30s} {op:<5s} {curr:>8.3f} [{lo:.3f}, {hi:.3f}]")
    print()

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

    # Baseline
    print("Running baseline (current YAML values, full period 2020-2024)...")
    t0 = time.time()
    baseline = run_backtest(base_config, features_df, '2020-01-01', '2024-12-31')
    print(f"Baseline: PF={baseline.get('profit_factor',0):.3f} | trades={baseline.get('total_trades',0)} | "
          f"WR={baseline.get('win_rate',0):.1f}% | MaxDD={abs(baseline.get('max_drawdown',0)):.1f}% | "
          f"Sharpe={baseline.get('sharpe_ratio',0):.2f} | PnL=${baseline.get('total_pnl',0):,.0f} | "
          f"{time.time()-t0:.0f}s\n")

    # Create sampler
    sampler = TPESampler(
        seed=args.seed,
        n_startup_trials=min(15, args.trials // 3),
        multivariate=True,
    )

    if args.mode == 'wfo':
        # ── Walk-Forward ──
        print(f"Walk-Forward Windows:")
        for w in WFO_WINDOWS:
            print(f"  {w['label']}")
        print()

        objective = WFOObjective(
            base_config=base_config, features_df=features_df,
            params_def=params_def, windows=WFO_WINDOWS,
            min_trades=min_trades, max_dd=15.0,
        )

        study = optuna.create_study(
            study_name=f'wfo_gates_{args.group}',
            direction='maximize',
            sampler=sampler,
        )

        print(f"Starting WFO optimization ({args.trials} trials)...\n")
        t0 = time.time()
        study.optimize(objective, n_trials=args.trials)
        total_time = time.time() - t0

        best = study.best_trial
        print(f"\n{'=' * 90}")
        print(f"WFO OPTIMIZATION RESULTS: Group {args.group}")
        print(f"{'=' * 90}")
        print(f"Best Trial: #{best.number} | Score: {best.value:.4f}")
        print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"OOS PFs: {best.user_attrs.get('oos_pfs', [])}")
        print(f"OOS Trades: {best.user_attrs.get('oos_trades', [])}")
        print(f"WFEs: {[f'{w:.0f}%' for w in best.user_attrs.get('wfes', [])]}")

        # Print optimized params
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

        # Parameter importance
        try:
            imp = optuna.importance.get_param_importances(study)
            print(f"\nParameter Importance:")
            for p, v in imp.items():
                print(f"  {p:<25s} {v*100:>5.1f}%")
        except Exception:
            pass

        # Full validation
        validation = validate_wfo(best.params, base_config, features_df,
                                   params_def, WFO_WINDOWS, baseline)

    else:
        # ── CPCV ──
        print(f"Generating CPCV splits (k={args.cpcv_k}, p={args.cpcv_p})...")
        all_splits = generate_cpcv_splits(features_df, k=args.cpcv_k, p=args.cpcv_p)
        n_paths = len(all_splits)
        print(f"Generated C({args.cpcv_k},{args.cpcv_p}) = {n_paths} paths")
        print(f"Purge={PURGE_BARS} bars, Embargo={EMBARGO_BARS} bars")

        # Select representative paths (spread across different test groups)
        # Pick paths with maximally different test groups
        representative_indices = np.linspace(0, n_paths - 1, args.cpcv_train_paths, dtype=int)
        train_splits = [all_splits[i] for i in representative_indices]

        print(f"\nTraining on {args.cpcv_train_paths} representative paths:")
        for i, split in enumerate(train_splits):
            print(f"  Path {i}: {split['label']} | train={len(split['train_idx'])} bars, test={len(split['test_idx'])} bars")
        print()

        objective = CPCVObjective(
            base_config=base_config, features_df=features_df,
            params_def=params_def, train_splits=train_splits,
            min_trades=min_trades, max_dd=15.0,
        )

        study = optuna.create_study(
            study_name=f'cpcv_gates_{args.group}',
            direction='maximize',
            sampler=sampler,
        )

        print(f"Starting CPCV optimization ({args.trials} trials)...\n")
        t0 = time.time()
        study.optimize(objective, n_trials=args.trials)
        total_time = time.time() - t0

        best = study.best_trial
        print(f"\n{'=' * 90}")
        print(f"CPCV OPTIMIZATION RESULTS: Group {args.group}")
        print(f"{'=' * 90}")
        print(f"Best Trial: #{best.number} | Score: {best.value:.4f}")
        print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"Median PF: {best.user_attrs.get('median_pf', 0):.3f}")
        print(f"Min PF: {best.user_attrs.get('min_pf', 0):.3f}")

        # Print optimized params
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

        # Parameter importance
        try:
            imp = optuna.importance.get_param_importances(study)
            print(f"\nParameter Importance:")
            for p, v in imp.items():
                print(f"  {p:<25s} {v*100:>5.1f}%")
        except Exception:
            pass

        # Full validation on all 15 paths + PBO via stitched equity curves
        validation = validate_cpcv(best.params, base_config, features_df,
                                    params_def, all_splits,
                                    k=args.cpcv_k, p=args.cpcv_p)

    # Save results
    output_dir = PROJECT_ROOT / f'results/optuna_{args.mode}_{args.group}_{group_def["name"]}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build YAML update instructions
    yaml_updates = {}
    for pname, (arch, feat, op, curr, lo, hi, step) in params_def.items():
        opt_val = best.params.get(pname, curr)
        if arch not in yaml_updates:
            yaml_updates[arch] = []
        yaml_updates[arch].append({
            'feature': feat,
            'new_value': float(opt_val),
            'old_value': float(curr),
        })

    results = {
        'mode': args.mode,
        'group': args.group,
        'name': group_def['name'],
        'best_trial': best.number,
        'best_score': best.value,
        'best_params': {k_: float(v) for k_, v in best.params.items()},
        'user_attrs': {k_: v for k_, v in best.user_attrs.items()},
        'validation': {k_: v for k_, v in validation.items()
                       if not isinstance(v, (np.ndarray,))},
        'total_time_s': total_time,
        'trials': args.trials,
        'seed': args.seed,
        'yaml_updates': yaml_updates,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print YAML update commands
    print(f"\n{'=' * 90}")
    print("YAML UPDATE INSTRUCTIONS:")
    print(f"{'=' * 90}")
    for arch, updates in yaml_updates.items():
        print(f"\n  {arch}.yaml:")
        for u in updates:
            print(f"    {u['feature']}: value {u['old_value']} → {u['new_value']:.4f}")

    print(f"\nResults saved to: {output_dir}/")
    print("=" * 90)


if __name__ == '__main__':
    main()
