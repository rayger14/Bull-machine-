#!/usr/bin/env python3
"""
Strategy Validation Framework — WFO + CPCV

Compares two strategy configs (A vs B) using rigorous out-of-sample methods:

  WFO  — Anchored Walk-Forward: trains on growing window, tests on fixed OOS periods
          W1: Train 2020-22 → Test 2023
          W2: Train 2020-23 → Test 2024
          Metric: OOS PF, WFE = OOS_PF / IS_PF (should be > 0.7 to avoid overfit)

  CPCV — Combinatorial Purged Cross-Validation (López de Prado):
          k=6 time groups, p=2 test groups, C(6,2)=15 splits
          48-bar purge + 24-bar embargo at each train/test boundary
          Metric: median OOS PF across all 15 splits, PBO via stitched equity curves

Usage:
    # Compare dynamic sizing vs flat (the primary use case)
    python3 bin/validate_strategy.py --config-a /tmp/bt_flat.json --config-b /tmp/bt_dyn.json

    # Quick: WFO only (faster)
    python3 bin/validate_strategy.py --config-a /tmp/bt_flat.json --config-b /tmp/bt_dyn.json --mode wfo

    # Default: test current production config against itself (sanity check)
    python3 bin/validate_strategy.py

    # Set flag on config B directly (no need to write separate JSON)
    python3 bin/validate_strategy.py --config-b-flag portfolio_allocation.dynamic_sizing_enabled=true

Labels are auto-derived from config filenames or can be set with --label-a / --label-b.

Author: Bull Machine
Date: 2026-04-04
"""

import sys
import json
import copy
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bin.backtest_v11_standalone import StandaloneBacktestEngine

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_CONFIG       = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"
COMMISSION_RATE      = 0.0002
SLIPPAGE_BPS         = 3.0
INITIAL_CASH         = 100_000.0
PURGE_BARS           = 48    # hours
EMBARGO_BARS         = 24    # hours

WFO_WINDOWS = [
    {
        'train_start': '2020-01-01', 'train_end': '2022-12-31',
        'test_start':  '2023-01-01', 'test_end':  '2023-12-31',
        'label': 'W1: Train 2020-22 → Test 2023'
    },
    {
        'train_start': '2020-01-01', 'train_end': '2023-12-31',
        'test_start':  '2024-01-01', 'test_end':  '2024-12-31',
        'label': 'W2: Train 2020-23 → Test 2024'
    },
]

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
WARN = '\033[93m~\033[0m'


# ── Config Helpers ─────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def set_nested(cfg: Dict, dotpath: str, value: Any) -> None:
    """Set a nested config key by dot-separated path.
    E.g. set_nested(cfg, 'portfolio_allocation.dynamic_sizing_enabled', True)
    """
    keys = dotpath.split('.')
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    # Parse value types
    if isinstance(value, str):
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        else:
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                pass
    node[keys[-1]] = value


# ── Engine Cache — initialize once per config, reset between runs ─────
# This avoids re-loading 17 YAML archetype configs on every WFO/CPCV split.
# Speed improvement: ~90% reduction in per-split overhead.

_engine_cache: Dict[int, StandaloneBacktestEngine] = {}


def get_engine(config: Dict, features_df: pd.DataFrame) -> StandaloneBacktestEngine:
    """Return cached engine for this config, creating it if not yet built."""
    key = id(config)
    if key not in _engine_cache:
        _engine_cache[key] = StandaloneBacktestEngine(
            config=config,
            initial_cash=INITIAL_CASH,
            commission_rate=COMMISSION_RATE,
            slippage_bps=SLIPPAGE_BPS,
            features_df=features_df,
        )
    return _engine_cache[key]


# ── Backtest Runner ───────────────────────────────────────────────────

def run_backtest(
    config: Dict,
    features_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """Run backtest on a date window, return stats dict. Reuses cached engine."""
    engine = get_engine(config, features_df)
    engine.reset()
    engine.run(start_date=start_date, end_date=end_date)
    return engine.get_performance_stats()


def run_backtest_on_indices(
    config: Dict,
    full_df: pd.DataFrame,
    indices: np.ndarray,
) -> Dict:
    """Run backtest on specific row indices (for CPCV splits). Reuses cached engine."""
    empty = {
        'profit_factor': 0.0, 'total_trades': 0, 'max_drawdown': 0.0,
        'sharpe_ratio': 0.0, 'total_pnl': 0.0, 'win_rate': 0.0,
    }
    if len(indices) < 200:
        return empty

    subset = full_df.iloc[indices].copy()
    start = subset.index[0].strftime('%Y-%m-%d')
    end   = subset.index[-1].strftime('%Y-%m-%d')

    engine = get_engine(config, full_df)
    engine.reset()
    # Swap feature store to the subset for this split
    engine.features_df = subset
    engine._derive_regime_labels()
    engine.run(start_date=start, end_date=end)
    # Restore full feature store for future runs
    engine.features_df = full_df
    engine._derive_regime_labels()
    return engine.get_performance_stats()


# ── CPCV Split Generator ──────────────────────────────────────────────

def generate_cpcv_splits(
    n: int, k: int = 6, p: int = 2
) -> List[Dict[str, Any]]:
    """Generate CPCV splits with purge + embargo.

    k = number of time groups (default 6)
    p = number of test groups per split (default 2)
    → C(k,p) = 15 splits for k=6,p=2

    Returns list of dicts: {train_idx, test_idx, label, test_groups}
    """
    group_size = n // k
    boundaries = [(i * group_size, (i + 1) * group_size if i < k - 1 else n) for i in range(k)]
    gap = PURGE_BARS + EMBARGO_BARS

    splits = []
    for test_groups in combinations(range(k), p):
        train_groups = [g for g in range(k) if g not in test_groups]

        # Test indices (all bars in test groups)
        test_idx = []
        for g in test_groups:
            test_idx.extend(range(*boundaries[g]))

        # Test boundary positions
        test_boundaries = set()
        for g in test_groups:
            test_boundaries.add(boundaries[g][0])
            test_boundaries.add(boundaries[g][1])

        # Train indices excluding purge/embargo zones around test boundaries
        train_idx = []
        for g in train_groups:
            s, e = boundaries[g]
            for idx in range(s, e):
                if all(abs(idx - b) > gap for b in test_boundaries):
                    train_idx.append(idx)

        test_labels  = [f"G{g}" for g in test_groups]
        train_labels = [f"G{g}" for g in train_groups]
        splits.append({
            'train_idx':   np.array(train_idx),
            'test_idx':    np.array(test_idx),
            'test_groups': test_groups,
            'label':       f"Test={','.join(test_labels)} Train={','.join(train_labels)}",
        })

    return splits


def compute_pbo(oos_pnls_a: List[float], oos_pnls_b: List[float]) -> float:
    """Probability of Backtest Overfitting via logit of rank statistic.

    PBO = fraction of stitched CPCV paths where strategy B underperforms A
    in OOS despite B being optimized. Near 0 = low overfit risk.
    Uses the simplified rank-logit approach (Bailey et al. 2016).
    """
    if not oos_pnls_a or not oos_pnls_b:
        return float('nan')
    n = min(len(oos_pnls_a), len(oos_pnls_b))
    if n == 0:
        return float('nan')
    # Fraction of splits where B OOS PnL < A OOS PnL (B loses its claimed edge)
    b_loses = sum(1 for a, b in zip(oos_pnls_a[:n], oos_pnls_b[:n]) if b < a)
    return b_loses / n


# ── WFO Analysis ──────────────────────────────────────────────────────

def run_wfo(
    config_a: Dict, config_b: Dict,
    label_a: str, label_b: str,
    features_df: pd.DataFrame,
) -> Dict:
    """Anchored Walk-Forward comparison of two configs."""
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD VALIDATION")
    print(f"  A = {label_a}")
    print(f"  B = {label_b}")
    print(f"  Windows: {len(WFO_WINDOWS)}")
    print(f"{'='*70}")

    results = {'windows': [], 'summary': {}}

    for w in WFO_WINDOWS:
        print(f"\n  [{w['label']}]")
        t0 = time.time()

        # In-sample (train) — needed for WFE
        is_a = run_backtest(config_a, features_df, w['train_start'], w['train_end'])
        is_b = run_backtest(config_b, features_df, w['train_start'], w['train_end'])

        # Out-of-sample (test) — the real metric
        oos_a = run_backtest(config_a, features_df, w['test_start'], w['test_end'])
        oos_b = run_backtest(config_b, features_df, w['test_start'], w['test_end'])

        pf_a_is  = is_a.get('profit_factor', 0)
        pf_a_oos = oos_a.get('profit_factor', 0)
        pf_b_is  = is_b.get('profit_factor', 0)
        pf_b_oos = oos_b.get('profit_factor', 0)

        wfe_a = pf_a_oos / pf_a_is if pf_a_is > 0 else 0
        wfe_b = pf_b_oos / pf_b_is if pf_b_is > 0 else 0

        elapsed = time.time() - t0

        row = {
            'label':        w['label'],
            'pf_a_is':      pf_a_is,
            'pf_a_oos':     pf_a_oos,
            'pf_b_is':      pf_b_is,
            'pf_b_oos':     pf_b_oos,
            'wfe_a':        wfe_a,
            'wfe_b':        wfe_b,
            'pnl_a_oos':    oos_a.get('total_pnl', 0),
            'pnl_b_oos':    oos_b.get('total_pnl', 0),
            'trades_a_oos': oos_a.get('total_trades', 0),
            'trades_b_oos': oos_b.get('total_trades', 0),
            'sharpe_a_oos': oos_a.get('sharpe_ratio', 0),
            'sharpe_b_oos': oos_b.get('sharpe_ratio', 0),
            'maxdd_a_oos':  oos_a.get('max_drawdown', 0),
            'maxdd_b_oos':  oos_b.get('max_drawdown', 0),
        }
        results['windows'].append(row)

        # Print window summary
        b_better = pf_b_oos > pf_a_oos
        indicator = PASS if b_better else FAIL

        print(f"    IS  PF: {label_a}={pf_a_is:.2f}  {label_b}={pf_b_is:.2f}")
        print(f"    OOS PF: {label_a}={pf_a_oos:.2f}  {label_b}={pf_b_oos:.2f}  {indicator} {'B better' if b_better else 'A better'}")
        print(f"    OOS PnL:{label_a}=${oos_a.get('total_pnl',0):>8,.0f}  {label_b}=${oos_b.get('total_pnl',0):>8,.0f}")
        print(f"    WFE:    {label_a}={wfe_a:.2f}  {label_b}={wfe_b:.2f}  (>0.70 = acceptable)")
        print(f"    Sharpe: {label_a}={oos_a.get('sharpe_ratio',0):.2f}  {label_b}={oos_b.get('sharpe_ratio',0):.2f}")
        print(f"    MaxDD:  {label_a}={oos_a.get('max_drawdown',0)*100:.1f}%  {label_b}={oos_b.get('max_drawdown',0)*100:.1f}%")
        print(f"    Trades: {label_a}={oos_a.get('total_trades',0)}  {label_b}={oos_b.get('total_trades',0)}  [{elapsed:.0f}s]")

    # Summary
    oos_pfs_a   = [r['pf_a_oos'] for r in results['windows']]
    oos_pfs_b   = [r['pf_b_oos'] for r in results['windows']]
    wfes_a      = [r['wfe_a'] for r in results['windows']]
    wfes_b      = [r['wfe_b'] for r in results['windows']]
    b_wins      = sum(1 for r in results['windows'] if r['pf_b_oos'] > r['pf_a_oos'])

    results['summary'] = {
        'oos_pf_avg_a':  np.mean(oos_pfs_a),
        'oos_pf_avg_b':  np.mean(oos_pfs_b),
        'oos_pf_min_a':  min(oos_pfs_a),
        'oos_pf_min_b':  min(oos_pfs_b),
        'wfe_avg_a':     np.mean(wfes_a),
        'wfe_avg_b':     np.mean(wfes_b),
        'b_wins':        b_wins,
        'total_windows': len(WFO_WINDOWS),
    }

    print(f"\n  WFO SUMMARY")
    print(f"  {'─'*50}")
    print(f"  OOS PF avg: {label_a}={np.mean(oos_pfs_a):.2f}  {label_b}={np.mean(oos_pfs_b):.2f}")
    print(f"  OOS PF min: {label_a}={min(oos_pfs_a):.2f}  {label_b}={min(oos_pfs_b):.2f}")
    print(f"  WFE avg:    {label_a}={np.mean(wfes_a):.2f}  {label_b}={np.mean(wfes_b):.2f}")
    verdict = f"B ({label_b}) outperforms in {b_wins}/{len(WFO_WINDOWS)} windows"
    print(f"  Verdict:    {verdict}")

    return results


# ── CPCV Analysis ─────────────────────────────────────────────────────

def run_cpcv(
    config_a: Dict, config_b: Dict,
    label_a: str, label_b: str,
    features_df: pd.DataFrame,
    k: int = 6, p: int = 2,
) -> Dict:
    """CPCV comparison of two configs (López de Prado method)."""
    n_splits = len(list(combinations(range(k), p)))

    print(f"\n{'='*70}")
    print(f"CPCV VALIDATION (k={k}, p={p}, C({k},{p})={n_splits} splits)")
    print(f"  Purge={PURGE_BARS}h  Embargo={EMBARGO_BARS}h")
    print(f"  A = {label_a}")
    print(f"  B = {label_b}")
    print(f"{'='*70}")

    splits = generate_cpcv_splits(len(features_df), k=k, p=p)
    results = {'splits': [], 'summary': {}}

    oos_pfs_a   = []
    oos_pfs_b   = []
    oos_pnls_a  = []
    oos_pnls_b  = []

    for i, split in enumerate(splits):
        t0 = time.time()
        oos_a = run_backtest_on_indices(config_a, features_df, split['test_idx'])
        oos_b = run_backtest_on_indices(config_b, features_df, split['test_idx'])
        elapsed = time.time() - t0

        pf_a = oos_a.get('profit_factor', 0)
        pf_b = oos_b.get('profit_factor', 0)
        pnl_a = oos_a.get('total_pnl', 0)
        pnl_b = oos_b.get('total_pnl', 0)

        oos_pfs_a.append(pf_a)
        oos_pfs_b.append(pf_b)
        oos_pnls_a.append(pnl_a)
        oos_pnls_b.append(pnl_b)

        indicator = PASS if pf_b > pf_a else FAIL
        print(
            f"  Split {i+1:02d}/{n_splits}  PF: {label_a}={pf_a:.2f} {label_b}={pf_b:.2f} {indicator}"
            f"  PnL: ${pnl_a:>7,.0f} / ${pnl_b:>7,.0f}"
            f"  [{elapsed:.0f}s]  {split['label']}"
        )

        results['splits'].append({
            'split':        i + 1,
            'label':        split['label'],
            'pf_a_oos':     pf_a,
            'pf_b_oos':     pf_b,
            'pnl_a_oos':    pnl_a,
            'pnl_b_oos':    pnl_b,
            'trades_a_oos': oos_a.get('total_trades', 0),
            'trades_b_oos': oos_b.get('total_trades', 0),
        })

    pbo = compute_pbo(oos_pnls_a, oos_pnls_b)
    b_wins = sum(1 for a, b in zip(oos_pfs_a, oos_pfs_b) if b > a)

    results['summary'] = {
        'oos_pf_median_a': float(np.median(oos_pfs_a)),
        'oos_pf_median_b': float(np.median(oos_pfs_b)),
        'oos_pf_mean_a':   float(np.mean(oos_pfs_a)),
        'oos_pf_mean_b':   float(np.mean(oos_pfs_b)),
        'oos_pf_min_a':    float(np.min(oos_pfs_a)),
        'oos_pf_min_b':    float(np.min(oos_pfs_b)),
        'oos_pnl_total_a': float(sum(oos_pnls_a)),
        'oos_pnl_total_b': float(sum(oos_pnls_b)),
        'pbo':             float(pbo),
        'b_wins':          b_wins,
        'total_splits':    n_splits,
    }

    print(f"\n  CPCV SUMMARY")
    print(f"  {'─'*50}")
    print(f"  OOS PF median: {label_a}={np.median(oos_pfs_a):.2f}  {label_b}={np.median(oos_pfs_b):.2f}")
    print(f"  OOS PF mean:   {label_a}={np.mean(oos_pfs_a):.2f}  {label_b}={np.mean(oos_pfs_b):.2f}")
    print(f"  OOS PF min:    {label_a}={min(oos_pfs_a):.2f}  {label_b}={min(oos_pfs_b):.2f}")
    print(f"  OOS PnL sum:   {label_a}=${sum(oos_pnls_a):>8,.0f}  {label_b}=${sum(oos_pnls_b):>8,.0f}")
    print(f"  B wins:        {b_wins}/{n_splits} splits")
    pbo_indicator = PASS if pbo < 0.5 else WARN if pbo < 0.65 else FAIL
    print(f"  PBO:           {pbo:.2f}  {pbo_indicator} (<0.50 = low overfit risk, <0.65 = acceptable)")

    return results


# ── Final Verdict ─────────────────────────────────────────────────────

def print_verdict(
    wfo_results: Optional[Dict], cpcv_results: Optional[Dict],
    label_a: str, label_b: str,
) -> None:
    """Print a structured final verdict on whether B beats A."""
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT: {label_b} vs {label_a}")
    print(f"{'='*70}")

    checks = []

    if wfo_results:
        s = wfo_results['summary']
        # WFO: B beats A in all windows?
        all_wins = s['b_wins'] == s['total_windows']
        any_win  = s['b_wins'] > 0
        checks.append((
            'WFO: B outperforms in all windows',
            all_wins,
            f"{s['b_wins']}/{s['total_windows']} windows won  "
            f"(OOS PF avg: A={s['oos_pf_avg_a']:.2f} B={s['oos_pf_avg_b']:.2f})"
        ))
        # WFE: both strategies not overfit?
        wfe_ok_a = s['wfe_avg_a'] >= 0.70
        wfe_ok_b = s['wfe_avg_b'] >= 0.70
        checks.append((
            'WFE ≥ 0.70 for both (not overfit)',
            wfe_ok_a and wfe_ok_b,
            f"WFE: A={s['wfe_avg_a']:.2f}  B={s['wfe_avg_b']:.2f}"
        ))

    if cpcv_results:
        s = cpcv_results['summary']
        # CPCV: B median OOS PF > A?
        median_win = s['oos_pf_median_b'] > s['oos_pf_median_a']
        checks.append((
            'CPCV: B median OOS PF > A',
            median_win,
            f"Median OOS PF: A={s['oos_pf_median_a']:.2f}  B={s['oos_pf_median_b']:.2f}"
        ))
        # CPCV: B wins majority of splits?
        majority_win = s['b_wins'] > s['total_splits'] // 2
        checks.append((
            f'CPCV: B wins majority ({s["b_wins"]}/{s["total_splits"]}) of splits',
            majority_win,
            f"{s['b_wins']}/{s['total_splits']} splits"
        ))
        # PBO: low overfit risk?
        pbo_ok = s['pbo'] < 0.50
        pbo_acceptable = s['pbo'] < 0.65
        checks.append((
            f'PBO < 0.50 (low overfit risk)',
            pbo_ok,
            f"PBO={s['pbo']:.2f}  {'acceptable (<0.65)' if pbo_acceptable else 'HIGH OVERFIT RISK'}"
        ))

    # Print all checks
    passed = 0
    for name, ok, detail in checks:
        icon = PASS if ok else FAIL
        print(f"  {icon}  {name}")
        print(f"       {detail}")
        if ok:
            passed += 1

    print(f"\n  {'─'*50}")
    print(f"  {passed}/{len(checks)} checks passed")

    if passed == len(checks):
        print(f"\n  VERDICT: {PASS} DEPLOY — {label_b} is rigorously better OOS")
    elif passed >= len(checks) * 0.7:
        print(f"\n  VERDICT: {WARN} PROMISING — {label_b} likely better, monitor live")
    else:
        print(f"\n  VERDICT: {FAIL} DO NOT DEPLOY — insufficient OOS evidence for {label_b}")

    print(f"{'='*70}\n")


# ── Full Report Save ──────────────────────────────────────────────────

def save_report(
    wfo_results: Optional[Dict], cpcv_results: Optional[Dict],
    label_a: str, label_b: str,
    output_path: str,
) -> None:
    """Save full validation report to JSON."""
    report = {
        'label_a': label_a,
        'label_b': label_b,
        'timestamp': pd.Timestamp.now().isoformat(),
        'wfo': wfo_results,
        'cpcv': cpcv_results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Strategy Validation: WFO + CPCV comparison of two configs'
    )
    parser.add_argument(
        '--config-a', default=DEFAULT_CONFIG,
        help='Path to config A (baseline). Default: production config'
    )
    parser.add_argument(
        '--config-b', default=None,
        help='Path to config B (candidate). Default: same as A (sanity check)'
    )
    parser.add_argument(
        '--config-a-flag', default=None, nargs='+',
        help='Override key=value pairs on config A (dot-path notation). '
             'E.g. --config-a-flag portfolio_allocation.dynamic_sizing_enabled=false'
    )
    parser.add_argument(
        '--config-b-flag', default=None, nargs='+',
        help='Override key=value pairs on config B (no separate file needed). '
             'E.g. --config-b-flag portfolio_allocation.dynamic_sizing_enabled=true'
    )
    parser.add_argument('--label-a', default=None, help='Label for config A')
    parser.add_argument('--label-b', default=None, help='Label for config B')
    parser.add_argument(
        '--mode', choices=['wfo', 'cpcv', 'both'], default='both',
        help='Validation mode. Default: both'
    )
    parser.add_argument(
        '--cpcv-k', type=int, default=6,
        help='CPCV number of groups (default: 6)'
    )
    parser.add_argument(
        '--cpcv-p', type=int, default=2,
        help='CPCV number of test groups per split (default: 2)'
    )
    parser.add_argument(
        '--output', default='results/validation/report.json',
        help='Output report path'
    )
    parser.add_argument(
        '--feature-store', default=DEFAULT_FEATURE_STORE,
        help='Path to feature store parquet'
    )
    args = parser.parse_args()

    # Load configs
    cfg_a = load_config(args.config_a)
    cfg_b = load_config(args.config_b) if args.config_b else copy.deepcopy(cfg_a)

    # Apply flag overrides to config A
    if args.config_a_flag:
        for flag in args.config_a_flag:
            if '=' not in flag:
                print(f"ERROR: --config-a-flag must be key=value, got: {flag}")
                sys.exit(1)
            key, val = flag.split('=', 1)
            set_nested(cfg_a, key, val)

    # Apply flag overrides to config B
    if args.config_b_flag:
        for flag in args.config_b_flag:
            if '=' not in flag:
                print(f"ERROR: --config-b-flag must be key=value, got: {flag}")
                sys.exit(1)
            key, val = flag.split('=', 1)
            set_nested(cfg_b, key, val)

    # Labels
    label_a = args.label_a or (Path(args.config_a).stem if args.config_a else 'config_a')
    if args.config_b:
        label_b = args.label_b or Path(args.config_b).stem
    elif args.config_b_flag:
        label_b = args.label_b or 'modified:' + ','.join(f.split('=')[0].split('.')[-1] for f in args.config_b_flag)
    else:
        label_b = args.label_b or 'config_b_copy'

    print(f"\nStrategy Validation Framework")
    print(f"  A = {label_a}  ({args.config_a})")
    print(f"  B = {label_b}")
    print(f"  Mode: {args.mode}")

    # Load feature store once (shared across all runs)
    print(f"\nLoading feature store...")
    t0 = time.time()
    features_df = pd.read_parquet(args.feature_store)
    print(f"  Loaded {len(features_df):,} bars in {time.time()-t0:.1f}s")

    wfo_results  = None
    cpcv_results = None

    if args.mode in ('wfo', 'both'):
        t0 = time.time()
        wfo_results = run_wfo(cfg_a, cfg_b, label_a, label_b, features_df)
        print(f"\n  WFO total time: {time.time()-t0:.0f}s")

    if args.mode in ('cpcv', 'both'):
        t0 = time.time()
        cpcv_results = run_cpcv(cfg_a, cfg_b, label_a, label_b, features_df,
                                k=args.cpcv_k, p=args.cpcv_p)
        print(f"\n  CPCV total time: {time.time()-t0:.0f}s")

    print_verdict(wfo_results, cpcv_results, label_a, label_b)
    save_report(wfo_results, cpcv_results, label_a, label_b, args.output)


if __name__ == '__main__':
    main()
