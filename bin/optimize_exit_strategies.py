#!/usr/bin/env python3
"""
Exit Strategy Optimizer - Find optimal exit confluence & thresholds

Uses Optuna + Real Backtest Engine to optimize exit strategy parameters across:
- Phase 2: Pattern exit confluence (2leg pullback, inside bar expansion)
- Phase 2: Structure invalidation gates (min hold, RSI thresholds, MTF alignment)
- Phase 3: Regime-aware trailing stop multipliers
- Phase 4: Re-entry confluence (RSI, 4H fusion, volume)

Objective: Maximize Total_PNL × sqrt(Win_Rate) × sqrt(Profit_Factor)

Search Space (All Phases):
- pattern_confluence_threshold: [1, 2, 3] (1/3, 2/3, or 3/3 required)
- structure_min_hold_bars: [8, 12, 16, 20]
- structure_rsi_long_threshold: [20, 25, 30] (lower = stricter)
- structure_rsi_short_threshold: [70, 75, 80] (higher = stricter)
- structure_vol_zscore_min: [0.5, 1.0, 1.5, 2.0]
- trailing_stop_base_mult: [1.5, 2.0, 2.5]
- trailing_stop_trending_mult: [2.0, 2.5, 3.0]
- reentry_confluence_threshold: [0, 2, 3] (0=disabled, 2=2/3, 3=3/3)
- reentry_window_btc_eth: [5, 7, 10] bars
- reentry_fusion_delta: [0.03, 0.05, 0.10] (how much below threshold to re-enter)

Output: Top 10 exit configs per asset → reports/exit_optimization/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import json
import optuna
from optuna.samplers import TPESampler
import argparse
from datetime import datetime
import re


def run_backtest_with_params(asset: str, start_date: str, end_date: str, params: dict) -> dict:
    """
    Run backtest_knowledge_v2.py with custom exit parameters via environment variables.

    Returns dict with: {pnl, trades, win_rate, profit_factor, max_drawdown}
    """
    # Build environment variables for exit parameters
    env_vars = {
        'EXIT_PATTERN_CONFLUENCE': str(params['pattern_confluence_threshold']),
        'EXIT_STRUCT_MIN_HOLD': str(params['structure_min_hold_bars']),
        'EXIT_STRUCT_RSI_LONG': str(params['structure_rsi_long_threshold']),
        'EXIT_STRUCT_RSI_SHORT': str(params['structure_rsi_short_threshold']),
        'EXIT_STRUCT_VOL_Z': str(params['structure_vol_zscore_min']),
        'EXIT_TRAILING_BASE': str(params['trailing_stop_base_mult']),
        'EXIT_TRAILING_TREND': str(params['trailing_stop_trending_mult']),
        'EXIT_REENTRY_CONF': str(params['reentry_confluence_threshold']),
        'EXIT_REENTRY_WINDOW': str(params['reentry_window_btc_eth']),
        'EXIT_REENTRY_DELTA': str(params['reentry_fusion_delta']),
    }

    # Run backtest (capture output)
    cmd = [
        'python3', 'bin/backtest_knowledge_v2.py',
        '--asset', asset,
        '--start', start_date,
        '--end', end_date
    ]

    # Merge with existing environment (FIX: was only passing env_vars)
    import os
    my_env = os.environ.copy()
    my_env.update(env_vars)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=my_env,
        timeout=300
    )

    # Parse output for metrics
    output = result.stdout

    # Extract metrics using regex
    pnl_match = re.search(r'Total PNL: \$([0-9.-]+)', output)
    trades_match = re.search(r'Total Trades: ([0-9]+)', output)
    win_rate_match = re.search(r'Win Rate: ([0-9.]+)%', output)
    pf_match = re.search(r'Profit Factor: ([0-9.]+)', output)
    dd_match = re.search(r'Max Drawdown: ([0-9.]+)%', output)

    return {
        'pnl': float(pnl_match.group(1)) if pnl_match else -10000,
        'trades': int(trades_match.group(1)) if trades_match else 0,
        'win_rate': float(win_rate_match.group(1)) / 100.0 if win_rate_match else 0.0,
        'profit_factor': float(pf_match.group(1)) if pf_match else 0.0,
        'max_drawdown': float(dd_match.group(1)) / 100.0 if dd_match else 1.0,
    }


def objective(trial: optuna.Trial, asset: str, start_date: str, end_date: str) -> float:
    """
    Optuna objective function: Maximize PNL × sqrt(Win_Rate) × sqrt(Profit_Factor)

    This metric balances:
    - Absolute returns (PNL)
    - Consistency (win rate)
    - Risk-adjusted returns (profit factor)
    """
    # Define search space
    params = {
        'pattern_confluence_threshold': trial.suggest_categorical('pattern_confluence', [1, 2, 3]),
        'structure_min_hold_bars': trial.suggest_categorical('struct_min_hold', [8, 12, 16, 20]),
        'structure_rsi_long_threshold': trial.suggest_categorical('struct_rsi_long', [20, 25, 30]),
        'structure_rsi_short_threshold': trial.suggest_categorical('struct_rsi_short', [70, 75, 80]),
        'structure_vol_zscore_min': trial.suggest_categorical('struct_vol_z', [0.5, 1.0, 1.5, 2.0]),
        'trailing_stop_base_mult': trial.suggest_float('trailing_base', 1.5, 2.5),
        'trailing_stop_trending_mult': trial.suggest_float('trailing_trend', 2.0, 3.0),
        'reentry_confluence_threshold': trial.suggest_categorical('reentry_conf', [0, 2, 3]),
        'reentry_window_btc_eth': trial.suggest_categorical('reentry_window', [5, 7, 10]),
        'reentry_fusion_delta': trial.suggest_float('reentry_delta', 0.03, 0.10),
    }

    # Run backtest
    metrics = run_backtest_with_params(asset, start_date, end_date, params)

    # Compute composite score
    pnl = metrics['pnl']
    win_rate = max(metrics['win_rate'], 0.01)  # Avoid zero
    profit_factor = max(metrics['profit_factor'], 0.01)
    trades = metrics['trades']

    # Penalize if too few trades (< 20)
    trade_penalty = 1.0 if trades >= 20 else (trades / 20.0)

    # Composite score: PNL × sqrt(WR) × sqrt(PF) × trade_penalty
    score = pnl * (win_rate ** 0.5) * (profit_factor ** 0.5) * trade_penalty

    # Log to trial user attrs for later analysis
    trial.set_user_attr('pnl', pnl)
    trial.set_user_attr('trades', trades)
    trial.set_user_attr('win_rate', win_rate)
    trial.set_user_attr('profit_factor', profit_factor)
    trial.set_user_attr('max_drawdown', metrics['max_drawdown'])

    return score


def main():
    parser = argparse.ArgumentParser(description='Optimize exit strategy parameters')
    parser.add_argument('--asset', type=str, required=True, choices=['BTC', 'ETH', 'SPY'],
                       help='Asset to optimize')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--output', type=str, default='reports/exit_optimization',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study_name = f"{args.asset}_exit_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    print(f"🔍 Starting exit strategy optimization for {args.asset}")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Trials: {args.trials}")
    print(f"   Output: {output_dir}")
    print("")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args.asset, args.start, args.end),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # Save results
    results = {
        'asset': args.asset,
        'period': f"{args.start} to {args.end}",
        'trials': args.trials,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'best_metrics': {
            'pnl': study.best_trial.user_attrs.get('pnl'),
            'trades': study.best_trial.user_attrs.get('trades'),
            'win_rate': study.best_trial.user_attrs.get('win_rate'),
            'profit_factor': study.best_trial.user_attrs.get('profit_factor'),
            'max_drawdown': study.best_trial.user_attrs.get('max_drawdown'),
        },
        'top_10_trials': []
    }

    # Get top 10 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -float('inf'), reverse=True)[:10]
    for i, trial in enumerate(sorted_trials, 1):
        results['top_10_trials'].append({
            'rank': i,
            'score': trial.value,
            'params': trial.params,
            'metrics': {
                'pnl': trial.user_attrs.get('pnl'),
                'trades': trial.user_attrs.get('trades'),
                'win_rate': trial.user_attrs.get('win_rate'),
                'profit_factor': trial.user_attrs.get('profit_factor'),
                'max_drawdown': trial.user_attrs.get('max_drawdown'),
            }
        })

    # Save to JSON
    output_path = output_dir / f"{args.asset}_{args.start}_{args.end}_exit_optimization.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest Score: {results['best_score']:.2f}")
    print(f"\nBest Parameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    print(f"\nBest Metrics:")
    print(f"  PNL: ${results['best_metrics']['pnl']:.2f}")
    print(f"  Trades: {results['best_metrics']['trades']}")
    print(f"  Win Rate: {results['best_metrics']['win_rate']:.1%}")
    print(f"  Profit Factor: {results['best_metrics']['profit_factor']:.2f}")
    print(f"  Max Drawdown: {results['best_metrics']['max_drawdown']:.1%}")
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
