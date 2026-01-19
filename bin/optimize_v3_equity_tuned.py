#!/usr/bin/env python3
"""
Knowledge-Aware Optimizer v3.1 - Equity Market Tuned

SPY-specific optimization with relaxed filters for RTH-only trading:
- Lower macro weight (equities less macro-sensitive than crypto)
- Reduced PTI penalty (equity fakeouts less common)
- Relaxed M1/M2 confirmation requirement
- Lower tier thresholds (fewer bars means need lower selectivity)
- Wider ATR stop ranges (equity volatility different from crypto)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
import optuna
from optuna.samplers import TPESampler
from typing import Dict, List
import logging

from backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_feature_store(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load pre-built MTF feature store."""
    feature_dir = Path('data/features_mtf')
    pattern = f"{asset}_1H_*.parquet"
    files = list(feature_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No feature store found for {asset} in {feature_dir}")

    feature_path = sorted(files)[-1]
    print(f"Loading feature store: {feature_path}")

    df = pd.read_parquet(feature_path)

    # Filter to date range
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def objective_equity(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """
    Equity-tuned objective function.
    
    Key differences from v3 crypto:
    - Lower macro weight range (0.01-0.10 vs 0.05-0.20)
    - Lower PTI weight range (0.01-0.10 vs 0.05-0.20)
    - Higher Wyckoff/Liquidity weight ranges (equity markets respect structure)
    - Lower tier thresholds (0.30-0.50 vs 0.40-0.60 for tier1)
    - M1/M2 and Macro confirmation optional (not forced)
    """
    # Sample domain weights - EQUITY TUNED
    wyckoff_weight = trial.suggest_float('wyckoff_weight', 0.25, 0.45)  # Higher floor
    liquidity_weight = trial.suggest_float('liquidity_weight', 0.25, 0.45)  # Higher floor
    momentum_weight = trial.suggest_float('momentum_weight', 0.10, 0.25)
    macro_weight = trial.suggest_float('macro_weight', 0.01, 0.10)  # REDUCED for equities
    pti_weight = trial.suggest_float('pti_weight', 0.01, 0.10)  # REDUCED (fewer fakeouts)

    # Check weight sum
    total_weight = wyckoff_weight + liquidity_weight + momentum_weight + macro_weight
    if total_weight > 0.95:
        return -1e6

    # Sample entry thresholds - RELAXED for RTH-only
    tier1_threshold = trial.suggest_float('tier1_threshold', 0.30, 0.50)  # LOWERED
    tier2_threshold = trial.suggest_float('tier2_threshold', 0.20, 0.40)  # LOWERED
    tier3_threshold = trial.suggest_float('tier3_threshold', 0.15, 0.30)  # LOWERED

    if not (tier1_threshold > tier2_threshold > tier3_threshold):
        return -1e6

    # Sample entry modifiers - OPTIONAL for equities
    require_m1m2 = trial.suggest_categorical('require_m1m2_confirmation', [True, False])
    require_macro = trial.suggest_categorical('require_macro_alignment', [True, False])

    # Sample exit parameters - WIDER STOPS for equities
    atr_stop_mult = trial.suggest_float('atr_stop_mult', 1.0, 2.5)  # Tighter stops
    trailing_atr_mult = trial.suggest_float('trailing_atr_mult', 1.0, 2.5)  # Tighter trails
    max_hold_bars = trial.suggest_int('max_hold_bars', 24, 168)  # 1 day to 1 week RTH

    # Sample position sizing
    max_risk_pct = trial.suggest_float('max_risk_pct', 0.01, 0.03)
    volatility_scaling = trial.suggest_categorical('volatility_scaling', [True, False])

    # Build params
    params = KnowledgeParams(
        wyckoff_weight=wyckoff_weight,
        liquidity_weight=liquidity_weight,
        momentum_weight=momentum_weight,
        macro_weight=macro_weight,
        pti_weight=pti_weight,
        tier1_threshold=tier1_threshold,
        tier2_threshold=tier2_threshold,
        tier3_threshold=tier3_threshold,
        require_m1m2_confirmation=require_m1m2,
        require_macro_alignment=require_macro,
        atr_stop_mult=atr_stop_mult,
        trailing_atr_mult=trailing_atr_mult,
        max_hold_bars=max_hold_bars,
        max_risk_pct=max_risk_pct,
        volatility_scaling=volatility_scaling,
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    # Run backtest
    try:
        backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
        results = backtest.run()
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        return -1e6

    # Log metrics
    trial.set_user_attr('total_pnl', results['total_pnl'])
    trial.set_user_attr('total_trades', results['total_trades'])
    trial.set_user_attr('profit_factor', results['profit_factor'])
    trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
    trial.set_user_attr('max_drawdown', results['max_drawdown'])
    trial.set_user_attr('win_rate', results['win_rate'])

    # Objective: Maximize profit factor × sqrt(trade_count) / (1 + max_drawdown)
    if results['total_trades'] == 0:
        return -1e6

    trade_penalty = np.sqrt(results['total_trades']) / np.sqrt(10)
    trade_penalty = min(trade_penalty, 1.0)

    dd_penalty = 1.0 / (1.0 + results['max_drawdown'])
    pf = max(results['profit_factor'], 0.1)

    score = pf * trade_penalty * dd_penalty

    if results['sharpe_ratio'] > 1.0:
        score *= (1.0 + results['sharpe_ratio'] / 10.0)

    return score


def optimize_equity(asset: str, start_date: str, end_date: str, n_trials: int = 200) -> List[Dict]:
    """Run equity-tuned Bayesian optimization."""
    print("=" * 80)
    print(f"Equity-Tuned Knowledge Optimization - {asset} ({start_date} → {end_date})")
    print("=" * 80)

    df = load_feature_store(asset, start_date, end_date)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f"{asset}_equity_tuned_optimization"
    )

    print(f"\nRunning {n_trials} trials with EQUITY-TUNED parameters...")
    print("Adjustments: Lower macro/PTI weights, relaxed thresholds, tighter stops.\n")

    study.optimize(
        lambda trial: objective_equity(trial, df),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )

    # Extract top 10
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1e9, reverse=True)[:10]

    results = []
    for i, trial in enumerate(top_trials):
        if trial.value is None or trial.value < 0:
            continue

        config = {
            'rank': i + 1,
            'score': trial.value,
            'params': trial.params,
            'metrics': {
                'total_pnl': trial.user_attrs.get('total_pnl', 0.0),
                'total_trades': trial.user_attrs.get('total_trades', 0),
                'profit_factor': trial.user_attrs.get('profit_factor', 0.0),
                'sharpe_ratio': trial.user_attrs.get('sharpe_ratio', 0.0),
                'max_drawdown': trial.user_attrs.get('max_drawdown', 0.0),
                'win_rate': trial.user_attrs.get('win_rate', 0.0)
            }
        }
        results.append(config)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Top 3 Equity-Tuned Configurations for {asset}")
    print("=" * 80)

    for config in results[:3]:
        print(f"\nRank {config['rank']}: Score = {config['score']:.3f}")
        print(f"  Domain Weights:")
        print(f"    Wyckoff: {config['params']['wyckoff_weight']:.3f}")
        print(f"    Liquidity: {config['params']['liquidity_weight']:.3f}")
        print(f"    Momentum: {config['params']['momentum_weight']:.3f}")
        print(f"    Macro: {config['params']['macro_weight']:.3f}")
        print(f"    PTI: {config['params']['pti_weight']:.3f}")
        print(f"  Entry Thresholds:")
        print(f"    Tier 1: {config['params']['tier1_threshold']:.3f}")
        print(f"    Tier 2: {config['params']['tier2_threshold']:.3f}")
        print(f"    Tier 3: {config['params']['tier3_threshold']:.3f}")
        print(f"  Metrics:")
        print(f"    PNL: ${config['metrics']['total_pnl']:.2f}")
        print(f"    Trades: {config['metrics']['total_trades']}")
        print(f"    Win Rate: {config['metrics']['win_rate']:.1%}")
        print(f"    Profit Factor: {config['metrics']['profit_factor']:.2f}")
        print(f"    Sharpe: {config['metrics']['sharpe_ratio']:.2f}")
        print(f"    Max DD: {config['metrics']['max_drawdown']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Equity-tuned knowledge-aware Bayesian optimization'
    )
    parser.add_argument('--asset', required=True, help='Asset to optimize (SPY, QQQ, etc.)')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-12-31', help='End date')
    parser.add_argument('--trials', type=int, default=200, help='Number of trials')

    args = parser.parse_args()

    results = optimize_equity(args.asset, args.start, args.end, args.trials)

    if not results:
        print("\nNo valid configurations found.")
        return

    # Save results
    output_dir = Path('reports/optuna_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{args.asset}_equity_tuned_best_configs.json'

    with open(output_path, 'w') as f:
        json.dump({
            'asset': args.asset,
            'period': f"{args.start} to {args.end}",
            'n_trials': args.trials,
            'timestamp': datetime.now().isoformat(),
            'optimizer_version': 'v3.1_equity_tuned',
            'features_used': 'ALL 69 MTF features (equity-tuned weights)',
            'tuning_notes': 'Lower macro/PTI, relaxed thresholds, tighter stops for RTH equity markets',
            'top_10_configs': results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
